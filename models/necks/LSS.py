# Copyright (c) Junjie.huang. All rights reserved.

import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmdet3d.models.builder import NECKS
from mmdet3d.models import builder
from mmdet.models.backbones.resnet import Bottleneck, BasicBlock
from mmcv.cnn import build_norm_layer
from mmdet3d.models.backbones.swin import SwinTransformer
from mmdet3d.ops.bev_pool import bev_pool
from mmdet.models import DETECTORS
import pdb
from mmdet3d.ops.bev_pool import bev_pool

@DETECTORS.register_module()
class EATransformerLSS(BaseModule):
    def __init__(self, final_dim=None, inputC=512, camC=64, downsample=16, grid=0.8):

        super(EATransformerLSS, self).__init__()
        
        self.grid = grid
        self.grid_conf = {
                'xbound': [-51.2, 51.2, 0.8],
                'ybound': [-51.2, 51.2, 0.8],
                'zbound': [-10.0, 10.0, 20.0],
                'dbound': [4.0, 45.0, 1.0], }
        self.dx, self.bx, self.nx = gen_dx_bx(self.grid_conf['xbound'],
                                              self.grid_conf['ybound'],
                                              self.grid_conf['zbound'],
                                              )
        self.final_dim = final_dim
        self.downsample = downsample
        self.fH, self.fW = self.final_dim[0] // self.downsample, self.final_dim[1] // self.downsample
        self.camC = camC
        self.inputC = inputC
        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape
        
        # self.dtransform = nn.Sequential(  # 提取深度特征
        #     nn.Conv2d(5, 8, 5, padding=2),
        #     nn.ReLU(),
        #     nn.Conv2d(8, 32, 5, stride=2, padding=2),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, 5, stride=2, padding=2),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, 5, stride=2, padding=2),
        #     nn.ReLU(),
        # )
        self.dtransform = nn.Sequential(  # 提取深度特征
            nn.Conv2d(5, 8, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, stride=2, padding=2),
            nn.ReLU(),
        )

        self.convnet = nn.Sequential(  # 提取图像语义
            nn.Conv2d(inputC, inputC, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(inputC, camC, 3, padding=1),
            nn.ReLU(),
        )

        self.prenet = nn.Sequential(  # 预测图像深度
            nn.Conv2d(inputC + 64, inputC, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(inputC, inputC, 3, padding=1),
            nn.ReLU()
        )

        self.attention1 = nn.Conv2d(inputC, 1, 1)
        self.depthnet = nn.Conv2d(inputC, self.D, 1)
        self.contextnet = nn.Conv2d(inputC, self.camC, 1)
        self.attention2 = nn.Conv2d(self.camC, 1, 1)

    

    

    def forward(self, x, rots, trans, lidar2img_rt=None, img_metas=None, post_rots=None, post_trans=None,
                extra_rots=None, extra_trans=None, depth_lidar=None):

        x, depth, img_metas = self.get_voxels(x, rots, trans, post_rots, post_trans, extra_rots, extra_trans, depth_lidar, img_metas)
        x = x.unsqueeze(0)
        return x
        
    def get_voxels(self, x, rots=None, trans=None, post_rots=None, post_trans=None, extra_rots=None, extra_trans=None, depth_lidar=None, img_metas=None):
        geom = self.get_geometry(rots, trans, post_rots, post_trans, extra_rots, extra_trans)
        x, depth, img_metas = self.get_cam_feats(x, depth_lidar, img_metas)
        x = self.voxel_pooling(geom, x)
        return x, depth, img_metas

    def get_geometry(self, rots, trans, post_rots=None, post_trans=None, extra_rots=None, extra_trans=None):  #将视锥体的点从图像坐标系映射到自车坐标系
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape
        # ADD
        # undo post-transformation
        # B x N x D x H x W x 3
        if post_rots is not None or post_trans is not None:
            if post_trans is not None:
                points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
            if post_rots is not None:
                points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
        else:
            points = self.frustum.repeat(B, N, 1, 1, 1, 1).unsqueeze(-1)  # B x N x D x H x W x 3 x 1

        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5)
        points = rots.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)

        if extra_rots is not None or extra_trans is not None:
            if extra_rots is not None:
                points = extra_rots.view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1)).squeeze(-1)
            if extra_trans is not None:
                points += extra_trans.view(B, N, 1, 1, 1, 3)
        return points

    def get_cam_feats(self, x, d, metas):  #针对图像特征和深度特征进行融合处理
        label_depth = d[:, :, :1]  #从深度图取出一个深度切片用作标签深度
        metas[0].update({'label_depth': label_depth})
        x = x.squeeze(0)  #删除batch=1维度
        B, N, C, fH, fW = x.shape
        d = d.view(B * N, *d.shape[2:])  #[18,5,256,704]
        x = x.view(B * N, C, fH, fW)     #[18,512,16,44]

        d = self.dtransform(d)  #深度特征提取  #[18,64,32,88]
        context = self.convnet(x)  # [18, 64, 16, 44] 提取语义特征
        x = torch.cat([x, d], dim=1)
        # depth_pre = self.upsample_depth(x)  # 增加x8上采样模块用来监督深度信息
        # metas[0].update({'pre_upsample': depth_pre})  # [b*6, 118, 256, 704]
        x = self.prenet(x)  # 合并深度和图像信息 cat好还是add好?  预测深度  [b*6, 128, 32, 88]  (用于计算loss)
        attention1 = self.attention1(x)  #[18,1,16,44]
        depth = self.depthnet(x) * attention1  #[18,41,16,44]
        # metas[0].update({'pre_depth': depth})
        depth = depth.softmax(dim=1)

        #depth = depth
        x = self.contextnet(x) + context  #[18,64,16,44]
        attention2 = self.attention2(x)   #[18,1,16,44]
        x = depth.unsqueeze(1) * (x * attention2).unsqueeze(2)  #[18,64,41,16,44]
        x = x.view(B, N, self.camC, self.D, fH, fW)             #[3,6,64,41,16,44]
        x = x.permute(0, 1, 3, 4, 5, 2)    #[3,6,41,16,44,64]
        return x, depth, metas

    def voxel_pooling(self, geom_feats, x):  #稠密图像特征或点云特征高效投影到BEV网格
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W
        batch_size = x.shape[0]
        # flatten x 
        x = x.reshape(Nprime, C)  #扁平化特征

        # flatten indices
        bx = self.bx.type_as(geom_feats)
        dx = self.dx.type_as(geom_feats)
        nx = self.nx.type_as(geom_feats).long()
        # flatten indices
        #geom_feats = ((geom_feats - (self.bx - self.dx / 2.)) / self.dx).long()  # 转成BEV视图
        geom_feats = ((geom_feats - (bx - dx / 2.)) / dx).long()
        geom_feats = geom_feats.view(Nprime, 3)  #扁平化几何点
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long) for ix in range(B)])
        batch_ix = batch_ix.to(geom_feats.device)
        geom_feats = torch.cat((geom_feats, batch_ix), 1)
        geom_feats = geom_feats.type_as(x).long()
        # filter out points that are outside box  过滤无效点
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < nx[0]) \
                & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < nx[1]) \
                & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]
        # get tensors from the same voxel next to each other
        # ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B) \
        #         + geom_feats[:, 1] * (self.nx[2] * B) \
        #         + geom_feats[:, 2] * B \
        #         + geom_feats[:, 3]
        # sorts = ranks.argsort()
        # x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]
        # # cumsum trick
        # if not self.use_quickcumsum:
        #     x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        # else:
        #     x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        # # griddify (B x C x Z x X x Y)
        # final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)
        # final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x

        #使用bev_pool
        final = bev_pool(x, geom_feats, B,
                             self.nx[2], self.nx[0], self.nx[1])
        final = final.transpose(dim0=-2, dim1=-1)

        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)  #[3,64,128,128]

        return final



    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.final_dim
        fH, fW = self.fH, self.fW
        ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)  #生成深度维度的采样点
        D, _, _ = ds.shape
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)  #图像平面的网格点
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)  #生成视锥点云
        return nn.Parameter(frustum, requires_grad=False)

def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor(
        [row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
    nx = torch.Tensor([(row[1] - row[0]) / row[2]
                      for row in [xbound, ybound, zbound]])

    return dx, bx, nx


class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = (ranks[1:] != ranks[:-1])

        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        kept, = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None
    

