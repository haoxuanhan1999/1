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


@DETECTORS.register_module()
class EATransformerLSS(BaseModule):
    def __init__(self, final_dim=None, numC_input=512, numC_Trans=64, downsample=16, grid=0.8):

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
        self.frustum = self.create_frustum()
        
        

    

    

    def forward(self, x, rots, trans, lidar2img_rt=None, img_metas=None, post_rots=None, post_trans=None,
                extra_rots=None, extra_trans=None, depth_lidar=None):

        x, depth, img_metas = self.get_voxels(x, rots, trans, post_rots, post_trans, extra_rots, extra_trans, depth_lidar, img_metas)
        
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
