import mmcv
import torch
from mmcv.parallel import DataContainer as DC
from mmcv.runner import BaseModule
from mmcv.runner import force_fp32
from os import path as osp
from torch import nn as nn
from torch.nn import functional as F

from mmdet3d.core import (Box3DMode, Coord3DMode, bbox3d2result,
                          merge_aug_bboxes_3d, show_result)
from mmdet3d.ops import Voxelization
from mmdet.core import multi_apply
from mmcv.cnn import ConvModule, xavier_init
from mmdet.models import DETECTORS
from mmdet3d.models.builder import NECKS
from mmdet3d.models import builder

from mmdet3d.models.detectors import MVXFasterRCNN
from .cam_stream_lss import LiftSplatShoot

@DETECTORS.register_module()
class EALSS_CAM(BaseModule):
    """Base class of Multi-modality VoxelNet."""
    def __init__(self, lss=False, lc_fusion=False, camera_stream=True,
                camera_depth_range=[4.0, 45.0, 1.0], img_depth_loss_weight=1.0,  img_depth_loss_method='kld',
                grid=0.8, num_views=6, se=False, final_dim=(900, 1600), pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                downsample=16, imc=512, lic=384, step=7, **kwargs):

        super(EALSS_CAM, self).__init__(**kwargs)
        self.num_views = num_views
        
        self.img_depth_loss_weight = img_depth_loss_weight
        self.img_depth_loss_method = img_depth_loss_method
        self.camera_depth_range = camera_depth_range
        self.lift = camera_stream
        self.step = step
        self.final_dim = final_dim
        self.se = se
        if camera_stream:
            self.lift_splat_shot_vis = LiftSplatShoot(lss=lss, grid=grid, inputC=imc, camC=64,
            pc_range=pc_range, final_dim=final_dim, downsample=downsample)
        


    def extract_feat(self, points, x, input, img_metas):
        
        B, S, N, C, H, W = x.shape  #获取图像特征维度参数
        lidar2img_rt = input[8]       #获取lidar2img矩阵 [1,3,6,4,4]
        lidar2img_rt = lidar2img_rt.squeeze(0)  #[3,6,4,4]
        post_rots = input[4].squeeze(0)         #[3,6,3,3]
        post_trans = input[5].squeeze(0)        #[3,6,3]

        rots = []  #lidar2img的rot
        trans = []  #lidar2img的trans
        rots_depth = []  #lidar2img的rots_depth
        trans_depth = [] ##lidar2img的trans_depth

        for sample_idx in range(lidar2img_rt.shape[0]):
            # 获取该时间点的所有视角的 lidar2img_rt 变换矩阵
            mats = lidar2img_rt[sample_idx]  # mats 的维度是 [6, 4, 4]
            mats_inv = torch.linalg.inv(mats)

            rots.append(mats_inv[:, :3, :3])  # [6, 3, 3] 提取旋转部分
            trans.append(mats_inv[:, :3, 3])  # [6, 3] 提取位移部分
            rots_depth.append(mats[:, :3, :3])  # [6, 3, 3] 提取深度旋转部分
            trans_depth.append(mats[:, :3, 3])  # [6, 3] 提取深度位移部分
                
            
        # 堆叠每个时间点的数据
        rots = torch.stack(rots, dim=0)  # [3, 6, 3, 3]
        trans = torch.stack(trans, dim=0)  # [3, 6, 3]
        rots_depth = torch.stack(rots_depth, dim=0)  # [3, 6, 3, 3]
        trans_depth = torch.stack(trans_depth, dim=0)  # [3, 6, 3]
        

        batch_size = len(points)  #遍历点云的时间数
        depth = torch.zeros(batch_size, 6, 1, 256, 704).cuda() # 创建大小,与图像一致
        for b in range(batch_size):
            cur_coords = points[b].float()[:, :, :3]  #取点的xyz
            cur_coords = cur_coords.squeeze(0)  #移除点云中的batchsize维度
            # lidar2image
            cur_coords = rots_depth[b].matmul(cur_coords.transpose(1, 0))
            cur_coords += trans_depth[b].reshape(-1, 3, 1)  #[6, 3, num_points]

            # get 2d coords
            dist = cur_coords[:, 2, :]  #获取深度Z轴
            cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-4, 1e4)  #限制深度范围
            cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]  #归一化，将点投影到图像坐标
            cur_coords = cur_coords[:, :2, :].transpose(1, 2)  #生成图像坐标[6,num_points,2]


            # imgaug
            ones = torch.ones((cur_coords.shape[0], cur_coords.shape[1], 1), device=cur_coords.device)
            cur_coords_homogeneous = torch.cat([cur_coords, ones], dim=-1)  # [6, num_points, 3]
            # 应用增强后的旋转和平移
            post_rots_expanded = post_rots[b] # [6, 3, 3]
            post_trans_expanded = post_trans[b]  # [6, 3]
            transformed_coords = (
                post_rots_expanded.matmul(cur_coords_homogeneous.transpose(2, 1)).transpose(2, 1)  # 旋转
                + post_trans_expanded.unsqueeze(1)  # 平移
            )

            cur_coords = transformed_coords[..., :2]  # 更新后的图像坐标
            
            # normalize coords for grid sample
            cur_coords = cur_coords[..., [1, 0]]  #生成图像坐标[6,num_points,2]

            #过滤有效点
            on_img = (
                (cur_coords[..., 0] < 256)
                & (cur_coords[..., 0] >= 0)
                & (cur_coords[..., 1] < 704)
                & (cur_coords[..., 1] >= 0)
            )

            for c in range(on_img.shape[0]):  #遍历摄像头
                masked_coords = cur_coords[c, on_img[c]].long()  # 提取第C个视角下投影到图像的点云的图像坐标
                masked_dist = dist[c, on_img[c]]  # 获取有效点深度
                depth[b, c, 0, masked_coords[:, 0], masked_coords[:, 1]] = masked_dist.float()  # 每个投影点的深度值填入深度图对应位置

        step = 7  #滑动窗口大小
        B, N, C, H, W = depth.size()  #深度图张量维度
        depth_tmp = depth.reshape(B*N, C, H, W)
        pad = int((step - 1) // 2)  #填充大小
        depth_tmp = F.pad(depth_tmp, [pad, pad, pad, pad], mode='constant', value=0)  #为每个深度图添加0填充
        patches = depth_tmp.unfold(dimension=2, size=step, step=1)
        patches = patches.unfold(dimension=3, size=step, step=1)
        max_depth, _ = patches.reshape(B, N, C, H, W, -1).max(dim=-1)  # [3, 6, 1, 256, 704]生成与深度图分辨率相同的最大深度图
        img_metas[0].update({'max_depth': max_depth})

        # 求解max_depth四个方向梯度, 随后concat depth, 以缓解深度跳变对深度预测模块的影响(梯度特征提供了深度图的局部变化信息，对建模边缘特性或局部特征非常有用)
        step = float(step)
        shift_list = [[step / H, 0.0 / W], [-step / H, 0.0 / W], [0.0 / H, step / W], [0.0 / H, -step / W]]
        max_depth_tmp = max_depth.reshape(B*N, C, H, W)
        output_list = []
        for shift in shift_list:
            transform_matrix =torch.tensor([[1, 0, shift[0]],[0, 1, shift[1]]]).unsqueeze(0).repeat(B*N, 1, 1).cuda()
            grid = F.affine_grid(transform_matrix, max_depth_tmp.shape).float()
            output = F.grid_sample(max_depth_tmp, grid, mode='nearest').reshape(B, N, C, H, W)  #平移后图像
            output = max_depth - output
            output_mask = ((output == max_depth) == False)
            output = output * output_mask
            output_list.append(output)
        grad = torch.cat(output_list, dim=2)  
        max_grad = torch.abs(grad).max(dim=2)[0].unsqueeze(2)
        img_metas[0].update({'max_grad': max_grad})  #最大梯度图
        #depth_ = depth
        depth = torch.cat([depth, grad], dim=2)  #[3,6,5,256,704]

        img_bev_feat, depth_dist, img_metas = self.lift_splat_shot_vis(x, rots, trans, lidar2img_rt=lidar2img_rt, img_metas=img_metas
                                                            ,depth_lidar=depth)
        
        return img_bev_feat


    def forward(self,
                points=None,
                img_feats=None,
                img=None,
                img_metas=None,
                ):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        img_bev_feats = self.extract_feat(
            points, x=img_feats, input=img, img_metas=img_metas)
        
        return img_bev_feats