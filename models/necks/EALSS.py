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
    def __init__(self, lss=False, lc_fusion=False, camera_stream=False,
                camera_depth_range=[4.0, 45.0, 1.0], img_depth_loss_weight=1.0,  img_depth_loss_method='kld',
                grid=0.6, num_views=6, se=False, final_dim=(900, 1600), pc_range=[-50, -50, -5, 50, 50, 3],
                downsample=4, imc=512, lic=384, step=7, **kwargs):

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
        


    def extract_feat(self, points, x, input):
        
        B, S, N, C, H, W = x.shape  #获取图像特征维度参数
        lidar2img_rt = input       #获取lidar2img矩阵 [1,3,6,4,4]
        lidar2img_rt = lidar2img_rt.squeeze(0)  #[3,6,4,4]

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
        depth = torch.zeros(batch_size, img_size[1], 1, img_size[3], img_size[4]).cuda() # 创建大小
        for b in range(batch_size):
            cur_coords = points[b].float()[:, :3]  #取点的xyz
            # lidar2image
            cur_coords = rots_depth[b].matmul(cur_coords.transpose(1, 0))
            cur_coords += trans_depth[b].reshape(-1, 3, 1)

            # get 2d coords
            dist = cur_coords[:, 2, :]
            cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-4, 1e4)
            cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]
            # imgaug

            cur_coords = cur_coords[:, :2, :].transpose(1, 2)
            # normalize coords for grid sample
            cur_coords = cur_coords[..., [1, 0]]

            on_img = (
                (cur_coords[..., 0] < img_size[3])
                & (cur_coords[..., 0] >= 0)
                & (cur_coords[..., 1] < img_size[4])
                & (cur_coords[..., 1] >= 0)
            )

            for c in range(on_img.shape[0]):
                masked_coords = cur_coords[c, on_img[c]].long()  # 点云投影到图像坐标
                masked_dist = dist[c, on_img[c]]  # 对应深度
                depth[b, c, 0, masked_coords[:, 0], masked_coords[:, 1]] = masked_dist.float()  # 稀疏的深度约束图（用于计算loss） 1, 6, 1, 448, 800

        step = 7
        B, N, C, H, W = depth.size()
        depth_tmp = depth.reshape(B*N, C, H, W)
        pad = int((step - 1) // 2)
        depth_tmp = F.pad(depth_tmp, [pad, pad, pad, pad], mode='constant', value=0)
        patches = depth_tmp.unfold(dimension=2, size=step, step=1)
        patches = patches.unfold(dimension=3, size=step, step=1)
        max_depth, _ = patches.reshape(B, N, C, H, W, -1).max(dim=-1)  # [2, 6, 1, 256, 704]
        img_metas[0].update({'max_depth': max_depth})

        # 求解max_depth四个方向梯度, 随后concat depth, 以缓解深度跳变对深度预测模块的影响
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
        grad = torch.cat(output_list, dim=2)  # [2, 6, 4, 256, 704]
        max_grad = torch.abs(grad).max(dim=2)[0].unsqueeze(2)
        img_metas[0].update({'max_grad': max_grad})
        #depth_ = depth
        depth = torch.cat([depth, grad], dim=2)  # [1, 6, 5, 448, 800]

        img_bev_feat, depth_dist, img_metas = self.lift_splat_shot_vis(img_feats_view, rots, trans, lidar2img_rt=lidar2img_rt, img_metas=img_metas
                                                            ,depth_lidar=depth)
        
        return img_bev_feat


    def forward(self,
                points=None,
                img_feats=None,
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
            points, x=img_feats, input=img_metas)
        
        return img_bev_feats