# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch
import torchvision
from PIL import Image
import numpy as np
import os
import pyquaternion
import imageio

from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations
from mmdet3d.core.points import BasePoints

import pdb


def get_rot(h):
    return torch.Tensor([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])


def img_transform(img, post_rot, post_tran,
                  resize, resize_dims, crop,
                  flip, rotate):
    # adjust image
    img = img.resize(resize_dims)
    img = img.crop(crop)
    if flip:
        img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
    img = img.rotate(rotate)

    # post-homography transformation
    post_rot *= resize
    post_tran -= torch.Tensor(crop[:2])
    if flip:
        A = torch.Tensor([[-1, 0], [0, 1]])
        b = torch.Tensor([crop[2] - crop[0], 0])
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b
    A = get_rot(rotate/180*np.pi)
    b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
    b = A.matmul(-b) + b
    post_rot = A.matmul(post_rot)
    post_tran = A.matmul(post_tran) + b

    return img, post_rot, post_tran


normalize_img = torchvision.transforms.Compose((
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
))


def decode_binary_labels(labels, nclass):
    bits = torch.pow(2, torch.arange(nclass))
    return (labels & bits.view(-1, 1, 1)) > 0

@PIPELINES.register_module()
class LoadPointsFromFile_OURS(object):
    """Load Points From File.

    Load sunrgbd and scannet points from file.

    Args:
        load_dim (int): The dimension of the loaded points.
            Defaults to 6.
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        use_dim (list[int]): Which dimensions of the points to be used.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool): Whether to use shifted height. Defaults to False.
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
    """

    def __init__(self,
                 coord_type,
                 load_dim=6,
                 use_dim=[0, 1, 2, 3],
                 shift_height=False,
                 file_client_args=dict(backend='disk')):
        self.shift_height = shift_height
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _remove_close(self, points, radius=1.0):
        """Removes point too close within a certain radius from origin.

        Args:
            points (np.ndarray): Sweep points.
            radius (float): Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray: Points after removing.
        """
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]
    
    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)

        return points
    
    def get_lidar_inputs(self, results):
        lidar_infos = results['lidar_info']    #针对多帧点云的键进行修改
        point_clouds = []
        for lidar_info in lidar_infos:
            points = self._load_points(lidar_info)
            points = points.reshape(-1, self.load_dim)
            points = points[:, self.use_dim]
            points = self._remove_close(points)
            attribute_dims = None

            # if self.shift_height:
            #     floor_height = np.percentile(points[:, 2], 0.99)
            #     height = points[:, 2] - floor_height
            #     points = np.concatenate([points, np.expand_dims(height, 1)], 1)
            #     attribute_dims = dict(height=3)

            # points_class = get_points_type(self.coord_type)
            # points = points_class(
            #     points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
            point_clouds.append(torch.tensor(points))
        
        

        # # 对每个点云数组应用lidar2ego变换
        # transformed_point_clouds = []
        # #构建lidar2ego变换矩阵
        # lidar2ego = torch.eye(4)
        # lidar2ego[:3, :3] = results['lidar2ego_rots']
        # lidar2ego[:3, 3] = results['lidar2ego_trans']
        
        # for points in point_clouds:
        #     # 添加一个维度，将 [N, 4] 转换为 [N, 1, 4]，以便进行矩阵乘法
        #     points_homogeneous = torch.cat([points[:, :3], torch.ones(points.shape[0], 1)], dim=1).T  # 转置以便矩阵乘法

        #     # 进行变换
        #     transformed_points = lidar2ego @ points_homogeneous

        #     # 转换回3D坐标
        #     transformed_points = transformed_points[:3].T

        #     #附加强度信息
        #     intensities = points[:, 3].unsqueeze(1)  # [N, 1]
        #     transformed_points= torch.cat([transformed_points, intensities], dim=1)

        #     transformed_point_clouds.append(transformed_points)

        return point_clouds
    
    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data. \
                Added key and value are described below.

                - points (np.ndarray): Point clouds data.
        """
        
        results['lidar_inputs'] = self.get_lidar_inputs(results)

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += 'shift_height={}, '.format(self.shift_height)
        repr_str += 'file_client_args={}), '.format(self.file_client_args)
        repr_str += 'load_dim={}, '.format(self.load_dim)
        repr_str += 'use_dim={})'.format(self.use_dim)
        return repr_str

@PIPELINES.register_module()
class LoadMultiViewImageFromFiles_MTL(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, is_train=False, using_ego=False, temporal_consist=False,
                 data_aug_conf={
                     'resize_lim': (0.193, 0.225),
                     'final_dim': (128, 352),
                     'rot_lim': (-5.4, 5.4),
                     'H': 900, 'W': 1600,
                     'rand_flip': True,
                     'bot_pct_lim': (0.0, 0.22),
                     'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                              'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
                     'Ncams': 5,
                 }, load_seg_gt=False, num_seg_classes=14, select_classes=None):

        self.is_train = is_train
        self.using_ego = using_ego
        self.data_aug_conf = data_aug_conf
        self.load_seg_gt = load_seg_gt
        self.num_seg_classes = num_seg_classes
        self.select_classes = range(
            num_seg_classes) if select_classes is None else select_classes

        self.temporal_consist = temporal_consist
        self.test_time_augmentation = self.data_aug_conf.get('test_aug', False)

    def sample_augmentation(self, specify_resize=None, specify_flip=None):
        H, W = self.data_aug_conf['H'], self.data_aug_conf['W']
        fH, fW = self.data_aug_conf['final_dim']
        if self.is_train:
            resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims

            crop_h = max(0, newH - fH)
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)

            flip = False
            if self.data_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf['rot_lim'])
        else:
            resize = max(fH / H, fW / W)
            resize = resize + 0.04
            if specify_resize is not None:
                resize = specify_resize

            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = max(0, newH - fH)
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if specify_flip is None else specify_flip
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def choose_cams(self):
        if self.is_train and self.data_aug_conf['Ncams'] < len(self.data_aug_conf['cams']):
            cams = np.random.choice(self.data_aug_conf['cams'], self.data_aug_conf['Ncams'],
                                    replace=False)
        else:
            cams = self.data_aug_conf['cams']
            cyclist = self.data_aug_conf.get('cyclist', False)
            if cyclist:
                start_id = np.random.choice(np.arange(len(cams)))
                cams = cams[start_id:] + cams[:start_id]
        return cams

    def get_img_inputs(self, results, specify_resize=None, specify_flip=None):
        img_infos = results['img_info']

        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []
        sensor2lidar_rots = []
        sensor2lidar_trans = []
        lidar2img_rts = []

        cams = self.choose_cams()
        if self.temporal_consist:
            cam_augments = {}
            for cam in cams:
                cam_augments[cam] = self.sample_augmentation(
                    specify_resize=specify_resize, specify_flip=specify_flip)

        for frame_id, img_info in enumerate(img_infos):
            imgs.append([])
            rots.append([])
            trans.append([])
            intrins.append([])
            post_rots.append([])
            post_trans.append([])
            sensor2lidar_rots.append([])
            sensor2lidar_trans.append([])
            lidar2img_rts.append([])

            for cam in cams:
                cam_data = img_info[cam]
                filename = cam_data['data_path']
                filename = os.path.join(
                    results['data_root'], filename.split('nuscenes/')[1])

                img = Image.open(filename)

                # img = imageio.imread(filename)
                # img = Image.fromarray(img)

                post_rot = torch.eye(2)
                post_tran = torch.zeros(2)

                intrin = torch.Tensor(cam_data['cam_intrinsic'])
                # extrinsics
                sensor2lidar_rot = torch.Tensor(cam_data['sensor2lidar_rotation'])
                sensor2lidar_tran = torch.Tensor(cam_data['sensor2lidar_translation'])

                # 计算lidar2img_rot和lidar2img_trans并合并为lidar2img_trs
                lidar2cam_r = np.linalg.inv(sensor2lidar_rot)
                lidar2cam_t = sensor2lidar_tran @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                viewpad = np.eye(4)
                viewpad[:intrin.shape[0], :intrin.shape[1]] = intrin
                lidar2img_rt = torch.Tensor(viewpad @ lidar2cam_rt.T)
                
                # 进一步转换到 LiDAR 坐标系
                if self.using_ego:
                    cam2lidar = torch.eye(4)
                    cam2lidar[:3, :3] = torch.Tensor(
                        cam_data['sensor2lidar_rotation'])
                    cam2lidar[:3, 3] = torch.Tensor(
                        cam_data['sensor2lidar_translation'])

                    lidar2ego = torch.eye(4)
                    lidar2ego[:3, :3] = results['lidar2ego_rots']
                    lidar2ego[:3, 3] = results['lidar2ego_trans']

                    cam2ego = lidar2ego @ cam2lidar

                    rot = cam2ego[:3, :3]
                    tran = cam2ego[:3, 3]

                # augmentation (resize, crop, horizontal flip, rotate)
                if self.temporal_consist:
                    resize, resize_dims, crop, flip, rotate = cam_augments[cam]
                else:
                    # generate augmentation for each time-step, each
                    resize, resize_dims, crop, flip, rotate = self.sample_augmentation(
                        specify_resize=specify_resize, specify_flip=specify_flip)

                img, post_rot2, post_tran2 = img_transform(img, post_rot, post_tran,
                                                           resize=resize,
                                                           resize_dims=resize_dims,
                                                           crop=crop,
                                                           flip=flip,
                                                           rotate=rotate)

                post_tran = torch.zeros(3)
                post_rot = torch.eye(3)
                post_tran[:2] = post_tran2
                post_rot[:2, :2] = post_rot2

                imgs[frame_id].append(normalize_img(img))
                intrins[frame_id].append(intrin)
                rots[frame_id].append(rot)
                trans[frame_id].append(tran)
                post_rots[frame_id].append(post_rot)
                post_trans[frame_id].append(post_tran)
                sensor2lidar_rots[frame_id].append(sensor2lidar_rot)  #增添多帧外参旋转平移矩阵从而获取lidar2img
                sensor2lidar_trans[frame_id].append(sensor2lidar_tran)
                lidar2img_rts[frame_id].append(lidar2img_rt)

        # [num_seq, num_cam, ...]
        imgs = torch.stack([torch.stack(x, dim=0) for x in imgs], dim=0)
        rots = torch.stack([torch.stack(x, dim=0) for x in rots], dim=0)
        trans = torch.stack([torch.stack(x, dim=0) for x in trans], dim=0)
        intrins = torch.stack([torch.stack(x, dim=0) for x in intrins], dim=0)
        post_rots = torch.stack([torch.stack(x, dim=0)
                                for x in post_rots], dim=0)
        post_trans = torch.stack([torch.stack(x, dim=0)
                                 for x in post_trans], dim=0)
        sensor2lidar_rots = torch.stack([torch.stack(x, dim=0) for x in sensor2lidar_rots], dim=0)
        sensor2lidar_trans = torch.stack([torch.stack(x, dim=0) for x in sensor2lidar_trans], dim=0)
        lidar2img_rts = torch.stack([torch.stack(x, dim=0) for x in lidar2img_rts], dim=0)

        return imgs, rots, trans, intrins, post_rots, post_trans, sensor2lidar_rots, sensor2lidar_trans, lidar2img_rts

    def __call__(self, results):
        if (not self.is_train) and self.test_time_augmentation:
            results['flip_aug'] = []
            results['scale_aug'] = []
            img_inputs = []
            for flip in self.data_aug_conf.get('tta_flip', [False, ]):
                for scale in self.data_aug_conf.get('tta_scale', [None, ]):
                    results['flip_aug'].append(flip)
                    results['scale_aug'].append(scale)
                    img_inputs.append(
                        self.get_img_inputs(results, scale, flip))

            results['img_inputs'] = img_inputs
        else:
            results['img_inputs'] = self.get_img_inputs(results)

        return results


@PIPELINES.register_module()
class LoadAnnotations3D_MTL(LoadAnnotations):
    """Load Annotations3D.

    Load instance mask and semantic mask of points and
    encapsulate the items into related fields.

    Args:
        with_bbox_3d (bool, optional): Whether to load 3D boxes.
            Defaults to True.
        with_label_3d (bool, optional): Whether to load 3D labels.
            Defaults to True.
        with_attr_label (bool, optional): Whether to load attribute label.
            Defaults to False.
        with_mask_3d (bool, optional): Whether to load 3D instance masks.
            for points. Defaults to False.
        with_seg_3d (bool, optional): Whether to load 3D semantic masks.
            for points. Defaults to False.
        with_bbox (bool, optional): Whether to load 2D boxes.
            Defaults to False.
        with_label (bool, optional): Whether to load 2D labels.
            Defaults to False.
        with_mask (bool, optional): Whether to load 2D instance masks.
            Defaults to False.
        with_seg (bool, optional): Whether to load 2D semantic masks.
            Defaults to False.
        with_bbox_depth (bool, optional): Whether to load 2.5D boxes.
            Defaults to False.
        poly2mask (bool, optional): Whether to convert polygon annotations
            to bitmasks. Defaults to True.
        seg_3d_dtype (dtype, optional): Dtype of 3D semantic masks.
            Defaults to int64
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details.
    """

    def __init__(self,
                 with_bbox_3d=True,
                 with_label_3d=True,
                 with_instance_tokens=False,
                 with_attr_label=False,
                 with_mask_3d=False,
                 with_seg_3d=False,
                 with_bbox=False,
                 with_label=False,
                 with_mask=False,
                 with_seg=False,
                 with_bbox_depth=False,
                 poly2mask=True,
                 seg_3d_dtype='int',
                 file_client_args=dict(backend='disk')):
        super().__init__(
            with_bbox,
            with_label,
            with_mask,
            with_seg,
            poly2mask,
            file_client_args=file_client_args)

        self.with_bbox_3d = with_bbox_3d
        self.with_bbox_depth = with_bbox_depth
        self.with_label_3d = with_label_3d
        self.with_attr_label = with_attr_label
        self.with_mask_3d = with_mask_3d
        self.with_seg_3d = with_seg_3d
        self.seg_3d_dtype = seg_3d_dtype
        self.with_instance_tokens = with_instance_tokens

    def _load_bboxes_3d(self, results):
        """Private function to load 3D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box annotations.
        """

        gt_bboxes_3d = []
        for ann_info in results['ann_info']:
            if ann_info is not None:
                gt_bboxes_3d.append(ann_info['gt_bboxes_3d'])
            else:
                gt_bboxes_3d.append(None)

        results['gt_bboxes_3d'] = gt_bboxes_3d
        results['bbox3d_fields'].append('gt_bboxes_3d')

        return results

    # modified
    def _load_labels_3d(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """

        gt_labels_3d = []
        for ann_info in results['ann_info']:
            if ann_info is not None:
                gt_labels_3d.append(ann_info['gt_labels_3d'])
            else:
                gt_labels_3d.append(None)
        results['gt_labels_3d'] = gt_labels_3d

        return results

    # added
    def _load_instance_tokens(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """

        instance_tokens = []
        for ann_info in results['ann_info']:
            if ann_info is not None:
                instance_tokens.append(ann_info['instance_tokens'])
            else:
                instance_tokens.append(None)
        results['instance_tokens'] = instance_tokens

        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box, label, mask and
                semantic segmentation annotations.
        """
        results = super().__call__(results)

        # 即使 without 3d bounding boxes, 也需要正常训练
        if self.with_bbox_3d:
            results = self._load_bboxes_3d(results)

        if self.with_label_3d:
            results = self._load_labels_3d(results)

        if self.with_instance_tokens:
            results = self._load_instance_tokens(results)

        # loading valid_flags
        gt_valid_flags = []
        for ann_info in results['ann_info']:
            if ann_info is not None:
                gt_valid_flags.append(ann_info['gt_valid_flag'])
            else:
                gt_valid_flags.append(None)
        results['gt_valid_flag'] = gt_valid_flags

        # loading visibility tokens
        gt_vis_tokens = []
        for ann_info in results['ann_info']:
            if ann_info is not None:
                gt_vis_tokens.append(ann_info['gt_vis_tokens'])
            else:
                gt_vis_tokens.append(None)
        results['gt_vis_tokens'] = gt_vis_tokens

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        indent_str = '    '
        repr_str = self.__class__.__name__ + '(\n'
        repr_str += f'{indent_str}with_bbox_3d={self.with_bbox_3d}, '
        repr_str += f'{indent_str}with_label_3d={self.with_label_3d}, '
        repr_str += f'{indent_str}with_attr_label={self.with_attr_label}, '
        repr_str += f'{indent_str}with_mask_3d={self.with_mask_3d}, '
        repr_str += f'{indent_str}with_seg_3d={self.with_seg_3d}, '
        repr_str += f'{indent_str}with_bbox={self.with_bbox}, '
        repr_str += f'{indent_str}with_label={self.with_label}, '
        repr_str += f'{indent_str}with_mask={self.with_mask}, '
        repr_str += f'{indent_str}with_seg={self.with_seg}, '
        repr_str += f'{indent_str}with_bbox_depth={self.with_bbox_depth}, '
        repr_str += f'{indent_str}poly2mask={self.poly2mask})'
        return repr_str
