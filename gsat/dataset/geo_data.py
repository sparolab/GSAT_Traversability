import os
import numpy as np
import torch

from .utils.io import load_bin
from .augmentation.geo_aug import data_augment
from .visualization.utils import visualize_pts_label_bev_with_grid

class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root,
        split,
        pts='point',
        label='label',
        aug=True,
        point_range_filter=None,
        label_point_range_filter=None,
    ):
        assert split in ['train', 'val', 'test']

        self.data_root = data_root
        self.split = split
        self.pts_dir = os.path.join(data_root, split, pts)
        self.label_dir = os.path.join(data_root, split, label)
        self.aug = aug

        self.point_data_infos = sorted([f for f in os.listdir(self.pts_dir) if f.endswith('.bin')])
        self.label_data_infos = sorted([f for f in os.listdir(self.label_dir) if f.endswith('.bin')])

        if point_range_filter is None:
            point_range_filter = [-12, -12, -2, 12, 12, 2]
        if label_point_range_filter is None:
            label_point_range_filter = [-12, -12, -2, 12, 12, 2]

        self.data_aug_config = dict(
            random_flip_ratio=0.5,
            pitch_flag=True,
            global_rot_scale_trans=dict(
                rot_range=[-np.pi / 3, np.pi / 3],
                scale_ratio_range=[1.0, 1.0],
                translation_std=[0.0, 0.0, 0.0],
            ),
            point_range_filter=point_range_filter,
            label_point_range_filter=label_point_range_filter,
        )

    def __len__(self):
        return len(self.point_data_infos)

    def __getitem__(self, index):
        filename = self.point_data_infos[index]

        pts = load_bin(os.path.join(self.pts_dir, filename))        # numpy (N,4)
        label_pts = load_bin(os.path.join(self.label_dir, filename))  # numpy (M,4)

        pts = torch.from_numpy(pts).float()
        label_pts = torch.from_numpy(label_pts).float()

        # visualize_pts_label_bev_with_grid(
        #     pts, label_pts,
        #     point_cloud_range=[-9, -9, -2, 9, 9, 2],
        #     voxel_size_xy=(0.15, 0.15),
        #     grid_every=1
        # )

        data_dict = {
            'pts': pts,
            'label_pts': label_pts,
            'filename': filename,
        }
        # if self.aug and self.split == 'train':
        #     data_dict = data_augment(self.data_root, data_dict, self.data_aug_config)

        #     if isinstance(data_dict['pts'], np.ndarray):
        #         data_dict['pts'] = torch.from_numpy(data_dict['pts']).float()
        #     if isinstance(data_dict['label_pts'], np.ndarray):
        #         data_dict['label_pts'] = torch.from_numpy(data_dict['label_pts']).float()

        data_dict = data_augment(self.data_root, data_dict, self.data_aug_config)

        if isinstance(data_dict['pts'], np.ndarray):
            data_dict['pts'] = torch.from_numpy(data_dict['pts']).float()
        if isinstance(data_dict['label_pts'], np.ndarray):
            data_dict['label_pts'] = torch.from_numpy(data_dict['label_pts']).float()

        return data_dict

    def get_item_for_viz(self, index):
        filename = self.point_data_infos[index]
        pts = load_bin(os.path.join(self.pts_dir, filename))
        label_pts = load_bin(os.path.join(self.label_dir, filename))
        pts = torch.from_numpy(pts).float()
        label_pts = torch.from_numpy(label_pts).float()

        data_dict = {"pts": pts, "label_pts": label_pts, "filename": filename}
        from .augmentation.geo_aug import point_range_filter

        pr_pts = self.data_aug_config["point_range_filter"]
        pr_lbl = self.data_aug_config["label_point_range_filter"]
        data_raw = point_range_filter({**data_dict}, pr_pts, "pts")
        data_raw = point_range_filter(data_raw, pr_lbl, "label_pts")
        pts_raw = data_raw["pts"]
        label_raw = data_raw["label_pts"]

        aug_info = {}
        data_aug = data_augment(self.data_root, {**data_dict}, self.data_aug_config, aug_info=aug_info)
        pts_aug = data_aug["pts"]
        label_aug = data_aug["label_pts"]
        if isinstance(pts_aug, np.ndarray):
            pts_aug = torch.from_numpy(pts_aug).float()
        if isinstance(label_aug, np.ndarray):
            label_aug = torch.from_numpy(label_aug).float()

        return pts_raw, label_raw, pts_aug, label_aug, filename, aug_info
