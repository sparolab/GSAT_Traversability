import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from functools import partial

def collate_fn(list_data):
    batched_pts_list = []
    batched_gt_pts_list = []

    for data_dict in list_data:
        pts= data_dict['pts']
        gt_pts = data_dict['label_pts']
        
        batched_pts_list.append(pts)
        batched_gt_pts_list.append(gt_pts)
        
    rt_data_dict = dict(
        batched_pts=batched_pts_list,
        batched_label_pts=batched_gt_pts_list,
    )

    return rt_data_dict


def get_dataloader(dataset, batch_size, num_workers, shuffle=True, drop_last=False):
    collate = collate_fn
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last, 
        collate_fn=collate,
    )
    return dataloader

