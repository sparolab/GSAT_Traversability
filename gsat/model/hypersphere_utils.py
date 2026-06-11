import numpy as np
import torch
import torch.nn.functional as F
from GSAT_Traversability.gsat.ops import Voxelization
from tqdm import tqdm

def points_to_voxel_grid(batched_label_pts, point_cloud_range, voxel_size, max_num_points, max_voxels):
    voxel_layer = Voxelization(
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        max_num_points=max_num_points,
        max_voxels=max_voxels
    )

    pillars, coors, npoints_per_pillar = [], [], []
    for pts in batched_label_pts:
        voxels_out, coors_out, num_points_per_voxel = voxel_layer(pts)
        pillars.append(voxels_out)
        coors.append(coors_out.long())
        npoints_per_pillar.append(num_points_per_voxel)
 
    pillars = torch.cat(pillars, dim=0)
    npoints_per_pillar = torch.cat(npoints_per_pillar, dim=0)

    coors_batch = []
    for i, cur_coors in enumerate(coors):
        coors_batch.append(F.pad(cur_coors, (1, 0), value=i))
    coors_batch = torch.cat(coors_batch, dim=0)

    device = pillars.device
    grid_x = int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])
    grid_y = int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])
    bs = len(batched_label_pts)

    voxel_grid = torch.full((bs, 1, grid_y, grid_x), fill_value=-1, device=device)

    for i in range(bs):
        mask = (coors_batch[:, 0] == i)
        if mask.sum() == 0:
            continue
        cur_coors = coors_batch[mask]
        pillar_intensity = pillars[mask, :, 3:4]
        pooled_intensity, _ = pillar_intensity.max(dim=1)
        for j in range(cur_coors.size(0)):
            x_idx = cur_coors[j, 1].item()
            y_idx = cur_coors[j, 2].item()
            voxel_grid[i, 0, y_idx, x_idx] = pooled_intensity[j]
                
    return voxel_grid

def center_extract(data_loader, gsat_feature, gsat_head, device, point_range, voxel_size, eps=0.1):
    n_samples = 0
    c = None

    gsat_feature.eval()
    gsat_head.eval()

    debug_n = 0
    max_debug_batches = 3
    with torch.no_grad():
        for data_idx, data_dict in enumerate(tqdm(data_loader, desc="hyper_center")):
            batched_pts = [pt.to(device) for pt in data_dict["batched_pts"]]
            batched_label_pts = [pt.to(device) for pt in data_dict["batched_label_pts"]]

            feature_map, _, feature_map_active = gsat_feature(batched_pts)
            latent_space, _, _, _ = gsat_head(feature_map_active)

            travel_input_train = points_to_voxel_grid(
                batched_label_pts=batched_label_pts,
                point_cloud_range=point_range,
                voxel_size=voxel_size,
                max_num_points=32,
                max_voxels=(16000, 40000)
            )

            if travel_input_train.shape[2:] != latent_space.shape[2:]:
                travel_input_train = F.interpolate(
                    travel_input_train.float(),
                    size=latent_space.shape[2:],
                    mode='nearest'
                )

            latent_dim = latent_space.shape[1]
            if c is None:
                c = torch.zeros(latent_dim, device=device)

            z_flat = latent_space.permute(0, 2, 3, 1).reshape(-1, latent_dim)
            mask = travel_input_train.view(-1) != -1
            z_pos = z_flat[mask]

            n_samples += z_pos.size(0)
            c += z_pos.sum(dim=0)

    if n_samples == 0:
        raise ValueError("No positive samples were found while extracting the hypersphere center.")

    c /= n_samples
    return c


def radius_extract(gsat_feature, gtis_head_model, data_loader, point_range, voxel_size, center_c, device):
    gsat_feature.eval()
    gtis_head_model.eval()
    all_distances = []

    with torch.no_grad():
        for data_dict in tqdm(data_loader):
            batched_pts = [pt.to(device) for pt in data_dict["batched_pts"]]
            batched_label_pts = [pt.to(device) for pt in data_dict["batched_label_pts"]]

            feature_map, _, feature_map_layer  = gsat_feature(batched_pts)
            latent_space, _, _, _ = gtis_head_model(feature_map_layer)

            travel_input = points_to_voxel_grid(
                batched_label_pts=batched_label_pts,
                point_cloud_range= point_range,
                voxel_size= voxel_size,
                max_num_points=32,
                max_voxels=(16000, 40000)
            )

            z = latent_space.permute(0, 2, 3, 1).contiguous().view(-1, latent_space.shape[1])
            mask = travel_input.view(-1) != -1
            z_valid = z[mask]

            dist = torch.sum((z_valid - center_c) ** 2, dim=1)
            all_distances.append(dist)

    all_dist = torch.cat(all_distances, dim=0)
    return all_dist.sqrt().mean().item()