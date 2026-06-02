from GSAT_Traversability.gsat.ops import Voxelization
import torch
import torch.nn.functional as F

def bev_representation(batched_pts, point_cloud_range, voxel_size, max_num_points, max_voxels):

    voxel_layer = Voxelization(
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        max_num_points=max_num_points,
        max_voxels=max_voxels
    )

    pillars, coors, npoints_per_pillar = [], [], []
    for pts in batched_pts:
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
    bs = len(batched_pts)

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
                
    return voxel_grid  # (B,1,grid_y,grid_x)