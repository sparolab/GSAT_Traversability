import os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def points_to_bev_occupancy(points, point_cloud_range, voxel_size_xy):
    x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range
    vx, vy = voxel_size_xy

    W = int((x_max - x_min) / vx)
    H = int((y_max - y_min) / vy)

    bev = torch.zeros((H, W), dtype=torch.float32)

    x = points[:, 0]
    y = points[:, 1]

    mask = (x >= x_min) & (x < x_max) & (y >= y_min) & (y < y_max)
    x = x[mask]
    y = y[mask]

    ix = ((x - x_min) / vx).floor().long()
    iy = ((y - y_min) / vy).floor().long()

    bev[iy, ix] = 1.0
    return bev

def visualize_pts_label_bev_with_grid(
    pts, label_pts,
    point_cloud_range,
    voxel_size_xy=(0.1, 0.1),
    grid_every=1,          
    grid_alpha=0.25,       
    grid_lw=0.5,           
    save_path=None,        
):

    bev_pts = points_to_bev_occupancy(pts, point_cloud_range, voxel_size_xy)
    bev_label = points_to_bev_occupancy(label_pts, point_cloud_range, voxel_size_xy)

    H, W = bev_pts.shape
    bev_rgb = np.zeros((H, W, 3), dtype=np.float32)

    # base blue
    bev_rgb[bev_pts.numpy() == 1] = np.array([0.2, 0.4, 1.0], dtype=np.float32)
    # label yellow overwrite
    bev_rgb[bev_label.numpy() == 1] = np.array([1.0, 1.0, 0.2], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(bev_rgb, origin="lower")

    ax.set_title("BEV (blue=pts, yellow=label) + grid")
    ax.set_xlabel("x bin")
    ax.set_ylabel("y bin")

    # ---- grid lines (cell boundaries) ----
    ax.set_xticks(np.arange(-0.5, W, grid_every), minor=True)
    ax.set_yticks(np.arange(-0.5, H, grid_every), minor=True)
    ax.grid(which="minor", alpha=grid_alpha, linewidth=grid_lw)
    ax.tick_params(which="minor", bottom=False, left=False)

    if save_path is not None:
        d = os.path.dirname(save_path)
        if d:
            os.makedirs(d, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def visualize_pts_label_bev_before_after(
    pts_raw, label_raw, pts_aug, label_aug,
    point_cloud_range,
    voxel_size_xy=(0.1, 0.1),
    save_path=None,
    pitch_applied=None,
    pitch_fail_reason=None,
):
    bev_raw_pts = points_to_bev_occupancy(pts_raw, point_cloud_range, voxel_size_xy)
    bev_raw_lbl = points_to_bev_occupancy(label_raw, point_cloud_range, voxel_size_xy)
    bev_aug_pts = points_to_bev_occupancy(pts_aug, point_cloud_range, voxel_size_xy)
    bev_aug_lbl = points_to_bev_occupancy(label_aug, point_cloud_range, voxel_size_xy)

    H, W = bev_raw_pts.shape
    rgb_raw = np.zeros((H, W, 3), dtype=np.float32)
    rgb_raw[bev_raw_pts.numpy() == 1] = [0.2, 0.4, 1.0]
    rgb_raw[bev_raw_lbl.numpy() == 1] = [1.0, 1.0, 0.2]

    rgb_aug = np.zeros((H, W, 3), dtype=np.float32)
    rgb_aug[bev_aug_pts.numpy() == 1] = [0.2, 0.4, 1.0]
    rgb_aug[bev_aug_lbl.numpy() == 1] = [1.0, 1.0, 0.2]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), facecolor="white")
    axes[0].imshow(rgb_raw, origin="lower")
    axes[0].set_title("Before Aug (raw)")
    axes[0].set_xlabel("x bin")
    axes[0].set_ylabel("y bin")

    pitch_caption = ""
    if pitch_applied is not None:
        if pitch_applied:
            pitch_caption = " | Pitch: applied"
        else:
            pitch_caption = f" | Pitch: not applied ({pitch_fail_reason or 'unknown'})"
    axes[1].imshow(rgb_aug, origin="lower")
    axes[1].set_title(f"After Aug (rot/flip/pitch){pitch_caption}")
    axes[1].set_xlabel("x bin")
    axes[1].set_ylabel("y bin")

    plt.tight_layout()
    if save_path is not None:
        d = os.path.dirname(save_path)
        if d:
            os.makedirs(d, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()