from __future__ import annotations

import os
import sys
import argparse
import numpy as np
import torch
import open3d as o3d

_GSAT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _GSAT_ROOT not in sys.path:
    sys.path.insert(0, _GSAT_ROOT)

from model import geo_feature, gsat_head

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CONFIG = os.path.join(_SCRIPT_DIR, "config", "infer_anomaly.yaml")

# Visualization colors: Normal (1) / Anomaly (0)
COLOR_NORMAL = np.array([0.0, 0.0, 1.0])   # Blue
COLOR_ANOMALY = np.array([1.0, 0.0, 0.0])  # Red


def load_config(path: str) -> dict:
    defaults = {
        "checkpoint_path": "",
        "input_folder": "",
        "point_range": [-9, -9, -2, 9, 9, 2],
        "voxel_size": [0.15, 0.15, 4],
        "threshold_scale": 1.0,
        "visualize_3d": True,
        "vis_point_size": 2.0,
    }
    if not os.path.isfile(path):
        return defaults
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        return defaults
    for k, v in defaults.items():
        if k not in cfg or cfg[k] is None:
            cfg[k] = v
    return cfg


def load_points_bin(bin_path: str):
    """KITTI format .bin (continuous x,y,z,intensity float32) -> pts (N,4) float32."""
    raw = np.fromfile(bin_path, dtype=np.float32)
    if raw.size % 4 != 0:
        raise ValueError(f"'{bin_path}' size is not a multiple of 4 (size={raw.size}).")
    pts = raw.reshape(-1, 4)
    return pts


def load_model(ckpt_path: str, device: torch.device, point_range: list, voxel_size: list):
    ckpt = torch.load(ckpt_path, map_location=device)
    geo = geo_feature(
        voxel_size=voxel_size,
        point_cloud_range=point_range,
    ).to(device)
    geo.load_state_dict(ckpt["feature_model"])
    geo.eval()
    head = gsat_head().to(device)
    head.load_state_dict(ckpt["head_model"])
    head.eval()
    center_c = ckpt.get("center_c")
    if center_c is not None:
        center_c = center_c.to(device)
    mean_radius_val = ckpt.get("mean_radius_val", None)
    if mean_radius_val is not None and hasattr(mean_radius_val, "item"):
        mean_radius_val = mean_radius_val.item()
    return geo, head, center_c, mean_radius_val


def _latent_anomaly_score(z_flat: torch.Tensor, center_c: torch.Tensor) -> torch.Tensor:
    #--------[Compute per-pixel anomaly score]--------#
    c = center_c.reshape(1, -1).to(device=z_flat.device, dtype=z_flat.dtype)
    return torch.sqrt(torch.sum((z_flat - c) ** 2, dim=1).clamp(min=0.0) + 1e-6)


@torch.no_grad()
def infer_anomaly(bin_path: str, geo, head, center_c,
                   point_range: list, voxel_size: list,
                   device: torch.device, threshold_scale: float,
                   mean_radius_val: float | None):
    """Single .bin file: Load -> ROI filter -> Inference -> Compute pred_map / dist_map.

    Returns:
        pred_map: (H, W) float32, 1=normal/traversable, 0=anomaly
        dist_map: (H, W) float32, anomaly score (larger value means more anomalous)
        thr: used threshold value
        points: ROI filtered points (N, 4)
        H, W: BEV grid size
    """

    # 1. Load point cloud
    points = load_points_bin(bin_path)
    x0, y0, z0, x1, y1, z1 = point_range

    # 2. Region of Interest (ROI) filtering
    mask = (
        (points[:, 0] >= x0) & (points[:, 0] <= x1) &
        (points[:, 1] >= y0) & (points[:, 1] <= y1) &
        (points[:, 2] >= z0) & (points[:, 2] <= z1)
    )
    points = points[mask]

    if len(points) == 0:
        return None

    # 3. Extract Geo Features
    batched = [torch.from_numpy(points).float().to(device)]
    fmap, _, fmap_layer = geo(batched)

    # 4. Extract latent space from Head
    latent, _, _, _ = head(fmap_layer)
    B, C, H, W = latent.shape
    z_flat = latent.permute(0, 2, 3, 1).reshape(-1, C)

    # 5. Compute Anomaly Score (distance)
    dist = _latent_anomaly_score(z_flat=z_flat, center_c=center_c)

    # 6. Reshape to distance map (BEV grid)
    dist_map = dist.view(H, W).cpu().numpy()

    # 7. Determine Threshold
    if mean_radius_val is not None:
        thr = float(threshold_scale * mean_radius_val)
    else:
        thr = float(np.nanmean(dist_map) * threshold_scale)

    # 8. Final prediction (distance < thr is normal(1), otherwise anomaly(0))
    pred_map = (dist_map < thr).astype(np.float32)

    return pred_map, dist_map, thr, points, H, W


#--------[Open3D 3D Point Cloud Anomaly Detection Visualization]--------#
# Mid (1-3 years experience)
def visualize_open3d(pred_map: np.ndarray, title: str, points: np.ndarray,
                     point_range: list, voxel_size: list, show: bool = True,
                     point_size_ref: list | None = None) -> bool:
    H, W = pred_map.shape
    x0, y0, z0, x1, y1, z1 = point_range
    vx, vy, vz = voxel_size

    if not show:
        return False

    # Convert point coordinates to grid indices
    grid_x = np.floor((points[:, 0] - x0) / vx).astype(int)
    grid_y = np.floor((points[:, 1] - y0) / vy).astype(int)

    # Filter indices within grid range
    valid_mask = (grid_x >= 0) & (grid_x < W) & (grid_y >= 0) & (grid_y < H)
    pts_valid = points[valid_mask]
    gx_valid = grid_x[valid_mask]
    gy_valid = grid_y[valid_mask]

    # Get prediction results for each point (1=normal, 0=anomaly)
    pts_preds = pred_map[gy_valid, gx_valid]

    # Create color array
    colors = np.zeros((len(pts_valid), 3))
    colors[pts_preds == 1] = COLOR_NORMAL   # Normal: Blue
    colors[pts_preds == 0] = COLOR_ANOMALY  # Anomaly: Red

    # Create Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_valid[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Add coordinate frame (size: 2.0m)
    coor = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])

    is_exit = [False]

    def next_callback(vis):
        # Save the current point size before closing the window
        if point_size_ref is not None:
            point_size_ref[0] = vis.get_render_option().point_size
        vis.close()
        return False

    def exit_callback(vis):
        is_exit[0] = True
        # Save the current point size before closing the window
        if point_size_ref is not None:
            point_size_ref[0] = vis.get_render_option().point_size
        vis.close()
        return False

    # Create VisualizerWithKeyCallback and configure window
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=f"Open3D - {title} [Space: Next, Esc: Exit]", width=1280, height=720)
    vis.add_geometry(pcd)
    vis.add_geometry(coor)

    # Set point size in renderer (use the referenced size if available)
    if point_size_ref is not None:
        vis.get_render_option().point_size = point_size_ref[0]

    # Register keyboard callbacks
    vis.register_key_callback(32, next_callback)   # Space
    vis.register_key_callback(256, exit_callback)  # Escape

    # Run visualizer
    vis.run()
    vis.destroy_window()

    return is_exit[0]


def infer_folder(input_folder: str, geo, head, center_c,
                  point_range: list, voxel_size: list,
                  threshold_scale: float, mean_radius_val: float | None,
                  device: torch.device, show_3d: bool = True, point_size: float = 2.0):
    bin_files = sorted([f for f in os.listdir(input_folder) if f.endswith(".bin")])
    if not bin_files:
        print(f"[Error] No .bin files found in '{input_folder}'.")
        return

    # Use a list as a mutable reference to persist point size changes across frames
    point_size_ref = [point_size]

    print(">>> Anomaly Inference (No GT labels)")
    for fname in bin_files:
        bin_path = os.path.join(input_folder, fname)
        out = infer_anomaly(
            bin_path, geo, head, center_c,
            point_range, voxel_size, device, threshold_scale, mean_radius_val,
        )
        if out is None:
            print(f"[{fname}] No points inside ROI, skipping")
            continue
        pred_map, dist_map, thr, points, H, W = out

        n_total = pred_map.size
        n_anomaly = int((pred_map == 0).sum())
        n_normal = int((pred_map == 1).sum())

        print(f"[{fname}] thr={thr:.4f}  normal_cells={n_normal}  anomaly_cells={n_anomaly}")

        if show_3d:
            is_exit = visualize_open3d(
                pred_map, fname, points, point_range, voxel_size, 
                show=show_3d, point_size_ref=point_size_ref
            )
            if is_exit:
                print(">>> [Exit] User pressed Esc to stop inference.")
                break


def main():
    parser = argparse.ArgumentParser(description="Anomaly Inference (without GT labels)")
    parser.add_argument("--config", type=str, default=_DEFAULT_CONFIG)
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ckpt_path = os.path.abspath(cfg["checkpoint_path"])
    input_folder = os.path.abspath(cfg["input_folder"])
    point_range = cfg["point_range"]
    voxel_size = cfg["voxel_size"]
    threshold_scale = float(cfg["threshold_scale"])
    show_3d = False if args.no_viz else bool(cfg.get("visualize_3d", True))
    point_size = float(cfg.get("vis_point_size", 2.0))

    if not os.path.isfile(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return
    if not os.path.isdir(input_folder):
        print(f"Input folder not found: {input_folder}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    geo, head, center_c, mean_radius_val = load_model(
        ckpt_path, device, point_range, voxel_size
    )
    if center_c is None:
        print("center_c not found in checkpoint.")
        return

    infer_folder(
        input_folder, geo, head, center_c,
        point_range, voxel_size, threshold_scale, mean_radius_val,
        device, show_3d=show_3d, point_size=point_size,
    )


if __name__ == "__main__":
    main()
