import os
import sys
import argparse
import numpy as np
import torch
import yaml
import open3d as o3d
import matplotlib.pyplot as plt
from plyfile import PlyData
from sklearn.metrics import roc_auc_score

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# model/ uses both `from ops import ...` and `from GSAT_Traversability.gsat... import ...`,
# so both the gsat dir and the src dir (two levels up) must be importable.
_SRC_DIR = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
for _p in (_SRC_DIR, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from anomaly_inference import load_model  # geo_feature + gsat_head from a train.py checkpoint

_DEFAULT_CONFIG = os.path.join(_SCRIPT_DIR, "config", "quantitative_eval.yaml")


def load_config(path):
    defaults = {
        "checkpoint_path": "",
        "sem_folder": "",
        "point_range": [-12, -12, -2, 12, 12, 2],
        "voxel_size": [0.15, 0.15, 4],
        "traversable_ids": [1],
        "show_metrics": True,
        "visualize_bev": False,
        "visualize_3d": False,
        "device": None,
    }
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    for k, v in defaults.items():
        if cfg.get(k) is None:
            cfg[k] = v
    return cfg


def load_semantic_ply(ply_path):
    plydata = PlyData.read(ply_path)
    if 'vertex' not in [e.name for e in plydata.elements]:
        raise ValueError(f"'{ply_path}' has no vertex element.")
    vertex = plydata['vertex']
    names = vertex.data.dtype.names
    for fld in ('x', 'y', 'z', 'intensity', 'label'):
        if fld not in names:
            raise ValueError(f"'{ply_path}' vertex is missing field '{fld}'.")

    data = vertex.data
    x_arr = np.array(data['x'], dtype=np.float32)
    y_arr = np.array(data['y'], dtype=np.float32)
    z_arr = np.array(data['z'], dtype=np.float32)
    i_arr = np.array(data['intensity'], dtype=np.float32)
    lbl_arr = np.array(data['label'], dtype=np.int32)

    pts = np.stack((x_arr, y_arr, z_arr, i_arr), axis=1)
    return pts, lbl_arr


def semantic_to_bev_mask(points, sem_labels,
                         pc_range, voxel_size, grid_h, grid_w,
                         traversable_ids):

    xmin, ymin, zmin, xmax, ymax, zmax = pc_range
    vx, vy, vz = voxel_size

    # 1) Keep only points inside pc_range
    mask = (
        (points[:, 0] >= xmin) & (points[:, 0] <  xmax) &
        (points[:, 1] >= ymin) & (points[:, 1] <  ymax) &
        (points[:, 2] >= zmin) & (points[:, 2] <  zmax)
    )
    pts = points[mask]
    lbl = sem_labels[mask]

    # 2) Compute BEV indices
    j_idxs = np.floor((pts[:, 0] - xmin) / vx).astype(np.int32)  # column index
    i_idxs = np.floor((pts[:, 1] - ymin) / vy).astype(np.int32)  # row index

    # 3) Keep only indices inside the grid
    valid = (
        (i_idxs >= 0) & (i_idxs < grid_h) &
        (j_idxs >= 0) & (j_idxs < grid_w)
    )
    i_idxs = i_idxs[valid]
    j_idxs = j_idxs[valid]
    lbl    = lbl[valid]

    cell_has_point = np.zeros((grid_h, grid_w), dtype=bool)
    cell_any_zero  = np.zeros((grid_h, grid_w), dtype=bool)

    for yi, xi, sem_val in zip(i_idxs, j_idxs, lbl):
        cell_has_point[yi, xi] = True  # cell contains a point
        if sem_val not in traversable_ids:
            cell_any_zero[yi, xi] = True  # cell contains a non-traversable point

    gt_mask = np.full((grid_h, grid_w), fill_value=-1, dtype=np.int8)  # -1 = unknown (no points)

    mask_non_trav = np.logical_and(cell_has_point, cell_any_zero)  # has points, at least one non-traversable
    gt_mask[mask_non_trav] = 0

    mask_all_trav = np.logical_and(cell_has_point, ~cell_any_zero)  # has points, all traversable
    gt_mask[mask_all_trav] = 1

    return gt_mask


def compute_metrics(pred_map, gt_map, score_map=None):
    valid_mask = (gt_map != -1)
    pred = pred_map[valid_mask].flatten()  # (N_valid,)
    gt   = gt_map[valid_mask].flatten()    # (N_valid,)

    TP = np.logical_and(pred == 1, gt == 1).sum()
    TN = np.logical_and(pred == 0, gt == 0).sum()
    FP = np.logical_and(pred == 1, gt == 0).sum()
    FN = np.logical_and(pred == 0, gt == 1).sum()

    precision = TP / (TP + FP + 1e-8)
    recall    = TP / (TP + FN + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    iou       = TP / (TP + FP + FN + 1e-8)

    # TPR (= recall), FPR
    tpr = recall
    fpr = FP / (FP + TN + 1e-8)

    # FNR = 1 - TPR = FN / (FN + TP)
    fnr = FN / (FN + TP + 1e-8)

    if score_map is not None:
        scores = score_map[valid_mask].flatten()
        try:
            auroc = roc_auc_score(gt, scores)
        except ValueError:
            auroc = float("nan")
    else:
        auroc = None

    return {
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
        'tpr': tpr,
        'fpr': fpr,
        'fnr': fnr,
        'auroc': auroc
    }


def visualize_3d_pointcloud(points, sem_labels, pred_map, gt_mask,
                            pc_range, voxel_size, grid_h, grid_w):
    xmin, ymin, zmin, xmax, ymax, zmax = pc_range
    vx, vy, vz = voxel_size

    mask3d = (
        (points[:, 0] >= xmin) & (points[:, 0] <= xmax) &
        (points[:, 1] >= ymin) & (points[:, 1] <= ymax) &
        (points[:, 2] >= zmin) & (points[:, 2] <= zmax)
    )
    pts3d = points[mask3d]

    xs = pts3d[:, 0]
    ys = pts3d[:, 1]
    j_idxs = np.floor((xs - xmin) / vx).astype(np.int32)
    i_idxs = np.floor((ys - ymin) / vy).astype(np.int32)

    valid_bev = (
        (i_idxs >= 0) & (i_idxs < grid_h) &
        (j_idxs >= 0) & (j_idxs < grid_w)
    )

    N = pts3d.shape[0]
    colors = np.zeros((N, 3), dtype=np.float32)

    for idx in range(N):
        if not valid_bev[idx]:
            colors[idx] = np.array([0.0, 0.0, 0.0])
            continue
        i = i_idxs[idx]
        j = j_idxs[idx]

        gt_val   = gt_mask[i, j]
        pred_val = pred_map[i, j]

        if   (gt_val == 1 and pred_val == 1):
            colors[idx] = np.array([0.0, 0.0, 1.0])   # blue: True Positive
        elif (gt_val == 0 and pred_val == 0):
            colors[idx] = np.array([0.5, 0.5, 0.5])   # gray: True Negative
        elif (gt_val == 1 and pred_val == 0):
            colors[idx] = np.array([1.0, 1.0, 0.0])   # yellow: False Negative
        elif (gt_val == 0 and pred_val == 1):
            colors[idx] = np.array([1.0, 0.0, 0.0])   # red: False Positive

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts3d[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="3D Prediction vs GT", width=800, height=600)
    vis.add_geometry(pcd)

    def close_on_space(vis):
        vis.close()
        return False

    vis.register_key_callback(ord(" "), close_on_space)
    print("3D visualization")
    vis.run()
    vis.destroy_window()


@torch.no_grad()
def infer_and_evaluate_semantic(semantic_ply_path, geo, head, center_c, thr, cfg, device,
                                visualize_bev=False, visualize_3d=False):
    pc_range = cfg["point_range"]
    voxel_size = cfg["voxel_size"]
    traversable_ids = set(cfg["traversable_ids"])

    points_sem, sem_labels = load_semantic_ply(semantic_ply_path)

    x0, y0, z0, x1, y1, z1 = pc_range
    mask3d = (
        (points_sem[:, 0] >= x0) & (points_sem[:, 0] <= x1) &
        (points_sem[:, 1] >= y0) & (points_sem[:, 1] <= y1) &
        (points_sem[:, 2] >= z0) & (points_sem[:, 2] <= z1)
    )
    points_sem = points_sem[mask3d]
    sem_labels = sem_labels[mask3d]
    if len(points_sem) == 0:
        return None

    # Same forward path as training: geo_feature -> fmap_layer -> gsat_head -> latent
    batched = [torch.from_numpy(points_sem).float().to(device)]
    _, _, fmap_layer = geo(batched)
    latent, _, _, _ = head(fmap_layer)

    _, C, grid_h, grid_w = latent.shape
    z_flat = latent.permute(0, 2, 3, 1).reshape(-1, C)
    dists = torch.sqrt(torch.sum((z_flat - center_c.view(1, -1)) ** 2, dim=1))
    dists_np = dists.view(grid_h, grid_w).cpu().numpy()

    # Inside the hypersphere -> traversable (1)
    pred_map = (dists_np <= thr).astype(np.uint8)

    gt_mask = semantic_to_bev_mask(
        points=points_sem,
        sem_labels=sem_labels,
        pc_range=pc_range,
        voxel_size=voxel_size,
        grid_h=grid_h,
        grid_w=grid_w,
        traversable_ids=traversable_ids
    )

    # Smaller distance means traversable (gt=1), so pass the negative distance as the AUROC score
    metrics = compute_metrics(pred_map, gt_mask, score_map=-dists_np)

    if visualize_bev:
        plt.figure(figsize=(8, 4))
        plt.suptitle(os.path.basename(semantic_ply_path))
        plt.subplot(1, 2, 1)
        plt.imshow(pred_map, cmap='gray', origin='lower')
        plt.title("Predicted (BEV)")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(gt_mask, cmap='gray', origin='lower')
        plt.title("GT (BEV)")
        plt.axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    if visualize_3d:
        visualize_3d_pointcloud(
            points_sem, sem_labels, pred_map, gt_mask,
            pc_range, voxel_size, grid_h, grid_w
        )

    return pred_map, gt_mask, metrics


def report_metrics(sem_folder, ply_files, geo, head, center_c, thr, cfg, device):
    """Compute and print per-file / macro / micro metrics. Returns the files that were evaluated."""
    per_file_metrics = {}
    total_TP = total_TN = total_FP = total_FN = 0

    precisions, recalls, f1s, ious = [], [], [], []
    tprs, fprs, fnrs, aurocs = [], [], [], []

    print(">>> Computing metrics")

    for fname in ply_files:
        ply_path = os.path.join(sem_folder, fname)

        out = infer_and_evaluate_semantic(ply_path, geo, head, center_c, thr, cfg, device)
        if out is None:
            print(f"[{fname}] No points inside point_range, skipping")
            continue
        _, _, m = out
        per_file_metrics[fname] = m

        total_TP += m['TP']
        total_TN += m['TN']
        total_FP += m['FP']
        total_FN += m['FN']

        precisions.append(m['precision'])
        recalls.append(m['recall'])
        f1s.append(m['f1'])
        ious.append(m['iou'])
        tprs.append(m['tpr'])
        fprs.append(m['fpr'])
        fnrs.append(m['fnr'])
        if m['auroc'] is not None:
            aurocs.append(m['auroc'])

    if not per_file_metrics:
        print("[Error] No file could be evaluated.")
        return []

    print("\n=== Per-file Metrics ===")
    for fname, m in per_file_metrics.items():
        auroc_str = f"{m['auroc']:.3f}" if m['auroc'] is not None else "N/A"
        print(f"[{fname}]  "
              f"Prec:{m['precision']:.3f}, Rec:{m['recall']:.3f}, "
              f"F1:{m['f1']:.3f}, IoU:{m['iou']:.3f}, "
              f"TPR:{m['tpr']:.3f}, FPR:{m['fpr']:.3f}, AUROC:{auroc_str}")

    # Report everything in %
    precisions = np.array(precisions) * 100
    recalls    = np.array(recalls) * 100
    f1s        = np.array(f1s) * 100
    ious       = np.array(ious) * 100
    tprs       = np.array(tprs) * 100
    fprs       = np.array(fprs) * 100
    fnrs       = np.array(fnrs) * 100
    aurocs     = np.array(aurocs) * 100

    def mean_std(arr):
        return arr.mean(), arr.std()

    def safe_mean(arr):
        return (arr.mean() if arr.size > 0 else float('nan'))

    p_mu, p_sigma = mean_std(precisions)
    r_mu, r_sigma = mean_std(recalls)
    f_mu, f_sigma = mean_std(f1s)

    # Macro (mean over files)
    print("\n=== Overall Results ===")
    print(f"Precision [%]:  μ={p_mu:.2f}  σ={p_sigma:.2f}")
    print(f"Recall    [%]:  μ={r_mu:.2f}  σ={r_sigma:.2f}")
    print(f"F1-score  [%]:  μ={f_mu:.2f}  σ={f_sigma:.2f}")
    print(f"IoU [%]:      μ={safe_mean(ious):.2f}")
    print(f"TPR [%]:      μ={safe_mean(tprs):.2f}")
    print(f"FPR [%]:      μ={safe_mean(fprs):.2f}")
    print(f"FNR [%]:      μ={safe_mean(fnrs):.2f}")
    print(f"AUROC [%]:    μ={safe_mean(aurocs):.2f}" if aurocs.size > 0 else "AUROC: N/A")

    # Micro (computed from the summed TP/TN/FP/FN over all files)
    def _safe_div(a, b): return a / (b + 1e-8)

    micro_precision = _safe_div(total_TP, total_TP + total_FP)
    micro_recall    = _safe_div(total_TP, total_TP + total_FN)
    micro_f1        = _safe_div(2*micro_precision*micro_recall, micro_precision + micro_recall)
    micro_iou       = _safe_div(total_TP, total_TP + total_FP + total_FN)
    micro_fpr       = _safe_div(total_FP, total_FP + total_TN)
    micro_fnr       = 1.0 - micro_recall

    print(f"\n[Micro] Precision={micro_precision*100:.2f}%, "
          f"Recall={micro_recall*100:.2f}%, F1={micro_f1*100:.2f}%, IoU={micro_iou*100:.2f}%")
    print(f"[Micro] FPR={micro_fpr*100:.2f}%, FNR={micro_fnr*100:.2f}%")

    print(f"\nTotal TP={total_TP}, FP={total_FP}, FN={total_FN}, TN={total_TN}")
    return list(per_file_metrics)


def evaluate_semantic_folder(sem_folder, geo, head, center_c, thr, cfg, device,
                             show_metrics=True, visualize_bev=False, visualize_3d=False):
    ply_files = sorted([f for f in os.listdir(sem_folder) if f.endswith('.ply')])
    if not ply_files:
        print(f"[Error] No .ply files found in '{sem_folder}'.")
        return

    if show_metrics:
        ply_files = report_metrics(sem_folder, ply_files, geo, head, center_c, thr, cfg, device)
        if not ply_files:
            return

    if not (visualize_bev or visualize_3d):
        if not show_metrics:
            print("show_metrics, visualize_bev and visualize_3d are all false; nothing to do.")
        return

    print("\n>>> Visualizing")
    for fname in ply_files:
        ply_path = os.path.join(sem_folder, fname)
        print(f"\n>> Visualizing: {fname}")
        infer_and_evaluate_semantic(
            ply_path, geo, head, center_c, thr, cfg, device,
            visualize_bev=visualize_bev,
            visualize_3d=visualize_3d
        )


def main():
    parser = argparse.ArgumentParser(description="Traversability metrics on semantic .ply files")
    parser.add_argument("--config", type=str, default=_DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="override checkpoint_path in the config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.checkpoint:
        cfg["checkpoint_path"] = args.checkpoint

    ckpt_path = os.path.abspath(cfg["checkpoint_path"])
    sem_folder = os.path.abspath(cfg["sem_folder"])
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not os.path.isdir(sem_folder):
        raise FileNotFoundError(f"Semantic folder not found: {sem_folder}")

    device = torch.device(cfg["device"] or ("cuda" if torch.cuda.is_available() else "cpu"))

    # point_range / voxel_size must match the values used in train.yaml
    geo, head, center_c, radius = load_model(ckpt_path, device, cfg["point_range"], cfg["voxel_size"])
    if center_c is None or radius is None:
        raise KeyError("Checkpoint must contain 'center_c' and 'mean_radius_val'.")

    # Classification threshold = the hypersphere radius saved with the checkpoint
    thr = float(radius)

    print(f"checkpoint : {ckpt_path}")
    print(f"sem_folder : {sem_folder}")
    print(f"radius     : {radius:.4f}")

    evaluate_semantic_folder(
        sem_folder, geo, head, center_c, thr, cfg, device,
        show_metrics=bool(cfg["show_metrics"]),
        visualize_bev=bool(cfg["visualize_bev"]),
        visualize_3d=bool(cfg["visualize_3d"]),
    )


if __name__ == "__main__":
    main()
