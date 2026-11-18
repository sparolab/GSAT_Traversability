#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rebuild per-pose (center-line) labels of a split dataset as robot-footprint labels.

Input : <input_root>/<split>/point/*.bin, <input_root>/<split>/label/*.bin
        label = (N, 4) float32 [x, y, z, tau], one point per future robot pose
Output: <output_root>/<split>/point/*.bin (copied as is)
        <output_root>/<split>/label/*.bin (one point per BEV cell covered by the footprint)

Each pose is expanded to the footprint polygon, oriented along the trajectory heading
estimated from neighbouring poses, and the result is reduced to one label per BEV cell
(max tau, cell center, z = 0). Cells closer than min_label_range to the sensor are dropped.
The input dataset is only read.
"""

import argparse
import os
import shutil
import sys

import numpy as np
import yaml


DEFAULTS = {
    "input_root": "",
    "output_root": "",
    "splits": ["train", "val", "test"],
    "copy_point_dir": True,
    "footprint_points": [[0.67, 0.45], [0.67, -0.45], [-0.67, -0.45], [-0.67, 0.45]],
    "footprint_sample_step_xy": [0.15, 0.15],
    "cell_size_xy": [0.15, 0.15],
    "point_range_xy": [-12.0, -12.0, 12.0, 12.0],
    "min_label_range": 0.0,
}


def load_config(config_path, key):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}
    if key not in config:
        print(f"[Error] config key '{key}' not found in {config_path}.")
        sys.exit(1)
    cfg = dict(DEFAULTS)
    cfg.update({k: v for k, v in config[key].items() if v is not None})
    return cfg


def load_label_bin(path):
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)


def estimate_headings(pose_pts):
    """2D heading of each pose from the neighbouring poses (central difference)."""
    count = pose_pts.shape[0]
    headings = np.zeros((count, 2), dtype=np.float32)
    prev = np.array([1.0, 0.0], dtype=np.float32)

    for i in range(count):
        if count == 1:
            diff = prev
        elif i == 0:
            diff = pose_pts[i + 1, :2] - pose_pts[i, :2]
        elif i == count - 1:
            diff = pose_pts[i, :2] - pose_pts[i - 1, :2]
        else:
            diff = pose_pts[i + 1, :2] - pose_pts[i - 1, :2]

        norm = np.linalg.norm(diff)
        if norm < 1e-8:
            headings[i] = prev
        else:
            headings[i] = (diff / norm).astype(np.float32)
            prev = headings[i]
    return headings


def _on_segment(px, py, ax, ay, bx, by, eps=1e-6):
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    if abs(abx * apy - aby * apx) > eps:
        return False
    dot = apx * abx + apy * aby
    return -eps <= dot <= abx * abx + aby * aby + eps


def _in_polygon(px, py, polygon):
    """Ray casting; points on the boundary count as inside."""
    px, py = float(px), float(py)
    inside = False
    n = polygon.shape[0]
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if _on_segment(px, py, float(x1), float(y1), float(x2), float(y2)):
            return True
        if ((y1 > py) != (y2 > py)) and (px < (x2 - x1) * (py - y1) / ((y2 - y1) + 1e-12) + x1):
            inside = not inside
    return inside


def sample_footprint(footprint_points, step_xy):
    """Grid samples inside the footprint polygon (robot frame, x forward, y left) plus its corners."""
    polygon = np.asarray(footprint_points, dtype=np.float32)
    if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2:
        raise ValueError("footprint_points needs at least 3 [x, y] points.")

    x_min, y_min = polygon.min(axis=0)
    x_max, y_max = polygon.max(axis=0)
    xs = np.arange(x_min, x_max + 1e-6, float(step_xy[0]), dtype=np.float32)
    ys = np.arange(y_min, y_max + 1e-6, float(step_xy[1]), dtype=np.float32)

    samples = [np.array([x, y], dtype=np.float32) for x in xs for y in ys if _in_polygon(x, y, polygon)]
    if not samples:
        samples = [polygon.mean(axis=0, dtype=np.float32)]

    merged = np.vstack([np.asarray(samples, dtype=np.float32), polygon])
    return np.unique(np.round(merged, decimals=6), axis=0).astype(np.float32)


def expand_to_footprint(pose_pts, footprint_xy):
    """Place the footprint samples at every pose, rotated to the pose heading."""
    if pose_pts.shape[0] == 0:
        return np.empty((0, 4), dtype=np.float32)

    headings = estimate_headings(pose_pts)
    expanded = []
    for (cx, cy, cz, tau), h in zip(pose_pts, headings):
        left = np.array([-h[1], h[0]], dtype=np.float32)
        for lx, ly in footprint_xy:
            expanded.append([cx + lx * h[0] + ly * left[0], cy + lx * h[1] + ly * left[1], cz, tau])
    return np.asarray(expanded, dtype=np.float32)


def to_bev_cells(points, cell_size_xy, point_range_xy):
    """One label per BEV cell (max tau), placed at the cell center with z = 0."""
    if points.size == 0:
        return np.empty((0, 4), dtype=np.float32)

    x_min, y_min, x_max, y_max = [float(v) for v in point_range_xy]
    vx, vy = float(cell_size_xy[0]), float(cell_size_xy[1])
    nx, ny = int((x_max - x_min) / vx), int((y_max - y_min) / vy)

    col = np.floor((points[:, 0] - x_min) / vx).astype(np.int32)
    row = np.floor((points[:, 1] - y_min) / vy).astype(np.int32)
    valid = (col >= 0) & (col < nx) & (row >= 0) & (row < ny)

    cell_tau = {}
    for r, c, tau in zip(row[valid], col[valid], points[valid, 3]):
        key = (int(r), int(c))
        if key not in cell_tau or float(tau) > cell_tau[key]:
            cell_tau[key] = float(tau)
    if not cell_tau:
        return np.empty((0, 4), dtype=np.float32)

    cells = [[x_min + (c + 0.5) * vx, y_min + (r + 0.5) * vy, 0.0, tau]
             for (r, c), tau in sorted(cell_tau.items())]
    return np.asarray(cells, dtype=np.float32)


def rebuild_split(cfg, split, footprint_xy):
    src = os.path.join(cfg["input_root"], split)
    dst = os.path.join(cfg["output_root"], split)
    src_label_dir = os.path.join(src, "label")
    if not os.path.isdir(src_label_dir):
        raise FileNotFoundError(f"label folder not found: {src_label_dir}")

    if cfg["copy_point_dir"]:
        src_point_dir = os.path.join(src, "point")
        dst_point_dir = os.path.join(dst, "point")
        os.makedirs(dst_point_dir, exist_ok=True)
        point_files = sorted(f for f in os.listdir(src_point_dir) if f.endswith(".bin"))
        for f in point_files:
            shutil.copy2(os.path.join(src_point_dir, f), os.path.join(dst_point_dir, f))
        print(f"[{split}] copied {len(point_files)} point files")

    dst_label_dir = os.path.join(dst, "label")
    os.makedirs(dst_label_dir, exist_ok=True)
    min_r = float(cfg["min_label_range"])

    n_pose, n_cell = [], []
    for f in sorted(f for f in os.listdir(src_label_dir) if f.endswith(".bin")):
        pose_pts = load_label_bin(os.path.join(src_label_dir, f))
        cells = to_bev_cells(expand_to_footprint(pose_pts, footprint_xy),
                             cfg["cell_size_xy"], cfg["point_range_xy"])
        if min_r > 0:
            cells = cells[np.hypot(cells[:, 0], cells[:, 1]) > min_r]
        cells.astype(np.float32).tofile(os.path.join(dst_label_dir, f))
        n_pose.append(len(pose_pts))
        n_cell.append(len(cells))

    if n_pose:
        print(f"[{split}] {len(n_pose)} label files, per frame: "
              f"{np.mean(n_pose):.1f} poses -> {np.mean(n_cell):.1f} footprint cells")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/rebuild_footprint_labels.yaml")
    parser.add_argument("--key", type=str, default="rellis")
    args = parser.parse_args()

    cfg = load_config(args.config, args.key)
    cfg["input_root"] = os.path.abspath(cfg["input_root"])
    cfg["output_root"] = os.path.abspath(cfg["output_root"])

    if not os.path.isdir(cfg["input_root"]):
        print(f"[Error] input_root not found: {cfg['input_root']}")
        sys.exit(1)
    if os.path.exists(cfg["output_root"]):
        print(f"[Error] output_root already exists, not overwriting: {cfg['output_root']}")
        sys.exit(1)

    footprint_xy = sample_footprint(cfg["footprint_points"], cfg["footprint_sample_step_xy"])
    print(f"input_root : {cfg['input_root']}")
    print(f"output_root: {cfg['output_root']}")
    print(f"footprint samples: {len(footprint_xy)}, min_label_range: {cfg['min_label_range']} m")

    for split in cfg["splits"]:
        rebuild_split(cfg, split, footprint_xy)


if __name__ == "__main__":
    main()
