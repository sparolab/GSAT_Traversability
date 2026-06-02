#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys

import yaml
import pandas as pd
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

from GSAT_Traversability.gsat.data_tools.utils.io import load_bin


def get_pose(translation, quaternion):
    r = R.from_quat(quaternion)
    rotation_matrix = r.as_matrix()

    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation_matrix
    pose[:3, 3] = translation
    return pose


def pitch_roll_rotation_ma(q_xyzw):
    roll, pitch, yaw = R.from_quat(q_xyzw).as_euler("xyz", degrees=False)
    R_rp = R.from_euler("xyz", [roll, pitch, 0.0], degrees=False).as_matrix()
    return R_rp


def transform_global2local(df_next, T_lidar2base=None):
    if df_next.empty:
        return np.zeros((0, 3), dtype=np.float32)

    if T_lidar2base is None:
        T_lidar2base = np.eye(4, dtype=np.float64)

    row0 = df_next.iloc[0]
    t0 = row0[["robot_posi_x", "robot_posi_y", "robot_posi_z"]].to_numpy(dtype=float)
    q0 = row0[["robot_ori_x", "robot_ori_y", "robot_ori_z", "robot_ori_w"]].to_numpy(dtype=float)

    T0_inv = np.linalg.inv(get_pose(t0, q0))

    local_pts = []
    for _, row in df_next.iterrows():
        t = row[["robot_posi_x", "robot_posi_y", "robot_posi_z"]].to_numpy(dtype=float)
        q = row[["robot_ori_x", "robot_ori_y", "robot_ori_z", "robot_ori_w"]].to_numpy(dtype=float)

        T = get_pose(t, q)
        T_local = T0_inv @ T @ T_lidar2base
        local_pts.append(T_local[:3, 3])

    return np.asarray(local_pts, dtype=np.float32)


def load_csv(file_path, time_stamp, save_trajectory_num, around_radius=0.0, filter_range=None):
    df = pd.read_csv(file_path)

    df_next = (
        df[df["TIMESTAMP"] >= time_stamp]
        .sort_values("TIMESTAMP")
        .head(save_trajectory_num)
    )

    if df_next.empty:
        raise ValueError(f"No rows with TIMESTAMP >= {time_stamp}")

    T_lidar2base = np.eye(4, dtype=np.float64)
    local_poses = transform_global2local(df_next, T_lidar2base)  # (K,3)

    travel_labels = df_next["Travel_label"].to_numpy(dtype=np.float32)

    filtered_pts = []
    filtered_lbl = []

    min_r = max(around_radius, filter_range) if filter_range is not None else around_radius
    for i, (x, y, z) in enumerate(local_poses):
        if np.hypot(x, y) >= min_r:
            filtered_pts.append([x, y, 0.0])
            filtered_lbl.append(travel_labels[i])

    if len(filtered_pts) == 0:
        print("Filtered local poses is empty.")
        return np.empty((0, 4), dtype=np.float32)

    filtered_pts = np.asarray(filtered_pts, dtype=np.float32)
    filtered_lbl = np.asarray(filtered_lbl, dtype=np.float32).reshape(-1, 1)

    return np.hstack([filtered_pts, filtered_lbl]).astype(np.float32)


def subsample_pose_pts_by_distance(pose_pts, min_pose_gap=0.2):
    if pose_pts.shape[0] == 0:
        return pose_pts.astype(np.float32)

    kept = [pose_pts[0]]
    last_xy = pose_pts[0, :2]

    for i in range(1, pose_pts.shape[0]):
        cur_xy = pose_pts[i, :2]
        if np.linalg.norm(cur_xy - last_xy) >= min_pose_gap:
            kept.append(pose_pts[i])
            last_xy = cur_xy

    return np.asarray(kept, dtype=np.float32)


def _compute_heading_from_trajectory(pose_pts, idx):
    K = pose_pts.shape[0]

    if K == 1:
        heading = np.array([1.0, 0.0], dtype=np.float32)
    elif idx == 0:
        diff = pose_pts[idx + 1, :2] - pose_pts[idx, :2]
        heading = diff / (np.linalg.norm(diff) + 1e-8)
    elif idx == K - 1:
        diff = pose_pts[idx, :2] - pose_pts[idx - 1, :2]
        heading = diff / (np.linalg.norm(diff) + 1e-8)
    else:
        diff = pose_pts[idx + 1, :2] - pose_pts[idx - 1, :2]
        heading = diff / (np.linalg.norm(diff) + 1e-8)

    return heading.astype(np.float32)


def augment_pose_by_rect_footprint(
    pose_pts,
    robot_width=0.6,
    robot_length=0.8,
    lateral_res=0.15,
    longitudinal_res=0.2,
    unique_xy_round=2
):
    if pose_pts.shape[0] == 0:
        return np.empty((0, 4), dtype=np.float32)

    half_w = robot_width / 2.0
    half_l = robot_length / 2.0

    lat_offsets = np.arange(-half_w, half_w + 1e-6, lateral_res, dtype=np.float32)
    lon_offsets = np.arange(-half_l, half_l + 1e-6, longitudinal_res, dtype=np.float32)

    aug_points = []

    for i in range(pose_pts.shape[0]):
        x, y, z, lbl = pose_pts[i]

        heading = _compute_heading_from_trajectory(pose_pts, i)
        lateral = np.array([-heading[1], heading[0]], dtype=np.float32)

        for lo in lon_offsets:
            for la in lat_offsets:
                px = x + lo * heading[0] + la * lateral[0]
                py = y + lo * heading[1] + la * lateral[1]
                pz = z
                aug_points.append([px, py, pz, lbl])

    aug_points = np.asarray(aug_points, dtype=np.float32)

    # Deduplicate nearby overlapping points
    if unique_xy_round is not None and aug_points.shape[0] > 0:
        xy = np.round(aug_points[:, :2], decimals=unique_xy_round)
        z = np.round(aug_points[:, 2:3], decimals=unique_xy_round)
        lbl = aug_points[:, 3:4]

        merged = np.concatenate([xy, z, lbl], axis=1)
        merged = np.unique(merged, axis=0)
        aug_points = merged.astype(np.float32)

    return aug_points


def _footprint_from_points(footprint_points):   
    if not footprint_points or len(footprint_points) < 2:
        return 1.0, 1.2
    pts = np.array(footprint_points)
    x_range = pts[:, 0].max() - pts[:, 0].min()
    y_range = pts[:, 1].max() - pts[:, 1].min()
    return float(y_range), float(x_range)


def train_data_setting(
    bin_path,
    label_path,
    idx,
    per_pose_num,
    save_path,
    filter_range,
    robot_width,
    robot_length,
    lateral_res,
    longitudinal_res,
    min_pose_gap,
    unique_xy_round,
    around_radius=0.0,
    level_points=True,
    visualize=False
):
    print(f" ==== Processing {bin_path}")
    print(f"======= file_idx : {idx} ==========")

    points = load_bin(bin_path)

    filename = os.path.basename(bin_path)
    timestamp = int(filename.replace(".bin", ""))

    df = pd.read_csv(label_path)
    point_pose = df[df["TIMESTAMP"] == timestamp]

    if point_pose.empty:
        print(f"Skipping file {bin_path} due to missing CSV data for TIMESTAMP={timestamp}")
        return

    # Filter near-range lidar points
    r2 = np.sum(points[:, :2] ** 2, axis=1)
    points = points[r2 > (filter_range ** 2)]

    # Roll/Pitch compensation (keep original when level_points=False)
    if level_points:
        q = point_pose[["robot_ori_x", "robot_ori_y", "robot_ori_z", "robot_ori_w"]].values[0].astype(float)
        R_rp = pitch_roll_rotation_ma(q)
        points_to_save = points.copy()
        points_to_save[:, :3] = points[:, :3] @ R_rp.T
    else:
        points_to_save = points.copy()

    # Save point cloud
    new_name = f"{idx:06d}.bin"

    points_save_dir = os.path.join(save_path, "point")
    os.makedirs(points_save_dir, exist_ok=True)
    points_save_path = os.path.join(points_save_dir, new_name)
    points_to_save.astype(np.float32).tofile(points_save_path)

    # Load future supervision trajectory (supervision is also filtered by filter_range)
    pose_pts = load_csv(
        file_path=label_path,
        time_stamp=timestamp,
        save_trajectory_num=per_pose_num,
        around_radius=around_radius,
        filter_range=filter_range
    )

    if pose_pts.size == 0:
        print("Label grid is empty after filtering. Skipping.")
        return

    original_pose_count = pose_pts.shape[0]

    # Reduce too-dense center poses
    pose_pts = subsample_pose_pts_by_distance(
        pose_pts,
        min_pose_gap=min_pose_gap
    )
    subsampled_pose_count = pose_pts.shape[0]

    # Expand each pose center to footprint rectangle
    pose_pts = augment_pose_by_rect_footprint(
        pose_pts,
        robot_width=robot_width,
        robot_length=robot_length,
        lateral_res=lateral_res,
        longitudinal_res=longitudinal_res,
        unique_xy_round=unique_xy_round
    )

    # Apply filter_range to supervision too: drop points within filter_range of the origin
    if filter_range is not None and filter_range > 0 and pose_pts.shape[0] > 0:
        r2_sup = np.sum(pose_pts[:, :2] ** 2, axis=1)
        keep = r2_sup > (filter_range ** 2)
        pose_pts = pose_pts[keep]

    print(f"Original supervision poses:   {original_pose_count}")
    print(f"Subsampled supervision poses: {subsampled_pose_count}")
    print(f"Augmented supervision points: {pose_pts.shape[0]}")

    # Save label
    label_save_dir = os.path.join(save_path, "label")
    os.makedirs(label_save_dir, exist_ok=True)
    label_save_path = os.path.join(label_save_dir, new_name)
    pose_pts.astype(np.float32).tofile(label_save_path)
    print(f"Saved label grid to: {label_save_path}")

    if visualize:
        traj_xyz = pose_pts[:, :3]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_to_save[:, :3])

        traj_pcd = o3d.geometry.PointCloud()
        traj_pcd.points = o3d.utility.Vector3dVector(traj_xyz)

        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)

        o3d.visualization.draw_geometries([pcd, traj_pcd, frame])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/data_preprocess.yaml")
    parser.add_argument("--key", type=str, default="gazebo_hill")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    #--------[Print an English message and exit if the requested key is missing]--------#
    if args.key not in config:
        print(f"[Error] config key '{args.key}' not found in {args.config}.")
        sys.exit(1)
    cfg = config[args.key]

    input_folder = cfg.get("data_folder", "/test/data_folder")
    output_folder = cfg.get("output_folder", "/test/save_folder")

    filter_range = cfg.get("filter_range", 3.5)
    per_pose_num = cfg.get("per_pose_num", 30)
    around_radius = cfg.get("around_radius", 0.0)
    level_points = cfg.get("level_points", True)

    # footprint: extract from footprint_points or set directly
    fp_pts = cfg.get("footprint_points", [])
    if fp_pts:
        _w, _l = _footprint_from_points(fp_pts)
        robot_width = cfg.get("robot_width", _w)
        robot_length = cfg.get("robot_length", _l)
    else:
        robot_width = cfg.get("robot_width", 0.6)
        robot_length = cfg.get("robot_length", 0.8)

    lateral_res = cfg.get("lateral_res", 0.15)
    longitudinal_res = cfg.get("longitudinal_res", 0.2)
    min_pose_gap = cfg.get("min_pose_gap", 0.2)
    unique_xy_round = cfg.get("unique_xy_round", 2)

    visualize = cfg.get("visualize", False)

    bin_dir = os.path.join(input_folder, "lidar")
    label_path = os.path.join(input_folder, "supervision.csv")

    if not os.path.isdir(bin_dir):
        print(f"Bin directory does not exist: {bin_dir}")
        sys.exit(1)

    if not os.path.isfile(label_path):
        print(f"CSV file does not exist: {label_path}")
        sys.exit(1)

    bin_files = sorted([f for f in os.listdir(bin_dir) if f.endswith(".bin")])
    if not bin_files:
        print(f"No bin files found in {bin_dir}. Exiting...")
        sys.exit(0)

    print(f"=== Found {len(bin_files)} bin files. Processing...")

    for idx, bin_file in enumerate(bin_files):
        if len(bin_files) - per_pose_num <= idx:
            continue

        bin_path = os.path.join(bin_dir, bin_file)

        train_data_setting(
            bin_path=bin_path,
            label_path=label_path,
            idx=idx,
            per_pose_num=per_pose_num,
            save_path=output_folder,
            filter_range=filter_range,
            robot_width=robot_width,
            robot_length=robot_length,
            lateral_res=lateral_res,
            longitudinal_res=longitudinal_res,
            min_pose_gap=min_pose_gap,
            unique_xy_round=unique_xy_round,
            around_radius=around_radius,
            level_points=level_points,
            visualize=visualize
        )


if __name__ == "__main__":
    main()