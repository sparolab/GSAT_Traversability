import torch
import numpy as np

from .geo_feature import ransac_plan_extract, plane_angle_extract, Rx, Ry

def pitch_augment(
    data_dict,
    flag,
    ransac_n_iter: int = 1200,
    ransac_dist: float = 0.013,
    ransac_min_inliers: int = 400,
    ransac_min_nz: float = 0.75,
    plane_dist_thresh: float = 0.07,
    above_plane_h: float = 0.20,
    uniform_pitch_between_minus_and_plus: bool = True, 
    visualize: bool = False
):
    if not flag:
        return data_dict, False, "pitch disabled"

    pts = data_dict['pts']      # (N,>=3) torch.Tensor
    label_pts = data_dict.get('label_pts', None)

    device, dtype = pts.device, pts.dtype
    gen = None

    xyz = pts[:, :3]

    abc, inliers, fail_info = ransac_plan_extract(
        xyz,
        n_iter=ransac_n_iter,
        dist_thresh=ransac_dist,
        min_inliers=ransac_min_inliers,
        min_nz=ransac_min_nz,
        generator=gen
    )
    if abc is None:
        reason = fail_info.get("reason", "unknown")
        bc = fail_info.get("best_cnt", "?")
        mi = fail_info.get("min_inliers", "?")
        if reason == "insufficient inliers":
            msg = f"insufficient inliers (best={bc}, min={mi})"
        elif reason == "too few points":
            msg = "too few points"
        elif reason == "no plane found":
            msg = "no plane found"
        else:
            msg = reason
        return data_dict, False, msg

    a, b, c = abc
    pitch_rad, roll_rad, _, _ = plane_angle_extract(a, b)

    R_roll = Rx(-roll_rad)

    original_pitch = pitch_rad
    target_pitch   = -original_pitch
    u = torch.rand((), device=device, dtype=dtype, generator=gen)
    final_pitch  = (1.0 - 0.9) * original_pitch + u * target_pitch
    delta_pitch  = final_pitch - original_pitch
    R_pitch = Ry(delta_pitch)
    R_total = R_pitch

    pts_rot = pts.clone()
    pts_rot[:, :3] = pts_rot[:, :3] @ R_total

    data_dict['pts'] = pts_rot
    data_dict['label_pts'] = label_pts
    return data_dict, True, None

def flip_augment(data_dict, random_flip_ratio):
    '''
    data_dict: dict(pts, label_pts, gt_labels, gt_names, difficulty)
    random_flip_ratio: float, 0-1
    return: data_dict
    '''
    random_flip_state = np.random.choice([True, False], p=[random_flip_ratio, 1-random_flip_ratio])
    if random_flip_state:
        pts, label_pts = data_dict['pts'], data_dict['label_pts']

        pts[:, 0] = -pts[:, 0] 
        label_pts[:, 0] = -label_pts[:, 0]

        data_dict.update({'label_pts': label_pts})
        data_dict.update({'pts': pts})
    return data_dict

def point_range_filter(data_dict, point_range, key):
    '''
    data_dict: dict(pts, gt_bboxes_3d, gt_labels, gt_names, difficulty)
    point_range: [x1, y1, z1, x2, y2, z2]
    '''
    pts = data_dict[key]
    x1,y1,z1,x2,y2,z2 = point_range
    keep = (
        (pts[:,0] > x1) & (pts[:,1] > y1) & (pts[:,2] > z1) &
        (pts[:,0] < x2) & (pts[:,1] < y2) & (pts[:,2] < z2)
    )
    data_dict[key] = pts[keep]
    return data_dict

def global_rot_scale_trans(data_dict, rot_range, scale_ratio_range, translation_std):
    '''
    data_dict: dict(pts, gt_bboxes_3d, gt_labels, gt_names, difficulty)
    rot_range: [a, b]
    scale_ratio_range: [c, d] 
    translation_std:  [e, f, g]
    return: data_dict
    '''
    pts, label_pts = data_dict['pts'], data_dict['label_pts']
    
    # 1. rotation
    rot_angle = np.random.uniform(rot_range[0], rot_range[1])
    rot_cos, rot_sin = np.cos(rot_angle), np.sin(rot_angle)
    # in fact, - rot_angle
    rot_mat = np.array([[rot_cos, rot_sin], 
                        [-rot_sin, rot_cos]]) # (2, 2)
    
    # 1.2 point rotation
    pts[:, :2] = pts[:, :2] @ rot_mat.T
    label_pts[:, :2] = label_pts[:, :2] @ rot_mat.T

    # 2. scaling
    scale_fator = np.random.uniform(scale_ratio_range[0], scale_ratio_range[1])
    pts[:, :3] *= scale_fator
    label_pts[:, :3] *= scale_fator

    # 3. translation
    trans_factor = np.random.normal(scale=translation_std, size=(1, 3))
    trans_factor[0, 2] = 0.0
    label_pts[:, :3] += trans_factor
    pts[:, :3] += trans_factor

    # 3. translation
    # trans_factor = np.random.normal(scale=translation_std, size=(1, 3))
    # pts[:, :3] += trans_factor
    # label_pts[:, :3] += trans_factor

    # std_z = translation_std[2]
    # trans_factor_z = np.random.normal(scale=std_z)

    # pts[:, 2] += trans_factor_z
    # label_pts[:, 2] += trans_factor_z

    data_dict.update({'label_pts': label_pts})
    data_dict.update({'pts': pts})
    return data_dict

def points_shuffle(data_dict):
    '''
    data_dict: dict(pts, gt_bboxes_3d, gt_labels, gt_names, difficulty)
    '''
    pts = data_dict['pts']
    indices = np.arange(0, len(pts))
    np.random.shuffle(indices)
    pts = pts[indices]
    data_dict.update({'pts': pts})
    return data_dict

def data_augment(data_root, data_dict, data_aug_config, aug_info=None):
    point_range = data_aug_config['point_range_filter']
    data_dict = point_range_filter(data_dict, point_range, "pts")

    point_range = data_aug_config['label_point_range_filter']
    data_dict = point_range_filter(data_dict, point_range, "label_pts")

    pitch_flag = data_aug_config['pitch_flag']
    data_dict, pitch_applied, pitch_fail_reason = pitch_augment(data_dict, pitch_flag)
    if aug_info is not None:
        aug_info['pitch_applied'] = pitch_applied
        aug_info['pitch_fail_reason'] = pitch_fail_reason

    random_flip_ratio = data_aug_config['random_flip_ratio']
    data_dict = flip_augment(data_dict, random_flip_ratio)

    global_rot_scale_trans_config = data_aug_config['global_rot_scale_trans']
    rot_range = global_rot_scale_trans_config['rot_range']
    scale_ratio_range = global_rot_scale_trans_config['scale_ratio_range']
    translation_std = global_rot_scale_trans_config['translation_std']
    data_dict = global_rot_scale_trans(data_dict, rot_range, scale_ratio_range, translation_std)
    data_dict = point_range_filter(data_dict, data_aug_config['point_range_filter'], "pts")
    data_dict = point_range_filter(data_dict, data_aug_config['label_point_range_filter'], "label_pts")
    data_dict = points_shuffle(data_dict)

    return data_dict
