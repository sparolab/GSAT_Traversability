import torch

def ransac_plan_extract(
    xyz, n_iter=400, dist_thresh=0.05, min_inliers=800,
    min_nz=0.7, eval_subsample=5000, generator=None
):
    N = xyz.shape[0]
    if N < 3:
        return None, None, {"reason": "too few points", "best_cnt": 0}

    device = xyz.device
    if eval_subsample and N > eval_subsample:
        eval_idx = torch.randint(0, N, (eval_subsample,), device=device, generator=generator)
        xyz_eval = xyz[eval_idx]
    else:
        xyz_eval = xyz
    early_exit_thresh = int(min_inliers * 3)
    best_mask, best_cnt = None, -1
    with torch.no_grad():
        for _ in range(n_iter):
            idx = torch.randint(0, N, (3,), device=device, generator=generator)
            p1, p2, p3 = xyz[idx]
            n = torch.cross(p2 - p1, p3 - p1)
            n_norm = torch.linalg.norm(n)
            if n_norm < 1e-9:
                continue
            n /= n_norm
            if torch.abs(n[2]) < min_nz:
                continue
            d = -torch.dot(n, p1)

            dist = torch.abs(xyz_eval @ n + d)
            mask_eval = dist < dist_thresh
            cnt = mask_eval.sum()
            if cnt > best_cnt:
                best_cnt = cnt
                dist_full = torch.abs(xyz @ n + d)
                best_mask = dist_full < dist_thresh
                if best_cnt > early_exit_thresh:
                    break

    bc = max(0, int(best_cnt.item())) if hasattr(best_cnt, "item") else max(0, int(best_cnt)) if best_cnt is not None else 0
    if best_mask is None or bc < min_inliers:
        reason = "insufficient inliers" if best_mask is not None else "no plane found"
        return None, None, {"reason": reason, "best_cnt": bc, "min_inliers": min_inliers}

    in_xyz = xyz[best_mask]
    X = torch.stack([in_xyz[:, 0], in_xyz[:, 1], torch.ones_like(in_xyz[:, 0])], dim=1)
    sol = torch.linalg.lstsq(X, in_xyz[:, 2].unsqueeze(1)).solution.squeeze(1)
    return (sol[0], sol[1], sol[2]), best_mask, None

def plane_angle_extract(a: torch.Tensor, b: torch.Tensor):
    n = torch.stack([a, b, -torch.ones((), device=a.device, dtype=a.dtype)])
    n = n / torch.linalg.norm(n)
    if n[2] < 0:
        n = -n
    pitch_rad = torch.atan2(n[0], n[2])   # about +y
    roll_rad  = torch.atan2(-n[1], n[2])  # about +x
    theta_rad = torch.atan(torch.sqrt(a*a + b*b))
    aspect_rad = torch.atan2(b, a)
    return pitch_rad, roll_rad, theta_rad, aspect_rad

def Rx(angle_rad: torch.Tensor):
    c, s = torch.cos(angle_rad), torch.sin(angle_rad)
    return torch.stack([
        torch.stack([torch.ones_like(c), torch.zeros_like(c), torch.zeros_like(c)], dim=-1),
        torch.stack([torch.zeros_like(c), c, -s], dim=-1),
        torch.stack([torch.zeros_like(c), s,  c], dim=-1),
    ], dim=-2)

def Ry(angle: torch.Tensor):
    c, s = torch.cos(angle), torch.sin(angle)
    R = torch.empty((3,3), device=angle.device, dtype=angle.dtype)
    R[0,0]=c; R[0,1]=0; R[0,2]=s
    R[1,0]=0; R[1,1]=1; R[1,2]=0
    R[2,0]=-s; R[2,1]=0; R[2,2]=c
    return R