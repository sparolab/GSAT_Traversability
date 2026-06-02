import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from GSAT_Traversability.gsat.ops import Voxelization

import os
import matplotlib
import matplotlib.pyplot as plt


def visualize_feature_map_grid(feature_map, stage_name, start_channel=0, num_channels=16, num_cols=8):
    C = feature_map.shape[1]
    total_channels = min(num_channels, C - start_channel)
    if total_channels <= 0:
        return

    num_rows = (total_channels + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(2 * num_cols, 2 * num_rows))
    axes = np.atleast_2d(axes)

    for i in range(total_channels):
        row, col = i // num_cols, i % num_cols
        ax = axes[row, col]
        ax.imshow(feature_map[0, i + start_channel].detach().cpu().numpy(), cmap='viridis')
        ax.set_title(f"Ch {i + start_channel}", fontsize=6)
        ax.axis("off")

    for i in range(total_channels, num_rows * num_cols):
        row, col = i // num_cols, i % num_cols
        fig.delaxes(axes[row, col])

    fig.suptitle(
        f"{stage_name} (ch {start_channel}~{start_channel + total_channels - 1})",
        fontsize=10
    )
    plt.tight_layout()

    if matplotlib.get_backend().lower() == "agg":
        out_path = os.path.abspath("geo_fmap_viz.png")
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[Viz] Feature map saved (headless): {out_path}")
    else:
        plt.show()


class PillarLayer(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels):
        super().__init__()
        self.voxel_layer = Voxelization(
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            max_num_points=max_num_points,
            max_voxels=max_voxels
        )

    @torch.no_grad()
    def forward(self, batched_pts):
        pillars, coors, npoints_per_pillar = [], [], []

        for i, pts in enumerate(batched_pts):
            voxels_out, coors_out, num_points_per_voxel_out = self.voxel_layer(pts)
            pillars.append(voxels_out)
            coors.append(coors_out.long())
            npoints_per_pillar.append(num_points_per_voxel_out)

        pillars = torch.cat(pillars, dim=0)
        npoints_per_pillar = torch.cat(npoints_per_pillar, dim=0)

        coors_batch = []
        for i, cur_coors in enumerate(coors):
            coors_batch.append(F.pad(cur_coors, (1, 0), value=i))
        coors_batch = torch.cat(coors_batch, dim=0)

        return pillars, coors_batch, npoints_per_pillar


class PillarEncoder(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, in_channel, out_channel):
        super().__init__()
        self.out_channel = out_channel
        self.vx, self.vy = voxel_size[0], voxel_size[1]
        self.x_offset = voxel_size[0] / 2 + point_cloud_range[0]
        self.y_offset = voxel_size[1] / 2 + point_cloud_range[1]
        self.x_l = int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])
        self.y_l = int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])

        self.conv = nn.Conv1d(in_channel, out_channel, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_channel, eps=1e-3, momentum=0.01)

    def forward(self, pillars, coors_batch, npoints_per_pillar):
        device = pillars.device
        num_pillars, num_points, _ = pillars.shape

        point_ids = torch.arange(num_points, device=device).unsqueeze(0)
        mask = point_ids < npoints_per_pillar.unsqueeze(1)
        mask_f = mask.float()

        xyz = pillars[:, :, :3]

        xyz_sum = (xyz * mask_f.unsqueeze(-1)).sum(dim=1, keepdim=True)
        xyz_mean = xyz_sum / npoints_per_pillar.clamp(min=1).view(-1, 1, 1).float()
        offset_pt_center = xyz - xyz_mean

        x_offset_pi_center = pillars[:, :, 0:1] - (
            coors_batch[:, None, 1:2].float() * self.vx + self.x_offset
        )
        y_offset_pi_center = pillars[:, :, 1:2] - (
            coors_batch[:, None, 2:3].float() * self.vy + self.y_offset
        )

        z = pillars[:, :, 2]
        z_sum = (z * mask_f).sum(dim=1, keepdim=True)
        z_mean = z_sum / npoints_per_pillar.clamp(min=1).unsqueeze(1).float()

        z_var = (((z - z_mean) ** 2) * mask_f).sum(dim=1, keepdim=True) / \
                npoints_per_pillar.clamp(min=1).unsqueeze(1).float()

        z_std = torch.sqrt(z_var.clamp(min=1e-6))
        z_3std_expanded = (3.0 * z_std).unsqueeze(1).expand(-1, num_points, -1)

        features = torch.cat([
            x_offset_pi_center,   # dx_pillar
            y_offset_pi_center,   # dy_pillar
            pillars[:, :, 2:3],   # raw z
            offset_pt_center,     # dx_mean, dy_mean, dz_mean
            z_3std_expanded       # z spread
        ], dim=-1)                # (Np, P, 7)

        features = features * mask_f.unsqueeze(-1)

        features = features.permute(0, 2, 1).contiguous()  # (Np, 7, P)
        features = F.relu(self.bn(self.conv(features)))
        pooling_features = torch.max(features, dim=-1)[0]  # (Np, out_channel)

        bs = int(coors_batch[-1, 0].item()) + 1

        batched_canvas = []
        point_num_canvas = []

        for i in range(bs):
            cur_mask = (coors_batch[:, 0] == i)
            cur_coors = coors_batch[cur_mask]
            cur_features = pooling_features[cur_mask]
            cur_num_points = npoints_per_pillar[cur_mask]

            canvas = torch.zeros(
                (self.x_l, self.y_l, self.out_channel),
                dtype=torch.float32,
                device=device
            )
            point_canvas = torch.zeros(
                (self.x_l, self.y_l, 1),
                dtype=torch.float32,
                device=device
            )

            if cur_coors.shape[0] > 0:
                canvas[cur_coors[:, 1], cur_coors[:, 2]] = cur_features
                point_canvas[cur_coors[:, 1], cur_coors[:, 2], 0] = cur_num_points.float()

            canvas = canvas.permute(2, 1, 0).contiguous()
            point_canvas = point_canvas.permute(2, 1, 0).contiguous()

            batched_canvas.append(canvas)
            point_num_canvas.append(point_canvas)

        batched_canvas = torch.stack(batched_canvas, dim=0)
        point_num_canvas = torch.stack(point_num_canvas, dim=0)

        return batched_canvas, point_num_canvas


class Backbone(nn.Module):
    def __init__(self, in_channel, out_channels, layer_nums, layer_strides=[2]):
        super().__init__()
        assert len(out_channels) == len(layer_nums)
        assert len(out_channels) == len(layer_strides)

        self.multi_blocks = nn.ModuleList()
        for i in range(len(layer_strides)):
            blocks = []
            blocks.append(
                nn.Conv2d(in_channel, out_channels[i], 3,
                          stride=layer_strides[i], bias=False, padding=1)
            )
            blocks.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
            blocks.append(nn.ReLU(inplace=True))

            for _ in range(layer_nums[i]):
                blocks.append(nn.Conv2d(out_channels[i], out_channels[i], 3, bias=False, padding=1))
                blocks.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
                blocks.append(nn.ReLU(inplace=True))

            in_channel = out_channels[i]
            self.multi_blocks.append(nn.Sequential(*blocks))

    def forward(self, x):
        outs = []
        for i in range(len(self.multi_blocks)):
            x = self.multi_blocks[i](x)
            outs.append(x)
        return outs


class ChannelScaleAttention2D(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # channel attention
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid()
        )

        # scale attention (scalar)
        self.scale_mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        pooled = self.avg_pool(x).view(b, c)

        ch_att = self.channel_mlp(pooled).view(b, c, 1, 1)
        sc_att = self.scale_mlp(pooled).view(b, 1, 1, 1)

        att = ch_att * sc_att
        out = x * (1.0 + att)   # residual attention

        return out, ch_att, sc_att


class ChannelScaleAttentionNeck(nn.Module):
    def __init__(self, in_channels, upsample_strides, out_channels, fusion_out=None, reduction=8):
        super().__init__()
        assert len(in_channels) == len(upsample_strides)
        assert len(upsample_strides) == len(out_channels)

        self.decoder_blocks = nn.ModuleList()
        self.att_blocks = nn.ModuleList()

        for i in range(len(in_channels)):
            block = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels[i],
                    out_channels[i],
                    upsample_strides[i],
                    stride=upsample_strides[i],
                    bias=False
                ),
                nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True)
            )
            self.decoder_blocks.append(block)
            self.att_blocks.append(
                ChannelScaleAttention2D(out_channels[i], reduction=reduction)
            )

        total_out = sum(out_channels)
        if fusion_out is None:
            fusion_out = total_out

        self.fuse = nn.Sequential(
            nn.Conv2d(total_out, fusion_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(fusion_out, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True)
        )

    def forward(self, xs, return_attention=False):
        outs = []
        att_info = []

        for i in range(len(self.decoder_blocks)):
            xi = self.decoder_blocks[i](xs[i])                 # upsample to same size
            xi_att, ch_att, sc_att = self.att_blocks[i](xi)   # channel + scale attention
            outs.append(xi_att)

            att_info.append({
                "channel_att": ch_att,   # (B, C, 1, 1)
                "scale_att": sc_att      # (B, 1, 1, 1)
            })

        fused_raw = torch.cat(outs, dim=1)
        fused_relu = self.fuse(fused_raw)

        if return_attention:
            return fused_raw, fused_relu, att_info
        return fused_raw, fused_relu


class geo_feature(nn.Module):
    def __init__(self,
                 voxel_size=[0.15, 0.15, 4],
                 point_cloud_range=[-12, -12, -2, 12, 12, 2],
                 max_num_points=32,
                 max_voxels=(16000, 40000)):

        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.pillar_layer = PillarLayer(
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            max_num_points=max_num_points,
            max_voxels=max_voxels
        )

        self.pillar_encoder = PillarEncoder(
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            in_channel=7,
            out_channel=16
        )

        self.backbone = Backbone(
            in_channel=16,
            out_channels=[16, 32, 64],
            layer_nums=[3, 5, 5],
            layer_strides=[1, 2, 2]
        )

        # channel + scale attention neck
        self.neck = ChannelScaleAttentionNeck(
            in_channels=[16, 32, 64],
            upsample_strides=[1, 2, 4],
            out_channels=[32, 32, 32],
            fusion_out=96,
            reduction=8
        )

        self.visualize_fmap = False
        self.return_attention = False

    def forward(self, batched_pts):
        pillars, coors_batch, npoints_per_pillar = self.pillar_layer(batched_pts)

        pillar_features, pillars_num = self.pillar_encoder(
            pillars, coors_batch, npoints_per_pillar
        )

        xs = self.backbone(pillar_features)

        if self.return_attention:
            fmap, fmap_relu, att_info = self.neck(xs, return_attention=True)
        else:
            fmap, fmap_relu = self.neck(xs)

        if self.visualize_fmap:
            visualize_feature_map_grid(
                fmap_relu,
                stage_name="Channel+Scale Attention Multi-Scale Feature Map",
                start_channel=0,
                num_channels=min(96, fmap_relu.shape[1]),
                num_cols=12
            )

        if self.return_attention:
            return fmap, pillars_num, fmap_relu, att_info

        return fmap, pillars_num, fmap_relu