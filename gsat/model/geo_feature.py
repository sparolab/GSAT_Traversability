import torch
import torch.nn as nn

from ops import Voxelization
import torch.nn.functional as F

class PillarLayer(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels):
        super().__init__()
        self.voxel_layer = Voxelization(voxel_size=voxel_size,
                                        point_cloud_range=point_cloud_range,
                                        max_num_points=max_num_points,
                                        max_voxels=max_voxels)

    @torch.no_grad()
    def forward(self, batched_pts):
        '''
        batched_pts: list[tensor], len(batched_pts) = bs
        return: 
               pillars: (p1 + p2 + ... + pb, num_points, c), 
               coors_batch: (p1 + p2 + ... + pb, 1 + 3), 
               num_points_per_pillar: (p1 + p2 + ... + pb, ), (b: batch size)
        '''

        pillars, coors, npoints_per_pillar = [], [], []
        for i, pts in enumerate(batched_pts):
            voxels_out, coors_out, num_points_per_voxel_out = self.voxel_layer(pts) 
            # voxels_out: (max_voxel, num_points, c), coors_out: (max_voxel, 3)
            # num_points_per_voxel_out: (max_voxel, )
            pillars.append(voxels_out)
            coors.append(coors_out.long())
            npoints_per_pillar.append(num_points_per_voxel_out)
        
        pillars = torch.cat(pillars, dim=0) # (p1 + p2 + ... + pb, num_points, c)
        npoints_per_pillar = torch.cat(npoints_per_pillar, dim=0) # (p1 + p2 + ... + pb, )
        coors_batch = []
        for i, cur_coors in enumerate(coors):
            coors_batch.append(F.pad(cur_coors, (1, 0), value=i))
        coors_batch = torch.cat(coors_batch, dim=0) # (p1 + p2 + ... + pb, 1 + 3)
        
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

        self.global_step = 0
            
    def forward(self, pillars, coors_batch, npoints_per_pillar):
        '''
        pillars: (p1 + p2 + ... + pb, num_points, c), c = 4
        coors_batch: (p1 + p2 + ... + pb, 1 + 3)
        npoints_per_pillar: (p1 + p2 + ... + pb, )
        return:  
            - batched_canvas: (bs, out_channel, y_l, x_l)
            - point_num_canvas: (bs, 1, y_l, x_l)  # 
        '''
        device = pillars.device

        # 1. calculate offset to the points center (in each pillar)
        offset_pt_center = pillars[:, :, :3] - torch.sum(pillars[:, :, :3], dim=1, keepdim=True) / npoints_per_pillar[:, None, None]

        # 2. calculate offset to the pillar center
        x_offset_pi_center = pillars[:, :, :1] - (coors_batch[:, None, 1:2] * self.vx + self.x_offset)
        y_offset_pi_center = pillars[:, :, 1:2] - (coors_batch[:, None, 2:3] * self.vy + self.y_offset)

        #===================add===================#

        z = pillars[:, :, 2]  # (N_pillars, N_points)

        valid_mask = (z != 0)
        z_masked = z.masked_fill(~valid_mask, float('nan'))  # (N_pillars, N_points)
        z_mean = torch.nanmean(z_masked, dim=1, keepdim=True)  # (N_pillars, 1)

        z_var = torch.nanmean((z_masked - z_mean) ** 2, dim=1, keepdim=True)
        z_var = torch.nan_to_num(z_var, nan=0.0)
        z_std = torch.sqrt(z_var.clamp(min=1e-6))

        z_std_expanded = z_std.unsqueeze(1).expand(-1, pillars.size(1), -1)    # (N_pillars, N_points, 1)

        z_3std_expanded = 3 * z_std_expanded

        ##real_eval_original_set, 3var## 
        features = torch.cat([pillars[:, :, :3], offset_pt_center, z_3std_expanded], dim=-1)
        ##################

        features[:, :, 0:1] = x_offset_pi_center  # tmp, x -> x-x_mean
        features[:, :, 1:2] = y_offset_pi_center  # tmp, y -> y-y_mean

        # 4. find mask for (0, 0, 0) and update the encoded features
        voxel_ids = torch.arange(0, pillars.size(1)).to(device)
        mask = voxel_ids[:, None] < npoints_per_pillar[None, :]
        mask = mask.permute(1, 0).contiguous()
        features *= mask[:, :, None]

        # 5. embedding
        features = features.permute(0, 2, 1).contiguous()
        features = F.relu(self.bn(self.conv(features)))
        pooling_features = torch.max(features, dim=-1)[0]

        # 6. pillar scatter
        batched_canvas = []
        point_num_canvas = []
        bs = coors_batch[-1, 0] + 1

        for i in range(bs):
            cur_coors_idx = coors_batch[:, 0] == i
            cur_coors = coors_batch[cur_coors_idx, :]
            cur_features = pooling_features[cur_coors_idx]
            cur_num_points = npoints_per_pillar[cur_coors_idx]

            # Feature map canvas
            canvas = torch.zeros((self.x_l, self.y_l, self.out_channel), dtype=torch.float32, device=device)
            canvas[cur_coors[:, 1], cur_coors[:, 2]] = cur_features
            canvas = canvas.permute(2, 1, 0).contiguous()
            batched_canvas.append(canvas)

            # Point num canvas
            point_canvas = torch.zeros((self.x_l, self.y_l, 1), dtype=torch.float32, device=device)
            point_canvas[cur_coors[:, 1], cur_coors[:, 2], 0] = cur_num_points.float()
            point_canvas = point_canvas.permute(2, 1, 0).contiguous()  # (1, H, W)
            point_num_canvas.append(point_canvas)

        batched_canvas = torch.stack(batched_canvas, dim=0)  # (bs, out_channel, y_l, x_l)
        point_num_canvas = torch.stack(point_num_canvas, dim=0)  # (bs, 1, y_l, x_l)

        return batched_canvas, point_num_canvas

class Backbone(nn.Module):
    def __init__(self, in_channel, out_channels, layer_nums, layer_strides=[2]):
        super().__init__()
        assert len(out_channels) == len(layer_nums)
        assert len(out_channels) == len(layer_strides)
        
        self.multi_blocks = nn.ModuleList()
        for i in range(len(layer_strides)):
            blocks = []
            blocks.append(nn.Conv2d(in_channel, out_channels[i], 3, stride=layer_strides[i], bias=False, padding=1))
            blocks.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
            blocks.append(nn.ReLU(inplace=True))

            for _ in range(layer_nums[i]):
                blocks.append(nn.Conv2d(out_channels[i], out_channels[i], 3, bias=False, padding=1))
                blocks.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
                blocks.append(nn.ReLU(inplace=True))

            in_channel = out_channels[i]
            self.multi_blocks.append(nn.Sequential(*blocks))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        out = self.multi_blocks[0](x)
        return out

class Neck(nn.Module):
    def __init__(self, in_channel, out_channel, upsample_stride=1):
        super().__init__()
        self.decoder_block = nn.Sequential(
            nn.ConvTranspose2d(in_channel, 
                               out_channel, 
                               upsample_stride, 
                               stride=upsample_stride,
                               bias=False)
        )
        self.neck_batch = nn.BatchNorm2d(out_channel, eps=1e-3, momentum=0.01)

        self.neck_relu = nn.ReLU(inplace=False)
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def forward(self, x):
        out = self.decoder_block(x)
        out_bn = self.neck_batch(out)
        out_relu = self.neck_relu(out_bn)
        return out, out_relu
    
class geo_feature(nn.Module):
    def __init__(self,    
                voxel_size=[0.15, 0.15, 4],
                point_cloud_range=[-12, -12, -2, 12, 12, 2],
                 max_num_points=32,
                 max_voxels=(16000, 40000)):
        
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pillar_layer = PillarLayer(voxel_size=voxel_size, 
                                        point_cloud_range=point_cloud_range, 
                                        max_num_points=max_num_points, 
                                        max_voxels=max_voxels)

        ##################encoder - output 16##################
        self.pillar_encoder = PillarEncoder(voxel_size=voxel_size, 
                                            point_cloud_range=point_cloud_range, 
                                            in_channel=7,
                                            out_channel=16)

        self.backbone = Backbone(in_channel=16, 
                                 out_channels=[16], 
                                 layer_nums=[3])

        self.neck = Neck(in_channel=16, 
                    out_channel=32, 
                    upsample_stride=1)
        ######################################################

        self.upsample_layer = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, batched_pts):
        pillars, coors_batch, npoints_per_pillar = self.pillar_layer(batched_pts)

        pillar_features, pillars_num = self.pillar_encoder(pillars, coors_batch, npoints_per_pillar)
        xs = self.backbone(pillar_features)
        x, x_out = self.neck(xs)
        fmap_relu = self.upsample_layer(x_out)
        fmap = self.upsample_layer(x)

        return fmap, pillars_num, fmap_relu