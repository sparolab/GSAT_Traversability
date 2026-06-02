import os
import sys
import argparse
import yaml
from GSAT_Traversability.gsat.model import center_extract, radius_extract
import torch
import torch.optim as optim
from tqdm import tqdm

from GSAT_Traversability.gsat.utils import setup_seed
from GSAT_Traversability.gsat.dataset import Dataset, get_dataloader, bev_representation
from GSAT_Traversability.gsat.model import geo_feature, gsat_head
from GSAT_Traversability.gsat.loss import loss_gsat
from GSAT_Traversability.gsat.scheduler import CosineAnnealingWarmupRestarts


def main(args):
    setup_seed(42)
    train_dataset = Dataset(
        data_root=args.dataset_dir,
        split='train',
        point_range_filter=args.point_range,
        label_point_range_filter=args.point_range,
    )
    val_dataset   = Dataset(
        data_root=args.dataset_dir,
        split='val',
        point_range_filter=args.point_range,
        label_point_range_filter=args.point_range,
    )
    train_loader = get_dataloader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True
    )
    val_loader = get_dataloader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        # shuffle=False,
        shuffle=True,
        drop_last=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    geo_feautre = geo_feature(
        voxel_size=args.voxel_size,
        point_cloud_range=args.point_range,
    ).to(device)
        
    for p in geo_feautre.parameters():  
        p.requires_grad = True

    head_model = gsat_head().to(device)    
    optimizer = optim.AdamW(
        list(geo_feautre.parameters()) + list(head_model.parameters()),
        lr=args.init_lr,
        weight_decay=1e-5
    )

    gsat_loss = loss_gsat()

    train_steps_per_epoch = len(train_loader)
    total_steps = train_steps_per_epoch * args.max_epoch
    ae_max_lr = args.init_lr

    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=total_steps,
        cycle_mult=1.0,
        max_lr=ae_max_lr,
        min_lr=ae_max_lr*0.1,
        warmup_steps=train_steps_per_epoch*1
    )
    center_c = None
    os.makedirs(args.saved_path, exist_ok=True)
    best_val_loss = float('inf')
    mean_radius_train = 0
    mean_radius_val = 0

    for epoch in range(args.max_epoch):
        print(f"\n===== Epoch {epoch+1}/{args.max_epoch} =====")
        print(f"--> Epoch {epoch+1}")

        if epoch % 5 == 0:
            geo_feautre.eval()
            head_model.eval()
            with torch.no_grad():
                center_c = center_extract(
                    train_loader, geo_feautre, head_model, device,
                    args.point_range, args.voxel_size, latent_dim=16
                )
        mean_radius_train = radius_extract(geo_feautre, head_model, train_loader, args.point_range, args.voxel_size, center_c, device='cuda')

        geo_feautre.train()
        head_model.train()

        train_loss_sum = 0.0
        train_metric_loss_sum = 0.0
        train_reg_loss_sum = 0.0
        train_recon_loss_sum = 0.0
        train_pos_metric_sum = 0.0
        train_unl_norm_metric_sum = 0.0
        train_unl_abnorm_metric_sum = 0.0

        num_batches = len(train_loader)

        for batch_idx, data in enumerate(tqdm(train_loader, desc=f"{epoch+1}/{args.max_epoch}")):
            pts, gt_pts = data["batched_pts"], data["batched_label_pts"]
            pts = [p.to(device) for p in pts]
            gt_pts = [g.to(device) for g in gt_pts]

            fmap, pillars_num, fmap_layer = geo_feautre(pts)
            latent_space, recon_out, recon_in, train_trave_out = head_model(fmap_layer)

            train_reg_in = bev_representation(
                batched_pts=gt_pts,
                point_cloud_range=args.point_range,
                voxel_size=args.voxel_size,
                max_num_points=args.max_num_points,
                max_voxels=(16000, 40000)
            )

            loss, loss_met, loss_reg, loss_rec, pos_met, un_no_met, un_ab_met = gsat_loss(
                train_reg_in, latent_space, center_c,
                train_trave_out, recon_out, fmap, epoch, mean_radius_train
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss_sum += loss.item()
            train_metric_loss_sum += loss_met.item()
            train_reg_loss_sum += loss_reg.item()
            train_recon_loss_sum += loss_rec.item()
            train_pos_metric_sum += pos_met.item()
            train_unl_norm_metric_sum += un_no_met.item()
            train_unl_abnorm_metric_sum += un_ab_met.item()

            avg_train_loss = train_loss_sum / (batch_idx + 1)
            avg_train_metric_loss = train_metric_loss_sum / (batch_idx + 1)
            avg_train_reg_loss = train_reg_loss_sum / (batch_idx + 1)
            avg_train_recon_loss = train_recon_loss_sum / (batch_idx + 1)


        do_val = (epoch + 1) % 2 == 1

        if do_val:
            geo_feautre.eval()
            head_model.eval()

            val_loss_sum = 0.0
            val_metric_loss_sum = 0.0
            val_reg_loss_sum = 0.0
            val_recon_loss_sum = 0.0
            val_pos_metric_sum = 0.0
            val_unl_norm_metric_sum = 0.0
            val_unl_abnorm_metric_sum = 0.0
            mean_radius_val = radius_extract(geo_feautre, head_model, val_loader, args.point_range, args.voxel_size, center_c, device='cuda')

            with torch.no_grad():
                for data in tqdm(val_loader, desc=f"Val {epoch+1}/{args.max_epoch}"):
                    
                    pts, gt_pts = data["batched_pts"], data["batched_label_pts"]
                    pts = [p.to(device) for p in pts]
                    gt_pts = [g.to(device) for g in gt_pts]

                    fmap_val, pillars_num, fmap_layer_val = geo_feautre(pts)
                    latent_space_val, recon_out_val, recon_in_val, travel_out_val = head_model(fmap_layer_val)

                    travel_input_val = bev_representation(
                        batched_pts=gt_pts,
                        point_cloud_range=args.point_range,
                        voxel_size=args.voxel_size,
                        max_num_points=args.max_num_points,
                        max_voxels=(16000, 40000)
                    )

                    mean_radius_val = None

                    loss, loss_met, loss_reg, loss_rec, pos_met, un_no_met, un_ab_met = gsat_loss(
                        travel_input_val, latent_space_val, center_c,
                        travel_out_val, recon_out_val, fmap_val, epoch, mean_radius_val
                    )

                    val_loss_sum += loss.item()
                    val_metric_loss_sum += loss_met.item()
                    val_reg_loss_sum += loss_reg.item()
                    val_recon_loss_sum += loss_rec.item()
                    val_pos_metric_sum += pos_met.item()
                    val_unl_norm_metric_sum += un_no_met.item()
                    val_unl_abnorm_metric_sum += un_ab_met.item()

            num_val_batches = len(val_loader)
            avg_val_loss = val_loss_sum / num_val_batches
            avg_val_metric_loss = val_metric_loss_sum / num_val_batches
            avg_val_reg_loss = val_reg_loss_sum / num_val_batches
            avg_val_recon_loss = val_recon_loss_sum / num_val_batches

            avg_pos_metric_loss_val = val_pos_metric_sum / num_val_batches
            avg_unl_norm_metric_loss_val = val_unl_norm_metric_sum / num_val_batches
            avg_unl_abnorm_metric_loss_val = val_unl_abnorm_metric_sum / num_val_batches

            print(f"Val Epoch {epoch+1} "
                f"Loss: {avg_val_loss:.4f} "
                f"Metric: {avg_val_metric_loss:.4f} "
                f"Reg: {avg_val_reg_loss:.4f} "
                f"Recon: {avg_val_recon_loss:.4f}")

            if (epoch + 1) % args.ckpt_freq_epoch == 0:
                mean_radius_val = radius_extract(
                    geo_feautre, head_model, val_loader,
                    args.point_range, args.voxel_size, center_c, device='cuda'
                )
                torch.save({
                    "epoch": epoch+1,
                    "feature_model" : geo_feautre.state_dict(),
                    "head_model": head_model.state_dict(),
                    "center_c": center_c.cpu(),
                    "mean_radius_val": mean_radius_val,
                    "all_pos_metric_loss": avg_pos_metric_loss_val,
                    "all_unlabel_norm_metric_loss": avg_unl_norm_metric_loss_val,
                    "all_unlabel_abnorm_metric_loss": avg_unl_abnorm_metric_loss_val
                }, os.path.join(args.saved_path, f"epoch_{epoch+1}.pth"))

            if avg_val_loss < best_val_loss:
                mean_radius_val = radius_extract(
                    geo_feautre, head_model, val_loader,
                    args.point_range, args.voxel_size, center_c, device='cuda'
                )
                best_val_loss = avg_val_loss
                torch.save({
                    "epoch": epoch+1,
                    "feature_model" : geo_feautre.state_dict(),
                    "head_model": head_model.state_dict(),
                    "center_c": center_c.cpu(),
                    "mean_radius_val": mean_radius_val,
                    "all_pos_metric_loss": avg_pos_metric_loss_val,
                    "all_unlabel_norm_metric_loss": avg_unl_norm_metric_loss_val,
                    "all_unlabel_abnorm_metric_loss": avg_unl_abnorm_metric_loss_val
                }, os.path.join(args.saved_path, f"best_model_{epoch+1}.pth"))
                print(f" New best! Validation Loss {best_val_loss:.4f} ▶ save: best_model")

    print("===== Training complete! =====")

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/train.yaml')
    parser.add_argument('--key', type=str, default='hill_example',
                        help='top-level config key to use (e.g. hill_example)')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    #--------[Print an English message and exit if the requested key is missing]--------#
    if args.key not in config:
        print(f"[Error] config key '{args.key}' not found in {args.config}.")
        sys.exit(1)
    cfg = config[args.key]

    args.dataset_dir = cfg.get("dataset_dir")
    args.saved_path = cfg.get("save_dir")
    args.batch_size = cfg.get("batch_size")
    args.num_workers = cfg.get("num_workers")
    args.init_lr = cfg.get("init_lr")
    args.max_epoch = cfg.get("max_epoch")
    args.ckpt_freq_epoch = cfg.get("checkpoint_freq_epoch")
    args.point_range = cfg.get("point_range")
    args.voxel_size = cfg.get("voxel_size")
    args.max_num_points = cfg.get("max_num_points")

    print(args.dataset_dir)
    print(args.saved_path)                 

    main(args)   