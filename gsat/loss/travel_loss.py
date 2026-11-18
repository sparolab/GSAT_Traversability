import torch
import torch.nn as nn
import torch.nn.functional as F


class loss_gsat(nn.Module):
    """Anomaly loss (positive / unlabeled-normal pull, unlabeled-anomaly push) + regression + reconstruction."""
    def __init__(self, eta=1.0, eps=1e-6, alpha=1.0):
        super().__init__()
        self.eta = eta
        self.eps = eps
        self.alpha = alpha

    def forward(self, travel_input, latent, center, travel_output,
                recon_output, recon_input, epoch, radius):
        # Distance to the hypersphere center
        B, C, H, W = latent.shape
        z_flat   = latent.permute(0, 2, 3, 1).reshape(-1, C)
        tgt_flat = travel_input.view(-1).float()

        distance_batch = torch.sqrt(torch.sum((z_flat - center)**2, dim=1))

        mask_pos = (tgt_flat != -1)
        mask_unl = ~mask_pos

        # 1) Anomaly loss
        device = latent.device

        pos_metric_loss = (
            distance_batch[mask_pos].mean()
            if mask_pos.any() else torch.tensor(0.0, device=device)
        )

        # Unlabeled cells inside the radius are normal (pulled), outside are anomalous (pushed)
        thr = radius
        mask_anom_full = mask_unl & (distance_batch > thr)
        mask_nom_full  = mask_unl & (distance_batch < thr)

        unlabel_norm_loss_metric = (
            distance_batch[mask_nom_full].mean()
            if mask_nom_full.any() else torch.tensor(0.0, device=device)
        )

        unlabel_abnorm_loss_metric = (
            (1.0 / (distance_batch[mask_anom_full] + self.eps)).mean()
            if mask_anom_full.any() else torch.tensor(0.0, device=device)
        )

        anom_loss = pos_metric_loss + unlabel_norm_loss_metric + unlabel_abnorm_loss_metric

        # 2) Regression loss: positive -> tau, anomalous -> 0
        target_for_loss = travel_input.clone().float()
        target_for_loss[target_for_loss == -1] = 0.0
        loss_travel_flat = F.mse_loss(travel_output.view(-1), target_for_loss.view(-1), reduction="none")

        loss_travel_positive = (
            loss_travel_flat[mask_pos].mean()
            if mask_pos.any() else torch.tensor(0.0, device=device)
        )

        travel_loss_anom = (
            loss_travel_flat[mask_anom_full].mean()
            if mask_anom_full.any() else torch.tensor(0.0, device=device)
        )

        travel_loss = loss_travel_positive + travel_loss_anom

        # 3) Reconstruction loss on positive cells
        # recon_input is the head input; detached so the backbone cannot lower this loss by moving the target
        loss_reco_all  = F.mse_loss(recon_output, recon_input.detach(), reduction="none")
        loss_reco_map  = loss_reco_all.mean(dim=1)
        loss_reco_flat = loss_reco_map.reshape(-1)
        recon_loss = (
            loss_reco_flat[mask_pos].mean()
            if mask_pos.any() else torch.tensor(0.0, device=latent.device)
        )

        # Total
        lambda_metric = 1.0
        lambda_reg = 1.0
        lambda_rec = 20.0

        final_loss_metric = lambda_metric*anom_loss
        final_travel_loss = lambda_reg*travel_loss
        final_loss_reco = lambda_rec*recon_loss
        loss = final_loss_metric + final_travel_loss + final_loss_reco

        return (
            loss,
            final_loss_metric.detach(),
            final_travel_loss.detach(),
            final_loss_reco.detach(),
            anom_loss,
            unlabel_norm_loss_metric,
            unlabel_abnorm_loss_metric
        )
