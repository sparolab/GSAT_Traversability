import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPAutoEncoder(nn.Module):
    def __init__(self, input_dim, 
                 hidden_dims_enc, 
                 bottle_neck_dim, 
                 hidden_dims_dec, 
                 recon_out_dim=None,
                 dropout_prob=0.2):

        super(MLPAutoEncoder, self).__init__()
        
        if recon_out_dim is None:
            recon_out_dim = input_dim
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h in hidden_dims_enc:
            encoder_layers.append(nn.Linear(prev_dim, h))
            encoder_layers.append(nn.BatchNorm1d(h))
            encoder_layers.append(nn.ReLU(inplace=False))
            # encoder_layers.append(nn.LeakyReLU(inplace=False))
            encoder_layers.append(nn.Dropout(p=dropout_prob))
            prev_dim = h
        self.encoder = nn.Sequential(*encoder_layers)
        
        self.fc_bottleneck = nn.Linear(prev_dim, bottle_neck_dim)
        self.bn_bottleneck = nn.BatchNorm1d(bottle_neck_dim)
        self.act_bottleneck = nn.ReLU(inplace=True)
        self.drop_bottleneck = nn.Dropout(p=dropout_prob)
        # Decoder
        decoder_layers = []
        prev_dim = bottle_neck_dim
        for h in hidden_dims_dec:
            decoder_layers.append(nn.Linear(prev_dim, h))
            decoder_layers.append(nn.BatchNorm1d(h))
            decoder_layers.append(nn.ReLU(inplace=False))
            # decoder_layers.append(nn.LeakyReLU(inplace=False))
            decoder_layers.append(nn.Dropout(p=dropout_prob))
            prev_dim = h
        decoder_layers.append(nn.Linear(prev_dim, recon_out_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        self.trav_fc = nn.Linear(bottle_neck_dim, 1)
        self.trav_sigmoid = nn.Sigmoid()


    def forward(self, x):
        """
        x: (B, input_dim, H, W)
        returns:
          latent_space: (B, bottle_neck_dim, H, W)
          recon_out:    (B, input_dim, H, W)
        """
        B, C, H, W = x.size()
        x_flat = x.permute(0, 2, 3, 1).contiguous().view(B * H * W, C)
        
        # Encoder
        enc = self.encoder(x_flat)                      # (B*H*W, hidden_dims_enc[-1])
        latent = self.fc_bottleneck(enc)                    # (B*H*W, bottle_neck_dim)
        latent_bn = self.bn_bottleneck(latent)
        latent_relu = self.act_bottleneck(latent_bn)
        latent_drop = self.drop_bottleneck(latent_relu)        

        # reco = self.decoder(latent_relu)
        reco = self.decoder(latent_drop)

        recon_out = reco.view(B, H, W, -1).permute(0, 3, 1, 2)  # (B, input_dim, H, W)
        latent_space = latent.view(B, H, W, -1).permute(0, 3, 1, 2)  # (B, bottle_neck_dim, H, W)

        trav = self.trav_sigmoid(self.trav_fc(latent_drop))
        trav = trav.view(B, H, W, 1).permute(0, 3, 1, 2)

        return latent_space, recon_out, trav

class gsat_head(nn.Module):
    def __init__(self):
        super().__init__()
        self.s_96_64_32_16 = MLPAutoEncoder(input_dim=96, hidden_dims_enc=[64,32],bottle_neck_dim=16,
                                    hidden_dims_dec=[32,64],recon_out_dim=96)
    def forward(self, feature_map):
        latent_space, recon_out, trave_out = self.s_96_64_32_16(feature_map)

        recon_input = feature_map
        return latent_space, recon_out, recon_input, trave_out