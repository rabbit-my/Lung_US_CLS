import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------
# Attention Module: CBAM
# ---------------------------------------------
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups, bias)
        self.bn = nn.BatchNorm2d(out_planes) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None: x = self.bn(x)
        if self.relu is not None: x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                pool = F.avg_pool2d(x, (x.size(2), x.size(3)))
            elif pool_type == 'max':
                pool = F.max_pool2d(x, (x.size(2), x.size(3)))
            channel_att_raw = self.mlp(pool)
            channel_att_sum = channel_att_raw if channel_att_sum is None else channel_att_sum + channel_att_raw
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size=7, padding=3, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

# ---------------------------------------------
# Encoder
# ---------------------------------------------
class ConvEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_cnn1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )
        self.encoder_cnn2 = nn.Sequential(
            nn.MaxPool2d(2),  # 256 → 128
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 128 → 64
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 64 → 32
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.encoder_cnn3 = nn.Sequential(
            nn.MaxPool2d(2),  # 32 → 16
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2)  # 16 → 8
        )
        self.flatten = nn.Flatten(start_dim=1)

    def forward(self, x):
        x1 = self.encoder_cnn1(x)
        x2 = self.encoder_cnn2(x1)
        x3 = self.encoder_cnn3(x2)
        x = self.flatten(x3)  # shape: (B, 256*8*8)
        z = [x1, x2]
        return x, z

class LinearEncoder(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.encoder_lin = nn.Sequential(
            nn.Linear(256 * 8 * 8, features),
            nn.BatchNorm1d(features),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.encoder_lin(x)

# ---------------------------------------------
# Decoder
# ---------------------------------------------
class LinearDecoder(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(features, 256 * 8 * 8),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.decoder_lin(x)

class ConvDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(256, 8, 8))
        self.decoder_conv = nn.Sequential(
            nn.Upsample(scale_factor=2),  # 8 → 16
            nn.ConvTranspose2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2),  # 16 → 32
            nn.ConvTranspose2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2),  # 32 → 64
            nn.ConvTranspose2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2),  # 64 → 128
            nn.ConvTranspose2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2),  # 128 → 256
            nn.ConvTranspose2d(16, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        return x

# ---------------------------------------------
# AutoEncoder (Full Model)
# ---------------------------------------------
class AutoEncoderDecoder(nn.Module):
    def __init__(self, features=1024):
        super().__init__()
        self.convenoder = ConvEncoder()
        self.linencoder = LinearEncoder(features)
        self.lindecoder = LinearDecoder(features)
        self.convdecoder = ConvDecoder()

    def forward(self, x):
        x, _ = self.convenoder(x)  # [B, 16384]
        y = self.linencoder(x)     # [B, features]
        x = self.lindecoder(y)     # [B, 16384]
        x = self.convdecoder(x)   # [B, 1, 256, 256]
        return x, y

# ---------------------------------------------
# ✅ 测试代码（可选）
# ---------------------------------------------
if __name__ == '__main__':
    inp = torch.randn(2, 1, 256, 256)
    model = AutoEncoderDecoder(features=1024)
    out, latent = model(inp)
    print("Output shape:", out.shape)      # → [2, 1, 256, 256]
    print("Latent shape:", latent.shape)   # → [2, 1024]
