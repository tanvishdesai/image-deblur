%%writefile model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- Degradation Estimator ---
class DegradationEstimator(nn.Module):
    def __init__(self, input_dim=3, output_dim=4, embed_dim=64):
        super().__init__()
        # Lightweight ResNet-style encoder
        self.conv1 = nn.Conv2d(input_dim, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(32, 64, stride=2)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 128, stride=2)
        self.layer4 = self._make_layer(128, embed_dim, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_out = nn.Linear(embed_dim, output_dim)
        self.fc_embed = nn.Linear(embed_dim, embed_dim) # Output embedding for conditioning

    def _make_layer(self, in_channels, out_channels, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        feat = x.flatten(1)
        
        deg_pred = torch.sigmoid(self.fc_out(feat)) # Predict normalized params [0, 1]
        deg_embed = self.fc_embed(feat) # Embedding for SFT
        
        return deg_pred, deg_embed

# --- SFT Layer (Spatial Feature Transform) ---
class SFTLayer(nn.Module):
    def __init__(self, channels, cond_dim=64):
        super().__init__()
        self.scale_net = nn.Sequential(
            nn.Linear(cond_dim, channels),
            nn.Sigmoid() # Scale is typically around 0-1 or 0-2
        )
        self.shift_net = nn.Sequential(
            nn.Linear(cond_dim, channels),
            nn.Tanh() # Shift around 0
        )

    def forward(self, x, cond):
        # x: [B, H, W, C] (Swin uses channels last in some impls, but standard is NCHW. 
        # Wait, Swin usually operates on [B, L, C]. Let's assume input is [B, L, C] or [B, H, W, C])
        
        # If x is [B, L, C]
        scale = self.scale_net(cond).unsqueeze(1) # [B, 1, C]
        shift = self.shift_net(cond).unsqueeze(1) # [B, 1, C]
        
        return x * (scale + 1) + shift

# --- Swin Transformer Components (Simplified) ---
# We'll implement a basic Swin Block with SFT injection.
# For brevity, we'll use a standard attention implementation or a simplified one.
# To ensure it runs on Kaggle without complex CUDA extensions, we stick to pure PyTorch.

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
        # Relative position bias (simplified)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=8, shift_size=0, mlp_ratio=4., cond_dim=64):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )
        
        # SFT Layers for conditioning
        self.sft1 = SFTLayer(dim, cond_dim)
        self.sft2 = SFTLayer(dim, cond_dim)

    def forward(self, x, cond):
        H, W = self.H, self.W
        B, L, C = x.shape
        
        shortcut = x
        x = self.norm1(x)
        x = self.sft1(x, cond) # Inject degradation info
        
        x = x.view(B, H, W, C)
        
        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Window partition
        x_windows = self.window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = self.window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
            
        x = x.view(B, H * W, C)
        x = shortcut + x
        
        # FFN
        shortcut = x
        x = self.norm2(x)
        x = self.sft2(x, cond) # Inject degradation info
        x = self.mlp(x)
        x = shortcut + x
        
        return x

    def window_partition(self, x, window_size):
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return windows

    def window_reverse(self, windows, window_size, H, W):
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size, mlp_ratio=4., cond_dim=64):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim, num_heads, window_size, 
                               shift_size=0 if (i % 2 == 0) else window_size // 2,
                               mlp_ratio=mlp_ratio, cond_dim=cond_dim)
            for i in range(depth)
        ])

    def forward(self, x, cond, H, W):
        for blk in self.blocks:
            blk.H, blk.W = H, W
            x = blk(x, cond)
        return x

# --- Main DAST Model ---
class DAST(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.scale = config.SCALE
        
        # Degradation Estimator
        self.deg_estimator = DegradationEstimator(
            output_dim=config.DEG_VECTOR_SIZE,
            embed_dim=config.DEG_EMBED_DIM
        )
        
        # Shallow Feature Extraction
        self.conv_first = nn.Conv2d(3, config.EMBED_DIM, 3, 1, 1)
        
        # Deep Feature Extraction
        self.layers = nn.ModuleList()
        for i in range(len(config.DEPTHS)):
            layer = BasicLayer(
                dim=config.EMBED_DIM,
                depth=config.DEPTHS[i],
                num_heads=config.NUM_HEADS[i],
                window_size=config.WINDOW_SIZE,
                mlp_ratio=config.MLP_RATIO,
                cond_dim=config.DEG_EMBED_DIM
            )
            self.layers.append(layer)
            
        self.norm = nn.LayerNorm(config.EMBED_DIM)
        
        # Reconstruction
        self.conv_after_body = nn.Conv2d(config.EMBED_DIM, config.EMBED_DIM, 3, 1, 1)
        self.upsample = nn.Sequential(
            nn.Conv2d(config.EMBED_DIM, config.EMBED_DIM * (self.scale ** 2), 3, 1, 1),
            nn.PixelShuffle(self.scale)
        )
        self.conv_last = nn.Conv2d(config.EMBED_DIM, 3, 3, 1, 1)

    def forward(self, x):
        # x: [B, 3, H, W]
        
        # 1. Estimate Degradation
        deg_pred, deg_embed = self.deg_estimator(x) # deg_pred: [B, 4], deg_embed: [B, 64]
        
        # 2. Shallow Features
        x_first = self.conv_first(x)
        
        # 3. Deep Features with SFT
        B, C, H, W = x_first.shape
        x_feat = x_first.flatten(2).transpose(1, 2) # [B, H*W, C]
        
        for layer in self.layers:
            x_feat = layer(x_feat, deg_embed, H, W)
            
        x_feat = self.norm(x_feat)
        x_feat = x_feat.transpose(1, 2).view(B, C, H, W)
        
        # 4. Reconstruction
        x_feat = self.conv_after_body(x_feat)
        x_feat = x_feat + x_first # Residual connection
        
        x_up = self.upsample(x_feat)
        x_out = self.conv_last(x_up)
        
        return x_out, deg_pred

if __name__ == "__main__":
    # Test instantiation
    from config import Config
    model = DAST(Config)
    dummy_input = torch.randn(1, 3, 64, 64)
    out, deg = model(dummy_input)
    print(f"Output shape: {out.shape}")
    print(f"Degradation prediction shape: {deg.shape}")
