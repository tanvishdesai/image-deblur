#!/usr/bin/env python3
"""
Enhanced Physics-Informed Neural Network for Image Deblurring
Based on Unrolled Richardson-Lucy Deconvolution with Deeper Architecture

Key improvements:
- Much deeper feature extraction networks
- Multi-scale processing with skip connections
- Enhanced Richardson-Lucy blocks with attention
- Progressive refinement architecture
- Advanced loss functions with perceptual components
- OPTIMIZED: Data generation pipeline moved to GPU for massive speedup.
- NEW: Resume training functionality from a specific checkpoint.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchvision.models as models

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
import random
from pathlib import Path
import cv2
from tqdm.auto import tqdm

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# =============================================================================
# ENHANCED BUILDING BLOCKS
# =============================================================================

class ResidualBlock(nn.Module):
    """Enhanced residual block with group normalization."""
    
    def __init__(self, channels, groups=8):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.gn1 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.gn2 = nn.GroupNorm(groups, channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return F.relu(out + residual)

class AttentionGate(nn.Module):
    """Spatial attention mechanism."""
    
    def __init__(self, in_channels):
        super(AttentionGate, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attention = self.conv(x)
        return x * attention

class MultiScaleFeatureExtractor(nn.Module):
    """Multi-scale feature extraction with dilated convolutions."""
    
    def __init__(self, in_channels, out_channels):
        super(MultiScaleFeatureExtractor, self).__init__()
        
        # Different dilation rates for multi-scale context
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, 3, padding=1, dilation=1),
            nn.GroupNorm(8, out_channels // 4),
            nn.ReLU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, 3, padding=2, dilation=2),
            nn.GroupNorm(8, out_channels // 4),
            nn.ReLU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, 3, padding=4, dilation=4),
            nn.GroupNorm(8, out_channels // 4),
            nn.ReLU()
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, 1),
            nn.GroupNorm(8, out_channels // 4),
            nn.ReLU()
        )
        
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        
        combined = torch.cat([b1, b2, b3, b4], dim=1)
        return self.fusion(combined)

# =============================================================================
# SYNTHETIC BLUR GENERATION (Enhanced and GPU-Accelerated)
# =============================================================================

def create_motion_blur_kernel(length, angle, kernel_size=15):
    """Create a motion blur kernel with given length and angle."""
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    
    angle_rad = np.radians(angle)
    dx = length / 2 * np.cos(angle_rad)
    dy = length / 2 * np.sin(angle_rad)
    
    num_points = max(1, int(np.ceil(length)))
    x_coords = np.linspace(center - dx, center + dx, num_points)
    y_coords = np.linspace(center - dy, center + dy, num_points)
    
    for i in range(num_points):
        x, y = int(round(x_coords[i])), int(round(y_coords[i]))
        if 0 <= x < kernel_size and 0 <= y < kernel_size:
            kernel[y, x] = 1.0
    
    if kernel.sum() == 0:
        kernel[center, center] = 1.0
    
    return kernel / kernel.sum()

# =============================================================================
# ENHANCED RICHARDSON-LUCY BLOCKS
# =============================================================================

class EnhancedRichardsonLucyBlock(nn.Module):
    """
    Enhanced Richardson-Lucy iteration with deep feature processing.
    FIXED: Uses efficient, vectorized convolution instead of slow for-loops.
    """
    
    def __init__(self, kernel_size=15, feature_channels=128):
        super(EnhancedRichardsonLucyBlock, self).__init__()
        self.kernel_size = kernel_size
        self.feature_channels = feature_channels
        self.eps = 1e-8
        
        # Learnable parameters
        self.alpha = nn.Parameter(torch.tensor(0.8))
        self.beta = nn.Parameter(torch.tensor(0.05))
        
        # Enhanced image processing network
        self.img_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            MultiScaleFeatureExtractor(64, feature_channels),
            ResidualBlock(feature_channels),
            ResidualBlock(feature_channels),
            AttentionGate(feature_channels)
        )
        
        self.img_decoder = nn.Sequential(
            ResidualBlock(feature_channels),
            nn.Conv2d(feature_channels, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh()  # Output in [-1, 1], will be scaled
        )
        
        # Enhanced kernel processing
        self.kernel_processor = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            ResidualBlock(32),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.ReLU()
        )
        
        # Feature fusion for updates
        self.update_fusion = nn.Sequential(
            nn.Conv2d(feature_channels + 3, feature_channels, 1),
            nn.GroupNorm(8, feature_channels),
            nn.ReLU(),
            ResidualBlock(feature_channels)
        )
    
    def forward(self, blurred, sharp_est, kernel_est):
        B, C, H, W = sharp_est.shape
        
        # Normalize kernel
        kernel_est = F.softmax(kernel_est.view(B, -1), dim=1).view_as(kernel_est)
        
        # Extract deep features from current estimate
        sharp_features = self.img_encoder(sharp_est)
        
        # --- Vectorized Physics-based forward model (convolution) ---
        pad_size = self.kernel_size // 2
        sharp_padded = F.pad(sharp_est, [pad_size] * 4, mode='reflect')
        
        # Reshape for grouped convolution:
        # Input: [B, C, H_pad, W_pad] -> [1, B*C, H_pad, W_pad]
        # Kernel: [B, 1, K, K] -> [B*C, 1, K, K]
        input_reshaped = sharp_padded.view(1, B * C, H + 2 * pad_size, W + 2 * pad_size)
        kernel_reshaped = kernel_est.repeat(1, C, 1, 1).view(B * C, 1, self.kernel_size, self.kernel_size)

        # Each of the B*C channels is its own group
        reconvolved_reshaped = F.conv2d(input_reshaped, kernel_reshaped, padding=0, groups=B * C)
        reconvolved = reconvolved_reshaped.view(B, C, H, W)

        # Compute ratio with numerical stability
        ratio = blurred / (reconvolved + self.eps)
        
        # --- Vectorized Richardson-Lucy update computation (correlation) ---
        kernel_flipped = torch.flip(kernel_est, [2, 3])
        ratio_padded = F.pad(ratio, [pad_size] * 4, mode='reflect')
        
        # Reshape for grouped convolution again
        ratio_reshaped = ratio_padded.view(1, B * C, H + 2 * pad_size, W + 2 * pad_size)
        kernel_flipped_reshaped = kernel_flipped.repeat(1, C, 1, 1).view(B * C, 1, self.kernel_size, self.kernel_size)
        
        rl_update_reshaped = F.conv2d(ratio_reshaped, kernel_flipped_reshaped, padding=0, groups=B * C)
        rl_update = rl_update_reshaped.view(B, C, H, W)
        
        # Fuse RL update with deep features
        combined_input = torch.cat([sharp_features, rl_update], dim=1)
        fused_features = self.update_fusion(combined_input)
        
        # Generate enhancement
        enhancement = self.img_decoder(fused_features)
        
        # Apply learned update with residual connection
        updated_sharp = sharp_est + self.alpha * enhancement * 0.1  # Scale down for stability
        updated_sharp = torch.clamp(updated_sharp, 0, 1)
        
        # Enhanced kernel update
        kernel_features = self.kernel_processor(kernel_est)
        updated_kernel = kernel_est + self.beta * (kernel_features - kernel_est)
        updated_kernel = F.relu(updated_kernel)
        updated_kernel = updated_kernel / (updated_kernel.sum(dim=[1, 2, 3], keepdim=True) + self.eps)
        
        return updated_sharp, updated_kernel

# =============================================================================
# MAIN ENHANCED NETWORK
# =============================================================================

class EnhancedPhysicsInformedDeblurNet(nn.Module):
    """Enhanced Physics-Informed Deblurring Network with deeper architecture."""
    
    def __init__(self, num_layers=7, kernel_size=15, feature_channels=128):
        super(EnhancedPhysicsInformedDeblurNet, self).__init__()
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.feature_channels = feature_channels
        
        # Enhanced initialization networks
        self.sharp_init = nn.Sequential(
            # Encoder
            nn.Conv2d(3, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            
            MultiScaleFeatureExtractor(64, 128),
            ResidualBlock(128),
            ResidualBlock(128),
            
            nn.Conv2d(128, 256, 3, padding=1, stride=2),  # Downsample
            nn.GroupNorm(8, 256),
            nn.ReLU(),
            ResidualBlock(256),
            ResidualBlock(256),
            AttentionGate(256),
            
            # Decoder
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # Upsample
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            ResidualBlock(128),
            
            nn.Conv2d(128, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid()
        )
        
        self.kernel_init = nn.Sequential(
            # Global feature extraction
            nn.AdaptiveAvgPool2d(8),  # Larger spatial context
            nn.Conv2d(3, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, kernel_size * kernel_size),
            nn.Softmax(dim=1)
        )
        
        # Enhanced physics blocks
        self.physics_blocks = nn.ModuleList([
            EnhancedRichardsonLucyBlock(kernel_size, feature_channels) 
            for _ in range(num_layers)
        ])
        
        # Final refinement network
        self.final_refine = nn.Sequential(
            MultiScaleFeatureExtractor(3, 64),
            ResidualBlock(64),
            AttentionGate(64),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh()  # Small residual correction
        )
    
    def forward(self, blurred):
        batch_size = blurred.size(0)
        
        # Enhanced initialization
        sharp_est = self.sharp_init(blurred)
        kernel_flat = self.kernel_init(blurred)
        kernel_est = kernel_flat.view(batch_size, 1, self.kernel_size, self.kernel_size)
        
        intermediates = []
        
        # Progressive refinement through physics blocks
        for i, block in enumerate(self.physics_blocks):
            sharp_est, kernel_est = block(blurred, sharp_est, kernel_est)
            intermediates.append((sharp_est.clone(), kernel_est.clone()))
        
        # Final refinement
        refinement = self.final_refine(sharp_est)
        final_output = torch.clamp(sharp_est + 0.05 * refinement, 0, 1)
        
        return final_output, kernel_est, intermediates

# =============================================================================
# ENHANCED DATASET CLASS (OPTIMIZED)
# =============================================================================

class EnhancedDeblurDataset(Dataset):
    """
    Enhanced dataset with data augmentation.
    OPTIMIZED: Only loads clean images. Blurring is done on-the-fly on GPU.
    """
    
    def __init__(self, data_dir, mode='train', patch_size=256, max_images=None):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.patch_size = patch_size
        
        # Find image files
        try:
            if mode == 'train':
                self.image_dir = self.data_dir / 'DIV2K_train_HR'
            else:
                self.image_dir = self.data_dir / 'DIV2K_valid_HR'
            
            self.image_paths = sorted(list(self.image_dir.glob('*.png')))
            
            if len(self.image_paths) == 0:
                print(f"Warning: No images found in {self.image_dir}. Trying alternative common paths.")
                if mode == 'train':
                    alt_path = self.data_dir / 'DIV2K_train_HR' / 'DIV2K_train_HR'
                else:
                    alt_path = self.data_dir / 'DIV2K_valid_HR' / 'DIV2K_valid_HR'
                if alt_path.exists():
                    self.image_dir = alt_path
                    self.image_paths = sorted(list(self.image_dir.glob('*.png')))

            if max_images:
                self.image_paths = self.image_paths[:max_images]
            
            if len(self.image_paths) == 0:
                raise FileNotFoundError(f"CRITICAL: No images found for mode '{mode}'. Please check your data directory structure in '{self.data_dir}'.")
            else:
                print(f"Found {len(self.image_paths)} images in {self.image_dir} for {mode} set")
            
        except Exception as e:
            print(f"Error setting up dataset: {e}")
            raise
        
        # Enhanced transforms with augmentation for training
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)
            ])
        else:
            self.transform = transforms.ToTensor()
    
    def __len__(self):
        # Create multiple patches from each image per epoch
        return len(self.image_paths) * (8 if self.mode == 'train' else 4)
    
    def __getitem__(self, idx):
        img_idx = idx // (8 if self.mode == 'train' else 4)
        
        img_path = self.image_paths[img_idx]
        image = Image.open(img_path).convert('RGB')
        
        # Apply transform first, which handles ToTensor()
        image_tensor = self.transform(image)
        
        # Enhanced cropping strategy
        _, h, w = image_tensor.shape
        if h < self.patch_size or w < self.patch_size:
            # Resize if image is smaller than patch size
            image_tensor = F.interpolate(
                image_tensor.unsqueeze(0), 
                size=(max(self.patch_size, h), max(self.patch_size, w)),
                mode='bilinear', align_corners=False
            ).squeeze(0)
            _, h, w = image_tensor.shape
        
        # Smart cropping
        if self.mode == 'train':
            top = torch.randint(0, h - self.patch_size + 1, (1,)).item()
            left = torch.randint(0, w - self.patch_size + 1, (1,)).item()
        else:
            # More systematic validation crops
            crop_idx = idx % 4
            if h <= self.patch_size or w <= self.patch_size:
                 top, left = 0, 0
            else:
                 h_step = (h - self.patch_size) // 1 
                 w_step = (w - self.patch_size) // 1
                 top = (crop_idx // 2) * h_step
                 left = (crop_idx % 2) * w_step

        clean_patch = image_tensor[:, top:top+self.patch_size, left:left+self.patch_size]
        
        # OPTIMIZED: Return only the clean patch. Blurring is done on GPU.
        return {'clean': clean_patch}

# =============================================================================
# ENHANCED LOSS FUNCTIONS
# =============================================================================

class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG features."""
    
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        # Use updated weights API
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.feature_extractor = nn.Sequential(*list(vgg.children())[:16]).to(device)  # Up to relu3_3
        
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        self.mse = nn.MSELoss()
        # VGG was trained on ImageNet, which has specific normalization
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, pred, target):
        pred_norm = self.normalize(pred)
        target_norm = self.normalize(target)

        pred_features = self.feature_extractor(pred_norm)
        target_features = self.feature_extractor(target_norm)
        return self.mse(pred_features, target_features)

class EnhancedCombinedLoss(nn.Module):
    """Enhanced loss with multiple components."""
    
    def __init__(self, img_weight=1.0, kernel_weight=0.1, perceptual_weight=0.2, ssim_weight=0.3):
        super(EnhancedCombinedLoss, self).__init__()
        self.img_weight = img_weight
        self.kernel_weight = kernel_weight
        self.perceptual_weight = perceptual_weight
        self.ssim_weight = ssim_weight
        
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss()
    
    def ssim_loss(self, pred, target):
        """Simplified SSIM loss."""
        mu1 = F.avg_pool2d(pred, 3, stride=1, padding=1)
        mu2 = F.avg_pool2d(target, 3, stride=1, padding=1)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.avg_pool2d(pred * pred, 3, stride=1, padding=1) - mu1_sq
        sigma2_sq = F.avg_pool2d(target * target, 3, stride=1, padding=1) - mu2_sq
        sigma12 = F.avg_pool2d(pred * target, 3, stride=1, padding=1) - mu1_mu2
        
        C1, C2 = 0.01**2, 0.03**2
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        return 1 - ssim_map.mean()
    
    def forward(self, pred_img, target_img, pred_kernel, target_kernel):
        # Image reconstruction losses
        mse_loss = self.mse_loss(pred_img, target_img)
        l1_loss = self.l1_loss(pred_img, target_img)
        perceptual_loss = self.perceptual_loss(pred_img, target_img)
        ssim_loss = self.ssim_loss(pred_img, target_img)
        
        # Kernel loss
        kernel_loss = self.mse_loss(pred_kernel, target_kernel)
        
        # Combined loss
        total_loss = (self.img_weight * (0.5 * mse_loss + 0.5 * l1_loss) + 
                     self.kernel_weight * kernel_loss + 
                     self.perceptual_weight * perceptual_loss + 
                     self.ssim_weight * ssim_loss)
        
        return total_loss, mse_loss, kernel_loss, perceptual_loss, ssim_loss

# =============================================================================
# METRICS
# =============================================================================

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images."""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def calculate_ssim(img1, img2, window_size=11):
    """Calculate SSIM between two images."""
    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size//2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size//2)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.avg_pool2d(img1 * img1, window_size, stride=1, padding=window_size//2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, window_size, stride=1, padding=window_size//2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size//2) - mu1_mu2
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

# =============================================================================
# TRAINING AND UTILITY FUNCTIONS (OPTIMIZED)
# =============================================================================

# In your training script

def apply_batch_motion_blur(clean_batch, kernel_size):
    """
    PERFORMANCE FIX: Generates and applies unique motion blurs to a batch 
    of images directly on the GPU.
    """
    B, C, H, W = clean_batch.shape
    device = clean_batch.device
    
    kernel_list = []
    for _ in range(B):
        # NEW "HARDER" SETTINGS:
        angle = random.uniform(0, 360)
        length = random.uniform(10, 45)  # Increased blur length range

        kernel_np = create_motion_blur_kernel(length, angle, kernel_size)
        kernel_list.append(torch.from_numpy(kernel_np).float())
    
    kernels_tensor = torch.stack(kernel_list).unsqueeze(1).to(device) 
    
    # Apply convolution to create blurred images
    pad_size = kernel_size // 2
    clean_padded = F.pad(clean_batch, [pad_size] * 4, mode='reflect')
    
    # Reshape for grouped convolution:
    # Input: [B, C, H_pad, W_pad] -> [1, B*C, H_pad, W_pad]
    # Kernel: [B, 1, K, K] -> [B*C, 1, K, K]
    input_reshaped = clean_padded.view(1, B * C, H + 2 * pad_size, W + 2 * pad_size)
    kernel_reshaped = kernels_tensor.repeat(1, C, 1, 1).view(B * C, 1, kernel_size, kernel_size)

    # Each of the B*C channels is its own group
    blurred_reshaped = F.conv2d(input_reshaped, kernel_reshaped, padding=0, groups=B * C)
    blurred_batch = blurred_reshaped.view(B, C, H, W)

    # Add noise
    noise_level = random.uniform(0, 0.04)  # Increased max noise level
    noise = torch.randn_like(blurred_batch) * noise_level
    blurred_batch = torch.clamp(blurred_batch + noise, 0, 1)

    return blurred_batch, kernels_tensor

def save_comparison_images(blurred, predicted, ground_truth, epoch, save_dir='results'):
    """Save comparison images from a validation batch."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Take the first image from the batch
    blurred_img = torch.clamp(blurred[0], 0, 1)
    pred_img = torch.clamp(predicted[0], 0, 1)
    gt_img = torch.clamp(ground_truth[0], 0, 1)
    
    comparison = torch.cat([blurred_img, pred_img, gt_img], dim=2)
    save_path = os.path.join(save_dir, f'comparison_epoch_{epoch:03d}.png')
    save_image(comparison, save_path, normalize=False)

def train_model(model, train_loader, val_loader, num_epochs=100, lr=1e-3, 
                results_dir='results', checkpoint_dir='checkpoints', 
                resume_training=False, resume_from_checkpoint=0, 
                explicit_checkpoint_path=None):  # NEW PARAMETER
    """Enhanced training loop with advanced optimizations and resume functionality."""
    model = model.to(device)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    criterion = EnhancedCombinedLoss(img_weight=1.0, kernel_weight=0.05, perceptual_weight=0.1, ssim_weight=0.2).to(device)

    start_epoch = 0
    best_psnr = 0.0
    history = {'train_losses': [], 'val_psnrs': [], 'val_ssims': []}

    # --- Resume Training Logic ---
    if resume_training:
        # NEW: Use explicit path if provided
        if explicit_checkpoint_path:
            checkpoint_path = explicit_checkpoint_path
            checkpoint_name = os.path.basename(explicit_checkpoint_path)
        # Original logic for automatic path selection
        elif resume_from_checkpoint == 0:
            checkpoint_path = os.path.join(checkpoint_dir, 'latest_model.pth')
            checkpoint_name = 'latest_model.pth'
        else:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{resume_from_checkpoint}.pth')
            checkpoint_name = f'checkpoint_epoch_{resume_from_checkpoint}.pth'

        if os.path.exists(checkpoint_path):
            print(f"Attempting to resume training from checkpoint: {checkpoint_name}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Rest of the loading logic remains the same...
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                print("Loading comprehensive checkpoint (new format).")
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
                start_epoch = checkpoint['epoch']
                best_psnr = checkpoint.get('best_psnr', 0.0)
                history = checkpoint.get('history', {'train_losses': [], 'val_psnrs': [], 'val_ssims': []})

                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)

                print(f"Successfully resumed from epoch {start_epoch}. Best PSNR was {best_psnr:.2f} dB.")
            else:
                print("WARNING: Loading legacy checkpoint (old format - model weights only).")
                print("Optimizer, scheduler, and history will be re-initialized.")
                model.load_state_dict(checkpoint)
                print("Model weights loaded successfully. Training will start from epoch 0.")
        else:
            print(f"WARNING: Checkpoint '{checkpoint_path}' not found. Starting training from scratch.")
    for epoch in range(start_epoch, num_epochs):
        # --- Training phase ---
        model.train()
        epoch_train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]', leave=False)
        for batch in progress_bar:
            clean = batch['clean'].to(device)
            blurred, kernel_gt = apply_batch_motion_blur(clean, model.kernel_size)

            optimizer.zero_grad()
            pred_clean, pred_kernel, _ = model(blurred)
            total_loss, mse, k_loss, p_loss, s_loss = criterion(pred_clean, clean, pred_kernel, kernel_gt)
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            epoch_train_loss += total_loss.item()
            progress_bar.set_postfix(loss=f'{total_loss.item():.4f}', mse=f'{mse.item():.4f}', k_loss=f'{k_loss.item():.4f}')
        
        # =================== START OF FIX ===================
        # Calculate the average training loss *before* attempting to use it.
        avg_train_loss = epoch_train_loss / len(train_loader)
        
        # Guard against empty/shorter history lists when resuming training
        if len(history['train_losses']) <= epoch:
            history['train_losses'].append(avg_train_loss)
        else:
            # Overwrite the value for this epoch if we are re-running it
            history['train_losses'][epoch] = avg_train_loss
        # =================== END OF FIX ===================

        # --- Validation phase ---
        model.eval()
        epoch_val_psnr = 0.0
        epoch_val_ssim = 0.0
        
        with torch.no_grad():
            val_progress_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]', leave=False)
            for i, batch in enumerate(val_progress_bar):
                clean = batch['clean'].to(device)
                blurred, _ = apply_batch_motion_blur(clean, model.kernel_size)
                pred_clean, _, _ = model(blurred)
                
                batch_psnr = calculate_psnr(pred_clean, clean)
                batch_ssim = calculate_ssim(pred_clean, clean)
                
                epoch_val_psnr += batch_psnr.item()
                epoch_val_ssim += batch_ssim.item()
                val_progress_bar.set_postfix(psnr=f'{batch_psnr.item():.2f}', ssim=f'{batch_ssim.item():.4f}')
                
                if i == 0:
                    save_comparison_images(blurred.cpu(), pred_clean.cpu(), clean.cpu(), epoch + 1, save_dir=results_dir)

        avg_val_psnr = epoch_val_psnr / len(val_loader)
        avg_val_ssim = epoch_val_ssim / len(val_loader)
        
        if len(history['val_psnrs']) <= epoch:
            history['val_psnrs'].append(avg_val_psnr)
            history['val_ssims'].append(avg_val_ssim)
        else:
            history['val_psnrs'][epoch] = avg_val_psnr
            history['val_ssims'][epoch] = avg_val_ssim

        print(f"\nEpoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | "
              f"Val PSNR: {avg_val_psnr:.2f} dB | Val SSIM: {avg_val_ssim:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.1e}")

        scheduler.step()

        # --- Save Comprehensive Checkpoint ---
        checkpoint_data = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_psnr': best_psnr,
            'history': history
        }

        if avg_val_psnr > best_psnr:
            best_psnr = avg_val_psnr
            checkpoint_data['best_psnr'] = best_psnr
            torch.save(checkpoint_data, os.path.join(checkpoint_dir, 'best_model.pth'))
            print(f"New best model saved with PSNR: {best_psnr:.2f} dB")
            
        torch.save(checkpoint_data, os.path.join(checkpoint_dir, 'latest_model.pth'))
        
        if (epoch + 1) % 5 == 0 or epoch + 1 == num_epochs:
            torch.save(checkpoint_data, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth'))
        
    print("Training finished.")
    return history

def plot_metrics(history, save_path):
    """Plots training and validation metrics and saves the figure."""
    epochs = range(1, len(history['train_losses']) + 1)
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('Training and Validation Metrics', fontsize=16)
    ax1.plot(epochs, history['train_losses'], 'o-', label='Training Loss')
    ax1.set_ylabel('Loss'); ax1.set_title('Training Loss'); ax1.legend(); ax1.grid(True)
    ax2.plot(epochs, history['val_psnrs'], 's-', label='Validation PSNR (dB)', color='tab:green')
    ax2.set_ylabel('PSNR (dB)', color='tab:green'); ax2.tick_params(axis='y', labelcolor='tab:green')
    ax2.set_xlabel('Epoch'); ax2.set_title('Validation Metrics'); ax2.grid(True)
    ax3 = ax2.twinx()
    ax3.plot(epochs, history['val_ssims'], '^-', label='Validation SSIM', color='tab:red')
    ax3.set_ylabel('SSIM', color='tab:red'); ax3.tick_params(axis='y', labelcolor='tab:red')
    lines, labels = ax2.get_legend_handles_labels(); lines2, labels2 = ax3.get_legend_handles_labels()
    ax3.legend(lines + lines2, labels + labels2, loc='lower right')
    fig.tight_layout(rect=[0, 0.03, 1, 0.96]); plt.savefig(save_path); plt.close()
    print(f"Metrics plot saved to {save_path}")

def run_inference(model_class_ref, model_path, image_path, output_dir, device):
    """Loads a model, performs deblurring on a full-size image, and saves the result."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading model from {model_path}...")
    # NOTE: Ensure these args match the saved model
    model_args = {'num_layers': 5, 'kernel_size': 15, 'feature_channels': 64}
    model = model_class_ref(**model_args).to(device)
    
    # MODIFIED: Load model state robustly from either comprehensive or legacy checkpoint.
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # New format
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Old format
        model.load_state_dict(checkpoint)

    model.eval()
    
    print(f"Processing image: {image_path}")
    image = Image.open(image_path).convert('RGB')
    clean_full_image = transforms.ToTensor()(image)
    
    # Use the same GPU-accelerated blur function
    blurred_full_image, _ = apply_batch_motion_blur(clean_full_image.unsqueeze(0).to(device), model.kernel_size)
    
    with torch.no_grad():
        pred_clean, _, _ = model(blurred_full_image)

    pred_clean_cpu = pred_clean.squeeze(0).cpu().clamp(0, 1)
    blurred_cpu = blurred_full_image.squeeze(0).cpu().clamp(0, 1)
    clean_cpu = clean_full_image.clamp(0, 1)
    
    psnr = calculate_psnr(pred_clean_cpu, clean_cpu)
    ssim = calculate_ssim(pred_clean_cpu.unsqueeze(0), clean_cpu.unsqueeze(0))
    print(f"Inference results - PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
    
    comparison = torch.cat([blurred_cpu, pred_clean_cpu, clean_cpu], dim=2)
    output_filename = Path(output_dir) / f"{Path(image_path).stem}_inference_comparison.png"
    save_image(comparison, output_filename)
    print(f"Saved inference comparison to {output_filename}")


# Add this helper function to your script
def load_partial_state_dict(model, checkpoint_path):
    """
    Loads weights from a checkpoint, skipping layers with mismatched shapes.
    """
    device = next(model.parameters()).device
    pretrained_dict = torch.load(checkpoint_path, map_location=device)
    
    # If it's a comprehensive checkpoint, get the model state
    if 'model_state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['model_state_dict']
    
    model_dict = model.state_dict()
    
    # 1. Filter out incompatible weights
    compatible_dict = {}
    incompatible_keys = []
    for k, v in pretrained_dict.items():
        if k in model_dict and v.shape == model_dict[k].shape:
            compatible_dict[k] = v
        else:
            incompatible_keys.append(k)

    # 2. Overwrite the new model's state dictionary with compatible weights
    model_dict.update(compatible_dict)
    
    # 3. Load the new state dictionary
    model.load_state_dict(model_dict)
    
    if incompatible_keys:
        print("Warning: The following keys were found in the checkpoint but were incompatible and skipped:")
        for key in incompatible_keys:
            print(f"  - {key}")
    print(f"Successfully loaded {len(compatible_dict)} compatible weight tensors.")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    """Main function to run the training and inference pipeline."""
    DATA_DIR = "/kaggle/input/div2k-dataset" 
    RESULTS_DIR = "results"
    CHECKPOINT_DIR = "checkpoints"
    INFERENCE_DIR = "inference_output"
    
    # --- Configuration ---
    NUM_EPOCHS = 15
    BATCH_SIZE = 8
    LEARNING_RATE = 2e-4
    PATCH_SIZE = 128
    NUM_WORKERS = 4
    
    # --- RESUME TRAINING CONFIGURATION ---
    RESUME_TRAINING = True
    RESUME_FROM_CHECKPOINT = 11
    
    # NEW: Explicit checkpoint path - Set this to your specific model path
    EXPLICIT_CHECKPOINT_PATH = "/kaggle/input/testing-physics-informed/checkpoints/checkpoint_epoch_11.pth"  # Example: "/path/to/your/model.pth"
    # Alternative examples:
    # EXPLICIT_CHECKPOINT_PATH = "/content/checkpoints/best_model.pth"
    # EXPLICIT_CHECKPOINT_PATH = "/kaggle/working/my_custom_checkpoint.pth"
    # EXPLICIT_CHECKPOINT_PATH = "checkpoints/checkpoint_epoch_25.pth"
    
    try:
        train_dataset = EnhancedDeblurDataset(data_dir=DATA_DIR, mode='train', patch_size=PATCH_SIZE)
        val_dataset = EnhancedDeblurDataset(data_dir=DATA_DIR, mode='val', patch_size=PATCH_SIZE)
        
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
            num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True if NUM_WORKERS > 0 else False
        )
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True if NUM_WORKERS > 0 else False
        )
        
        model = EnhancedPhysicsInformedDeblurNet(num_layers=5, kernel_size=51, feature_channels=64)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model created. Number of trainable parameters: {num_params / 1e6:.2f}M")
        if RESUME_TRAINING:
            checkpoint_path = "/kaggle/input/k/vikramdesai/testing-physics-informed/checkpoints/checkpoint_epoch_25.pth"
            if os.path.exists(checkpoint_path):
                print(f"Performing surgical weight transfer from: {checkpoint_path}")
                load_partial_state_dict(model, checkpoint_path)
                # IMPORTANT: Do NOT load the optimizer/scheduler state, as the new layers 
                # have different parameters. Start the optimizer fresh.
            else:
                print("Checkpoint not found. Starting from scratch.")
       
        history = train_model(
            model, train_loader, val_loader, 
            num_epochs=NUM_EPOCHS, lr=LEARNING_RATE,
            results_dir=RESULTS_DIR, checkpoint_dir=CHECKPOINT_DIR,
            resume_training=False,
        )
        
        # Rest of the function remains the same...
        plot_metrics(history, os.path.join(RESULTS_DIR, 'training_metrics.png'))
        
        if len(val_dataset.image_paths) > 0:
            test_image_path = val_dataset.image_paths[0]
            best_model_path = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
            
            if os.path.exists(best_model_path):
                print("\nRunning inference on a full-size sample image...")
                run_inference(
                    EnhancedPhysicsInformedDeblurNet,
                    best_model_path, 
                    str(test_image_path),
                    INFERENCE_DIR,
                    device
                )
            else:
                print(f"Could not find best_model.pth at '{best_model_path}' for inference. Skipping.")

    except (FileNotFoundError, RuntimeError) as e:
        print("\n" + "="*60)
        print(f"ERROR: {e}")
        print("Please check your data paths and ensure the DIV2K dataset is available.")
        print(f"Expected structure: {Path(DATA_DIR) / 'DIV2K_train_HR/*.png'}")
        print("Aborting.")
        print("="*60)
if __name__ == '__main__':
    main()