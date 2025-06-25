import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import os
import numpy as np
import math
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import json # Added for saving results

print("="*80)
print("NOVEL GAN-BASED HIERARCHICAL MULTI-SCALE MOTION DEBLURRING NETWORK")
print("Architecture: Generator + Discriminator + Perceptual Loss + Adversarial Training")
print("="*80)

# ============================================================================
# SECTION 1: DATASET PREPARATION AND LOADING
# ============================================================================
print("\n[SECTION 1] Setting up dataset and data loading...")

class GoPro_Dataset(Dataset):
    def __init__(self, root_dir, transform=None, max_samples=None):
        print(f"Initializing dataset from: {root_dir}")
        self.root_dir = root_dir
        self.transform = transform
        self.blur_images = []
        self.sharp_images = []
        
        # Traverse directory structure
        sequence_dirs = [d for d in os.listdir(root_dir) if d.startswith('GOPR0')]
        print(f"Found {len(sequence_dirs)} sequence directories")
        
        sample_count = 0
        for seq_dir in sequence_dirs:
            seq_path = os.path.join(root_dir, seq_dir)
            if os.path.isdir(seq_path):
                blur_path = os.path.join(seq_path, 'blur')
                sharp_path = os.path.join(seq_path, 'sharp')
                
                if os.path.exists(blur_path) and os.path.exists(sharp_path):
                    blur_files = sorted([f for f in os.listdir(blur_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
                    sharp_files = sorted([f for f in os.listdir(sharp_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
                    
                    for blur_file, sharp_file in zip(blur_files, sharp_files):
                        self.blur_images.append(os.path.join(blur_path, blur_file))
                        self.sharp_images.append(os.path.join(sharp_path, sharp_file))
                        sample_count += 1
                        
                        if max_samples and sample_count >= max_samples:
                            break
            
            if max_samples and sample_count >= max_samples:
                break
        
        print(f"Dataset initialized with {len(self.blur_images)} image pairs")
        
    def __len__(self):
        return len(self.blur_images)
    
    def __getitem__(self, idx):
        blur_img = Image.open(self.blur_images[idx]).convert('RGB')
        sharp_img = Image.open(self.sharp_images[idx]).convert('RGB')
        
        if self.transform:
            # Apply same transform to both images
            seed = torch.randint(0, 2147483647, (1,)).item()
            torch.manual_seed(seed)
            blur_img = self.transform(blur_img)
            torch.manual_seed(seed)
            sharp_img = self.transform(sharp_img)
        
        return blur_img, sharp_img

# ============================================================================
# SECTION 2: ENHANCED NETWORK ARCHITECTURE COMPONENTS
# ============================================================================
print("\n[SECTION 2] Building enhanced network architecture components...")

class ChannelAttention(nn.Module):
    """Channel Attention Module for feature recalibration"""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.conv2(self.relu(self.conv1(self.avg_pool(x))))
        max_out = self.conv2(self.relu(self.conv1(self.max_pool(x))))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class SpatialAttention(nn.Module):
    """Spatial Attention Module for spatial feature enhancement"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return x * self.sigmoid(out)

class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, in_channels):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels)
        self.spatial_attention = SpatialAttention()
        
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class ResidualBlock(nn.Module):
    """Enhanced Residual Block with CBAM attention"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.attention = CBAM(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Skip connection adjustment
        self.skip = nn.Sequential()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.attention(out)
        out += residual
        return self.relu(out)

class FeaturePyramidBlock(nn.Module):
    """Multi-scale feature extraction using feature pyramid"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, 1),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, 1),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels//4, out_channels//4, 3, padding=1),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, 1),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels//4, out_channels//4, 3, padding=2, dilation=2),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True)
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, 1),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels//4, out_channels//4, 3, padding=4, dilation=4),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b3, b4], dim=1)

class ProgressiveRefinementModule(nn.Module):
    """Progressive refinement for hierarchical deblurring"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.coarse_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            ResidualBlock(out_channels, out_channels)
        )
        
        self.fine_branch = nn.Sequential(
            nn.Conv2d(in_channels + out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            FeaturePyramidBlock(out_channels, out_channels),
            ResidualBlock(out_channels, out_channels)
        )
        
    def forward(self, x):
        coarse = self.coarse_branch(x)
        fine_input = torch.cat([x, coarse], dim=1)
        fine = self.fine_branch(fine_input)
        return coarse + fine

# ============================================================================
# SECTION 3: GENERATOR (HIERARCHICAL DEBLUR NET)
# ============================================================================
print("\n[SECTION 3] Constructing Generator (Deblurring Network)...")

class HierarchicalDeblurNet(nn.Module):
    """Generator: Hierarchical Multi-Scale Motion Deblurring Network"""
    def __init__(self):
        super().__init__()
        print("Initializing HierarchicalDeblurNet (Generator)...")
        
        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.encoder2 = ProgressiveRefinementModule(64, 128)
        self.encoder3 = ProgressiveRefinementModule(128, 256)
        self.encoder4 = ProgressiveRefinementModule(256, 512)
        
        # Bottleneck with multi-scale processing
        self.bottleneck = nn.Sequential(
            FeaturePyramidBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            CBAM(512)
        )
        
        # Decoder with skip connections
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.refine4 = ProgressiveRefinementModule(512, 256)
        
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.refine3 = ProgressiveRefinementModule(256, 128)
        
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.refine2 = ProgressiveRefinementModule(128, 64)
        
        # Final output layers
        self.final_conv = nn.Sequential(
            ResidualBlock(128, 64),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh()
        )
        
        print("Generator architecture initialized successfully!")
        self._print_network_info()
    
    def _print_network_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Generator parameters: {total_params:,}")
        print(f"Generator trainable parameters: {trainable_params:,}")
    
    def forward(self, x):
        # Store input for skip connection
        input_img = x
        
        # Encoder with multi-scale downsampling
        e1 = self.encoder1(x)  # 64 channels
        e2 = self.encoder2(F.avg_pool2d(e1, 2))  # 128 channels, 1/2 size
        e3 = self.encoder3(F.avg_pool2d(e2, 2))  # 256 channels, 1/4 size
        e4 = self.encoder4(F.avg_pool2d(e3, 2))  # 512 channels, 1/8 size
        
        # Bottleneck processing
        bottleneck = self.bottleneck(e4)
        
        # Decoder with progressive refinement
        d4 = self.decoder4(bottleneck)
        d4 = self.refine4(torch.cat([d4, e3], dim=1))
        
        d3 = self.decoder3(d4)
        d3 = self.refine3(torch.cat([d3, e2], dim=1))
        
        d2 = self.decoder2(d3)
        d2 = self.refine2(torch.cat([d2, e1], dim=1))
        
        # Final refinement with input skip connection
        final_features = torch.cat([d2, e1], dim=1)
        residual = self.final_conv(final_features)
        
        # Residual learning: add input to learn the residual
        return input_img + residual

# ============================================================================
# SECTION 4: PATCHGAN DISCRIMINATOR
# ============================================================================
print("\n[SECTION 4] Constructing PatchGAN Discriminator...")

class PatchGANDiscriminator(nn.Module):
    """PatchGAN Discriminator for adversarial training"""
    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_sigmoid=False):
        super().__init__()
        print("Initializing PatchGAN Discriminator...")
        
        def discriminator_block(in_filters, out_filters, stride=2, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride, 1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        layers = []
        
        # First layer (no normalization)
        layers.extend(discriminator_block(input_nc, ndf, normalize=False))
        
        # Gradually increase the number of filters
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            layers.extend(discriminator_block(ndf * nf_mult_prev, ndf * nf_mult))
        
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        layers.extend(discriminator_block(ndf * nf_mult_prev, ndf * nf_mult, stride=1))
        
        # Final layer
        layers.append(nn.Conv2d(ndf * nf_mult, 1, 4, 1, 1))
        if use_sigmoid:
            layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
        
        print("Discriminator architecture initialized successfully!")
        self._print_network_info()
    
    def _print_network_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Discriminator parameters: {total_params:,}")
        print(f"Discriminator trainable parameters: {trainable_params:,}")
    
    def forward(self, input):
        return self.model(input)

# ============================================================================
# SECTION 5: PERCEPTUAL LOSS (VGG-BASED)
# ============================================================================
print("\n[SECTION 5] Setting up VGG-based Perceptual Loss...")

class VGGPerceptualLoss(nn.Module):
    """VGG-based perceptual loss for better visual quality"""
    def __init__(self, feature_layers=[2, 7, 12, 21, 30]):
        super().__init__()
        # Load pre-trained VGG19
        self.vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.eval() # Updated
        
        # Freeze VGG parameters
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        self.feature_layers = feature_layers
        self.criterion = nn.L1Loss()
        
        # Register VGG normalization constants as buffers (automatically handled for device)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        print(f"VGG Perceptual Loss initialized with layers: {feature_layers}")
    
    def forward(self, input, target):
        # Denormalize images from [-1, 1] to [0, 1] for VGG
        input_vgg = (input + 1) / 2
        target_vgg = (target + 1) / 2
        
        # Normalize for VGG (ImageNet stats) - buffers automatically on correct device
        input_vgg = (input_vgg - self.mean) / self.std
        target_vgg = (target_vgg - self.mean) / self.std
        
        # Extract features
        input_features = []
        target_features = []
        
        x_input = input_vgg
        x_target = target_vgg
        
        for i, layer in enumerate(self.vgg):
            x_input = layer(x_input)
            x_target = layer(x_target)
            
            if i in self.feature_layers:
                input_features.append(x_input)
                target_features.append(x_target)
        
        # Calculate perceptual loss
        perceptual_loss = 0
        for input_feat, target_feat in zip(input_features, target_features):
            perceptual_loss += self.criterion(input_feat, target_feat)
        
        return perceptual_loss

# ============================================================================
# SECTION 6: GRADIENT/EDGE LOSS
# ============================================================================
print("\n[SECTION 6] Setting up Gradient/Edge Loss...")

class GradientLoss(nn.Module):
    """Gradient loss for edge sharpness enhancement"""
    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss()
        
        # Sobel kernels for gradient calculation
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.sobel_x = sobel_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        self.sobel_y = sobel_y.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        
        print("Gradient Loss initialized with Sobel operators")
    
    def get_gradients(self, img):
        # Move kernels to same device as image
        sobel_x = self.sobel_x.to(img.device)
        sobel_y = self.sobel_y.to(img.device)
        
        # Calculate gradients for each channel
        grad_x = F.conv2d(img, sobel_x, padding=1, groups=3)
        grad_y = F.conv2d(img, sobel_y, padding=1, groups=3)
        
        # Calculate gradient magnitude
        grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        return grad_magnitude
    
    def forward(self, input, target):
        input_grad = self.get_gradients(input)
        target_grad = self.get_gradients(target)
        return self.criterion(input_grad, target_grad)

# ============================================================================
# SECTION 7: COMPREHENSIVE LOSS FUNCTIONS
# ============================================================================
print("\n[SECTION 7] Setting up comprehensive loss functions...")

class ComprehensiveLoss(nn.Module):
    """Comprehensive loss combining pixel, perceptual, and gradient losses"""
    def __init__(self, lambda_pixel=1.0, lambda_perceptual=0.01, lambda_gradient=0.1):
        super().__init__()
        self.lambda_pixel = lambda_pixel
        self.lambda_perceptual = lambda_perceptual
        self.lambda_gradient = lambda_gradient
        
        self.pixel_loss = nn.L1Loss()
        self.perceptual_loss = VGGPerceptualLoss()
        self.gradient_loss = GradientLoss()
        
        print(f"Comprehensive Loss initialized:")
        print(f"- Pixel Loss weight: {lambda_pixel}")
        print(f"- Perceptual Loss weight: {lambda_perceptual}")
        print(f"- Gradient Loss weight: {lambda_gradient}")
    
    def forward(self, pred, target):
        pixel_l = self.pixel_loss(pred, target)
        perceptual_l = self.perceptual_loss(pred, target)
        gradient_l = self.gradient_loss(pred, target)
        
        total_loss = (self.lambda_pixel * pixel_l + 
                     self.lambda_perceptual * perceptual_l + 
                     self.lambda_gradient * gradient_l)
        
        return {
            'total': total_loss,
            'pixel': pixel_l,
            'perceptual': perceptual_l,
            'gradient': gradient_l
        }

class GANLoss(nn.Module):
    """GAN loss for adversarial training"""
    def __init__(self, use_lsgan=True):
        super().__init__()
        self.use_lsgan = use_lsgan
        if use_lsgan:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        
        print(f"GAN Loss initialized with {'LSGAN' if use_lsgan else 'vanilla GAN'}")
    
    def forward(self, prediction, target_is_real):
        if self.use_lsgan:
            target = torch.ones_like(prediction) if target_is_real else torch.zeros_like(prediction)
        else: # For BCEWithLogitsLoss
            target = torch.ones_like(prediction) if target_is_real else torch.zeros_like(prediction)
        
        return self.criterion(prediction, target)

# ============================================================================
# SECTION 8: EVALUATION METRICS
# ============================================================================
print("\n[SECTION 8] Setting up evaluation metrics...")

def calculate_psnr_batch(pred, target):
    """Calculate PSNR for a batch of images"""
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    psnr_values = []
    for i in range(pred_np.shape[0]):
        # Convert from [-1, 1] to [0, 1]
        pred_img = (pred_np[i].transpose(1, 2, 0) + 1) / 2
        target_img = (target_np[i].transpose(1, 2, 0) + 1) / 2
        
        # Clip values to valid range
        pred_img = np.clip(pred_img, 0, 1)
        target_img = np.clip(target_img, 0, 1)
        
        psnr = peak_signal_noise_ratio(target_img, pred_img, data_range=1.0)
        psnr_values.append(psnr)
    
    return np.mean(psnr_values)

def calculate_ssim_batch(pred, target):
    """Calculate SSIM for a batch of images"""
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    ssim_values = []
    for i in range(pred_np.shape[0]):
        pred_img = (pred_np[i].transpose(1, 2, 0) + 1) / 2
        target_img = (target_np[i].transpose(1, 2, 0) + 1) / 2
        
        pred_img = np.clip(pred_img, 0, 1)
        target_img = np.clip(target_img, 0, 1)
        
        # Get image dimensions
        height, width = pred_img.shape[:2]
        
        # Calculate appropriate window size
        min_dim = min(height, width)
        win_size = min(7, min_dim)
        if win_size % 2 == 0:
            win_size -= 1
        if win_size < 3: # Ensure win_size is at least 3 and odd
            win_size = 3 
        
        try:
            ssim = structural_similarity(
                target_img, pred_img, 
                channel_axis=2, # For RGB images
                data_range=1.0,
                win_size=win_size
            )
        except Exception as e:
            # print(f"SSIM calculation failed for image {i}: {e}")
            ssim = 0.0 # Fallback value
            
        ssim_values.append(ssim)
    
    return np.mean(ssim_values)

# ============================================================================
# SECTION 9: GAN TRAINING CONFIGURATION
# ============================================================================
print("\n[SECTION 9] Setting up GAN training configuration...")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Hyperparameters
BATCH_SIZE = 4
LEARNING_RATE_G = 1e-4  # Generator learning rate
LEARNING_RATE_D = 4e-4  # Discriminator learning rate (higher than generator)
NUM_EPOCHS = 950        # TARGET TOTAL number of epochs for the model
IMAGE_SIZE = 256

# Loss weights
LAMBDA_PIXEL = 100.0      # High weight for pixel reconstruction
LAMBDA_PERCEPTUAL = 1.0   # Perceptual loss weight
LAMBDA_GRADIENT = 10.0    # Gradient loss weight
LAMBDA_ADV = 1.0          # Adversarial loss weight

# --- Resume Training Configuration ---
RESUME_TRAINING = True  # << SET TO True TO RESUME TRAINING >>
# Path to the checkpoint to resume from.
# If None or empty, and RESUME_TRAINING is True, will try to use `best_model_path` (defined in Sec 11).
CHECKPOINT_TO_RESUME_PATH = "/kaggle/input/deblur-650/best_deblur_gan_model.pth" # Example: 'best_deblur_gan_model.pth' or 'specific_checkpoint_epoch_50.pth'
# --- End Resume Training Configuration ---


print(f"GAN Training Configuration:")
print(f"- Batch Size: {BATCH_SIZE}")
print(f"- Generator LR: {LEARNING_RATE_G}")
print(f"- Discriminator LR: {LEARNING_RATE_D}")
print(f"- Target Total Epochs: {NUM_EPOCHS}")
print(f"- Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"- Loss Weights: Pixel={LAMBDA_PIXEL}, Perceptual={LAMBDA_PERCEPTUAL}, Gradient={LAMBDA_GRADIENT}, Adversarial={LAMBDA_ADV}")
if RESUME_TRAINING:
    print(f"- Resuming Training: True")
    print(f"- Checkpoint to Resume: {CHECKPOINT_TO_RESUME_PATH if CHECKPOINT_TO_RESUME_PATH else 'Default (best_model_path)'}")


# Enhanced data transforms with more augmentation
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(degrees=5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ============================================================================
# SECTION 10: DATASET LOADING
# ============================================================================
print("\n[SECTION 10] Loading datasets...")

# Adjust these paths as per your environment
TRAIN_PATH = '/kaggle/input/gopro-data/train'
TEST_PATH = '/kaggle/input/gopro-data/test'

# Fallback for local testing if Kaggle paths don't exist
if not os.path.exists(TRAIN_PATH) or not os.path.isdir(TRAIN_PATH):
    print(f"Warning: Train path {TRAIN_PATH} not found or not a directory. Using './train_dummy' for testing.")
    TRAIN_PATH = './train_dummy' 
    if not os.path.exists(TRAIN_PATH): os.makedirs(TRAIN_PATH, exist_ok=True)
    # Create dummy structure if needed for GoPro_Dataset initialization
    if not os.path.exists(os.path.join(TRAIN_PATH, 'GOPR0000')):
        os.makedirs(os.path.join(TRAIN_PATH, 'GOPR0000', 'blur'), exist_ok=True)
        os.makedirs(os.path.join(TRAIN_PATH, 'GOPR0000', 'sharp'), exist_ok=True)
        # Create dummy images
        Image.new('RGB', (60, 60), color = 'red').save(os.path.join(TRAIN_PATH, 'GOPR0000', 'blur', 'dummy_blur.png'))
        Image.new('RGB', (60, 60), color = 'green').save(os.path.join(TRAIN_PATH, 'GOPR0000', 'sharp', 'dummy_sharp.png'))


if not os.path.exists(TEST_PATH) or not os.path.isdir(TEST_PATH):
    print(f"Warning: Test path {TEST_PATH} not found or not a directory. Using './test_dummy' for testing.")
    TEST_PATH = './test_dummy'
    if not os.path.exists(TEST_PATH): os.makedirs(TEST_PATH, exist_ok=True)
    if not os.path.exists(os.path.join(TEST_PATH, 'GOPR0000')):
        os.makedirs(os.path.join(TEST_PATH, 'GOPR0000', 'blur'), exist_ok=True)
        os.makedirs(os.path.join(TEST_PATH, 'GOPR0000', 'sharp'), exist_ok=True)
        Image.new('RGB', (60, 60), color = 'blue').save(os.path.join(TEST_PATH, 'GOPR0000', 'blur', 'dummy_blur.png'))
        Image.new('RGB', (60, 60), color = 'yellow').save(os.path.join(TEST_PATH, 'GOPR0000', 'sharp', 'dummy_sharp.png'))

try:
    print("Loading training dataset...")
    # Limiting samples for faster testing if needed, adjust max_samples
    train_dataset = GoPro_Dataset(TRAIN_PATH, transform=train_transform, max_samples=100 if "dummy" in TRAIN_PATH else 1000) 
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    
    print("Loading test dataset...")
    test_dataset = GoPro_Dataset(TEST_PATH, transform=test_transform, max_samples=50 if "dummy" in TEST_PATH else 200)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    if len(train_dataset) == 0 or len(test_dataset) == 0:
        raise ValueError("Datasets are empty. Check paths and data.")

except Exception as e:
    print(f"Error loading real datasets: {e}")
    print("Using minimal dummy data loaders for code execution.")
    # Create minimal dummy data loaders if real data loading fails completely
    dummy_blur_tensor = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
    dummy_sharp_tensor = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
    train_loader = [(dummy_blur_tensor, dummy_sharp_tensor) for _ in range(5)] # 5 batches
    test_loader = [(dummy_blur_tensor, dummy_sharp_tensor) for _ in range(2)]  # 2 batches


# ============================================================================
# SECTION 11: MODEL INITIALIZATION
# ============================================================================
print("\n[SECTION 11] Initializing GAN models, optimizers, and schedulers...")

# Initialize networks
generator = HierarchicalDeblurNet().to(device)
discriminator = PatchGANDiscriminator().to(device)

# Optimizers (initialize first, states will be loaded if resuming)
optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE_G, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE_D, betas=(0.5, 0.999))

# Schedulers (initialize with current NUM_EPOCHS, states will be loaded if resuming)
scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=NUM_EPOCHS, eta_min=1e-7) # Adjusted eta_min
scheduler_D = optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=NUM_EPOCHS, eta_min=1e-7) # Adjusted eta_min

# Loss functions
comprehensive_loss = ComprehensiveLoss(LAMBDA_PIXEL, LAMBDA_PERCEPTUAL, LAMBDA_GRADIENT).to(device)
gan_loss = GANLoss(use_lsgan=True).to(device) # Using LSGAN

# Initialize training state variables
start_epoch_idx = 0  # 0-indexed epoch to start from
best_psnr = 0.0
train_losses_G = []
train_losses_D = []
train_psnrs = []
best_model_path = 'best_deblur_gan_model.pth'  # Fixed name for the best model

# Function to initialize weights (call only if not resuming or resume fails)
def init_weights(net, init_type='kaiming', init_gain=0.02): # kaiming is good for ReLUs
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(f'initialization method [{init_type}] is not implemented')
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)
    
    print(f'Initializing network with {init_type} initialization')
    net.apply(init_func)


# Load checkpoint if resuming
if RESUME_TRAINING:
    # Determine the path for the checkpoint to resume from
    _resume_path_to_try = CHECKPOINT_TO_RESUME_PATH if CHECKPOINT_TO_RESUME_PATH else best_model_path
    
    if os.path.exists(_resume_path_to_try):
        print(f"Attempting to resume training from checkpoint: {_resume_path_to_try}")
        try:
            checkpoint = torch.load(_resume_path_to_try, map_location=device, weights_only=False)
            
            generator.load_state_dict(checkpoint['generator_state_dict'])
            discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
            
            # Schedulers are initialized with the new NUM_EPOCHS (T_max).
            # Loading state_dict will restore their last_epoch and base_lrs,
            # effectively continuing the schedule within the new T_max.
            scheduler_G.load_state_dict(checkpoint['scheduler_G_state_dict'])
            scheduler_D.load_state_dict(checkpoint['scheduler_D_state_dict'])
            
            start_epoch_idx = checkpoint['epoch']  # This was saved as 'epoch + 1' (0-indexed for next run)
            best_psnr = checkpoint.get('best_psnr', 0.0) # .get for backward compatibility
            train_losses_G = checkpoint.get('train_losses_G', [])
            train_losses_D = checkpoint.get('train_losses_D', [])
            train_psnrs = checkpoint.get('train_psnrs', [])
            
            print(f"Successfully resumed training.")
            print(f"Starting from epoch {start_epoch_idx + 1} (0-indexed: {start_epoch_idx}). Target total epochs: {NUM_EPOCHS}")
            print(f"Previous best PSNR: {best_psnr:.2f} dB. Loaded {len(train_losses_G)} historical loss records.")
            print(f"Current G LR: {scheduler_G.get_last_lr()[0]:.2e}, D LR: {scheduler_D.get_last_lr()[0]:.2e}")

            if start_epoch_idx >= NUM_EPOCHS:
                print(f"Warning: Resumed epoch ({start_epoch_idx}) is already >= target NUM_EPOCHS ({NUM_EPOCHS}).")
                print(f"Increase NUM_EPOCHS if you want to train further. Otherwise, will proceed to evaluation.")
        
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting from scratch and initializing weights.")
            start_epoch_idx = 0 # Reset epoch index
            best_psnr = 0.0     # Reset best PSNR
            train_losses_G, train_losses_D, train_psnrs = [], [], [] # Reset history
            init_weights(generator)
            init_weights(discriminator)

    else:
        print(f"Warning: RESUME_TRAINING is True, but checkpoint path '{_resume_path_to_try}' not found.")
        print("Starting new training and initializing model weights...")
        init_weights(generator)
        init_weights(discriminator)
else:
    print("Starting new training. Initializing model weights...")
    init_weights(generator)
    init_weights(discriminator)

print("GAN models, optimizers, and schedulers setup complete!")


# ============================================================================
# SECTION 12: GAN TRAINING LOOP
# ============================================================================
print("\n[SECTION 12] Starting GAN training loop...")
print("="*80)

# Debug function for intermediate outputs
def save_intermediate_outputs(generator, blur_img, epoch_display_num, batch_idx): # epoch_display_num is 1-indexed
    """Save intermediate outputs for debugging"""
    if batch_idx == 0 and epoch_display_num % 10 == 0:  # Save every 10 epochs, first batch
        if not os.path.exists("intermediate_outputs"):
            os.makedirs("intermediate_outputs")
            
        with torch.no_grad():
            generator.eval()
            
            x = blur_img[:1]  # Take first image
            
            # Forward through encoder
            e1 = generator.encoder1(x)
            # ... (rest of the forward pass to get residual, simplified here for brevity)
            residual = generator(x) - x # Deblurred - Blurred = Residual
            
            # Save residual visualization
            residual_vis = (residual[0].detach().cpu() + 1) / 2 # Assuming Tanh output, bring to [0,1]
            residual_vis = residual_vis.permute(1, 2, 0).numpy()
            residual_vis = np.clip(residual_vis, 0, 1)
            
            plt.figure(figsize=(8, 8))
            plt.imshow(residual_vis)
            plt.title(f'Learned Residual - Epoch {epoch_display_num}')
            plt.axis('off')
            plt.savefig(f'intermediate_outputs/residual_epoch_{epoch_display_num}.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            generator.train()

if start_epoch_idx < NUM_EPOCHS: # Only train if there are epochs left to run
    for epoch in range(start_epoch_idx, NUM_EPOCHS): # epoch is 0-indexed
        epoch_display_num = epoch + 1
        print(f"\nEPOCH {epoch_display_num}/{NUM_EPOCHS}")
        print("-" * 60)
        
        generator.train()
        discriminator.train()
        
        epoch_loss_G = 0.0
        epoch_loss_D = 0.0
        epoch_psnr = 0.0
        num_batches = 0
        
        start_time = time.time()
        
        # Use tqdm for progress bar over train_loader
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch_display_num} Training", leave=False)
        for batch_idx, (blur_imgs, sharp_imgs) in enumerate(progress_bar):
            # Move data to device
            blur_imgs = blur_imgs.to(device)
            sharp_imgs = sharp_imgs.to(device)
            
            # ===========================
            # Train Discriminator
            # ===========================
            optimizer_D.zero_grad()
            
            # Real images
            pred_real = discriminator(sharp_imgs)
            loss_D_real = gan_loss(pred_real, True)
            
            # Fake images
            with torch.no_grad(): # Detach generator output from its graph for D training
                fake_imgs_detached = generator(blur_imgs).detach()
            pred_fake = discriminator(fake_imgs_detached)
            loss_D_fake = gan_loss(pred_fake, False)
            
            # Total discriminator loss
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()
            
            # ===========================
            # Train Generator
            # ===========================
            optimizer_G.zero_grad()
            
            # Generate fake images (now within G's computation graph)
            fake_imgs = generator(blur_imgs)
            
            # Adversarial loss
            pred_fake_for_G = discriminator(fake_imgs) # Re-evaluate fake images for G loss
            loss_G_adv = gan_loss(pred_fake_for_G, True) * LAMBDA_ADV
            
            # Comprehensive loss (pixel + perceptual + gradient)
            loss_dict = comprehensive_loss(fake_imgs, sharp_imgs)
            loss_G_comprehensive = loss_dict['total']
            
            # Total generator loss
            loss_G = loss_G_adv + loss_G_comprehensive
            loss_G.backward()
            
            # Gradient clipping for Generator (optional, but can help stability)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            optimizer_G.step()
            
            # Calculate metrics
            batch_psnr = calculate_psnr_batch(fake_imgs, sharp_imgs)
            
            epoch_loss_G += loss_G.item()
            epoch_loss_D += loss_D.item()
            epoch_psnr += batch_psnr
            num_batches += 1
            
            # Save intermediate outputs for debugging
            if batch_idx == 0:
                save_intermediate_outputs(generator, blur_imgs, epoch_display_num, batch_idx)
            
            # Update tqdm progress bar
            progress_bar.set_postfix({
                'G_Loss': f"{loss_G.item():.4f}",
                'D_Loss': f"{loss_D.item():.4f}",
                'PSNR': f"{batch_psnr:.2f}dB"
            })
        
        progress_bar.close() # Close tqdm bar for the epoch

        # Calculate epoch averages
        avg_loss_G = epoch_loss_G / num_batches if num_batches > 0 else 0
        avg_loss_D = epoch_loss_D / num_batches if num_batches > 0 else 0
        avg_psnr = epoch_psnr / num_batches if num_batches > 0 else 0
        epoch_time = time.time() - start_time
        
        train_losses_G.append(avg_loss_G)
        train_losses_D.append(avg_loss_D)
        train_psnrs.append(avg_psnr)
        
        # Update learning rates
        scheduler_G.step()
        scheduler_D.step()
        
        current_lr_G = optimizer_G.param_groups[0]['lr']
        current_lr_D = optimizer_D.param_groups[0]['lr']
        
        print(f"\nEpoch {epoch_display_num} Summary:")
        print(f"- Generator Loss: {avg_loss_G:.4f}")
        print(f"- Discriminator Loss: {avg_loss_D:.4f}")
        print(f"- Average Training PSNR: {avg_psnr:.2f} dB")
        print(f"- Generator LR: {current_lr_G:.2e}")
        print(f"- Discriminator LR: {current_lr_D:.2e}")
        print(f"- Epoch Time: {epoch_time:.1f}s")
        
        # Save best model
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save({
                'epoch': epoch + 1, # Save the next 0-indexed epoch number to start from
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'scheduler_G_state_dict': scheduler_G.state_dict(),
                'scheduler_D_state_dict': scheduler_D.state_dict(),
                'best_psnr': best_psnr,
                'train_losses_G': train_losses_G,
                'train_losses_D': train_losses_D,
                'train_psnrs': train_psnrs,
                'num_epochs_config': NUM_EPOCHS # Save the NUM_EPOCHS config at time of saving
            }, best_model_path)
            print(f"*** NEW BEST GAN MODEL SAVED! PSNR: {best_psnr:.2f} dB (Epoch {epoch_display_num}) ***")
        
        # Memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
else:
    print("Training skipped as start_epoch_idx >= NUM_EPOCHS.")


print("\n" + "="*80)
print("GAN TRAINING PHASE COMPLETED!")
print(f"Best Training PSNR achieved: {best_psnr:.2f} dB")
print(f"Model checkpoints and history saved in '{best_model_path}'")
print("="*80)

# ============================================================================
# SECTION 13: MODEL EVALUATION
# ============================================================================
print("\n[SECTION 13] Evaluating GAN model on test set...")

# Load best model for evaluation (even if training was skipped)
if os.path.exists(best_model_path):
    print(f"Loading best trained GAN model from: {best_model_path}")
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    loaded_epoch = checkpoint.get('epoch', 'N/A') # epoch saved is next epoch to run
    if isinstance(loaded_epoch, int): loaded_epoch -=1 # actual last completed epoch (0-indexed)
    
    print(f"Loaded model from checkpoint (last completed epoch {loaded_epoch if isinstance(loaded_epoch,int) else 'N/A'}). Training PSNR: {checkpoint.get('best_psnr', 'N/A'):.2f} dB")
else:
    print(f"Warning: Best model path {best_model_path} not found. Evaluation might use an untrained model.")


# Evaluation mode
generator.eval()

# Test evaluation
test_psnrs = []
test_ssims = []

if len(test_loader) == 0 or (isinstance(test_loader, list) and len(test_loader[0][0])==0 ): # Check for dummy or empty loader
    print("Test loader is empty or invalid. Skipping evaluation.")
    final_psnr, final_ssim, psnr_std, ssim_std = 0.0, 0.0, 0.0, 0.0
else:
    total_batches = len(test_loader)
    print(f"Starting evaluation on {total_batches} test batches...")
    eval_progress_bar = tqdm(test_loader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for batch_idx, (blur_imgs, sharp_imgs) in enumerate(eval_progress_bar):
            blur_imgs = blur_imgs.to(device)
            sharp_imgs = sharp_imgs.to(device)
            
            # Generate deblurred images
            pred_imgs = generator(blur_imgs)
            
            # Calculate metrics
            batch_psnr = calculate_psnr_batch(pred_imgs, sharp_imgs)
            batch_ssim = calculate_ssim_batch(pred_imgs, sharp_imgs)
            
            test_psnrs.append(batch_psnr)
            test_ssims.append(batch_ssim)
            
            eval_progress_bar.set_postfix({
                'PSNR': f"{batch_psnr:.2f}dB",
                'SSIM': f"{batch_ssim:.4f}"
            })
    eval_progress_bar.close()

    # Calculate final metrics
    final_psnr = np.mean(test_psnrs) if test_psnrs else 0
    final_ssim = np.mean(test_ssims) if test_ssims else 0
    psnr_std = np.std(test_psnrs) if test_psnrs else 0
    ssim_std = np.std(test_ssims) if test_ssims else 0


print("\n" + "="*80)
print("FINAL GAN EVALUATION RESULTS")
print("="*80)
print(f"Test Set PSNR: {final_psnr:.2f} ± {psnr_std:.2f} dB")
print(f"Test Set SSIM: {final_ssim:.4f} ± {ssim_std:.4f}")
if test_psnrs:
    print(f"Number of test samples evaluated: {len(test_psnrs) * BATCH_SIZE if not isinstance(test_loader, list) else len(test_psnrs) * test_loader[0][0].size(0)}") # Adjust for dummy loader
else:
    print("Number of test samples evaluated: 0")
print("="*80)

# ============================================================================
# SECTION 14: ENHANCED SAMPLE VISUALIZATION
# ============================================================================
print("\n[SECTION 14] Generating enhanced sample comparisons...")

def denormalize_image(tensor):
    """Convert tensor from [-1, 1] to [0, 1] range"""
    return (tensor + 1.0) / 2.0

def tensor_to_numpy(tensor):
    """Convert tensor to numpy array for visualization"""
    return tensor.detach().cpu().permute(1, 2, 0).numpy() # Adjusted for CHW to HWC

def calculate_ssim_safe(img1, img2, data_range=1.0):
    """Safe SSIM calculation"""
    height, width = img1.shape[:2]
    min_dim = min(height, width)
    win_size = min(7, min_dim)
    if win_size % 2 == 0: win_size -= 1
    if win_size < 3: win_size = 3
    
    try:
        ssim_val = structural_similarity(
            img1, img2, 
            channel_axis=2, # For RGB HWC images
            data_range=data_range,
            win_size=win_size,
            multichannel=True # Deprecated, use channel_axis instead for skimage >= 0.19
        )
    except TypeError: # For older skimage that uses multichannel
         ssim_val = structural_similarity(
            img1, img2, 
            data_range=data_range,
            win_size=win_size,
            multichannel=True 
        )
    except Exception as e:
        print(f"SSIM calculation failed: {e}")
        ssim_val = 0.0
    return ssim_val

# Get sample for visualization
sample_blur, sample_sharp = None, None
if len(test_loader) > 0 and not (isinstance(test_loader, list) and len(test_loader[0][0])==0 ):
    with torch.no_grad():
        # Iterate to get the first actual batch if using DataLoader
        data_iter = iter(test_loader)
        try:
            blur_imgs, sharp_imgs = next(data_iter)
            sample_blur = blur_imgs[:1].to(device)
            sample_sharp = sharp_imgs[:1].to(device)
        except StopIteration:
            print("Test loader is empty, cannot get sample for visualization.")
        except Exception as e:
            print(f"Error getting sample from test_loader: {e}")

if sample_blur is not None and sample_sharp is not None:
    generator.eval()
    with torch.no_grad():
        sample_pred = generator(sample_blur)
    
    # Denormalize images
    blur_img_vis = denormalize_image(sample_blur[0])
    sharp_img_vis = denormalize_image(sample_sharp[0])
    pred_img_vis = denormalize_image(sample_pred[0])
    
    # Convert to numpy
    blur_np = np.clip(tensor_to_numpy(blur_img_vis), 0, 1)
    sharp_np = np.clip(tensor_to_numpy(sharp_img_vis), 0, 1)
    pred_np = np.clip(tensor_to_numpy(pred_img_vis), 0, 1)
    
    # Calculate metrics for this sample
    sample_psnr = peak_signal_noise_ratio(sharp_np, pred_np, data_range=1.0)
    sample_ssim = calculate_ssim_safe(sharp_np, pred_np, data_range=1.0)
    
    print(f"Sample Image Metrics:")
    print(f"- PSNR: {sample_psnr:.2f} dB")
    print(f"- SSIM: {sample_ssim:.4f}")
    
    # Create enhanced comparison
    try:
        if not os.path.exists("visualizations"):
            os.makedirs("visualizations")
            
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        axes[0, 0].imshow(blur_np)
        axes[0, 0].set_title('Input (Blurred)', fontsize=14); axes[0, 0].axis('off')
        
        axes[0, 1].imshow(pred_np)
        axes[0, 1].set_title(f'GAN Deblurred\nPSNR: {sample_psnr:.2f}dB | SSIM: {sample_ssim:.4f}', fontsize=14); axes[0, 1].axis('off')
        
        axes[0, 2].imshow(sharp_np)
        axes[0, 2].set_title('Ground Truth (Sharp)', fontsize=14); axes[0, 2].axis('off')
        
        crop_size = max(64, min(sharp_np.shape[0]//4, sharp_np.shape[1]//4)) # Dynamic crop size
        center_y, center_x = sharp_np.shape[0]//2, sharp_np.shape[1]//2
        y1, y2 = center_y - crop_size//2, center_y + crop_size//2
        x1, x2 = center_x - crop_size//2, center_x + crop_size//2
        
        axes[1, 0].imshow(blur_np[y1:y2, x1:x2])
        axes[1, 0].set_title('Blurred (Detail)', fontsize=12); axes[1, 0].axis('off')
        
        axes[1, 1].imshow(pred_np[y1:y2, x1:x2])
        axes[1, 1].set_title('Deblurred (Detail)', fontsize=12); axes[1, 1].axis('off')
        
        axes[1, 2].imshow(sharp_np[y1:y2, x1:x2])
        axes[1, 2].set_title('Sharp (Detail)', fontsize=12); axes[1, 2].axis('off')
        
        plt.tight_layout(pad=1.5)
        plt.savefig('visualizations/gan_deblurring_comparison.png', dpi=300, bbox_inches='tight')
        plt.close(fig) # Close figure to free memory
        print("Enhanced GAN comparison saved as 'visualizations/gan_deblurring_comparison.png'")
        
        blur_sharp_psnr = peak_signal_noise_ratio(sharp_np, blur_np, data_range=1.0)
        improvement = sample_psnr - blur_sharp_psnr
        
        print(f"\nSample GAN Deblurring Performance:")
        print(f"- Input PSNR vs GT: {blur_sharp_psnr:.2f} dB")
        print(f"- Output PSNR vs GT: {sample_psnr:.2f} dB")
        print(f"- PSNR Improvement: {improvement:.2f} dB")
        
    except Exception as e:
        print(f"Could not generate sample visualizations: {e}")
else:
    print("Skipping sample visualization as no valid test samples were loaded.")


# ============================================================================
# SECTION 15: TRAINING PROGRESS VISUALIZATION
# ============================================================================
print("\n[SECTION 15] Generating training progress visualization...")

if train_losses_G and train_losses_D and train_psnrs:
    try:
        if not os.path.exists("visualizations"):
            os.makedirs("visualizations")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12)) # Adjusted size
        epochs_ran = range(1, len(train_losses_G) + 1)

        axes[0, 0].plot(epochs_ran, train_losses_G, label='Generator Loss', color='blue', alpha=0.8)
        axes[0, 0].plot(epochs_ran, train_losses_D, label='Discriminator Loss', color='red', alpha=0.8)
        axes[0, 0].set_title('GAN Training Losses')
        axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend(); axes[0, 0].grid(True, linestyle='--', alpha=0.7)
        
        axes[0, 1].plot(epochs_ran, train_psnrs, color='green', alpha=0.8)
        axes[0, 1].set_title('Training PSNR Progress')
        axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('PSNR (dB)')
        axes[0, 1].grid(True, linestyle='--', alpha=0.7)
        
        loss_ratio = [g/d if d != 0 else np.nan for g, d in zip(train_losses_G, train_losses_D)] # Handle d=0
        axes[1, 0].plot(epochs_ran, loss_ratio, color='purple', alpha=0.8)
        axes[1, 0].set_title('Generator/Discriminator Loss Ratio')
        axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('G_Loss / D_Loss')
        axes[1, 0].grid(True, linestyle='--', alpha=0.7)
        
        if test_psnrs: # Only plot if test_psnrs is not empty
            axes[1, 1].hist(test_psnrs, bins=15, alpha=0.75, color='orange', edgecolor='black')
            axes[1, 1].axvline(final_psnr, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {final_psnr:.2f}dB')
            axes[1, 1].set_title('Test Set PSNR Distribution')
            axes[1, 1].set_xlabel('PSNR (dB)'); axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].legend(); axes[1, 1].grid(True, linestyle='--', alpha=0.7)
        else:
            axes[1, 1].text(0.5, 0.5, 'No test PSNR data to display', ha='center', va='center', transform=axes[1,1].transAxes)
            axes[1, 1].set_title('Test Set PSNR Distribution'); axes[1, 1].axis('off')

        plt.tight_layout(pad=1.5)
        plt.savefig('visualizations/gan_training_analysis.png', dpi=300, bbox_inches='tight')
        plt.close(fig) # Close figure
        print("GAN training analysis saved as 'visualizations/gan_training_analysis.png'")
        
    except Exception as e:
        print(f"Could not generate training plots: {e}")
else:
    print("Skipping training progress visualization as no training data (losses/PSNR) is available.")


# ============================================================================
# SECTION 16: FINAL RESULTS AND SUMMARY
# ============================================================================
print("\n" + "="*80)
print("GAN-BASED DEBLURRING PERFORMANCE SUMMARY")
print("="*80)
print(f"Architecture: GAN with Hierarchical Generator + PatchGAN Discriminator")
print(f"Loss Components: Adversarial + Pixel + Perceptual (VGG) + Gradient")
print(f"Target Total Epochs Configured: {NUM_EPOCHS}")
print(f"Epochs Actually Trained in this run: {len(train_losses_G) - (start_epoch_idx if RESUME_TRAINING and os.path.exists(CHECKPOINT_TO_RESUME_PATH if CHECKPOINT_TO_RESUME_PATH else best_model_path) else 0)}")
print(f"Total Epochs Completed for Model: {len(train_losses_G)}") # Total number of entries in history
print(f"Best Training PSNR: {best_psnr:.2f} dB")
print(f"Final Test PSNR: {final_psnr:.2f} dB (Std: {psnr_std:.2f})")
print(f"Final Test SSIM: {final_ssim:.4f} (Std: {ssim_std:.4f})")
print(f"Generator Parameters: {sum(p.numel() for p in generator.parameters()):,}")
print(f"Discriminator Parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
print("="*80)

# Save comprehensive results to a JSON file
results_dict = {
    'model_info': {
        'architecture': 'GAN-based Hierarchical Deblurring',
        'generator_parameters': sum(p.numel() for p in generator.parameters()),
        'discriminator_parameters': sum(p.numel() for p in discriminator.parameters()),
    },
    'training_config': {
        'target_total_epochs': NUM_EPOCHS,
        'batch_size': BATCH_SIZE,
        'image_size': IMAGE_SIZE,
        'learning_rate_g': LEARNING_RATE_G,
        'learning_rate_d': LEARNING_RATE_D,
        'loss_weights': {
            'pixel': LAMBDA_PIXEL,
            'perceptual': LAMBDA_PERCEPTUAL,
            'gradient': LAMBDA_GRADIENT,
            'adversarial': LAMBDA_ADV
        },
        'resumed_training': RESUME_TRAINING,
        'checkpoint_resumed_from': CHECKPOINT_TO_RESUME_PATH if RESUME_TRAINING and CHECKPOINT_TO_RESUME_PATH else (best_model_path if RESUME_TRAINING else None)
    },
    'training_summary': {
        'epochs_trained_this_run': len(train_losses_G) - (checkpoint.get('epoch', 0) if RESUME_TRAINING and 'checkpoint' in locals() and checkpoint else 0),
        'total_epochs_model_trained_for': len(train_losses_G),
        'best_training_psnr_db': float(f"{best_psnr:.2f}") if best_psnr else 0.0,
    },
    'evaluation_results': {
        'final_test_psnr_db': float(f"{final_psnr:.2f}") if final_psnr else 0.0,
        'final_test_psnr_std_db': float(f"{psnr_std:.2f}") if psnr_std else 0.0,
        'final_test_ssim': float(f"{final_ssim:.4f}") if final_ssim else 0.0,
        'final_test_ssim_std': float(f"{ssim_std:.4f}") if ssim_std else 0.0,
    },
    'training_history_arrays': { # Optional: save full history if needed, can make JSON large
        # 'train_losses_G': train_losses_G,
        # 'train_losses_D': train_losses_D,
        # 'train_psnrs': train_psnrs,
    }
}

if not os.path.exists("results"):
    os.makedirs("results")
with open('results/gan_deblurring_results.json', 'w') as f:
    json.dump(results_dict, f, indent=2)
print("Comprehensive results saved to 'results/gan_deblurring_results.json'")

print(f"\nFinal Answer: Test PSNR = {final_psnr:.2f} dB, Test SSIM = {final_ssim:.4f}")