#!/usr/bin/env python3
"""
Real-ESRGAN Single Image Super-Resolution Implementation
Author: Research Implementation
Description: Modern approach to SISR using Real-ESRGAN architecture with DF2K dataset
"""

import os
import sys
import time
import random
import warnings
from datetime import datetime
from pathlib import Path
import math
import json

# Core ML libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# ============================ CHANGE 1: Import custom_fwd ============================
from torch.cuda.amp import autocast, GradScaler, custom_fwd
from torch.utils.data import Dataset, DataLoader

# Computer Vision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import cv2
import numpy as np
from PIL import Image

# Monitoring and visualization
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorboard
from torch.utils.tensorboard import SummaryWriter

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

print("="*80)
print("🚀 REAL-ESRGAN SINGLE IMAGE SUPER-RESOLUTION IMPLEMENTATION")
print("="*80)
print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"🐍 Python version: {sys.version}")
print(f"🔥 PyTorch version: {torch.__version__}")
print(f"💻 CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"🎮 GPU device: {torch.cuda.get_device_name()}")
    print(f"🎮 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print("-"*80)

class Config:
    """Configuration class for Real-ESRGAN training and inference"""
    
    def __init__(self):
        print("⚙️  Initializing configuration...")
        
        # Dataset configuration
        self.dataset_path = "/kaggle/input/df2kdata"
        self.scale_factor = 4  # Can be 2, 3, or 4
        self.degradation_type = "bicubic"  # "bicubic" or "unknown"
        
        # Model architecture configuration
        self.num_in_ch = 3  # RGB channels
        self.num_out_ch = 3  # RGB channels
        self.num_feat = 64  # Number of feature channels
        self.num_block = 23  # Number of RRDB blocks
        self.num_grow_ch = 32  # Growth channels in dense blocks
        
        # Training configuration
        self.batch_size = 8  # Adjusted for memory efficiency
        self.learning_rate = 2e-4
        self.weight_decay = 0
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.epsilon = 1e-8
        
        # Training schedule
        self.num_epochs = 200
        self.warmup_epochs = 10
        self.lr_scheduler_type = "cosine"  # "cosine", "step", "multistep"
        self.lr_decay_factor = 0.5
        self.lr_decay_steps = [100, 200, 300, 400]
        
        # Loss function weights
        self.pixel_loss_weight = 1.0
        self.perceptual_loss_weight = 1.0
        self.gan_loss_weight = 0.1
        self.l1_loss_weight = 1.0
        
        # Data augmentation
        self.use_augmentation = True
        self.crop_size = 128  # Training patch size
        self.use_flip = True
        self.use_rotation = True
        self.color_jitter = True
        
        # Hardware configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_workers = 4 if torch.cuda.is_available() else 2
        self.pin_memory = True
        self.use_mixed_precision = True
        
        # Checkpoint and logging
        self.save_dir = "experiments/real_esrgan"
        self.checkpoint_freq = 10  # Save every N epochs
        self.validation_freq = 5   # Validate every N epochs
        self.log_freq = 50         # Log every N iterations
        self.save_images_freq = 100 # Save sample images every N iterations
        
        # Resume training
        self.resume_training = True
        self.checkpoint_path = "/kaggle/input/gan-sisr-199-epoch/experiments/real_esrgan/checkpoints"
        
        # Validation configuration
        self.val_batch_size = 1
        self.save_validation_images = True
        
        print(f"✅ Configuration initialized:")
        print(f"   📊 Dataset: {self.dataset_path}")
        print(f"   🔍 Scale factor: {self.scale_factor}x")
        print(f"   🎯 Batch size: {self.batch_size}")
        print(f"   🎓 Learning rate: {self.learning_rate}")
        print(f"   🎮 Device: {self.device}")
        print(f"   🏠 Save directory: {self.save_dir}")
        print(f"   🔄 Resume training: {self.resume_training}")
        if self.resume_training and self.checkpoint_path:
            print(f"   📁 Checkpoint path: {self.checkpoint_path}")
        
    def create_directories(self):
        """Create necessary directories for training"""
        print("📁 Creating directories...")
        
        dirs_to_create = [
            self.save_dir,
            f"{self.save_dir}/checkpoints",
            f"{self.save_dir}/logs", 
            f"{self.save_dir}/samples",
            f"{self.save_dir}/validation"
        ]
        
        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            print(f"   ✅ Created: {dir_path}")
            
    def save_config(self):
        """Save configuration to JSON file"""
        config_dict = {k: v for k, v in self.__dict__.items() 
                      if not k.startswith('_') and not callable(v)}
        config_dict['device'] = str(config_dict['device'])
        
        config_path = f"{self.save_dir}/config.json"
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=4)
        print(f"💾 Configuration saved to: {config_path}")

# Initialize global configuration
config = Config()
config.create_directories()
config.save_config()

print("✅ Section 1 Complete: Imports and Configuration")
print("-"*80)

# ============================================================================
# SECTION 2: DATASET HANDLING AND DATA LOADING
# ============================================================================

class DF2KDataset(Dataset):
    """Custom dataset class for DF2K super-resolution dataset"""
    
    def __init__(self, config, mode='train', transform=None):
        print(f"🗃️  Initializing DF2K Dataset for {mode} mode...")
        
        self.config = config
        self.mode = mode
        self.transform = transform
        self.scale_factor = config.scale_factor
        
        # Define paths based on mode
        if mode == 'train':
            self.hr_path = os.path.join(config.dataset_path, "DF2K_train_HR")
            if config.degradation_type == "bicubic":
                self.lr_path = os.path.join(config.dataset_path, "DF2K_train_LR_bicubic", f"X{config.scale_factor}")
            else:
                self.lr_path = os.path.join(config.dataset_path, "DF2K_train_LR_unknown", f"X{config.scale_factor}")
        else:  # validation
            self.hr_path = os.path.join(config.dataset_path, "DF2K_valid_HR")
            self.lr_path = os.path.join(config.dataset_path, "DF2K_valid_LR_bicubic", f"X{config.scale_factor}")
            
        print(f"   📂 HR Path: {self.hr_path}")
        print(f"   📂 LR Path: {self.lr_path}")
        
        # Get image pairs
        self.image_pairs = self._get_image_pairs()
        
        print(f"   📊 Found {len(self.image_pairs)} image pairs")
        print(f"   🔍 Scale factor: {self.scale_factor}x")
        
    def _get_image_pairs(self):
        """Get list of HR-LR image pairs"""
        print(f"   🔍 Scanning for image pairs...")
        
        pairs = []
        
        if not os.path.exists(self.hr_path):
            raise FileNotFoundError(f"HR path not found: {self.hr_path}")
        if not os.path.exists(self.lr_path):
            raise FileNotFoundError(f"LR path not found: {self.lr_path}")
            
        hr_files = sorted([f for f in os.listdir(self.hr_path) if f.endswith('.png')])
        
        for hr_file in hr_files:
            # Extract image number
            img_num = hr_file.split('.')[0]
            
            # Find corresponding LR file
            lr_file = f"{img_num}x{self.scale_factor}.png"
            lr_full_path = os.path.join(self.lr_path, lr_file)
            hr_full_path = os.path.join(self.hr_path, hr_file)
            
            if os.path.exists(lr_full_path):
                pairs.append((hr_full_path, lr_full_path))
            else:
                print(f"   ⚠️  Missing LR image: {lr_file}")
                
        print(f"   ✅ Found {len(pairs)} valid pairs")
        return pairs
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        try:
            hr_path, lr_path = self.image_pairs[idx]
            
            # Load images
            hr_img = self._load_image(hr_path)
            lr_img = self._load_image(lr_path)
            
            # Apply transforms
            if self.transform:
                hr_img, lr_img = self.transform(hr_img, lr_img)
                
            return {
                'lr': lr_img,
                'hr': hr_img,
                'hr_path': hr_path,
                'lr_path': lr_path
            }
            
        except Exception as e:
            print(f"   ❌ Error loading image at index {idx}: {str(e)}")
            # Return a random valid sample as fallback
            return self.__getitem__(random.randint(0, len(self.image_pairs) - 1))
    
    def _load_image(self, path):
        """Load and preprocess image"""
        try:
            # Use PIL for consistent loading
            img = Image.open(path).convert('RGB')
            img = np.array(img)
            
            # Convert to float and normalize to [0, 1]
            img = img.astype(np.float32) / 255.0
            
            # Convert to tensor (H, W, C) -> (C, H, W)
            img = torch.from_numpy(img).permute(2, 0, 1)
            
            return img
            
        except Exception as e:
            print(f"   ❌ Error loading image {path}: {str(e)}")
            raise

class DataAugmentation:
    """Data augmentation for super-resolution training"""
    
    def __init__(self, config):
        print("🎨 Initializing data augmentation...")
        self.config = config
        self.crop_size = config.crop_size
        self.scale_factor = config.scale_factor
        self.use_flip = config.use_flip
        self.use_rotation = config.use_rotation
        self.color_jitter = config.color_jitter
        
        print(f"   ✂️  Crop size: {self.crop_size}")
        print(f"   🔄 Flip: {self.use_flip}")
        print(f"   🔄 Rotation: {self.use_rotation}")
        print(f"   🌈 Color jitter: {self.color_jitter}")
        
    def __call__(self, hr_img, lr_img):
        """Apply augmentation to HR and LR image pair"""
        
        # Random crop (ensure corresponding patches)
        hr_img, lr_img = self._random_crop_pair(hr_img, lr_img)
        
        # Random horizontal flip
        if self.use_flip and random.random() < 0.5:
            hr_img = torch.flip(hr_img, [2])  # Flip width dimension
            lr_img = torch.flip(lr_img, [2])
            
        # Random vertical flip
        if self.use_flip and random.random() < 0.5:
            hr_img = torch.flip(hr_img, [1])  # Flip height dimension
            lr_img = torch.flip(lr_img, [1])
            
        # Random rotation (90, 180, 270 degrees)
        if self.use_rotation and random.random() < 0.5:
            k = random.randint(1, 3)
            hr_img = torch.rot90(hr_img, k, [1, 2])
            lr_img = torch.rot90(lr_img, k, [1, 2])
            
        # Color jittering
        if self.color_jitter and random.random() < 0.3:
            hr_img, lr_img = self._color_jitter(hr_img, lr_img)
            
        return hr_img, lr_img
    
    def _random_crop_pair(self, hr_img, lr_img):
        """Random crop HR and corresponding LR patches"""
        C, H, W = hr_img.shape
        
        # Calculate LR crop size
        lr_crop_size = self.crop_size // self.scale_factor
        
        # Random crop coordinates for HR
        if H > self.crop_size and W > self.crop_size:
            top = random.randint(0, H - self.crop_size)
            left = random.randint(0, W - self.crop_size)
            
            hr_patch = hr_img[:, top:top+self.crop_size, left:left+self.crop_size]
            
            # Corresponding LR crop
            lr_top = top // self.scale_factor
            lr_left = left // self.scale_factor
            lr_patch = lr_img[:, lr_top:lr_top+lr_crop_size, lr_left:lr_left+lr_crop_size]
        else:
            # If images are smaller than crop size, use the whole image
            hr_patch = hr_img
            lr_patch = lr_img
            
        return hr_patch, lr_patch
    
    def _color_jitter(self, hr_img, lr_img):
        """Apply color jittering to both images"""
        # Random brightness
        brightness_factor = random.uniform(0.8, 1.2)
        hr_img = torch.clamp(hr_img * brightness_factor, 0, 1)
        lr_img = torch.clamp(lr_img * brightness_factor, 0, 1)
        
        # Random contrast
        contrast_factor = random.uniform(0.8, 1.2)
        hr_mean = hr_img.mean()
        lr_mean = lr_img.mean()
        hr_img = torch.clamp((hr_img - hr_mean) * contrast_factor + hr_mean, 0, 1)
        lr_img = torch.clamp((lr_img - lr_mean) * contrast_factor + lr_mean, 0, 1)
        
        return hr_img, lr_img

def create_data_loaders(config):
    """Create training and validation data loaders"""
    print("🔄 Creating data loaders...")
    
    # Initialize transforms
    train_transform = DataAugmentation(config) if config.use_augmentation else None
    val_transform = None  # No augmentation for validation
    
    # Create datasets
    print("   📚 Creating training dataset...")
    train_dataset = DF2KDataset(config, mode='train', transform=train_transform)
    
    print("   📚 Creating validation dataset...")
    val_dataset = DF2KDataset(config, mode='val', transform=val_transform)
    
    # Create data loaders
    print("   🔄 Creating training data loader...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True
    )
    
    print("   🔄 Creating validation data loader...")
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.val_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False
    )
    
    print(f"   ✅ Training samples: {len(train_dataset)}")
    print(f"   ✅ Validation samples: {len(val_dataset)}")
    print(f"   ✅ Training batches: {len(train_loader)}")
    print(f"   ✅ Validation batches: {len(val_loader)}")
    
    return train_loader, val_loader

def test_data_loading():
    """Test data loading functionality"""
    print("🧪 Testing data loading...")
    
    try:
        train_loader, val_loader = create_data_loaders(config)
        
        # Test training loader
        print("   🔍 Testing training loader...")
        for i, batch in enumerate(train_loader):
            lr_batch = batch['lr']
            hr_batch = batch['hr']
            
            print(f"   📊 Batch {i+1}:")
            print(f"      LR shape: {lr_batch.shape}")
            print(f"      HR shape: {hr_batch.shape}")
            print(f"      LR range: [{lr_batch.min():.3f}, {lr_batch.max():.3f}]")
            print(f"      HR range: [{hr_batch.min():.3f}, {hr_batch.max():.3f}]")
            
            if i >= 2:  # Test first 3 batches
                break
                
        print("   ✅ Data loading test successful!")
        return train_loader, val_loader
        
    except Exception as e:
        print(f"   ❌ Data loading test failed: {str(e)}")
        return None, None

print("✅ Section 2 Complete: Dataset Handling")
print("-"*80)

# ============================================================================
# SECTION 3: REAL-ESRGAN NETWORK ARCHITECTURE
# ============================================================================

class DenseBlock(nn.Module):
    """Dense Block for RRDB (Residual in Residual Dense Block)"""
    
    def __init__(self, num_feat=64, num_grow_ch=32):
        super(DenseBlock, self).__init__()
        print(f"   🧱 Initializing Dense Block - Features: {num_feat}, Growth: {num_grow_ch}")
        
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights using Kaiming normal"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        
        # Empirical scaling factor from ESRGAN paper
        return x5 * 0.2 + x

class RRDB(nn.Module):
    """Residual in Residual Dense Block"""
    
    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        print(f"   🏗️  Initializing RRDB - Features: {num_feat}, Growth: {num_grow_ch}")
        
        self.RDB1 = DenseBlock(num_feat, num_grow_ch)
        self.RDB2 = DenseBlock(num_feat, num_grow_ch)
        self.RDB3 = DenseBlock(num_feat, num_grow_ch)
        
    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        
        # Empirical scaling factor from ESRGAN paper
        return out * 0.2 + x

class PixelShuffleUpsampler(nn.Module):
    """Pixel shuffle upsampling block"""
    
    def __init__(self, num_feat, scale_factor):
        super(PixelShuffleUpsampler, self).__init__()
        print(f"   📈 Initializing Pixel Shuffle Upsampler - Scale: {scale_factor}")
        
        self.scale_factor = scale_factor
        
        if scale_factor == 2:
            self.conv = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(2)
        elif scale_factor == 3:
            self.conv = nn.Conv2d(num_feat, num_feat * 9, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(3)
        elif scale_factor == 4:
            self.conv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.conv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.pixel_shuffle1 = nn.PixelShuffle(2)
            self.pixel_shuffle2 = nn.PixelShuffle(2)
        else:
            raise ValueError(f"Unsupported scale factor: {scale_factor}")
            
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        if self.scale_factor in [2, 3]:
            out = self.pixel_shuffle(self.conv(x))
            out = self.lrelu(out)
        elif self.scale_factor == 4:
            out = self.pixel_shuffle1(self.conv1(x))
            out = self.lrelu(out)
            out = self.pixel_shuffle2(self.conv2(out))
            out = self.lrelu(out)
        
        return out

class RealESRGAN(nn.Module):
    """Real-ESRGAN Generator Network"""
    
    def __init__(self, config):
        super(RealESRGAN, self).__init__()
        print("🏭 Initializing Real-ESRGAN Generator...")
        
        self.config = config
        self.num_in_ch = config.num_in_ch
        self.num_out_ch = config.num_out_ch
        self.num_feat = config.num_feat
        self.num_block = config.num_block
        self.num_grow_ch = config.num_grow_ch
        self.scale_factor = config.scale_factor
        
        print(f"   📊 Architecture Details:")
        print(f"      Input channels: {self.num_in_ch}")
        print(f"      Output channels: {self.num_out_ch}")
        print(f"      Feature channels: {self.num_feat}")
        print(f"      RRDB blocks: {self.num_block}")
        print(f"      Growth channels: {self.num_grow_ch}")
        print(f"      Scale factor: {self.scale_factor}")
        
        # Shallow feature extraction
        self.conv_first = nn.Conv2d(self.num_in_ch, self.num_feat, 3, 1, 1)
        
        # Deep feature extraction (RRDB blocks)
        self.body = nn.ModuleList()
        for i in range(self.num_block):
            self.body.append(RRDB(self.num_feat, self.num_grow_ch))
            
        # Feature fusion
        self.conv_body = nn.Conv2d(self.num_feat, self.num_feat, 3, 1, 1)
        
        # Upsampling
        self.upsampler = PixelShuffleUpsampler(self.num_feat, self.scale_factor)
        
        # Final convolution layers
        self.conv_up1 = nn.Conv2d(self.num_feat, self.num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(self.num_feat, self.num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(self.num_feat, self.num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(self.num_feat, self.num_out_ch, 3, 1, 1)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        # Initialize weights
        self._initialize_weights()
        
        # Count parameters
        self._count_parameters()
        
    def _initialize_weights(self):
        """Initialize network weights"""
        print("   🎲 Initializing network weights...")
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def _count_parameters(self):
        """Count total parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"   📊 Network Statistics:")
        print(f"      Total parameters: {total_params:,}")
        print(f"      Trainable parameters: {trainable_params:,}")
        print(f"      Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
        
    def forward(self, x):
        """Forward pass"""
        # Shallow feature extraction
        feat = self.conv_first(x)
        
        # Deep feature extraction
        body_feat = feat
        for block in self.body:
            body_feat = block(body_feat)
            
        # Feature fusion
        body_feat = self.conv_body(body_feat)
        feat = feat + body_feat  # Global residual connection
        
        # Upsampling
        feat = self.upsampler(feat)
        
        # Final convolutions
        feat = self.lrelu(self.conv_up1(feat))
        feat = self.lrelu(self.conv_up2(feat))
        feat = self.lrelu(self.conv_hr(feat))
        out = self.conv_last(feat)
        
        return out

class Discriminator(nn.Module):
    """VGG-style discriminator for adversarial training"""
    
    def __init__(self, num_in_ch=3, num_feat=64):
        super(Discriminator, self).__init__()
        print("🕵️  Initializing VGG-style Discriminator...")
        
        self.num_in_ch = num_in_ch
        self.num_feat = num_feat
        
        # Convolutional layers
        self.conv0_0 = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv2d(num_feat, num_feat, 4, 2, 1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(num_feat, affine=True)
        
        self.conv1_0 = nn.Conv2d(num_feat, num_feat * 2, 3, 1, 1, bias=False)
        self.bn1_0 = nn.BatchNorm2d(num_feat * 2, affine=True)
        self.conv1_1 = nn.Conv2d(num_feat * 2, num_feat * 2, 4, 2, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(num_feat * 2, affine=True)
        
        self.conv2_0 = nn.Conv2d(num_feat * 2, num_feat * 4, 3, 1, 1, bias=False)
        self.bn2_0 = nn.BatchNorm2d(num_feat * 4, affine=True)
        self.conv2_1 = nn.Conv2d(num_feat * 4, num_feat * 4, 4, 2, 1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(num_feat * 4, affine=True)
        
        self.conv3_0 = nn.Conv2d(num_feat * 4, num_feat * 8, 3, 1, 1, bias=False)
        self.bn3_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
        self.conv3_1 = nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(num_feat * 8, affine=True)
        
        self.conv4_0 = nn.Conv2d(num_feat * 8, num_feat * 8, 3, 1, 1, bias=False)
        self.bn4_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
        self.conv4_1 = nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(num_feat * 8, affine=True)
        
        # Final layers
        self.linear1 = nn.Linear(num_feat * 8 * 4 * 4, 100)
        self.linear2 = nn.Linear(100, 1)
        
        # Activation
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        # Initialize weights
        self._initialize_weights()
        self._count_parameters()
        
    def _initialize_weights(self):
        """Initialize discriminator weights"""
        print("   🎲 Initializing discriminator weights...")
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)
                
    def _count_parameters(self):
        """Count discriminator parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        print(f"   📊 Discriminator parameters: {total_params:,}")
        
    def forward(self, x):
        """Forward pass"""
        feat = self.lrelu(self.conv0_0(x))
        feat = self.lrelu(self.bn0_1(self.conv0_1(feat)))
        
        feat = self.lrelu(self.bn1_0(self.conv1_0(feat)))
        feat = self.lrelu(self.bn1_1(self.conv1_1(feat)))
        
        feat = self.lrelu(self.bn2_0(self.conv2_0(feat)))
        feat = self.lrelu(self.bn2_1(self.conv2_1(feat)))
        
        feat = self.lrelu(self.bn3_0(self.conv3_0(feat)))
        feat = self.lrelu(self.bn3_1(self.conv3_1(feat)))
        
        feat = self.lrelu(self.bn4_0(self.conv4_0(feat)))
        feat = self.lrelu(self.bn4_1(self.conv4_1(feat)))
        
        # Global average pooling and fully connected
        feat = feat.view(feat.size(0), -1)
        feat = self.lrelu(self.linear1(feat))
        out = self.linear2(feat)
        
        return out

def create_models(config):
    """Create generator and discriminator models"""
    print("🏭 Creating models...")
    
    # Create generator
    generator = RealESRGAN(config)
    
    # Create discriminator
    discriminator = Discriminator(config.num_out_ch)
    
    # Move to device
    generator = generator.to(config.device)
    discriminator = discriminator.to(config.device)
    
    print(f"   ✅ Models created and moved to {config.device}")
    
    return generator, discriminator

def test_models():
    """Test model creation and forward pass"""
    print("🧪 Testing models...")
    
    try:
        generator, discriminator = create_models(config)
        
        # Test input
        batch_size = 2
        lr_size = config.crop_size // config.scale_factor
        hr_size = config.crop_size
        
        lr_input = torch.randn(batch_size, 3, lr_size, lr_size).to(config.device)
        hr_target = torch.randn(batch_size, 3, hr_size, hr_size).to(config.device)
        
        print(f"   🔍 Testing forward pass...")
        print(f"      LR input shape: {lr_input.shape}")
        print(f"      HR target shape: {hr_target.shape}")
        
        # Test generator
        with torch.no_grad():
            sr_output = generator(lr_input)
            print(f"      SR output shape: {sr_output.shape}")
            
            # Test discriminator
            disc_real = discriminator(hr_target)
            disc_fake = discriminator(sr_output)
            print(f"      Discriminator real: {disc_real.shape}")
            print(f"      Discriminator fake: {disc_fake.shape}")
            
        print("   ✅ Model test successful!")
        return generator, discriminator
        
    except Exception as e:
        print(f"   ❌ Model test failed: {str(e)}")
        return None, None

print("✅ Section 3 Complete: Network Architecture")
print("-"*80)

# ============================================================================
# SECTION 4: LOSS FUNCTIONS AND PERCEPTUAL LOSS
# ============================================================================

class VGGFeatureExtractor(nn.Module):
    """VGG19 feature extractor for perceptual loss"""
    
    def __init__(self, layer_name_list, use_input_norm=True, range_norm=False, requires_grad=False):
        super(VGGFeatureExtractor, self).__init__()
        print("🎨 Initializing VGG19 Feature Extractor for Perceptual Loss...")
        
        self.layer_name_list = layer_name_list
        self.use_input_norm = use_input_norm
        self.range_norm = range_norm
        
        print(f"   📋 Extracting features from layers: {layer_name_list}")
        print(f"   🔄 Input normalization: {use_input_norm}")
        print(f"   📏 Range normalization: {range_norm}")
        
        # Load pre-trained VGG19
        from torchvision.models import vgg19
        self.vgg = vgg19(pretrained=True)
        
        # Input normalization parameters (ImageNet stats)
        if self.use_input_norm:
            self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
            
        # Extract features from specified layers
        self.feature_layers = nn.ModuleDict()
        max_idx = 0
        
        # Map layer names to indices
        layer_mapping = {
            'conv1_1': 0, 'relu1_1': 1, 'conv1_2': 2, 'relu1_2': 3, 'pool1': 4,
            'conv2_1': 5, 'relu2_1': 6, 'conv2_2': 7, 'relu2_2': 8, 'pool2': 9,
            'conv3_1': 10, 'relu3_1': 11, 'conv3_2': 12, 'relu3_2': 13, 'conv3_3': 14, 'relu3_3': 15, 'conv3_4': 16, 'relu3_4': 17, 'pool3': 18,
            'conv4_1': 19, 'relu4_1': 20, 'conv4_2': 21, 'relu4_2': 22, 'conv4_3': 23, 'relu4_3': 24, 'conv4_4': 25, 'relu4_4': 26, 'pool4': 27,
            'conv5_1': 28, 'relu5_1': 29, 'conv5_2': 30, 'relu5_2': 31, 'conv5_3': 32, 'relu5_3': 33, 'conv5_4': 34, 'relu5_4': 35, 'pool5': 36
        }
        
        for layer_name in layer_name_list:
            if layer_name in layer_mapping:
                idx = layer_mapping[layer_name]
                max_idx = max(max_idx, idx)
                
        # Extract layers up to max_idx
        self.features = nn.Sequential(*list(self.vgg.features.children())[:max_idx + 1])
        
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = requires_grad
            
        print(f"   ✅ VGG19 feature extractor initialized")
        
    def normalize_inputs(self, x):
        """Normalize inputs to ImageNet statistics"""
        if self.range_norm:
            x = (x + 1) / 2  # Convert from [-1, 1] to [0, 1]
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        return x
    
    # ============================ CHANGE 2: Add custom_fwd decorator ============================
    # @custom_fwd(cast_inputs=torch.float32)
    def forward(self, x):
        """Extract VGG features"""
        if self.use_input_norm:
            x = self.normalize_inputs(x)
            
        features = {}
        layer_mapping = {
            0: 'conv1_1', 1: 'relu1_1', 2: 'conv1_2', 3: 'relu1_2', 4: 'pool1',
            5: 'conv2_1', 6: 'relu2_1', 7: 'conv2_2', 8: 'relu2_2', 9: 'pool2',
            10: 'conv3_1', 11: 'relu3_1', 12: 'conv3_2', 13: 'relu3_2', 14: 'conv3_3', 15: 'relu3_3', 16: 'conv3_4', 17: 'relu3_4', 18: 'pool3',
            19: 'conv4_1', 20: 'relu4_1', 21: 'conv4_2', 22: 'relu4_2', 23: 'conv4_3', 24: 'relu4_3', 25: 'conv4_4', 26: 'relu4_4', 27: 'pool4',
            28: 'conv5_1', 29: 'relu5_1', 30: 'conv5_2', 31: 'relu5_2', 32: 'conv5_3', 33: 'relu5_3', 34: 'conv5_4', 35: 'relu5_4', 36: 'pool5'
        }
        
        for idx, layer in enumerate(self.features):
            x = layer(x)
            layer_name = layer_mapping.get(idx)
            if layer_name in self.layer_name_list:
                features[layer_name] = x
                
        return features

class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG features"""
    
    def __init__(self, layer_weights={'relu3_4': 1.0, 'relu4_4': 1.0}, criterion='l1', device='cuda'):
        super(PerceptualLoss, self).__init__()
        print("🔍 Initializing Perceptual Loss...")
        
        self.layer_weights = layer_weights
        self.criterion_type = criterion
        self.device = device
        
        print(f"   📊 Layer weights: {layer_weights}")
        print(f"   📏 Criterion: {criterion}")
        print(f"   🎮 Device: {device}")
        
        # Initialize VGG feature extractor
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            use_input_norm=True,
            range_norm=False,
            requires_grad=False
        )
        
        # Move VGG to device
        self.vgg = self.vgg.to(device)
        
        # Loss criterion
        if criterion == 'l1':
            self.criterion = nn.L1Loss()
        elif criterion == 'l2':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported criterion: {criterion}")
            
        print("   ✅ Perceptual loss initialized")
        
    def forward(self, sr, hr):
        """Compute perceptual loss"""
        # VGG network is not autocast-safe, so we disable it for this block
        # and manually cast inputs to float32.
        with torch.cuda.amp.autocast(enabled=False):
            sr_32 = sr.float()
            hr_32 = hr.float()
            
            # Extract features
            sr_features = self.vgg(sr_32)
            hr_features = self.vgg(hr_32)
        
            # Compute loss for each layer (still in float32)
            perceptual_loss = 0
            for layer_name, weight in self.layer_weights.items():
                if layer_name in sr_features and layer_name in hr_features:
                    layer_loss = self.criterion(sr_features[layer_name], hr_features[layer_name])
                    perceptual_loss += weight * layer_loss
                
        return perceptual_loss

class PixelLoss(nn.Module):
    """Pixel-wise loss (L1 and L2)"""
    
    def __init__(self, loss_weight=1.0, reduction='mean', criterion='l1'):
        super(PixelLoss, self).__init__()
        print(f"📐 Initializing Pixel Loss - Weight: {loss_weight}, Criterion: {criterion}")
        
        self.loss_weight = loss_weight
        self.reduction = reduction
        
        if criterion == 'l1':
            self.criterion = nn.L1Loss(reduction=reduction)
        elif criterion == 'l2':
            self.criterion = nn.MSELoss(reduction=reduction)
        else:
            raise ValueError(f"Unsupported criterion: {criterion}")
            
        print("   ✅ Pixel loss initialized")
        
    def forward(self, sr, hr):
        """Compute pixel loss"""
        return self.loss_weight * self.criterion(sr, hr)

class AdversarialLoss(nn.Module):
    """Adversarial loss for GAN training"""
    
    def __init__(self, gan_type='vanilla', real_label_val=1.0, fake_label_val=0.0):
        super(AdversarialLoss, self).__init__()
        print(f"⚔️  Initializing Adversarial Loss - Type: {gan_type}")
        
        self.gan_type = gan_type
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val
        
        if gan_type == 'vanilla':
            self.criterion = nn.BCEWithLogitsLoss()
        elif gan_type == 'lsgan':
            self.criterion = nn.MSELoss()
        elif gan_type == 'wgan':
            self.criterion = None
        else:
            raise ValueError(f"Unsupported GAN type: {gan_type}")
            
        print("   ✅ Adversarial loss initialized")
        
    def get_target_tensor(self, prediction, target_is_real):
        """Create target tensor"""
        if target_is_real:
            target_tensor = self.real_label_val
        else:
            target_tensor = self.fake_label_val
            
        return torch.full_like(prediction, target_tensor, dtype=prediction.dtype, device=prediction.device)
        
    def forward(self, prediction, target_is_real):
        """Compute adversarial loss"""
        if self.gan_type == 'wgan':
            if target_is_real:
                return -prediction.mean()
            else:
                return prediction.mean()
        else:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            return self.criterion(prediction, target_tensor)

class CombinedLoss(nn.Module):
    """Combined loss function for Real-ESRGAN training"""
    
    def __init__(self, config):
        super(CombinedLoss, self).__init__()
        print("🎯 Initializing Combined Loss Function...")
        
        self.config = config
        
        # Initialize individual loss functions
        self.pixel_loss = PixelLoss(
            loss_weight=config.pixel_loss_weight,
            criterion='l1'
        )
        
        self.perceptual_loss = PerceptualLoss(
            layer_weights={'relu3_4': 1.0, 'relu4_4': 1.0},
            criterion='l1',
            device=config.device
        )
        
        self.adversarial_loss = AdversarialLoss(
            gan_type='vanilla'
        )
        
        print(f"   📊 Loss weights:")
        print(f"      Pixel loss: {config.pixel_loss_weight}")
        print(f"      Perceptual loss: {config.perceptual_loss_weight}")
        print(f"      Adversarial loss: {config.gan_loss_weight}")
        print("   ✅ Combined loss initialized")
        
    def forward(self, sr, hr, discriminator=None, mode='generator'):
        """Compute combined loss"""
        losses = {}
        total_loss = 0
        
        # Pixel loss
        pixel_loss = self.pixel_loss(sr, hr)
        losses['pixel'] = pixel_loss
        total_loss += pixel_loss
        
        # Perceptual loss
        perceptual_loss = self.perceptual_loss(sr, hr) * self.config.perceptual_loss_weight
        losses['perceptual'] = perceptual_loss
        total_loss += perceptual_loss
        
        # Adversarial loss (only during GAN training)
        if discriminator is not None and mode == 'generator':
            fake_pred = discriminator(sr)
            adversarial_loss = self.adversarial_loss(fake_pred, True) * self.config.gan_loss_weight
            losses['adversarial'] = adversarial_loss
            total_loss += adversarial_loss
        
        losses['total'] = total_loss
        return losses

def compute_discriminator_loss(discriminator, real_imgs, fake_imgs, adversarial_loss_fn):
    """Compute discriminator loss"""
    # Real images
    real_pred = discriminator(real_imgs)
    real_loss = adversarial_loss_fn(real_pred, True)
    
    # Fake images
    fake_pred = discriminator(fake_imgs.detach())
    fake_loss = adversarial_loss_fn(fake_pred, False)
    
    # Total discriminator loss
    disc_loss = (real_loss + fake_loss) * 0.5
    
    return disc_loss, real_pred, fake_pred

def test_loss_functions():
    """Test loss function implementations"""
    print("🧪 Testing loss functions...")
    
    try:
        # Create test tensors
        batch_size = 2
        channels = 3
        height, width = 128, 128
        
        sr = torch.randn(batch_size, channels, height, width).to(config.device)
        hr = torch.randn(batch_size, channels, height, width).to(config.device)
        
        print(f"   🔍 Test tensor shapes: {sr.shape}")
        
        # Test pixel loss
        pixel_loss_fn = PixelLoss()
        pixel_loss = pixel_loss_fn(sr, hr)
        print(f"   📐 Pixel loss: {pixel_loss.item():.6f}")
        
        # Test perceptual loss
        perceptual_loss_fn = PerceptualLoss(device=config.device)
        perceptual_loss = perceptual_loss_fn(sr, hr)
        print(f"   🔍 Perceptual loss: {perceptual_loss.item():.6f}")
        
        # Test adversarial loss
        adversarial_loss_fn = AdversarialLoss()
        fake_pred = torch.randn(batch_size, 1).to(config.device)
        adv_loss = adversarial_loss_fn(fake_pred, True)
        print(f"   ⚔️  Adversarial loss: {adv_loss.item():.6f}")
        
        # Test combined loss
        combined_loss_fn = CombinedLoss(config)
        losses = combined_loss_fn(sr, hr, mode='generator')
        print(f"   🎯 Combined losses:")
        for name, loss in losses.items():
            print(f"      {name}: {loss.item():.6f}")
            
        print("   ✅ Loss function test successful!")
        return combined_loss_fn
        
    except Exception as e:
        print(f"   ❌ Loss function test failed: {str(e)}")
        return None

print("✅ Section 4 Complete: Loss Functions")
print("-"*80)

# ... (The rest of the script remains unchanged) ...
# ...
# The final parts of the script will now work correctly as this change
# affects both the training and validation loops where the error occurred.
# ...

# ============================================================================
# SECTION 5: TRAINING LOOP AND OPTIMIZATION
# ============================================================================

class Trainer:
    """Main trainer class for Real-ESRGAN"""
    
    def __init__(self, config, generator, discriminator, train_loader, val_loader):
        print("🏋️  Initializing Real-ESRGAN Trainer...")
        
        self.config = config
        self.generator = generator
        self.discriminator = discriminator
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Initialize optimizers
        self._setup_optimizers()
        
        # Initialize loss functions
        self._setup_loss_functions()
        
        # Initialize learning rate schedulers
        self._setup_schedulers()
        
        # Initialize mixed precision training
        self.scaler_g = GradScaler() if config.use_mixed_precision else None
        self.scaler_d = GradScaler() if config.use_mixed_precision else None
        
        # Initialize logging
        self._setup_logging()
        
        # Training state
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_psnr = 0.0
        
        print("   ✅ Trainer initialized successfully")
        
    def _setup_optimizers(self):
        """Setup optimizers for generator and discriminator"""
        print("   ⚙️  Setting up optimizers...")
        
        # Generator optimizer
        self.optimizer_g = optim.Adam(
            self.generator.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.epsilon,
            weight_decay=self.config.weight_decay
        )
        
        # Discriminator optimizer
        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.epsilon,
            weight_decay=self.config.weight_decay
        )
        
        print(f"      Generator parameters: {sum(p.numel() for p in self.generator.parameters()):,}")
        print(f"      Discriminator parameters: {sum(p.numel() for p in self.discriminator.parameters()):,}")
        
    def _setup_loss_functions(self):
        """Setup loss functions"""
        print("   🎯 Setting up loss functions...")
        
        self.combined_loss = CombinedLoss(self.config)
        self.adversarial_loss = AdversarialLoss()
        
    def _setup_schedulers(self):
        """Setup learning rate schedulers"""
        print("   📈 Setting up learning rate schedulers...")
        
        if self.config.lr_scheduler_type == "cosine":
            self.scheduler_g = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_g, T_max=self.config.num_epochs
            )
            self.scheduler_d = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_d, T_max=self.config.num_epochs
            )
        elif self.config.lr_scheduler_type == "multistep":
            self.scheduler_g = optim.lr_scheduler.MultiStepLR(
                self.optimizer_g, 
                milestones=self.config.lr_decay_steps,
                gamma=self.config.lr_decay_factor
            )
            self.scheduler_d = optim.lr_scheduler.MultiStepLR(
                self.optimizer_d,
                milestones=self.config.lr_decay_steps,
                gamma=self.config.lr_decay_factor
            )
        else:
            self.scheduler_g = None
            self.scheduler_d = None
            
        print(f"      Scheduler type: {self.config.lr_scheduler_type}")
        
    def _setup_logging(self):
        """Setup tensorboard logging"""
        print("   📊 Setting up logging...")
        
        self.writer = SummaryWriter(log_dir=f"{self.config.save_dir}/logs")
        self.train_losses = {'pixel': [], 'perceptual': [], 'adversarial': [], 'total': []}
        self.val_metrics = {'psnr': [], 'ssim': []}
        
    def save_checkpoint(self, epoch, is_best=False):
        """Save training checkpoint"""
        print(f"   💾 Saving checkpoint for epoch {epoch}...")
        
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'best_psnr': self.best_psnr,
            'config': self.config.__dict__
        }
        
        if self.scheduler_g is not None:
            checkpoint['scheduler_g_state_dict'] = self.scheduler_g.state_dict()
            checkpoint['scheduler_d_state_dict'] = self.scheduler_d.state_dict()
            
        if self.scaler_g is not None:
            checkpoint['scaler_g_state_dict'] = self.scaler_g.state_dict()
            checkpoint['scaler_d_state_dict'] = self.scaler_d.state_dict()
            
        # Save regular checkpoint
        checkpoint_path = f"{self.config.save_dir}/checkpoints/checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path, _use_new_zipfile_serialization=False)
        
        # Save best checkpoint
        if is_best:
            best_path = f"{self.config.save_dir}/checkpoints/best_model.pth"
            torch.save(checkpoint, best_path, _use_new_zipfile_serialization=False)
            print(f"      🏆 New best model saved! PSNR: {self.best_psnr:.4f}")
            
        print(f"      ✅ Checkpoint saved: {checkpoint_path}")
        
    def find_latest_checkpoint(self, checkpoint_dir):
        """Find the latest checkpoint file in a directory"""
        if not os.path.isdir(checkpoint_dir):
            print(f"      ❌ Checkpoint directory not found: {checkpoint_dir}")
            return None
            
        # List available checkpoints first
        available_checkpoints = self.list_available_checkpoints(checkpoint_dir)
        
        # Look for best model first
        best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
        if os.path.exists(best_model_path):
            print(f"      🏆 Using best model: {best_model_path}")
            return best_model_path
            
        # Look for latest epoch checkpoint
        checkpoint_files = []
        for file in os.listdir(checkpoint_dir):
            if file.startswith("checkpoint_epoch_") and file.endswith(".pth"):
                try:
                    epoch_num = int(file.replace("checkpoint_epoch_", "").replace(".pth", ""))
                    checkpoint_files.append((epoch_num, os.path.join(checkpoint_dir, file)))
                except ValueError:
                    continue
                    
        if checkpoint_files:
            # Sort by epoch number and get the latest
            latest_epoch, latest_file = max(checkpoint_files, key=lambda x: x[0])
            print(f"      📊 Using latest epoch checkpoint: {latest_file} (epoch {latest_epoch})")
            return latest_file
            
        # If no epoch checkpoints, try to use any .pth file
        if available_checkpoints:
            latest_file = available_checkpoints[0]['path']  # Already sorted by modification time
            print(f"      🔄 Using most recent checkpoint: {latest_file}")
            return latest_file
            
        print(f"      ❌ No valid checkpoint files found in: {checkpoint_dir}")
        return None

    def list_available_checkpoints(self, checkpoint_dir):
        """List all available checkpoints in directory"""
        if not os.path.isdir(checkpoint_dir):
            print(f"      📂 Checkpoint directory not found: {checkpoint_dir}")
            return []
            
        print(f"      📂 Scanning checkpoint directory: {checkpoint_dir}")
        
        # List all .pth files
        checkpoint_files = []
        try:
            for file in os.listdir(checkpoint_dir):
                if file.endswith(".pth"):
                    full_path = os.path.join(checkpoint_dir, file)
                    file_size = os.path.getsize(full_path) / (1024 * 1024)  # MB
                    mod_time = datetime.fromtimestamp(os.path.getmtime(full_path))
                    
                    checkpoint_files.append({
                        'name': file,
                        'path': full_path,
                        'size_mb': file_size,
                        'modified': mod_time
                    })
                    
            # Sort by modification time (newest first)
            checkpoint_files.sort(key=lambda x: x['modified'], reverse=True)
            
            print(f"      📊 Found {len(checkpoint_files)} checkpoint files:")
            for i, ckpt in enumerate(checkpoint_files[:5]):  # Show first 5
                print(f"         {i+1}. {ckpt['name']} ({ckpt['size_mb']:.1f} MB, {ckpt['modified'].strftime('%Y-%m-%d %H:%M')})")
                
            if len(checkpoint_files) > 5:
                print(f"         ... and {len(checkpoint_files) - 5} more files")
                
        except Exception as e:
            print(f"      ❌ Error scanning directory: {str(e)}")
            
        return checkpoint_files

    def load_checkpoint(self, checkpoint_path):
        """Load training checkpoint"""
        print(f"   📁 Loading checkpoint from {checkpoint_path}...")
        
        actual_checkpoint_path = None
        if os.path.isfile(checkpoint_path):
            print(f"      📁 Checkpoint path is a file. Loading directly.")
            actual_checkpoint_path = checkpoint_path
        elif os.path.isdir(checkpoint_path):
            print(f"      📁 Checkpoint path is a directory. Searching for latest checkpoint...")
            actual_checkpoint_path = self.find_latest_checkpoint(checkpoint_path)
        else:
            print(f"      ❌ Checkpoint path does not exist or is not a file/directory: {checkpoint_path}")
            return False

        if actual_checkpoint_path is None:
            print(f"      ❌ No valid checkpoint found at path: {checkpoint_path}")
            return False
            
        try:
            print(f"      🔄 Loading checkpoint file: {actual_checkpoint_path}")
            checkpoint = torch.load(actual_checkpoint_path, map_location=self.config.device, weights_only=False)
            
            print(f"      📋 Checkpoint contents: {list(checkpoint.keys())}")
            
            # Load model states
            print(f"      🏗️  Loading generator state...")
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            
            print(f"      🕵️  Loading discriminator state...")
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            
            # Load optimizer states
            print(f"      ⚙️  Loading optimizer states...")
            self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
            self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
            
            # Load training state
            self.current_epoch = checkpoint['epoch']
            self.best_psnr = checkpoint.get('best_psnr', 0.0)
            
            # Load scheduler states if available
            if 'scheduler_g_state_dict' in checkpoint and self.scheduler_g is not None:
                print(f"      📈 Loading scheduler states...")
                self.scheduler_g.load_state_dict(checkpoint['scheduler_g_state_dict'])
                self.scheduler_d.load_state_dict(checkpoint['scheduler_d_state_dict'])
                
            # Load scaler states if available (for mixed precision)
            if 'scaler_g_state_dict' in checkpoint and self.scaler_g is not None:
                print(f"      🎯 Loading scaler states...")
                self.scaler_g.load_state_dict(checkpoint['scaler_g_state_dict'])
                self.scaler_d.load_state_dict(checkpoint['scaler_d_state_dict'])
                
            print(f"      ✅ Checkpoint loaded successfully")
            print(f"      📊 Resuming from epoch {self.current_epoch}")
            print(f"      🏆 Best PSNR: {self.best_psnr:.4f}")
            
            # Display some training info if available
            if 'config' in checkpoint:
                saved_config = checkpoint['config']
                print(f"      🔧 Original scale factor: {saved_config.get('scale_factor', 'unknown')}")
                print(f"      🔧 Original batch size: {saved_config.get('batch_size', 'unknown')}")
                
            return True
            
        except Exception as e:
            print(f"      ❌ Failed to load checkpoint: {str(e)}")
            print(f"      💡 Hint: If you're using PyTorch 2.6+, this might be a weights_only issue")
            return False
            
    def train_epoch(self, epoch):
        """Train for one epoch"""
        print(f"🚀 Training Epoch {epoch}/{self.config.num_epochs}")
        
        self.generator.train()
        self.discriminator.train()
        
        epoch_losses = {'pixel': 0, 'perceptual': 0, 'adversarial': 0, 'discriminator': 0, 'total': 0}
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            lr_imgs = batch['lr'].to(self.config.device)
            hr_imgs = batch['hr'].to(self.config.device)
            
            batch_size = lr_imgs.size(0)
            
            # =====================================
            # Train Generator
            # =====================================
            self.optimizer_g.zero_grad()
            
            if self.config.use_mixed_precision:
                with autocast():
                    sr_imgs = self.generator(lr_imgs)
                    g_losses = self.combined_loss(
                        sr_imgs, hr_imgs, 
                        discriminator=self.discriminator,
                        mode='generator'
                    )
                    
                self.scaler_g.scale(g_losses['total']).backward()
                self.scaler_g.step(self.optimizer_g)
                self.scaler_g.update()
            else:
                sr_imgs = self.generator(lr_imgs)
                g_losses = self.combined_loss(
                    sr_imgs, hr_imgs,
                    discriminator=self.discriminator,
                    mode='generator'
                )
                g_losses['total'].backward()
                self.optimizer_g.step()
                
            # =====================================
            # Train Discriminator
            # =====================================
            self.optimizer_d.zero_grad()
            
            if self.config.use_mixed_precision:
                with autocast():
                    d_loss, _, _ = compute_discriminator_loss(
                        self.discriminator, hr_imgs, sr_imgs, self.adversarial_loss
                    )
                    
                self.scaler_d.scale(d_loss).backward()
                self.scaler_d.step(self.optimizer_d)
                self.scaler_d.update()
            else:
                d_loss, _, _ = compute_discriminator_loss(
                    self.discriminator, hr_imgs, sr_imgs, self.adversarial_loss
                )
                d_loss.backward()
                self.optimizer_d.step()
                
            # =====================================
            # Update metrics
            # =====================================
            epoch_losses['pixel'] += g_losses['pixel'].item()
            epoch_losses['perceptual'] += g_losses['perceptual'].item()
            epoch_losses['adversarial'] += g_losses.get('adversarial', torch.tensor(0)).item()
            epoch_losses['discriminator'] += d_loss.item()
            epoch_losses['total'] += g_losses['total'].item()
            
            # =====================================
            # Logging and visualization
            # =====================================
            if batch_idx % self.config.log_freq == 0:
                self._log_training_progress(epoch, batch_idx, g_losses, d_loss)
                
            if batch_idx % self.config.save_images_freq == 0:
                self._save_sample_images(epoch, batch_idx, lr_imgs, hr_imgs, sr_imgs)
                
            # Update progress bar
            progress_bar.set_postfix({
                'G_loss': f"{g_losses['total'].item():.4f}",
                'D_loss': f"{d_loss.item():.4f}",
                'Pixel': f"{g_losses['pixel'].item():.4f}",
                'Perceptual': f"{g_losses['perceptual'].item():.4f}"
            })
            
            self.current_iteration += 1
            
        # Average losses for epoch
        num_batches = len(self.train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
            
        return epoch_losses
        
    def _log_training_progress(self, epoch, batch_idx, g_losses, d_loss):
        """Log training progress to tensorboard"""
        iteration = epoch * len(self.train_loader) + batch_idx
        
        # Log generator losses
        self.writer.add_scalar('Train/Generator/Total', g_losses['total'].item(), iteration)
        self.writer.add_scalar('Train/Generator/Pixel', g_losses['pixel'].item(), iteration)
        self.writer.add_scalar('Train/Generator/Perceptual', g_losses['perceptual'].item(), iteration)
        
        if 'adversarial' in g_losses:
            self.writer.add_scalar('Train/Generator/Adversarial', g_losses['adversarial'].item(), iteration)
            
        # Log discriminator loss
        self.writer.add_scalar('Train/Discriminator/Loss', d_loss.item(), iteration)
        
        # Log learning rates
        self.writer.add_scalar('Train/LR/Generator', self.optimizer_g.param_groups[0]['lr'], iteration)
        self.writer.add_scalar('Train/LR/Discriminator', self.optimizer_d.param_groups[0]['lr'], iteration)
        
    def _save_sample_images(self, epoch, batch_idx, lr_imgs, hr_imgs, sr_imgs):
        """Save sample images during training"""
        with torch.no_grad():
            # Take first image from batch
            lr_sample = lr_imgs[0:1]
            hr_sample = hr_imgs[0:1]
            sr_sample = sr_imgs[0:1]
            
            # Clamp to valid range
            lr_sample = torch.clamp(lr_sample, 0, 1)
            hr_sample = torch.clamp(hr_sample, 0, 1)
            sr_sample = torch.clamp(sr_sample, 0, 1)
            
            # Save images
            save_path = f"{self.config.save_dir}/samples"
            save_image(lr_sample, f"{save_path}/epoch_{epoch}_batch_{batch_idx}_lr.png")
            save_image(hr_sample, f"{save_path}/epoch_{epoch}_batch_{batch_idx}_hr.png")
            save_image(sr_sample, f"{save_path}/epoch_{epoch}_batch_{batch_idx}_sr.png")
            
            # Save comparison grid - resize LR to match SR size
            lr_upsampled = F.interpolate(lr_sample, size=sr_sample.shape[-2:], mode='bicubic', align_corners=False)
            comparison = torch.cat([lr_upsampled, sr_sample, hr_sample], dim=3)
            save_image(comparison, f"{save_path}/epoch_{epoch}_batch_{batch_idx}_comparison.png")
            
    def train(self):
        """Main training loop"""
        print("🎯 Starting Real-ESRGAN Training")
        print("="*80)
        
        # Resume from checkpoint if specified
        if self.config.resume_training and self.config.checkpoint_path:
            print(f"🔄 Attempting to resume training from: {self.config.checkpoint_path}")
            checkpoint_loaded = self.load_checkpoint(self.config.checkpoint_path)
            if checkpoint_loaded:
                start_epoch = self.current_epoch + 1
                print(f"✅ Successfully resumed training from epoch {self.current_epoch}")
            else:
                print("⚠️  Failed to load checkpoint, starting from scratch")
                start_epoch = 1
        else:
            print("🏁 Starting training from scratch")
            start_epoch = 1
            
        for epoch in range(start_epoch, self.config.num_epochs + 1):
            self.current_epoch = epoch
            
            # Train one epoch
            train_losses = self.train_epoch(epoch)
            
            # Update learning rate
            if self.scheduler_g is not None:
                self.scheduler_g.step()
                self.scheduler_d.step()
                
            # Validation
            if epoch % self.config.validation_freq == 0:
                val_metrics = self.validate(epoch)
                
                # Check if best model
                current_psnr = val_metrics.get('psnr', 0)
                is_best = current_psnr > self.best_psnr
                if is_best:
                    self.best_psnr = current_psnr
                    
            else:
                is_best = False
                
            # Save checkpoint
            if epoch % self.config.checkpoint_freq == 0:
                self.save_checkpoint(epoch, is_best)
                
            # Log epoch summary
            print(f"📊 Epoch {epoch} Summary:")
            print(f"   Generator Loss: {train_losses['total']:.6f}")
            print(f"   Pixel Loss: {train_losses['pixel']:.6f}")
            print(f"   Perceptual Loss: {train_losses['perceptual']:.6f}")
            print(f"   Adversarial Loss: {train_losses['adversarial']:.6f}")
            print(f"   Discriminator Loss: {train_losses['discriminator']:.6f}")
            print("-"*60)
            
        print("✅ Training completed successfully!")
        self.writer.close()

def create_trainer(config):
    """Create trainer instance"""
    print("🏭 Creating trainer...")
    
    try:
        # Create data loaders
        train_loader, val_loader = create_data_loaders(config)
        
        # Create models
        generator, discriminator = create_models(config)
        
        # Create trainer
        trainer = Trainer(config, generator, discriminator, train_loader, val_loader)
        
        print("   ✅ Trainer created successfully")
        return trainer
        
    except Exception as e:
        print(f"   ❌ Failed to create trainer: {str(e)}")
        return None

print("✅ Section 5 Complete: Training Loop")
print("-"*80)

# ============================================================================
# SECTION 6: VALIDATION AND EVALUATION
# ============================================================================

def calculate_psnr(img1, img2, max_value=1.0):
    """Calculate Peak Signal-to-Noise Ratio (PSNR)"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    
    psnr = 20 * torch.log10(max_value / torch.sqrt(mse))
    return psnr.item()

def calculate_ssim(img1, img2, window_size=11, size_average=True):
    """Calculate Structural Similarity Index (SSIM)"""
    def gaussian_window(window_size, sigma=1.5):
        """Create gaussian window"""
        coords = torch.arange(window_size, dtype=torch.float32)
        coords -= window_size // 2
        
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        
        return g.unsqueeze(1) * g.unsqueeze(0)
    
    def ssim_single(img1, img2, window, window_size, size_average=True):
        """Calculate SSIM for single image pair"""
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=img1.size(1))
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=img2.size(1))
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=img1.size(1)) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=img2.size(1)) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=img1.size(1)) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
    
    # Ensure images are in the same device
    img1 = img1.to(img2.device)
    
    # Create gaussian window
    window = gaussian_window(window_size).to(img1.device)
    window = window.unsqueeze(0).unsqueeze(0)
    window = window.expand(img1.size(1), 1, window_size, window_size)
    
    return ssim_single(img1, img2, window, window_size, size_average).item()

class ImageQualityMetrics:
    """Calculate various image quality metrics"""
    
    def __init__(self):
        print("📊 Initializing Image Quality Metrics...")
        self.metrics = {}
        
    def calculate_metrics(self, sr_imgs, hr_imgs):
        """Calculate all metrics for a batch of images"""
        batch_size = sr_imgs.size(0)
        
        psnr_values = []
        ssim_values = []
        
        for i in range(batch_size):
            sr_img = sr_imgs[i:i+1]
            hr_img = hr_imgs[i:i+1]
            
            # Clamp images to valid range
            sr_img = torch.clamp(sr_img, 0, 1)
            hr_img = torch.clamp(hr_img, 0, 1)
            
            # Calculate PSNR
            psnr = calculate_psnr(sr_img, hr_img)
            psnr_values.append(psnr)
            
            # Calculate SSIM
            ssim = calculate_ssim(sr_img, hr_img)
            ssim_values.append(ssim)
            
        metrics = {
            'psnr': np.mean(psnr_values),
            'ssim': np.mean(ssim_values),
            'psnr_std': np.std(psnr_values),
            'ssim_std': np.std(ssim_values)
        }
        
        return metrics

# Add validation method to Trainer class
def add_validation_to_trainer():
    """Add validation method to Trainer class"""
    
    def validate(self, epoch):
        """Validate the model"""
        print(f"🔍 Validating Epoch {epoch}...")
        
        self.generator.eval()
        self.discriminator.eval()
        
        val_losses = {'pixel': 0, 'perceptual': 0, 'total': 0}
        metrics_calculator = ImageQualityMetrics()
        
        all_psnr = []
        all_ssim = []
        
        progress_bar = tqdm(self.val_loader, desc="Validation")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                lr_imgs = batch['lr'].to(self.config.device)
                hr_imgs = batch['hr'].to(self.config.device)
                
                # Generate super-resolution images (without autocast for validation)
                sr_imgs = self.generator(lr_imgs)
                
                # Calculate validation loss (without adversarial component and without autocast)
                v_losses = self.combined_loss(sr_imgs, hr_imgs, mode='validation')
                
                # Accumulate losses
                val_losses['pixel'] += v_losses['pixel'].item()
                val_losses['perceptual'] += v_losses['perceptual'].item()
                val_losses['total'] += v_losses['total'].item()
                
                # Calculate metrics
                metrics = metrics_calculator.calculate_metrics(sr_imgs, hr_imgs)
                all_psnr.append(metrics['psnr'])
                all_ssim.append(metrics['ssim'])
                
                # Update progress bar
                progress_bar.set_postfix({
                    'PSNR': f"{metrics['psnr']:.2f}",
                    'SSIM': f"{metrics['ssim']:.4f}",
                    'Loss': f"{v_losses['total'].item():.4f}"
                })
                
                # Save validation images for first few batches
                if batch_idx < 5 and self.config.save_validation_images:
                    self._save_validation_images(epoch, batch_idx, lr_imgs, hr_imgs, sr_imgs)
                    
        # Average metrics
        num_batches = len(self.val_loader)
        for key in val_losses:
            val_losses[key] /= num_batches
            
        avg_psnr = np.mean(all_psnr)
        avg_ssim = np.mean(all_ssim)
        std_psnr = np.std(all_psnr)
        std_ssim = np.std(all_ssim)
        
        # Log validation metrics
        self.writer.add_scalar('Validation/Loss/Total', val_losses['total'], epoch)
        self.writer.add_scalar('Validation/Loss/Pixel', val_losses['pixel'], epoch)
        self.writer.add_scalar('Validation/Loss/Perceptual', val_losses['perceptual'], epoch)
        self.writer.add_scalar('Validation/Metrics/PSNR', avg_psnr, epoch)
        self.writer.add_scalar('Validation/Metrics/SSIM', avg_ssim, epoch)
        
        print(f"   📊 Validation Results:")
        print(f"      PSNR: {avg_psnr:.4f} ± {std_psnr:.4f}")
        print(f"      SSIM: {avg_ssim:.6f} ± {std_ssim:.6f}")
        print(f"      Total Loss: {val_losses['total']:.6f}")
        print(f"      Pixel Loss: {val_losses['pixel']:.6f}")
        print(f"      Perceptual Loss: {val_losses['perceptual']:.6f}")
        
        return {
            'psnr': avg_psnr,
            'ssim': avg_ssim,
            'psnr__std': std_psnr,
            'ssim_std': std_ssim,
            'losses': val_losses
        }
    
    def _save_validation_images(self, epoch, batch_idx, lr_imgs, hr_imgs, sr_imgs):
        """Save validation images"""
        # Take first image from batch
        lr_sample = lr_imgs[0:1]
        hr_sample = hr_imgs[0:1] 
        sr_sample = sr_imgs[0:1]
        
        # Clamp to valid range
        lr_sample = torch.clamp(lr_sample, 0, 1)
        hr_sample = torch.clamp(hr_sample, 0, 1)
        sr_sample = torch.clamp(sr_sample, 0, 1)
        
        # Calculate metrics for this sample
        psnr = calculate_psnr(sr_sample, hr_sample)
        ssim = calculate_ssim(sr_sample, hr_sample)
        
        # Save images
        save_path = f"{self.config.save_dir}/validation"
        save_image(lr_sample, f"{save_path}/epoch_{epoch}_val_{batch_idx}_lr.png")
        save_image(hr_sample, f"{save_path}/epoch_{epoch}_val_{batch_idx}_hr.png")
        save_image(sr_sample, f"{save_path}/epoch_{epoch}_val_{batch_idx}_sr_psnr{psnr:.2f}_ssim{ssim:.4f}.png")
        
        # Create comparison grid - resize LR to match SR size
        lr_upsampled = F.interpolate(lr_sample, size=sr_sample.shape[-2:], mode='bicubic', align_corners=False)
        comparison = torch.cat([lr_upsampled, sr_sample, hr_sample], dim=3)
        save_image(comparison, f"{save_path}/epoch_{epoch}_val_{batch_idx}_comparison.png")
    
    # Add methods to Trainer class
    Trainer.validate = validate
    Trainer._save_validation_images = _save_validation_images

# Execute the function to add validation methods
add_validation_to_trainer()

class Evaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self, config, generator):
        print("🔍 Initializing Model Evaluator...")
        
        self.config = config
        self.generator = generator
        self.metrics_calculator = ImageQualityMetrics()
        
        print("   ✅ Evaluator initialized")
        
    def evaluate_dataset(self, data_loader, save_results=True):
        """Evaluate model on entire dataset"""
        print("📊 Evaluating model on dataset...")
        
        self.generator.eval()
        
        all_metrics = []
        total_time = 0
        
        results_dir = f"{self.config.save_dir}/evaluation_results"
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        
        progress_bar = tqdm(data_loader, desc="Evaluation")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                lr_imgs = batch['lr'].to(self.config.device)
                hr_imgs = batch['hr'].to(self.config.device)
                
                # Measure inference time
                start_time = time.time()
                
                if self.config.use_mixed_precision:
                    with autocast():
                        sr_imgs = self.generator(lr_imgs)
                else:
                    sr_imgs = self.generator(lr_imgs)
                    
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                
                batch_time = end_time - start_time
                total_time += batch_time
                
                # Calculate metrics
                metrics = self.metrics_calculator.calculate_metrics(sr_imgs, hr_imgs)
                metrics['inference_time'] = batch_time / lr_imgs.size(0)  # Per image
                all_metrics.append(metrics)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'PSNR': f"{metrics['psnr']:.2f}",
                    'SSIM': f"{metrics['ssim']:.4f}",
                    'Time': f"{metrics['inference_time']*1000:.1f}ms"
                })
                
                # Save sample results
                if save_results and batch_idx < 10:
                    self._save_evaluation_results(batch_idx, lr_imgs, hr_imgs, sr_imgs, metrics, results_dir)
                    
        # Aggregate results
        aggregate_metrics = self._aggregate_metrics(all_metrics)
        aggregate_metrics['total_inference_time'] = total_time
        aggregate_metrics['avg_inference_time_per_image'] = total_time / len(data_loader.dataset)
        
        # Save detailed results
        if save_results:
            self._save_detailed_results(aggregate_metrics, all_metrics, results_dir)
            
        return aggregate_metrics
    
    def _aggregate_metrics(self, all_metrics):
        """Aggregate metrics across all batches"""
        psnr_values = [m['psnr'] for m in all_metrics]
        ssim_values = [m['ssim'] for m in all_metrics]
        time_values = [m['inference_time'] for m in all_metrics]
        
        aggregate = {
            'psnr_mean': np.mean(psnr_values),
            'psnr_std': np.std(psnr_values),
            'psnr_min': np.min(psnr_values),
            'psnr_max': np.max(psnr_values),
            'ssim_mean': np.mean(ssim_values),
            'ssim_std': np.std(ssim_values),
            'ssim_min': np.min(ssim_values),
            'ssim_max': np.max(ssim_values),
            'inference_time_mean': np.mean(time_values),
            'inference_time_std': np.std(time_values),
            'num_samples': len(all_metrics)
        }
        
        return aggregate
    
    def _save_evaluation_results(self, batch_idx, lr_imgs, hr_imgs, sr_imgs, metrics, results_dir):
        """Save evaluation results for a batch"""
        # Take first image from batch
        lr_sample = torch.clamp(lr_imgs[0:1], 0, 1)
        hr_sample = torch.clamp(hr_imgs[0:1], 0, 1)
        sr_sample = torch.clamp(sr_imgs[0:1], 0, 1)
        
        # Save individual images
        save_image(lr_sample, f"{results_dir}/sample_{batch_idx:03d}_lr.png")
        save_image(hr_sample, f"{results_dir}/sample_{batch_idx:03d}_hr.png")
        save_image(sr_sample, f"{results_dir}/sample_{batch_idx:03d}_sr.png")
        
        # Create comparison - resize LR to match SR size
        lr_upsampled = F.interpolate(lr_sample, size=sr_sample.shape[-2:], mode='bicubic', align_corners=False)
        comparison = torch.cat([lr_upsampled, sr_sample, hr_sample], dim=3)
        save_image(comparison, f"{results_dir}/sample_{batch_idx:03d}_comparison.png")
        
    def _save_detailed_results(self, aggregate_metrics, all_metrics, results_dir):
        """Save detailed evaluation results"""
        # Save aggregate metrics
        with open(f"{results_dir}/aggregate_metrics.json", 'w') as f:
            json.dump(aggregate_metrics, f, indent=4)
            
        # Save per-sample metrics
        with open(f"{results_dir}/per_sample_metrics.json", 'w') as f:
            json.dump(all_metrics, f, indent=4)
            
        # Create summary report
        report = f"""
Real-ESRGAN Evaluation Report
=============================

Dataset: {self.config.dataset_path}
Scale Factor: {self.config.scale_factor}x
Number of Samples: {aggregate_metrics['num_samples']}

Performance Metrics:
- PSNR: {aggregate_metrics['psnr_mean']:.4f} ± {aggregate_metrics['psnr_std']:.4f} dB
- SSIM: {aggregate_metrics['ssim_mean']:.6f} ± {aggregate_metrics['ssim_std']:.6f}

Inference Performance:
- Average Time per Image: {aggregate_metrics['inference_time_mean']*1000:.2f} ± {aggregate_metrics['inference_time_std']*1000:.2f} ms
- Total Evaluation Time: {aggregate_metrics['total_inference_time']:.2f} seconds

Range Statistics:
- PSNR Range: [{aggregate_metrics['psnr_min']:.2f}, {aggregate_metrics['psnr_max']:.2f}] dB
- SSIM Range: [{aggregate_metrics['ssim_min']:.4f}, {aggregate_metrics['ssim_max']:.4f}]
"""
        
        with open(f"{results_dir}/evaluation_report.txt", 'w') as f:
            f.write(report)
            
        print("📊 Evaluation Results:")
        print(report)

def test_evaluation():
    """Test evaluation functionality"""
    print("🧪 Testing evaluation...")
    
    try:
        # Create test data
        batch_size = 4
        channels = 3
        lr_size = 32
        hr_size = 128
        
        sr_imgs = torch.rand(batch_size, channels, hr_size, hr_size)
        hr_imgs = torch.rand(batch_size, channels, hr_size, hr_size)
        
        # Test metrics calculation
        metrics_calc = ImageQualityMetrics()
        metrics = metrics_calc.calculate_metrics(sr_imgs, hr_imgs)
        
        print(f"   📊 Test metrics:")
        print(f"      PSNR: {metrics['psnr']:.4f}")
        print(f"      SSIM: {metrics['ssim']:.6f}")
        
        print("   ✅ Evaluation test successful!")
        return True
        
    except Exception as e:
        print(f"   ❌ Evaluation test failed: {str(e)}")
        return False

print("✅ Section 6 Complete: Validation and Evaluation")
print("-"*80) 

# ============================================================================
# SECTION 7: MAIN EXECUTION AND INFERENCE
# ============================================================================

class InferenceEngine:
    """Inference engine for Real-ESRGAN"""
    
    def __init__(self, config, model_path=None):
        print("🚀 Initializing Real-ESRGAN Inference Engine...")
        
        self.config = config
        self.model_path = model_path
        
        # Create generator model
        self.generator = RealESRGAN(config).to(config.device)
        
        # Load trained model if path provided
        if model_path:
            self.load_model(model_path)
        else:
            print("   ⚠️  No model path provided, using random weights")
            
        self.generator.eval()
        print("   ✅ Inference engine initialized")
        
    def find_model_checkpoint(self, model_path):
        """Find the best model checkpoint to load"""
        if os.path.isfile(model_path):
            return model_path
            
        if os.path.isdir(model_path):
            # Look for best model first
            best_model_path = os.path.join(model_path, "best_model.pth")
            if os.path.exists(best_model_path):
                print(f"      🏆 Found best model: {best_model_path}")
                return best_model_path
                
            # Look for latest epoch checkpoint
            checkpoint_files = []
            for file in os.listdir(model_path):
                if file.startswith("checkpoint_epoch_") and file.endswith(".pth"):
                    try:
                        epoch_num = int(file.replace("checkpoint_epoch_", "").replace(".pth", ""))
                        checkpoint_files.append((epoch_num, os.path.join(model_path, file)))
                    except ValueError:
                        continue
                        
            if checkpoint_files:
                latest_epoch, latest_file = max(checkpoint_files, key=lambda x: x[0])
                print(f"      📊 Found latest checkpoint: {latest_file} (epoch {latest_epoch})")
                return latest_file
                
        return None

    def load_model(self, model_path):
        """Load trained model weights"""
        print(f"   📁 Loading model from: {model_path}")
        
        # Find the actual model file
        actual_model_path = self.find_model_checkpoint(model_path)
        if actual_model_path is None:
            print(f"      ❌ No valid model checkpoint found")
            return
            
        try:
            checkpoint = torch.load(actual_model_path, map_location=self.config.device, weights_only=False)
            
            if 'generator_state_dict' in checkpoint:
                self.generator.load_state_dict(checkpoint['generator_state_dict'])
                print(f"      ✅ Model loaded successfully")
                if 'best_psnr' in checkpoint:
                    print(f"      🏆 Model PSNR: {checkpoint['best_psnr']:.4f}")
                if 'epoch' in checkpoint:
                    print(f"      📊 Model epoch: {checkpoint['epoch']}")
            else:
                # Direct state dict
                self.generator.load_state_dict(checkpoint)
                print(f"      ✅ Model state dict loaded")
                
        except Exception as e:
            print(f"      ❌ Failed to load model: {str(e)}")
            
    def inference_single_image(self, lr_image_path, output_path=None, save_comparison=True):
        """Perform inference on a single image"""
        print(f"🔍 Processing image: {lr_image_path}")
        
        try:
            # Load and preprocess image
            lr_img = self._load_and_preprocess_image(lr_image_path)
            
            # Perform inference
            with torch.no_grad():
                if self.config.use_mixed_precision:
                    with autocast():
                        sr_img = self.generator(lr_img)
                else:
                    sr_img = self.generator(lr_img)
                    
            # Post-process result
            sr_img = torch.clamp(sr_img, 0, 1)
            
            # Save result
            if output_path is None:
                output_path = lr_image_path.replace('.', f'_sr_{self.config.scale_factor}x.')
                
            self._save_image(sr_img, output_path)
            
            # Save comparison if requested
            if save_comparison:
                comparison_path = output_path.replace('.', '_comparison.')
                self._save_comparison(lr_img, sr_img, comparison_path)
                
            print(f"   ✅ Result saved to: {output_path}")
            return sr_img, output_path
            
        except Exception as e:
            print(f"   ❌ Inference failed: {str(e)}")
            return None, None
            
    def inference_batch(self, input_dir, output_dir, file_extensions=('.png', '.jpg', '.jpeg')):
        """Perform inference on all images in a directory"""
        print(f"🔍 Processing directory: {input_dir}")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Find all images
        image_files = []
        for ext in file_extensions:
            image_files.extend(Path(input_dir).glob(f"*{ext}"))
            image_files.extend(Path(input_dir).glob(f"*{ext.upper()}"))
            
        print(f"   📊 Found {len(image_files)} images")
        
        results = []
        progress_bar = tqdm(image_files, desc="Processing images")
        
        for image_path in progress_bar:
            output_path = Path(output_dir) / f"{image_path.stem}_sr_{self.config.scale_factor}x{image_path.suffix}"
            
            sr_img, result_path = self.inference_single_image(
                str(image_path), 
                str(output_path),
                save_comparison=True
            )
            
            if sr_img is not None:
                results.append((str(image_path), result_path))
                progress_bar.set_postfix({'processed': len(results)})
                
        print(f"   ✅ Processed {len(results)}/{len(image_files)} images")
        return results
        
    def _load_and_preprocess_image(self, image_path):
        """Load and preprocess image for inference"""
        try:
            # Load image
            img = Image.open(image_path).convert('RGB')
            img = np.array(img)
            
            # Convert to float and normalize
            img = img.astype(np.float32) / 255.0
            
            # Convert to tensor and add batch dimension
            img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
            img = img.to(self.config.device)
            
            return img
            
        except Exception as e:
            print(f"   ❌ Failed to load image {image_path}: {str(e)}")
            raise
            
    def _save_image(self, tensor, output_path):
        """Save tensor as image"""
        try:
            save_image(tensor, output_path)
        except Exception as e:
            print(f"   ❌ Failed to save image {output_path}: {str(e)}")
            raise
            
    def _save_comparison(self, lr_img, sr_img, comparison_path):
        """Save LR-SR comparison"""
        try:
            # Resize LR to match SR for comparison
            lr_resized = F.interpolate(lr_img, size=sr_img.shape[-2:], mode='bicubic', align_corners=False)
            
            # Create comparison
            comparison = torch.cat([lr_resized, sr_img], dim=3)
            save_image(comparison, comparison_path)
            
        except Exception as e:
            print(f"   ⚠️  Failed to save comparison: {str(e)}")

def run_comprehensive_test():
    """Run comprehensive test of all components"""
    print("🧪 Running Comprehensive Test Suite")
    print("="*80)
    
    # Test configuration
    print("📋 Testing configuration...")
    assert config.scale_factor in [2, 3, 4], "Invalid scale factor"
    print("   ✅ Configuration test passed")
    
    # Test data loading (if dataset exists)
    print("📚 Testing data loading...")
    try:
        # Only test if DF2K exists
        if os.path.exists(config.dataset_path):
            train_loader, val_loader = test_data_loading()
            if train_loader is not None:
                print("   ✅ Data loading test passed")
            else:
                print("   ⚠️  Data loading test skipped (no dataset)")
        else:
            print("   ⚠️  Data loading test skipped (no dataset found)")
    except Exception as e:
        print(f"   ⚠️  Data loading test failed: {str(e)}")
    
    # Test model creation
    print("🏭 Testing model creation...")
    generator, discriminator = test_models()
    if generator is not None:
        print("   ✅ Model creation test passed")
    else:
        print("   ❌ Model creation test failed")
        return False
    
    # Test loss functions
    print("🎯 Testing loss functions...")
    loss_fn = test_loss_functions()
    if loss_fn is not None:
        print("   ✅ Loss function test passed")
    else:
        print("   ❌ Loss function test failed")
        return False
    
    # Test evaluation
    print("📊 Testing evaluation...")
    eval_success = test_evaluation()
    if eval_success:
        print("   ✅ Evaluation test passed")
    else:
        print("   ❌ Evaluation test failed")
        return False
    
    print("🎉 All tests passed successfully!")
    return True

def main():
    """Main execution function"""
    print("🎯 Real-ESRGAN Main Execution")
    print("="*80)
    
    # Print system information
    print(f"💻 System Information:")
    print(f"   Python: {sys.version}")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("-"*60)
    
    # Run comprehensive tests
    test_success = run_comprehensive_test()
    
    if not test_success:
        print("❌ Tests failed. Please fix issues before training.")
        return
    
    print("\n" + "="*80)
    print("🚀 TRAINING OPTIONS")
    print("="*80)
    print("1. 🏋️  Start training from scratch")
    print("2. 🔄 Resume training from checkpoint")
    print("3. 🔍 Run inference on images")
    print("4. 📊 Evaluate trained model")
    print("5. 🧪 Run tests only")
    print("6. ❌ Exit")
    print("-"*60)
    
    try:
        print("🏋️  Starting training from scratch...")
        trainer = create_trainer(config)
        if trainer:
            trainer.train()
        else:
            print("❌ Failed to create trainer")
 
            
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")

if __name__ == "__main__":
    print("🚀 Real-ESRGAN Single Image Super-Resolution")
    print("   Modern PyTorch Implementation with DF2K Dataset")
    print("   Comprehensive training, validation, and inference pipeline")
    print("="*80)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    # Run main function
    main()

print("✅ Section 7 Complete: Main Execution and Inference")
print("="*80)
print("🎉 REAL-ESRGAN IMPLEMENTATION COMPLETE!")
print("   All sections successfully implemented:")
print("   1. ✅ Imports and Configuration")
print("   2. ✅ Dataset Handling")
print("   3. ✅ Network Architecture")
print("   4. ✅ Loss Functions")
print("   5. ✅ Training Loop")
print("   6. ✅ Validation and Evaluation")
print("   7. ✅ Main Execution and Inference")
print("="*80)