# =================================================================================================
# SECTION 1: IMPORTS AND INITIAL SETUP
# =================================================================================================
# --- Standard Library Imports ---
import os
import glob
import random
import time

# --- Third-party Library Imports ---
import numpy as np
from PIL import Image, ImageFilter
import torch
import torch.nn as nn
import torch.nn.functional as F  # <<< MODIFICATION: ADDED FOR INTERPOLATION
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models import vgg19
import torchvision.utils as vutils
from tqdm import tqdm
import cv2
import math
from scipy import ndimage
from scipy.ndimage import gaussian_filter

print("✅ SECTION 1: IMPORTS AND INITIAL SETUP COMPLETE")
print("-" * 80)

# =================================================================================================
# SECTION 2: CONFIGURATION
# =================================================================================================
# This class holds all hyperparameters and configuration settings in one place.
# This makes it easy to tune the model and training process.

class Config:
    # --- Dataset and Paths ---
    # !!! IMPORTANT: Change this to the root directory of your DF2K dataset !!!
    DATASET_PATH = "/kaggle/input/df2kdata"
    OUTPUT_DIR = "training_outputs" # Directory to save checkpoints and images
    SCALE_FACTOR = 4  # The super-resolution scale (e.g., 4x)
    # We will use bicubic degradation for this example.
    # To use 'unknown', change 'LR_bicubic' to 'LR_unknown'.
    LR_FOLDER = f"DF2K_train_LR_bicubic/X{SCALE_FACTOR}"
    HR_FOLDER = "DF2K_train_HR"

    # --- Resume Training Options ---
    RESUME_TRAINING = True  # Set to True to resume from checkpoint
    RESUME_CHECKPOINT_DIR = "/kaggle/input/degraded-100-complete/training_outputs/checkpoints"  # Path to checkpoint directory (e.g., "training_outputs/checkpoints")
    RESUME_EPOCH = 90  # Specific epoch to resume from (e.g., 10). If None, will find latest

    # --- Training Parameters ---
    NUM_EPOCHS = 180 # Increased for demonstration
    BATCH_SIZE = 8
    # Patch sizes for training. LR patches are cropped and corresponding HR patches are derived.
    LR_PATCH_SIZE = 64
    HR_PATCH_SIZE = LR_PATCH_SIZE * SCALE_FACTOR  # Should be 256 for 64x4

    # --- Optimizer Parameters ---
    # Learning rates for the generator and discriminator
    LR_G = 1e-4
    LR_D = 1e-4
    # Betas for Adam optimizer
    BETA1 = 0.9
    BETA2 = 0.999

    # --- Loss Function Weights ---
    # These weights balance the contribution of each loss component.
    W_L1 = 1.0          # L1 pixel loss
    W_PERCEPTUAL = 1.0  # Perceptual (VGG) loss
    W_GAN = 0.1         # Adversarial (GAN) loss

    # --- Novel Architecture Options ---
    USE_NOVEL_DEGRADATION = True  # Use sophisticated degradation pipeline
    USE_TRANSFORMER_GENERATOR = True  # Use Transformer-based generator instead of RRDB
    USE_MULTISCALE_DISCRIMINATOR = True  # Use multi-scale discriminator
    
    # Transformer Generator Parameters (only used if USE_TRANSFORMER_GENERATOR = True)
    TRANSFORMER_NUM_FEAT = 96     # Number of features in transformer
    TRANSFORMER_NUM_BLOCK = 6     # Number of RSTB blocks
    TRANSFORMER_NUM_HEAD = 6      # Number of attention heads
    TRANSFORMER_WINDOW_SIZE = 8   # Window size for attention

    # --- Hardware and Logging ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Number of worker threads for DataLoader
    SAVE_EVERY_N_EPOCHS = 2 # Save checkpoints and comparison images every N epochs

config = Config()

# Create output directory
os.makedirs(os.path.join(config.OUTPUT_DIR, "checkpoints"), exist_ok=True)
os.makedirs(os.path.join(config.OUTPUT_DIR, "images"), exist_ok=True)


print("✅ SECTION 2: CONFIGURATION COMPLETE")
print(f"   - Device: {config.DEVICE}")
print(f"   - Scale Factor: {config.SCALE_FACTOR}x")
print(f"   - LR/HR Patch Size: {config.LR_PATCH_SIZE}/{config.HR_PATCH_SIZE}")
print(f"   - Batch Size: {config.BATCH_SIZE}")
print(f"   - Dataset Path: {config.DATASET_PATH}")
print(f"   - Output Directory: {config.OUTPUT_DIR}")
print(f"   - Resume Training: {config.RESUME_TRAINING}")
if config.RESUME_TRAINING:
    print(f"   - Resume Checkpoint Dir: {config.RESUME_CHECKPOINT_DIR or 'default (training_outputs/checkpoints)'}")
    print(f"   - Resume Epoch: {config.RESUME_EPOCH or 'latest available'}")
print("\n🔥 NOVEL CONTRIBUTIONS ENABLED:")
print(f"   - Novel Degradation Pipeline: {'✅ ENABLED' if config.USE_NOVEL_DEGRADATION else '❌ DISABLED'}")
print(f"   - Transformer Generator: {'✅ ENABLED' if config.USE_TRANSFORMER_GENERATOR else '❌ DISABLED (using RRDB)'}")
print(f"   - Multi-Scale Discriminator: {'✅ ENABLED' if config.USE_MULTISCALE_DISCRIMINATOR else '❌ DISABLED (using U-Net)'}")
if config.USE_TRANSFORMER_GENERATOR:
    print(f"     * Features: {config.TRANSFORMER_NUM_FEAT}, Blocks: {config.TRANSFORMER_NUM_BLOCK}")
    print(f"     * Attention Heads: {config.TRANSFORMER_NUM_HEAD}, Window Size: {config.TRANSFORMER_WINDOW_SIZE}")
print("-" * 80)

# =================================================================================================
# SECTION 2.5: NOVEL DEGRADATION PIPELINE
# =================================================================================================
# This section implements a sophisticated, realistic degradation pipeline that goes beyond
# simple bicubic downsampling to include multiple real-world corruptions.

class NovelDegradationPipeline:
    """
    Advanced degradation pipeline that simulates realistic image corruptions:
    - Chromatic aberration
    - Camera sensor noise patterns  
    - Motion blur (linear and rotational)
    - JPEG compression artifacts
    - Lens distortion
    - Random resizing with different interpolation methods
    """
    
    def __init__(self, scale_factor=4):
        self.scale_factor = scale_factor
        
    def apply_chromatic_aberration(self, img_array):
        """
        Simulates chromatic aberration by shifting color channels.
        This is a common lens imperfection in real cameras.
        """
        h, w, c = img_array.shape
        if c != 3:
            return img_array
            
        # Random shift parameters
        shift_r = np.random.uniform(-1.5, 1.5)
        shift_b = np.random.uniform(-1.5, 1.5)
        
        # Apply shifts to red and blue channels
        img_shifted = img_array.copy()
        
        # Red channel shift
        if shift_r != 0:
            img_shifted[:, :, 0] = ndimage.shift(img_array[:, :, 0], shift_r, mode='nearest')
        
        # Blue channel shift  
        if shift_b != 0:
            img_shifted[:, :, 2] = ndimage.shift(img_array[:, :, 2], shift_b, mode='nearest')
            
        return img_shifted
    
    def apply_sensor_noise(self, img_array):
        """
        Simulates camera sensor noise patterns including:
        - Shot noise (Poisson)
        - Read noise (Gaussian)
        - Dark current noise
        """
        img_float = img_array.astype(np.float32) / 255.0
        
        # Shot noise (signal-dependent Poisson noise)
        if np.random.random() < 0.7:
            shot_noise_scale = np.random.uniform(0.01, 0.05)
            img_float = np.random.poisson(img_float / shot_noise_scale) * shot_noise_scale
        
        # Read noise (Gaussian)
        if np.random.random() < 0.8:
            read_noise_std = np.random.uniform(0.005, 0.02)
            read_noise = np.random.normal(0, read_noise_std, img_float.shape)
            img_float += read_noise
        
        # Dark current noise (position-dependent)
        if np.random.random() < 0.3:
            dark_current = np.random.uniform(0.001, 0.008)
            img_float += dark_current
            
        return np.clip(img_float * 255.0, 0, 255).astype(np.uint8)
    
    def apply_motion_blur(self, img_array):
        """
        Applies motion blur to simulate camera shake or object movement.
        Includes both linear and rotational motion blur.
        """
        if np.random.random() < 0.4:  # 40% chance of motion blur
            blur_type = np.random.choice(['linear', 'rotational'])
            
            if blur_type == 'linear':
                # Linear motion blur
                kernel_size = np.random.randint(5, 15)
                angle = np.random.uniform(0, 180)
                
                # Create motion blur kernel
                kernel = np.zeros((kernel_size, kernel_size))
                center = kernel_size // 2
                for i in range(kernel_size):
                    x = int(center + (i - center) * np.cos(np.radians(angle)))
                    y = int(center + (i - center) * np.sin(np.radians(angle)))
                    if 0 <= x < kernel_size and 0 <= y < kernel_size:
                        kernel[y, x] = 1
                kernel /= np.sum(kernel)
                
                # Apply convolution
                if img_array.ndim == 3:
                    blurred = np.zeros_like(img_array)
                    for c in range(img_array.shape[2]):
                        blurred[:, :, c] = cv2.filter2D(img_array[:, :, c], -1, kernel)
                    return blurred
                else:
                    return cv2.filter2D(img_array, -1, kernel)
            
            else:  # rotational blur
                # Simplified rotational blur using Gaussian
                sigma = np.random.uniform(0.5, 2.0)
                return gaussian_filter(img_array, sigma=sigma)
                
        return img_array
    
    def apply_jpeg_compression(self, img_array, pil_img=None):
        """
        Applies JPEG compression artifacts with random quality levels.
        """
        if np.random.random() < 0.6:  # 60% chance of JPEG compression
            quality = np.random.randint(30, 85)
            
            if pil_img is None:
                pil_img = Image.fromarray(img_array)
            
            # Save to memory buffer with JPEG compression
            import io
            buffer = io.BytesIO()
            pil_img.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            
            # Load back from buffer
            compressed_img = Image.open(buffer)
            return np.array(compressed_img)
            
        return img_array
    
    def apply_lens_distortion(self, img_array):
        """
        Applies subtle lens distortion (barrel or pincushion).
        """
        if np.random.random() < 0.3:  # 30% chance of lens distortion
            h, w = img_array.shape[:2]
            distortion_coeff = np.random.uniform(-0.1, 0.1)
            
            # Create distortion map
            center_x, center_y = w // 2, h // 2
            y, x = np.mgrid[0:h, 0:w]
            
            # Normalize coordinates
            x_norm = (x - center_x) / center_x
            y_norm = (y - center_y) / center_y
            
            # Apply radial distortion
            r_squared = x_norm**2 + y_norm**2
            distortion_factor = 1 + distortion_coeff * r_squared
            
            x_distorted = x_norm * distortion_factor * center_x + center_x
            y_distorted = y_norm * distortion_factor * center_y + center_y
            
            # Clip to image bounds
            x_distorted = np.clip(x_distorted, 0, w-1)
            y_distorted = np.clip(y_distorted, 0, h-1)
            
            # Apply distortion
            if img_array.ndim == 3:
                distorted = np.zeros_like(img_array)
                for c in range(img_array.shape[2]):
                    distorted[:, :, c] = cv2.remap(
                        img_array[:, :, c], 
                        x_distorted.astype(np.float32), 
                        y_distorted.astype(np.float32),
                        cv2.INTER_LINEAR
                    )
                return distorted
            else:
                return cv2.remap(
                    img_array, 
                    x_distorted.astype(np.float32), 
                    y_distorted.astype(np.float32),
                    cv2.INTER_LINEAR
                )
                
        return img_array
    
    def apply_random_resize(self, img_array):
        """
        Applies random resizing with different interpolation methods before final downsampling.
        This simulates the various resize operations that might occur in real image processing pipelines.
        """
        if np.random.random() < 0.5:  # 50% chance of intermediate resize
            h, w = img_array.shape[:2]
            
            # Random intermediate scale
            intermediate_scale = np.random.uniform(0.8, 1.2)
            new_h, new_w = int(h * intermediate_scale), int(w * intermediate_scale)
            
            # Random interpolation method
            interpolation = np.random.choice([
                cv2.INTER_LINEAR,
                cv2.INTER_CUBIC, 
                cv2.INTER_LANCZOS4,
                cv2.INTER_AREA
            ])
            
            # Resize and then back to original size
            resized = cv2.resize(img_array, (new_w, new_h), interpolation=interpolation)
            return cv2.resize(resized, (w, h), interpolation=cv2.INTER_CUBIC)
            
        return img_array
    
    def degrade_image(self, hr_pil_image):
        """
        Applies the complete degradation pipeline to a high-resolution PIL image.
        Returns a degraded low-resolution PIL image.
        """
        # Convert to numpy array
        hr_array = np.array(hr_pil_image)
        
        # Apply degradations in random order for more variety
        degradations = [
            self.apply_chromatic_aberration,
            self.apply_sensor_noise,
            self.apply_motion_blur,
            self.apply_lens_distortion,
            self.apply_random_resize
        ]
        
        # Shuffle and apply degradations
        np.random.shuffle(degradations)
        degraded_array = hr_array.copy()
        
        for degradation_func in degradations:
            degraded_array = degradation_func(degraded_array)
        
        # Convert back to PIL for JPEG compression
        degraded_pil = Image.fromarray(degraded_array)
        degraded_array = self.apply_jpeg_compression(degraded_array, degraded_pil)
        degraded_pil = Image.fromarray(degraded_array)
        
        # Final downsampling to LR resolution
        lr_size = (hr_pil_image.size[0] // self.scale_factor, 
                   hr_pil_image.size[1] // self.scale_factor)
        
        # Use random interpolation for final downsampling
        interpolation_methods = [
            Image.BICUBIC, Image.BILINEAR, Image.LANCZOS
        ]
        interpolation = np.random.choice(interpolation_methods)
        
        lr_pil = degraded_pil.resize(lr_size, interpolation)
        
        return lr_pil

print("✅ SECTION 2.5: NOVEL DEGRADATION PIPELINE COMPLETE")
print("   - Chromatic aberration simulation")
print("   - Camera sensor noise patterns")
print("   - Motion blur (linear and rotational)")
print("   - JPEG compression artifacts")
print("   - Lens distortion effects")
print("   - Random resizing with multiple interpolation methods")
print("-" * 80)

# =================================================================================================
# SECTION 3: DATASET AND DATALOADER
# =================================================================================================
# This section defines the custom PyTorch Dataset for DF2K.
# It handles finding image pairs, applying augmentations (random cropping, flipping),
# and preparing tensors for the models.

class DF2KDataset(Dataset):
    def __init__(self, config, use_novel_degradation=True):
        """
        Initializes the dataset object.
        Args:
            config (Config): The configuration object.
            use_novel_degradation (bool): If True, uses our novel degradation pipeline.
                                        If False, loads pre-degraded LR images.
        """
        print("[DATASET] Initializing DF2KDataset...")
        self.config = config
        self.use_novel_degradation = use_novel_degradation
        self.hr_path = os.path.join(config.DATASET_PATH, config.HR_FOLDER)
        
        if not use_novel_degradation:
            self.lr_path = os.path.join(config.DATASET_PATH, config.LR_FOLDER)

        # Find all HR images. We will derive LR paths from these.
        self.hr_image_files = sorted(glob.glob(os.path.join(self.hr_path, "*.png")))
        print(f"[DATASET] Found {len(self.hr_image_files)} high-resolution training images.")
        
        # Initialize degradation pipeline if using novel degradations
        if use_novel_degradation:
            self.degradation_pipeline = NovelDegradationPipeline(scale_factor=config.SCALE_FACTOR)
            print("[DATASET] Novel degradation pipeline initialized.")
        else:
            print("[DATASET] Using pre-degraded LR images.")

        # Define image transformations
        self.to_tensor = T.ToTensor() # Converts PIL Image to tensor in range [0, 1]

    def __len__(self):
        """Returns the total number of images in the dataset."""
        return len(self.hr_image_files)

    def __getitem__(self, index):
        """
        Retrieves one sample (LR and HR image pair) from the dataset.
        """
        # --- 1. Load HR image ---
        hr_image_path = self.hr_image_files[index]
        hr_image = Image.open(hr_image_path).convert("RGB")

        if self.use_novel_degradation:
            # --- 2. Get random HR patch first ---
            hr_patch = self.get_random_hr_patch(hr_image)
            
            # --- 3. Apply novel degradation pipeline to create LR ---
            lr_patch = self.degradation_pipeline.degrade_image(hr_patch)
            
        else:
            # --- 2. Derive and load corresponding LR image (original approach) ---
            filename = os.path.basename(hr_image_path)
            lr_image_name = f"{filename.split('.')[0]}x{self.config.SCALE_FACTOR}.png"
            lr_image_path = os.path.join(self.lr_path, lr_image_name)
            lr_image = Image.open(lr_image_path).convert("RGB")

            # --- 3. Get random patches (sub-image cropping) ---
            lr_patch, hr_patch = self.get_random_patches(lr_image, hr_image)

        # --- 4. Apply data augmentation ---
        lr_patch, hr_patch = self.augment(lr_patch, hr_patch)

        # --- 5. Convert to Tensors ---
        lr_tensor = self.to_tensor(lr_patch)
        hr_tensor = self.to_tensor(hr_patch)

        return lr_tensor, hr_tensor

    def get_random_hr_patch(self, hr_img):
        """Crops a random HR patch that will be used to generate LR via degradation."""
        hr_w, hr_h = hr_img.size
        hr_patch_size = self.config.HR_PATCH_SIZE
        
        # Get random top-left coordinates for the HR patch
        hr_x = random.randint(0, hr_w - hr_patch_size)
        hr_y = random.randint(0, hr_h - hr_patch_size)
        
        # Crop HR patch
        hr_patch = hr_img.crop((hr_x, hr_y, hr_x + hr_patch_size, hr_y + hr_patch_size))
        
        return hr_patch

    def get_random_patches(self, lr_img, hr_img):
        """Crops random corresponding patches from LR and HR images."""
        lr_w, lr_h = lr_img.size
        lr_patch_size = self.config.LR_PATCH_SIZE
        hr_patch_size = self.config.HR_PATCH_SIZE

        # Get random top-left coordinates for the LR patch
        lr_x = random.randint(0, lr_w - lr_patch_size)
        lr_y = random.randint(0, lr_h - lr_patch_size)

        # Derive corresponding HR coordinates
        hr_x = lr_x * self.config.SCALE_FACTOR
        hr_y = lr_y * self.config.SCALE_FACTOR

        # Crop patches
        lr_patch = lr_img.crop((lr_x, lr_y, lr_x + lr_patch_size, lr_y + lr_patch_size))
        hr_patch = hr_img.crop((hr_x, hr_y, hr_x + hr_patch_size, hr_y + hr_patch_size))

        return lr_patch, hr_patch

    def augment(self, lr_img, hr_img):
        """Applies random horizontal flip and rotation."""
        # Horizontal flip
        if random.random() > 0.5:
            lr_img = lr_img.transpose(Image.FLIP_LEFT_RIGHT)
            hr_img = hr_img.transpose(Image.FLIP_LEFT_RIGHT)

        # Rotation (0, 90, 180, 270 degrees)
        rotation_angle = random.choice([0, 90, 180, 270])
        if rotation_angle != 0:
            lr_img = lr_img.rotate(rotation_angle)
            hr_img = hr_img.rotate(rotation_angle)

        return lr_img, hr_img

print("✅ SECTION 3: DATASET AND DATALOADER SETUP COMPLETE")
print("-" * 80)

# =================================================================================================
# SECTION 4: MODEL ARCHITECTURE (Real-ESRGAN Components)
# =================================================================================================
# This section implements the core components of the Real-ESRGAN architecture.
# Generator: Residual-in-Residual Dense Block (RRDBNet)
# Discriminator: U-Net based discriminator for patch-level feedback.

# --- Generator Architecture ---

class ResidualDenseBlock(nn.Module):
    """Residual Dense Block, the core of the RRDB."""
    def __init__(self, nf=64, gc=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x # Residual scaling

class RRDB(nn.Module):
    """Residual-in-Residual Dense Block."""
    def __init__(self, nf=64):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(nf)
        self.rdb2 = ResidualDenseBlock(nf)
        self.rdb3 = ResidualDenseBlock(nf)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x

class RRDBNet(nn.Module):
    """The main generator network (RRDBNet)."""
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, scale=4):
        super(RRDBNet, self).__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)

        # RRDB body
        self.body = nn.Sequential(*[RRDB(nf) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)

        # --- *** MAJOR BUG FIX & REVISION IN UPSAMPLING BLOCK *** ---
        # The original code was incorrect. For nn.PixelShuffle(r), the preceding
        # convolution must produce r*r times the channels. Here, r=2 for each step.
        # Total upscaling is 4x, so we have two 2x upsampling steps.
        num_upsamples = int(np.log2(scale))
        self.upsample_blocks = nn.ModuleList()
        for _ in range(num_upsamples):
            # Each PixelShuffle(2) needs channels to be scaled by 2^2=4
            self.upsample_blocks.append(
                nn.Sequential(
                    nn.Conv2d(nf, nf * 4, 3, 1, 1),
                    nn.PixelShuffle(2),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
        # --- END OF REVISION ---

        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat

        # Apply upsampling blocks
        for block in self.upsample_blocks:
            feat = block(feat)

        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


# =================================================================================================
# SECTION 4.1: NOVEL TRANSFORMER-BASED GENERATOR ARCHITECTURE
# =================================================================================================
# This section implements a Transformer-based generator inspired by SwinIR,
# replacing the traditional RRDB blocks with self-attention mechanisms.

class WindowAttention(nn.Module):
    """Window-based Multi-Head Self Attention (W-MSA) module with relative position bias."""
    
    def __init__(self, dim, window_size, num_heads, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x

def window_partition(x, window_size):
    """Partition into non-overlapping windows."""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """Reverse window partition."""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block."""
    
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim),
        )

    def forward(self, x):
        H, W = x.shape[1], x.shape[2]
        B, L, C = x.shape[0], H * W, x.shape[3]

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H, W, C)

        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x

class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB)."""
    
    def __init__(self, dim, num_heads, window_size, depth, mlp_ratio=4.):
        super(RSTB, self).__init__()
        
        self.dim = dim
        self.window_size = window_size
        
        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, num_heads=num_heads, window_size=window_size,
                                shift_size=0 if (i % 2 == 0) else window_size // 2,
                                mlp_ratio=mlp_ratio)
            for i in range(depth)])
        
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # B, H, W, C
        
        for blk in self.blocks:
            x = blk(x)
        
        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        x = self.conv(x)
        
        return x

class TransformerESRGAN(nn.Module):
    """Transformer-based Super-Resolution GAN Generator."""
    
    def __init__(self, in_nc=3, out_nc=3, nf=64, num_feat=64, num_block=6, num_head=6, window_size=8, scale=4):
        super(TransformerESRGAN, self).__init__()
        self.scale = scale
        self.window_size = window_size
        
        # Shallow feature extraction
        self.conv_first = nn.Conv2d(in_nc, num_feat, 3, 1, 1)
        
        # Deep feature extraction
        self.num_block = num_block
        self.blocks = nn.ModuleList([
            RSTB(dim=num_feat, num_heads=num_head, window_size=window_size, depth=6, mlp_ratio=2.)
            for _ in range(num_block)
        ])
        
        self.conv_after_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        
        # Upsampling
        num_upsamples = int(np.log2(scale))
        self.upsample_blocks = nn.ModuleList()
        for _ in range(num_upsamples):
            self.upsample_blocks.append(
                nn.Sequential(
                    nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1),
                    nn.PixelShuffle(2),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
        
        # Final output
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, out_nc, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        # Shallow feature extraction
        x = self.conv_first(x)
        
        # Deep feature extraction
        residual = x
        for block in self.blocks:
            x = x + block(x)
        
        x = self.conv_after_body(x)
        x = x + residual
        
        # Upsampling
        for block in self.upsample_blocks:
            x = block(x)
        
        # Final output
        x = self.conv_last(self.lrelu(self.conv_hr(x)))
        
        return x


# --- Discriminator Architecture ---
class UNetDiscriminator(nn.Module):
    """U-Net based Discriminator for realistic feedback."""
    def __init__(self, in_nc=3, nf=64):
        super(UNetDiscriminator, self).__init__()

        # Downsampling blocks
        self.conv0_0 = nn.Conv2d(in_nc, nf, 3, 1, 1)
        self.conv0_1 = nn.Conv2d(nf, nf, 4, 2, 1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(nf)

        self.conv1_0 = nn.Conv2d(nf, nf * 2, 3, 1, 1, bias=False)
        self.conv1_1 = nn.Conv2d(nf * 2, nf * 2, 4, 2, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(nf * 2)

        self.conv2_0 = nn.Conv2d(nf * 2, nf * 4, 3, 1, 1, bias=False)
        self.conv2_1 = nn.Conv2d(nf * 4, nf * 4, 4, 2, 1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(nf * 4)

        self.conv3_0 = nn.Conv2d(nf * 4, nf * 8, 3, 1, 1, bias=False)
        self.conv3_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(nf * 8)

        # Final output convolution
        self.conv4 = nn.Conv2d(nf * 8, 1, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x0 = self.lrelu(self.conv0_0(x))
        x0 = self.lrelu(self.bn0_1(self.conv0_1(x0)))

        x1 = self.lrelu(self.bn1_1(self.conv1_1(self.conv1_0(x0))))
        x2 = self.lrelu(self.bn2_1(self.conv2_1(self.conv2_0(x1))))
        x3 = self.lrelu(self.bn3_1(self.conv3_1(self.conv3_0(x2))))

        out = self.conv4(x3)
        return out

# =================================================================================================
# SECTION 4.2: NOVEL MULTI-SCALE DISCRIMINATOR ARCHITECTURE
# =================================================================================================
# This section implements an improved multi-scale discriminator that provides
# feedback at multiple scales for better texture and detail preservation.

class MultiScaleDiscriminator(nn.Module):
    """Multi-scale discriminator that operates at different resolutions."""
    
    def __init__(self, in_nc=3, nf=64, num_scales=3):
        super(MultiScaleDiscriminator, self).__init__()
        self.num_scales = num_scales
        
        # Create discriminators for different scales
        self.discriminators = nn.ModuleList()
        for i in range(num_scales):
            self.discriminators.append(self._make_discriminator(in_nc, nf))
        
        # Downsampling layers for multi-scale input
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
    
    def _make_discriminator(self, in_nc, nf):
        """Creates a single discriminator network."""
        return nn.Sequential(
            # Layer 1
            nn.Conv2d(in_nc, nf, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            
            # Layer 2
            nn.Conv2d(nf, nf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 2),
            nn.LeakyReLU(0.2, True),
            
            # Layer 3
            nn.Conv2d(nf * 2, nf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 4),
            nn.LeakyReLU(0.2, True),
            
            # Layer 4
            nn.Conv2d(nf * 4, nf * 8, 4, 1, 1, bias=False),
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2, True),
            
            # Output layer
            nn.Conv2d(nf * 8, 1, 4, 1, 1)
        )
    
    def forward(self, x):
        outputs = []
        input_img = x
        
        for i in range(self.num_scales):
            outputs.append(self.discriminators[i](input_img))
            if i != self.num_scales - 1:  # Don't downsample for the last scale
                input_img = self.downsample(input_img)
        
        return outputs

class SpectralNormDiscriminator(nn.Module):
    """Discriminator with Spectral Normalization for training stability."""
    
    def __init__(self, in_nc=3, nf=64):
        super(SpectralNormDiscriminator, self).__init__()
        
        self.features = nn.Sequential(
            # Layer 1: No spectral norm on first layer
            nn.Conv2d(in_nc, nf, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            
            # Layer 2
            nn.utils.spectral_norm(nn.Conv2d(nf, nf * 2, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(nf * 2),
            nn.LeakyReLU(0.2, True),
            
            # Layer 3
            nn.utils.spectral_norm(nn.Conv2d(nf * 2, nf * 4, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(nf * 4),
            nn.LeakyReLU(0.2, True),
            
            # Layer 4
            nn.utils.spectral_norm(nn.Conv2d(nf * 4, nf * 8, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2, True),
            
            # Layer 5
            nn.utils.spectral_norm(nn.Conv2d(nf * 8, nf * 16, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(nf * 16),
            nn.LeakyReLU(0.2, True),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(nf * 16, 1))
        )
    
    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features)
        return output

print("✅ SECTION 4: MODEL ARCHITECTURE COMPLETE")
print("   - Original Generator: RRDBNet")
print("   - Novel Generator: TransformerESRGAN (Swin Transformer-based)")
print("   - Original Discriminator: UNetDiscriminator")
print("   - Novel Discriminators: MultiScaleDiscriminator, SpectralNormDiscriminator")
print("-" * 80)

# =================================================================================================
# SECTION 5: LOSS FUNCTIONS AND UTILITIES
# =================================================================================================
# This section defines the loss functions and a new utility function for saving comparison images.

class PerceptualLoss(nn.Module):
    """
    Calculates the VGG-based perceptual loss.
    Uses features from the VGG19 network to compare generated and real images.
    """
    def __init__(self, device):
        super(PerceptualLoss, self).__init__()
        print("[LOSS] Initializing Perceptual Loss (VGG19)...")
        vgg = vgg19(pretrained=True).features[:35].eval().to(device)
        self.vgg = vgg
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.loss_fn = nn.L1Loss()
        print("[LOSS] VGG19 loaded and frozen.")

    def forward(self, generated_img, target_img):
        vgg_gen = self.vgg(generated_img)
        vgg_target = self.vgg(target_img)
        return self.loss_fn(vgg_gen, vgg_target)

# <<< MODIFICATION: NEW FUNCTION TO SAVE COMPARISON IMAGES >>>
def save_comparison_image(lr_tensor, sr_tensor, hr_tensor, epoch, config):
    """
    Saves a side-by-side comparison of LR, SR, and HR images.
    - lr_tensor: The low-resolution input tensor.
    - sr_tensor: The super-resolved output from the generator.
    - hr_tensor: The high-resolution ground truth tensor.
    """
    # We only need one image from the batch for comparison
    lr_img = lr_tensor[0].cpu()
    sr_img = sr_tensor[0].cpu()
    hr_img = hr_tensor[0].cpu()

    # Upscale the LR image to the same size as HR/SR for visual comparison
    lr_upscaled = F.interpolate(
        lr_img.unsqueeze(0),
        size=(config.HR_PATCH_SIZE, config.HR_PATCH_SIZE),
        mode='bicubic',
        align_corners=False
    ).squeeze(0)

    # Create a grid of the three images
    comparison_grid = vutils.make_grid(
        [lr_upscaled, sr_img, hr_img],
        nrow=3,
        normalize=True,
        scale_each=True,
        pad_value=1 # Add white padding between images
    )

    # Save the grid
    filepath = os.path.join(config.OUTPUT_DIR, "images", f"comparison_epoch_{epoch:03d}.png")
    vutils.save_image(comparison_grid, filepath)
    # No need to print here, it will be printed in the main loop


print("✅ SECTION 5: LOSS FUNCTIONS & UTILITIES SETUP COMPLETE")
print("-" * 80)

# =================================================================================================
# SECTION 5.5: CHECKPOINT MANAGEMENT UTILITIES
# =================================================================================================

def save_checkpoint(generator, discriminator, optimizer_g, optimizer_d, epoch, config):
    """
    Saves model states, optimizer states, and epoch information in a single checkpoint file.
    """
    checkpoint_data = {
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_g_state_dict': optimizer_g.state_dict(),
        'optimizer_d_state_dict': optimizer_d.state_dict(),
        'epoch': epoch,
        'config': {
            'SCALE_FACTOR': config.SCALE_FACTOR,
            'LR_G': config.LR_G,
            'LR_D': config.LR_D,
            'BETA1': config.BETA1,
            'BETA2': config.BETA2
        }
    }
    
    checkpoint_path = os.path.join(config.OUTPUT_DIR, "checkpoints", f"checkpoint_epoch_{epoch:03d}.pth")
    torch.save(checkpoint_data, checkpoint_path)
    
    print(f"✅ Checkpoint saved for epoch {epoch}: {os.path.basename(checkpoint_path)}")

def load_checkpoint_from_individual_files(checkpoint_dir, epoch, generator, discriminator, optimizer_g, optimizer_d, config):
    """
    Loads from individual generator and discriminator files (for backward compatibility).
    Returns the epoch number if successful, None if files don't exist.
    """
    g_path = os.path.join(checkpoint_dir, f"generator_epoch_{epoch:03d}.pth")
    d_path = os.path.join(checkpoint_dir, f"discriminator_epoch_{epoch:03d}.pth")
    
    if os.path.exists(g_path) and os.path.exists(d_path):
        print(f"[RESUME] Loading from individual model files (epoch {epoch})")
        generator.load_state_dict(torch.load(g_path, map_location=config.DEVICE))
        discriminator.load_state_dict(torch.load(d_path, map_location=config.DEVICE))
        print("⚠️  WARNING: Optimizer states not restored (loaded from individual files)")
        return epoch
    return None

def find_latest_checkpoint(checkpoint_dir):
    """
    Finds the latest checkpoint file in the given directory.
    Returns the path to the latest checkpoint and the epoch number.
    If no comprehensive checkpoints found, tries to find individual model files.
    """
    if not os.path.exists(checkpoint_dir):
        return None, 0
    
    # First, look for comprehensive checkpoint files
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth"))
    if checkpoint_files:
        epochs = []
        for file in checkpoint_files:
            filename = os.path.basename(file)
            try:
                epoch_num = int(filename.split('_')[2].split('.')[0])
                epochs.append((epoch_num, file))
            except (IndexError, ValueError):
                continue
        
        if epochs:
            latest_epoch, latest_file = max(epochs, key=lambda x: x[0])
            return latest_file, latest_epoch
    
    # If no comprehensive checkpoints, look for individual generator files as fallback
    generator_files = glob.glob(os.path.join(checkpoint_dir, "generator_epoch_*.pth"))
    if generator_files:
        epochs = []
        for file in generator_files:
            filename = os.path.basename(file)
            try:
                epoch_num = int(filename.split('_')[2].split('.')[0])
                epochs.append(epoch_num)
            except (IndexError, ValueError):
                continue
        
        if epochs:
            latest_epoch = max(epochs)
            return "individual_files", latest_epoch
    
    return None, 0

def load_checkpoint(checkpoint_path, generator, discriminator, optimizer_g, optimizer_d, config):
    """
    Loads model states, optimizer states, and epoch information from checkpoint.
    Handles both comprehensive checkpoints and individual model files.
    Returns the epoch number to resume from.
    """
    if checkpoint_path == "individual_files":
        # Handle loading from individual files
        checkpoint_dir = config.RESUME_CHECKPOINT_DIR or os.path.join(config.OUTPUT_DIR, "checkpoints")
        epoch = load_checkpoint_from_individual_files(
            checkpoint_dir, config.RESUME_EPOCH, generator, discriminator, optimizer_g, optimizer_d, config
        )
        if epoch is not None:
            return epoch
        else:
            print(f"❌ ERROR: Could not load individual model files for epoch {config.RESUME_EPOCH}")
            return 1
    
    print(f"[RESUME] Loading comprehensive checkpoint from: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
        
        # Load model states
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        
        # Load optimizer states
        optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        
        # Get epoch information
        resume_epoch = checkpoint['epoch']
        
        # Verify config compatibility (optional warnings)
        saved_config = checkpoint.get('config', {})
        if saved_config.get('SCALE_FACTOR') != config.SCALE_FACTOR:
            print(f"⚠️  WARNING: Scale factor mismatch. Saved: {saved_config.get('SCALE_FACTOR')}, Current: {config.SCALE_FACTOR}")
        
        print(f"✅ Comprehensive checkpoint loaded successfully. Resuming from epoch {resume_epoch}")
        return resume_epoch
        
    except Exception as e:
        print(f"❌ ERROR loading checkpoint: {str(e)}")
        return 1

def validate_resume_config(config):
    """
    Validates the resume configuration and returns the starting epoch and checkpoint path.
    """
    if not config.RESUME_TRAINING:
        return 1, None
    
    checkpoint_dir = config.RESUME_CHECKPOINT_DIR or os.path.join(config.OUTPUT_DIR, "checkpoints")
    
    if config.RESUME_EPOCH is not None:
        # Resume from specific epoch - first try comprehensive checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{config.RESUME_EPOCH:03d}.pth")
        if os.path.exists(checkpoint_path):
            return config.RESUME_EPOCH + 1, checkpoint_path
        
        # If comprehensive checkpoint doesn't exist, try individual files
        g_path = os.path.join(checkpoint_dir, f"generator_epoch_{config.RESUME_EPOCH:03d}.pth")
        d_path = os.path.join(checkpoint_dir, f"discriminator_epoch_{config.RESUME_EPOCH:03d}.pth")
        if os.path.exists(g_path) and os.path.exists(d_path):
            print(f"⚠️  WARNING: Using individual model files for epoch {config.RESUME_EPOCH} (optimizer states will not be restored)")
            return config.RESUME_EPOCH + 1, "individual_files"
        
        print(f"❌ ERROR: No checkpoint found for epoch {config.RESUME_EPOCH}")
        return 1, None
    else:
        # Find and resume from latest checkpoint
        latest_checkpoint, latest_epoch = find_latest_checkpoint(checkpoint_dir)
        if latest_checkpoint is None:
            print(f"⚠️  WARNING: No checkpoints found in {checkpoint_dir}. Starting fresh training.")
            return 1, None
        
        if latest_checkpoint == "individual_files":
            print(f"⚠️  WARNING: Using individual model files for epoch {latest_epoch} (optimizer states will not be restored)")
        
        return latest_epoch + 1, latest_checkpoint

print("✅ SECTION 5.5: CHECKPOINT MANAGEMENT UTILITIES COMPLETE")
print("-" * 80)

# =================================================================================================
# SECTION 6: TRAINING ORCHESTRATION
# =================================================================================================
# This is the main training loop. It handles everything from initialization to logging and saving.

def train():
    """Main training function."""
    print("[TRAIN] Starting training orchestration...")

    # --- 1. Initialize Dataloader ---
    dataset = DF2KDataset(config, use_novel_degradation=config.USE_NOVEL_DEGRADATION)
    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True # Important for consistent batch sizes
    )
    print(f"[TRAIN] DataLoader created with {len(dataloader)} batches per epoch.")

    # --- 2. Initialize Models ---
    if config.USE_TRANSFORMER_GENERATOR:
        generator = TransformerESRGAN(
            num_feat=config.TRANSFORMER_NUM_FEAT,
            num_block=config.TRANSFORMER_NUM_BLOCK,
            num_head=config.TRANSFORMER_NUM_HEAD,
            window_size=config.TRANSFORMER_WINDOW_SIZE,
            scale=config.SCALE_FACTOR
        ).to(config.DEVICE)
        print("[TRAIN] Transformer-based generator initialized.")
    else:
        generator = RRDBNet(scale=config.SCALE_FACTOR).to(config.DEVICE)
        print("[TRAIN] RRDB-based generator initialized.")
    
    if config.USE_MULTISCALE_DISCRIMINATOR:
        discriminator = MultiScaleDiscriminator().to(config.DEVICE)
        print("[TRAIN] Multi-scale discriminator initialized.")
    else:
        discriminator = UNetDiscriminator().to(config.DEVICE)
        print("[TRAIN] U-Net discriminator initialized.")

    # --- 3. Initialize Optimizers ---
    optimizer_g = optim.Adam(generator.parameters(), lr=config.LR_G, betas=(config.BETA1, config.BETA2))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=config.LR_D, betas=(config.BETA1, config.BETA2))
    print("[TRAIN] Adam optimizers initialized.")

    # --- 4. Initialize Loss Functions ---
    l1_loss = nn.L1Loss().to(config.DEVICE)
    perceptual_loss = PerceptualLoss(device=config.DEVICE)
    adversarial_loss = nn.BCEWithLogitsLoss().to(config.DEVICE)
    print("[TRAIN] Loss functions initialized.")

    # --- 5. Handle Resume Training ---
    start_epoch, checkpoint_path = validate_resume_config(config)
    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, generator, discriminator, optimizer_g, optimizer_d, config)
        print(f"[RESUME] Training will resume from epoch {start_epoch}")
    else:
        print(f"[TRAIN] Starting fresh training from epoch 1")

    # --- 6. The Main Training Loop ---
    print("\n" + "="*30 + " STARTING TRAINING " + "="*30)
    start_time = time.time()

    for epoch in range(start_epoch, config.NUM_EPOCHS + 1):
        generator.train()
        discriminator.train()

        epoch_progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{config.NUM_EPOCHS}", leave=True)

        for lr_imgs, hr_imgs in epoch_progress_bar:
            lr_imgs = lr_imgs.to(config.DEVICE)
            hr_imgs = hr_imgs.to(config.DEVICE)

            # --- Train Discriminator ---
            optimizer_d.zero_grad()
            fake_sr_imgs = generator(lr_imgs).detach()
            real_pred = discriminator(hr_imgs)
            fake_pred = discriminator(fake_sr_imgs)
            
            if config.USE_MULTISCALE_DISCRIMINATOR:
                # Multi-scale discriminator returns a list of predictions
                loss_d_real = 0
                loss_d_fake = 0
                for real_p, fake_p in zip(real_pred, fake_pred):
                    loss_d_real += adversarial_loss(real_p, torch.ones_like(real_p))
                    loss_d_fake += adversarial_loss(fake_p, torch.zeros_like(fake_p))
                loss_d_real /= len(real_pred)
                loss_d_fake /= len(fake_pred)
            else:
                # Single-scale discriminator
                loss_d_real = adversarial_loss(real_pred, torch.ones_like(real_pred))
                loss_d_fake = adversarial_loss(fake_pred, torch.zeros_like(fake_pred))
            
            loss_d = (loss_d_real + loss_d_fake) / 2
            loss_d.backward()
            optimizer_d.step()

            # --- Train Generator ---
            optimizer_g.zero_grad()
            fake_sr_imgs = generator(lr_imgs)
            fake_pred_g = discriminator(fake_sr_imgs)

            # Calculate generator losses
            loss_g_l1 = config.W_L1 * l1_loss(fake_sr_imgs, hr_imgs)
            loss_g_perceptual = config.W_PERCEPTUAL * perceptual_loss(fake_sr_imgs, hr_imgs)
            
            if config.USE_MULTISCALE_DISCRIMINATOR:
                # Multi-scale adversarial loss
                loss_g_gan = 0
                for fake_p in fake_pred_g:
                    loss_g_gan += adversarial_loss(fake_p, torch.ones_like(fake_p))
                loss_g_gan = config.W_GAN * loss_g_gan / len(fake_pred_g)
            else:
                # Single-scale adversarial loss
                loss_g_gan = config.W_GAN * adversarial_loss(fake_pred_g, torch.ones_like(fake_pred_g))
            
            loss_g = loss_g_l1 + loss_g_perceptual + loss_g_gan
            loss_g.backward()
            optimizer_g.step()

            # --- Logging with TQDM ---
            epoch_progress_bar.set_postfix(
                G_Loss=f"{loss_g.item():.4f}",
                D_Loss=f"{loss_d.item():.4f}",
                G_L1=f"{loss_g_l1.item():.4f}",
                G_Perc=f"{loss_g_perceptual.item():.4f}",
                G_GAN=f"{loss_g_gan.item():.4f}"
            )

        # --- End of Epoch Saving ---
        if epoch % config.SAVE_EVERY_N_EPOCHS == 0:
            # Save complete checkpoint with optimizer states
            save_checkpoint(generator, discriminator, optimizer_g, optimizer_d, epoch, config)

            # Save comparison image
            generator.eval() # Set to eval mode for consistent output
            with torch.no_grad():
                # Use the last batch from the epoch for the comparison image
                fake_sr_for_image = generator(lr_imgs)
                save_comparison_image(lr_imgs, fake_sr_for_image, hr_imgs, epoch, config)
            generator.train() # Set back to train mode

            tqdm.write(f"\n✅ Checkpoint and comparison image saved for epoch {epoch}")


    print("\n" + "="*30 + " TRAINING FINISHED " + "="*30)
    total_time = time.time() - start_time
    print(f"Total training time: {total_time / 3600:.2f} hours")

print("✅ SECTION 6: TRAINING ORCHESTRATION COMPLETE")
print("-" * 80)

# =================================================================================================
# SECTION 7: MAIN EXECUTION BLOCK
# =================================================================================================

if __name__ == '__main__':
    print("✅ SECTION 7: MAIN EXECUTION BLOCK")
    print("Initiating the training process...")

    # Check if dataset path exists
    if not os.path.exists(config.DATASET_PATH) or not os.path.exists(os.path.join(config.DATASET_PATH, config.HR_FOLDER)):
        print("\n" + "!"*80)
        print("! ERROR: Dataset path not found.")
        print(f"! Please check if the `DATASET_PATH` in the Config class is correct.")
        print(f"! Current path: '{config.DATASET_PATH}'")
        print(f"! The script expects the following structure inside: '{config.HR_FOLDER}' and '{config.LR_FOLDER}'")
        print("!"*80)
    else:
        # Launch the training
        train()