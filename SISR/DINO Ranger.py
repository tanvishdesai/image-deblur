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
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.utils as vutils
from tqdm import tqdm
from einops import rearrange # <<< NOVEL CONTRIBUTION: For Swin Transformer implementation

print("✅ SECTION 1: IMPORTS AND INITIAL SETUP COMPLETE")
print("-" * 80)

# =================================================================================================
# SECTION 2: CONFIGURATION
# =================================================================================================
# This class holds all hyperparameters and configuration settings in one place.

class Config:
    # --- Dataset and Paths ---
    DATASET_PATH = "/kaggle/input/df2kdata"
    OUTPUT_DIR = "training_outputs_novel" # Directory for new model
    SCALE_FACTOR = 4
    LR_FOLDER = f"DF2K_train_LR_bicubic/X{SCALE_FACTOR}"
    HR_FOLDER = "DF2K_train_HR"

    # --- Resume Training Options ---
    RESUME_TRAINING = False
    RESUME_CHECKPOINT_DIR = None
    RESUME_EPOCH = None
    RESUME_CHECKPOINT_PATH = None  # Direct path to a specific checkpoint file

    # --- Training Parameters ---
    NUM_EPOCHS = 100
    BATCH_SIZE = 4 # Reduced batch size due to larger model
    LR_PATCH_SIZE = 64
    HR_PATCH_SIZE = LR_PATCH_SIZE * SCALE_FACTOR

    # --- Optimizer Parameters ---
    LR_G = 1e-4
    LR_D = 1e-4
    BETA1 = 0.9
    BETA2 = 0.999

    # --- Loss Function Weights ---
    W_L1 = 1.0          # L1 pixel loss
    W_PERCEPTUAL = 1.0  # Perceptual (DINO) loss
    W_GAN = 0.1         # Adversarial (GAN) loss
    W_FFT = 0.8         # <<< NOVEL CONTRIBUTION: Weight for the new Frequency Loss

    # --- Hardware and Logging ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
    SAVE_EVERY_N_EPOCHS = 2

config = Config()

os.makedirs(os.path.join(config.OUTPUT_DIR, "checkpoints"), exist_ok=True)
os.makedirs(os.path.join(config.OUTPUT_DIR, "images"), exist_ok=True)


print("✅ SECTION 2: CONFIGURATION COMPLETE")
print(f"   - Device: {config.DEVICE}")
print(f"   - Novelty: Hybrid Swin-Transformer Generator, DINO Perceptual Loss, FFT Loss")
print(f"   - Batch Size: {config.BATCH_SIZE} (Reduced for larger model)")
print(f"   - Output Directory: {config.OUTPUT_DIR}")
print("-" * 80)


# =================================================================================================
# SECTION 3: DATASET AND DATALOADER (Unchanged, but crucial for context)
# =================================================================================================
class DF2KDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.hr_path = os.path.join(config.DATASET_PATH, config.HR_FOLDER)
        self.lr_path = os.path.join(config.DATASET_PATH, config.LR_FOLDER)
        self.hr_image_files = sorted(glob.glob(os.path.join(self.hr_path, "*.png")))
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.hr_image_files)

    def __getitem__(self, index):
        hr_image_path = self.hr_image_files[index]
        hr_image = Image.open(hr_image_path).convert("RGB")
        filename = os.path.basename(hr_image_path)
        lr_image_name = f"{filename.split('.')[0]}x{self.config.SCALE_FACTOR}.png"
        lr_image_path = os.path.join(self.lr_path, lr_image_name)
        lr_image = Image.open(lr_image_path).convert("RGB")
        lr_patch, hr_patch = self.get_random_patches(lr_image, hr_image)
        lr_patch, hr_patch = self.augment(lr_patch, hr_patch)
        return self.to_tensor(lr_patch), self.to_tensor(hr_patch)

    def get_random_patches(self, lr_img, hr_img):
        lr_w, lr_h = lr_img.size
        lr_patch_size = self.config.LR_PATCH_SIZE
        hr_patch_size = self.config.HR_PATCH_SIZE
        lr_x = random.randint(0, lr_w - lr_patch_size)
        lr_y = random.randint(0, lr_h - lr_patch_size)
        hr_x = lr_x * self.config.SCALE_FACTOR
        hr_y = lr_y * self.config.SCALE_FACTOR
        lr_patch = lr_img.crop((lr_x, lr_y, lr_x + lr_patch_size, lr_y + lr_patch_size))
        hr_patch = hr_img.crop((hr_x, hr_y, hr_x + hr_patch_size, hr_y + hr_patch_size))
        return lr_patch, hr_patch

    def augment(self, lr_img, hr_img):
        if random.random() > 0.5:
            lr_img = lr_img.transpose(Image.FLIP_LEFT_RIGHT)
            hr_img = hr_img.transpose(Image.FLIP_LEFT_RIGHT)
        rotation_angle = random.choice([0, 90, 180, 270])
        if rotation_angle != 0:
            lr_img = lr_img.rotate(rotation_angle)
            hr_img = hr_img.rotate(rotation_angle)
        return lr_img, hr_img

print("✅ SECTION 3: DATASET AND DATALOADER SETUP COMPLETE")
print("-" * 80)


# =================================================================================================
# SECTION 4: <<< NOVEL CONTRIBUTION >>> MODEL ARCHITECTURE (Hybrid Swin-Transformer GAN)
# =================================================================================================
# We replace the RRDBNet with a more modern hybrid architecture.

# --- Helper functions for Swin Transformer ---
def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

# --- Core Swin Transformer Block ---
class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask, persistent=False)


    def forward(self, x):
        # =========================================================================
        # START OF THE FIX
        # =========================================================================
        H, W = self.input_resolution
        B, L, C = x.shape
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
        # Dynamically expand attn_mask for batch size and num_heads
        if self.attn_mask is not None:
            # The input to MHA is x_windows: (B * num_windows, L_win, C)
            # self.attn_mask is: (num_windows, L_win, L_win)
            # We need to expand it to the shape MHA expects: (B * num_windows * num_heads, L_win, L_win)
            
            # Get B (batch size) by dividing attention input batch size by num_windows
            num_windows = self.attn_mask.shape[0]
            B_attn = x_windows.shape[0] // num_windows
            
            # Repeat the base mask for each item in the batch
            batch_mask = self.attn_mask.repeat(B_attn, 1, 1) # -> (B*num_windows, L_win, L_win)
            
            # Expand for heads and reshape to the 3D tensor MHA requires
            final_attn_mask = batch_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            final_attn_mask = final_attn_mask.view(-1, self.window_size * self.window_size, self.window_size * self.window_size)

            attn_windows, _ = self.attn(x_windows, x_windows, x_windows, attn_mask=final_attn_mask)
        else:
            # No mask needed for regular W-MSA
            attn_windows, _ = self.attn(x_windows, x_windows, x_windows, attn_mask=None)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x
        # =========================================================================
        # END OF THE FIX
        # =========================================================================

# --- New Hybrid Block replacing RRDB ---
class HybridBlock(nn.Module):
    def __init__(self, dim=64, input_resolution=(64, 64), num_heads=8, window_size=8):
        super().__init__()
        self.swin1 = SwinTransformerBlock(dim, input_resolution, num_heads, window_size, shift_size=0)
        self.swin2 = SwinTransformerBlock(dim, input_resolution, num_heads, window_size, shift_size=window_size//2)
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        res = x
        x = rearrange(x, 'b c h w -> b (h w) c') # to sequence
        x = self.swin1(x)
        x = self.swin2(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W) # to image
        x = self.conv(x)
        return x + res

# --- The main generator network (HybridSwinNet) ---
class HybridSwinNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, scale=4, input_res=(64,64)):
        super(HybridSwinNet, self).__init__()
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
        
        self.body = nn.Sequential(*[HybridBlock(dim=nf, input_resolution=input_res) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)

        num_upsamples = int(np.log2(scale))
        self.upsample_blocks = nn.ModuleList()
        for _ in range(num_upsamples):
            self.upsample_blocks.append(
                nn.Sequential(
                    nn.Conv2d(nf, nf * 4, 3, 1, 1),
                    nn.PixelShuffle(2),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
        
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        for block in self.upsample_blocks:
            feat = block(feat)
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


# --- Discriminator Architecture (Unchanged) ---
class UNetDiscriminator(nn.Module):
    def __init__(self, in_nc=3, nf=64):
        super(UNetDiscriminator, self).__init__()
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

print("✅ SECTION 4: NOVEL MODEL ARCHITECTURE COMPLETE (FIX APPLIED)")
print("   - Generator: HybridSwinNet (Swin-Transformer + CNN)")
print("   - Discriminator: UNetDiscriminator (Unchanged)")
print("-" * 80)

# =================================================================================================
# SECTION 5: <<< NOVEL CONTRIBUTION >>> LOSS FUNCTIONS AND UTILITIES
# =================================================================================================

# --- Novel Perceptual Loss using DINO ViT ---
class DINOPerceptualLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        print("[LOSS] Initializing DINO Perceptual Loss...")
        # dino_vits16 is a good balance of speed and feature quality
        self.dino = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', force_reload=True).to(device)
        self.dino.eval()
        for param in self.dino.parameters():
            param.requires_grad = False
        self.loss_fn = nn.L1Loss()
        # DINO expects specific normalization
        self.transform = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        print("[LOSS] DINOv1 ViT-S/16 loaded and frozen.")

    def forward(self, generated_img, target_img):
        # Apply DINO-specific normalization
        gen_norm = self.transform(generated_img)
        target_norm = self.transform(target_img)
        # Extract features (class token)
        dino_gen = self.dino.get_intermediate_layers(gen_norm, n=1)[0]
        dino_target = self.dino.get_intermediate_layers(target_norm, n=1)[0]
        return self.loss_fn(dino_gen, dino_target)

# --- Novel Frequency Domain (FFT) Loss ---
class FrequencyLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.loss_fn = nn.L1Loss().to(device)

    def forward(self, generated_img, target_img):
        gen_fft = torch.fft.fftn(generated_img, dim=(-2, -1))
        gen_fft_mag = torch.abs(gen_fft)
        
        target_fft = torch.fft.fftn(target_img, dim=(-2, -1))
        target_fft_mag = torch.abs(target_fft)

        return self.loss_fn(gen_fft_mag, target_fft_mag)

# --- Image Saving Utility (Unchanged) ---
def save_comparison_image(lr_tensor, sr_tensor, hr_tensor, epoch, config):
    lr_img, sr_img, hr_img = lr_tensor[0].cpu(), sr_tensor[0].cpu(), hr_tensor[0].cpu()
    lr_upscaled = F.interpolate(
        lr_img.unsqueeze(0),
        size=(config.HR_PATCH_SIZE, config.HR_PATCH_SIZE),
        mode='bicubic',
        align_corners=False
    ).squeeze(0)
    comparison_grid = vutils.make_grid([lr_upscaled, sr_img, hr_img], nrow=3, normalize=True, scale_each=True, pad_value=1)
    filepath = os.path.join(config.OUTPUT_DIR, "images", f"comparison_epoch_{epoch:03d}.png")
    vutils.save_image(comparison_grid, filepath)


print("✅ SECTION 5: NOVEL LOSS FUNCTIONS & UTILITIES SETUP COMPLETE")
print("   - Perceptual Loss: DINO ViT-S/16")
print("   - Detail Loss: Frequency-Domain (FFT) L1 Loss")
print("-" * 80)

# =================================================================================================
# SECTION 5.5: CHECKPOINT MANAGEMENT UTILITIES (Unchanged)
# =================================================================================================
def save_checkpoint(generator, discriminator, optimizer_g, optimizer_d, epoch, config):
    state = {'epoch': epoch, 'generator_state_dict': generator.state_dict(), 'discriminator_state_dict': discriminator.state_dict(), 'optimizer_g_state_dict': optimizer_g.state_dict(), 'optimizer_d_state_dict': optimizer_d.state_dict()}
    path = os.path.join(config.OUTPUT_DIR, "checkpoints", f"checkpoint_epoch_{epoch:03d}.pth")
    torch.save(state, path)
def load_checkpoint(checkpoint_path, generator, discriminator, optimizer_g, optimizer_d, config):
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
    optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
    return checkpoint['epoch']
def validate_resume_config(config):
    """
    Validates resume configuration and returns (start_epoch, checkpoint_path).
    Priority: RESUME_CHECKPOINT_PATH > RESUME_EPOCH > Latest checkpoint
    """
    if not config.RESUME_TRAINING: 
        return 1, None
    
    # Priority 1: Direct checkpoint path
    if config.RESUME_CHECKPOINT_PATH is not None:
        if os.path.exists(config.RESUME_CHECKPOINT_PATH):
            try:
                # Try to load checkpoint to validate it
                checkpoint = torch.load(config.RESUME_CHECKPOINT_PATH, map_location='cpu')
                epoch = checkpoint.get('epoch', 0)
                print(f"[RESUME] Found valid checkpoint at: {config.RESUME_CHECKPOINT_PATH}")
                return epoch + 1, config.RESUME_CHECKPOINT_PATH
            except Exception as e:
                print(f"ERROR: Failed to load checkpoint from {config.RESUME_CHECKPOINT_PATH}: {e}")
                return 1, None
        else:
            print(f"ERROR: Checkpoint file not found: {config.RESUME_CHECKPOINT_PATH}")
            return 1, None
    
    # Priority 2: Specific epoch in checkpoint directory
    checkpoint_dir = config.RESUME_CHECKPOINT_DIR or os.path.join(config.OUTPUT_DIR, "checkpoints")
    if config.RESUME_EPOCH is not None:
        path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{config.RESUME_EPOCH:03d}.pth")
        if os.path.exists(path):
            print(f"[RESUME] Found checkpoint for epoch {config.RESUME_EPOCH}")
            return config.RESUME_EPOCH + 1, path
        else:
            print(f"ERROR: No checkpoint found for epoch {config.RESUME_EPOCH} in {checkpoint_dir}")
            return 1, None
    
    # Priority 3: Latest checkpoint in directory
    if not os.path.exists(checkpoint_dir):
        print(f"ERROR: Checkpoint directory not found: {checkpoint_dir}")
        return 1, None
        
    files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth"))
    if not files:
        print(f"ERROR: No checkpoint files found in {checkpoint_dir}")
        return 1, None
        
    latest_file = max(files, key=os.path.getctime)
    try:
        latest_epoch = int(os.path.basename(latest_file).split('_')[2].split('.')[0])
        print(f"[RESUME] Found latest checkpoint: {latest_file} (epoch {latest_epoch})")
        return latest_epoch + 1, latest_file
    except Exception as e:
        print(f"ERROR: Failed to parse epoch from checkpoint filename: {latest_file}")
        return 1, None

def setup_resume_training(config, checkpoint_path=None, checkpoint_dir=None, epoch=None):
    """
    Helper function to easily configure resume training.
    
    Args:
        config: Configuration object
        checkpoint_path: Direct path to a specific checkpoint file (highest priority)
        checkpoint_dir: Directory containing checkpoints
        epoch: Specific epoch to resume from
    """
    config.RESUME_TRAINING = True
    if checkpoint_path:
        config.RESUME_CHECKPOINT_PATH = checkpoint_path
        print(f"[SETUP] Resume training configured with checkpoint: {checkpoint_path}")
    elif checkpoint_dir:
        config.RESUME_CHECKPOINT_DIR = checkpoint_dir
        if epoch:
            config.RESUME_EPOCH = epoch
            print(f"[SETUP] Resume training configured for epoch {epoch} in directory: {checkpoint_dir}")
        else:
            print(f"[SETUP] Resume training configured with latest checkpoint in directory: {checkpoint_dir}")
    else:
        print(f"[SETUP] Resume training configured with default settings")

def parse_command_line_args():
    """
    Parse command line arguments for resume training.
    Returns updated config or None if no resume args provided.
    """
    import argparse
    parser = argparse.ArgumentParser(description='DINO Ranger Training with Resume Support')
    parser.add_argument('--resume', action='store_true', help='Enable resume training')
    parser.add_argument('--checkpoint-path', type=str, help='Direct path to checkpoint file')
    parser.add_argument('--checkpoint-dir', type=str, help='Directory containing checkpoints')
    parser.add_argument('--epoch', type=int, help='Specific epoch to resume from')
    
    args = parser.parse_args()
    
    if args.resume or args.checkpoint_path or args.checkpoint_dir or args.epoch:
        return args
    return None

print("✅ SECTION 5.5: CHECKPOINT MANAGEMENT UTILITIES LOADED")
print("   - Enhanced resume training with direct checkpoint path support")
print("   - Command line argument support for resume options")
print("-" * 80)

# =================================================================================================
# SECTION 6: <<< MODIFIED FOR NOVELTY >>> TRAINING ORCHESTRATION
# =================================================================================================
def train():
    print("[TRAIN] Starting training orchestration for NOVEL model...")

    # --- 1. Initialize Dataloader ---
    dataset = DF2KDataset(config)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True)
    print(f"[TRAIN] DataLoader created with {len(dataloader)} batches per epoch.")

    # --- 2. Initialize Models ---
    # <<< MODIFIED FOR NOVELTY >>> Instantiate the HybridSwinNet
    generator = HybridSwinNet(
        scale=config.SCALE_FACTOR,
        input_res=(config.LR_PATCH_SIZE, config.LR_PATCH_SIZE)
    ).to(config.DEVICE)
    discriminator = UNetDiscriminator().to(config.DEVICE)
    print("[TRAIN] Novel models initialized (HybridSwinNet, UNetDiscriminator).")

    # --- 3. Initialize Optimizers ---
    optimizer_g = optim.Adam(generator.parameters(), lr=config.LR_G, betas=(config.BETA1, config.BETA2))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=config.LR_D, betas=(config.BETA1, config.BETA2))

    # --- 4. Initialize Loss Functions ---
    # <<< MODIFIED FOR NOVELTY >>> Use DINO and add FFT loss
    l1_loss = nn.L1Loss().to(config.DEVICE)
    perceptual_loss = DINOPerceptualLoss(device=config.DEVICE)
    frequency_loss = FrequencyLoss(device=config.DEVICE)
    adversarial_loss = nn.BCEWithLogitsLoss().to(config.DEVICE)
    print("[TRAIN] Novel loss functions initialized (DINO, FFT, L1, GAN).")

    # --- 5. Handle Resume Training ---
    start_epoch, checkpoint_path = validate_resume_config(config)
    if checkpoint_path:
        loaded_epoch = load_checkpoint(checkpoint_path, generator, discriminator, optimizer_g, optimizer_d, config)
        print(f"[RESUME] Checkpoint loaded. Resuming from epoch {loaded_epoch + 1}")
        start_epoch = loaded_epoch + 1
    else:
        print("[TRAIN] Starting fresh training from epoch 1.")

    # --- 6. The Main Training Loop ---
    print("\n" + "="*30 + " STARTING NOVEL TRAINING " + "="*30)
    for epoch in range(start_epoch, config.NUM_EPOCHS + 1):
        generator.train()
        discriminator.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{config.NUM_EPOCHS}", leave=True)

        for lr_imgs, hr_imgs in pbar:
            lr_imgs = lr_imgs.to(config.DEVICE)
            hr_imgs = hr_imgs.to(config.DEVICE)

            # --- Train Discriminator ---
            optimizer_d.zero_grad()
            fake_sr_imgs = generator(lr_imgs).detach()
            real_pred, fake_pred = discriminator(hr_imgs), discriminator(fake_sr_imgs)
            loss_d = (adversarial_loss(real_pred, torch.ones_like(real_pred)) + adversarial_loss(fake_pred, torch.zeros_like(fake_pred))) / 2
            loss_d.backward()
            optimizer_d.step()

            # --- Train Generator ---
            optimizer_g.zero_grad()
            fake_sr_imgs = generator(lr_imgs)
            fake_pred_g = discriminator(fake_sr_imgs)

            # <<< MODIFIED FOR NOVELTY >>> Calculate all generator losses including FFT
            loss_g_l1 = config.W_L1 * l1_loss(fake_sr_imgs, hr_imgs)
            loss_g_perceptual = config.W_PERCEPTUAL * perceptual_loss(fake_sr_imgs, hr_imgs)
            loss_g_gan = config.W_GAN * adversarial_loss(fake_pred_g, torch.ones_like(fake_pred_g))
            loss_g_fft = config.W_FFT * frequency_loss(fake_sr_imgs, hr_imgs)
            
            loss_g = loss_g_l1 + loss_g_perceptual + loss_g_gan + loss_g_fft
            loss_g.backward()
            optimizer_g.step()

            # --- Logging with TQDM ---
            pbar.set_postfix(
                G_Loss=f"{loss_g.item():.4f}", D_Loss=f"{loss_d.item():.4f}",
                G_L1=f"{loss_g_l1.item():.4f}", G_DINO=f"{loss_g_perceptual.item():.4f}",
                G_GAN=f"{loss_g_gan.item():.4f}", G_FFT=f"{loss_g_fft.item():.4f}" # <-- New loss term
            )

        # --- End of Epoch Saving ---
        if epoch % config.SAVE_EVERY_N_EPOCHS == 0:
            save_checkpoint(generator, discriminator, optimizer_g, optimizer_d, epoch, config)
            generator.eval()
            with torch.no_grad():
                save_comparison_image(lr_imgs, generator(lr_imgs), hr_imgs, epoch, config)
            generator.train()
            tqdm.write(f"\n✅ Checkpoint and comparison image saved for epoch {epoch}")

    print("\n" + "="*30 + " NOVEL TRAINING FINISHED " + "="*30)

print("✅ SECTION 6: TRAINING ORCHESTRATION COMPLETE (MODIFIED FOR NOVELTY)")
print("-" * 80)

# =================================================================================================
# SECTION 7: MAIN EXECUTION BLOCK
# =================================================================================================

if __name__ == '__main__':
    print("✅ SECTION 7: MAIN EXECUTION BLOCK")
    print("Initiating the training process for the novel model...")

    if not os.path.exists(config.DATASET_PATH) or not os.path.exists(os.path.join(config.DATASET_PATH, config.HR_FOLDER)):
        print("\n" + "!"*80 + "\n! ERROR: Dataset path not found. Please check `DATASET_PATH` in Config.\n" + "!"*80)
        # In a real script, you might exit here. For this interactive environment, we'll proceed.
    
    # First-time run of DINO model will download weights.
    print("\nNOTE: The first time you run this, PyTorch Hub will download the DINO model weights.")
    print("This may take a moment and requires an internet connection.\n")
    # Added a try-except block to gracefully handle potential environment issues (like no dataset).
    try:
        train()
    except Exception as e:
        print(f"\nAN ERROR OCCURRED DURING TRAINING: {e}")
        import traceback
        traceback.print_exc()