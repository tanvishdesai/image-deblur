# =================================================================================================
# DINO Ranger with Degradation (Corrected & Verified)
#
# This script combines the best features of both original files:
# 1. Generator/Discriminator: The robust TransformerESRGAN and MultiScaleDiscriminator from `degradation.py`.
# 2. Degradation: The NovelDegradationPipeline for realistic on-the-fly image degradation.
# 3. Losses: The advanced DINO Perceptual Loss and Frequency (FFT) Loss from `DINO Ranger.py`.
# =================================================================================================


# =================================================================================================
# SECTION 1: IMPORTS AND INITIAL SETUP
# =================================================================================================
# --- Standard Library Imports ---
import os
import glob
import random
import time
import io

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

# --- Additional imports for Novel Pipeline & Architecture ---
import cv2
from scipy import ndimage
from scipy.ndimage import gaussian_filter

print("✅ SECTION 1: IMPORTS AND INITIAL SETUP COMPLETE")
print("-" * 80)


# =================================================================================================
# SECTION 2: CONFIGURATION
# =================================================================================================
class Config:
    # --- Dataset and Paths ---
    DATASET_PATH = "/kaggle/input/df2kdata"
    OUTPUT_DIR = "training_outputs_dino_degradation"
    SCALE_FACTOR = 4
    HR_FOLDER = "DF2K_train_HR"

    # --- Resume Training Options ---
    RESUME_TRAINING = True
    RESUME_CHECKPOINT_DIR = "/kaggle/input/proposed-merge-80-epoch/training_outputs_dino_degradation/checkpoints"
    RESUME_EPOCH = 80

    # --- Training Parameters ---
    NUM_EPOCHS = 180
    BATCH_SIZE = 4
    LR_PATCH_SIZE = 64
    HR_PATCH_SIZE = LR_PATCH_SIZE * SCALE_FACTOR

    # --- Optimizer Parameters ---
    LR_G = 1e-4
    LR_D = 1e-4
    BETA1 = 0.9
    BETA2 = 0.999

    # --- Loss Function Weights ---
    W_L1 = 1.0
    W_PERCEPTUAL = 1.0  # DINO loss weight
    W_GAN = 0.1
    W_FFT = 0.8         # Frequency loss weight

    # --- Transformer Generator Parameters ---
    TRANSFORMER_NUM_FEAT = 96
    TRANSFORMER_NUM_BLOCK = 6
    TRANSFORMER_NUM_HEAD = 6
    TRANSFORMER_WINDOW_SIZE = 8

    # --- Hardware and Logging ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
    SAVE_EVERY_N_EPOCHS = 2

config = Config()

os.makedirs(os.path.join(config.OUTPUT_DIR, "checkpoints"), exist_ok=True)
os.makedirs(os.path.join(config.OUTPUT_DIR, "images"), exist_ok=True)

print("✅ SECTION 2: CONFIGURATION COMPLETE")
print(f"   - Device: {config.DEVICE}")
print(f"   - Generator: TransformerESRGAN")
print(f"   - Discriminator: MultiScaleDiscriminator")
print(f"   - Degradation: On-the-fly Novel Degradation Pipeline")
print(f"   - Losses: L1 + DINO Perceptual + FFT + GAN")
print(f"   - Output Directory: {config.OUTPUT_DIR}")
print("-" * 80)


# =================================================================================================
# SECTION 2.5: NOVEL DEGRADATION PIPELINE
# =================================================================================================
class NovelDegradationPipeline:
    def __init__(self, scale_factor=4):
        self.scale_factor = scale_factor

    def apply_chromatic_aberration(self, img_array):
        h, w, c = img_array.shape
        if c != 3: return img_array
        shift_r = np.random.uniform(-1.5, 1.5)
        shift_b = np.random.uniform(-1.5, 1.5)
        img_shifted = img_array.copy()
        if shift_r != 0: img_shifted[:, :, 0] = ndimage.shift(img_array[:, :, 0], shift_r, mode='nearest')
        if shift_b != 0: img_shifted[:, :, 2] = ndimage.shift(img_array[:, :, 2], shift_b, mode='nearest')
        return img_shifted

    def apply_sensor_noise(self, img_array):
        img_float = img_array.astype(np.float32) / 255.0
        if np.random.random() < 0.7:
            shot_noise_scale = np.random.uniform(0.01, 0.05)
            img_float = np.random.poisson(img_float / shot_noise_scale) * shot_noise_scale
        if np.random.random() < 0.8:
            read_noise_std = np.random.uniform(0.005, 0.02)
            img_float += np.random.normal(0, read_noise_std, img_float.shape)
        if np.random.random() < 0.3:
            dark_current = np.random.uniform(0.001, 0.008)
            img_float += dark_current
        return np.clip(img_float * 255.0, 0, 255).astype(np.uint8)

    def apply_motion_blur(self, img_array):
        if np.random.random() < 0.4:
            if np.random.choice(['linear', 'rotational']) == 'linear':
                kernel_size = np.random.randint(5, 15)
                angle = np.random.uniform(0, 180)
                kernel = np.zeros((kernel_size, kernel_size))
                center = kernel_size // 2
                for i in range(kernel_size):
                    x = int(center + (i - center) * np.cos(np.radians(angle)))
                    y = int(center + (i - center) * np.sin(np.radians(angle)))
                    if 0 <= x < kernel_size and 0 <= y < kernel_size: kernel[y, x] = 1
                kernel /= np.sum(kernel)
                blurred = img_array.copy()
                for c in range(img_array.shape[2]):
                    blurred[:, :, c] = cv2.filter2D(img_array[:, :, c], -1, kernel)
                return blurred
            else:
                sigma = np.random.uniform(0.5, 2.0)
                return gaussian_filter(img_array, sigma=sigma)
        return img_array

    def apply_jpeg_compression(self, img_array):
        if np.random.random() < 0.6:
            quality = np.random.randint(30, 85)
            pil_img = Image.fromarray(img_array)
            buffer = io.BytesIO()
            pil_img.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            return np.array(Image.open(buffer))
        return img_array

    def degrade_image(self, hr_pil_image):
        hr_array = np.array(hr_pil_image)
        degradations = [self.apply_chromatic_aberration, self.apply_sensor_noise, self.apply_motion_blur]
        np.random.shuffle(degradations)
        degraded_array = hr_array.copy()
        for degradation_func in degradations:
            degraded_array = degradation_func(degraded_array)
        degraded_array = self.apply_jpeg_compression(degraded_array)
        degraded_pil = Image.fromarray(degraded_array)
        lr_size = (hr_pil_image.size[0] // self.scale_factor, hr_pil_image.size[1] // self.scale_factor)
        interpolation = np.random.choice([Image.BICUBIC, Image.BILINEAR, Image.LANCZOS])
        return degraded_pil.resize(lr_size, interpolation)

print("✅ SECTION 2.5: NOVEL DEGRADATION PIPELINE COMPLETE")
print("-" * 80)


# =================================================================================================
# SECTION 3: DATASET AND DATALOADER
# =================================================================================================
class DF2KDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.hr_path = os.path.join(config.DATASET_PATH, config.HR_FOLDER)
        self.hr_image_files = sorted(glob.glob(os.path.join(self.hr_path, "*.png")))
        self.degradation_pipeline = NovelDegradationPipeline(scale_factor=config.SCALE_FACTOR)
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.hr_image_files)

    def __getitem__(self, index):
        hr_image = Image.open(self.hr_image_files[index]).convert("RGB")
        hr_patch = self.get_random_hr_patch(hr_image)
        lr_patch = self.degradation_pipeline.degrade_image(hr_patch)
        lr_patch, hr_patch = self.augment(lr_patch, hr_patch)
        return self.to_tensor(lr_patch), self.to_tensor(hr_patch)

    def get_random_hr_patch(self, hr_img):
        hr_w, hr_h = hr_img.size
        hr_patch_size = self.config.HR_PATCH_SIZE
        hr_x = random.randint(0, hr_w - hr_patch_size)
        hr_y = random.randint(0, hr_h - hr_patch_size)
        return hr_img.crop((hr_x, hr_y, hr_x + hr_patch_size, hr_y + hr_patch_size))

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
print("   - Using NovelDegradationPipeline for on-the-fly data generation.")
print("-" * 80)


# =================================================================================================
# SECTION 4: MODEL ARCHITECTURE (TransformerESRGAN)
# =================================================================================================
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim, self.window_size, self.num_heads = dim, window_size, num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))
        coords = torch.stack(torch.meshgrid([torch.arange(window_size), torch.arange(window_size)]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1))
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(N, N, -1).permute(2, 0, 1).contiguous()
        attn += relative_position_bias.unsqueeze(0)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(x)

def window_partition(x, window_size):
    B, H, W, C = x.shape
    return x.view(B, H // window_size, window_size, W // window_size, window_size, C).permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    return windows.view(B, H // window_size, W // window_size, window_size, window_size, -1).permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, shift_size=0, mlp_ratio=4.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim))
        self.window_size, self.shift_size = window_size, shift_size

    def forward(self, x, H, W):
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)
        shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)) if self.shift_size > 0 else x
        x_windows = window_partition(shifted_x, self.window_size).view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows)
        shifted_x = window_reverse(attn_windows.view(-1, self.window_size, self.window_size, C), self.window_size, H, W)
        x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)) if self.shift_size > 0 else shifted_x
        x = (shortcut + x.view(B, H * W, C)) + self.mlp(self.norm2(x.view(B, H * W, C)))
        return x

class RSTB(nn.Module):
    def __init__(self, dim, num_heads, window_size, depth=6, mlp_ratio=2.):
        super(RSTB, self).__init__()
        self.blocks = nn.ModuleList([SwinTransformerBlock(dim, num_heads, window_size, 0 if (i % 2 == 0) else window_size // 2, mlp_ratio) for i in range(depth)])
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        res = x
        x = x.permute(0, 2, 3, 1).view(B, H * W, C)
        for blk in self.blocks:
            x = blk(x, H, W)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        return self.conv(x) + res

class TransformerESRGAN(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, num_feat=96, num_block=6, num_head=6, window_size=8, scale=4):
        super(TransformerESRGAN, self).__init__()
        self.conv_first = nn.Conv2d(in_nc, num_feat, 3, 1, 1)
        self.body = nn.ModuleList([RSTB(num_feat, num_head, window_size) for _ in range(num_block)])
        self.conv_after_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.upsample = nn.Sequential(*[
            nn.Sequential(nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1), nn.PixelShuffle(2), nn.LeakyReLU(0.2, True))
            for _ in range(int(np.log2(scale)))
        ])
        self.conv_last = nn.Sequential(nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.LeakyReLU(0.2, True), nn.Conv2d(num_feat, out_nc, 3, 1, 1))

    def forward(self, x):
        res = self.conv_first(x)
        x = res
        for block in self.body:
            x = block(x)
        x = self.conv_after_body(x) + res
        x = self.upsample(x)
        return self.conv_last(x)

class MultiScaleDiscriminator(nn.Module):
    def __init__(self, in_nc=3, nf=64, num_scales=3):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([self._make_discriminator(in_nc, nf) for _ in range(num_scales)])
        self.downsample = nn.AvgPool2d(2)
    def _make_discriminator(self, in_nc, nf):
        return nn.Sequential(
            nn.Conv2d(in_nc, nf, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(nf * 2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf * 2, nf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(nf * 4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf * 4, nf * 8, 4, 1, 1, bias=False), nn.BatchNorm2d(nf * 8), nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf * 8, 1, 4, 1, 1)
        )
    def forward(self, x):
        outputs = []
        for disc in self.discriminators:
            outputs.append(disc(x))
            x = self.downsample(x)
        return outputs

print("✅ SECTION 4: ROBUST MODEL ARCHITECTURE COMPLETE")
print("   - Generator: TransformerESRGAN (from degradation.py)")
print("   - Discriminator: MultiScaleDiscriminator (from degradation.py)")
print("-" * 80)


# =================================================================================================
# SECTION 5: NOVEL LOSS FUNCTIONS (DINO + FFT)
# =================================================================================================
class DINOPerceptualLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        print("[LOSS] Initializing DINO Perceptual Loss...")
        self.dino = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', force_reload=True).to(device)
        self.dino.eval()
        for param in self.dino.parameters(): param.requires_grad = False
        self.loss_fn = nn.L1Loss()
        self.transform = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        print("[LOSS] DINOv1 ViT-S/16 loaded and frozen.")

    def forward(self, generated_img, target_img):
        dino_gen = self.dino.get_intermediate_layers(self.transform(generated_img), n=1)[0]
        dino_target = self.dino.get_intermediate_layers(self.transform(target_img), n=1)[0]
        return self.loss_fn(dino_gen, dino_target)

class FrequencyLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.loss_fn = nn.L1Loss().to(device)
    def forward(self, gen_img, target_img):
        return self.loss_fn(torch.abs(torch.fft.fftn(gen_img, dim=(-2, -1))), torch.abs(torch.fft.fftn(target_img, dim=(-2, -1))))

def save_comparison_image(lr, sr, hr, epoch, cfg):
    grid = vutils.make_grid([F.interpolate(lr[0].cpu().unsqueeze(0), size=sr.shape[-2:], mode='bicubic', align_corners=False).squeeze(0), sr[0].cpu(), hr[0].cpu()], nrow=3, normalize=True, pad_value=1)
    vutils.save_image(grid, os.path.join(cfg.OUTPUT_DIR, "images", f"comparison_epoch_{epoch:03d}.png"))

print("✅ SECTION 5: NOVEL LOSS FUNCTIONS & UTILITIES SETUP COMPLETE")
print("-" * 80)


# =================================================================================================
# SECTION 6: TRAINING ORCHESTRATION
# =================================================================================================
def train():
    print("[TRAIN] Starting training orchestration...")
    dataset = DF2KDataset(config)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True)

    generator = TransformerESRGAN(
        num_feat=config.TRANSFORMER_NUM_FEAT, num_block=config.TRANSFORMER_NUM_BLOCK,
        num_head=config.TRANSFORMER_NUM_HEAD, window_size=config.TRANSFORMER_WINDOW_SIZE, scale=config.SCALE_FACTOR
    ).to(config.DEVICE)
    discriminator = MultiScaleDiscriminator().to(config.DEVICE)

    optimizer_g = optim.Adam(generator.parameters(), lr=config.LR_G, betas=(config.BETA1, config.BETA2))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=config.LR_D, betas=(config.BETA1, config.BETA2))

    l1_loss = nn.L1Loss().to(config.DEVICE)
    perceptual_loss = DINOPerceptualLoss(device=config.DEVICE)
    frequency_loss = FrequencyLoss(device=config.DEVICE)
    adversarial_loss = nn.BCEWithLogitsLoss().to(config.DEVICE)

    # Resume training if specified
    start_epoch = 1
    if config.RESUME_TRAINING:
        print(f"[RESUME] Attempting to resume training from epoch {config.RESUME_EPOCH}...")
        generator_checkpoint = os.path.join(config.RESUME_CHECKPOINT_DIR, f"generator_epoch_{config.RESUME_EPOCH:03d}.pth")
        discriminator_checkpoint = os.path.join(config.RESUME_CHECKPOINT_DIR, f"discriminator_epoch_{config.RESUME_EPOCH:03d}.pth")
        
        if os.path.exists(generator_checkpoint) and os.path.exists(discriminator_checkpoint):
            print(f"[RESUME] Loading generator from: {generator_checkpoint}")
            generator.load_state_dict(torch.load(generator_checkpoint, map_location=config.DEVICE))
            print(f"[RESUME] Loading discriminator from: {discriminator_checkpoint}")
            discriminator.load_state_dict(torch.load(discriminator_checkpoint, map_location=config.DEVICE))
            start_epoch = config.RESUME_EPOCH + 1
            print(f"[RESUME] ✅ Successfully resumed from epoch {config.RESUME_EPOCH}. Starting from epoch {start_epoch}")
        else:
            print(f"[RESUME] ⚠️  Checkpoint files not found. Starting from epoch 1.")
            print(f"[RESUME] Generator checkpoint: {generator_checkpoint}")
            print(f"[RESUME] Discriminator checkpoint: {discriminator_checkpoint}")

    print("\n" + "="*30 + " STARTING TRAINING " + "="*30)
    for epoch in range(start_epoch, config.NUM_EPOCHS + 1):
        generator.train()
        discriminator.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{config.NUM_EPOCHS}", leave=True)

        for lr_imgs, hr_imgs in pbar:
            lr_imgs, hr_imgs = lr_imgs.to(config.DEVICE), hr_imgs.to(config.DEVICE)

            # --- Train Discriminator ---
            optimizer_d.zero_grad()
            fake_sr_imgs = generator(lr_imgs).detach()
            real_preds, fake_preds = discriminator(hr_imgs), discriminator(fake_sr_imgs)
            loss_d_real, loss_d_fake = 0, 0
            for real_p, fake_p in zip(real_preds, fake_preds):
                loss_d_real += adversarial_loss(real_p, torch.ones_like(real_p))
                loss_d_fake += adversarial_loss(fake_p, torch.zeros_like(fake_p))
            loss_d = (loss_d_real + loss_d_fake) / len(real_preds)
            loss_d.backward()
            optimizer_d.step()

            # --- Train Generator ---
            optimizer_g.zero_grad()
            fake_sr_imgs = generator(lr_imgs)
            fake_preds_g = discriminator(fake_sr_imgs)
            loss_g_l1 = config.W_L1 * l1_loss(fake_sr_imgs, hr_imgs)
            loss_g_perceptual = config.W_PERCEPTUAL * perceptual_loss(fake_sr_imgs, hr_imgs)
            loss_g_fft = config.W_FFT * frequency_loss(fake_sr_imgs, hr_imgs)
            loss_g_gan = 0
            for fake_p in fake_preds_g:
                loss_g_gan += adversarial_loss(fake_p, torch.ones_like(fake_p))
            loss_g_gan = config.W_GAN * loss_g_gan / len(fake_preds_g)
            loss_g = loss_g_l1 + loss_g_perceptual + loss_g_gan + loss_g_fft
            loss_g.backward()
            optimizer_g.step()

            pbar.set_postfix(G_Loss=f"{loss_g.item():.4f}", D_Loss=f"{loss_d.item():.4f}", G_DINO=f"{loss_g_perceptual.item():.4f}", G_FFT=f"{loss_g_fft.item():.4f}")

        if epoch % config.SAVE_EVERY_N_EPOCHS == 0:
            torch.save(generator.state_dict(), os.path.join(config.OUTPUT_DIR, "checkpoints", f"generator_epoch_{epoch:03d}.pth"))
            torch.save(discriminator.state_dict(), os.path.join(config.OUTPUT_DIR, "checkpoints", f"discriminator_epoch_{epoch:03d}.pth"))
            generator.eval()
            with torch.no_grad():
                save_comparison_image(lr_imgs, generator(lr_imgs), hr_imgs, epoch, config)
            tqdm.write(f"\n✅ Checkpoint and comparison image saved for epoch {epoch}")

    print("\n" + "="*30 + " TRAINING FINISHED " + "="*30)

print("✅ SECTION 6: TRAINING ORCHESTRATION COMPLETE")
print("-" * 80)


# =================================================================================================
# SECTION 7: MAIN EXECUTION BLOCK
# =================================================================================================
if __name__ == '__main__':
    print("✅ SECTION 7: MAIN EXECUTION BLOCK")
    print("Initiating the training process...")
    if not os.path.exists(config.DATASET_PATH):
        print("\n" + "!"*80 + "\n! ERROR: Dataset path not found. Please check `DATASET_PATH` in Config.\n" + "!"*80)
    else:
        try:
            train()
        except Exception as e:
            print(f"\nAN ERROR OCCURRED DURING TRAINING: {e}")
            import traceback
            traceback.print_exc()