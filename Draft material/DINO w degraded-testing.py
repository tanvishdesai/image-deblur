import os
import glob
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms as T
from einops import rearrange
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips
import pyiqa
import warnings
from tqdm import tqdm
import torch.nn.functional as F

# --- Suppress specific warnings ---
warnings.filterwarnings("ignore", category=UserWarning, message="The given NumPy array is not writeable")

# =================================================================================================
# SECTION 1: SETUP AND INSTRUCTIONS
# =================================================================================================
# This script is designed for testing the HybridSwinNet SISR model trained with 'proposed_solution.py'.
#
# To run this script, you need to install the following packages:
# pip install torch torchvision torchaudio
# pip install numpy pillow scikit-image tqdm einops
# pip install lpips
# pip install pyiqa
#
# Place this script in the same directory as your trained checkpoint file
# and the dataset folders (BSD100, Set5, Set14, Urban100).
# =================================================================================================


# =================================================================================================
# SECTION 2: MODEL ARCHITECTURE (Copied from proposed_solution.py)
# =================================================================================================

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
        if self.attn_mask is not None:
            num_windows = self.attn_mask.shape[0]
            B_attn = x_windows.shape[0] // num_windows
            batch_mask = self.attn_mask.repeat(B_attn, 1, 1)
            final_attn_mask = batch_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            final_attn_mask = final_attn_mask.view(-1, self.window_size * self.window_size, self.window_size * self.window_size)
            attn_windows, _ = self.attn(x_windows, x_windows, x_windows, attn_mask=final_attn_mask)
        else:
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

# --- The main generator network (HybridSwinNet from proposed_solution.py) ---
class HybridSwinNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, scale=4, input_res=(64,64)):
        super(HybridSwinNet, self).__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
        
        self.body = nn.Sequential(*[HybridBlock(dim=nf, input_resolution=input_res) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)

        # Upsampling blocks
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

    def forward(self, x, target_scale=None):
        if target_scale is None:
            target_scale = self.scale

        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        
        # --- Multi-scale Upsampling Logic (adapted for dynamic testing) ---
        if target_scale == 2:
            # Assumes base scale >= 2
            feat = self.upsample_blocks[0](feat)
        elif target_scale == 3:
            # Assumes base scale >= 2, upsample by 2x then interpolate
            feat = self.upsample_blocks[0](feat)
            feat = F.interpolate(feat, scale_factor=1.5, mode='bicubic', align_corners=False)
        elif target_scale == 4:
            # Assumes base scale == 4
            for block in self.upsample_blocks:
                feat = block(feat)
        else:
            # Fallback for any other scale
            feat = F.interpolate(feat, scale_factor=target_scale, mode='bicubic', align_corners=False)

        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out

# =================================================================================================
# SECTION 3: CONFIGURATION FOR TESTING
# =================================================================================================
class TestConfig:
    # --- IMPORTANT: Path to the checkpoint file from 'proposed_solution.py' ---
    CHECKPOINT_PATH = "/kaggle/input/dino-ranger-w-degrade/training_outputs_novel/checkpoints/checkpoint_epoch_514.pth" # <<< UPDATE THIS PATH if needed
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # SCALE_FACTOR is now determined dynamically from the dataset folders.

    # --- HybridSwinNet Parameters (must match the trained model from proposed_solution.py) ---
    HYBRID_NUM_FEAT = 64
    HYBRID_NUM_BLOCK = 16
    HYBRID_NUM_HEAD = 8
    HYBRID_WINDOW_SIZE = 8
    INFERENCE_PATCH_SIZE = 64 # The model was trained on 64x64 patches

    # Datasets to evaluate (using Kaggle paths as requested)
    DATASETS = {
        "Set5": "/kaggle/input/sisr-testing-use-ssim-psnr-lpips-niqe/Set5/Set5",
        "Set14": "/kaggle/input/sisr-testing-use-ssim-psnr-lpips-niqe/Set14/Set14",
        "BSD100": "/kaggle/input/sisr-testing-use-ssim-psnr-lpips-niqe/BSD100/BSD100",
        "Urban100": "/kaggle/input/sisr-testing-use-ssim-psnr-lpips-niqe/Urban100/Urban100",
    }
    
    # The model was trained with 64x64 LR patches, but for testing we use full images.
    # We still need the input_res for the Swin blocks, but it's dynamic now.
    # We'll handle this during model initialization.

# =================================================================================================
# SECTION 4: METRIC CALCULATION AND UTILITIES
# =================================================================================================
def rgb2ycbcr(img_np):
    """Converts a numpy array RGB image to YCbCr (Y channel)."""
    y = np.dot(img_np, [65.481, 128.553, 24.966]) + 16.0
    return y / 255.0

def calculate_psnr_ssim(sr_np, hr_np, scale):
    """Calculates PSNR and SSIM on the Y channel of YCbCr color space."""
    # Shave border
    shave_size = scale
    hr_shaved = hr_np[shave_size:-shave_size, shave_size:-shave_size]
    sr_shaved = sr_np[shave_size:-shave_size, shave_size:-shave_size]

    # Convert to Y channel
    hr_y = rgb2ycbcr(hr_shaved / 255.0)
    sr_y = rgb2ycbcr(sr_shaved / 255.0)
    
    psnr = peak_signal_noise_ratio(hr_y, sr_y, data_range=1.0)
    ssim = structural_similarity(hr_y, sr_y, data_range=1.0, win_size=11, gaussian_weights=True)
    
    return psnr, ssim

def tensor_to_numpy(tensor):
    """Converts a CHW tensor [0,1] to HWC numpy array [0,255]."""
    return tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255.0

def sliding_window_inference(model, lr_tensor, patch_size=64, overlap=32, scale=4):
    """
    Performs sliding-window inference on a low-resolution tensor.
    This function is used to handle images of arbitrary sizes with models
    that are trained on fixed-size patches.
    """
    b, c, h, w = lr_tensor.shape
    sr_h, sr_w = h * scale, w * scale
    
    # Create tensors to store the summed output and a count for averaging
    sr_sum = torch.zeros((b, c, sr_h, sr_w), device=lr_tensor.device)
    sr_count = torch.zeros_like(sr_sum)
    
    stride = patch_size - overlap
    
    # Generate patch coordinates, ensuring the whole image is covered
    h_steps = list(range(0, h - patch_size + 1, stride))
    if h_steps[-1] != h - patch_size:
        h_steps.append(h - patch_size)

    w_steps = list(range(0, w - patch_size + 1, stride))
    if w_steps[-1] != w - patch_size:
        w_steps.append(w - patch_size)

    for h_i in h_steps:
        for w_i in w_steps:
            lr_patch = lr_tensor[:, :, h_i:h_i+patch_size, w_i:w_i+patch_size]
            
            with torch.no_grad():
                # Pass the target scale to the model's forward pass
                sr_patch = model(lr_patch, target_scale=scale)
            
            # Add the super-resolved patch to the summation tensor
            sr_h_start, sr_w_start = h_i * scale, w_i * scale
            sr_h_end, sr_w_end = sr_h_start + sr_patch.shape[2], sr_w_start + sr_patch.shape[3]
            
            sr_sum[:, :, sr_h_start:sr_h_end, sr_w_start:sr_w_end] += sr_patch
            sr_count[:, :, sr_h_start:sr_h_end, sr_w_start:sr_w_end] += 1
            
    # Average the results in overlapping regions
    sr_img = sr_sum / sr_count
    return sr_img.clamp(0, 1)

# =================================================================================================
# SECTION 5: MAIN EVALUATION SCRIPT
# =================================================================================================
def main():
    print("--- Starting SISR Model Evaluation ---")
    config = TestConfig()
    
    # --- Create directory for comparison images ---
    os.makedirs("top_psnr_results", exist_ok=True)

    # --- 1. Load Model ---
    if not os.path.exists(config.CHECKPOINT_PATH):
        print(f"FATAL: Checkpoint file not found at '{config.CHECKPOINT_PATH}'")
        return

    print(f"Loading generator model from: {config.CHECKPOINT_PATH}")
    
    # Always initialize the generator with parameters matching the checkpoint's architecture
    generator = HybridSwinNet(
        nf=config.HYBRID_NUM_FEAT,
        nb=config.HYBRID_NUM_BLOCK,
        scale=4, # Base scale is 4x
        input_res=(config.INFERENCE_PATCH_SIZE, config.INFERENCE_PATCH_SIZE)
    ).to(config.DEVICE)
    
    try:
        # Proposed solution 4 saves a comprehensive checkpoint dictionary.
        # We need to extract the generator's state dictionary from it.
        checkpoint = torch.load(config.CHECKPOINT_PATH, map_location=config.DEVICE)
        
        # Check if the checkpoint is the new comprehensive format or the old one
        if 'generator_state_dict' in checkpoint:
            # New format from proposed_solution_4.py
            generator.load_state_dict(checkpoint['generator_state_dict'])
        elif 'model_state_dict' in checkpoint:
             # Standard PyTorch checkpoint format
            generator.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Assume the checkpoint is just the generator's state_dict
            generator.load_state_dict(checkpoint)

    except Exception as e:
         print(f"FATAL: Could not load checkpoint. Error: {e}")
         print("Please ensure the checkpoint file is valid and its architecture matches the model.")
         return

    generator.eval()
    print("Model loaded successfully.")

    # --- 2. Initialize Metric Functions ---
    print("Initializing metric functions (LPIPS, NIQE)...")
    lpips_fn = lpips.LPIPS(net='alex').to(config.DEVICE)
    niqe_fn = pyiqa.create_metric('niqe', device=config.DEVICE)
    to_tensor = T.ToTensor()

    # --- 3. Evaluation Loop ---
    all_results = {}
    for name, path in config.DATASETS.items():
        print(f"\n--- Evaluating Dataset: {name} ---")
        
        # Dynamically find available scales (SRF_2, SRF_3, etc.)
        scale_folders = glob.glob(os.path.join(path, 'image_SRF_*'))
        if not scale_folders:
            print(f"Warning: No 'image_SRF_*' folders found in {path}. Skipping.")
            continue

        for lr_folder in sorted(scale_folders):
            try:
                scale = int(lr_folder.split('_')[-1])
            except (ValueError, IndexError):
                tqdm.write(f"Warning: Could not determine scale from folder name '{os.path.basename(lr_folder)}'. Skipping.")
                continue

            print(f"\n-- Testing Scale: {scale}x --")
            if scale == 3:
                print("   (Note: Using 2x upsampler + 1.5x bicubic interpolation for 3x scale)")

            lr_image_paths = sorted(glob.glob(os.path.join(lr_folder, '*_LR.png')))
            if not lr_image_paths:
                lr_image_paths = sorted(glob.glob(os.path.join(lr_folder, '*_LR.*'))) # Check for other extensions
            
            if not lr_image_paths:
                print(f"Warning: No LR images found in {lr_folder}. Skipping scale {scale}x.")
                continue

            total_metrics = {'psnr': 0.0, 'ssim': 0.0, 'lpips': 0.0, 'niqe': 0.0}
            num_images = 0
            top_samples = []
            
            for lr_path in tqdm(lr_image_paths, desc=f"Evaluating {name} (x{scale})"):
                hr_path = lr_path.replace('_LR', '_HR')
                if not os.path.exists(hr_path):
                     hr_path = hr_path.replace('.png','.bmp') # for some datasets
                     if not os.path.exists(hr_path):
                        tqdm.write(f"Warning: Could not find HR pair for {os.path.basename(lr_path)}. Skipping.")
                        continue
                
                num_images += 1

                # Load images
                lr_img_pil = Image.open(lr_path).convert("RGB")
                hr_img_pil = Image.open(hr_path).convert("RGB")

                # Prepare tensors
                lr_tensor = to_tensor(lr_img_pil).unsqueeze(0).to(config.DEVICE)
                hr_tensor_lpips = to_tensor(hr_img_pil).unsqueeze(0).to(config.DEVICE)

                # Pad image if smaller than patch size
                _, _, h, w = lr_tensor.shape
                padded = False
                patch_size = config.INFERENCE_PATCH_SIZE # The model was trained on this patch size
                if h < patch_size or w < patch_size:
                    pad_h = max(0, patch_size - h)
                    pad_w = max(0, patch_size - w)
                    lr_tensor = F.pad(lr_tensor, (0, pad_w, 0, pad_h), 'reflect')
                    padded = True

                # Generate SR image using sliding window inference
                with torch.no_grad():
                    sr_tensor = sliding_window_inference(generator, lr_tensor, scale=scale, patch_size=patch_size)
                
                # Crop back if padded
                if padded:
                    sr_tensor = sr_tensor[:, :, :h * scale, :w * scale]

                # --- Calculate Metrics ---
                sr_np = tensor_to_numpy(sr_tensor)
                hr_np = np.array(hr_img_pil)

                # PSNR & SSIM
                psnr, ssim = calculate_psnr_ssim(sr_np, hr_np, scale)
                total_metrics['psnr'] += psnr
                total_metrics['ssim'] += ssim
                
                # Store sample for top-5 comparison image generation
                sr_img_pil = Image.fromarray(sr_np.round().astype(np.uint8))
                top_samples.append({
                    'psnr': psnr,
                    'lr': lr_img_pil.copy(),
                    'sr': sr_img_pil,
                    'hr': hr_img_pil.copy(),
                    'name': os.path.basename(lr_path).replace('_LR.png', '')
                })

                # LPIPS
                # LPIPS expects tensors in range [-1, 1]
                sr_tensor_lpips = (sr_tensor * 2) - 1 
                hr_tensor_lpips = (hr_tensor_lpips * 2) - 1
                lpips_score = lpips_fn(sr_tensor_lpips, hr_tensor_lpips).item()
                total_metrics['lpips'] += lpips_score
                
                # NIQE (on SR image)
                # niqe_fn expects tensor in range [0, 1]
                niqe_score = niqe_fn(sr_tensor).item()
                total_metrics['niqe'] += niqe_score
                
            if num_images > 0:
                dataset_scale_key = f"{name}_x{scale}"
                all_results[dataset_scale_key] = {k: v / num_images for k, v in total_metrics.items()}

            # --- Save Top 5 PSNR Comparison Images ---
            if top_samples:
                tqdm.write(f"\nGenerating top 5 comparison images for {name} (x{scale})...")
                top_samples.sort(key=lambda x: x['psnr'], reverse=True)
                
                output_dir = "top_psnr_results"
                os.makedirs(output_dir, exist_ok=True)

                for i, sample in enumerate(top_samples[:5]):
                    lr_img, sr_img, hr_img = sample['lr'], sample['sr'], sample['hr']
                    
                    hr_w, hr_h = hr_img.size
                    lr_resized = lr_img.resize((hr_w, hr_h), Image.BICUBIC)

                    # Create a new image to hold the three images side-by-side
                    comparison_img = Image.new('RGB', (hr_w * 3, hr_h))
                    
                    # Paste the images
                    comparison_img.paste(lr_resized, (0, 0))
                    comparison_img.paste(sr_img, (hr_w, 0))
                    comparison_img.paste(hr_img, (hr_w * 2, 0))
                    
                    # Save the final image
                    psnr_val = sample['psnr']
                    save_path = os.path.join(output_dir, f"{name}_x{scale}_{sample['name']}_rank{i+1}_psnr_{psnr_val:.2f}_dino-ranger-w-degrade.png")
                    comparison_img.save(save_path)
                    tqdm.write(f"Saved: {save_path}")

    # --- 4. Display Final Results ---
    print("\n\n" + "="*80)
    print("--- FINAL EVALUATION RESULTS ---")
    print("="*80)
    header = f"{'Dataset':<20}{'PSNR':>10}{'SSIM':>10}{'LPIPS':>10}{'NIQE':>10}"
    print(header)
    print("-" * (len(header) + 5))
    
    for name, metrics in all_results.items():
        print(f"{name:<20}{metrics['psnr']:>10.2f}{metrics['ssim']:>10.4f}{metrics['lpips']:>10.4f}{metrics['niqe']:>10.2f}")
    
    print("="*80)
    print("Evaluation finished.")

if __name__ == '__main__':
    main() 