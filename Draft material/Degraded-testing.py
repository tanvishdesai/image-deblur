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
# This script is designed for testing the HybridSwinNet SISR model.
#
# To run this script, you need to install the following packages:
# pip install torch torchvision torchaudio
# pip install numpy pillow scikit-image tqdm einops
# pip install lpips
# pip install pyiqa
#
# Place this script in the same directory as your 'checkpoint_epoch_514.pth' file
# and the dataset folders (BSD100, Set5, Set14, Urban100).
# =================================================================================================


# =================================================================================================
# SECTION 2: MODEL ARCHITECTURE (Copied from proposed_solution.py)
# =================================================================================================
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
        # The 'scale' parameter in __init__ is used to build the upsampling blocks.
        # We will initialize the model with scale=4 to match the checkpoint,
        # but the forward pass will handle different target scales.
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
        
        # Upsampling - build for scale 4 to match checkpoint
        self.upsample_blocks = nn.ModuleList()
        for _ in range(2): # Two PixelShuffle blocks for 4x
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

    def forward(self, x, target_scale=None):
        """
        Forward pass with flexible upsampling for different target scales.
        """
        if target_scale is None:
            target_scale = self.scale

        # Shallow feature extraction
        x = self.conv_first(x)
        
        # Deep feature extraction
        residual = x
        for block in self.blocks:
            x = x + block(x)
        
        x = self.conv_after_body(x)
        x = x + residual
        
        # --- Multi-scale Upsampling Logic ---
        if target_scale == 2:
            # Use one 2x upsampler block
            x = self.upsample_blocks[0](x)
        elif target_scale == 3:
            # Use one 2x upsampler and then interpolate to 3x
            x = self.upsample_blocks[0](x)
            x = F.interpolate(x, scale_factor=1.5, mode='bicubic', align_corners=False)
        elif target_scale == 4:
            # Use both 2x upsampler blocks for 4x
            for block in self.upsample_blocks:
                x = block(x)
        else:
            # Fallback for any other scale
            x = F.interpolate(x, scale_factor=target_scale, mode='bicubic', align_corners=False)
        
        # Final output
        x = self.conv_last(self.lrelu(self.conv_hr(x)))
        
        return x

# =================================================================================================
# SECTION 3: CONFIGURATION FOR TESTING
# =================================================================================================
class TestConfig:
    # --- IMPORTANT: Path to the comprehensive checkpoint file from proposed solution 4 ---
    CHECKPOINT_PATH = "/kaggle/input/degraded-100-complete/training_outputs/checkpoints/checkpoint_epoch_500.pth" # <<< UPDATE THIS PATH
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # SCALE_FACTOR is now determined dynamically from the dataset folders.

    # --- TransformerESRGAN Parameters (must match the trained model) ---
    TRANSFORMER_NUM_FEAT = 96
    TRANSFORMER_NUM_BLOCK = 6
    TRANSFORMER_NUM_HEAD = 6
    TRANSFORMER_WINDOW_SIZE = 8

    # Datasets to evaluate
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
    
    # Always initialize the generator with scale=4 to match the checkpoint's architecture
    generator = TransformerESRGAN(
        num_feat=config.TRANSFORMER_NUM_FEAT,
        num_block=config.TRANSFORMER_NUM_BLOCK,
        num_head=config.TRANSFORMER_NUM_HEAD,
        window_size=config.TRANSFORMER_WINDOW_SIZE,
        scale=4
    ).to(config.DEVICE)
    
    try:
        # Proposed solution 4 saves a comprehensive checkpoint dictionary.
        # We need to extract the generator's state dictionary from it.
        checkpoint = torch.load(config.CHECKPOINT_PATH, map_location=config.DEVICE)
        generator.load_state_dict(checkpoint['generator_state_dict'])
    except Exception as e:
         print(f"FATAL: Could not load checkpoint. Error: {e}")
         print("Please ensure the checkpoint file is a valid comprehensive checkpoint from 'proposed solution 4'")
         print("and that it contains the 'generator_state_dict' key.")
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
                patch_size = 64 # The model was trained on 64x64 patches
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
                    save_path = os.path.join(output_dir, f"{name}_x{scale}_{sample['name']}_rank{i+1}_psnr_{psnr_val:.2f}_degraded.png")
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