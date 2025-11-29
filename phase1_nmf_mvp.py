"""phase1_nmf_mvp.py

Minimal implementation of Phase 1 (MVP) from the Neural Motion Fields roadmap.

This script performs *per-image optimisation* of a simple Neural Motion Field (NMF)
model that learns to reconstruct a blurry image produced by Monte-Carlo temporal
integration of the model.  After optimisation, the model is evaluated at a
fixed time-slice (t = 0.5) to produce a predicted sharp image.

Key Features
------------
1. SIREN-style sinusoidal activations (Sitzmann et al.) to represent high-frequency
   image content.
2. Differentiable blur renderer that repeatedly queries the NMF at random time
   samples and averages the results (Monte-Carlo integration).
3. Simple MSE loss with Adam optimisation.
4. Optional generation of synthetic blur from high-resolution images so that the
   script can be run on datasets containing only sharp images (e.g. DIV2K / DF2K).

Usage
-----
python phase1_nmf_mvp.py \
    --data_dir /path/to/DF2K_train_HR \
    --output_dir ./nmf_results \
    --n_iters 2000 \
    --n_time_samples 8 \
    --blur_kernel 21

The above command will iterate over every image in *data_dir*, generate a blurred
counterpart on-the-fly (if one does not already exist), optimise a dedicated NMF
model for each image, and save both the blurred input and the recovered sharp
image to *output_dir*.

Dependencies
------------
* Python 3.8+
* PyTorch 1.10+
* torchvision (for basic image utilities)
* Pillow
* OpenCV-Python (only for generating synthetic blur)

Tip: enable CUDA for much faster optimisation.
"""

import glob
import math
import os
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace  # Replaces argparse for hard-coded config

# -----------------------------
# SIREN building blocks
# -----------------------------
class SineLayer(nn.Module):
    """Fully-connected layer with sine activation (SIREN).

    Args:
        in_features (int):  input feature dimension
        out_features (int): output feature dimension
        is_first (bool):    whether this is the first layer in the network
        omega_0 (float):    frequency factor from the SIREN paper
    """

    def __init__(self, in_features: int, out_features: int, *, is_first: bool = False, omega_0: float = 30.0):
        super().__init__()
        self.in_features = in_features
        self.is_first = is_first
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                # First layer uses symmetric uniform init in [-1/in, 1/in]
                bound = 1 / self.in_features
                self.linear.weight.uniform_(-bound, bound)
            else:
                # Subsequent layers: SIREN specific init
                bound = math.sqrt(6 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return torch.sin(self.omega_0 * self.linear(x))


class NMFModel(nn.Module):
    """Minimal Neural Motion Field (MVP).

    Signature: f_theta(x, y, t) -> (r, g, b, a)
    For the MVP we ignore motion and simply predict RGB (+ alpha) at each
    spatio-temporal coordinate.
    """

    def __init__(self, hidden_dim: int = 256, hidden_layers: int = 4, omega_0: float = 30.0):
        super().__init__()
        layers = [
            SineLayer(3, hidden_dim, is_first=True, omega_0=omega_0)
        ]
        for _ in range(hidden_layers - 1):
            layers.append(SineLayer(hidden_dim, hidden_dim, omega_0=omega_0))
        self.net = nn.Sequential(*layers)
        self.final_linear = nn.Linear(hidden_dim, 4)  # 3 RGB + alpha

    def forward(self, coords_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        h = self.net(coords_t)
        out = self.final_linear(h)
        rgb = torch.sigmoid(out[..., :3])  # (N, 3), range (0,1)
        alpha = torch.sigmoid(out[..., 3:])  # (N, 1)
        return rgb, alpha


# -----------------------------
# Utility functions
# -----------------------------

def load_image(path: str, device: torch.device) -> torch.Tensor:
    """Load image as torch.Tensor in (C,H,W) range [0,1]."""
    img = Image.open(path).convert("RGB")
    img_np = np.array(img).astype(np.float32) / 255.0
    img_t = torch.from_numpy(img_np).permute(2, 0, 1).to(device)
    return img_t


def pil_save(tensor: torch.Tensor, path: str):
    """Save torch Tensor image (C,H,W) in range [0,1] to path."""
    tensor = tensor.detach().cpu().clamp(0, 1)
    img_np = (tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    Image.fromarray(img_np).save(path)


def gaussian_blur_np(img_np: np.ndarray, ksize: int = 21, sigma: float = 3.0) -> np.ndarray:
    """Apply a Gaussian blur (OpenCV) to an HxWx3 numpy array."""
    if ksize % 2 == 0:
        ksize += 1  # kernel must be odd
    return cv2.GaussianBlur(img_np, (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT101)


def generate_synthetic_blur(sharp: torch.Tensor, ksize: int = 21, sigma: float = 3.0) -> torch.Tensor:
    """Generate a blurred version of a sharp image tensor (C,H,W)."""
    sharp_np = sharp.permute(1, 2, 0).cpu().numpy()
    blur_np = gaussian_blur_np(sharp_np, ksize, sigma)
    blur_t = torch.from_numpy(blur_np).permute(2, 0, 1).to(sharp.device)
    return blur_t


def get_coordinate_grid(h: int, w: int, device: torch.device) -> torch.Tensor:
    """Return flattened (h*w, 2) grid of (x,y) in [-1,1]."""
    ys = torch.linspace(0, 1, steps=h, device=device)
    xs = torch.linspace(0, 1, steps=w, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # (H,W)
    coords = torch.stack([grid_x, grid_y], dim=-1)  # (H,W,2)
    coords = coords.reshape(-1, 2)
    coords = coords * 2.0 - 1.0  # [0,1] -> [-1,1]
    return coords


def render_blur(model: nn.Module, coords: torch.Tensor, h: int, w: int, n_samples: int = 8, *, batch_size: int = 4096) -> torch.Tensor:
    """Monte-Carlo rendering of a blurry image (C,H,W)."""
    n_pix = coords.shape[0]
    device = coords.device
    accumulated = torch.zeros((n_pix, 3), device=device)
    # Process in manageable batches to avoid OOM
    for _ in range(n_samples):
        t = torch.rand((n_pix, 1), device=device)
        input_ = torch.cat([coords, t], dim=1)  # (N,3)
        for start in range(0, n_pix, batch_size):
            rgb, _ = model(input_[start : start + batch_size])
            accumulated[start : start + batch_size] += rgb.detach()
    blur_flat = accumulated / n_samples  # (N,3)
    blur = blur_flat.reshape(h, w, 3).permute(2, 0, 1)  # (C,H,W)
    return blur


def render_sharp(model: nn.Module, coords: torch.Tensor, h: int, w: int, *, t_val: float = 0.5, batch_size: int = 4096) -> torch.Tensor:
    """Render sharp image by querying NMF at a fixed time slice in smaller chunks."""
    n_pix = coords.shape[0]
    t = torch.full((n_pix, 1), t_val, device=coords.device)
    input_ = torch.cat([coords, t], dim=1)
    rgb_acc = []
    for start in range(0, n_pix, batch_size):
        rgb_chunk, _ = model(input_[start : start + batch_size])
        rgb_acc.append(rgb_chunk.detach())
    rgb_flat = torch.cat(rgb_acc, dim=0)
    sharp = rgb_flat.reshape(h, w, 3).permute(2, 0, 1)
    return sharp


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return 100.0
    return 20 * math.log10(1.0 / math.sqrt(mse))


# -----------------------------
# Main optimisation routine
# -----------------------------

def optimise_single_image(img_path: str, args, device: torch.device):
    """Optimise NMF on a single image using minibatch sampling to reduce memory."""
    img_name = Path(img_path).stem
    sharp = load_image(img_path, device)
    c, h, w = sharp.shape
    coords = get_coordinate_grid(h, w, device)

    # Prepare blurred input B (either from disk or synthetic)
    if args.provided_blur_dir:
        candidate = os.path.join(args.provided_blur_dir, os.path.basename(img_path))
        if os.path.isfile(candidate):
            blurred = load_image(candidate, device)
        else:
            raise FileNotFoundError(f"Blurred version not found: {candidate}")
    else:
        blurred = generate_synthetic_blur(sharp, ksize=args.blur_kernel, sigma=args.blur_sigma)

    blurred_flat = blurred.reshape(-1, c)

    # Instantiate NMF model
    model = NMFModel(hidden_dim=args.hidden_dim, hidden_layers=args.hidden_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    pixels_per_batch = args.pixels_per_batch if hasattr(args, "pixels_per_batch") else 2048

    # Optimisation loop with minibatch sampling
    progress = tqdm(range(args.n_iters), desc=f"Optimising {img_name}")
    for step in progress:
        optimizer.zero_grad()

        # Sample a random subset of pixels
        idx = torch.randint(0, h * w, (pixels_per_batch,), device=device)
        coords_batch = coords[idx]
        targets_batch = blurred_flat[idx]

        # Monte-Carlo integration over time samples
        coords_rep = coords_batch.repeat_interleave(args.n_time_samples, dim=0)
        t = torch.rand((coords_rep.shape[0], 1), device=device)
        inp = torch.cat([coords_rep, t], dim=1)

        preds, _ = model(inp)
        preds = preds.reshape(pixels_per_batch, args.n_time_samples, 3).mean(dim=1)

        loss = F.mse_loss(preds, targets_batch)
        loss.backward()
        optimizer.step()

        if (step + 1) % 100 == 0:
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

    # After optimisation: render sharp image (chunked)
    sharp_pred = render_sharp(model, coords, h, w, batch_size=4096)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    pil_save(blurred.cpu(), os.path.join(args.output_dir, f"{img_name}_blur.png"))
    pil_save(sharp_pred, os.path.join(args.output_dir, f"{img_name}_sharp_pred.png"))
    pil_save(sharp, os.path.join(args.output_dir, f"{img_name}_sharp_gt.png"))

    sharp_psnr = psnr(sharp_pred, sharp)

    # After loading or generating the blurred image, compute and print baseline PSNR
    baseline_psnr = psnr(blurred, sharp)
    print(f"{img_name}: input blur PSNR={baseline_psnr:.2f} dB")

    return sharp_psnr


# -----------------------------
# Configuration is now hard-coded via SimpleNamespace; no CLI required
# -----------------------------

def main():
    args = SimpleNamespace(
        # Directory containing high-resolution (sharp) images. Change as needed.
        data_dir="/kaggle/input/df2kdata/DF2K_train_HR",
        # Where to save outputs (blurred copy, recovered sharp, ground truth sharp)
        output_dir="nmf_outputs",
        # Optionally provide a directory with existing blurred inputs (else synthetic blur is used)
        provided_blur_dir=None,
        # Optimisation hyper-parameters
        n_iters=10000,
        n_time_samples=8,
        hidden_dim=256,
        hidden_layers=4,
        lr=1e-4,
        max_images=10,
        pixels_per_batch=2048,
        # Synthetic blur parameters
        blur_kernel=21,
        blur_sigma=3.0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Collect image files (png/jpg/jpeg)
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    img_files = []
    for ext in exts:
        img_files.extend(glob.glob(os.path.join(args.data_dir, "**", ext), recursive=True))
    if not img_files:
        raise RuntimeError(f"No image files found in {args.data_dir}")

    if args.max_images > 0:
        img_files = img_files[: args.max_images]

    psnr_scores = []
    for img_path in img_files:
        score = optimise_single_image(img_path, args, device)
        psnr_scores.append(score)
        print(f"Finished {Path(img_path).name}: PSNR={score:.2f} dB")

    avg_psnr = sum(psnr_scores) / len(psnr_scores)
    print(f"Average PSNR over {len(psnr_scores)} image(s): {avg_psnr:.2f} dB")


if __name__ == "__main__":
    main() 