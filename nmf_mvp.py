import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
import numpy as np
from tqdm import tqdm
import os
import imageio
from scipy.ndimage import convolve
import cv2 # Using cv2 for the dummy image creation


class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class NMFModel(nn.Module):
    def __init__(self, in_features=3, out_features=3, hidden_features=256, hidden_layers=4, omega_0=30.):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                     is_first=False, omega_0=omega_0))

        final_linear = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / omega_0,
                                          np.sqrt(6 / hidden_features) / omega_0)
        self.net.append(final_linear)

        self.net = nn.Sequential(*self.net)

    def forward(self, coords_t):
        output = self.net(coords_t)
        return output

def render_full_image(model, h, w, n_samples, device, is_blurry=True):
    model.eval()
    
    pixels_y, pixels_x = torch.meshgrid(torch.linspace(-1, 1, h, device=device), torch.linspace(-1, 1, w, device=device), indexing='ij')
    coords = torch.stack([pixels_x, pixels_y], dim=-1).view(-1, 2)
    
    if is_blurry:
        coords = coords.repeat_interleave(n_samples, dim=0)
        t = torch.rand(coords.shape[0], 1, device=device)
    else:
        t = torch.full((coords.shape[0], 1), 0.5, device=device)

    coords_t = torch.cat([coords, t], dim=-1)
    
    rendered_colors = []
    batch_size = 4096 * 32
    for i in range(0, coords_t.shape[0], batch_size):
        batch_coords_t = coords_t[i:i+batch_size]
        out = model(batch_coords_t)
        rendered_colors.append(torch.sigmoid(out).cpu())
    
    rendered_colors = torch.cat(rendered_colors, dim=0)

    if is_blurry:
        rendered_image = rendered_colors.view(h, w, n_samples, -1)
        final_image = torch.mean(rendered_image, dim=2)
    else:
        final_image = rendered_colors.view(h, w, -1)
        
    return final_image.permute(2, 0, 1)

def total_variation_loss(img):
    # img is (C, H, W)
    # Squeeze to (C, H, W) if it's (1, C, H, W)
    if img.dim() == 4 and img.size(0) == 1:
        img = img.squeeze(0)

    # We don't want to calculate TV loss on the channel dimension
    tv_h = torch.pow(img[:, 1:, :] - img[:, :-1, :], 2).sum()
    tv_w = torch.pow(img[:, :, 1:] - img[:, :, :-1], 2).sum()
    return tv_h + tv_w

def optimize_nmf_for_image(blurry_image_path, n_iters=2000, hidden_features=256, hidden_layers=4,
                           n_samples=64, lr=1e-4, output_dir='output_mvp', pixels_per_batch=4096,
                           tv_weight=1e-4, tv_patch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    blurry_img_pil = Image.open(blurry_image_path).convert('RGB')
    
    # Resize large images to speed up training, similar to your logs
    if max(blurry_img_pil.size) > 512:
        blurry_img_pil.thumbnail((512, 512), Image.Resampling.LANCZOS)
        print(f"Resized image to: {blurry_img_pil.size}")

    transform = ToTensor()
    blurry_img_tensor = transform(blurry_img_pil).permute(1, 2, 0).to(device)
    h, w, c = blurry_img_tensor.shape
    
    print(f"Image resolution: {h}x{w}")

    pixels_y, pixels_x = torch.meshgrid(torch.linspace(-1, 1, h, device=device), torch.linspace(-1, 1, w, device=device), indexing='ij')
    pixel_coords = torch.stack([pixels_x, pixels_y], dim=-1).view(-1, 2)
    blurry_img_flat = blurry_img_tensor.view(-1, c)

    nmf_model = NMFModel(in_features=3, out_features=c, hidden_features=hidden_features, hidden_layers=hidden_layers).to(device)
    optimizer = torch.optim.Adam(nmf_model.parameters(), lr=lr)

    os.makedirs(output_dir, exist_ok=True)

    pbar = tqdm(range(n_iters))
    for i in pbar:
        nmf_model.train()
        optimizer.zero_grad()
        
        # --- 1. Reconstruction Loss (as before) ---
        pixel_indices = torch.randint(0, h * w, (pixels_per_batch,), device=device)
        batch_coords = pixel_coords[pixel_indices]
        batch_gt_colors = blurry_img_flat[pixel_indices]

        batch_coords_repeated = batch_coords.repeat_interleave(n_samples, dim=0)
        t = torch.rand(pixels_per_batch * n_samples, 1, device=device)
        coords_t = torch.cat([batch_coords_repeated, t], dim=-1)
        
        rendered_output = nmf_model(coords_t)
        rendered_output = torch.sigmoid(rendered_output)
        predicted_colors = rendered_output.view(pixels_per_batch, n_samples, c).mean(dim=1)
        
        recon_loss = F.mse_loss(predicted_colors, batch_gt_colors)
        
        # --- 2. Total Variation (TV) Loss (The new part) ---
        loss = recon_loss
        tv_loss_val = 0
        if tv_weight > 0:
            # Render a small sharp patch to calculate TV loss on
            patch_y_start = torch.randint(0, h - tv_patch_size, (1,)).item()
            patch_x_start = torch.randint(0, w - tv_patch_size, (1,)).item()
            
            patch_y, patch_x = torch.meshgrid(
                torch.linspace(-1 + 2*patch_y_start/h, -1 + 2*(patch_y_start+tv_patch_size-1)/h, tv_patch_size, device=device),
                torch.linspace(-1 + 2*patch_x_start/w, -1 + 2*(patch_x_start+tv_patch_size-1)/w, tv_patch_size, device=device),
                indexing='ij'
            )
            patch_coords = torch.stack([patch_x, patch_y], dim=-1).view(-1, 2)
            patch_t = torch.full((patch_coords.shape[0], 1), 0.5, device=device) # t=0.5 for sharp
            patch_coords_t = torch.cat([patch_coords, patch_t], dim=-1)
            
            sharp_patch_out = torch.sigmoid(nmf_model(patch_coords_t))
            sharp_patch_img = sharp_patch_out.view(tv_patch_size, tv_patch_size, c).permute(2, 0, 1)
            
            tv_loss = total_variation_loss(sharp_patch_img)
            tv_loss_val = tv_loss.item()
            loss = loss + tv_weight * tv_loss

        loss.backward()
        optimizer.step()
        
        pbar.set_description(f"Loss: {loss.item():.6f} (Recon: {recon_loss.item():.6f}, TV: {tv_loss_val:.2f})")
        
        if i % 500 == 0 or i == n_iters - 1:
            with torch.no_grad():
                sharp_image = render_full_image(nmf_model, h, w, n_samples, device, is_blurry=False)
                sharp_image_pil = ToPILImage()((sharp_image.cpu().clamp(0, 1)))
                sharp_image_pil.save(os.path.join(output_dir, f'sharp_frame_iter_{i}.png'))


def create_synthetic_blur(image_path, kernel_size=31, output_path='synthetic_blurred.png'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img_pil = Image.open(image_path).convert('RGB')
    
    # Optional: Resize large images to speed up the process
    # if max(img_pil.size) > 1024:
    #     img_pil.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    
    img_np = np.array(img_pil) / 255.0

    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    kernel[center, :] = 1 # Horizontal motion blur
    kernel /= kernel.sum()

    blurred_np = np.zeros_like(img_np)
    for i in range(3):
        blurred_np[..., i] = convolve(img_np[..., i], kernel, mode='reflect')
        
    blurred_pil = Image.fromarray((blurred_np * 255).astype(np.uint8))
    blurred_pil.save(output_path)
    return output_path

# <<< NEW, MORE INTELLIGENT MAIN EXECUTION BLOCK >>>
if __name__ == '__main__':
    # --- Step 1: Define your dataset path here ---
    # This path is based on the Kaggle environment from your traceback.
    # Change it if your dataset is located elsewhere.
    dataset_path = "/kaggle/input/df2kdata/DF2K_train_HR" 
    
    # --- Step 2: Define how many images you want to process ---
    num_images_to_process = 2 # Let's process 2 images as a demonstration
    
    image_paths_to_process = []
    
    # --- Step 3: Check if the dataset exists and get image paths ---
    if os.path.exists(dataset_path) and os.path.isdir(dataset_path):
        print(f"Found dataset at: {dataset_path}")
        image_files = sorted([f for f in os.listdir(dataset_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        if not image_files:
            print("Dataset directory is empty. Falling back to dummy image.")
            image_paths_to_process.append(None) # Use None as a signal for dummy image
        else:
            selected_files = image_files[:num_images_to_process]
            image_paths_to_process = [os.path.join(dataset_path, f) for f in selected_files]
            print(f"Selected {len(image_paths_to_process)} images to process: {selected_files}")
            
    else:
        print(f"Dataset path '{dataset_path}' not found. Falling back to dummy image.")
        image_paths_to_process.append(None) # Use None as a signal for dummy image

    # --- Step 4: Loop through the selected images and process them ---
    for source_image_path in image_paths_to_process:
        base_output_dir = 'output_mvp_batched'
        
        if source_image_path is None:
            # This is the fallback logic
            print("\n--- Processing Dummy Image ---")
            image_name = "dummy_image"
            start_image_path = "dummy_start_image.png"
            if not os.path.exists(start_image_path):
                dummy_np = np.array(Image.new('RGB', (256, 256), color='red'))
                cv2.circle(dummy_np, (128, 128), 60, (0, 255, 0), -1)
                cv2.putText(dummy_np, "TEST", (60, 140), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
                Image.fromarray(dummy_np).save(start_image_path)
        else:
            # This is the main logic for real images
            image_name = os.path.splitext(os.path.basename(source_image_path))[0]
            start_image_path = source_image_path
            print(f"\n--- Processing Image: {image_name} ---")

        # Create a dedicated output folder for this specific image
        image_output_dir = os.path.join(base_output_dir, image_name)
        
        # Create the synthetic blur for the current image
        blurred_image_path = create_synthetic_blur(
            start_image_path, 
            kernel_size=31, 
            output_path=os.path.join(image_output_dir, "synthetic_blurred.png")
        )
        print(f"Synthetically blurred image saved to: {blurred_image_path}")

        # Run the optimization process
        optimize_nmf_for_image(
            blurry_image_path=blurred_image_path,
            n_iters=4001, # Let's try fewer iterations first with the new loss
            hidden_features=256,
            hidden_layers=4,
            n_samples=32,
            pixels_per_batch=4096,
            lr=5e-4,
            output_dir=image_output_dir, # Use the specific output directory
            tv_weight=2e-4, # This is the new parameter to encourage sharpness
            tv_patch_size=64 # The size of the patch for TV loss
        )

    print("\nAll processing finished.")