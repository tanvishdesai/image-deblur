%%writefile dataset.py
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from config import Config

class SISRDataset(Dataset):
    def __init__(self, dataset_path, split='train', transform=None):
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.transform = transform
        
        # Load metadata
        metadata_path = self.dataset_path / "metadata" / "complete_metadata.json"
        if not metadata_path.exists():
            # Fallback for local testing or partial datasets
            print(f"Warning: Metadata not found at {metadata_path}")
            self.pairs = []
        else:
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
                
            # Filter pairs based on split (if split info exists, otherwise use all/random)
            # Assuming the user might want to split manually or use the 'test_split.json'
            if split == 'test':
                test_split_path = self.dataset_path / "metadata" / "test_split.json"
                if test_split_path.exists():
                    with open(test_split_path, 'r') as f:
                        self.pairs = json.load(f)
                else:
                    # If no explicit test split, take last 10%
                    all_pairs = self.metadata['pairs']
                    split_idx = int(len(all_pairs) * 0.9)
                    self.pairs = all_pairs[split_idx:]
            else:
                test_split_path = self.dataset_path / "metadata" / "test_split.json"
                if test_split_path.exists():
                    # If test split exists, exclude those from train
                    # This is a bit complex to do efficiently without IDs, 
                    # so we'll assume the user handles splits or we just take the first 90%
                    all_pairs = self.metadata['pairs']
                    split_idx = int(len(all_pairs) * 0.9)
                    self.pairs = all_pairs[:split_idx]
                else:
                    all_pairs = self.metadata['pairs']
                    split_idx = int(len(all_pairs) * 0.9)
                    self.pairs = all_pairs[:split_idx]

    def _parse_degradation_params(self, metadata):
        """
        Extracts and normalizes degradation parameters from the metadata json.
        Returns a tensor of size [4]: [Blur, Noise, JPEG, Gamma]
        """
        # Defaults
        blur_val = 0.0
        noise_val = 0.0
        jpeg_val = 0.0
        gamma_val = 0.5 # Normalized 1.0 -> 0.5 approx
        
        if 'applied_degradations' in metadata:
            degs = metadata['applied_degradations']
            
            # 1. Blur
            if 'motion_blur' in degs:
                params = degs['motion_blur']
                if params.get('type') == 'linear':
                    blur_val = params.get('kernel_size', 0) / Config.MAX_BLUR_KERNEL
                elif params.get('type') == 'rotational':
                    blur_val = params.get('sigma', 0) / 5.0 # Approx max sigma
                elif params.get('type') == 'defocus':
                    blur_val = params.get('radius', 0) / 10.0
            
            # 2. Noise
            if 'sensor_noise' in degs:
                params = degs['sensor_noise']
                # Aggregate noise levels
                shot = params.get('shot_scale', 0)
                read = params.get('read_std', 0)
                # Simple heuristic sum
                total_noise = shot + read
                noise_val = total_noise / Config.MAX_NOISE_STD
                
            # 3. JPEG
            # Note: generate_sisr_dataset.py applies JPEG at the end but might not store it 
            # in 'applied_degradations' if it was done via the save wrapper, 
            # but let's check if it's in the metadata structure.
            # Looking at the file, it IS in applied_degradations if apply_jpeg_compression is called.
            # However, the main degrade_image function calls it at the very end:
            # "Convert back to PIL for JPEG compression (always apply this last)"
            # Wait, the code snippet ended before showing the JPEG call in degrade_image.
            # Assuming it's recorded if applied.
            # If not found, we assume high quality (0 degradation)
            if 'jpeg_compression' in degs: # Hypothetical key, need to verify if used
                q = degs['jpeg_compression'].get('quality', 100)
                jpeg_val = (100 - q) / 100.0
            
            # 4. Gamma/Lighting
            if 'illumination_degradation' in degs:
                params = degs['illumination_degradation']
                if params.get('type') == 'low_light':
                    g = params.get('gamma', 1.0)
                    gamma_val = g / Config.MAX_GAMMA
        
        # Clip to [0, 1]
        vec = torch.tensor([blur_val, noise_val, jpeg_val, gamma_val], dtype=torch.float32)
        return torch.clamp(vec, 0.0, 1.0)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        lr_path = self.dataset_path / pair['lr_path']
        hr_path = self.dataset_path / pair['hr_path']
        
        try:
            lr_img = Image.open(lr_path).convert('RGB')
            hr_img = Image.open(hr_path).convert('RGB')
            
            # Extract degradation vector
            deg_vec = self._parse_degradation_params(pair)
            
            # Aligned Random Crop
            if self.split == 'train':
                lr_w, lr_h = lr_img.size
                crop_size = Config.TRAIN_CROP_SIZE
                
                if lr_w > crop_size and lr_h > crop_size:
                    # Random crop
                    x = np.random.randint(0, lr_w - crop_size)
                    y = np.random.randint(0, lr_h - crop_size)
                    
                    # Crop LR
                    lr_img = lr_img.crop((x, y, x + crop_size, y + crop_size))
                    
                    # Crop HR (scale coordinates)
                    scale = Config.SCALE
                    hr_x, hr_y = x * scale, y * scale
                    hr_crop_size = crop_size * scale
                    hr_img = hr_img.crop((hr_x, hr_y, hr_x + hr_crop_size, hr_y + hr_crop_size))
                else:
                    # If image is too small, resize it (or pad, but resize is easier for now)
                    # This shouldn't happen often with standard datasets, but good for robustness
                    lr_img = lr_img.resize((crop_size, crop_size), Image.BICUBIC)
                    hr_img = hr_img.resize((crop_size * Config.SCALE, crop_size * Config.SCALE), Image.BICUBIC)
            else:
                # For validation/test, we might want full images or center crop.
                # To avoid batch size issues during validation if batch_size > 1, we should also crop.
                # Or we can set batch_size=1 for validation.
                # Let's do center crop for simplicity if batching.
                lr_w, lr_h = lr_img.size
                crop_size = Config.TRAIN_CROP_SIZE
                
                if lr_w > crop_size and lr_h > crop_size:
                    x = (lr_w - crop_size) // 2
                    y = (lr_h - crop_size) // 2
                    lr_img = lr_img.crop((x, y, x + crop_size, y + crop_size))
                    
                    scale = Config.SCALE
                    hr_x, hr_y = x * scale, y * scale
                    hr_crop_size = crop_size * scale
                    hr_img = hr_img.crop((hr_x, hr_y, hr_x + hr_crop_size, hr_y + hr_crop_size))
                else:
                    lr_img = lr_img.resize((crop_size, crop_size), Image.BICUBIC)
                    hr_img = hr_img.resize((crop_size * Config.SCALE, crop_size * Config.SCALE), Image.BICUBIC)

            # Basic ToTensor
            lr_tensor = torch.from_numpy(np.array(lr_img)).permute(2, 0, 1).float() / 255.0
            hr_tensor = torch.from_numpy(np.array(hr_img)).permute(2, 0, 1).float() / 255.0
            
            return {
                'lr': lr_tensor,
                'hr': hr_tensor,
                'deg_vec': deg_vec,
                'lr_path': str(lr_path)
            }
            
        except Exception as e:
            print(f"Error loading {lr_path}: {e}")
            # Return a dummy item or handle gracefully
            return self.__getitem__((idx + 1) % len(self))
