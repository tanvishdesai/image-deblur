"""
SOTA Models Evaluation on Merged SISR Benchmark Dataset (Optimized Version)

This script evaluates state-of-the-art super-resolution models on a merged dataset
created by combining multiple category-specific datasets using dataset_merger.py.

This version is optimized for speed using asynchronous data loading and
GPU-accelerated metrics. It conditionally uses torch.compile on compatible hardware.

USAGE:
1. Update paths and batching settings in the get_configuration() function.
2. Run: python test_sota_models.py

The script will:
- Load the merged dataset metadata.
- Initialize available SOTA models (DSCF-SR, SeemoRe variants, Real-ESRGAN, SwinIR, HAT, StableSR).
- Evaluate each model on the test set using efficient data loading.
- Save results and comparison images.
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# --- New Imports for Optimization ---
from torch.utils.data import Dataset, DataLoader
import torchmetrics
# --- End New Imports ---


MODEL_DISPLAY_NAMES = {
    "DSCF-SR": "DSCF-SR",
    "SeemoRe_B_X4": "SeemoRe-B",
    "SeemoRe_L_X4": "SeemoRe-L",
    "SeemoRe_T_X4": "SeemoRe-T",
    "RealESRGAN_x4": "Real-ESRGAN",
    "SwinIR": "SwinIR",
    "StableSR": "StableSR",
}

CATEGORY_DISPLAY_NAMES = {
    "urban_architecture": "Urban Architecture",
    "natural_landscapes": "Natural Landscapes",
    "portraits_people": "Portraits & People",
    "objects_macro": "Macro Objects",
}


def add_sota_models_to_path(sota_models_path):
    """Add SOTA models to Python path for imports."""
    if not os.path.isabs(sota_models_path):
        sota_models_path = os.path.abspath(sota_models_path)

    paths_to_add = {
        "DSCF-SR": ["", "utils", "models"],
        "seemoredetails": ["", "basicsr"],
        "Real-ESRGAN": [""],
        "SwinIR": ["", "models"],
        "HAT": [""],
    }

    for model_dir, subdirs in paths_to_add.items():
        base_path = Path(sota_models_path) / model_dir
        if base_path.exists():
            for subdir in subdirs:
                full_path = str(base_path / subdir)
                if full_path not in sys.path:
                    sys.path.insert(0, full_path)
            print(f"✓ Added paths for {model_dir}")
        else:
            print(f"⚠ {model_dir} path not found: {base_path}")


class SISRDataset(Dataset):
    """Custom PyTorch Dataset for loading SISR data efficiently."""
    def __init__(self, pairs, dataset_path, data_range=1.0):
        self.pairs = pairs
        self.dataset_path = Path(dataset_path)
        self.data_range = data_range

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        hr_path = self.dataset_path / pair['hr_path']
        lr_path = self.dataset_path / pair['lr_path']

        try:
            lr_img = Image.open(lr_path).convert('RGB')
            lr_array = np.array(lr_img, dtype=np.float32)
            if self.data_range == 1.0:
                lr_array /= 255.0
            lr_tensor = torch.from_numpy(lr_array).permute(2, 0, 1)

            hr_img = Image.open(hr_path).convert('RGB')
            hr_array = np.array(hr_img, dtype=np.uint8) # Ensure HR is uint8 for metrics
            
            return lr_tensor, hr_array, str(pair['hr_path']), str(pair['lr_path']), json.dumps(pair)
        except Exception as e:
            print(f"⚠ Error loading pair {idx} ({lr_path}): {e}. Skipping.")
            return None

def collate_fn(batch):
    """
    Custom collate function to handle images of varying sizes.
    Instead of stacking images into a single tensor (which requires them all to be the same size),
    it keeps them as a list.
    """
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None, None, None, None, None

    # Unzip the batch into separate lists
    lr_tensors, hr_arrays, hr_paths, lr_paths, pairs_json = zip(*batch)
    
    # Return the items as lists instead of trying to stack them into a single tensor
    return list(lr_tensors), list(hr_arrays), list(hr_paths), list(lr_paths), list(pairs_json)


class MetricsCalculator:
    """Utility functions for LPIPS and luminance-based evaluation."""

    RGB_TO_Y_WEIGHTS = torch.tensor([65.481, 128.553, 24.966], dtype=torch.float32).view(1, 3, 1, 1)

    @staticmethod
    def calculate_lpips(sr_tensor, hr_tensor, lpips_fn):
        """Calculates LPIPS for a single pair of images on the GPU."""
        if lpips_fn is None:
            return None
        
        # LPIPS expects input in range [-1, 1]
        sr_lpips = sr_tensor * 2 - 1
        hr_lpips = hr_tensor * 2 - 1
        
        with torch.no_grad():
            lpips_score = lpips_fn(sr_lpips, hr_lpips)
        
        return lpips_score.squeeze().item()

    @staticmethod
    def rgb_to_y_channel(tensor_255):
        """
        Convert an RGB tensor in [0, 255] to Y channel using BT.601 coefficients.

        This mirrors common SISR reporting practice more closely than raw RGB metrics.
        """
        weights = MetricsCalculator.RGB_TO_Y_WEIGHTS.to(device=tensor_255.device, dtype=tensor_255.dtype)
        return (tensor_255 * weights).sum(dim=1, keepdim=True) / 255.0 + 16.0

class SwinIRWrapper(torch.nn.Module):
    """Wraps SwinIR to handle required window-size padding during inference."""
    def __init__(self, model, window_size=8, scale=4):
        super().__init__()
        self.model = model
        self.window_size = window_size
        self.scale = scale

    def forward(self, x):
        _, _, h_old, w_old = x.size()
        h_pad = (self.window_size - h_old % self.window_size) % self.window_size
        w_pad = (self.window_size - w_old % self.window_size) % self.window_size
        x = F.pad(x, (0, w_pad, 0, h_pad), mode='reflect')
        output = self.model(x)
        return output[..., :h_old * self.scale, :w_old * self.scale]


class SOTAModelEvaluator:
    """Evaluate SOTA models on the generated dataset."""

    METRIC_NAMES = ["psnr_rgb", "ssim_rgb", "psnr_y", "ssim_y", "lpips"]

    def __init__(
        self,
        sota_models_path,
        dataset_path,
        device='cuda',
        save_all_comparisons=False,
        save_worst_comparisons=False,
        num_worst_to_save=5,
        default_batch_size=4,
        default_num_workers=2,
    ):
        self.sota_models_path = Path(sota_models_path)
        self.dataset_path = Path(dataset_path)
        self.device = device
        self.save_all_comparisons = save_all_comparisons
        self.save_worst_comparisons = save_worst_comparisons
        self.num_worst_to_save = num_worst_to_save
        self.default_batch_size = default_batch_size
        self.default_num_workers = default_num_workers
        
        print(f"Using device: {device}")
        if device.startswith('cuda'):
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            torch.backends.cudnn.benchmark = True

        # ★★★ FIX: CHECK IF TORCH.COMPILE IS SUPPORTED BY THE HARDWARE ★★★
        self.use_torch_compile = False
        if hasattr(torch, 'compile') and self.device.startswith('cuda'):
            major, _ = torch.cuda.get_device_capability(self.device)
            if major >= 7:
                self.use_torch_compile = True
                print(f"✓ torch.compile is enabled (GPU Capability: {major}.0 >= 7.0).")
            else:
                print(f"⚠ torch.compile is disabled (GPU Capability {major}.0 < 7.0).")
        else:
            print("✓ torch.compile is disabled (not available or not using CUDA).")
        
        self.psnr_metric_rgb = torchmetrics.PeakSignalNoiseRatio(data_range=255.0).to(self.device)
        self.ssim_metric_rgb = torchmetrics.StructuralSimilarityIndexMeasure(data_range=255.0).to(self.device)
        self.psnr_metric_y = torchmetrics.PeakSignalNoiseRatio(data_range=255.0).to(self.device)
        self.ssim_metric_y = torchmetrics.StructuralSimilarityIndexMeasure(data_range=255.0).to(self.device)
        print("✓ Initialized GPU-accelerated metrics (torchmetrics)")

        self.lpips_fn = None
        try:
            import lpips
            self.lpips_fn = lpips.LPIPS(net='alex').to(self.device)
            print("✓ LPIPS loaded successfully")
        except ImportError:
            print("⚠ LPIPS not available, skipping LPIPS calculation")

        self.load_dataset_metadata()
        self.models = {}
        self.initialize_models()

    def validate_merged_dataset(self):
        print(f"🔍 Validating merged dataset structure at: {self.dataset_path}")
        if not all((self.dataset_path / d).exists() for d in ['HR', 'LR', 'metadata']):
            print("❌ Missing one or more required directories: HR, LR, metadata")
            return False
        if not (self.dataset_path / "metadata" / "complete_metadata.json").exists():
            print("❌ metadata/complete_metadata.json not found!")
            return False
        return True

    def load_dataset_metadata(self):
        print("\n📂 Loading merged dataset metadata...")
        if not self.validate_merged_dataset():
            raise ValueError("Dataset validation failed.")
        metadata_file = self.dataset_path / "metadata" / "complete_metadata.json"
        with open(metadata_file, 'r') as f:
            self.dataset_metadata = json.load(f)

        split_summary_file = self.dataset_path / "metadata" / "split_summary.json"
        self.split_summary = None
        if split_summary_file.exists():
            with open(split_summary_file, 'r') as f:
                self.split_summary = json.load(f)

        test_split_candidates = [
            self.dataset_path / "metadata" / "test_split.json",
            self.dataset_path / "metadata" / "test-split.json",
        ]
        test_split_file = next((path for path in test_split_candidates if path.exists()), None)
        if test_split_file is not None:
            with open(test_split_file, 'r') as f:
                self.test_pairs = json.load(f)
        else:
            self.test_pairs = self.dataset_metadata.get('pairs', [])
        print(f"✓ Loaded {len(self.test_pairs)} test pairs")
        if self.split_summary:
            print(f"✓ Split summary detected: {json.dumps(self.split_summary.get('unique_hr_per_split', {}))}")

    def initialize_models(self):
        print("\n🔧 Initializing SOTA models...")
        self.initialize_dscf_model()
        self.initialize_seemore_models()
        self.initialize_realesrgan_model()
        self.initialize_swinir_model()
        self.initialize_hat_model()
        self.initialize_stablesr_precomputed()

    def initialize_dscf_model(self):
        dscf_path = self.sota_models_path / "DSCF-SR"
        if not dscf_path.exists(): return
        try:
            from team23_DSCF import DSCF
            model_path = dscf_path / "model_zoo" / "team23_DSCF.pth"
            if not model_path.exists():
                print("⚠ DSCF-SR model weights not found"); return
            
            model = DSCF(3, 3, feature_channels=26, upscale=4)
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict, strict=False)
            model.eval().to(self.device)

            # ★★★ FIX: ONLY COMPILE IF SUPPORTED ★★★
            if self.use_torch_compile:
                try: 
                    model = torch.compile(model)
                    print("✓ DSCF-SR compiled with torch.compile")
                except Exception as e: 
                    print(f"⚠ Could not compile DSCF-SR: {e}")

            self.models['DSCF-SR'] = {
                'model': model,
                'data_range': 1.0,
                'scale': 4,
                'runtime': {'batch_size': 4, 'num_workers': 1, 'empty_cache_every': 8},
            }
            print("✓ DSCF-SR model loaded successfully")
        except Exception as e: print(f"⚠ Error loading DSCF-SR model: {e}")

    def initialize_seemore_models(self):
        seemore_path = self.sota_models_path / "seemoredetails"
        if not seemore_path.exists(): return
        try:
            from basicsr.archs.seemore_arch import SeemoRe
            variants = {
                'SeemoRe_B_X4': {'checkpoint': 'checkpoints/SeemoRe_B_X4/net_g_latest.pth', 'scale': 4, 'num_experts': 3, 'num_layers': 8, 'embedding_dim': 48},
                'SeemoRe_L_X4': {'checkpoint': 'checkpoints/SeemoRe_L_X4/net_g_latest.pth', 'scale': 4, 'num_experts': 3, 'num_layers': 12, 'embedding_dim': 64},
                'SeemoRe_T_X4': {'checkpoint': 'checkpoints/SeemoRe_T_X4/net_g_latest.pth', 'scale': 4, 'num_experts': 3, 'num_layers': 6, 'embedding_dim': 36}
            }
            for name, cfg in variants.items():
                try:
                    ckpt_path = seemore_path / cfg['checkpoint']
                    if not ckpt_path.exists(): print(f"⚠ {name} checkpoint not found"); continue
                    
                    model = SeemoRe(scale=cfg['scale'], in_chans=3, num_experts=cfg['num_experts'], img_range=1.0, num_layers=cfg['num_layers'], embedding_dim=cfg['embedding_dim'], use_shuffle=True, lr_space='exp', topk=1, recursive=2, global_kernel_size=11)
                    ckpt = torch.load(ckpt_path, map_location=self.device)
                    state_dict = ckpt.get('params', ckpt.get('params_ema', ckpt))
                    model.load_state_dict(state_dict)
                    model.eval().to(self.device)
                    
                    # ★★★ FIX: ONLY COMPILE IF SUPPORTED ★★★
                    if self.use_torch_compile:
                        try: 
                            model = torch.compile(model)
                            print(f"✓ {name} compiled with torch.compile")
                        except Exception as e: 
                            print(f"⚠ Could not compile {name}: {e}")
                    
                    self.models[name] = {
                        'model': model,
                        'data_range': 1.0,
                        'scale': cfg['scale'],
                        'runtime': {'batch_size': 4, 'num_workers': 1, 'empty_cache_every': 8},
                    }
                    print(f"✓ {name} model loaded successfully")
                except Exception as e: print(f"⚠ Error loading {name}: {e}")
        except Exception as e: print(f"⚠ Error initializing SeemoRe models: {e}")

    def initialize_realesrgan_model(self):
        realesrgan_path = self.sota_models_path / "Real-ESRGAN"
        if not realesrgan_path.exists(): return
        try:
            from RealESRGAN import RealESRGAN
            model = RealESRGAN(self.device, scale=4)
            model_path = realesrgan_path / "weights" / "RealESRGAN_x4.pth"
            model.load_weights(str(model_path), download=True)
            self.models['RealESRGAN_x4'] = {
                'model': model,
                'data_range': 255.0,
                'scale': 4,
                'runtime': {'batch_size': 1, 'num_workers': 0, 'empty_cache_every': 1},
            }
            print("✓ RealESRGAN_x4 model loaded successfully")
        except Exception as e: print(f"⚠ Error initializing Real-ESRGAN model: {e}")

    def initialize_swinir_model(self):
        """Load SwinIR classical SR x4 model (JingyunLiang/SwinIR)."""
        swinir_path = self.sota_models_path / "SwinIR"
        if not swinir_path.exists(): return
        try:
            from network_swinir import SwinIR as SwinIRNet

            model_path = swinir_path / "model_zoo" / "swinir" / "001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth"
            if not model_path.exists():
                print("⚠ SwinIR model weights not found"); return

            model = SwinIRNet(
                upscale=4, in_chans=3, img_size=64, window_size=8,
                img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180,
                num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2,
                upsampler='pixelshuffle', resi_connection='1conv'
            )
            ckpt = torch.load(model_path, map_location=self.device)
            model.load_state_dict(ckpt.get('params', ckpt), strict=True)
            model.eval().to(self.device)

            wrapped = SwinIRWrapper(model, window_size=8, scale=4)
            if self.use_torch_compile:
                try:
                    wrapped = torch.compile(wrapped)
                    print("✓ SwinIR compiled with torch.compile")
                except Exception as e:
                    print(f"⚠ Could not compile SwinIR: {e}")

            self.models['SwinIR'] = {
                'model': wrapped,
                'data_range': 1.0,
                'scale': 4,
                'runtime': {'batch_size': 2, 'num_workers': 1, 'empty_cache_every': 4},
            }
            print("✓ SwinIR model loaded successfully")
        except Exception as e: print(f"⚠ Error loading SwinIR model: {e}")

    def initialize_hat_model(self):
        """Load HAT x4 model with ImageNet pretraining (XPixelGroup/HAT)."""
        hat_path = self.sota_models_path / "HAT"
        if not hat_path.exists(): return
        try:
            import types
            from timm.models.layers import to_2tuple, trunc_normal_

            class _NoOpRegistry:
                @staticmethod
                def register(obj=None):
                    return obj if obj is not None else (lambda x: x)

            arch_util = types.ModuleType('basicsr.archs.arch_util')
            arch_util.to_2tuple = to_2tuple
            arch_util.trunc_normal_ = trunc_normal_
            registry_mod = types.ModuleType('basicsr.utils.registry')
            registry_mod.ARCH_REGISTRY = _NoOpRegistry()

            for k in [k for k in sys.modules if k == 'basicsr' or k.startswith('basicsr.')]:
                del sys.modules[k]
            sys.modules.update({
                'basicsr': types.ModuleType('basicsr'),
                'basicsr.archs': types.ModuleType('basicsr.archs'),
                'basicsr.archs.arch_util': arch_util,
                'basicsr.utils': types.ModuleType('basicsr.utils'),
                'basicsr.utils.registry': registry_mod,
            })

            import importlib.util
            hat_arch_file = hat_path / "hat" / "archs" / "hat_arch.py"
            if not hat_arch_file.exists():
                print(f"⚠ hat_arch.py not found at {hat_arch_file}"); return
            spec = importlib.util.spec_from_file_location("hat_arch", str(hat_arch_file))
            hat_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(hat_module)
            HAT = hat_module.HAT

            model_path = None
            candidates = [
                hat_path / "experiments" / "pretrained_models" / "HAT_SRx4_ImageNet-pretrain.pth",
                hat_path / "pretrained_models" / "HAT_SRx4_ImageNet-pretrain.pth",
                hat_path / "HAT_SRx4_ImageNet-pretrain.pth",
            ]
            for c in candidates:
                if c.exists():
                    model_path = c; break
            if model_path is None:
                found = list(hat_path.rglob("HAT*SRx4*.pth"))
                if found:
                    model_path = found[0]
            if model_path is None:
                print(f"⚠ HAT model weights not found in {hat_path}"); return
            print(f"  Found HAT checkpoint: {model_path.name}")

            ckpt = torch.load(model_path, map_location=self.device)
            state_dict = ckpt.get('params_ema', ckpt.get('params', ckpt))

            layer_ids = {int(k.split('.')[1]) for k in state_dict
                         if k.startswith('layers.') and '.residual_group.' in k}
            num_groups = max(layer_ids) + 1 if layer_ids else 6
            embed_dim = state_dict.get(
                'layers.0.residual_group.blocks.0.norm1.weight',
                torch.zeros(180)
            ).shape[0]

            model = HAT(
                upscale=4, in_chans=3, img_size=64, window_size=16,
                compress_ratio=3, squeeze_factor=30, conv_scale=0.01,
                overlap_ratio=0.5, img_range=1.,
                depths=[6] * num_groups, embed_dim=embed_dim,
                num_heads=[6] * num_groups, mlp_ratio=2,
                upsampler='pixelshuffle', resi_connection='1conv'
            )
            model.load_state_dict(state_dict, strict=True)
            model.eval().to(self.device)

            hat_variant = model_path.stem.split('_SRx')[0]
            if self.use_torch_compile:
                try:
                    model = torch.compile(model)
                    print(f"✓ {hat_variant} compiled with torch.compile")
                except Exception as e:
                    print(f"⚠ Could not compile {hat_variant}: {e}")

            self.models[hat_variant] = {
                'model': model,
                'data_range': 1.0,
                'scale': 4,
                'runtime': {'batch_size': 1, 'num_workers': 0, 'empty_cache_every': 1},
            }
            print(f"✓ {hat_variant} loaded (groups={num_groups}, embed_dim={embed_dim})")
        except Exception as e: print(f"⚠ Error loading HAT model: {e}")

    def initialize_stablesr_precomputed(self):
        """Register pre-computed StableSR outputs for metric evaluation.

        StableSR uses a full Stable Diffusion pipeline and should be run
        separately via its official inference script. Place the generated
        SR images in {sota_models_path}/StableSR/outputs/ and this method
        will pick them up by matching filenames with the LR test images.
        """
        stablesr_output_dir = self.sota_models_path / "StableSR" / "outputs"
        if not stablesr_output_dir.exists(): return

        sr_lookup = {}
        for ext in ('*.png', '*.jpg', '*.jpeg'):
            for sr_path in stablesr_output_dir.glob(ext):
                sr_lookup[sr_path.stem] = sr_path

        if not sr_lookup:
            print("⚠ StableSR output directory exists but contains no images"); return

        self.models['StableSR'] = {
            'model': sr_lookup,
            'data_range': 255.0,
            'scale': 4,
            'precomputed': True,
            'runtime': {'batch_size': 4, 'num_workers': 1, 'empty_cache_every': 8},
        }
        print(f"✓ StableSR pre-computed outputs registered ({len(sr_lookup)} images)")

    def postprocess_image(self, tensor, data_range=1.0):
        if data_range == 1.0:
            tensor = torch.clamp(tensor, 0, 1) * 255.0
        else:
            tensor = torch.clamp(tensor, 0, 255.0)
        return tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)

    def get_runtime_settings(self, model_name, model_info, batch_size, num_workers):
        runtime = dict(model_info.get('runtime', {}))
        runtime.setdefault('batch_size', batch_size or self.default_batch_size)
        runtime.setdefault('num_workers', num_workers if num_workers is not None else self.default_num_workers)
        runtime.setdefault('empty_cache_every', 8 if self.device.startswith('cuda') else 0)
        return runtime

    def cleanup_cuda(self):
        if self.device.startswith('cuda'):
            torch.cuda.empty_cache()

    def calculate_pair_metrics(self, sr_tensor_metrics, hr_tensor, model_info):
        sr_y = MetricsCalculator.rgb_to_y_channel(sr_tensor_metrics)
        hr_y = MetricsCalculator.rgb_to_y_channel(hr_tensor.float())

        self.psnr_metric_rgb.reset()
        self.ssim_metric_rgb.reset()
        self.psnr_metric_y.reset()
        self.ssim_metric_y.reset()

        metrics = {
            'psnr_rgb': self.psnr_metric_rgb(sr_tensor_metrics, hr_tensor).item(),
            'ssim_rgb': self.ssim_metric_rgb(sr_tensor_metrics, hr_tensor).item(),
            'psnr_y': self.psnr_metric_y(sr_y, hr_y).item(),
            'ssim_y': self.ssim_metric_y(sr_y, hr_y).item(),
            'lpips': MetricsCalculator.calculate_lpips(
                sr_tensor_metrics.squeeze(0) / 255.0,
                hr_tensor.squeeze(0).float() / 255.0,
                self.lpips_fn
            ) if self.lpips_fn is not None else None,
        }
        return metrics

    def build_metric_record(self, batch_idx, item_idx, batch_size, hr_paths, lr_paths, pair_meta, pair_metrics):
        return {
            'pair_index': batch_idx * batch_size + item_idx,
            'pair_id': pair_meta.get('lr_path', lr_paths[item_idx]),
            'hr_path': hr_paths[item_idx],
            'lr_path': lr_paths[item_idx],
            'category': pair_meta.get('category', 'unknown'),
            'difficulty_level': pair_meta.get('degradation_metadata', {}).get('difficulty_level', 'unknown'),
            'variant_index': pair_meta.get('variant_index'),
            **pair_metrics,
        }

    def evaluate_model(self, model_name, model_info, sample_size, batch_size, num_workers):
        runtime = self.get_runtime_settings(model_name, model_info, batch_size, num_workers)
        effective_batch_size = runtime['batch_size']
        effective_num_workers = runtime['num_workers']
        empty_cache_every = runtime['empty_cache_every']

        print(
            f"\n{'='*60}\nEvaluating {model_name} "
            f"(batch_size={effective_batch_size}, num_workers={effective_num_workers})\n{'='*60}"
        )
        
        test_pairs = self.test_pairs[:sample_size] if sample_size else self.test_pairs
        if not test_pairs:
            print("❌ No test pairs to evaluate.")
            return None
             
        dataset = SISRDataset(test_pairs, self.dataset_path, model_info['data_range'])
        dataloader = DataLoader(
            dataset,
            effective_batch_size,
            shuffle=False,
            num_workers=effective_num_workers,
            pin_memory=self.device.startswith('cuda'),
            collate_fn=collate_fn
        )

        all_metrics, failed_pairs, worst_pairs = [], [], []
        model = model_info['model']
        results_dir = Path(f"results/{model_name}")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        success_count = 0
        
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc=f"Evaluating {model_name}")):
            lr_tensors, hr_arrays, hr_paths, lr_paths, pairs_json = batch_data

            if lr_tensors is None:
                continue

            for i in range(len(lr_tensors)):
                try:
                    lr_tensor = lr_tensors[i].unsqueeze(0).to(self.device)
                    hr_array = hr_arrays[i]
                    pair_meta = json.loads(pairs_json[i])
                    
                    with torch.inference_mode():
                        if model_info.get('precomputed'):
                            sr_lookup = model_info['model']
                            lr_stem = Path(lr_paths[i]).stem
                            if lr_stem not in sr_lookup:
                                failed_pairs.append({
                                    'item_index': batch_idx * effective_batch_size + i,
                                    'pair_id': pair_meta.get('lr_path', lr_paths[i]),
                                    'reason': f"Missing precomputed output for {lr_stem}",
                                })
                                continue
                            sr_img = np.array(Image.open(sr_lookup[lr_stem]).convert('RGB'))
                            sr_tensor = torch.from_numpy(sr_img).permute(2, 0, 1).unsqueeze(0).to(self.device).float()
                        elif 'RealESRGAN' in model_name:
                            lr_img_for_esrgan = self.postprocess_image(lr_tensor, model_info['data_range'])
                            sr_array = np.array(model.predict(Image.fromarray(lr_img_for_esrgan)))
                            sr_tensor = torch.from_numpy(sr_array).permute(2, 0, 1).unsqueeze(0).to(self.device).float()
                        else:
                            sr_tensor = model(lr_tensor)

                    hr_tensor = torch.from_numpy(hr_array).permute(2, 0, 1).unsqueeze(0).to(self.device)
                    _, _, h_hr, w_hr = hr_tensor.shape
                    if sr_tensor.shape[2:] != (h_hr, w_hr):
                        sr_tensor = F.interpolate(sr_tensor, size=(h_hr, w_hr), mode='bicubic', align_corners=False)

                    sr_tensor_metrics = (
                        torch.clamp(sr_tensor, 0, 1) * 255.0
                        if model_info['data_range'] == 1.0
                        else torch.clamp(sr_tensor, 0, 255.0)
                    ).float()
                    hr_tensor = hr_tensor.float()

                    pair_metrics = self.calculate_pair_metrics(sr_tensor_metrics, hr_tensor, model_info)
                    metrics = self.build_metric_record(
                        batch_idx,
                        i,
                        effective_batch_size,
                        hr_paths,
                        lr_paths,
                        pair_meta,
                        pair_metrics,
                    )
                    all_metrics.append(metrics)
                    success_count += 1
                    
                    if self.save_worst_comparisons:
                        worst_pairs.append({
                            'metrics': metrics,
                            'sr_img': self.postprocess_image(sr_tensor, model_info['data_range']),
                            'pair': pair_meta,
                            'sort_psnr': metrics['psnr_y'],
                        })
                        worst_pairs.sort(key=lambda x: x['sort_psnr'])
                        if len(worst_pairs) > self.num_worst_to_save:
                            worst_pairs.pop()

                    del lr_tensor, hr_tensor, sr_tensor, sr_tensor_metrics
                    if empty_cache_every and success_count % empty_cache_every == 0:
                        self.cleanup_cuda()

                except Exception as e:
                    print(f"❌ Error on item {i} in batch {batch_idx}: {e}")
                    failed_pairs.append({
                        'item_index': batch_idx * effective_batch_size + i,
                        'pair_id': json.loads(pairs_json[i]).get('lr_path', lr_paths[i]),
                        'reason': str(e),
                    })
                    self.cleanup_cuda()
        
        total_time = time.time() - start_time
        if all_metrics:
            print(f"\n⏱️ Performance for {model_name}: {total_time:.2f}s for {len(all_metrics)} images ({len(all_metrics)/total_time:.2f} img/s)")
            agg_metrics = self.calculate_aggregate_metrics(all_metrics)
            if self.save_worst_comparisons:
                self.save_worst_comparisons_images(model_name, worst_pairs, results_dir)
            self.save_results(model_name, all_metrics, agg_metrics, failed_pairs, results_dir)
            return {
                'aggregate_metrics': agg_metrics,
                'detailed_metrics': all_metrics,
                'failed_pairs': failed_pairs,
                'runtime': {
                    'seconds': total_time,
                    'images_per_second': len(all_metrics) / total_time if total_time > 0 else None,
                    'evaluated_pairs': len(all_metrics),
                    'expected_pairs': len(test_pairs),
                    'batch_size': effective_batch_size,
                    'num_workers': effective_num_workers,
                }
            }
        else:
            print(f"❌ No successful evaluations for {model_name}")
            return None

    def calculate_aggregate_metrics(self, all_metrics):
        agg = {'evaluated_pairs': len(all_metrics)}

        for metric in self.METRIC_NAMES:
            values = [m[metric] for m in all_metrics if m.get(metric) is not None]
            if values:
                agg[f'{metric}_overall'] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                }

        categories = sorted(set(m['category'] for m in all_metrics))
        agg['by_category'] = {}
        for cat in categories:
            cat_metrics = [m for m in all_metrics if m['category'] == cat]
            agg['by_category'][cat] = {'evaluated_pairs': len(cat_metrics)}
            for metric in self.METRIC_NAMES:
                values = [m[metric] for m in cat_metrics if m.get(metric) is not None]
                if values:
                    agg['by_category'][cat][metric] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                    }
        return agg

    def save_results(self, model_name, all_metrics, agg_metrics, failed_pairs, results_dir):
        with open(results_dir / "detailed_metrics.json", 'w') as f: json.dump(all_metrics, f, indent=2)
        with open(results_dir / "aggregate_metrics.json", 'w') as f: json.dump(agg_metrics, f, indent=2)
        with open(results_dir / "failed_pairs.json", 'w') as f: json.dump(failed_pairs, f, indent=2)
        
        print(f"\n📊 {model_name} Summary:")
        if 'psnr_y_overall' in agg_metrics:
            print(f"   PSNR-Y: {agg_metrics['psnr_y_overall']['mean']:.4f} ± {agg_metrics['psnr_y_overall']['std']:.4f}")
        if 'ssim_y_overall' in agg_metrics:
            print(f"   SSIM-Y: {agg_metrics['ssim_y_overall']['mean']:.4f} ± {agg_metrics['ssim_y_overall']['std']:.4f}")
        if 'psnr_rgb_overall' in agg_metrics:
            print(f"   PSNR-RGB: {agg_metrics['psnr_rgb_overall']['mean']:.4f} ± {agg_metrics['psnr_rgb_overall']['std']:.4f}")
        if 'ssim_rgb_overall' in agg_metrics:
            print(f"   SSIM-RGB: {agg_metrics['ssim_rgb_overall']['mean']:.4f} ± {agg_metrics['ssim_rgb_overall']['std']:.4f}")
        if 'lpips_overall' in agg_metrics and agg_metrics['lpips_overall']:
            print(f"   LPIPS: {agg_metrics['lpips_overall']['mean']:.4f} ± {agg_metrics['lpips_overall']['std']:.4f}")
        print(f"   Evaluated pairs: {agg_metrics['evaluated_pairs']}")

    def save_worst_comparisons_images(self, model_name, worst_pairs, results_dir):
        print(f"\n💾 Saving {len(worst_pairs)} worst comparison images for {model_name}...")
        worst_dir = results_dir / "worst_samples"
        worst_dir.mkdir(exist_ok=True)
        for idx, data in enumerate(worst_pairs):
            hr_path = self.dataset_path / data['pair']['hr_path']
            try:
                hr_img = Image.open(hr_path)
                sr_img_pil = Image.fromarray(data['sr_img'])
                comp = Image.new('RGB', (hr_img.width * 2, hr_img.height))
                comp.paste(sr_img_pil, (0, 0))
                comp.paste(hr_img, (hr_img.width, 0))
                draw = ImageDraw.Draw(comp)
                text = (
                    f"PSNR-Y: {data['metrics']['psnr_y']:.2f}, "
                    f"SSIM-Y: {data['metrics']['ssim_y']:.4f}, "
                    f"LPIPS: {data['metrics']['lpips']:.4f}" if data['metrics'].get('lpips') is not None
                    else f"PSNR-Y: {data['metrics']['psnr_y']:.2f}, SSIM-Y: {data['metrics']['ssim_y']:.4f}"
                )
                draw.text((10, 10), text, fill="white")
                comp.save(worst_dir / f"worst_{idx+1:02d}_psnrY_{data['metrics']['psnr_y']:.2f}.png")
            except Exception as e:
                print(f"Could not save worst sample image: {e}")

    def compute_common_subset_results(self, all_results):
        successful_models = [name for name, result in all_results.items() if result and result.get('detailed_metrics')]
        if not successful_models:
            return {}

        common_ids = None
        for model_name in successful_models:
            pair_ids = {record['pair_id'] for record in all_results[model_name]['detailed_metrics']}
            common_ids = pair_ids if common_ids is None else common_ids & pair_ids

        common_results = {}
        for model_name in successful_models:
            filtered = [
                record for record in all_results[model_name]['detailed_metrics']
                if record['pair_id'] in common_ids
            ]
            common_results[model_name] = {
                'aggregate_metrics': self.calculate_aggregate_metrics(filtered),
                'detailed_metrics': filtered,
            }
        return common_results

    def write_paper_ready_tables(self, all_results):
        common_results = self.compute_common_subset_results(all_results)
        if not common_results:
            return

        results_root = Path("results")
        results_root.mkdir(exist_ok=True)

        sorted_models = sorted(common_results.keys())
        sorted_categories = sorted({
            record['category']
            for result in common_results.values()
            for record in result['detailed_metrics']
        })

        payload = {
            'common_subset_size': next(iter(common_results.values()))['aggregate_metrics']['evaluated_pairs'],
            'models': {},
        }
        for model_name in sorted_models:
            payload['models'][model_name] = common_results[model_name]['aggregate_metrics']

        with open(results_root / "paper_ready_tables.json", 'w') as f:
            json.dump(payload, f, indent=2)

        md_lines = [
            f"# Paper-ready Tables",
            "",
            f"Common evaluated subset size: **{payload['common_subset_size']}** pairs.",
            "",
            "## Overall (common subset)",
            "",
            "| Model | PSNR-Y | SSIM-Y | PSNR-RGB | SSIM-RGB | LPIPS |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for model_name in sorted_models:
            agg = common_results[model_name]['aggregate_metrics']
            md_lines.append(
                "| "
                f"{MODEL_DISPLAY_NAMES.get(model_name, model_name)} | "
                f"{agg['psnr_y_overall']['mean']:.4f} | "
                f"{agg['ssim_y_overall']['mean']:.4f} | "
                f"{agg['psnr_rgb_overall']['mean']:.4f} | "
                f"{agg['ssim_rgb_overall']['mean']:.4f} | "
                f"{agg['lpips_overall']['mean']:.4f} |"
            )

        md_lines.extend(["", "## Category-wise PSNR-Y / SSIM-Y / LPIPS (common subset)", ""])
        for category in sorted_categories:
            md_lines.extend([
                f"### {CATEGORY_DISPLAY_NAMES.get(category, category)}",
                "",
                "| Model | PSNR-Y | SSIM-Y | LPIPS |",
                "| --- | ---: | ---: | ---: |",
            ])
            for model_name in sorted_models:
                cat_metrics = common_results[model_name]['aggregate_metrics']['by_category'][category]
                md_lines.append(
                    "| "
                    f"{MODEL_DISPLAY_NAMES.get(model_name, model_name)} | "
                    f"{cat_metrics['psnr_y']['mean']:.4f} | "
                    f"{cat_metrics['ssim_y']['mean']:.4f} | "
                    f"{cat_metrics['lpips']['mean']:.4f} |"
                )
            md_lines.append("")

        with open(results_root / "paper_ready_tables.md", 'w', encoding='utf-8') as f:
            f.write("\n".join(md_lines))

        tex_lines = [
            "% Auto-generated paper-ready tables on the common evaluated subset",
            f"% Common subset size: {payload['common_subset_size']} pairs",
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Overall performance on the corrected test split (common evaluated subset).}",
            "\\begin{tabular}{@{}lccccc@{}}",
            "\\toprule",
            "Method & PSNR-Y$\\uparrow$ & SSIM-Y$\\uparrow$ & PSNR-RGB$\\uparrow$ & SSIM-RGB$\\uparrow$ & LPIPS$\\downarrow$ \\\\",
            "\\midrule",
        ]
        for model_name in sorted_models:
            agg = common_results[model_name]['aggregate_metrics']
            tex_lines.append(
                f"{MODEL_DISPLAY_NAMES.get(model_name, model_name)} & "
                f"{agg['psnr_y_overall']['mean']:.2f} & "
                f"{agg['ssim_y_overall']['mean']:.3f} & "
                f"{agg['psnr_rgb_overall']['mean']:.2f} & "
                f"{agg['ssim_rgb_overall']['mean']:.3f} & "
                f"{agg['lpips_overall']['mean']:.3f} \\\\"
            )
        tex_lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ])

        for category in sorted_categories:
            tex_lines.extend([
                "\\begin{table}[t]",
                "\\centering",
                f"\\caption{{Category-wise performance on {CATEGORY_DISPLAY_NAMES.get(category, category)} (common evaluated subset).}}",
                "\\begin{tabular}{@{}lccc@{}}",
                "\\toprule",
                "Method & PSNR-Y$\\uparrow$ & SSIM-Y$\\uparrow$ & LPIPS$\\downarrow$ \\\\",
                "\\midrule",
            ])
            for model_name in sorted_models:
                cat_metrics = common_results[model_name]['aggregate_metrics']['by_category'][category]
                tex_lines.append(
                    f"{MODEL_DISPLAY_NAMES.get(model_name, model_name)} & "
                    f"{cat_metrics['psnr_y']['mean']:.2f} & "
                    f"{cat_metrics['ssim_y']['mean']:.3f} & "
                    f"{cat_metrics['lpips']['mean']:.3f} \\\\"
                )
            tex_lines.extend([
                "\\bottomrule",
                "\\end{tabular}",
                "\\end{table}",
                "",
            ])

        with open(results_root / "paper_ready_tables.tex", 'w', encoding='utf-8') as f:
            f.write("\n".join(tex_lines))

        print("\n📄 Paper-ready tables saved to:")
        print("   results/paper_ready_tables.json")
        print("   results/paper_ready_tables.md")
        print("   results/paper_ready_tables.tex")

    def evaluate_all_models(self, sample_size, batch_size, num_workers, selected_models=None):
        print(f"\n{'='*80}\nEVALUATING ALL MODELS\n{'='*80}")
        targets = self.models
        if selected_models:
            targets = {k: v for k, v in self.models.items() if k in selected_models}
            skipped = set(selected_models) - set(targets.keys())
            if skipped:
                print(f"⚠ Requested but not loaded: {skipped}")
            print(f"▶ Running: {list(targets.keys())}")
        all_results = {}
        for model_name, model_info in targets.items():
            results = self.evaluate_model(model_name, model_info, sample_size, batch_size, num_workers)
            if results:
                all_results[model_name] = results
        self.generate_comparison_report(all_results)
        self.write_paper_ready_tables(all_results)
        return all_results
    
    def generate_comparison_report(self, all_results):
        print(f"\n{'='*80}\nMODEL COMPARISON REPORT\n{'='*80}")
        if not all_results:
            print("No results to compare.")
            return
        
        print(f"{'Model':<20} {'PSNR-Y':<12} {'SSIM-Y':<12} {'PSNR-RGB':<12} {'SSIM-RGB':<12} {'LPIPS':<12} {'N':<8}")
        print("-" * 96)
        compact_results = {}
        for model, result in sorted(all_results.items()):
            agg = result['aggregate_metrics']
            compact_results[model] = {
                'aggregate_metrics': agg,
                'runtime': result['runtime'],
                'failed_pairs': len(result['failed_pairs']),
            }
            print(
                f"{MODEL_DISPLAY_NAMES.get(model, model):<20} "
                f"{agg['psnr_y_overall']['mean']:<12.4f} "
                f"{agg['ssim_y_overall']['mean']:<12.4f} "
                f"{agg['psnr_rgb_overall']['mean']:<12.4f} "
                f"{agg['ssim_rgb_overall']['mean']:<12.4f} "
                f"{agg['lpips_overall']['mean']:<12.4f} "
                f"{agg['evaluated_pairs']:<8}"
            )
        
        Path("results").mkdir(exist_ok=True)
        with open("results/model_comparison.json", 'w') as f:
            json.dump(compact_results, f, indent=2)
        print(f"\n📄 Detailed comparison saved to: results/model_comparison.json")

def get_configuration():
    """Get configuration for the evaluation. Modify these paths and settings."""
    config = {
        # === IMPORTANT: UPDATE THESE PATHS ===
        'dataset_path': "/kaggle/input/sisr-benchmark-unified",
        'sota_models_path': "/kaggle/input/datasets/vasuaashadesai/testing-models-sisr-cusotm-data/SOTA models",
        
        # === TUNING PARAMETERS ===
        'batch_size': 4,      # Base value. Safer model-specific settings will override this when needed.
        'num_workers': 2,     # Base value. Safer model-specific settings will override this when needed.
        
        # === MODEL SELECTION (for parallel Kaggle runs) ===
        # Set to None to evaluate ALL models, or pass a list of names to evaluate
        # only specific ones. Each parallel notebook can set a different model.
        # Valid names: "DSCF-SR", "SeemoRe_B_X4", "SeemoRe_L_X4", "SeemoRe_T_X4",
        #              "RealESRGAN_x4", "SwinIR", "HAT-L", "StableSR"
        'selected_models': None,  # e.g. ["SwinIR"] or ["HAT-L", "DSCF-SR"]

        # === OTHER SETTINGS ===
        'sample_size': None,  # None = all test pairs, or set to a number (e.g., 100) for testing.
        'device': "cuda" if torch.cuda.is_available() else "cpu",
        'save_all_comparisons': False,
        'save_worst_comparisons': True,
        'num_worst_to_save': 10,
    }
    return config

def main():
    config = get_configuration()
    print(f"{'='*60}\nConfiguration:\n{json.dumps(config, indent=2)}\n{'='*60}")

    if not Path(config['dataset_path']).exists() or not Path(config['sota_models_path']).exists():
        print("❌ Critical Error: Dataset or SOTA models path not found. Please update get_configuration().")
        return

    add_sota_models_to_path(config['sota_models_path'])

    try:
        evaluator = SOTAModelEvaluator(
            sota_models_path=config['sota_models_path'],
            dataset_path=config['dataset_path'],
            device=config['device'],
            save_all_comparisons=config['save_all_comparisons'],
            save_worst_comparisons=config['save_worst_comparisons'],
            num_worst_to_save=config['num_worst_to_save'],
            default_batch_size=config['batch_size'],
            default_num_workers=config['num_workers'],
        )
        
        if not evaluator.models:
            print("❌ Critical Error: No models were loaded. Check paths and initialization logic.")
            return

        evaluator.evaluate_all_models(
            sample_size=config['sample_size'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            selected_models=config['selected_models']
        )
        print(f"\n{'='*80}\n🎉 EVALUATION COMPLETE!\n{'='*80}")

    except Exception as e:
        print(f"❌ An unhandled error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
