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
- Initialize available SOTA models (DSCF-SR, SeemoRe variants, Real-ESRGAN).
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


def add_sota_models_to_path(sota_models_path):
    """Add SOTA models to Python path for imports."""
    if not os.path.isabs(sota_models_path):
        sota_models_path = os.path.abspath(sota_models_path)

    paths_to_add = {
        "DSCF-SR": ["", "utils", "models"],
        "seemoredetails": ["", "basicsr"],
        "Real-ESRGAN": [""]
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
    """Houses the LPIPS calculation logic."""
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

class SOTAModelEvaluator:
    """Evaluate SOTA models on the generated dataset."""

    def __init__(self, sota_models_path, dataset_path, device='cuda', save_all_comparisons=False, save_worst_comparisons=False, num_worst_to_save=5):
        self.sota_models_path = Path(sota_models_path)
        self.dataset_path = Path(dataset_path)
        self.device = device
        self.save_all_comparisons = save_all_comparisons
        self.save_worst_comparisons = save_worst_comparisons
        self.num_worst_to_save = num_worst_to_save
        
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
        
        self.psnr_metric = torchmetrics.PeakSignalNoiseRatio(data_range=255.0).to(self.device)
        self.ssim_metric = torchmetrics.StructuralSimilarityIndexMeasure(data_range=255.0).to(self.device)
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
        with open(metadata_file, 'r') as f: self.dataset_metadata = json.load(f)
        test_split_file = self.dataset_path / "metadata" / "test_split.json"
        if test_split_file.exists():
            with open(test_split_file, 'r') as f: self.test_pairs = json.load(f)
        else:
            self.test_pairs = self.dataset_metadata.get('pairs', [])
        print(f"✓ Loaded {len(self.test_pairs)} test pairs")

    def initialize_models(self):
        print("\n🔧 Initializing SOTA models...")
        self.initialize_dscf_model()
        self.initialize_seemore_models()
        self.initialize_realesrgan_model()

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

            self.models['DSCF-SR'] = {'model': model, 'data_range': 1.0, 'scale': 4}
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
                    
                    self.models[name] = {'model': model, 'data_range': 1.0, 'scale': cfg['scale']}
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
            self.models['RealESRGAN_x4'] = {'model': model, 'data_range': 255.0, 'scale': 4}
            print("✓ RealESRGAN_x4 model loaded successfully")
        except Exception as e: print(f"⚠ Error initializing Real-ESRGAN model: {e}")

    def postprocess_image(self, tensor, data_range=1.0):
        if data_range == 1.0:
            tensor = torch.clamp(tensor, 0, 1) * 255.0
        else:
            tensor = torch.clamp(tensor, 0, 255.0)
        return tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)

    def evaluate_model(self, model_name, model_info, sample_size, batch_size, num_workers):
        print(f"\n{'='*60}\nEvaluating {model_name} with batch_size={batch_size}\n{'='*60}")
        
        test_pairs = self.test_pairs[:sample_size] if sample_size else self.test_pairs
        if not test_pairs: print("❌ No test pairs to evaluate."); return None
            
        dataset = SISRDataset(test_pairs, self.dataset_path, model_info['data_range'])
        dataloader = DataLoader(dataset, batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)

        all_metrics, failed_pairs, worst_pairs = [], [], []
        model = model_info['model']
        results_dir = Path(f"results/{model_name}"); results_dir.mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc=f"Evaluating {model_name}")):
            lr_tensors, hr_arrays, hr_paths, lr_paths, pairs_json = batch_data

            if lr_tensors is None: continue

            for i in range(len(lr_tensors)):
                try:
                    lr_tensor = lr_tensors[i].unsqueeze(0).to(self.device)
                    hr_array = hr_arrays[i]
                    pair_meta = json.loads(pairs_json[i])
                    
                    with torch.no_grad():
                        if 'RealESRGAN' in model_name:
                            lr_img_for_esrgan = self.postprocess_image(lr_tensor, model_info['data_range'])
                            sr_array = np.array(model.predict(Image.fromarray(lr_img_for_esrgan)))
                            sr_tensor = torch.from_numpy(sr_array).permute(2, 0, 1).unsqueeze(0).to(self.device).float()
                        else:
                            sr_tensor = model(lr_tensor)

                    hr_tensor = torch.from_numpy(hr_array).permute(2, 0, 1).unsqueeze(0).to(self.device)
                    _, _, h_hr, w_hr = hr_tensor.shape
                    if sr_tensor.shape[2:] != (h_hr, w_hr):
                         sr_tensor = F.interpolate(sr_tensor, size=(h_hr, w_hr), mode='bicubic', align_corners=False)

                    sr_tensor_metrics = (torch.clamp(sr_tensor, 0, 1) * 255.0) if model_info['data_range'] == 1.0 else torch.clamp(sr_tensor, 0, 255)

                    psnr = self.psnr_metric(sr_tensor_metrics, hr_tensor).item()
                    ssim = self.ssim_metric(sr_tensor_metrics, hr_tensor).item()
                    lpips_val = MetricsCalculator.calculate_lpips(
                        sr_tensor.squeeze(0) / (255.0 if model_info['data_range'] == 255.0 else 1.0),
                        hr_tensor.squeeze(0).float() / 255.0,
                        self.lpips_fn
                    )

                    metrics = {
                        'pair_index': batch_idx * batch_size + i,
                        'psnr': psnr, 'ssim': ssim, 'lpips': lpips_val,
                        'hr_path': hr_paths[i], 'category': pair_meta.get('category', 'unknown')
                    }
                    all_metrics.append(metrics)
                    
                    if self.save_worst_comparisons:
                        worst_pairs.append({
                            'metrics': metrics,
                            'sr_img': self.postprocess_image(sr_tensor, model_info['data_range']),
                            'pair': pair_meta, 'psnr': psnr
                        })
                        worst_pairs.sort(key=lambda x: x['psnr'])
                        if len(worst_pairs) > self.num_worst_to_save: worst_pairs.pop()

                except Exception as e:
                    print(f"❌ Error on item {i} in batch {batch_idx}: {e}"); failed_pairs.append({'item_index': batch_idx * batch_size + i, 'reason': str(e)})
        
        total_time = time.time() - start_time
        if all_metrics:
            print(f"\n⏱️ Performance for {model_name}: {total_time:.2f}s for {len(all_metrics)} images ({len(all_metrics)/total_time:.2f} img/s)")
            agg_metrics = self.calculate_aggregate_metrics(all_metrics)
            if self.save_worst_comparisons: self.save_worst_comparisons_images(model_name, worst_pairs, results_dir)
            self.save_results(model_name, all_metrics, agg_metrics, failed_pairs, results_dir)
            return agg_metrics
        else:
            print(f"❌ No successful evaluations for {model_name}"); return None

    def calculate_aggregate_metrics(self, all_metrics):
        agg = {}
        metrics_names = [k for k in all_metrics[0] if isinstance(all_metrics[0][k], (int, float))]
        
        for metric in metrics_names:
            values = [m[metric] for m in all_metrics if m.get(metric) is not None]
            if values: agg[f'{metric}_overall'] = {'mean': np.mean(values), 'std': np.std(values)}
        
        categories = set(m['category'] for m in all_metrics)
        agg['by_category'] = {}
        for cat in categories:
            cat_metrics = [m for m in all_metrics if m['category'] == cat]
            agg['by_category'][cat] = {}
            for metric in metrics_names:
                values = [m[metric] for m in cat_metrics if m.get(metric) is not None]
                if values: agg['by_category'][cat][metric] = {'mean': np.mean(values), 'std': np.std(values)}
        return agg

    def save_results(self, model_name, all_metrics, agg_metrics, failed_pairs, results_dir):
        with open(results_dir / "detailed_metrics.json", 'w') as f: json.dump(all_metrics, f, indent=2)
        with open(results_dir / "aggregate_metrics.json", 'w') as f: json.dump(agg_metrics, f, indent=2)
        with open(results_dir / "failed_pairs.json", 'w') as f: json.dump(failed_pairs, f, indent=2)
        
        print(f"\n📊 {model_name} Summary:")
        if 'psnr_overall' in agg_metrics: print(f"   PSNR: {agg_metrics['psnr_overall']['mean']:.4f} ± {agg_metrics['psnr_overall']['std']:.4f}")
        if 'ssim_overall' in agg_metrics: print(f"   SSIM: {agg_metrics['ssim_overall']['mean']:.4f} ± {agg_metrics['ssim_overall']['std']:.4f}")
        if 'lpips_overall' in agg_metrics and agg_metrics['lpips_overall']: print(f"   LPIPS: {agg_metrics['lpips_overall']['mean']:.4f} ± {agg_metrics['lpips_overall']['std']:.4f}")

    def save_worst_comparisons_images(self, model_name, worst_pairs, results_dir):
        print(f"\n💾 Saving {len(worst_pairs)} worst comparison images for {model_name}...")
        worst_dir = results_dir / "worst_samples"; worst_dir.mkdir(exist_ok=True)
        for idx, data in enumerate(worst_pairs):
            hr_path = self.dataset_path / data['pair']['hr_path']
            try:
                hr_img = Image.open(hr_path); sr_img_pil = Image.fromarray(data['sr_img'])
                comp = Image.new('RGB', (hr_img.width * 2, hr_img.height))
                comp.paste(sr_img_pil, (0, 0)); comp.paste(hr_img, (hr_img.width, 0))
                draw = ImageDraw.Draw(comp)
                text = f"PSNR: {data['psnr']:.2f}, SSIM: {data['metrics']['ssim']:.4f}"
                draw.text((10, 10), text, fill="white")
                comp.save(worst_dir / f"worst_{idx+1:02d}_psnr_{data['psnr']:.2f}.png")
            except Exception as e: print(f"Could not save worst sample image: {e}")

    def evaluate_all_models(self, sample_size, batch_size, num_workers):
        print(f"\n{'='*80}\nEVALUATING ALL MODELS\n{'='*80}")
        all_results = {}
        for model_name, model_info in self.models.items():
            results = self.evaluate_model(model_name, model_info, sample_size, batch_size, num_workers)
            if results: all_results[model_name] = results
        self.generate_comparison_report(all_results)
        return all_results
    
    def generate_comparison_report(self, all_results):
        print(f"\n{'='*80}\nMODEL COMPARISON REPORT\n{'='*80}")
        if not all_results: print("No results to compare."); return
        
        print(f"{'Model':<20} {'PSNR':<20} {'SSIM':<20} {'LPIPS':<20}")
        print("-" * 80)
        for model, results in sorted(all_results.items()):
            psnr = f"{results.get('psnr_overall', {}).get('mean', 0):.4f}"
            ssim = f"{results.get('ssim_overall', {}).get('mean', 0):.4f}"
            lpips = f"{results.get('lpips_overall', {}).get('mean', 0):.4f}" if results.get('lpips_overall') else 'N/A'
            print(f"{model:<20} {psnr:<20} {ssim:<20} {lpips:<20}")
        
        Path("results").mkdir(exist_ok=True)
        with open("results/model_comparison.json", 'w') as f: json.dump(all_results, f, indent=2)
        print(f"\n📄 Detailed comparison saved to: results/model_comparison.json")

def get_configuration():
    """Get configuration for the evaluation. Modify these paths and settings."""
    config = {
        # === IMPORTANT: UPDATE THESE PATHS ===
        'dataset_path': "/kaggle/input/sisr-benchmark-unified",
        'sota_models_path': "/kaggle/input/testing-models-sisr-cusotm-data/SOTA models",
        
        # === TUNING PARAMETERS ===
        'batch_size': 16,     # Adjust based on VRAM and num_workers. Larger is better for I/O.
        'num_workers': 2,     # Number of CPU cores for data loading. Kaggle usually has 2-4.
        
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
            num_worst_to_save=config['num_worst_to_save']
        )
        
        if not evaluator.models:
            print("❌ Critical Error: No models were loaded. Check paths and initialization logic.")
            return

        evaluator.evaluate_all_models(
            sample_size=config['sample_size'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers']
        )
        print(f"\n{'='*80}\n🎉 EVALUATION COMPLETE!\n{'='*80}")

    except Exception as e:
        print(f"❌ An unhandled error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()