
from proposed_model.dataset import SISRDataset
from proposed_model.config import Config
from torch.utils.data import DataLoader
import torch

def verify_dataset():
    print("Verifying dataset loading...")
    try:
        dataset = SISRDataset(Config.DATASET_PATH, split='train')
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        batch = next(iter(loader))
        lr = batch['lr']
        hr = batch['hr']
        deg = batch['deg_vec']
        
        print(f"Batch loaded successfully.")
        print(f"LR Shape: {lr.shape}")
        print(f"HR Shape: {hr.shape}")
        print(f"Degradation Vector Shape: {deg.shape}")
        
        expected_lr_size = (4, 3, Config.TRAIN_CROP_SIZE, Config.TRAIN_CROP_SIZE)
        expected_hr_size = (4, 3, Config.TRAIN_CROP_SIZE * Config.SCALE, Config.TRAIN_CROP_SIZE * Config.SCALE)
        
        if lr.shape == expected_lr_size and hr.shape == expected_hr_size:
            print("✓ Shapes are correct.")
        else:
            print(f"❌ Shape mismatch! Expected LR {expected_lr_size}, got {lr.shape}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_dataset()
