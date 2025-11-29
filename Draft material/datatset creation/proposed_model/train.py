
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from pathlib import Path

from config import Config
from dataset import SISRDataset
from model import DAST

def train():
    # Setup
    Path(Config.SAVE_DIR).mkdir(parents=True, exist_ok=True)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")
    
    # Data
    train_dataset = SISRDataset(Config.DATASET_PATH, split='train')
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, 
                              num_workers=Config.NUM_WORKERS, pin_memory=True)
    
    print(f"Training on {len(train_dataset)} samples")
    
    # Model
    model = DAST(Config).to(device)
    
    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS, eta_min=Config.MIN_LR)
    
    # Losses
    criterion_pixel = nn.L1Loss()
    criterion_deg = nn.MSELoss()
    
    # Training Loop
    best_loss = float('inf')
    
    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        epoch_pixel_loss = 0
        epoch_deg_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.NUM_EPOCHS}")
        
        for batch in pbar:
            lr = batch['lr'].to(device)
            hr = batch['hr'].to(device)
            deg_gt = batch['deg_vec'].to(device)
            
            optimizer.zero_grad()
            
            # Forward
            sr, deg_pred = model(lr)
            
            # Loss
            loss_pixel = criterion_pixel(sr, hr)
            loss_deg = criterion_deg(deg_pred, deg_gt)
            
            total_loss = Config.LAMBDA_PIXEL * loss_pixel + Config.LAMBDA_DEG * loss_deg
            
            # Backward
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Logging
            epoch_loss += total_loss.item()
            epoch_pixel_loss += loss_pixel.item()
            epoch_deg_loss += loss_deg.item()
            
            pbar.set_postfix({
                'Loss': f"{total_loss.item():.4f}",
                'Pix': f"{loss_pixel.item():.4f}",
                'Deg': f"{loss_deg.item():.4f}"
            })
            
        scheduler.step()
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.5f}")
        
        # Save Checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, f"{Config.SAVE_DIR}/best_model.pth")
            print("Saved Best Model")
            
        # Regular save
        if (epoch + 1) % 10 == 0:
             torch.save(model.state_dict(), f"{Config.SAVE_DIR}/epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train()
