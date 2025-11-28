%%writefile config.py
import os
import torch

class Config:
    # Paths
    DATASET_PATH = "/kaggle/input/sisr-benchmark-unified"  # Kaggle path
    # DATASET_PATH = r"C:\Users\DELL\Desktop\code_playground\image deblur\Draft material\datatset creation\dataset" # Local path for testing
    
    SAVE_DIR = "./experiments/DAST_v1"
    
    # Model Architecture
    SCALE = 4
    TRAIN_CROP_SIZE = 64 # LR crop size (HR will be 256)
    EMBED_DIM = 96
    DEPTHS = [6, 6, 6, 6]
    NUM_HEADS = [6, 6, 6, 6]
    WINDOW_SIZE = 8
    MLP_RATIO = 2.0
    
    # Degradation Estimator
    DEG_VECTOR_SIZE = 4  # [Blur, Noise, JPEG, Gamma]
    DEG_EMBED_DIM = 64
    
    # Training
    BATCH_SIZE = 16 # Adjust based on VRAM
    LR = 2e-4
    MIN_LR = 1e-7
    WEIGHT_DECAY = 0.05
    NUM_EPOCHS = 100 # User mentioned "thousands", but we'll set a reasonable default that can be increased
    NUM_WORKERS = 4
    
    # Loss Weights
    LAMBDA_PIXEL = 1.0
    LAMBDA_DEG = 0.5  # Weight for degradation estimation loss
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Normalization constants (approximate max values from generate_sisr_dataset.py)
    MAX_BLUR_KERNEL = 25.0
    MAX_NOISE_STD = 0.1  # Combined estimate
    MAX_GAMMA = 3.0
