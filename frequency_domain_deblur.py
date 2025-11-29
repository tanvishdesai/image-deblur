import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("FREQUENCY DOMAIN DEBLURRING NETWORK (FourierNet)")
print("Revolutionary Architecture: Processing in Fourier Domain")
print("Novel Approach: Global Receptive Field Through Frequency Analysis")
print("="*80)

# ============================================================================
# SECTION 1: COMPLEX-VALUED NEURAL NETWORK COMPONENTS
# ============================================================================
print("\n[SECTION 1] Building complex-valued neural network components...")

class ComplexLinear(nn.Module):
    """Linear layer for complex-valued data"""
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Real and imaginary weight matrices
        self.weight_real = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_imag = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        
        if bias:
            self.bias_real = nn.Parameter(torch.zeros(out_features))
            self.bias_imag = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias_real', None)
            self.register_parameter('bias_imag', None)
    
    def forward(self, input_complex):
        """
        Complex linear transformation: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
        """
        real, imag = input_complex
        
        # Complex matrix multiplication
        output_real = F.linear(real, self.weight_real) - F.linear(imag, self.weight_imag)
        output_imag = F.linear(real, self.weight_imag) + F.linear(imag, self.weight_real)
        
        # Add bias if present
        if self.bias_real is not None:
            output_real += self.bias_real
            output_imag += self.bias_imag
        
        return output_real, output_imag

class ComplexReLU(nn.Module):
    """ReLU activation for complex numbers (applies to both real and imaginary parts)"""
    
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
    
    def forward(self, input_complex):
        real, imag = input_complex
        return self.relu(real), self.relu(imag)

class ComplexModReLU(nn.Module):
    """Modified ReLU that preserves phase information"""
    
    def __init__(self, bias=0.1):
        super().__init__()
        self.bias = bias
    
    def forward(self, input_complex):
        real, imag = input_complex
        
        # Calculate magnitude and phase
        magnitude = torch.sqrt(real**2 + imag**2 + 1e-8)
        phase_real = real / (magnitude + 1e-8)
        phase_imag = imag / (magnitude + 1e-8)
        
        # Apply ReLU to magnitude with bias
        magnitude_activated = F.relu(magnitude + self.bias)
        
        # Reconstruct complex number
        output_real = magnitude_activated * phase_real
        output_imag = magnitude_activated * phase_imag
        
        return output_real, output_imag

class ComplexConv2d(nn.Module):
    """2D Convolution for complex-valued data"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Real and imaginary convolution layers
        self.conv_real = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.conv_imag = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        
        if bias:
            self.bias_real = nn.Parameter(torch.zeros(out_channels, 1, 1))
            self.bias_imag = nn.Parameter(torch.zeros(out_channels, 1, 1))
        else:
            self.register_parameter('bias_real', None)
            self.register_parameter('bias_imag', None)
    
    def forward(self, input_complex):
        real, imag = input_complex
        
        # Complex convolution: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
        output_real = self.conv_real(real) - self.conv_imag(imag)
        output_imag = self.conv_real(imag) + self.conv_imag(real)
        
        if self.bias_real is not None:
            output_real += self.bias_real
            output_imag += self.bias_imag
        
        return output_real, output_imag

# ============================================================================
# SECTION 2: FREQUENCY DOMAIN PROCESSING MODULES
# ============================================================================
print("\n[SECTION 2] Building frequency domain processing modules...")

class FrequencyProcessor(nn.Module):
    """Core frequency domain processing unit"""
    
    def __init__(self, num_channels=3, hidden_dim=64):
        super().__init__()
        self.num_channels = num_channels
        self.hidden_dim = hidden_dim
        
        # Complex-valued processing layers
        self.freq_processor = nn.ModuleList([
            ComplexConv2d(num_channels, hidden_dim, 3, padding=1),
            ComplexModReLU(),
            ComplexConv2d(hidden_dim, hidden_dim, 3, padding=1),
            ComplexModReLU(),
            ComplexConv2d(hidden_dim, num_channels, 3, padding=1)
        ])
        
        # Learnable frequency mask (complex-valued)
        self.freq_mask_real = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.freq_mask_imag = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        
        print(f"FrequencyProcessor initialized: {num_channels} channels, {hidden_dim} hidden dim")
    
    def forward(self, freq_real, freq_imag):
        """Process complex frequency domain data"""
        
        # Apply learnable frequency mask
        masked_real = freq_real * self.freq_mask_real - freq_imag * self.freq_mask_imag
        masked_imag = freq_real * self.freq_mask_imag + freq_imag * self.freq_mask_real
        
        # Process through complex network layers
        x_real, x_imag = masked_real, masked_imag
        
        for layer in self.freq_processor:
            if isinstance(layer, (ComplexConv2d, ComplexLinear)):
                x_real, x_imag = layer((x_real, x_imag))
            elif isinstance(layer, (ComplexReLU, ComplexModReLU)):
                x_real, x_imag = layer((x_real, x_imag))
        
        return x_real, x_imag

class AdaptiveFrequencyFilter(nn.Module):
    """Learnable frequency-selective filter"""
    
    def __init__(self, num_channels=3):
        super().__init__()
        self.num_channels = num_channels
        
        # Learnable frequency response (both magnitude and phase)
        self.magnitude_filter = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.phase_filter = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        
        # Frequency-dependent scaling
        self.freq_scaling = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        
    def forward(self, freq_real, freq_imag):
        """Apply adaptive frequency filtering"""
        
        # Calculate current magnitude and phase
        magnitude = torch.sqrt(freq_real**2 + freq_imag**2 + 1e-8)
        phase = torch.atan2(freq_imag, freq_real + 1e-8)
        
        # Apply learnable magnitude and phase adjustments
        new_magnitude = magnitude * self.magnitude_filter * self.freq_scaling
        new_phase = phase + self.phase_filter
        
        # Convert back to real and imaginary
        filtered_real = new_magnitude * torch.cos(new_phase)
        filtered_imag = new_magnitude * torch.sin(new_phase)
        
        return filtered_real, filtered_imag

class MultiScaleFrequencyProcessor(nn.Module):
    """Multi-scale processing in frequency domain"""
    
    def __init__(self, num_channels=3, scales=[1, 2, 4]):
        super().__init__()
        self.scales = scales
        
        # Processors for different frequency scales
        self.processors = nn.ModuleList([
            FrequencyProcessor(num_channels, 32) for _ in scales
        ])
        
        # Scale-specific filters
        self.scale_filters = nn.ModuleList([
            AdaptiveFrequencyFilter(num_channels) for _ in scales
        ])
        
        # Combination weights
        self.scale_weights = nn.Parameter(torch.ones(len(scales)))
        
    def forward(self, freq_real, freq_imag):
        """Multi-scale frequency processing"""
        
        scale_outputs_real = []
        scale_outputs_imag = []
        
        for i, (scale, processor, filter_layer) in enumerate(zip(self.scales, self.processors, self.scale_filters)):
            # Downsample frequency representation
            if scale > 1:
                h, w = freq_real.shape[-2:]
                scale_real = F.interpolate(freq_real, size=(h//scale, w//scale), mode='bilinear')
                scale_imag = F.interpolate(freq_imag, size=(h//scale, w//scale), mode='bilinear')
            else:
                scale_real, scale_imag = freq_real, freq_imag
            
            # Process at this scale
            processed_real, processed_imag = processor(scale_real, scale_imag)
            filtered_real, filtered_imag = filter_layer(processed_real, processed_imag)
            
            # Upsample back if needed
            if scale > 1:
                h, w = freq_real.shape[-2:]
                filtered_real = F.interpolate(filtered_real, size=(h, w), mode='bilinear')
                filtered_imag = F.interpolate(filtered_imag, size=(h, w), mode='bilinear')
            
            scale_outputs_real.append(filtered_real)
            scale_outputs_imag.append(filtered_imag)
        
        # Weighted combination of scales
        weights = F.softmax(self.scale_weights, dim=0)
        output_real = sum(w * out for w, out in zip(weights, scale_outputs_real))
        output_imag = sum(w * out for w, out in zip(weights, scale_outputs_imag))
        
        return output_real, output_imag

# ============================================================================
# SECTION 3: FOURIERNET ARCHITECTURE
# ============================================================================
print("\n[SECTION 3] Building FourierNet Architecture...")

class FourierNet(nn.Module):
    """
    Revolutionary FourierNet: Deblurring primarily in frequency domain
    
    Key innovation: Operates almost entirely in Fourier space where:
    - Convolution becomes element-wise multiplication
    - Global receptive field by definition
    - Deblurring is theoretically division in frequency domain
    """
    
    def __init__(self, num_channels=3, num_freq_blocks=6):
        super().__init__()
        print(f"Initializing FourierNet with {num_freq_blocks} frequency processing blocks...")
        
        self.num_channels = num_channels
        self.num_freq_blocks = num_freq_blocks
        
        # Input preprocessing (minimal spatial processing)
        self.input_prep = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, 3, padding=1)
        )
        
        # Multi-scale frequency processors
        self.freq_blocks = nn.ModuleList([
            MultiScaleFrequencyProcessor(num_channels) for _ in range(num_freq_blocks)
        ])
        
        # Learnable deconvolution mask (the core innovation)
        self.deconv_mask_real = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.deconv_mask_imag = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        
        # Frequency domain regularization
        self.freq_regularizer = AdaptiveFrequencyFilter(num_channels)
        
        # Output refinement (minimal spatial processing)
        self.output_refine = nn.Sequential(
            nn.Conv2d(num_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, num_channels, 3, padding=1),
            nn.Tanh()
        )
        
        # Learnable mixing of frequency and spatial processing
        self.freq_spatial_weight = nn.Parameter(torch.tensor(0.8))  # Favor frequency processing
        
        print("FourierNet initialized successfully!")
        self._print_network_info()
    
    def _print_network_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frequency blocks: {self.num_freq_blocks}")
        print("Architecture: Frequency-Domain Dominant Processing")
    
    def forward(self, blurred_image):
        """
        Main forward pass: Frequency domain processing pipeline
        """
        batch_size, channels, height, width = blurred_image.shape
        
        # Minimal input preprocessing
        spatial_features = self.input_prep(blurred_image)
        
        # Convert to frequency domain (the key step)
        freq_complex = torch.fft.fft2(spatial_features)
        freq_real = freq_complex.real
        freq_imag = freq_complex.imag
        
        # Store original frequency content for residual connection
        orig_freq_real, orig_freq_imag = freq_real.clone(), freq_imag.clone()
        
        # Multi-block frequency domain processing
        for i, freq_block in enumerate(self.freq_blocks):
            freq_real, freq_imag = freq_block(freq_real, freq_imag)
            
            # Residual connections in frequency domain
            if i > 0:  # Skip first block for residual
                freq_real = freq_real + 0.1 * orig_freq_real
                freq_imag = freq_imag + 0.1 * orig_freq_imag
        
        # Apply learnable deconvolution mask (core deblurring operation)
        # In frequency domain: Deblurring ≈ Y(f) / H(f) where H(f) is blur kernel
        deconv_real = freq_real * self.deconv_mask_real - freq_imag * self.deconv_mask_imag
        deconv_imag = freq_real * self.deconv_mask_imag + freq_imag * self.deconv_mask_real
        
        # Frequency domain regularization (prevent amplification of noise)
        reg_real, reg_imag = self.freq_regularizer(deconv_real, deconv_imag)
        
        # Convert back to spatial domain
        deblurred_complex = torch.complex(reg_real, reg_imag)
        deblurred_spatial = torch.fft.ifft2(deblurred_complex).real
        
        # Minimal output refinement
        spatial_refined = self.output_refine(deblurred_spatial)
        
        # Learnable mixing of frequency-processed and input features
        freq_weight = torch.sigmoid(self.freq_spatial_weight)
        output = freq_weight * spatial_refined + (1 - freq_weight) * blurred_image
        
        # Return detailed outputs for analysis
        return {
            'output': output,
            'freq_processed': deblurred_spatial,
            'freq_real': reg_real,
            'freq_imag': reg_imag,
            'deconv_mask_real': self.deconv_mask_real,
            'deconv_mask_imag': self.deconv_mask_imag,
            'freq_weight': freq_weight
        }

# ============================================================================
# SECTION 4: FREQUENCY-AWARE LOSS FUNCTIONS
# ============================================================================
print("\n[SECTION 4] Setting up frequency-aware loss functions...")

class FrequencyAwareLoss(nn.Module):
    """Loss function that incorporates frequency domain knowledge"""
    
    def __init__(self, lambda_spatial=1.0, lambda_freq=0.5, lambda_perceptual=0.1):
        super().__init__()
        self.lambda_spatial = lambda_spatial
        self.lambda_freq = lambda_freq
        self.lambda_perceptual = lambda_perceptual
        
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        
        print(f"Frequency-Aware Loss initialized:")
        print(f"- Spatial loss weight: {lambda_spatial}")
        print(f"- Frequency loss weight: {lambda_freq}")
        print(f"- Perceptual loss weight: {lambda_perceptual}")
    
    def forward(self, outputs, target):
        """
        Calculate frequency-aware loss
        """
        predicted = outputs['output']
        
        # 1. Spatial domain loss
        spatial_loss = self.l1_loss(predicted, target)
        
        # 2. Frequency domain loss
        freq_loss = self.compute_frequency_loss(predicted, target)
        
        # 3. Perceptual loss (high-frequency content preservation)
        perceptual_loss = self.compute_perceptual_loss(predicted, target)
        
        total_loss = (self.lambda_spatial * spatial_loss + 
                     self.lambda_freq * freq_loss +
                     self.lambda_perceptual * perceptual_loss)
        
        return {
            'total': total_loss,
            'spatial': spatial_loss,
            'frequency': freq_loss,
            'perceptual': perceptual_loss
        }
    
    def compute_frequency_loss(self, pred, target):
        """Loss in frequency domain"""
        # Convert to frequency domain
        pred_freq = torch.fft.fft2(pred)
        target_freq = torch.fft.fft2(target)
        
        # Loss on magnitude spectrum (more important for visual quality)
        pred_magnitude = torch.abs(pred_freq)
        target_magnitude = torch.abs(target_freq)
        magnitude_loss = self.l1_loss(pred_magnitude, target_magnitude)
        
        # Loss on phase (important for detail preservation)
        pred_phase = torch.angle(pred_freq)
        target_phase = torch.angle(target_freq)
        phase_loss = self.l1_loss(torch.sin(pred_phase), torch.sin(target_phase))
        
        return magnitude_loss + 0.1 * phase_loss
    
    def compute_perceptual_loss(self, pred, target):
        """Simple perceptual loss using gradients (high-frequency content)"""
        # Sobel filters for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=pred.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=pred.device)
        
        sobel_x = sobel_x.view(1, 1, 3, 3).repeat(pred.shape[1], 1, 1, 1)
        sobel_y = sobel_y.view(1, 1, 3, 3).repeat(pred.shape[1], 1, 1, 1)
        
        # Calculate gradients
        pred_grad_x = F.conv2d(pred, sobel_x, padding=1, groups=pred.shape[1])
        pred_grad_y = F.conv2d(pred, sobel_y, padding=1, groups=pred.shape[1])
        target_grad_x = F.conv2d(target, sobel_x, padding=1, groups=target.shape[1])
        target_grad_y = F.conv2d(target, sobel_y, padding=1, groups=target.shape[1])
        
        # Gradient magnitude
        pred_grad_mag = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + 1e-8)
        target_grad_mag = torch.sqrt(target_grad_x**2 + target_grad_y**2 + 1e-8)
        
        return self.l1_loss(pred_grad_mag, target_grad_mag)

# ============================================================================
# SECTION 5: TRAINING UTILITIES
# ============================================================================
print("\n[SECTION 5] Setting up training utilities...")

class FourierNetTrainer:
    """Comprehensive trainer for FourierNet"""
    
    def __init__(self, model, device, config):
        self.model = model.to(device)
        self.device = device
        self.config = config
        
        # Optimizer (using different LR for frequency and spatial components)
        freq_params = []
        spatial_params = []
        
        for name, param in model.named_parameters():
            if 'freq' in name or 'deconv_mask' in name:
                freq_params.append(param)
            else:
                spatial_params.append(param)
        
        self.optimizer = optim.Adam([
            {'params': freq_params, 'lr': config['learning_rate'] * 0.5},  # Lower LR for frequency components
            {'params': spatial_params, 'lr': config['learning_rate']}
        ], weight_decay=1e-5)
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config['num_epochs']
        )
        
        self.criterion = FrequencyAwareLoss(
            lambda_spatial=config.get('lambda_spatial', 1.0),
            lambda_freq=config.get('lambda_freq', 0.5),
            lambda_perceptual=config.get('lambda_perceptual', 0.1)
        ).to(device)
        
        print("FourierNet trainer initialized with frequency-aware optimization!")
    
    def train_epoch(self, train_loader, epoch):
        """Train one epoch"""
        self.model.train()
        epoch_loss = 0.0
        epoch_loss_components = {'spatial': 0.0, 'frequency': 0.0, 'perceptual': 0.0}
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"FourierNet Epoch {epoch}", leave=False)
        
        for batch_idx, (blur_imgs, sharp_imgs) in enumerate(progress_bar):
            blur_imgs = blur_imgs.to(self.device)
            sharp_imgs = sharp_imgs.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(blur_imgs)
            
            # Calculate loss
            loss_dict = self.criterion(outputs, sharp_imgs)
            total_loss = loss_dict['total']
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping for frequency components
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track losses
            epoch_loss += total_loss.item()
            for key in epoch_loss_components:
                epoch_loss_components[key] += loss_dict[key].item()
            num_batches += 1
            
            progress_bar.set_postfix({
                'Loss': f"{total_loss.item():.4f}",
                'Freq': f"{loss_dict['frequency'].item():.4f}"
            })
        
        progress_bar.close()
        
        # Update learning rate
        self.scheduler.step()
        
        # Calculate averages
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        avg_components = {k: v / num_batches for k, v in epoch_loss_components.items()}
        
        return avg_loss, avg_components

def create_fourier_visualization(model, test_loader, device, save_path='visualizations/fourier_analysis.png'):
    """Create comprehensive FourierNet visualization"""
    
    model.eval()
    with torch.no_grad():
        for blur_imgs, sharp_imgs in test_loader:
            blur_imgs = blur_imgs.to(device)
            sharp_imgs = sharp_imgs.to(device)
            
            # Get detailed outputs
            outputs = model(blur_imgs[:1])
            
            # Extract visualizations
            sample_blur = blur_imgs[0]
            sample_sharp = sharp_imgs[0] 
            sample_pred = outputs['output'][0]
            freq_real = outputs['freq_real'][0]
            freq_imag = outputs['freq_imag'][0]
            
            # Create visualization
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            
            # Denormalize for display
            def denorm(tensor):
                return torch.clamp((tensor + 1) / 2, 0, 1).cpu().permute(1, 2, 0).numpy()
            
            # Spatial domain results
            axes[0, 0].imshow(denorm(sample_blur))
            axes[0, 0].set_title('Input (Blurred)')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(denorm(sample_pred))
            axes[0, 1].set_title('FourierNet Output')
            axes[0, 1].axis('off')
            
            axes[0, 2].imshow(denorm(sample_sharp))
            axes[0, 2].set_title('Ground Truth')
            axes[0, 2].axis('off')
            
            # Frequency domain analysis
            freq_magnitude = torch.sqrt(freq_real**2 + freq_imag**2)[0].cpu().numpy()
            axes[0, 3].imshow(np.log(freq_magnitude + 1), cmap='hot')
            axes[0, 3].set_title('Frequency Magnitude (log)')
            axes[0, 3].axis('off')
            
            # Input vs Output frequency comparison
            input_freq = torch.fft.fft2(sample_blur)
            input_mag = torch.abs(input_freq)[0].cpu().numpy()
            output_freq = torch.fft.fft2(sample_pred)
            output_mag = torch.abs(output_freq)[0].cpu().numpy()
            
            axes[1, 0].imshow(np.log(input_mag + 1), cmap='viridis')
            axes[1, 0].set_title('Input Frequency Spectrum')
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(np.log(output_mag + 1), cmap='viridis')
            axes[1, 1].set_title('Output Frequency Spectrum')
            axes[1, 1].axis('off')
            
            # Learned deconvolution masks
            deconv_real = outputs['deconv_mask_real'][0, 0].cpu().numpy()
            deconv_imag = outputs['deconv_mask_imag'][0, 0].cpu().numpy()
            deconv_mag = np.sqrt(deconv_real**2 + deconv_imag**2)
            
            axes[1, 2].imshow(deconv_mag, cmap='hot')
            axes[1, 2].set_title('Learned Deconv Mask')
            axes[1, 2].axis('off')
            
            # Frequency weight
            freq_weight = outputs['freq_weight'].item()
            axes[1, 3].text(0.5, 0.5, f'Frequency Weight:\n{freq_weight:.3f}', 
                           ha='center', va='center', fontsize=16, transform=axes[1,3].transAxes)
            axes[1, 3].set_title('Network Analysis')
            axes[1, 3].axis('off')
            
            plt.tight_layout()
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"FourierNet visualization saved to {save_path}")
            break

if __name__ == "__main__":
    print("FourierNet - Revolutionary Frequency Domain Deblurring")
    print("Key Innovation: Global receptive field through Fourier analysis")
    print("Theoretical Foundation: Deblurring as division in frequency domain")
    print("Ready for breakthrough experiments!") 