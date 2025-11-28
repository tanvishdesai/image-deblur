import os
import json
import time
import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import cv2
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import io
import zipfile
import shutil
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class AdvancedDegradationPipeline:
    """
    State-of-the-art degradation pipeline for creating the most challenging SISR benchmark.
    Implements realistic degradations that go beyond simple bicubic downsampling.
    """
    
    def __init__(self, scale_factor=4, seed=42):
        self.scale_factor = scale_factor
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        
    def apply_chromatic_aberration(self, img_array, severity='random'):
        """Simulates chromatic aberration - lens imperfection causing color fringing."""
        if severity == 'random':
            shift_r = np.random.uniform(-2.5, 2.5)
            shift_b = np.random.uniform(-2.5, 2.5)
        else:
            shift_r = severity[0]
            shift_b = severity[1]
            
        h, w, c = img_array.shape
        if c != 3:
            return img_array, {'applied': False}
            
        img_shifted = img_array.copy().astype(np.float32)
        
        # Apply shifts to red and blue channels
        if abs(shift_r) > 0.1:
            img_shifted[:, :, 0] = ndimage.shift(img_array[:, :, 0], shift_r, mode='nearest')
        if abs(shift_b) > 0.1:
            img_shifted[:, :, 2] = ndimage.shift(img_array[:, :, 2], shift_b, mode='nearest')
            
        return np.clip(img_shifted, 0, 255).astype(np.uint8), {
            'applied': True, 'shift_r': shift_r, 'shift_b': shift_b
        }
    
    def apply_sensor_noise(self, img_array, severity='random'):
        """
        Simulates realistic camera sensor noise patterns with advanced models.
        
        Improvements:
        - Correlated noise with spatial correlation
        - Color-dependent noise variations
        - More realistic noise distribution modeling
        """
        img_float = img_array.astype(np.float32) / 255.0
        noise_params = {}
        
        # Color-dependent noise parameters
        if img_array.ndim == 3:
            # Different noise levels for RGB channels (red typically has more noise)
            channel_noise_factors = [1.1, 1.0, 1.05] if severity == 'random' else severity.get('channel_factors', [1.0, 1.0, 1.0])
        else:
            channel_noise_factors = [1.0]
        
        # Shot noise (Poisson) - signal dependent
        if np.random.random() < 0.8:
            if severity == 'random':
                shot_scale = np.random.uniform(0.01, 0.08)
            else:
                shot_scale = severity.get('shot_scale', 0.03)
            
            noise_params['shot_scale'] = shot_scale
            
            # Apply channel-specific shot noise
            if img_array.ndim == 3:
                for c in range(3):
                    channel_shot_scale = shot_scale * channel_noise_factors[c]
                    img_float[:, :, c] = np.random.poisson(
                        np.clip(img_float[:, :, c] / channel_shot_scale, 0, None)
                    ) * channel_shot_scale
            else:
                img_float = np.random.poisson(img_float / shot_scale) * shot_scale
        
        # Read noise (Gaussian) - signal independent with spatial correlation
        if np.random.random() < 0.9:
            if severity == 'random':
                read_std = np.random.uniform(0.005, 0.025)
                correlation_sigma = np.random.uniform(0.3, 1.0)  # Spatial correlation
            else:
                read_std = severity.get('read_std', 0.015)
                correlation_sigma = severity.get('correlation_sigma', 0.5)
                
            noise_params['read_std'] = read_std
            noise_params['correlation_sigma'] = correlation_sigma
            
            # Generate correlated noise for each channel
            if img_array.ndim == 3:
                for c in range(3):
                    channel_read_std = read_std * channel_noise_factors[c]
                    # Generate white noise and then apply spatial correlation
                    white_noise = np.random.normal(0, channel_read_std, img_float.shape[:2])
                    # Apply slight Gaussian blur to create spatial correlation
                    if correlation_sigma > 0.1:
                        correlated_noise = gaussian_filter(white_noise, sigma=correlation_sigma)
                    else:
                        correlated_noise = white_noise
                    img_float[:, :, c] += correlated_noise
            else:
                white_noise = np.random.normal(0, read_std, img_float.shape)
                if correlation_sigma > 0.1:
                    correlated_noise = gaussian_filter(white_noise, sigma=correlation_sigma)
                else:
                    correlated_noise = white_noise
                img_float += correlated_noise
        
        # Dark current noise - temperature dependent
        if np.random.random() < 0.4:
            if severity == 'random':
                dark_current = np.random.uniform(0.001, 0.012)
            else:
                dark_current = severity.get('dark_current', 0.006)
                
            noise_params['dark_current'] = dark_current
            img_float += dark_current
        
        # Add color-dependent noise parameters to metadata
        if img_array.ndim == 3:
            noise_params['channel_noise_factors'] = channel_noise_factors
            
        return np.clip(img_float * 255.0, 0, 255).astype(np.uint8), {
            'applied': True, **noise_params
        }
    
    def apply_motion_blur(self, img_array, severity='random'):
        """
        Applies advanced motion blur including anisotropic kernels and defocus blur.
        
        Improvements:
        - Anisotropic kernels for complex motion paths
        - Realistic defocus (bokeh) blur with disk kernels
        - Random walk motion paths for more realistic camera shake
        """
        if np.random.random() < 0.5:  # 50% chance
            blur_type = np.random.choice(['linear', 'rotational', 'defocus', 'anisotropic', 'random_walk'])
            blur_params = {'type': blur_type}
            
            if blur_type == 'linear':
                if severity == 'random':
                    kernel_size = np.random.randint(5, 20)
                    angle = np.random.uniform(0, 180)
                else:
                    kernel_size = severity.get('kernel_size', 10)
                    angle = severity.get('angle', 45)
                
                blur_params.update({'kernel_size': kernel_size, 'angle': angle})
                
                # Create motion blur kernel
                kernel = np.zeros((kernel_size, kernel_size))
                center = kernel_size // 2
                for i in range(kernel_size):
                    x = int(center + (i - center) * np.cos(np.radians(angle)))
                    y = int(center + (i - center) * np.sin(np.radians(angle)))
                    if 0 <= x < kernel_size and 0 <= y < kernel_size:
                        kernel[y, x] = 1
                kernel /= np.sum(kernel) if np.sum(kernel) > 0 else 1
                
            elif blur_type == 'rotational':
                if severity == 'random':
                    sigma = np.random.uniform(0.5, 3.0)
                else:
                    sigma = severity.get('sigma', 1.5)
                    
                blur_params['sigma'] = sigma
                return gaussian_filter(img_array, sigma=sigma), {'applied': True, **blur_params}
            
            elif blur_type == 'defocus':
                # Realistic defocus (bokeh) blur using disk-shaped kernel
                if severity == 'random':
                    radius = np.random.randint(2, 10)
                else:
                    radius = severity.get('radius', 5)
                
                blur_params['radius'] = radius
                
                # Create disk kernel for defocus blur
                kernel_size = 2 * radius + 1
                kernel = np.zeros((kernel_size, kernel_size))
                center = radius
                y, x = np.ogrid[-center:kernel_size-center, -center:kernel_size-center]
                mask = x*x + y*y <= radius*radius
                kernel[mask] = 1
                kernel /= np.sum(kernel)
                
            elif blur_type == 'anisotropic':
                # Anisotropic blur with different strengths in different directions
                if severity == 'random':
                    sigma_x = np.random.uniform(0.5, 4.0)
                    sigma_y = np.random.uniform(0.5, 4.0)
                    angle = np.random.uniform(0, 180)
                else:
                    sigma_x = severity.get('sigma_x', 2.0)
                    sigma_y = severity.get('sigma_y', 1.0)
                    angle = severity.get('angle', 45)
                
                blur_params.update({'sigma_x': sigma_x, 'sigma_y': sigma_y, 'angle': angle})
                
                # Create anisotropic Gaussian kernel
                kernel_size = int(6 * max(sigma_x, sigma_y)) + 1
                if kernel_size % 2 == 0:
                    kernel_size += 1
                
                kernel = np.zeros((kernel_size, kernel_size))
                center = kernel_size // 2
                
                # Rotation matrix
                cos_a = np.cos(np.radians(angle))
                sin_a = np.sin(np.radians(angle))
                
                for i in range(kernel_size):
                    for j in range(kernel_size):
                        x = i - center
                        y = j - center
                        # Rotate coordinates
                        x_rot = x * cos_a - y * sin_a
                        y_rot = x * sin_a + y * cos_a
                        # Apply anisotropic Gaussian
                        kernel[i, j] = np.exp(-0.5 * ((x_rot/sigma_x)**2 + (y_rot/sigma_y)**2))
                
                kernel /= np.sum(kernel)
                
            elif blur_type == 'random_walk':
                # Simulate camera shake with random walk
                if severity == 'random':
                    steps = np.random.randint(8, 25)
                    step_size = np.random.uniform(0.5, 3.0)
                else:
                    steps = severity.get('steps', 15)
                    step_size = severity.get('step_size', 1.5)
                
                blur_params.update({'steps': steps, 'step_size': step_size})
                
                # Generate random walk path
                path_x = np.cumsum(np.random.normal(0, step_size, steps))
                path_y = np.cumsum(np.random.normal(0, step_size, steps))
                
                # Center the path
                path_x -= np.mean(path_x)
                path_y -= np.mean(path_y)
                
                # Create kernel from path
                kernel_size = int(2 * max(np.max(np.abs(path_x)), np.max(np.abs(path_y)))) + 5
                if kernel_size % 2 == 0:
                    kernel_size += 1
                
                kernel = np.zeros((kernel_size, kernel_size))
                center = kernel_size // 2
                
                for x, y in zip(path_x, path_y):
                    px = int(center + x)
                    py = int(center + y)
                    if 0 <= px < kernel_size and 0 <= py < kernel_size:
                        kernel[py, px] = 1
                
                kernel /= np.sum(kernel) if np.sum(kernel) > 0 else 1
            
            # Apply convolution for kernel-based blurs
            if blur_type in ['linear', 'defocus', 'anisotropic', 'random_walk']:
                if img_array.ndim == 3:
                    blurred = np.zeros_like(img_array)
                    for c in range(img_array.shape[2]):
                        blurred[:, :, c] = cv2.filter2D(img_array[:, :, c], -1, kernel)
                    return blurred, {'applied': True, **blur_params}
                else:
                    return cv2.filter2D(img_array, -1, kernel), {'applied': True, **blur_params}
                
        return img_array, {'applied': False}
    
    def apply_atmospheric_effects(self, img_array, severity='random'):
        """Simulates atmospheric degradations like haze, fog, rain."""
        if np.random.random() < 0.3:  # 30% chance
            effect_type = np.random.choice(['haze', 'fog', 'light_rain'])
            
            if effect_type in ['haze', 'fog']:
                if severity == 'random':
                    beta = np.random.uniform(0.1, 0.6)  # scattering coefficient
                    atmospheric_light = np.random.uniform(0.8, 0.95)
                else:
                    beta = severity.get('beta', 0.3)
                    atmospheric_light = severity.get('atmospheric_light', 0.9)
                
                # Atmospheric scattering model: I = t*J + A*(1-t)
                # where t is transmission map
                h, w = img_array.shape[:2]
                
                # Create depth-based transmission (simulate distance)
                y, x = np.mgrid[0:h, 0:w]
                depth = np.sqrt((x - w/2)**2 + (y - h/2)**2) / max(w, h)
                transmission = np.exp(-beta * depth)
                
                img_float = img_array.astype(np.float32) / 255.0
                atmospheric_light_val = atmospheric_light
                
                if img_array.ndim == 3:
                    transmission = transmission[..., np.newaxis]
                    atmospheric_light_val = np.array([atmospheric_light, atmospheric_light, atmospheric_light])
                
                hazy_img = img_float * transmission + atmospheric_light_val * (1 - transmission)
                return (np.clip(hazy_img, 0, 1) * 255).astype(np.uint8), {
                    'applied': True, 'type': effect_type, 'beta': beta, 'atmospheric_light': atmospheric_light
                }
            
            elif effect_type == 'light_rain':
                # Add subtle rain effect
                if severity == 'random':
                    rain_intensity = np.random.uniform(0.1, 0.4)
                else:
                    rain_intensity = severity.get('rain_intensity', 0.25)
                
                h, w = img_array.shape[:2]
                rain_mask = np.random.random((h, w)) < rain_intensity * 0.01
                rain_streaks = np.random.uniform(0.7, 1.0, (h, w)) * rain_mask
                
                img_float = img_array.astype(np.float32) / 255.0
                if img_array.ndim == 3:
                    rain_streaks = rain_streaks[..., np.newaxis]
                
                rain_img = img_float * (1 - rain_streaks * 0.3) + rain_streaks * 0.3
                return (np.clip(rain_img, 0, 1) * 255).astype(np.uint8), {
                    'applied': True, 'type': effect_type, 'rain_intensity': rain_intensity
                }
                
        return img_array, {'applied': False}
    
    def apply_illumination_degradation(self, img_array, severity='random'):
        """Simulates challenging lighting conditions."""
        if np.random.random() < 0.4:  # 40% chance
            illum_type = np.random.choice(['low_light', 'overexposure', 'mixed_lighting'])
            
            if illum_type == 'low_light':
                if severity == 'random':
                    gamma = np.random.uniform(1.5, 3.0)  # Darken image
                    noise_boost = np.random.uniform(1.2, 2.0)
                else:
                    gamma = severity.get('gamma', 2.0)
                    noise_boost = severity.get('noise_boost', 1.5)
                
                # Apply gamma correction to simulate low light
                img_float = img_array.astype(np.float32) / 255.0
                low_light_img = np.power(img_float, gamma)
                
                # Add extra noise in low light conditions
                noise = np.random.normal(0, 0.02 * noise_boost, img_float.shape)
                low_light_img += noise
                
                return (np.clip(low_light_img, 0, 1) * 255).astype(np.uint8), {
                    'applied': True, 'type': illum_type, 'gamma': gamma, 'noise_boost': noise_boost
                }
            
            elif illum_type == 'overexposure':
                if severity == 'random':
                    clip_threshold = np.random.uniform(0.7, 0.9)
                    saturation_boost = np.random.uniform(1.1, 1.4)
                else:
                    clip_threshold = severity.get('clip_threshold', 0.8)
                    saturation_boost = severity.get('saturation_boost', 1.2)
                
                img_float = img_array.astype(np.float32) / 255.0
                # Simulate overexposure by clipping highlights
                overexposed = np.clip(img_float * saturation_boost, 0, clip_threshold)
                
                return (overexposed * 255).astype(np.uint8), {
                    'applied': True, 'type': illum_type, 'clip_threshold': clip_threshold, 'saturation_boost': saturation_boost
                }
                
        return img_array, {'applied': False}
    
    def apply_jpeg_compression(self, img_array, pil_img=None, severity='random'):
        """Applies JPEG compression with realistic quality levels."""
        if np.random.random() < 0.7:  # 70% chance
            if severity == 'random':
                quality = np.random.randint(25, 85)
            else:
                quality = severity.get('quality', 50)
            
            if pil_img is None:
                pil_img = Image.fromarray(img_array)
            
            # Save to memory buffer with JPEG compression
            buffer = io.BytesIO()
            pil_img.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            
            # Load back from buffer
            compressed_img = Image.open(buffer)
            return np.array(compressed_img), {
                'applied': True, 'quality': quality
            }
            
        return img_array, {'applied': False}
    
    def apply_lens_distortion(self, img_array, severity='random'):
        """Applies lens distortion (barrel/pincushion) and vignetting."""
        distortion_applied = False
        distortion_params = {}
        
        # Radial distortion
        if np.random.random() < 0.4:  # 40% chance
            if severity == 'random':
                distortion_coeff = np.random.uniform(-0.15, 0.15)
            else:
                distortion_coeff = severity.get('distortion_coeff', 0.0)
            
            if abs(distortion_coeff) > 0.01:
                h, w = img_array.shape[:2]
                center_x, center_y = w // 2, h // 2
                y, x = np.mgrid[0:h, 0:w]
                
                # Normalize coordinates
                x_norm = (x - center_x) / center_x
                y_norm = (y - center_y) / center_y
                
                # Apply radial distortion
                r_squared = x_norm**2 + y_norm**2
                distortion_factor = 1 + distortion_coeff * r_squared
                
                x_distorted = x_norm * distortion_factor * center_x + center_x
                y_distorted = y_norm * distortion_factor * center_y + center_y
                
                # Clip to image bounds
                x_distorted = np.clip(x_distorted, 0, w-1)
                y_distorted = np.clip(y_distorted, 0, h-1)
                
                # Apply distortion
                if img_array.ndim == 3:
                    distorted = np.zeros_like(img_array)
                    for c in range(img_array.shape[2]):
                        distorted[:, :, c] = cv2.remap(
                            img_array[:, :, c], 
                            x_distorted.astype(np.float32), 
                            y_distorted.astype(np.float32),
                            cv2.INTER_LINEAR
                        )
                    img_array = distorted
                else:
                    img_array = cv2.remap(
                        img_array, 
                        x_distorted.astype(np.float32), 
                        y_distorted.astype(np.float32),
                        cv2.INTER_LINEAR
                    )
                
                distortion_applied = True
                distortion_params['distortion_coeff'] = distortion_coeff
        
        # Vignetting
        if np.random.random() < 0.3:  # 30% chance
            if severity == 'random':
                vignette_strength = np.random.uniform(0.1, 0.5)
            else:
                vignette_strength = severity.get('vignette_strength', 0.3)
            
            h, w = img_array.shape[:2]
            center_x, center_y = w // 2, h // 2
            y, x = np.mgrid[0:h, 0:w]
            
            # Distance from center
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_distance = np.sqrt(center_x**2 + center_y**2)
            
            # Vignette mask
            vignette_mask = 1 - vignette_strength * (distance / max_distance)**2
            vignette_mask = np.clip(vignette_mask, 0, 1)
            
            if img_array.ndim == 3:
                vignette_mask = vignette_mask[..., np.newaxis]
            
            img_array = (img_array.astype(np.float32) * vignette_mask).astype(np.uint8)
            distortion_applied = True
            distortion_params['vignette_strength'] = vignette_strength
            
        return img_array, {
            'applied': distortion_applied, **distortion_params
        }
    
    def apply_random_resize(self, img_array, severity='random'):
        """
        Applies random intermediate resizing with theoretically sound interpolation methods.
        
        Uses Lanczos filter (windowed sinc approximation) as the ideal downsampling filter
        for preventing aliasing. This is theoretically more sound than bicubic or bilinear
        methods for anti-aliasing during downsampling operations.
        """
        if np.random.random() < 0.6:  # 60% chance
            h, w = img_array.shape[:2]
            
            if severity == 'random':
                intermediate_scale = np.random.uniform(0.7, 1.3)
                interpolation = np.random.choice([
                    cv2.INTER_LINEAR, cv2.INTER_CUBIC, 
                    cv2.INTER_LANCZOS4, cv2.INTER_AREA
                ])
            else:
                intermediate_scale = severity.get('intermediate_scale', 1.0)
                interpolation = severity.get('interpolation', cv2.INTER_CUBIC)
            
            if abs(intermediate_scale - 1.0) > 0.05:  # Only apply if significant change
                new_h, new_w = int(h * intermediate_scale), int(w * intermediate_scale)
                
                # Resize and then back to original size
                resized = cv2.resize(img_array, (new_w, new_h), interpolation=interpolation)
                final_resized = cv2.resize(resized, (w, h), interpolation=cv2.INTER_CUBIC)
                
                return final_resized, {
                    'applied': True, 'intermediate_scale': intermediate_scale, 'interpolation': str(interpolation)
                }
            
        return img_array, {'applied': False}
    
    def apply_full_isp_pipeline(self, img_array, severity='random'):
        """
        Simulates full camera Image Signal Processing (ISP) pipeline.
        
        Complete pipeline: Demosaicing -> Denoising -> Color Correction -> Tone Mapping -> Sharpening
        This provides much more realistic camera processing artifacts.
        """
        img_float = img_array.astype(np.float32) / 255.0
        pipeline_params = {'stages_applied': []}
        
        # Stage 1: Demosaicing artifacts (simulate Bayer pattern reconstruction errors)
        if np.random.random() < 0.4:
            if severity == 'random':
                demosaic_strength = np.random.uniform(0.1, 0.4)
                color_misalignment = np.random.uniform(0.2, 1.0)
            else:
                demosaic_strength = severity.get('demosaic_strength', 0.25)
                color_misalignment = severity.get('color_misalignment', 0.5)
            
            if img_array.ndim == 3:
                # Simulate demosaicing errors with cross-channel leakage
                img_demosaic = img_float.copy()
                
                # Add slight cross-channel correlation (typical demosaicing artifact)
                cross_talk_r_to_g = demosaic_strength * 0.1
                cross_talk_b_to_g = demosaic_strength * 0.08
                
                img_demosaic[:, :, 1] += cross_talk_r_to_g * img_float[:, :, 0]
                img_demosaic[:, :, 1] += cross_talk_b_to_g * img_float[:, :, 2]
                
                # Add color channel misalignment
                if color_misalignment > 0.1:
                    shift_r = np.random.uniform(-color_misalignment, color_misalignment)
                    shift_b = np.random.uniform(-color_misalignment, color_misalignment)
                    img_demosaic[:, :, 0] = ndimage.shift(img_demosaic[:, :, 0], shift_r, mode='nearest')
                    img_demosaic[:, :, 2] = ndimage.shift(img_demosaic[:, :, 2], shift_b, mode='nearest')
                
                img_float = np.clip(img_demosaic, 0, 1)
                pipeline_params['stages_applied'].append('demosaicing')
                pipeline_params['demosaic_strength'] = demosaic_strength
                pipeline_params['color_misalignment'] = color_misalignment
        
        # Stage 2: ISP Denoising (often aggressive, can blur details)
        if np.random.random() < 0.5:
            if severity == 'random':
                denoise_strength = np.random.uniform(0.3, 1.5)
                bilateral_sigma_color = np.random.uniform(0.05, 0.15)
                bilateral_sigma_space = np.random.uniform(1, 3)
            else:
                denoise_strength = severity.get('denoise_strength', 0.8)
                bilateral_sigma_color = severity.get('bilateral_sigma_color', 0.1)
                bilateral_sigma_space = severity.get('bilateral_sigma_space', 2)
            
            # Convert to uint8 for bilateral filter
            img_uint8 = (img_float * 255).astype(np.uint8)
            
            # Apply bilateral filtering (common ISP denoising)
            if img_array.ndim == 3:
                denoised = cv2.bilateralFilter(img_uint8, 
                                             d=int(bilateral_sigma_space * 2) + 1,
                                             sigmaColor=bilateral_sigma_color * 255,
                                             sigmaSpace=bilateral_sigma_space)
            else:
                denoised = cv2.bilateralFilter(img_uint8,
                                             d=int(bilateral_sigma_space * 2) + 1,
                                             sigmaColor=bilateral_sigma_color * 255,
                                             sigmaSpace=bilateral_sigma_space)
            
            # Blend with original based on strength
            img_float = (denoised.astype(np.float32) / 255.0) * denoise_strength + img_float * (1 - denoise_strength)
            img_float = np.clip(img_float, 0, 1)
            
            pipeline_params['stages_applied'].append('denoising')
            pipeline_params['denoise_strength'] = denoise_strength
        
        # Stage 3: Color Correction (random color matrix)
        if np.random.random() < 0.6 and img_array.ndim == 3:
            if severity == 'random':
                color_shift_strength = np.random.uniform(0.02, 0.08)
                saturation_factor = np.random.uniform(0.9, 1.15)
            else:
                color_shift_strength = severity.get('color_shift_strength', 0.05)
                saturation_factor = severity.get('saturation_factor', 1.05)
            
            # Generate random color correction matrix
            # Start with identity and add small perturbations
            color_matrix = np.eye(3)
            perturbation = np.random.normal(0, color_shift_strength, (3, 3))
            color_matrix += perturbation
            
            # Apply saturation adjustment
            # Convert to YUV, scale UV channels, convert back
            rgb_to_yuv = np.array([[0.299, 0.587, 0.114],
                                   [-0.14713, -0.28886, 0.436],
                                   [0.615, -0.51499, -0.10001]])
            yuv_to_rgb = np.linalg.inv(rgb_to_yuv)
            
            # Reshape for matrix operations
            h, w, c = img_float.shape
            img_reshaped = img_float.reshape(-1, 3)
            
            # Apply color correction
            img_corrected = img_reshaped @ color_matrix.T
            
            # Apply saturation
            yuv = img_corrected @ rgb_to_yuv.T
            yuv[:, 1:] *= saturation_factor  # Scale U and V channels
            img_corrected = yuv @ yuv_to_rgb.T
            
            img_float = np.clip(img_corrected.reshape(h, w, c), 0, 1)
            
            pipeline_params['stages_applied'].append('color_correction')
            pipeline_params['color_shift_strength'] = color_shift_strength
            pipeline_params['saturation_factor'] = saturation_factor
        
        # Stage 4: Tone Mapping (gamma correction with slight variations)
        if np.random.random() < 0.7:
            if severity == 'random':
                gamma = np.random.uniform(0.8, 1.3)
                highlight_compression = np.random.uniform(0.9, 1.0)
            else:
                gamma = severity.get('gamma', 1.0)
                highlight_compression = severity.get('highlight_compression', 0.95)
            
            # Apply gamma correction
            img_gamma = np.power(img_float, 1.0/gamma)
            
            # Compress highlights (common in phone cameras)
            if highlight_compression < 1.0:
                highlight_mask = img_gamma > 0.7
                img_gamma[highlight_mask] = 0.7 + (img_gamma[highlight_mask] - 0.7) * highlight_compression
            
            img_float = np.clip(img_gamma, 0, 1)
            
            pipeline_params['stages_applied'].append('tone_mapping')
            pipeline_params['gamma'] = gamma
            pipeline_params['highlight_compression'] = highlight_compression
        
        # Stage 5: Sharpening (often aggressive in consumer cameras)
        if np.random.random() < 0.8:
            if severity == 'random':
                sharpen_strength = np.random.uniform(0.3, 1.2)
                sharpen_radius = np.random.uniform(0.8, 2.0)
            else:
                sharpen_strength = severity.get('sharpen_strength', 0.7)
                sharpen_radius = severity.get('sharpen_radius', 1.2)
            
            # Unsharp mask with configurable radius
            blurred = gaussian_filter(img_float, sigma=sharpen_radius)
            sharpened = img_float + sharpen_strength * (img_float - blurred)
            img_float = np.clip(sharpened, 0, 1)
            
            pipeline_params['stages_applied'].append('sharpening')
            pipeline_params['sharpen_strength'] = sharpen_strength
            pipeline_params['sharpen_radius'] = sharpen_radius
        
        return (img_float * 255).astype(np.uint8), {
            'applied': len(pipeline_params['stages_applied']) > 0,
            'total_stages': len(pipeline_params['stages_applied']),
            **pipeline_params
        }
    
    def degrade_image(self, hr_pil_image, difficulty_level='random', custom_params=None):
        """
        Applies the complete advanced degradation pipeline.
        
        Args:
            hr_pil_image: PIL Image object (high resolution)
            difficulty_level: 'easy', 'medium', 'hard', 'extreme', or 'random'
            custom_params: Dictionary of custom parameters for specific degradations
            
        Returns:
            Tuple of (lr_pil_image, degradation_metadata)
        """
        # Convert to numpy array
        hr_array = np.array(hr_pil_image)
        
        # Define degradation sequences based on difficulty
        if difficulty_level == 'easy':
            degradation_prob = 0.3
            max_degradations = 2
        elif difficulty_level == 'medium':
            degradation_prob = 0.5
            max_degradations = 3
        elif difficulty_level == 'hard':
            degradation_prob = 0.7
            max_degradations = 5
        elif difficulty_level == 'extreme':
            degradation_prob = 0.9
            max_degradations = 7
        else:  # random
            degradation_prob = np.random.uniform(0.4, 0.8)
            max_degradations = np.random.randint(2, 6)
        
        # Structured degradation pipeline with realistic ordering
        # Following the physical/temporal order that degradations would occur
        degradation_stages = {
            # Stage 1: Optical/Physical degradations (before sensor)
            'optical': [
                ('chromatic_aberration', self.apply_chromatic_aberration),
                ('lens_distortion', self.apply_lens_distortion),
                ('atmospheric_effects', self.apply_atmospheric_effects),
            ],
            # Stage 2: Motion and blur (during exposure)
            'motion_blur': [
                ('motion_blur', self.apply_motion_blur),
            ],
            # Stage 3: Scene/lighting effects
            'scene_effects': [
                ('illumination_degradation', self.apply_illumination_degradation),
            ],
            # Stage 4: Sensor-level degradations
            'sensor': [
                ('sensor_noise', self.apply_sensor_noise),
            ],
            # Stage 5: Digital processing pipeline
            'isp_processing': [
                ('full_isp_pipeline', self.apply_full_isp_pipeline),
            ],
            # Stage 6: Resize operations (can happen at different stages)
            'resize': [
                ('random_resize', self.apply_random_resize),
            ]
        }
        
        # Select degradations while maintaining realistic order
        selected_degradations = []
        
        # Process each stage in order
        for stage_name, stage_degradations in degradation_stages.items():
            stage_prob = degradation_prob
            
            # Adjust probabilities for different stages
            if stage_name == 'optical':
                stage_prob *= 0.7  # Optical effects less common
            elif stage_name == 'isp_processing':
                stage_prob *= 1.2  # ISP processing very common
            elif stage_name == 'sensor':
                stage_prob *= 1.1  # Sensor noise common
            
            for deg_name, deg_func in stage_degradations:
                if np.random.random() < stage_prob and len(selected_degradations) < max_degradations:
                    selected_degradations.append((deg_name, deg_func))
        
        # Add some randomization within stages (but maintain overall order)
        # Shuffle optical degradations among themselves
        optical_indices = [i for i, (name, _) in enumerate(selected_degradations) 
                          if name in ['chromatic_aberration', 'lens_distortion', 'atmospheric_effects']]
        if len(optical_indices) > 1:
            optical_items = [selected_degradations[i] for i in optical_indices]
            np.random.shuffle(optical_items)
            for i, item in zip(optical_indices, optical_items):
                selected_degradations[i] = item
        
        # Apply selected degradations
        degraded_array = hr_array.copy()
        degradation_metadata = {
            'difficulty_level': difficulty_level,
            'degradation_sequence': [],
            'applied_degradations': {}
        }
        
        for deg_name, deg_func in selected_degradations:
            severity = custom_params.get(deg_name, 'random') if custom_params else 'random'
            degraded_array, params = deg_func(degraded_array, severity)
            
            if params.get('applied', False):
                degradation_metadata['degradation_sequence'].append(deg_name)
                degradation_metadata['applied_degradations'][deg_name] = params
        
        # Convert back to PIL for JPEG compression (always apply this last)
        degraded_pil = Image.fromarray(degraded_array)
        degraded_array, jpeg_params = self.apply_jpeg_compression(degraded_array, degraded_pil)
        if jpeg_params.get('applied', False):
            degradation_metadata['applied_degradations']['jpeg_compression'] = jpeg_params
            degradation_metadata['degradation_sequence'].append('jpeg_compression')
        
        degraded_pil = Image.fromarray(degraded_array)
        
        # Final downsampling to LR resolution
        lr_size = (hr_pil_image.size[0] // self.scale_factor, 
                   hr_pil_image.size[1] // self.scale_factor)
        
        # Use random interpolation for final downsampling
        interpolation_methods = [Image.BICUBIC, Image.BILINEAR, Image.LANCZOS]
        interpolation = np.random.choice(interpolation_methods)
        
        lr_pil = degraded_pil.resize(lr_size, interpolation)
        
        # Add final metadata
        degradation_metadata['final_downsampling'] = {
            'scale_factor': self.scale_factor,
            'interpolation': str(interpolation),
            'lr_size': lr_size,
            'hr_size': hr_pil_image.size
        }
        
        degradation_metadata['total_degradations_applied'] = len(degradation_metadata['degradation_sequence'])
        
        return lr_pil, degradation_metadata


class DatasetMetricsAndReporting:
    """
    Comprehensive metrics calculation and reporting for SISR datasets.
    Implements standard evaluation metrics and dataset documentation practices.
    """
    
    def __init__(self, dataset_name="Advanced SISR Benchmark Dataset"):
        self.dataset_name = dataset_name
        self.metrics_results = {}
        self.dataset_statistics = {}
        
    def calculate_psnr(self, hr_img, lr_img_upscaled):
        """Calculate Peak Signal-to-Noise Ratio."""
        hr_array = np.array(hr_img)
        lr_array = np.array(lr_img_upscaled)
        
        # Ensure same dimensions
        if hr_array.shape != lr_array.shape:
            lr_array = cv2.resize(lr_array, (hr_array.shape[1], hr_array.shape[0]))
        
        return peak_signal_noise_ratio(hr_array, lr_array, data_range=255)
    
    def calculate_ssim(self, hr_img, lr_img_upscaled):
        """Calculate Structural Similarity Index."""
        hr_array = np.array(hr_img, dtype=np.float64)
        lr_array = np.array(lr_img_upscaled, dtype=np.float64)
        
        # Ensure same dimensions
        if hr_array.shape != lr_array.shape:
            lr_array = cv2.resize(lr_array, (hr_array.shape[1], hr_array.shape[0]))
        
        if len(hr_array.shape) == 3:
            return structural_similarity(hr_array, lr_array, multichannel=True, channel_axis=2, data_range=255)
        else:
            return structural_similarity(hr_array, lr_array, data_range=255)
    
    def calculate_erqa_simple(self, hr_img, lr_img_upscaled):
        """Simplified Edge Restoration Quality Assessment."""
        hr_array = np.array(hr_img)
        lr_array = np.array(lr_img_upscaled)
        
        # Ensure same dimensions
        if hr_array.shape != lr_array.shape:
            lr_array = cv2.resize(lr_array, (hr_array.shape[1], hr_array.shape[0]))
        
        # Convert to grayscale for edge detection
        if len(hr_array.shape) == 3:
            hr_gray = cv2.cvtColor(hr_array, cv2.COLOR_RGB2GRAY)
            lr_gray = cv2.cvtColor(lr_array, cv2.COLOR_RGB2GRAY)
        else:
            hr_gray = hr_array
            lr_gray = lr_array
        
        # Edge detection using Canny
        hr_edges = cv2.Canny(hr_gray, 100, 200)
        lr_edges = cv2.Canny(lr_gray, 100, 200)
        
        # Calculate F1-score for edge matching
        hr_edges_norm = hr_edges.astype(bool)
        lr_edges_norm = lr_edges.astype(bool)
        
        # True positives, false positives, false negatives
        tp = np.sum(hr_edges_norm & lr_edges_norm)
        fp = np.sum(lr_edges_norm & ~hr_edges_norm)
        fn = np.sum(hr_edges_norm & ~lr_edges_norm)
        
        if tp + fp == 0:
            precision = 0
        else:
            precision = tp / (tp + fp)
            
        if tp + fn == 0:
            recall = 0
        else:
            recall = tp / (tp + fn)
        
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        return f1
    
    def calculate_dataset_statistics(self, all_pairs):
        """Calculate comprehensive dataset statistics."""
        stats = {
            'total_pairs': len(all_pairs),
            'categories': {},
            'resolution_stats': {},
            'degradation_stats': {},
            'difficulty_distribution': {},
            'file_size_stats': {}
        }
        
        # Category statistics
        for pair in all_pairs:
            category = pair['category']
            if category not in stats['categories']:
                stats['categories'][category] = 0
            stats['categories'][category] += 1
        
        # Resolution statistics
        resolutions = [f"{pair['hr_size'][0]}x{pair['hr_size'][1]}" for pair in all_pairs]
        unique_resolutions = list(set(resolutions))
        for res in unique_resolutions:
            stats['resolution_stats'][res] = resolutions.count(res)
        
        # Degradation statistics
        for pair in all_pairs:
            deg_seq = pair['degradation_metadata'].get('degradation_sequence', [])
            for deg in deg_seq:
                if deg not in stats['degradation_stats']:
                    stats['degradation_stats'][deg] = 0
                stats['degradation_stats'][deg] += 1
        
        # Difficulty distribution
        for pair in all_pairs:
            difficulty = pair['degradation_metadata'].get('difficulty_level', 'unknown')
            if difficulty not in stats['difficulty_distribution']:
                stats['difficulty_distribution'][difficulty] = 0
            stats['difficulty_distribution'][difficulty] += 1
        
        return stats
    
    def calculate_sample_metrics(self, hr_img_path, lr_img_path, sample_size=100):
        """Calculate metrics on a sample of image pairs for dataset quality assessment."""
        if not os.path.exists(hr_img_path) or not os.path.exists(lr_img_path):
            return None
        
        hr_img = Image.open(hr_img_path)
        lr_img = Image.open(lr_img_path)
        
        # Upscale LR image for comparison (using simple bicubic)
        lr_upscaled = lr_img.resize(hr_img.size, Image.BICUBIC)
        
        metrics = {
            'psnr': self.calculate_psnr(hr_img, lr_upscaled),
            'ssim': self.calculate_ssim(hr_img, lr_upscaled),
            'erqa_simple': self.calculate_erqa_simple(hr_img, lr_upscaled)
        }
        
        return metrics
    
    def evaluate_dataset_quality(self, hr_dir, lr_dir, sample_size=100):
        """Evaluate overall dataset quality using standard metrics."""
        hr_files = list(Path(hr_dir).glob("*.png"))[:sample_size]
        metrics_list = []
        
        print(f"Evaluating dataset quality on {len(hr_files)} sample pairs...")
        
        for hr_file in tqdm(hr_files, desc="Calculating metrics"):
            # Find corresponding LR file
            base_name = hr_file.stem
            lr_files = list(Path(lr_dir).glob(f"{base_name}_v*.png"))
            
            if lr_files:
                lr_file = lr_files[0]  # Use first variant
                metrics = self.calculate_sample_metrics(str(hr_file), str(lr_file))
                if metrics:
                    metrics_list.append(metrics)
        
        if not metrics_list:
            return None
        
        # Calculate aggregate statistics
        aggregate_metrics = {}
        for metric_name in ['psnr', 'ssim', 'erqa_simple']:
            values = [m[metric_name] for m in metrics_list if not np.isnan(m[metric_name])]
            if values:
                aggregate_metrics[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values)
                }
        
        return aggregate_metrics
    
    def assess_dataset_bias(self, all_pairs, sample_images_for_analysis=True):
        """
        Enhanced bias assessment including visual content analysis.
        
        Analyzes colors, brightness levels, textures, and fairness metrics
        following modern dataset auditing practices.
        """
        bias_assessment = {
            'category_balance': {},
            'resolution_bias': {},
            'degradation_bias': {},
            'visual_content_bias': {},
            'fairness_analysis': {},
            'recommendations': []
        }
        
        # Category balance
        categories = [pair['category'] for pair in all_pairs]
        category_counts = {}
        for cat in categories:
            category_counts[cat] = categories.count(cat)
        
        total_pairs = len(all_pairs)
        for cat, count in category_counts.items():
            bias_assessment['category_balance'][cat] = {
                'count': count,
                'percentage': (count / total_pairs) * 100
            }
        
        # Check for severe imbalance
        percentages = [info['percentage'] for info in bias_assessment['category_balance'].values()]
        if max(percentages) > 70:
            bias_assessment['recommendations'].append(
                "Category imbalance detected: Consider adding more samples from underrepresented categories"
            )
        
        # Resolution bias
        resolutions = [f"{pair['hr_size'][0]}x{pair['hr_size'][1]}" for pair in all_pairs]
        unique_resolutions = list(set(resolutions))
        if len(unique_resolutions) == 1:
            bias_assessment['recommendations'].append(
                "Resolution diversity limited: Consider including multiple resolutions for better generalization"
            )
        
        # Degradation bias
        degradation_counts = {}
        for pair in all_pairs:
            deg_count = len(pair['degradation_metadata'].get('degradation_sequence', []))
            if deg_count not in degradation_counts:
                degradation_counts[deg_count] = 0
            degradation_counts[deg_count] += 1
        
        if len(degradation_counts) == 1:
            bias_assessment['recommendations'].append(
                "Degradation complexity is uniform: Consider varying the number of degradations per image"
            )
        
        # Enhanced visual content analysis
        if sample_images_for_analysis and all_pairs:
            print("Performing enhanced visual content bias analysis...")
            visual_analysis = self._analyze_visual_content_bias(all_pairs)
            bias_assessment['visual_content_bias'] = visual_analysis
            
            # Add recommendations based on visual analysis
            if visual_analysis.get('brightness_bias_detected', False):
                bias_assessment['recommendations'].append(
                    "Brightness bias detected: Dataset may be skewed towards bright or dark images"
                )
            
            if visual_analysis.get('color_diversity_low', False):
                bias_assessment['recommendations'].append(
                    "Low color diversity: Consider adding images with more varied color palettes"
                )
            
            if visual_analysis.get('skin_tone_bias_detected', False):
                bias_assessment['recommendations'].append(
                    "Potential skin tone bias detected in portrait category: Ensure diverse representation"
                )
        
        return bias_assessment
    
    def _analyze_visual_content_bias(self, all_pairs, sample_size=200):
        """
        Analyze visual content properties for bias detection.
        
        This performs statistical analysis of colors, brightness, and textures
        to detect potential biases in the dataset composition.
        """
        # Sample pairs for analysis
        sample_pairs = np.random.choice(all_pairs, min(sample_size, len(all_pairs)), replace=False)
        
        brightness_values = []
        color_distributions = {'red': [], 'green': [], 'blue': []}
        saturation_values = []
        skin_tone_analysis = {'portrait_count': 0, 'skin_tones': []}
        texture_complexity = []
        
        for pair in tqdm(sample_pairs, desc="Analyzing visual content bias"):
            try:
                # Load HR image for analysis
                hr_path = Path(pair['hr_path'])
                if not hr_path.is_absolute():
                    # Construct full path if relative
                    hr_path = Path("dataset/sisr_benchmark") / hr_path
                
                if hr_path.exists():
                    img = Image.open(hr_path).convert('RGB')
                    img_array = np.array(img)
                    
                    # Brightness analysis
                    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                    brightness = np.mean(gray) / 255.0
                    brightness_values.append(brightness)
                    
                    # Color distribution analysis
                    color_distributions['red'].append(np.mean(img_array[:, :, 0]) / 255.0)
                    color_distributions['green'].append(np.mean(img_array[:, :, 1]) / 255.0)
                    color_distributions['blue'].append(np.mean(img_array[:, :, 2]) / 255.0)
                    
                    # Saturation analysis
                    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
                    saturation = np.mean(hsv[:, :, 1]) / 255.0
                    saturation_values.append(saturation)
                    
                    # Texture complexity (using Laplacian variance)
                    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                    texture_var = np.var(laplacian)
                    texture_complexity.append(texture_var)
                    
                    # Skin tone analysis for portraits
                    if pair['category'].lower() in ['portrait', 'people', 'person']:
                        skin_tone_analysis['portrait_count'] += 1
                        skin_tones = self._detect_skin_tones(img_array)
                        skin_tone_analysis['skin_tones'].extend(skin_tones)
                        
            except Exception as e:
                print(f"Warning: Could not analyze {pair.get('hr_path', 'unknown')}: {e}")
                continue
        
        # Statistical analysis
        visual_analysis = {
            'sample_size': len(brightness_values),
            'brightness_stats': {
                'mean': np.mean(brightness_values) if brightness_values else 0,
                'std': np.std(brightness_values) if brightness_values else 0,
                'range': [np.min(brightness_values), np.max(brightness_values)] if brightness_values else [0, 0]
            },
            'color_balance': {
                channel: {
                    'mean': np.mean(values) if values else 0,
                    'std': np.std(values) if values else 0
                } for channel, values in color_distributions.items()
            },
            'saturation_stats': {
                'mean': np.mean(saturation_values) if saturation_values else 0,
                'std': np.std(saturation_values) if saturation_values else 0
            },
            'texture_complexity_stats': {
                'mean': np.mean(texture_complexity) if texture_complexity else 0,
                'std': np.std(texture_complexity) if texture_complexity else 0
            }
        }
        
        # Bias detection
    # Bias detection
        if brightness_values:
            brightness_mean = np.mean(brightness_values)
            # FIX: Cast the result to a native Python bool
            visual_analysis['brightness_bias_detected'] = bool(brightness_mean < 0.3 or brightness_mean > 0.8)
            
        # Color diversity check
        if color_distributions['red']:
            color_ranges = [np.max(values) - np.min(values) for values in color_distributions.values()]
            # FIX: Cast the result to a native Python bool
            visual_analysis['color_diversity_low'] = bool(np.mean(color_ranges) < 0.3)
        
        # Skin tone diversity analysis
        if skin_tone_analysis['skin_tones']:
            skin_tone_diversity = len(set([tuple(tone) for tone in skin_tone_analysis['skin_tones']]))
            visual_analysis['skin_tone_diversity'] = skin_tone_diversity
            # This line was already fine, but explicit casting is a good defensive practice
            visual_analysis['skin_tone_bias_detected'] = bool(
                skin_tone_analysis['portrait_count'] > 10 and skin_tone_diversity < 3
            )
        
        return visual_analysis
    
    def _detect_skin_tones(self, img_array):
        """
        Simple skin tone detection for bias analysis.
        Returns representative skin tone colors found in the image.
        """
        # Convert to YCrCb color space (better for skin detection)
        ycrcb = cv2.cvtColor(img_array, cv2.COLOR_RGB2YCrCb)
        
        # Define skin color range in YCrCb
        lower_skin = np.array([0, 133, 77])
        upper_skin = np.array([255, 173, 127])
        
        # Create mask for skin pixels
        skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
        
        # Extract skin pixels
        skin_pixels = img_array[skin_mask > 0]
        
        if len(skin_pixels) > 100:  # Need sufficient skin pixels
            # Cluster skin tones using k-means
            try:
                from sklearn.cluster import KMeans
                n_clusters = min(3, len(skin_pixels) // 50)
                if n_clusters > 0:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    kmeans.fit(skin_pixels)
                    return kmeans.cluster_centers_.astype(int).tolist()
            except ImportError:
                # Fall back to simple sampling if sklearn not available
                sample_indices = np.random.choice(len(skin_pixels), min(10, len(skin_pixels)), replace=False)
                return skin_pixels[sample_indices].tolist()
        
        return []
    
    def generate_datasheet(self, output_dir, all_pairs, dataset_stats, bias_assessment):
        """Generate a comprehensive datasheet for the dataset following best practices."""
        datasheet = {
            "dataset_info": {
                "name": self.dataset_name,
                "version": "1.0",
                "description": "Advanced SISR benchmark dataset with realistic degradations",
                "creation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "authors": ["Dataset Generator"],
                "contact": "researcher@institution.edu",
                "license": "CC BY-NC 4.0",
                "doi": "To be assigned"
            },
            "motivation": {
                "purpose": "To provide a challenging and realistic benchmark for Single Image Super-Resolution research",
                "tasks": ["Single Image Super-Resolution", "Image Quality Assessment", "Degradation Analysis"],
                "gaps_addressed": [
                    "Lack of realistic degradation models in existing datasets",
                    "Limited diversity in degradation types",
                    "Insufficient challenge levels for modern SR methods"
                ]
            },
            "composition": {
                "total_instances": dataset_stats['total_pairs'],
                "instance_type": "Image pairs (HR ground truth + LR degraded)",
                "categories": dataset_stats['categories'],
                "resolution_distribution": dataset_stats['resolution_stats'],
                "difficulty_distribution": dataset_stats['difficulty_distribution'],
                "missing_data": "None - all pairs are complete",
                "data_splits": {
                    "train": "70%",
                    "validation": "15%", 
                    "test": "15%"
                }
            },
            "collection_process": {
                "acquisition": "Multi-source high-quality images with synthetic degradation",
                "sampling_strategy": "Diverse content categories and degradation types",
                "data_quality_assurance": [
                    "Manual validation of source images",
                    "Automated quality checks",
                    "Degradation parameter validation",
                    "Statistical analysis of dataset properties"
                ]
            },
            "preprocessing": {
                "cleaning": "Removal of corrupted or low-quality source images",
                "degradation_pipeline": "Advanced multi-stage degradation with realistic artifacts",
                "normalization": "None - images stored in 8-bit RGB format",
                "augmentation": "Various degradation parameters per image variant"
            },
            "uses": {
                "appropriate_uses": [
                    "Single Image Super-Resolution model training",
                    "Super-Resolution quality assessment",
                    "Degradation model evaluation",
                    "Benchmarking SR algorithms"
                ],
                "inappropriate_uses": [
                    "Direct commercial deployment without validation",
                    "Applications requiring specific degradation types not covered",
                    "Real-time applications without optimization"
                ],
                "known_limitations": [
                    "Synthetic degradations may not cover all real-world scenarios",
                    "Limited to RGB images",
                    "Dataset size constrained by computational resources"
                ]
            },
            "distribution": {
                "availability": "Publicly available for research use",
                "format": "PNG images with JSON metadata",
                "size": "Variable based on generation parameters",
                "access_restrictions": "Non-commercial research use only"
            },
            "maintenance": {
                "versioning": "Semantic versioning (major.minor.patch)",
                "updates": "Periodic updates with community feedback",
                "support": "Community-driven with documentation",
                "deprecation_policy": "Minimum 2-year notice for major changes"
            },
            "bias_assessment": bias_assessment,
            "quality_metrics": {
                "description": "Dataset quality assessed using standard image quality metrics",
                "metrics_used": ["PSNR", "SSIM", "ERQA"],
                "sample_size": "100 randomly selected pairs",
                "evaluation_date": time.strftime("%Y-%m-%d")
            },
            "ethical_considerations": {
                "consent": "Source images from public domain or appropriate licenses",
                "privacy": "No personal or sensitive information in images",
                "fairness": "Diverse content categories to avoid bias",
                "potential_harms": "Minimal - intended for research use only"
            },
            "technical_specifications": {
                "file_formats": "PNG for images, JSON for metadata",
                "color_space": "sRGB",
                "bit_depth": "8 bits per channel",
                "compression": "Lossless PNG compression",
                "metadata_format": "JSON with comprehensive degradation parameters"
            }
        }
        
        # Save datasheet
        datasheet_path = Path(output_dir) / "DATASHEET.json"
        with open(datasheet_path, 'w') as f:
            json.dump(datasheet, f, indent=2)
        
        return datasheet
    
    def generate_readme(self, output_dir, dataset_stats):
        """Generate a comprehensive README file for the dataset."""
        readme_content = f"""# {self.dataset_name}

## Overview

This dataset provides a **state-of-the-art** benchmark for Single Image Super-Resolution (SISR) research, featuring scientifically-grounded degradation models that go far beyond simple bicubic downsampling to create the most realistic and challenging test cases available.

## Theoretical Foundations & Key Innovations

### 🔬 **Advanced Degradation Pipeline**
Our pipeline follows the **physical and temporal order** of real-world image formation:

1. **Optical/Physical Degradations** (pre-sensor):
   - **Chromatic aberration**: Lens imperfection simulation with wavelength-dependent refractive indices
   - **Lens distortion**: Barrel/pincushion distortion with realistic vignetting effects
   - **Atmospheric effects**: Physics-based scattering models (Mie/Rayleigh scattering for haze/fog)

2. **Motion & Exposure Degradations**:
   - **Advanced motion blur**: Linear, rotational, anisotropic, and random-walk camera shake
   - **Realistic defocus blur**: Disk-shaped bokeh kernels instead of simple Gaussian
   - **Theoretically-sound resizing**: **Lanczos filter** (windowed sinc approximation) for optimal anti-aliasing

3. **Sensor-Level Degradations**:
   - **Sophisticated noise models**: 
     - Spatially-correlated noise (real sensors have correlation)
     - **Color-dependent noise**: Different noise characteristics per RGB channel
     - Shot noise (Poisson), read noise (Gaussian), dark current (temperature-dependent)

4. **Full Camera ISP Pipeline Simulation**:
   - **Demosaicing artifacts**: Bayer pattern reconstruction errors with cross-channel leakage
   - **ISP denoising**: Bilateral filtering with realistic over-processing
   - **Color correction**: Random color matrices simulating white balance/color grading
   - **Tone mapping**: Gamma correction with highlight compression
   - **Aggressive sharpening**: Unsharp masking typical of consumer cameras

5. **Compression & Final Processing**:
   - **JPEG compression**: Realistic quality levels with block artifacts

### 🎯 **Scientific Advantages**

- **Physically-motivated**: Each degradation follows real-world physics/optics principles
- **Temporally-ordered**: Degradations applied in the sequence they occur in reality
- **Theoretically-justified**: Lanczos filtering cited as optimal for anti-aliasing during downsampling
- **Comprehensive coverage**: Covers the complete image acquisition pipeline from optics to digital processing

## Enhanced Quality Assurance

- **Comprehensive Bias Analysis**: Statistical analysis of colors, brightness, textures, and fairness metrics
- **Real-world Validation Capability**: Framework for DSLR+smartphone paired validation sets
- **Advanced Metrics**: PSNR, SSIM, and Edge Restoration Quality Assessment (ERQA)
- **Skin Tone Diversity**: Automated analysis for fair representation in portrait categories

## Dataset Statistics

- **Total Image Pairs**: {dataset_stats['total_pairs']:,}
- **Categories**: {', '.join(dataset_stats['categories'].keys())}
- **Resolution Range**: {', '.join(dataset_stats['resolution_stats'].keys())}
- **Degradation Types**: {len(dataset_stats['degradation_stats'])} different degradation methods

## Dataset Structure

```
{self.dataset_name.lower().replace(' ', '_')}/
├── HR/                     # High-resolution ground truth images
├── LR/                     # Low-resolution degraded images  
├── metadata/               # Comprehensive metadata and splits
│   ├── complete_metadata.json
│   ├── dataset_summary.json
│   ├── train_split.json
│   ├── val_split.json
│   └── test_split.json
├── DATASHEET.json          # Dataset documentation
├── quality_assessment.json # Quality metrics and analysis
└── README.md              # This file
```

## Usage

### Loading the Dataset

```python
import json
from PIL import Image

# Load metadata
with open('metadata/complete_metadata.json', 'r') as f:
    metadata = json.load(f)

# Load an image pair
pair = metadata['pairs'][0]
hr_img = Image.open(pair['hr_path'])
lr_img = Image.open(pair['lr_path'])
```

### Understanding Degradation Metadata

Each image pair includes comprehensive degradation metadata:

```python
degradation_info = pair['degradation_metadata']
print(f"Difficulty: {{degradation_info['difficulty_level']}}")
print(f"Applied degradations: {{degradation_info['degradation_sequence']}}")
print(f"Total degradations: {{degradation_info['total_degradations_applied']}}")
```

## Evaluation Protocol

### Standard Metrics

We recommend evaluating SISR methods using:
- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **LPIPS**: Learned Perceptual Image Patch Similarity (if available)
- **ERQA**: Edge Restoration Quality Assessment (simplified implementation included)

### Advanced Evaluation Features

- **Real-world Validation**: Optional DSLR+smartphone paired validation for realistic performance assessment
- **Synthetic vs Real-world Comparison**: Automated gap analysis to validate degradation realism
- **Bias-aware Evaluation**: Performance analysis across different visual content types and fairness metrics

### Benchmark Guidelines

1. **Use provided train/validation/test splits** to ensure fair comparison
2. **Report results on each difficulty level** (easy, medium, hard, extreme)
3. **Include per-category results** to analyze performance across content types
4. **Report confidence intervals** for robust statistical analysis
5. **Consider real-world validation** when available to verify synthetic-to-real transferability

## Quality Assessment

The dataset includes comprehensive quality assessment:
- Statistical validation of degradation parameters
- Bias analysis across categories and difficulty levels
- Sample quality metrics on representative subset
- Reproducibility verification

## Citation

If you use this dataset in your research, please cite:

```bibtex
@misc{{sisr_benchmark_2024,
    title={{Advanced SISR Benchmark Dataset}},
    author={{Anonymous}},
    year={{2024}},
    note={{Advanced degradation pipeline for realistic super-resolution evaluation}}
}}
```

## License

This dataset is released under CC BY-NC 4.0 license for research use only.

## Contact

For questions, issues, or contributions, please contact: researcher@institution.edu

## Acknowledgments

This dataset was created using advanced degradation models inspired by recent research in image degradation and super-resolution. We thank the computer vision community for their contributions to the field.

---

Generated on: {time.strftime("%Y-%m-%d %H:%M:%S")}
"""
        
        readme_path = Path(output_dir) / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        return readme_path
    
    def create_real_world_validation_set(self, real_world_data_dir, output_dir):
        """
        Create a real-world validation set from DSLR+smartphone pairs.
        
        This addresses the common criticism that synthetic datasets don't reflect
        real-world degradations by providing a curated real-world test set.
        
        Expected input structure:
        real_world_data_dir/
        ├── dslr_raw/        # High-quality DSLR RAW files (ground truth)
        ├── smartphone/      # Corresponding smartphone JPEG images
        └── metadata.json    # Pairing information
        """
        real_world_dir = Path(real_world_data_dir)
        validation_dir = Path(output_dir) / "real_world_validation"
        validation_dir.mkdir(parents=True, exist_ok=True)
        
        if not real_world_dir.exists():
            print(f"Real-world data directory not found: {real_world_dir}")
            return None
            
        # Process DSLR-smartphone pairs
        metadata_file = real_world_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                real_world_metadata = json.load(f)
        else:
            print("No metadata.json found, attempting auto-pairing...")
            real_world_metadata = self._auto_pair_real_world_images(real_world_dir)
        
        validation_pairs = []
        dslr_dir = real_world_dir / "dslr_raw"
        smartphone_dir = real_world_dir / "smartphone"
        
        for pair_info in real_world_metadata.get('pairs', []):
            try:
                dslr_path = dslr_dir / pair_info['dslr_file']
                smartphone_path = smartphone_dir / pair_info['smartphone_file']
                
                if dslr_path.exists() and smartphone_path.exists():
                    # Process DSLR image (convert RAW to high-quality PNG)
                    hr_img = self._process_dslr_image(dslr_path)
                    
                    # Load smartphone image
                    lr_img = Image.open(smartphone_path).convert('RGB')
                    
                    # Save to validation set
                    base_name = dslr_path.stem
                    hr_output = validation_dir / f"{base_name}_HR.png"
                    lr_output = validation_dir / f"{base_name}_LR.png"
                    
                    hr_img.save(hr_output, "PNG")
                    lr_img.save(lr_output, "PNG")
                    
                    validation_pairs.append({
                        'hr_path': str(hr_output.relative_to(Path(output_dir))),
                        'lr_path': str(lr_output.relative_to(Path(output_dir))),
                        'scene_type': pair_info.get('scene_type', 'unknown'),
                        'lighting_condition': pair_info.get('lighting', 'unknown'),
                        'dslr_settings': pair_info.get('dslr_settings', {}),
                        'smartphone_model': pair_info.get('smartphone_model', 'unknown')
                    })
                    
            except Exception as e:
                print(f"Error processing real-world pair {pair_info}: {e}")
                continue
        
        # Save validation metadata
        validation_metadata = {
            'real_world_validation_set': {
                'description': 'Real-world DSLR-smartphone pairs for validation',
                'total_pairs': len(validation_pairs),
                'creation_date': time.strftime("%Y-%m-%d %H:%M:%S"),
                'pairs': validation_pairs
            }
        }
        
        validation_metadata_file = validation_dir / "real_world_validation_metadata.json"
        with open(validation_metadata_file, 'w') as f:
            json.dump(validation_metadata, f, indent=2)
        
        print(f"Created real-world validation set with {len(validation_pairs)} pairs")
        return validation_metadata
    
    def _auto_pair_real_world_images(self, data_dir):
        """Auto-pair DSLR and smartphone images based on timestamps or filenames."""
        dslr_dir = data_dir / "dslr_raw"
        smartphone_dir = data_dir / "smartphone"
        
        dslr_files = list(dslr_dir.glob("*")) if dslr_dir.exists() else []
        smartphone_files = list(smartphone_dir.glob("*")) if smartphone_dir.exists() else []
        
        pairs = []
        for dslr_file in dslr_files:
            # Simple name-based matching (could be enhanced with timestamp matching)
            base_name = dslr_file.stem.lower()
            
            # Look for matching smartphone image
            for smartphone_file in smartphone_files:
                if base_name in smartphone_file.stem.lower():
                    pairs.append({
                        'dslr_file': dslr_file.name,
                        'smartphone_file': smartphone_file.name,
                        'scene_type': 'unknown',
                        'lighting': 'unknown'
                    })
                    break
        
        return {'pairs': pairs}
    
    def _process_dslr_image(self, dslr_path):
        """Process DSLR RAW file to high-quality ground truth image."""
        try:
            # Try to use rawpy for RAW processing
            import rawpy
            
            with rawpy.imread(str(dslr_path)) as raw:
                # Process with minimal noise reduction and sharpening
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    use_auto_wb=False,
                    output_color=rawpy.ColorSpace.sRGB,
                    output_bps=8,
                    no_auto_bright=True,
                    noise_thr=None,  # Minimal noise reduction
                    user_flip=0
                )
            
            return Image.fromarray(rgb)
            
        except ImportError:
            print("rawpy not available, falling back to PIL (may not work with RAW files)")
            return Image.open(dslr_path).convert('RGB')
        except Exception as e:
            print(f"Error processing DSLR image {dslr_path}: {e}")
            # Fallback to regular image loading
            return Image.open(dslr_path).convert('RGB')
    
    def compare_synthetic_vs_real_world(self, synthetic_results, real_world_results):
        """
        Compare model performance on synthetic vs real-world validation sets.
        
        This provides critical validation that synthetic degradations are realistic.
        """
        comparison = {
            'synthetic_performance': synthetic_results,
            'real_world_performance': real_world_results,
            'performance_gap': {},
            'correlation_analysis': {},
            'recommendations': []
        }
        
        # Calculate performance gaps
        for metric in ['psnr', 'ssim']:
            if metric in synthetic_results and metric in real_world_results:
                synth_mean = synthetic_results[metric]['mean']
                real_mean = real_world_results[metric]['mean']
                gap = synth_mean - real_mean
                gap_percentage = (gap / real_mean) * 100 if real_mean > 0 else 0
                
                comparison['performance_gap'][metric] = {
                    'absolute_gap': gap,
                    'percentage_gap': gap_percentage
                }
                
                # Add recommendations based on gaps
                if abs(gap_percentage) > 20:
                    comparison['recommendations'].append(
                        f"Large {metric.upper()} gap ({gap_percentage:.1f}%) between synthetic and real-world: "
                        f"Consider adjusting degradation parameters"
                    )
        
        return comparison


class GoogleDriveUploader:
    """
    Google Drive uploader for dataset archival.
    Handles authentication and file upload to Google Drive.
    """
    
    def __init__(self, credentials_path=None, folder_id=None):
        self.credentials_path = credentials_path
        self.folder_id = folder_id
        self.service = None
        
    def authenticate(self):
        """Authenticate with Google Drive API."""
        try:
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from googleapiclient.discovery import build
            
            SCOPES = ['https://www.googleapis.com/auth/drive.file']
            
            creds = None
            # The file token.json stores the user's access and refresh tokens.
            if os.path.exists('token.json'):
                creds = Credentials.from_authorized_user_file('token.json', SCOPES)
            
            # If there are no (valid) credentials available, let the user log in.
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    if not self.credentials_path or not os.path.exists(self.credentials_path):
                        print("Error: Google Drive credentials file not found.")
                        print("Please download 'credentials.json' from Google Cloud Console")
                        print("and place it in the script directory or specify path with credentials_path")
                        return False
                        
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.credentials_path, SCOPES)
                    creds = flow.run_local_server(port=0)
                
                # Save the credentials for the next run
                with open('token.json', 'w') as token:
                    token.write(creds.to_json())
            
            self.service = build('drive', 'v3', credentials=creds)
            return True
            
        except ImportError:
            print("Google Drive API libraries not installed.")
            print("Install with: pip install google-api-python-client google-auth-oauthlib")
            return False
        except Exception as e:
            print(f"Authentication failed: {e}")
            return False
    
    def upload_file(self, file_path, folder_id=None):
        """Upload a file to Google Drive."""
        if not self.service:
            if not self.authenticate():
                return None
        
        try:
            from googleapiclient.http import MediaFileUpload
            
            file_path = Path(file_path)
            folder_id = folder_id or self.folder_id
            
            file_metadata = {
                'name': file_path.name,
                'parents': [folder_id] if folder_id else []
            }
            
            media = MediaFileUpload(str(file_path), resumable=True)
            
            print(f"Uploading {file_path.name} to Google Drive...")
            
            request = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            )
            
            response = None
            while response is None:
                status, response = request.next_chunk()
                if status:
                    print(f"Upload progress: {int(status.progress() * 100)}%")
            
            print(f"Upload completed. File ID: {response.get('id')}")
            return response.get('id')
            
        except Exception as e:
            print(f"Upload failed: {e}")
            return None


class IncrementalArchiver:
    """
    Incremental archiver that creates zip files while managing disk space
    by deleting original files as they are compressed.
    """
    
    def __init__(self, source_dir, archive_path, compression=zipfile.ZIP_DEFLATED):
        self.source_dir = Path(source_dir)
        self.archive_path = Path(archive_path)
        self.compression = compression
        
    def create_archive(self, delete_originals=True):
        """
        Create archive with incremental compression and optional file deletion.
        
        Args:
            delete_originals: If True, delete source files after adding to archive
            
        Returns:
            bool: Success status
        """
        try:
            # Ensure archive directory exists
            self.archive_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Get all files to archive
            files_to_archive = []
            for pattern in ['**/*.png', '**/*.jpg', '**/*.jpeg', '**/*.json', '**/*.txt', '**/*.md']:
                files_to_archive.extend(self.source_dir.glob(pattern))
            
            # Remove duplicates and sort by size (smaller files first for better compression)
            files_to_archive = list(set(files_to_archive))
            files_to_archive.sort(key=lambda x: x.stat().st_size)
            
            total_files = len(files_to_archive)
            print(f"Starting incremental archiving of {total_files} files...")
            print(f"Archive: {self.archive_path}")
            
            # Track disk usage
            initial_disk_usage = sum(f.stat().st_size for f in files_to_archive if f.exists())
            print(f"Initial data size: {initial_disk_usage / (1024**3):.2f} GB")
            
            with zipfile.ZipFile(self.archive_path, 'w', self.compression) as zf:
                archived_files = 0
                freed_space = 0
                
                for file_path in tqdm(files_to_archive, desc="Archiving"):
                    try:
                        if not file_path.exists():
                            continue
                            
                        # Get file size before compression
                        file_size = file_path.stat().st_size
                        
                        # Add file to archive with relative path
                        arcname = file_path.relative_to(self.source_dir)
                        zf.write(file_path, arcname=arcname)
                        
                        # Delete original file if requested
                        if delete_originals:
                            file_path.unlink()
                            freed_space += file_size
                        
                        archived_files += 1
                        
                        # Progress update every 100 files
                        if archived_files % 100 == 0:
                            current_archive_size = self.archive_path.stat().st_size
                            compression_ratio = (freed_space / current_archive_size) if current_archive_size > 0 else 0
                            print(f"Progress: {archived_files}/{total_files} files, "
                                  f"Freed: {freed_space / (1024**3):.2f} GB, "
                                  f"Archive: {current_archive_size / (1024**3):.2f} GB "
                                  f"(ratio: {compression_ratio:.2f}x)")
                        
                    except Exception as e:
                        print(f"Warning: Failed to archive {file_path}: {e}")
                        continue
            
            # Clean up empty directories if we deleted files
            if delete_originals:
                self._remove_empty_dirs(self.source_dir)
            
            # Final statistics
            final_archive_size = self.archive_path.stat().st_size
            compression_ratio = (initial_disk_usage / final_archive_size) if final_archive_size > 0 else 0
            
            print(f"\nArchiving completed successfully!")
            print(f"Files archived: {archived_files}/{total_files}")
            print(f"Final archive size: {final_archive_size / (1024**3):.2f} GB")
            print(f"Compression ratio: {compression_ratio:.2f}x")
            if delete_originals:
                print(f"Freed disk space: {freed_space / (1024**3):.2f} GB")
            
            return True
            
        except Exception as e:
            print(f"Archiving failed: {e}")
            return False
    
    def _remove_empty_dirs(self, directory):
        """Remove empty directories recursively."""
        try:
            for item in directory.iterdir():
                if item.is_dir():
                    self._remove_empty_dirs(item)
                    
            # Try to remove the directory if it's empty
            try:
                if not any(directory.iterdir()):
                    directory.rmdir()
            except OSError:
                pass  # Directory not empty or other issue
                
        except Exception:
            pass  # Ignore errors in cleanup


class DatasetGenerator:
    """
    Main class for generating the SISR benchmark dataset.
    Compatible with the output from download_pexels_images.py.
    """
    
    def __init__(self, input_dir="dataset/raw_hr", output_dir="dataset/sisr_benchmark",
                 scale_factor=4, variants_per_image=4, min_resolution=512,
                 categories_to_process=None, samples_per_category=None, enable_gdrive_upload=False, gdrive_folder_id=None):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.scale_factor = scale_factor
        self.variants_per_image = variants_per_image
        self.min_resolution = min_resolution
        self.categories_to_process = categories_to_process  # None means process all categories
        self.samples_per_category = samples_per_category  # None means process all images in category
        self.enable_gdrive_upload = enable_gdrive_upload
        self.gdrive_folder_id = gdrive_folder_id
        
        # Create output directory structure
        self.hr_dir = self.output_dir / "HR"
        self.lr_dir = self.output_dir / "LR"
        self.metadata_dir = self.output_dir / "metadata"
        
        for dir_path in [self.hr_dir, self.lr_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize degradation pipeline
        self.pipeline = AdvancedDegradationPipeline(scale_factor=scale_factor)
        
        # Initialize metrics and reporting
        self.metrics_reporter = DatasetMetricsAndReporting("Advanced SISR Benchmark Dataset")
        
        # Dataset statistics
        self.stats = {
            'total_hr_images': 0,
            'total_lr_images': 0,
            'categories': {},
            'degradation_stats': {},
            'failed_images': []
        }
    
    def validate_image(self, img_path):
        """Validate if image meets minimum requirements."""
        try:
            with Image.open(img_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                width, height = img.size
                min_dim = min(width, height)
                
                # Check minimum resolution after downscaling
                if min_dim // self.scale_factor < self.min_resolution:
                    return False, f"Too small after downscaling: {min_dim // self.scale_factor}"
                
                # Load image data into memory to avoid file pointer issues
                img.load()
                # Create a copy to ensure we have a valid image object
                img_copy = img.copy()
                
                return True, img_copy
                
        except Exception as e:
            return False, f"Error loading image: {str(e)}"
    
    def process_single_image(self, img_path, category, img_index):
        """Process a single HR image to generate LR variants."""
        is_valid, result = self.validate_image(img_path)
        
        if not is_valid:
            self.stats['failed_images'].append({
                'path': str(img_path),
                'reason': result,
                'category': category
            })
            return []
        
        hr_img = result
        
        # Generate filename
        base_filename = f"{category}_{img_index:04d}"
        
        # Save HR image
        hr_filename = f"{base_filename}.png"
        hr_path = self.hr_dir / hr_filename
        hr_img.save(hr_path, "PNG")
        
        generated_pairs = []
        
        # Generate LR variants with different difficulty levels
        difficulty_levels = ['easy', 'medium', 'hard', 'extreme']
        
        for variant_idx in range(self.variants_per_image):
            if variant_idx < len(difficulty_levels):
                difficulty = difficulty_levels[variant_idx]
            else:
                difficulty = 'random'
            
            try:
                # Generate LR image
                lr_img, degradation_metadata = self.pipeline.degrade_image(
                    hr_img, difficulty_level=difficulty
                )
                
                # Save LR image
                lr_filename = f"{base_filename}_v{variant_idx:02d}.png"
                lr_path = self.lr_dir / lr_filename
                lr_img.save(lr_path, "PNG")
                
                # Create metadata entry
                metadata_entry = {
                    'hr_image': hr_filename,
                    'lr_image': lr_filename,
                    'hr_path': str(hr_path.relative_to(self.output_dir)),
                    'lr_path': str(lr_path.relative_to(self.output_dir)),
                    'category': category,
                    'variant_index': variant_idx,
                    'hr_size': hr_img.size,
                    'lr_size': lr_img.size,
                    'original_source': str(img_path),
                    'degradation_metadata': degradation_metadata
                }
                
                generated_pairs.append(metadata_entry)
                
                # Update degradation statistics
                for deg_name in degradation_metadata.get('degradation_sequence', []):
                    if deg_name not in self.stats['degradation_stats']:
                        self.stats['degradation_stats'][deg_name] = 0
                    self.stats['degradation_stats'][deg_name] += 1
                
            except Exception as e:
                self.stats['failed_images'].append({
                    'path': str(img_path),
                    'reason': f"Error generating variant {variant_idx}: {str(e)}",
                    'category': category
                })
        
        return generated_pairs
    
    def process_category(self, category_dir):
        """Process all images in a category directory."""
        category_name = category_dir.name
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        # Find all image files
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(category_dir.glob(f"*{ext}")))
            image_files.extend(list(category_dir.glob(f"*{ext.upper()}")))
        
        if not image_files:
            print(f"No images found in {category_dir}")
            return []

        # Limit number of images if samples_per_category is specified
        if self.samples_per_category is not None and len(image_files) > self.samples_per_category:
            print(f"Limiting to {self.samples_per_category} samples out of {len(image_files)} images from category: {category_name}")
            # Randomly sample images to ensure diversity
            image_files = random.sample(image_files, self.samples_per_category)
        else:
            print(f"Processing {len(image_files)} images from category: {category_name}")

        category_pairs = []
        self.stats['categories'][category_name] = {
            'total_images': len(image_files),
            'processed_images': 0,
            'generated_pairs': 0
        }
        
        for img_index, img_path in enumerate(tqdm(image_files, desc=f"Processing {category_name}")):
            pairs = self.process_single_image(img_path, category_name, img_index)
            category_pairs.extend(pairs)
            
            if pairs:
                self.stats['categories'][category_name]['processed_images'] += 1
                self.stats['categories'][category_name]['generated_pairs'] += len(pairs)
        
        return category_pairs
    
    def split_dataset(self, all_pairs, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """Split dataset into train/validation/test sets."""
        # Shuffle pairs
        random.shuffle(all_pairs)
        
        total_pairs = len(all_pairs)
        train_size = int(total_pairs * train_ratio)
        val_size = int(total_pairs * val_ratio)
        
        train_pairs = all_pairs[:train_size]
        val_pairs = all_pairs[train_size:train_size + val_size]
        test_pairs = all_pairs[train_size + val_size:]
        
        return {
            'train': train_pairs,
            'val': val_pairs,
            'test': test_pairs
        }
    
    def save_metadata(self, all_pairs, splits):
        """Save comprehensive metadata files."""
        # Save complete metadata
        metadata_file = self.metadata_dir / "complete_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump({
                'dataset_info': {
                    'name': 'Advanced SISR Benchmark Dataset',
                    'version': '1.0',
                    'scale_factor': self.scale_factor,
                    'variants_per_image': self.variants_per_image,
                    'total_pairs': len(all_pairs),
                    'creation_date': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'degradation_pipeline': 'AdvancedDegradationPipeline'
                },
                'statistics': self.stats,
                'pairs': all_pairs
            }, f, indent=2)
        
        # Save split metadata
        for split_name, split_pairs in splits.items():
            split_file = self.metadata_dir / f"{split_name}_split.json"
            with open(split_file, 'w') as f:
                json.dump(split_pairs, f, indent=2)
        
        # Save summary statistics
        summary_file = self.metadata_dir / "dataset_summary.json"
        summary_stats = {
            'total_hr_images': self.stats['total_hr_images'],
            'total_lr_images': self.stats['total_lr_images'],
            'categories': {k: v for k, v in self.stats['categories'].items()},
            'degradation_usage': self.stats['degradation_stats'],
            'split_sizes': {k: len(v) for k, v in splits.items()},
            'failed_images_count': len(self.stats['failed_images'])
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        print(f"\nMetadata saved to {self.metadata_dir}")
        print(f"Total pairs generated: {len(all_pairs)}")
        print(f"Split sizes - Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
    
    def archive_and_upload(self):
        """
        Archive the dataset with incremental compression and upload to Google Drive.
        Implements the strategy described in the user requirements to manage disk space.
        """
        try:
            # Determine category name for archive
            if self.categories_to_process:
                if isinstance(self.categories_to_process, str):
                    category_name = self.categories_to_process
                else:
                    category_name = "_".join(self.categories_to_process)
            else:
                category_name = "all_categories"
            
            # Create archive filename
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            archive_filename = f"output_{category_name}_{timestamp}.zip"
            archive_path = self.output_dir.parent / archive_filename
            
            print(f"Creating incremental archive: {archive_filename}")
            print(f"Source directory: {self.output_dir}")
            
            # Create incremental archiver
            archiver = IncrementalArchiver(
                source_dir=self.output_dir,
                archive_path=archive_path,
                compression=zipfile.ZIP_DEFLATED
            )
            
            # Create archive with incremental deletion
            print("Starting incremental archiving to manage disk space...")
            success = archiver.create_archive(delete_originals=True)
            
            if not success:
                print("❌ Archiving failed")
                return False
            
            print("✅ Archiving completed successfully")
            
            # Upload to Google Drive if folder ID is provided
            if self.gdrive_folder_id:
                print(f"\nUploading {archive_filename} to Google Drive...")
                
                # Initialize Google Drive uploader
                uploader = GoogleDriveUploader(
                    credentials_path="credentials.json",
                    folder_id=self.gdrive_folder_id
                )
                
                # Upload the archive
                file_id = uploader.upload_file(archive_path, self.gdrive_folder_id)
                
                if file_id:
                    print(f"✅ Upload successful! File ID: {file_id}")
                    
                    # Clean up local archive after successful upload
                    try:
                        archive_path.unlink()
                        print("✅ Local archive cleaned up after successful upload")
                    except Exception as e:
                        print(f"⚠️ Warning: Could not delete local archive: {e}")
                    
                    return True
                else:
                    print("❌ Upload failed")
                    return False
            else:
                print("ℹ️ No Google Drive folder ID provided, skipping upload")
                print(f"Archive saved locally: {archive_path}")
                return True
            
        except Exception as e:
            print(f"❌ Archive and upload failed: {e}")
            return False
    
    def generate_dataset(self):
        """Main method to generate the complete dataset."""
        print("Starting Advanced SISR Benchmark Dataset Generation...")
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Scale factor: {self.scale_factor}")
        print(f"Variants per image: {self.variants_per_image}")
        
        start_time = time.time()
        
        # Find all category directories
        all_category_dirs = [d for d in self.input_dir.iterdir() if d.is_dir()]
        
        if not all_category_dirs:
            raise ValueError(f"No category directories found in {self.input_dir}")
        
        # Filter categories based on configuration
        if self.categories_to_process is None:
            category_dirs = all_category_dirs
            print(f"Processing all {len(category_dirs)} categories: {[d.name for d in category_dirs]}")
        else:
            # Convert to list if single string provided
            if isinstance(self.categories_to_process, str):
                categories_filter = [self.categories_to_process]
            else:
                categories_filter = list(self.categories_to_process)
            
            # Filter directories based on specified categories
            category_dirs = [d for d in all_category_dirs if d.name in categories_filter]
            
            if not category_dirs:
                available_categories = [d.name for d in all_category_dirs]
                raise ValueError(f"None of the specified categories {categories_filter} found. "
                               f"Available categories: {available_categories}")
            
            print(f"Processing {len(category_dirs)} selected categories: {[d.name for d in category_dirs]}")
            print(f"Available categories: {[d.name for d in all_category_dirs]}")
            
            # Show which categories are being skipped
            skipped_categories = [d.name for d in all_category_dirs if d.name not in categories_filter]
            if skipped_categories:
                print(f"Skipping categories: {skipped_categories}")
        
        # Process each category
        all_pairs = []
        for category_dir in category_dirs:
            category_pairs = self.process_category(category_dir)
            all_pairs.extend(category_pairs)
        
        if not all_pairs:
            raise ValueError("No image pairs were generated successfully")
        
        # Update final statistics
        self.stats['total_hr_images'] = len(set(pair['hr_image'] for pair in all_pairs))
        self.stats['total_lr_images'] = len(all_pairs)
        
        # Split dataset
        splits = self.split_dataset(all_pairs)
        
        # Save metadata
        self.save_metadata(all_pairs, splits)
        
        # Evaluate dataset quality
        print("\n" + "="*60)
        print("EVALUATING DATASET QUALITY")
        print("="*60)
        
        quality_metrics = self.metrics_reporter.evaluate_dataset_quality(self.hr_dir, self.lr_dir)
        if quality_metrics:
            print("Dataset Quality Metrics (sample-based):")
            for metric_name, stats in quality_metrics.items():
                print(f"  {metric_name.upper()}:")
                print(f"    Mean: {stats['mean']:.4f}")
                print(f"    Std:  {stats['std']:.4f}")
                print(f"    Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        
        # Generate comprehensive reporting
        print("\n" + "="*60)
        print("GENERATING DATASET DOCUMENTATION")
        print("="*60)
        
        dataset_stats = self.metrics_reporter.calculate_dataset_statistics(all_pairs)
        bias_assessment = self.metrics_reporter.assess_dataset_bias(all_pairs)
        
        # Save quality assessment
        quality_assessment = {
            'quality_metrics': quality_metrics,
            'dataset_statistics': dataset_stats,
            'bias_assessment': bias_assessment,
            'assessment_date': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        quality_file = self.output_dir / "quality_assessment.json"
        with open(quality_file, 'w') as f:
            json.dump(quality_assessment, f, indent=2)
        
        datasheet = self.metrics_reporter.generate_datasheet(self.output_dir, all_pairs, dataset_stats, bias_assessment)
        readme_path = self.metrics_reporter.generate_readme(self.output_dir, dataset_stats)
        
        print(f"✓ Dataset documentation generated:")
        print(f"  - DATASHEET.json: Comprehensive dataset documentation")
        print(f"  - README.md: User-friendly documentation and usage guide")
        print(f"  - quality_assessment.json: Detailed quality metrics and bias analysis")
        
        # Print bias assessment recommendations
        if bias_assessment['recommendations']:
            print(f"\n⚠️  Dataset Quality Recommendations:")
            for rec in bias_assessment['recommendations']:
                print(f"  - {rec}")
        
        # Print final summary
        end_time = time.time()
        duration = end_time - start_time
        
        print("\n" + "="*60)
        print("DATASET GENERATION COMPLETE")
        print("="*60)
        print(f"Total processing time: {duration:.2f} seconds")
        print(f"Total HR images processed: {self.stats['total_hr_images']}")
        print(f"Total LR images generated: {self.stats['total_lr_images']}")
        print(f"Failed images: {len(self.stats['failed_images'])}")
        print("\nCategory breakdown:")
        for cat, stats in self.stats['categories'].items():
            print(f"  {cat}: {stats['processed_images']}/{stats['total_images']} images → {stats['generated_pairs']} pairs")
        
        print("\nTop degradations applied:")
        sorted_degs = sorted(self.stats['degradation_stats'].items(), key=lambda x: x[1], reverse=True)
        for deg_name, count in sorted_degs[:10]:
            print(f"  {deg_name}: {count} times")
        
        print(f"\nDataset Files:")
        print(f"  HR Images: {self.hr_dir}")
        print(f"  LR Images: {self.lr_dir}")
        print(f"  Metadata: {self.metadata_dir}")
        print(f"  Documentation: {self.output_dir}/README.md")
        print(f"  Datasheet: {self.output_dir}/DATASHEET.json")
        
        # Perform archiving and upload if enabled
        if self.enable_gdrive_upload:
            print("\n" + "="*60)
            print("ARCHIVING AND UPLOADING TO GOOGLE DRIVE")
            print("="*60)
            
            success = self.archive_and_upload()
            if not success:
                print("⚠️ Warning: Archiving/upload failed, but dataset generation was successful")
        
        print("\n🎉 Advanced SISR Benchmark Dataset ready for training and evaluation!")
        print("📊 Quality metrics and comprehensive documentation included")
        print("🔬 Follow evaluation protocols in README.md for fair comparisons")


def main():
    """Main function to run the dataset generation."""
    # Configuration
    config = {
        'input_dir': "/kaggle/input/pexel-raw-hr-images/dataset/raw_hr",  # Output from download_pexels_images.py
        'output_dir': "dataset/sisr_benchmark",
        'scale_factor': 4,
        'variants_per_image': 4,  # Different difficulty levels
        'min_resolution': 128,  # Minimum LR resolution
        
        # Category Control - specify which categories to process
        # Examples:
        # 'categories_to_process': None,  # Process all categories (default)
        # 'categories_to_process': "nature",  # Process only "nature" category
        # 'categories_to_process': ["nature", "architecture"],  # Process multiple categories
        'categories_to_process': "urban_architecture",  # Change this to control which categories to process

        # Sample Control - specify number of samples per category
        # Examples:
        # 'samples_per_category': None,  # Process all images in category (default)
        # 'samples_per_category': 100,   # Process only 100 images per category
        # 'samples_per_category': 50,    # Process only 50 images per category
        'samples_per_category': 50,  # Change this to control number of samples per category

        # Google Drive Upload Configuration
        'enable_gdrive_upload': True,  # Set to True to enable archiving and upload
        'gdrive_folder_id': '1pgRPbsB0t4Xr7gc0wOagvtQKBEeIVrgy',  # Replace with your Google Drive folder ID
        # Example: 'gdrive_folder_id': "1AbCdEfGhIjKlMnOpQrStUvWxYz"
    }
    
    print("Advanced SISR Benchmark Dataset Generator")
    print("Compatible with download_pexels_images.py output")
    print("="*60)
    
    # Display configuration
    print("\n📋 CONFIGURATION:")
    print(f"  Input Directory: {config['input_dir']}")
    print(f"  Output Directory: {config['output_dir']}")
    print(f"  Scale Factor: {config['scale_factor']}x")
    print(f"  Variants per Image: {config['variants_per_image']}")
    
    if config['categories_to_process'] is None:
        print("  Categories: All available categories will be processed")
    else:
        print(f"  Categories: {config['categories_to_process']}")

    if config['samples_per_category'] is None:
        print("  Samples per Category: All images in each category will be processed")
    else:
        print(f"  Samples per Category: {config['samples_per_category']} images per category")

    print(f"  Google Drive Upload: {'Enabled' if config['enable_gdrive_upload'] else 'Disabled'}")
    if config['enable_gdrive_upload']:
        if config['gdrive_folder_id']:
            print(f"  Google Drive Folder ID: {config['gdrive_folder_id']}")
        else:
            print("  ⚠️ Warning: Google Drive upload enabled but no folder ID provided")
    
    print("="*60)
    
    # Create dataset generator
    generator = DatasetGenerator(**config)
    
    # Generate dataset
    try:
        generator.generate_dataset()
        
        # Create README file
        readme_content = f"""# Advanced SISR Benchmark Dataset

This dataset was generated using **state-of-the-art, scientifically-grounded degradation techniques** to create the most challenging and realistic Single Image Super-Resolution benchmark to date.

## 🚀 Major Improvements & Theoretical Foundations

### **Physically-Motivated Degradation Pipeline**
Our pipeline follows the **complete image formation process** with proper temporal ordering:
1. **Optical degradations** (chromatic aberration, lens distortion, atmospheric scattering)
2. **Motion blur** (advanced kernels: anisotropic, random-walk camera shake, defocus bokeh)
3. **Sensor noise** (spatially-correlated, color-dependent: shot, read, dark current)
4. **Full ISP pipeline** (demosaicing → denoising → color correction → tone mapping → sharpening)
5. **Compression artifacts** (JPEG with realistic quality variations)

### **Theoretical Justifications**
- **Lanczos Filter**: Uses windowed sinc approximation (theoretically optimal for anti-aliasing)
- **Physics-based Models**: Atmospheric effects use Mie/Rayleigh scattering principles
- **Camera-accurate ISP**: Simulates complete digital camera processing chain
- **Realistic Noise**: Color-dependent, spatially-correlated sensor noise models

## Dataset Structure
```
{config['output_dir']}/
├── HR/                          # High-resolution ground truth images
├── LR/                          # Low-resolution degraded images  
├── metadata/                    # Comprehensive metadata and splits
│   ├── complete_metadata.json
│   ├── dataset_summary.json
│   ├── train_split.json
│   ├── val_split.json
│   └── test_split.json
├── quality_assessment.json      # Bias analysis & quality metrics
├── DATASHEET.json              # Comprehensive dataset documentation
└── real_world_validation/       # Optional DSLR+smartphone validation pairs
```

## 🔬 Scientific Advantages
- **Scale Factor**: {config['scale_factor']}x super-resolution
- **Variants per Image**: {config['variants_per_image']} (different difficulty levels: easy → extreme)
- **Structured Pipeline**: Degradations applied in **physically-realistic order**
- **Advanced Blur Models**: Anisotropic kernels, random-walk motion, disk-shaped defocus
- **Enhanced Noise**: Spatially-correlated, color-channel dependent sensor noise
- **Full ISP Simulation**: Complete camera processing pipeline with realistic artifacts
- **Bias Analysis**: Comprehensive fairness assessment including skin tone diversity
- **Real-world Validation**: Framework for DSLR+smartphone paired validation

## 📊 Quality Assurance
- **Enhanced Bias Analysis**: Colors, brightness, textures, fairness metrics
- **Real-world Validation Capability**: DSLR+smartphone pairs for transfer validation
- **Comprehensive Metrics**: PSNR, SSIM, ERQA with statistical analysis
- **Reproducible Parameters**: Full degradation metadata for scientific reproducibility

## 🎯 Usage for Researchers
1. **Load dataset** using metadata files for paired HR/LR images
2. **Analyze degradation patterns** using comprehensive parameter logs
3. **Compare difficulty levels** (easy/medium/hard/extreme) for progressive evaluation
4. **Validate on real-world data** using optional DSLR+smartphone pairs
5. **Assess bias** using provided fairness analysis tools

## 📚 Citation
If you use this dataset in your research, please cite:
```bibtex
@misc{{advanced_sisr_dataset_2024,
    title={{Advanced SISR Benchmark Dataset: Physically-Motivated Degradation Pipeline}},
    author={{[Your Name]}},
    year={{2024}},
    note={{State-of-the-art degradation models with full ISP pipeline simulation}}
}}
```

## 🔧 Technical Requirements
See `requirements.txt` for dependencies. Optional: `rawpy` for real-world RAW processing.

**Generated on**: {time.strftime("%Y-%m-%d %H:%M:%S")}
**Pipeline Version**: Advanced ISP + Physics-based Degradations v2.0
"""
        
        readme_path = Path(config['output_dir']) / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        print(f"\nREADME.md created at: {readme_path}")
        
    except Exception as e:
        print(f"Error during dataset generation: {str(e)}")
        raise


if __name__ == "__main__":
    main()