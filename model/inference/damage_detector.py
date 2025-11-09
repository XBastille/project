"""
Production damage detection pipeline
Minimal implementation - all contained in model folder
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import torch
from torch import nn
import math
from datetime import datetime, timedelta

# Add parent to path
model_dir = Path(__file__).parent.parent
sys.path.insert(0, str(model_dir))

# Earth Engine import
try:
    import ee
    HAS_GEE = True
except ImportError:
    ee = None
    HAS_GEE = False


def preprocess_inputs(x):
    """Preprocess images for xView2 models (normalize to [-1, 1])"""
    x = np.asarray(x, dtype='float32')
    x /= 127
    x -= 1
    return x


def calculate_padding(image, patch_size=256):
    """Calculate required padding for patch-based inference"""
    height, width = image.shape[-3:-1]
    xx = (width // patch_size + 1) * patch_size
    yy = (height // patch_size + 1) * patch_size
    x_pad = xx - width
    y_pad = yy - height
    x0 = x_pad // 2
    x1 = x_pad - x0
    y0 = y_pad // 2
    y1 = y_pad - y0
    return (y0, y1, x0, x1)


def apply_padding(image, patch_size=256):
    """Add padding to image for patch-based inference"""
    pad_y0, pad_y1, pad_x0, pad_x1 = calculate_padding(image, patch_size)
    return np.pad(image, ((pad_y0, pad_y1), (pad_x0, pad_x1), (0, 0)), mode='constant')


def segment_buildings(model, img, threshold=0.5, patch_size=256):
    """
    Segment buildings from satellite imagery
    Args:
        model: Keras segmentation model
        img: numpy array (H, W, C), float values 0-1
        threshold: confidence threshold
    Returns:
        mask: binary mask of buildings (H, W, 1)
    """
    padded = apply_padding(img, patch_size)
    number_of_rows = math.ceil(img.shape[0] / patch_size)
    number_of_columns = math.ceil(img.shape[1] / patch_size)
    prediction = np.zeros((number_of_rows * patch_size, number_of_columns * patch_size, 1), dtype="float32")
    
    for i in range(number_of_rows):
        for j in range(number_of_columns):
            slc = padded[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] / 256.0
            pred = model.predict(np.expand_dims(slc, axis=0), verbose=0)
            prediction[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = pred[0]
    
    prediction = (prediction > threshold).astype(np.uint8)
    y0, y1, x0, x1 = calculate_padding(img, patch_size)
    return prediction[y0:-y1 if y1 > 0 else None, x0:-x1 if x1 > 0 else None]


class DamageDetector:
    """
    Minimal damage detection system
    Uses: Building Segmentation + xView2 Damage Classification
    """
    
    def __init__(self, 
                 segmentation_model_path=None,
                 damage_weights_path=None,
                 use_gpu=False):
        """
        Initialize detector
        Args:
            segmentation_model_path: path to building segmentation model
            damage_weights_path: path to xView2 damage classification weights
            use_gpu: whether to use GPU for inference
        """
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        
        # Set default paths
        if segmentation_model_path is None:
            segmentation_model_path = model_dir / 'segment' / 'building_segmentation_model.keras'
        
        if damage_weights_path is None:
            damage_weights_path = model_dir / 'weights' / 'weights' / 'res34_cls2_0_tuned_best'
        
        self.seg_model_path = Path(segmentation_model_path)
        self.damage_weights_path = Path(damage_weights_path)
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load building segmentation and damage classification models"""
        # Load building segmentation
        print("[LOADING] Building segmentation model...")
        if self.seg_model_path.exists():
            try:
                from keras.models import load_model
                self.seg_model = load_model(self.seg_model_path)
                print(f"  [OK] Loaded from {self.seg_model_path.name}")
            except Exception as e:
                print(f"  [WARNING] Could not load model: {e}")
                print(f"  [INFO] Will use fallback (no building segmentation)")
                self.seg_model = None
        else:
            print(f"  [WARNING] Model not found: {self.seg_model_path}")
            self.seg_model = None
        
        # Load damage classification
        print("[LOADING] Damage classification model (ResNet34)...")
        if self.damage_weights_path.exists():
            from zoo_models import Res34_Unet_Double
            
            self.damage_model = Res34_Unet_Double(pretrained=False)
            
            # Load weights
            try:
                checkpoint = torch.load(self.damage_weights_path, map_location=self.device, weights_only=False)
            except TypeError:
                # Older PyTorch version
                checkpoint = torch.load(self.damage_weights_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Remove 'module.' prefix if present (DataParallel)
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k.replace('module.', '') if k.startswith('module.') else k
                new_state_dict[name] = v
            
            self.damage_model.load_state_dict(new_state_dict)
            self.damage_model = self.damage_model.to(self.device)
            self.damage_model.eval()
            print(f"  [OK] Loaded from {self.damage_weights_path.name}")
        else:
            print(f"  [WARNING] Weights not found: {self.damage_weights_path}")
            self.damage_model = None
    
    def detect_buildings(self, image):
        """
        Detect buildings in image
        Args:
            image: numpy array (H, W, 3), uint8 [0-255]
        Returns:
            mask: binary mask of buildings (H, W, 1)
        """
        if self.seg_model is None:
            # Return empty mask
            return np.zeros((image.shape[0], image.shape[1], 1), dtype=np.uint8)
        
        img_normalized = image.astype(float) / 255.0
        mask = segment_buildings(self.seg_model, img_normalized)
        return mask
    
    def classify_damage(self, before_img, after_img):
        """
        Classify damage between before/after images
        Args:
            before_img: numpy array (H, W, 3), uint8 [0-255]
            after_img: numpy array (H, W, 3), uint8 [0-255]
        Returns:
            damage_map: (H, W, 5) - probabilities for each damage class
        """
        if self.damage_model is None:
            # Fallback to simple change detection
            return self._simple_damage_detection(before_img, after_img)
        
        # Resize to 1024x1024 for xView2 model
        h, w = before_img.shape[:2]
        before_resized = cv2.resize(before_img, (1024, 1024))
        after_resized = cv2.resize(after_img, (1024, 1024))
        
        # Concatenate before and after (6 channels)
        img_combined = np.concatenate([before_resized, after_resized], axis=2)
        img_combined = preprocess_inputs(img_combined)
        
        # Convert to tensor
        inp = torch.from_numpy(img_combined.transpose((2, 0, 1))).float()
        inp = inp.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            pred = self.damage_model(inp)
            pred = torch.sigmoid(pred)
            pred = pred.cpu().numpy()[0]
        
        # Transpose to (H, W, C)
        pred = pred.transpose(1, 2, 0)
        
        # Debug: Print prediction statistics
        print(f"  [DEBUG] Damage map shape: {pred.shape}")
        print(f"  [DEBUG] Damage predictions per class:")
        for i, label in enumerate(['No Building', 'No Damage', 'Minor', 'Major', 'Destroyed']):
            class_mean = pred[:, :, i].mean()
            class_max = pred[:, :, i].max()
            print(f"    Class {i} ({label}): mean={class_mean:.4f}, max={class_max:.4f}")
        
        # Resize back to original size
        pred = cv2.resize(pred, (w, h))
        
        return pred
    
    def _simple_damage_detection(self, before_img, after_img):
        """Simple rule-based damage detection (fallback)"""
        diff = np.abs(after_img.astype(float) - before_img.astype(float))
        change_magnitude = diff.mean()
        
        damage_map = np.zeros((before_img.shape[0], before_img.shape[1], 5))
        
        if change_magnitude > 50:
            damage_map[:, :, 3] = 0.7  # Major damage
            damage_map[:, :, 4] = 0.3  # Destroyed
        elif change_magnitude > 30:
            damage_map[:, :, 2] = 0.8  # Minor damage
        else:
            damage_map[:, :, 1] = 0.9  # No damage
        
        return damage_map
    
    def assess(self, before_img, after_img):
        """
        Complete damage assessment
        Args:
            before_img: numpy array (H, W, 3), uint8
            after_img: numpy array (H, W, 3), uint8
        Returns:
            dict with assessment results
        """
        # Classify damage on whole image (skip building segmentation)
        damage_map = self.classify_damage(before_img, after_img)
        
        # Analyze damage across entire image
        # xView2 outputs damage probabilities for each pixel
        avg_damage = damage_map.mean(axis=(0, 1))  # Average across all pixels
        damage_class = np.argmax(avg_damage)
        
        # Count non-zero predictions as "building pixels"
        non_background = damage_map[:, :, 1:].sum(axis=2) > 0.1  # Classes 1-4
        num_buildings = np.sum(non_background)
        
        # Create simplified building mask from damage predictions
        building_mask = non_background.astype(np.uint8)[:, :, np.newaxis]
        
        # Map to decision
        damage_labels = ['No Building', 'No Damage', 'Minor Damage', 'Major Damage', 'Destroyed']
        damage_level = damage_labels[damage_class]
        
        if damage_class >= 3:
            decision = 'APPROVE'
            priority = 'HIGH'
        elif damage_class == 2:
            decision = 'MANUAL REVIEW'
            priority = 'MEDIUM'
        else:
            decision = 'REJECT'
            priority = 'LOW'
        
        confidence = float(avg_damage[damage_class]) * 100
        
        return {
            'before_image': before_img,
            'after_image': after_img,
            'building_mask': building_mask,
            'damage_map': damage_map,
            'damage_class': damage_class,
            'damage_level': damage_level,
            'damage_probabilities': avg_damage.tolist(),
            'confidence': confidence,
            'decision': decision,
            'priority': priority,
            'num_buildings': int(num_buildings)
        }


if __name__ == "__main__":
    # Quick test
    print("="*70)
    print("DAMAGE DETECTOR - QUICK TEST")
    print("="*70)
    
    detector = DamageDetector(use_gpu=torch.cuda.is_available())
    
    # Test with random images
    before = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    after = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    result = detector.assess(before, after)
    
    print(f"\nDecision: {result['decision']}")
    print(f"Damage Level: {result['damage_level']}")
    print(f"Confidence: {result['confidence']:.1f}%")
    print(f"Buildings: {result['num_buildings']} pixels")
    print("\n" + "="*70)
