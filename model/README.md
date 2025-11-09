# AlphaEarth Insurance AI - Damage Detection Model

Minimal, production-ready damage detection from satellite imagery for hackathon demo.

## Quick Test

```bash
cd model/inference
python damage_detector.py
```

## Structure

```
model/
├── README.md
├── __init__.py
├── zoo_models.py              # xView2 ResNet34 architecture
├── segment/
│   ├── building_segmentation_model.keras
│   └── utils.py
├── weights/weights/
│   └── res34_cls2_0_tuned_best  # Pre-trained damage classification
└── inference/
    ├── __init__.py
    ├── damage_detector.py     # Main detector class
    └── run_assessment.py      # CLI tool
```

## Usage

### CLI Tool

```bash
cd model/inference
python run_assessment.py \
  --before ../maxar_pre.png \
  --after ../maxar_buildings.png \
  --name "test_assessment" \
  --output results
```

**Arguments:**
- `--before` - Path to before-event image
- `--after` - Path to after-event image  
- `--name` - Output file prefix
- `--output` - Output directory (default: `results`)
- `--gpu` - Use GPU acceleration

### Python API

```python
import sys
sys.path.append('model/inference')

from damage_detector import DamageDetector
import cv2

# Initialize
detector = DamageDetector(use_gpu=True)

# Load images (RGB, uint8)
before = cv2.imread('before.png')
after = cv2.imread('after.png')
before = cv2.cvtColor(before, cv2.COLOR_BGR2RGB)
after = cv2.cvtColor(after, cv2.COLOR_BGR2RGB)

# Assess damage
result = detector.assess(before, after)

# Results
print(f"Decision: {result['decision']}")       # APPROVE/REJECT/MANUAL REVIEW
print(f"Damage: {result['damage_level']}")     # No Damage/Minor/Major/Destroyed
print(f"Confidence: {result['confidence']}%")  # 0-100
print(f"Buildings: {result['num_buildings']}")
```

## Output Files

When using `run_assessment.py`, outputs are saved to the specified directory:

- `{name}_{timestamp}_before.png` - Before image
- `{name}_{timestamp}_after.png` - After image
- `{name}_{timestamp}_buildings.png` - Building mask (white = building)
- `{name}_{timestamp}_damage.png` - Damage heatmap (color-coded by severity)
- `{name}_{timestamp}_report.json` - Full assessment report

### JSON Report Format

```json
{
  "assessment": {
    "damage_level": "Major Damage",
    "damage_class": 3,
    "confidence": 85.3,
    "decision": "APPROVE",
    "priority": "HIGH"
  },
  "metrics": {
    "num_buildings": 45123,
    "damage_probabilities": {
      "no_building": 0.01,
      "no_damage": 0.05,
      "minor_damage": 0.09,
      "major_damage": 0.70,
      "destroyed": 0.15
    }
  },
  "timestamp": "20251109_143022"
}
```

## Models

### Building Segmentation (Optional)
- **File:** `segment/building_segmentation_model.keras`
- **Type:** Keras U-Net
- **Input:** RGB image (any size, processed in 256x256 patches)
- **Output:** Binary mask (building/non-building)
- **Note:** If model fails to load, system uses fallback (no segmentation)

### Damage Classification
- **File:** `weights/weights/res34_cls2_0_tuned_best`
- **Type:** PyTorch ResNet34 U-Net (xView2 first place solution)
- **Input:** 6-channel (before+after RGB), resized to 1024x1024
- **Output:** 5-channel probability map
  - Channel 0: No Building
  - Channel 1: No Damage
  - Channel 2: Minor Damage
  - Channel 3: Major Damage
  - Channel 4: Destroyed

## Decision Logic

```python
if damage_class >= 3:  # Major or Destroyed
    decision = 'APPROVE'
    priority = 'HIGH'
elif damage_class == 2:  # Minor
    decision = 'MANUAL REVIEW'
    priority = 'MEDIUM'
else:  # No damage or no building
    decision = 'REJECT'
    priority = 'LOW'
```

## Requirements

```
torch>=2.0
torchvision>=0.15
opencv-python
numpy
keras (optional - for building segmentation)
```

## GPU Support

To use GPU:
```bash
python run_assessment.py --gpu --before X --after Y
```

Requires CUDA-compatible GPU and PyTorch with CUDA support.

## Notes

- **Minimal Design:** Single ResNet34 model, no ensemble
- **Self-Contained:** All code in `model/` folder
- **Fallback:** Works even if building segmentation fails to load
- **Production Ready:** Error handling, structured output, clean code
