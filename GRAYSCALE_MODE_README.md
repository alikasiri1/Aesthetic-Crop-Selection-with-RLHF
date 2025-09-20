# Grayscale Mode for Aesthetic Crop Selection

This document explains how to use the new grayscale mode functionality in the Aesthetic Crop Selection with RLHF system.

## Overview

The system now supports processing images in grayscale mode instead of RGB. When grayscale mode is enabled:

- Images are converted to grayscale before processing
- The neural network architecture adapts to handle single-channel input
- The aesthetic scorer processes grayscale images
- All observations are single-channel instead of 3-channel

## Key Changes

### 1. Command Line Interface
Added `--gray_mode` flag to enable grayscale processing:

```bash
# RGB mode (default)
python actor/train_ppo.py --image image.jpg --timesteps 20000

# Grayscale mode
python actor/train_ppo.py --image image.jpg --timesteps 20000 --gray_mode
```

### 2. Neural Network Architecture
- **CustomCNN**: Automatically adapts input channels (1 for grayscale, 3 for RGB)
- **AestheticScorer**: Modifies first convolutional layer to accept single-channel input
- **Weight Initialization**: RGB weights are averaged to initialize grayscale weights

### 3. Image Processing Pipeline
- **FrameSelectorEnv**: Converts images to grayscale when `gray_mode=True`
- **Observation Space**: Changes from `(H, W, 3)` to `(H, W, 1)` for grayscale
- **AestheticScorerPipeline**: Uses appropriate normalization for grayscale images

## Usage Examples

### Training with Grayscale Mode

```python
# Direct training
python actor/train_ppo.py \
    --image path/to/image.jpg \
    --timesteps 20000 \
    --save_dir checkpoints_gray \
    --gray_mode

# Using the helper script
python run_gray_training.py
```

### Testing Grayscale Functionality

```python
# Test both RGB and grayscale modes
python test_gray_mode.py
```

### Programmatic Usage

```python
from rl_env.frame_selector_env import FrameSelectorEnv
from scorer.aesthetic_scorer import AestheticScorerPipeline

# Grayscale mode
scorer = AestheticScorerPipeline(gray_mode=True)
env = FrameSelectorEnv(
    image_path="image.jpg",
    scorer=scorer,
    gray_mode=True
)

obs, info = env.reset()
print(f"Observation shape: {obs.shape}")  # (H, W, 1) for grayscale
```

## Technical Details

### Image Conversion
- Input images are converted from BGR to grayscale using `cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)`
- For compatibility with existing code, grayscale images are converted back to 3-channel BGR format internally
- Observations are properly formatted as single-channel arrays

### Neural Network Modifications
- **EfficientNet**: First conv layer (`conv_stem`) modified to accept 1 input channel
- **ResNet**: First conv layer (`conv1`) modified to accept 1 input channel
- Weights are initialized by averaging the RGB channel weights

### Normalization
- **RGB mode**: Uses ImageNet normalization `mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`
- **Grayscale mode**: Uses simple normalization `mean=[0.5], std=[0.5]`

## Performance Considerations

### Memory Usage
- Grayscale mode uses ~1/3 less memory for image data
- Neural network input size is reduced from 3 channels to 1 channel

### Training Speed
- Slightly faster due to reduced input size
- May require different learning rates or training schedules

### Model Compatibility
- Models trained in RGB mode cannot be directly used in grayscale mode
- Models trained in grayscale mode cannot be directly used in RGB mode
- Separate model checkpoints should be saved for each mode

## File Structure

```
Aesthetic-Crop-Selection-with-RLHF/
├── actor/
│   ├── train_ppo.py          # Updated with --gray_mode flag
│   └── gym_wrapper.py        # Updated observation space
├── rl_env/
│   └── frame_selector_env.py # Updated image processing
├── scorer/
│   └── aesthetic_scorer.py   # Updated for grayscale support
├── test_gray_mode.py         # Test script
├── run_gray_training.py      # Training helper script
└── GRAYSCALE_MODE_README.md  # This file
```

## Troubleshooting

### Common Issues

1. **Shape Mismatch Errors**
   - Ensure `gray_mode` parameter is consistent across all components
   - Check that observation space matches the expected input channels

2. **Model Loading Errors**
   - RGB and grayscale models are not compatible
   - Train separate models for each mode

3. **Image Processing Errors**
   - Ensure input images are valid and readable
   - Check that OpenCV can properly load the image

### Debug Tips

1. Use the test script to verify functionality:
   ```bash
   python test_gray_mode.py
   ```

2. Check observation shapes:
   ```python
   print(f"RGB obs shape: {obs_rgb.shape}")      # Should be (H, W, 3)
   print(f"Gray obs shape: {obs_gray.shape}")    # Should be (H, W, 1)
   ```

3. Verify image conversion:
   ```python
   # Check if image is properly converted to grayscale
   print(f"Image shape: {image.shape}")
   print(f"Image dtype: {image.dtype}")
   ```

## Future Enhancements

- Support for other color spaces (HSV, LAB, etc.)
- Automatic mode detection based on input image
- Mixed-mode training (both RGB and grayscale)
- Advanced grayscale-specific augmentation techniques
