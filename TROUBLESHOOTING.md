# 🔧 Troubleshooting Guide

This guide helps you resolve common issues with the Car Parking Detection System.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Runtime Errors](#runtime-errors)
- [Performance Issues](#performance-issues)
- [Detection Issues](#detection-issues)
- [Configuration Issues](#configuration-issues)
- [Platform-Specific Issues](#platform-specific-issues)
- [Getting Help](#getting-help)

---

## Installation Issues

### Problem: `ModuleNotFoundError: No module named 'cv2'`

**Solution:**
```bash
pip install opencv-python
# Or for full features:
pip install opencv-contrib-python
```

### Problem: `ModuleNotFoundError: No module named 'ultralytics'`

**Solution:**
```bash
pip install ultralytics
```

### Problem: Torch/CUDA Installation Fails

**Symptoms:**
- `RuntimeError: Couldn't install torch`
- CUDA version mismatch errors

**Solution:**
```bash
# For CPU-only (no GPU):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Problem: `pip` Version Too Old

**Solution:**
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Runtime Errors

### Problem: "Video file not found" or "Image file not found"

**Symptoms:**
```
❌ Error: Image not found at 'carParkImg.png'
```

**Solutions:**
1. Check file path is correct:
```bash
ls -la carParkImg.png
```

2. Use absolute path:
```bash
python run.py --image /full/path/to/carParkImg.png
```

3. Verify file permissions:
```bash
chmod 644 carParkImg.png
```

### Problem: "Failed to load YOLO model"

**Symptoms:**
```
IOError: Failed to load YOLO model from yolov8n.pt
```

**Solutions:**
1. Delete existing model and re-download:
```bash
rm yolov8n.pt
python run.py --image carParkImg.png
# Model will auto-download
```

2. Manually download model:
```bash
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

3. Check internet connection (required for first run)

### Problem: "Cannot open video file"

**Symptoms:**
```
IOError: Cannot open video file: carPark.mp4
```

**Solutions:**
1. Verify video codec:
```bash
ffmpeg -i carPark.mp4
```

2. Convert to compatible format:
```bash
ffmpeg -i input.mp4 -codec copy output.mp4
```

3. Try different video:
```bash
python run.py --video path/to/different/video.mp4
```

---

## Performance Issues

### Problem: Slow Processing / Low FPS

**Symptoms:**
- Processing takes very long
- Video playback is laggy
- High CPU usage

**Solutions:**

1. **Enable GPU acceleration:**
```python
# Verify CUDA is available
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

2. **Reduce image resolution:**
Edit `config.py`:
```python
# Add image resizing
MAX_IMAGE_WIDTH = 1280
MAX_IMAGE_HEIGHT = 720
```

3. **Use smaller YOLO model:**
Edit `config.py`:
```python
MODEL_PATH = 'yolov8n.pt'  # Fastest (current)
# vs
MODEL_PATH = 'yolov8s.pt'  # Slower but more accurate
```

4. **Reduce frame processing:**
For videos, process every Nth frame:
```python
# In process_video(), add:
frame_skip = 3  # Process every 3rd frame
```

### Problem: High Memory Usage

**Symptoms:**
```
RuntimeError: CUDA out of memory
MemoryError: Unable to allocate array
```

**Solutions:**

1. **Use CPU instead of GPU:**
```bash
export CUDA_VISIBLE_DEVICES=""
python run.py --image carParkImg.png
```

2. **Reduce batch size:**
Edit `car_detector.py`:
```python
results = self.model(image, conf=self.confidence_threshold, batch=1)
```

3. **Close other applications**

4. **Restart Python:**
```bash
# Clear memory and restart
python run.py --image carParkImg.png
```

---

## Detection Issues

### Problem: No Parking Spaces Detected

**Symptoms:**
- Warning: "Please select parking spaces first!"
- Empty parking lot display

**Solutions:**

1. **Select parking spaces:**
   - Click and drag to mark parking areas
   - Press `S` to save layout
   - Press `D` to detect

2. **Check saved positions:**
```bash
ls -la CarParkPos
# Should exist and have size > 0
```

3. **Reset and redraw:**
   - Press `R` to reset
   - Carefully redraw parking spaces
   - Press `S` to save

### Problem: Inaccurate Vehicle Detection

**Symptoms:**
- Occupied spaces marked as empty
- Empty spaces marked as occupied
- Inconsistent detection

**Solutions:**

1. **Adjust occupancy threshold:**
Edit `config.py`:
```python
OCCUPANCY_THRESHOLD = 900  # Default
# Try lower value for more sensitivity:
OCCUPANCY_THRESHOLD = 700  # More sensitive
# Try higher value for less sensitivity:
OCCUPANCY_THRESHOLD = 1100  # Less sensitive
```

2. **Adjust YOLO confidence:**
Edit `config.py`:
```python
CONFIDENCE_THRESHOLD = 0.5  # Default
# Try lower for more detections:
CONFIDENCE_THRESHOLD = 0.3  # More detections
# Try higher for fewer false positives:
CONFIDENCE_THRESHOLD = 0.7  # Fewer detections
```

3. **Check parking space dimensions:**
Edit `config.py`:
```python
PARKING_WIDTH = 107  # Adjust to match your lot
PARKING_HEIGHT = 48  # Adjust to match your lot
```

### Problem: Special Parking Not Detected

**Symptoms:**
- Yellow-marked disabled parking not recognized
- All spaces shown as regular

**Solutions:**

1. **Verify yellow markings are visible:**
   - Check image quality
   - Ensure yellow lines are clear
   - Good lighting conditions

2. **Adjust yellow detection threshold:**
Edit `car_detector.py`:
```python
# Line ~60
return yellow_percentage > 3  # Default
# Try:
return yellow_percentage > 1.5  # More sensitive
```

---

## Configuration Issues

### Problem: Reports Not Generated

**Symptoms:**
- No files in `reports/` directory
- CSV not created

**Solutions:**

1. **Check directories exist:**
```bash
ls -la reports/ data/
# Should see directories
```

2. **Run setup:**
```python
python setup_directories.py
```

3. **Check permissions:**
```bash
chmod 755 reports/ data/
```

4. **Check disk space:**
```bash
df -h .
```

### Problem: Configuration Changes Not Applied

**Symptoms:**
- Changes to `config.py` have no effect
- Old values still being used

**Solutions:**

1. **Remove Python cache:**
```bash
rm -rf __pycache__/
rm -f *.pyc
```

2. **Restart application:**
```bash
# Exit completely and restart
python run.py --image carParkImg.png
```

3. **Verify config syntax:**
```python
python -m py_compile config.py
```

---

## Platform-Specific Issues

### Windows

#### Problem: "python: command not found"

**Solution:**
```cmd
# Use py instead of python
py run.py --image carParkImg.png

# Or use python3
python3 run.py --image carParkImg.png
```

#### Problem: Path Issues with Backslashes

**Solution:**
```python
# Use forward slashes or raw strings
image_path = "C:/path/to/image.png"
# Or
image_path = r"C:\path\to\image.png"
```

### macOS

#### Problem: "zsh: command not found: python"

**Solution:**
```bash
# Use python3
python3 run.py --image carParkImg.png

# Or create alias
alias python=python3
```

#### Problem: GUI Window Not Appearing

**Solution:**
```bash
# Install PyQt5
pip install PyQt5

# Or use different backend
export MPLBACKEND=TkAgg
python run.py --image carParkImg.png
```

### Linux

#### Problem: "ImportError: libGL.so.1"

**Solution:**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install libgl1-mesa-glx

# Fedora/RHEL
sudo dnf install mesa-libGL

# Arch
sudo pacman -S mesa
```

#### Problem: "Qt platform plugin" Error

**Solution:**
```bash
# Install Qt dependencies
sudo apt-get install libxcb-xinerama0

# Or use headless mode
export QT_QPA_PLATFORM=offscreen
python run.py --image carParkImg.png
```

---

## Getting Help

If you still have issues after trying these solutions:

### 1. Check Logs

```bash
# View application log
cat parking_detection.log

# View Python error details
python run.py --verbose --image carParkImg.png
```

### 2. Gather Information

When asking for help, include:
- Operating System (Windows/macOS/Linux)
- Python version: `python --version`
- Package versions: `pip list`
- Error message (full traceback)
- Steps to reproduce

### 3. Search Existing Issues

Check if someone else had the same problem:
- https://github.com/8harath/Car-Parking-Detection/issues

### 4. Create New Issue

If your problem is new:
1. Go to: https://github.com/8harath/Car-Parking-Detection/issues/new
2. Use the template
3. Provide all information from step 2
4. Include screenshots if applicable

### 5. Contact Developer

For urgent issues:
- Email: 8harath.k@gmail.com
- Include "[Parking Detection]" in subject line

---

## Quick Diagnostic Script

Run this to check your setup:

```python
# diagnostic.py
import sys

print(f"Python: {sys.version}")

try:
    import cv2
    print(f"✓ OpenCV: {cv2.__version__}")
except ImportError as e:
    print(f"✗ OpenCV: {e}")

try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"✗ PyTorch: {e}")

try:
    from ultralytics import YOLO
    print(f"✓ Ultralytics: Installed")
except ImportError as e:
    print(f"✗ Ultralytics: {e}")

try:
    import numpy, pandas, matplotlib, seaborn
    print(f"✓ Data libraries: All installed")
except ImportError as e:
    print(f"✗ Data libraries: {e}")

print("\nIf all checks show ✓, your environment is ready!")
```

Run it:
```bash
python diagnostic.py
```

---

## Common Error Codes

| Code | Meaning | Solution |
|------|---------|----------|
| Exit 0 | Success | - |
| Exit 1 | File not found | Check file paths |
| Exit 1 | Permission denied | Check file permissions |
| Exit 1 | IO Error | Check disk space |

---

**Still stuck? Don't hesitate to ask for help!**

[⬆ Back to README](README.md)
