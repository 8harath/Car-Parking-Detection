# 🚗 Advanced Car Parking Space Detection System

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-green.svg)](https://opencv.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Latest-orange.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**An intelligent parking space detection system powered by computer vision and deep learning**

[Features](#-features) •
[Quick Start](#-quick-start) •
[Documentation](#-documentation) •
[Demo](#-demo) •
[Contributing](#-contributing)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Demo](#-demo)
- [Quick Start](#-quick-start)
- [Detailed Setup](#-detailed-setup)
- [Usage](#-usage)
- [Configuration](#%EF%B8%8F-configuration)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)
- [Contact](#-contact)

---

## 🎯 Overview

This project provides a **production-ready** parking space detection system that combines classical computer vision techniques with state-of-the-art deep learning (YOLOv8) to analyze parking lots in real-time.

**Perfect for:**
- Smart city applications
- Parking lot management systems
- Real-time parking guidance
- Parking analytics and reporting
- Research and educational purposes

### Key Capabilities

✅ **Real-time Detection** - Process live video feeds or static images
✅ **ML-Powered** - YOLOv8 for accurate vehicle detection
✅ **Interactive Setup** - Drag-and-drop parking space selection
✅ **Comprehensive Reports** - Visual, text, and CSV outputs
✅ **Special Parking Detection** - Identifies disabled parking spaces
✅ **Historical Tracking** - CSV data logging for analytics
✅ **Production Ready** - Robust error handling and logging

---

## 🌟 Features

### 1. **Interactive Parking Space Selection**
- 🖱️ Drag-and-select multiple parking spaces at once
- 📐 Grid-based automatic space allocation
- 👁️ Visual feedback during selection
- ↩️ Undo/Redo functionality
- 🗑️ Right-click to remove individual spaces
- 🔍 Zoom and pan capabilities

### 2. **Intelligent Vehicle Detection**
- 🤖 YOLOv8-based deep learning model
- 🚙 Detects: Cars, Motorcycles, Buses, Trucks
- 📊 Confidence scores for each detection
- 🎯 Real-time detection capabilities
- 👀 Handles occlusions and partial visibility
- ⚡ GPU acceleration support

### 3. **Comprehensive Reporting**
- 📈 **Visual Reports**: Multi-panel analytics dashboard
- 📝 **Text Reports**: Detailed statistics and metrics
- 💾 **CSV Export**: Historical data for analysis
- 🎨 **Color-coded Display**: Green (empty) / Red (occupied)
- ⏱️ Real-time statistics display
- 📊 Occupancy rate tracking

### 4. **Advanced Features**
- ♿ Special parking space detection (yellow markings)
- 🔄 Video loop playback
- 📍 Precise coordinate tracking
- 🛡️ Robust error handling
- 📝 Comprehensive logging
- ⚙️ Centralized configuration

---

## 🎬 Demo

### Visual Output Examples

The system generates comprehensive reports with:

1. **Parking Lot Visualization** - Color-coded parking spaces with occupancy status
2. **Statistics Dashboard** - Bar charts showing total, occupied, and available spaces
3. **Space Type Distribution** - Pie charts for regular vs. special parking
4. **Occupancy Analysis** - Detailed breakdown by parking type

*Reports are saved in `reports/` directory with timestamp*

---

## 🚀 Quick Start

Get up and running in under 5 minutes!

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster processing

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/8harath/Car-Parking-Detection.git
cd Car-Parking-Detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the application
python run.py --image carParkImg.png
```

### First Run

1. **Select Parking Spaces**: Click and drag to mark parking areas
2. **Save Layout**: Press `S` to save your parking space configuration
3. **Detect Vehicles**: Press `D` to run detection and generate reports
4. **View Results**: Check the `reports/` directory for outputs

**Keyboard Shortcuts:**
- `D` - Detect vehicles & generate reports
- `S` - Save parking layout
- `R` - Reset all selections
- `Z` - Undo last selection
- `Q` - Quit application

---

## 🛠️ Detailed Setup

### 1. System Requirements

**Minimum:**
- Python 3.8+
- 4GB RAM
- 2GB free disk space

**Recommended:**
- Python 3.10+
- 8GB RAM
- NVIDIA GPU with CUDA support
- 5GB free disk space

### 2. Installation Steps

#### Option A: Using pip (Recommended)

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Unix or MacOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Option B: Using conda

```bash
# Create conda environment
conda create -n parking-detection python=3.10
conda activate parking-detection

# Install dependencies
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
# Test imports
python -c "import cv2, numpy, pandas, ultralytics; print('✓ All dependencies installed')"

# Check GPU availability (optional)
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

---

## 💻 Usage

### Basic Usage

#### Process an Image

```bash
python run.py --image carParkImg.png
```

#### Process a Video

```bash
python run.py --video carPark.mp4
```

### Advanced Usage

#### With Custom Configuration

```python
from enhanced_parking_detector import EnhancedParkingDetector

# Initialize detector
detector = EnhancedParkingDetector(
    image_path='path/to/image.png',
    video_path='path/to/video.mp4'
)

# Process image
detector.process_image()

# Or process video
detector.process_video()
```

#### Custom Thresholds

Edit `config.py`:

```python
# Adjust detection sensitivity
OCCUPANCY_THRESHOLD = 900  # Lower = more sensitive
CONFIDENCE_THRESHOLD = 0.5  # YOLOv8 confidence (0-1)

# Adjust parking space dimensions
PARKING_WIDTH = 107
PARKING_HEIGHT = 48
```

---

## ⚙️ Configuration

All configuration is centralized in `config.py`:

### Key Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PARKING_WIDTH` | 107 | Parking space width in pixels |
| `PARKING_HEIGHT` | 48 | Parking space height in pixels |
| `OCCUPANCY_THRESHOLD` | 900 | Pixel count threshold for occupancy |
| `CONFIDENCE_THRESHOLD` | 0.5 | YOLOv8 detection confidence |
| `MODEL_PATH` | 'yolov8n.pt' | Path to YOLO model |
| `CSV_APPEND_MODE` | True | Append to CSV vs. overwrite |

### Directory Structure

The system automatically creates:

```
reports/       # Generated reports
data/          # CSV data files
models/        # ML models
```

---

## 📁 Project Structure

```
Car-Parking-Detection/
├── config.py                      # Centralized configuration
├── enhanced_parking_detector.py   # Main detection logic
├── car_detector.py                # YOLO-based vehicle detection
├── main.py                        # Video processing entry point
├── run.py                         # CLI entry point
├── setup_directories.py           # Directory initialization
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── TROUBLESHOOTING.md            # Common issues and solutions
├── LICENSE                        # MIT License
├── .gitignore                    # Git ignore rules
├── carParkImg.png                # Sample parking lot image
├── carPark.mp4                   # Sample parking lot video
├── CarParkPos                    # Saved parking positions
├── reports/                      # Generated reports
│   ├── parking_report_YYYYMMDD_HHMMSS.png
│   └── parking_report_YYYYMMDD_HHMMSS.txt
└── data/                         # CSV data files
    └── parking_status.csv
```

---

## 🔍 How It Works

### Architecture Overview

```
┌─────────────────┐
│   Input         │
│ (Image/Video)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Space Selection │
│   (Manual UI)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Image Processing│
│  (CV Pipeline)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Vehicle Detection│
│    (YOLOv8)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Space Occupancy  │
│   Analysis      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Report Generation│
│ (Visual/CSV)    │
└─────────────────┘
```

### Detection Pipeline

1. **Pre-processing**
   - Grayscale conversion
   - Gaussian blur (noise reduction)
   - Adaptive thresholding
   - Median blur
   - Morphological operations (dilation)

2. **Vehicle Detection**
   - YOLOv8 inference on full image
   - Filter for vehicle classes (car, truck, bus, motorcycle)
   - Apply confidence threshold
   - Match detections to parking spaces

3. **Occupancy Analysis**
   - Extract each parking space region
   - Count non-zero pixels
   - Compare against threshold
   - Mark as occupied/empty

4. **Special Parking Detection**
   - Convert to HSV color space
   - Detect yellow markings
   - Identify horizontal/vertical lines
   - Mark as special parking space

---

## 🐛 Troubleshooting

### Common Issues

#### "ModuleNotFoundError: No module named 'cv2'"

```bash
pip install opencv-python
```

#### "YOLOv8 model not found"

The model will download automatically on first run. Ensure internet connection.

#### "CUDA out of memory"

```python
# In config.py, use CPU instead of GPU
# Or reduce image resolution
```

#### "No parking spaces detected"

Press `R` to reset and redraw parking spaces. Make sure to press `S` to save.

For more issues, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. 🐛 **Report Bugs**: Open an issue with details
2. 💡 **Suggest Features**: Share your ideas
3. 🔧 **Submit PRs**: Fork, create branch, submit PR
4. 📖 **Improve Docs**: Help us improve documentation
5. ⭐ **Star the Repo**: Show your support!

### Development Setup

```bash
# Fork and clone your fork
git clone https://github.com/YOUR_USERNAME/Car-Parking-Detection.git

# Create feature branch
git checkout -b feature/amazing-feature

# Make changes and commit
git commit -m "Add amazing feature"

# Push and create PR
git push origin feature/amazing-feature
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

### Project Attribution
This project was implemented by **Bharath K** from **Jain Deemed to be University** as part of his project submission for the PNT Lab selection process conducted by IIT Tirupati Navishkar.

### Credits
- **Original Inspiration**: [Murtaza's Computer Vision Zone](https://www.computervision.zone/)
- **YOLOv8 Model**: [Ultralytics](https://github.com/ultralytics/ultralytics)
- **Libraries**: OpenCV, PyTorch, Matplotlib, Pandas, NumPy

### Special Thanks
- 🏛️ **IIT Tirupati Navishkar** - For the selection process opportunity
- 🎓 **Jain Deemed to be University** - For academic support
- 👥 **Open Source Community** - For amazing tools and libraries

---

## 📞 Contact

**Bharath K**
📧 Email: 8harath.k@gmail.com
🌐 Website: [8harath.me](https://8harath.me)
💻 GitHub: [@8harath](https://github.com/8harath)

---

## 🌟 Star History

If you find this project useful, please consider giving it a ⭐!

---

## 📊 Project Status

- ✅ Core Features: Complete
- ✅ Documentation: Comprehensive
- ✅ Testing: In Progress
- 🔄 Continuous Improvement

---

<div align="center">

**[⬆ Back to Top](#-advanced-car-parking-space-detection-system)**

Made with ❤️ by Bharath K

</div>
