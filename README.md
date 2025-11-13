# 🚗 Advanced Car Parking Space Detection System

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-green.svg)](https://opencv.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Latest-orange.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**An intelligent parking space detection system powered by computer vision and deep learning**

[Features](#-features) •
[Quick Start](#-quick-start) •
[For Students](#-for-students--academic-use) •
[How It Works](#-how-it-works) •
[Contributing](#-contributing)

</div>

---

## 🎓 What Does This Do? (In Simple Terms)

Imagine you're managing a parking lot and need to know:
- ✅ How many parking spaces are empty right now?
- ✅ Which specific spots are available?
- ✅ How full is the parking lot over time?

**This system does exactly that - automatically!**

You provide an image or video of your parking lot, and the system:
1. Lets you mark where the parking spaces are (just click and drag)
2. Uses AI to detect cars in those spaces
3. Shows you which spaces are empty (green) and occupied (red)
4. Generates detailed reports with charts and statistics

**No manual counting needed!** The computer does all the work.

### Real-World Example

```
Input:  Picture of parking lot with 50 spaces
Output: "35 spaces occupied, 15 available"
        + Visual map showing which spots are free
        + Detailed report with charts
        + CSV file for data analysis
```

---

## 📋 Table of Contents

- [What Does This Do?](#-what-does-this-do-in-simple-terms)
- [For Students & Academic Use](#-for-students--academic-use)
- [Overview](#-overview)
- [Features](#-features)
- [Quick Start](#-quick-start)
- [Step-by-Step Tutorial](#-step-by-step-tutorial)
- [Detailed Setup](#-detailed-setup)
- [Usage](#-usage)
- [Configuration](#%EF%B8%8F-configuration)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🎓 For Students & Academic Use

### Why This Project is Perfect for Learning

This project is **excellent for academic purposes** because it demonstrates:

**Computer Vision Concepts:**
- Image processing (grayscale, blurring, thresholding)
- Object detection using deep learning (YOLOv8)
- Real-time video processing
- Color space transformations (RGB to HSV)
- Morphological operations

**Software Engineering Skills:**
- Clean code architecture
- Configuration management
- Error handling and logging
- File I/O operations
- CLI application development

**Data Science & Analytics:**
- Data collection and storage (CSV)
- Statistical analysis
- Data visualization (charts, graphs)
- Report generation

### Academic Projects You Can Build

✅ **Final Year Project** - Complete system for smart parking
✅ **Computer Vision Assignment** - Object detection implementation
✅ **Machine Learning Project** - YOLO model application
✅ **IoT Project** - Combine with sensors for real-time monitoring
✅ **Data Analysis Project** - Analyze parking patterns over time
✅ **Research Paper** - Study parking occupancy patterns

### Learning Outcomes

After studying this project, you'll understand:
1. How to use OpenCV for image processing
2. How to apply YOLOv8 for object detection
3. How to build interactive GUI applications
4. How to generate professional reports
5. How to structure a real-world Python project
6. How to document code professionally

### Recommended Study Path

```
Week 1: Understand the basics (image processing, OpenCV)
Week 2: Study YOLO and object detection
Week 3: Run the project and experiment with parameters
Week 4: Modify and add your own features
Week 5: Create a presentation/report on your learnings
```

### Citations & Academic Use

If you use this project in academic work, please cite:

```
Bharath K. (2024). Advanced Car Parking Space Detection System.
GitHub repository. https://github.com/8harath/Car-Parking-Detection
```

**Note:** This project was created for the PNT Lab selection process at IIT Tirupati Navishkar by Bharath K from Jain Deemed to be University.

---

## 🎯 Overview

This project provides a **production-ready** parking space detection system that combines classical computer vision techniques with state-of-the-art deep learning (YOLOv8) to analyze parking lots in real-time.

**Perfect for:**
- Academic projects and research
- Smart city applications
- Parking lot management systems
- Real-time parking guidance
- Parking analytics and reporting
- Learning computer vision and ML

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

## 📖 Step-by-Step Tutorial

### Complete Beginner's Guide

This tutorial assumes you've never used the system before. Follow these steps exactly:

#### Step 1: Installation (5 minutes)

```bash
# Open your terminal/command prompt

# Navigate to where you want to save the project
cd Desktop  # or any folder you prefer

# Download the project
git clone https://github.com/8harath/Car-Parking-Detection.git

# Enter the project folder
cd Car-Parking-Detection

# Install required libraries
pip install -r requirements.txt
```

**What just happened?** You downloaded the code and installed all the Python libraries it needs.

#### Step 2: Run the Application (1 minute)

```bash
# Start the program
python run.py --image carParkImg.png
```

**What you'll see:** A window will open showing a parking lot image.

#### Step 3: Mark Parking Spaces (3 minutes)

This is the most important step! You need to tell the computer where the parking spaces are.

**How to mark spaces:**

1. **Click and hold** your left mouse button at the **top-left** corner of a parking space
2. **Drag** to the **bottom-right** corner of that space
3. **Release** the mouse button
4. You'll see a grid of purple rectangles appear - these are your marked spaces!

**Tips:**
- You can mark multiple spaces at once by dragging over them
- If you make a mistake, **right-click** on a space to remove it
- Press `Z` to undo your last selection
- Press `R` to start over completely

**Practice:** Try marking 5-10 spaces to get the hang of it.

#### Step 4: Save Your Layout (10 seconds)

```
Press 'S' on your keyboard
```

**What happened?** The computer saved your parking space positions. You won't have to mark them again!

#### Step 5: Detect Vehicles (30 seconds)

```
Press 'D' on your keyboard
```

**What happens:**
1. The screen will show: "🚗 Detecting vehicles..."
2. The AI will analyze each parking space
3. Green spaces = Empty
4. Red spaces = Occupied
5. Reports are generated automatically

#### Step 6: View Your Results (2 minutes)

The program creates several files for you:

**1. Visual Report** (`reports/parking_report_YYYYMMDD_HHMMSS.png`)
   - Open this image to see charts and statistics
   - Shows a colorful dashboard with graphs

**2. Text Report** (`reports/parking_report_YYYYMMDD_HHMMSS.txt`)
   - Open with any text editor
   - Contains detailed numbers and percentages

**3. CSV Data** (`data/parking_status.csv`)
   - Open with Excel or Google Sheets
   - Great for tracking over time

#### Step 7: Understanding the Output

When you look at the visual report, you'll see:

```
┌─────────────────────────────────────────┐
│  Parking Lot Image (with colors)        │
│  Green = Empty  |  Red = Occupied       │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│  Bar Chart: Parking Statistics          │
│  Shows total, occupied, available       │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│  Pie Charts: Space Distribution         │
│  Regular vs. Special parking            │
└─────────────────────────────────────────┘
```

The text report tells you:
- Total number of parking spaces
- How many are occupied
- How many are available
- Occupancy percentage
- Timestamp of detection

### Common First-Time Questions

**Q: "Do I need to mark spaces every time I run it?"**
A: No! Once you press `S` to save, the spaces are remembered. Next time just press `D` to detect.

**Q: "Can I use my own parking lot image?"**
A: Yes! Just use: `python run.py --image path/to/your/image.jpg`

**Q: "What if I mark the wrong spot?"**
A: Right-click on it to remove, or press `Z` to undo, or press `R` to reset everything.

**Q: "How accurate is the detection?"**
A: The AI (YOLOv8) is about 90-95% accurate. You can adjust sensitivity in `config.py`.

**Q: "Can this work with a video?"**
A: Yes! Use: `python run.py --video your_video.mp4`

**Q: "Where are my reports saved?"**
A: In the `reports/` folder inside the project directory.

### Next Steps After Tutorial

Once you're comfortable with the basics, try:

1. **Use your own images** - Take a photo of any parking lot
2. **Adjust thresholds** - Edit `config.py` to fine-tune detection
3. **Try video mode** - Process a video of a parking lot over time
4. **Analyze patterns** - Open the CSV in Excel to see trends
5. **Modify the code** - Add your own features!

### Video Tutorial

*Coming soon: YouTube video walkthrough*

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
