# Advanced Car Parking Space Detection System

## Project Attribution
This project was implemented by **Bharath K** from **Jain Deemed to be University** as part of his project submission. This project serves as a criterion for his selection process for PNT Lab, which is conducted by IIT Tirupati Navishkar.

## Overview
This project provides an intelligent parking space detection system that combines computer vision and machine learning to analyze parking lots. It can process both images and video feeds to detect parking spaces, identify vehicles, and generate comprehensive reports about parking lot utilization.

## How It Works

### 1. Parking Space Selection Process
1. **Initial Setup**
   - Load the parking lot image/video
   - Create a window with interactive controls
   - Initialize the YOLO model for vehicle detection

2. **Space Selection**
   - Use left-click and drag to select parking spaces
   - The system automatically creates a grid of parking spaces
   - Each space is saved with its coordinates
   - Right-click to remove incorrectly placed spaces
   - Use 'Z' to undo last selection
   - Use 'R' to reset all selections

3. **Space Management**
   - Spaces are saved in `CarParkPos` file
   - Each space has predefined dimensions (120x43 pixels)
   - Spaces can be modified or removed at any time
   - Layout can be saved using 'S' key

### 2. Vehicle Detection Process
1. **Image Processing Pipeline**
   - Convert image to grayscale
   - Apply Gaussian blur for noise reduction
   - Use adaptive thresholding for better contrast
   - Apply median blur to remove remaining noise
   - Dilate image to connect components

2. **ML-Based Detection**
   - Use YOLOv8 model for vehicle detection
   - Process each frame/image
   - Detect vehicles with confidence scores
   - Classify vehicle types (car, truck, bus, motorcycle)
   - Handle occlusions and partial visibility

3. **Space Occupancy Analysis**
   - For each parking space:
     - Extract the region
     - Count non-zero pixels
     - Compare against threshold (900)
     - Mark as occupied or empty
   - Update status in real-time

### 3. Reporting System
1. **Real-time Display**
   - Show parking lot with color-coded spaces
   - Green: Empty spaces
   - Red: Occupied spaces
   - Display current statistics

2. **Report Generation**
   - Press 'D' to generate reports
   - Create visual report with:
     - Parking lot visualization
     - Statistics charts
     - Vehicle distribution
   - Generate text report with detailed statistics
   - Update CSV file with current status

### 4. Technical Implementation
1. **Core Components**
   - `enhanced_parking_detector.py`: Main implementation
   - `car_detector.py`: Vehicle detection logic
   - `main.py`: Application entry point

2. **Dependencies**
   - OpenCV for image processing
   - YOLOv8 for object detection
   - NumPy for numerical operations
   - Pandas for data handling
   - Matplotlib/Seaborn for visualization

3. **Data Flow**
   ```
   Input (Image/Video) → Space Selection → Vehicle Detection → 
   Occupancy Analysis → Report Generation → Output (Reports/CSV)
   ```

### 5. Getting Started
1. **Setup Requirements**
   - Python 3.8 or higher
   - CUDA-capable GPU (recommended)
   - Required Python packages (see requirements.txt)

2. **Running the Project**
   ```bash
   # 1. Clone the repository
   git clone [repository-url]
   
   # 2. Install dependencies
   pip install -r requirements.txt
   
   # 3. Run the application
   python enhanced_parking_detector.py
   ```

3. **Basic Workflow**
   - Load parking lot image/video
   - Select parking spaces
   - Save the layout
   - Run detection
   - View generated reports

### 6. Best Practices
1. **Space Selection**
   - Select spaces in a grid pattern
   - Ensure spaces don't overlap
   - Use zoom for precise selection
   - Save layout frequently

2. **Detection**
   - Ensure good lighting conditions
   - Avoid camera movement
   - Check detection confidence
   - Verify space occupancy

3. **Reporting**
   - Generate reports periodically
   - Check CSV data for trends
   - Monitor detection accuracy
   - Update space layout if needed

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Contributing](#contributing)
- [License](#license)
- [Credits](#credits)
- [Acknowledgments](#acknowledgments)

## Features

### 1. Interactive Parking Space Selection
- Drag-and-select multiple parking spaces at once
- Grid-based automatic space allocation
- Visual feedback during selection
- Right-click to remove individual spaces
- Clear all spaces with a single keypress
- Undo functionality for space selection
- Zoom and pan capabilities for better space selection

### 2. ML-Based Vehicle Detection
- Uses YOLOv8 for accurate vehicle detection
- Detects multiple vehicle types:
  - Cars
  - Motorcycles
  - Buses
  - Trucks
- Provides confidence scores for each detection
- Real-time detection capabilities
- Handles occlusions and partial vehicle visibility

### 3. Comprehensive Reporting
- Visual reports with multiple plots:
  1. Parking lot with detections
  2. Parking statistics
  3. Detection confidence distribution
  4. Vehicle type distribution
- Detailed text reports
- CSV data export
- Real-time statistics display
- Historical data tracking

### 4. Edge Case Handling
- Irregular parking detection
- Large vehicle handling
- Environmental variations
- Lighting condition adaptation
- Moving vehicle detection
- Special parking zones
- Weather condition handling
- Night-time detection capabilities

## Installation

### Prerequisites
- Python 3.8 or higher
- OpenCV
- CUDA-capable GPU (recommended for better performance)

### Setup Steps
1. Clone this repository:
   ```bash
   git clone https://github.com/8harath/Car-Parking-Detection/
   cd car-parking-detection
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv env
   # On Windows
   env\Scripts\activate
   # On Unix or MacOS
   source env/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the YOLOv8 model (automatically handled on first run)

## Usage

### 1. Select Parking Spaces

Run the main script:
```bash
python enhanced_parking_detector.py
```

#### Interactive Controls:
- **Left-click and drag**: Select multiple parking spaces
- **Right-click**: Remove individual parking spaces
- **'R' key**: Reset all selections
- **'Z' key**: Undo last selection
- **'D' key**: Detect cars and generate reports
- **'S' key**: Save current layout
- **'Q' key**: Quit the application
- **Mouse wheel**: Zoom in/out
- **Middle mouse button + drag**: Pan the view

### 2. Generate Reports

Press 'D' to:
- Detect vehicles in the parking lot
- Generate comprehensive reports
- Save visual and text reports
- Update CSV data

### 3. Report Contents

#### Visual Report (`parking_report_[timestamp].png`):
- Parking lot visualization with detections
- Parking statistics bar chart
- Detection confidence distribution
- Vehicle type distribution pie chart

#### Text Report (`parking_report_[timestamp].txt`):
- Total parking spaces
- Detected vehicles count
- Available spaces
- Vehicle type distribution
- Detection confidence statistics
- Timestamp and duration information

#### CSV Data (`parking_status.csv`):
- Total slots
- Occupied slots
- Available slots
- Timestamp
- Vehicle type distribution

## Project Structure

```
car-parking-detection/
├── enhanced_parking_detector.py  # Main detection script
├── car_detector.py              # ML-based car detection
├── main.py                      # Application entry point
├── requirements.txt             # Project dependencies
├── reports/                     # Generated reports directory
├── carPark.mp4                 # Sample video
├── carParkImg.png              # Sample image
└── yolov8n.pt                  # YOLO model weights
```

## Technical Details

### ML Model
- Uses YOLOv8 for vehicle detection
- COCO dataset classes for vehicle types
- Confidence threshold for detections
- Real-time inference capabilities
- GPU acceleration support

### Image Processing
- Grayscale conversion
- Gaussian blur
- Adaptive thresholding
- Median blur for noise reduction
- Dilation for component connection
- Edge detection
- Contour analysis

### Space Detection
- Grid-based space allocation
- Automatic space size calculation
- Occupancy detection using pixel analysis
- Dynamic space adjustment
- Multi-space correlation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Contribution Guidelines
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Credits

- Original inspiration from [Murtaza's Computer Vision Zone](https://www.computervision.zone/)
- Enhanced by Bharath K from Jain Deemed to be University
- YOLOv8 model from [Ultralytics](https://github.com/ultralytics/ultralytics)

## Acknowledgments

- OpenCV for computer vision capabilities
- YOLOv8 for object detection
- Matplotlib and Seaborn for visualization
- Pandas for data handling
- IIT Tirupati Navishkar for the selection process opportunity
- Jain Deemed to be University for academic support

## Contact

For any queries or support, please contact:
- Bharath K (Project Developer)
- Email: 8harath.k@gmail.com
- Webpage: 8harath.me

## Future Enhancements

1. Real-time video streaming support
2. Mobile application integration
3. Cloud-based deployment
4. Multi-camera support
5. Advanced analytics dashboard
6. Machine learning model improvements
7. Weather condition handling
8. Night-time detection enhancements
