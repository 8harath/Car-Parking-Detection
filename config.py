"""
Configuration file for Car Parking Detection System
All configurable parameters are centralized here for easy management
"""

# Parking Space Dimensions
PARKING_WIDTH = 107
PARKING_HEIGHT = 48

# Detection Thresholds
OCCUPANCY_THRESHOLD = 900  # Pixel count threshold for space occupancy
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for vehicle detection

# YOLO Model Configuration
MODEL_PATH = 'yolov8n.pt'
CAR_CLASSES = [2, 3, 5, 7]  # COCO classes: cars, motorcycles, buses, trucks

# File Paths
POSITION_FILE = 'CarParkPos'
DEFAULT_VIDEO_PATH = 'carPark.mp4'
DEFAULT_IMAGE_PATH = 'carParkImg.png'

# Directory Structure
REPORTS_DIR = 'reports'
DATA_DIR = 'data'
MODELS_DIR = 'models'

# CSV Configuration
CSV_FILE = 'parking_status.csv'
CSV_APPEND_MODE = True  # Set to True to keep historical data

# UI Configuration
BORDER_SIZE = 50
WINDOW_NAME = "Parking Detection"

# Zoom Configuration
MIN_ZOOM = 0.5
MAX_ZOOM = 3.0
ZOOM_STEP = 0.1
PAN_STEP = 50

# Colors (BGR format)
COLOR_EMPTY = (0, 255, 0)  # Green
COLOR_OCCUPIED = (0, 0, 255)  # Red
COLOR_SPECIAL = (0, 255, 255)  # Yellow
COLOR_REGULAR = (255, 0, 255)  # Magenta
COLOR_SELECTION = (0, 255, 0)  # Green

# Special Parking Detection
YELLOW_PERCENTAGE_THRESHOLD = 3  # Minimum percentage of yellow pixels for special parking
MIN_LINE_LENGTH_RATIO = 0.3  # Minimum line length as ratio of space dimension

# Report Configuration
REPORT_DPI = 300
REPORT_FIGSIZE = (15, 10)
