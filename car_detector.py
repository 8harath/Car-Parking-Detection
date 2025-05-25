from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

class CarDetector:
    def __init__(self):
        # Load YOLOv8 model
        self.model = YOLO('yolov8n.pt')
        self.car_classes = [2, 3, 5, 7]  # COCO classes for cars, motorcycles, buses, trucks
        self.confidence_threshold = 0.5  # Minimum confidence for detections
        
    def detect_cars(self, image):
        # Run YOLOv8 inference with confidence threshold
        results = self.model(image, conf=self.confidence_threshold)
        return results[0]
    
    def detect_special_parking(self, image, x, y, width, height):
        # Extract the parking space region
        space_region = image[y:y+height, x:x+width]
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(space_region, cv2.COLOR_BGR2HSV)
        
        # Define yellow color ranges for special parking markings
        # Multiple ranges to handle different shades of yellow
        yellow_ranges = [
            (np.array([20, 100, 100]), np.array([30, 255, 255])),  # Standard yellow
            (np.array([15, 100, 100]), np.array([25, 255, 255])),  # Slightly darker yellow
            (np.array([25, 100, 100]), np.array([35, 255, 255]))   # Slightly lighter yellow
        ]
        
        # Create combined mask for all yellow ranges
        yellow_mask = np.zeros_like(hsv[:,:,0])
        for lower, upper in yellow_ranges:
            mask = cv2.inRange(hsv, lower, upper)
            yellow_mask = cv2.bitwise_or(yellow_mask, mask)
        
        # Apply morphological operations to remove noise
        kernel = np.ones((3,3), np.uint8)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
        
        # Calculate percentage of yellow pixels
        yellow_percentage = (np.sum(yellow_mask > 0) / (width * height)) * 100
        
        # Check for yellow lines (horizontal and vertical)
        horizontal_lines = cv2.HoughLinesP(yellow_mask, 1, np.pi/180, threshold=50, minLineLength=width*0.3, maxLineGap=20)
        vertical_lines = cv2.HoughLinesP(yellow_mask, 1, np.pi/180, threshold=50, minLineLength=height*0.3, maxLineGap=20)
        
        # Consider it a special space if:
        # 1. Has significant yellow pixels (>3%)
        # 2. Has clear yellow lines
        has_yellow_lines = (horizontal_lines is not None and len(horizontal_lines) > 0) or \
                          (vertical_lines is not None and len(vertical_lines) > 0)
        
        return yellow_percentage > 3 or has_yellow_lines
    
    def check_space_occupancy(self, image, x, y, width, height):
        # Extract the parking space region
        space_region = image[y:y+height, x:x+width]
        
        # Convert to grayscale
        gray = cv2.cvtColor(space_region, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Calculate percentage of non-zero pixels
        occupancy_percentage = (np.sum(thresh > 0) / (width * height)) * 100
        
        return occupancy_percentage > 30  # Return True if more than 30% of the space is occupied
    
    def process_detections(self, results, parking_spaces, image):
        # Get bounding boxes for detected vehicles
        detections = []
        space_status = []
        
        for pos in parking_spaces:
            x, y = pos
            width, height = 107, 48  # Standard parking space dimensions
            
            # Check if it's a special parking space
            is_special = self.detect_special_parking(image, x, y, width, height)
            
            # Check occupancy
            is_occupied = self.check_space_occupancy(image, x, y, width, height)
            
            # Check for vehicle detection in this space
            vehicle_detected = False
            vehicle_type = None
            confidence = 0
            
            for box in results.boxes:
                if int(box.cls) in self.car_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # Check if the detected vehicle overlaps with this parking space
                    if (x1 < x + width and x2 > x and 
                        y1 < y + height and y2 > y):
                        vehicle_detected = True
                        vehicle_type = int(box.cls)
                        confidence = float(box.conf[0])
                        break
            
            space_status.append({
                'position': (x, y),
                'is_special': is_special,
                'is_occupied': is_occupied,
                'vehicle_detected': vehicle_detected,
                'vehicle_type': vehicle_type,
                'confidence': confidence
            })
            
            if vehicle_detected:
                detections.append({
                    'bbox': (x, y, x + width, y + height),
                    'confidence': confidence,
                    'class_id': vehicle_type,
                    'is_special': is_special
                })
        
        return detections, space_status
    
    def generate_report(self, image, parking_spaces, detections, space_status, output_dir='reports'):
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Original image with detections
        plt.subplot(2, 2, 1)
        img_with_detections = image.copy()
        
        # Draw parking spaces with different colors based on type and status
        for status in space_status:
            x, y = status['position']
            if status['is_special']:
                color = (0, 255, 255)  # Yellow for special parking
            else:
                color = (255, 0, 255)  # Magenta for regular parking
            
            # Draw rectangle
            cv2.rectangle(img_with_detections, (x, y), (x + 107, y + 48), color, 2)
            
            # Add status label
            label = "Special " if status['is_special'] else ""
            label += "Occupied" if status['is_occupied'] else "Empty"
            cv2.putText(img_with_detections, label, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        plt.imshow(cv2.cvtColor(img_with_detections, cv2.COLOR_BGR2RGB))
        plt.title('Parking Lot Analysis')
        plt.axis('off')
        
        # Plot 2: Parking Statistics
        plt.subplot(2, 2, 2)
        total_spaces = len(parking_spaces)
        special_spaces = sum(1 for s in space_status if s['is_special'])
        regular_spaces = total_spaces - special_spaces
        occupied_spaces = sum(1 for s in space_status if s['is_occupied'])
        
        stats = {
            'Total Spaces': total_spaces,
            'Special Spaces': special_spaces,
            'Regular Spaces': regular_spaces,
            'Occupied': occupied_spaces,
            'Available': total_spaces - occupied_spaces
        }
        
        plt.bar(stats.keys(), stats.values(), color=['blue', 'yellow', 'magenta', 'red', 'green'])
        plt.title('Parking Status')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(stats.values()):
            plt.text(i, v, str(v), ha='center', va='bottom')
        
        # Plot 3: Space Type Distribution
        plt.subplot(2, 2, 3)
        space_types = ['Regular', 'Special']
        space_counts = [regular_spaces, special_spaces]
        
        # Only create pie chart if there are spaces
        if sum(space_counts) > 0:
            plt.pie(space_counts, labels=space_types, autopct='%1.1f%%',
                   colors=['magenta', 'yellow'])
            plt.title('Parking Space Types')
        else:
            plt.text(0.5, 0.5, 'No parking spaces detected',
                    ha='center', va='center', fontsize=12)
            plt.title('Parking Space Types')
            plt.axis('off')
        
        # Plot 4: Occupancy by Space Type
        plt.subplot(2, 2, 4)
        special_occupied = sum(1 for s in space_status if s['is_special'] and s['is_occupied'])
        regular_occupied = sum(1 for s in space_status if not s['is_special'] and s['is_occupied'])
        
        # Only create bar chart if there are spaces
        if sum(space_counts) > 0:
            occupancy_data = {
                'Special': [special_occupied, special_spaces - special_occupied],
                'Regular': [regular_occupied, regular_spaces - regular_occupied]
            }
            
            x = np.arange(2)
            width = 0.35
            
            plt.bar(x - width/2, [occupancy_data['Special'][0], occupancy_data['Regular'][0]], 
                    width, label='Occupied', color='red')
            plt.bar(x + width/2, [occupancy_data['Special'][1], occupancy_data['Regular'][1]], 
                    width, label='Available', color='green')
            
            plt.title('Occupancy by Space Type')
            plt.xticks(x, ['Special', 'Regular'])
            plt.legend()
        else:
            plt.text(0.5, 0.5, 'No parking spaces detected',
                    ha='center', va='center', fontsize=12)
            plt.title('Occupancy by Space Type')
            plt.axis('off')
        
        # Save the report
        plt.tight_layout()
        report_path = os.path.join(output_dir, f'parking_report_{timestamp}.png')
        plt.savefig(report_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate concise text report
        text_report_path = os.path.join(output_dir, f'parking_report_{timestamp}.txt')
        with open(text_report_path, 'w') as f:
            f.write(f"Parking Lot Analysis Report\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Space Distribution:\n")
            f.write(f"Total Parking Spaces: {total_spaces}\n")
            f.write(f"Special Parking Spaces: {special_spaces}\n")
            f.write(f"Regular Parking Spaces: {regular_spaces}\n\n")
            
            f.write("Occupancy Status:\n")
            f.write(f"Total Occupied Spaces: {occupied_spaces}\n")
            f.write(f"Total Available Spaces: {total_spaces - occupied_spaces}\n")
            f.write(f"Special Spaces Occupied: {special_occupied}\n")
            f.write(f"Regular Spaces Occupied: {regular_occupied}\n\n")
            
            f.write("Occupancy Rates:\n")
            f.write(f"Overall Occupancy: {(occupied_spaces/total_spaces)*100:.1f}%\n")
            
            # Add checks for division by zero
            if special_spaces > 0:
                f.write(f"Special Spaces Occupancy: {(special_occupied/special_spaces)*100:.1f}%\n")
            else:
                f.write("Special Spaces Occupancy: N/A (No special spaces detected)\n")
                
            if regular_spaces > 0:
                f.write(f"Regular Spaces Occupancy: {(regular_occupied/regular_spaces)*100:.1f}%\n")
            else:
                f.write("Regular Spaces Occupancy: N/A (No regular spaces detected)\n")
        
        return report_path, text_report_path 