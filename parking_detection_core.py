import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import os
from vehicle_detection import VehicleDetector

class ParkingDetectionCore:
    def __init__(self, video_path=None, image_path=None, model_path='yolov8n.pt'):
        self.video_path = video_path
        self.image_path = image_path
        self.cap = None
        self.posList = []
        self.width = 120
        self.height = 43
        self.history = []
        self.occupancy_history = []
        self.last_available_slots = 0
        self.last_vehicle_types = {}
        self.last_confidences = []
        
        # Initialize vehicle detector
        self.vehicle_detector = VehicleDetector(model_path=model_path)
        
        # Load video or image
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
        elif image_path:
            self.current_image = cv2.imread(image_path)
            self.original_image = self.current_image.copy()
        
        # Create reports directory if it doesn't exist
        if not os.path.exists('reports'):
            os.makedirs('reports')
        
        # Load existing parking positions if available
        self.load_parking_positions()

    def load_parking_positions(self):
        try:
            if os.path.exists('CarParkPos'):
                with open('CarParkPos', 'r') as f:
                    for line in f:
                        x, y = map(int, line.strip().split(','))
                        self.posList.append((x, y))
        except Exception as e:
            print(f"Error loading parking positions: {e}")

    def save_parking_positions(self):
        try:
            with open('CarParkPos', 'w') as f:
                for pos in self.posList:
                    f.write(f"{pos[0]},{pos[1]}\n")
        except Exception as e:
            print(f"Error saving parking positions: {e}")

    def clear_all_markings(self):
        self.posList = []
        self.history = []
        self.current_image = self.original_image.copy() if hasattr(self, 'original_image') else None
        self.save_parking_positions()

    def undo_last_selection(self):
        if self.history:
            self.posList = self.history.pop()
            self.save_parking_positions()

    def process_frame(self, frame):
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 1)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 25, 16)
        median = cv2.medianBlur(thresh, 5)
        kernel = np.ones((3, 3), np.uint8)
        dilate = cv2.dilate(median, kernel, iterations=1)
        return dilate

    def check_parking_space(self, processed_img, frame):
        available_slots = 0
        occupied_slots = 0
        vehicle_types = {}
        confidences = []
        
        for pos in self.posList:
            x, y = pos
            space_crop = processed_img[y:y + self.height, x:x + self.width]
            count = cv2.countNonZero(space_crop)
            
            if count < 900:
                color = (0, 255, 0)  # Green for available
                available_slots += 1
            else:
                color = (0, 0, 255)  # Red for occupied
                occupied_slots += 1
                
                # Detect vehicle type in occupied space
                space_img = frame[y:y + self.height, x:x + self.width]
                detections = self.vehicle_detector.detect(space_img)
                
                for detection in detections:
                    vehicle_type = detection['class_name']
                    confidence = detection['confidence']
                    
                    if vehicle_type not in vehicle_types:
                        vehicle_types[vehicle_type] = 0
                    vehicle_types[vehicle_type] += 1
                    confidences.append(confidence)
            
            cv2.rectangle(frame, (x, y), (x + self.width, y + self.height), color, 2)
        
        # Update statistics
        self.last_available_slots = available_slots
        self.last_vehicle_types = vehicle_types
        self.last_confidences = confidences
        self.occupancy_history.append(occupied_slots)
        
        return available_slots, occupied_slots

    def process_image(self):
        if not hasattr(self, 'current_image') or self.current_image is None:
            return
        
        # Save current state for undo
        self.history.append(self.posList.copy())
        
        # Process the image
        processed_img = self.process_frame(self.current_image)
        available_slots, occupied_slots = self.check_parking_space(processed_img, self.current_image)
        
        # Generate report
        self.generate_report(available_slots, occupied_slots)
        
        return self.current_image

    def generate_report(self, available_slots, occupied_slots):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"reports/parking_report_{timestamp}.csv"
        
        data = {
            'Timestamp': [datetime.now()],
            'Total Spaces': [len(self.posList)],
            'Available Spaces': [available_slots],
            'Occupied Spaces': [occupied_slots],
            'Occupancy Rate': [f"{(occupied_slots/len(self.posList)*100):.2f}%"]
        }
        
        # Add vehicle type distribution
        for vehicle_type, count in self.last_vehicle_types.items():
            data[f'{vehicle_type} Count'] = [count]
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(data)
        df.to_csv(report_path, index=False)
        
        return report_path

    def __del__(self):
        if self.cap is not None:
            self.cap.release()
 