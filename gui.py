import sys
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QComboBox, 
                            QSpinBox, QDoubleSpinBox, QTabWidget, QGroupBox,
                            QGridLayout, QFileDialog, QMessageBox, QSlider)
from PyQt5.QtCore import Qt, QTimer, QPoint
from PyQt5.QtGui import QImage, QPixmap, QMouseEvent
import numpy as np
from enhanced_parking_detector import EnhancedParkingDetector
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class ImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.setMouseTracking(True)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.start_point = event.pos()
            self.end_point = event.pos()
        elif event.button() == Qt.RightButton:
            # Handle right-click to remove spaces
            if self.parent.detector:
                x = event.pos().x()
                y = event.pos().y()
                # Convert coordinates to image space
                x = int(x / self.width() * self.parent.detector.current_image.shape[1])
                y = int(y / self.height() * self.parent.detector.current_image.shape[0])
                
                # Remove spaces in the clicked area
                for pos in self.parent.detector.posList[:]:
                    x1, y1 = pos
                    if x1 < x < x1 + self.parent.detector.width and y1 < y < y1 + self.parent.detector.height:
                        self.parent.detector.posList.remove(pos)
                self.parent.detector.save_parking_positions()
                self.parent.update_frame()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.drawing:
            self.end_point = event.pos()
            self.parent.update_frame()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            if self.parent.detector:
                # Convert coordinates to image space
                x1 = int(self.start_point.x() / self.width() * self.parent.detector.current_image.shape[1])
                y1 = int(self.start_point.y() / self.height() * self.parent.detector.current_image.shape[0])
                x2 = int(self.end_point.x() / self.width() * self.parent.detector.current_image.shape[1])
                y2 = int(self.end_point.y() / self.height() * self.parent.detector.current_image.shape[0])
                
                # Calculate grid of parking spaces
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                # Calculate number of spaces that fit
                num_spaces_x = (x2 - x1) // self.parent.detector.width
                num_spaces_y = (y2 - y1) // self.parent.detector.height
                
                # Add all spaces in the grid
                for i in range(num_spaces_x):
                    for j in range(num_spaces_y):
                        pos = (x1 + i * self.parent.detector.width, y1 + j * self.parent.detector.height)
                        if pos not in self.parent.detector.posList:
                            self.parent.detector.posList.append(pos)
                
                self.parent.detector.save_parking_positions()
                self.parent.update_frame()

    def wheelEvent(self, event):
        if self.parent.detector:
            # Zoom in/out with mouse wheel
            delta = event.angleDelta().y()
            if delta > 0:
                self.parent.zoom_in()
            else:
                self.parent.zoom_out()

class ParkingDetectionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.detector = None
        self.video_path = None
        self.image_path = None
        self.is_processing = False
        self.zoom_scale = 1.0
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Parking Space Detection System')
        self.setGeometry(100, 100, 1200, 800)

        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # Create left panel for controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(300)
        layout.addWidget(left_panel)

        # Create right panel for video/image display and statistics
        right_panel = QTabWidget()
        layout.addWidget(right_panel)

        # Add tabs for different views
        self.video_tab = QWidget()
        self.stats_tab = QWidget()
        right_panel.addTab(self.video_tab, "Video/Image")
        right_panel.addTab(self.stats_tab, "Statistics")

        # Setup video tab
        video_layout = QVBoxLayout(self.video_tab)
        self.video_label = ImageLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        video_layout.addWidget(self.video_label)

        # Setup statistics tab
        stats_layout = QVBoxLayout(self.stats_tab)
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        stats_layout.addWidget(self.canvas)

        # Add control groups
        self.add_file_controls(left_layout)
        self.add_detection_controls(left_layout)
        self.add_settings_controls(left_layout)
        self.add_zoom_controls(left_layout)
        self.add_keyboard_shortcuts(left_layout)

        # Initialize timer for video processing
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Initialize statistics update timer
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self.update_statistics)
        self.stats_timer.start(1000)  # Update every second

    def add_file_controls(self, layout):
        group = QGroupBox("File Controls")
        group_layout = QVBoxLayout()

        # Video file selection
        video_btn = QPushButton("Load Video")
        video_btn.clicked.connect(self.load_video)
        group_layout.addWidget(video_btn)

        # Image file selection
        image_btn = QPushButton("Load Image")
        image_btn.clicked.connect(self.load_image)
        group_layout.addWidget(image_btn)

        group.setLayout(group_layout)
        layout.addWidget(group)

    def add_detection_controls(self, layout):
        group = QGroupBox("Detection Controls")
        group_layout = QVBoxLayout()

        # Start/Stop detection
        self.detect_btn = QPushButton("Start Detection")
        self.detect_btn.clicked.connect(self.toggle_detection)
        group_layout.addWidget(self.detect_btn)

        # Save layout
        save_btn = QPushButton("Save Layout")
        save_btn.clicked.connect(self.save_layout)
        group_layout.addWidget(save_btn)

        # Reset layout
        reset_btn = QPushButton("Reset Layout")
        reset_btn.clicked.connect(self.reset_layout)
        group_layout.addWidget(reset_btn)

        group.setLayout(group_layout)
        layout.addWidget(group)

    def add_settings_controls(self, layout):
        group = QGroupBox("Settings")
        group_layout = QGridLayout()

        # Confidence threshold
        group_layout.addWidget(QLabel("Confidence Threshold:"), 0, 0)
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.1, 1.0)
        self.confidence_spin.setValue(0.5)
        self.confidence_spin.setSingleStep(0.1)
        group_layout.addWidget(self.confidence_spin, 0, 1)

        # Space size
        group_layout.addWidget(QLabel("Space Width:"), 1, 0)
        self.width_spin = QSpinBox()
        self.width_spin.setRange(50, 200)
        self.width_spin.setValue(120)
        group_layout.addWidget(self.width_spin, 1, 1)

        group_layout.addWidget(QLabel("Space Height:"), 2, 0)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(20, 100)
        self.height_spin.setValue(43)
        group_layout.addWidget(self.height_spin, 2, 1)

        # Apply settings
        apply_btn = QPushButton("Apply Settings")
        apply_btn.clicked.connect(self.apply_settings)
        group_layout.addWidget(apply_btn, 3, 0, 1, 2)

        group.setLayout(group_layout)
        layout.addWidget(group)

    def add_zoom_controls(self, layout):
        group = QGroupBox("Zoom Controls")
        group_layout = QVBoxLayout()

        # Zoom in button
        zoom_in_btn = QPushButton("Zoom In (+)")
        zoom_in_btn.clicked.connect(self.zoom_in)
        group_layout.addWidget(zoom_in_btn)

        # Zoom out button
        zoom_out_btn = QPushButton("Zoom Out (-)")
        zoom_out_btn.clicked.connect(self.zoom_out)
        group_layout.addWidget(zoom_out_btn)

        # Zoom reset button
        zoom_reset_btn = QPushButton("Reset Zoom")
        zoom_reset_btn.clicked.connect(self.reset_zoom)
        group_layout.addWidget(zoom_reset_btn)

        # Zoom slider
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setMinimum(50)  # 50% zoom
        self.zoom_slider.setMaximum(200)  # 200% zoom
        self.zoom_slider.setValue(100)  # 100% zoom
        self.zoom_slider.valueChanged.connect(self.zoom_slider_changed)
        group_layout.addWidget(self.zoom_slider)

        group.setLayout(group_layout)
        layout.addWidget(group)

    def add_keyboard_shortcuts(self, layout):
        group = QGroupBox("Keyboard Shortcuts")
        group_layout = QVBoxLayout()

        shortcuts = [
            "R: Reset all selections",
            "Z: Undo last selection",
            "D: Detect & generate report",
            "S: Save current layout",
            "Space: Start/Stop detection",
            "Esc: Quit application"
        ]

        for shortcut in shortcuts:
            group_layout.addWidget(QLabel(shortcut))

        group.setLayout(group_layout)
        layout.addWidget(group)

    def load_video(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", 
                                                 "Video Files (*.mp4 *.avi *.mov)")
        if file_name:
            self.video_path = file_name
            self.detector = EnhancedParkingDetector(video_path=file_name)
            self.timer.start(30)  # 30ms = ~33fps

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", 
                                                 "Image Files (*.png *.jpg *.jpeg)")
        if file_name:
            self.image_path = file_name
            self.detector = EnhancedParkingDetector(image_path=file_name)
            self.update_frame()

    def toggle_detection(self):
        if self.is_processing:
            self.timer.stop()
            self.detect_btn.setText("Start Detection")
        else:
            self.timer.start(30)
            self.detect_btn.setText("Stop Detection")
        self.is_processing = not self.is_processing

    def save_layout(self):
        if self.detector:
            self.detector.save_parking_positions()
            QMessageBox.information(self, "Success", "Layout saved successfully!")

    def reset_layout(self):
        if self.detector:
            self.detector.clear_all_markings()
            self.update_frame()
            QMessageBox.information(self, "Reset", "All selections have been cleared!")

    def undo_last_selection(self):
        if self.detector and self.detector.history:
            self.detector.undo_last_selection()
            self.update_frame()
            QMessageBox.information(self, "Undo", "Last selection has been removed!")

    def apply_settings(self):
        if self.detector:
            self.detector.width = self.width_spin.value()
            self.detector.height = self.height_spin.value()
            # Update other settings as needed

    def zoom_in(self):
        self.zoom_scale = min(2.0, self.zoom_scale + 0.1)
        self.zoom_slider.setValue(int(self.zoom_scale * 100))
        self.update_frame()

    def zoom_out(self):
        self.zoom_scale = max(0.5, self.zoom_scale - 0.1)
        self.zoom_slider.setValue(int(self.zoom_scale * 100))
        self.update_frame()

    def reset_zoom(self):
        self.zoom_scale = 1.0
        self.zoom_slider.setValue(100)
        self.update_frame()

    def zoom_slider_changed(self, value):
        self.zoom_scale = value / 100.0
        self.update_frame()

    def update_frame(self):
        if not self.detector:
            return

        if self.video_path:
            ret, frame = self.detector.cap.read()
            if not ret:
                self.detector.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.detector.cap.read()

        elif self.image_path:
            frame = self.detector.current_image.copy() if self.detector.current_image is not None else cv2.imread(self.image_path)
        else:
            return

        # Process frame
        processed_img = self.detector.process_frame(frame)
        available_slots, occupied_slots = self.detector.check_parking_space(processed_img, frame)

        # Draw selection rectangle if drawing
        if self.video_label.drawing and self.video_label.start_point and self.video_label.end_point:
            cv2.rectangle(frame, 
                        (self.video_label.start_point.x(), self.video_label.start_point.y()),
                        (self.video_label.end_point.x(), self.video_label.end_point.y()),
                        (0, 255, 0), 2)

        # Convert frame to QImage
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Apply zoom
        scaled_size = qt_image.size() * self.zoom_scale
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            scaled_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled_pixmap)

    def update_statistics(self):
        if not self.detector or not self.detector.posList:
            return

        # Clear previous plots
        self.figure.clear()

        # Create subplots
        ax1 = self.figure.add_subplot(221)
        ax2 = self.figure.add_subplot(222)
        ax3 = self.figure.add_subplot(223)
        ax4 = self.figure.add_subplot(224)

        # Plot 1: Parking Space Distribution
        total_spaces = len(self.detector.posList)
        if hasattr(self.detector, 'last_available_slots'):
            available = self.detector.last_available_slots
            occupied = total_spaces - available
            ax1.bar(['Available', 'Occupied'], [available, occupied])
            ax1.set_title('Parking Space Distribution')

        # Plot 2: Vehicle Types (if available)
        if hasattr(self.detector, 'last_vehicle_types'):
            vehicle_types = self.detector.last_vehicle_types
            ax2.pie(vehicle_types.values(), labels=vehicle_types.keys(), autopct='%1.1f%%')
            ax2.set_title('Vehicle Type Distribution')

        # Plot 3: Detection Confidence (if available)
        if hasattr(self.detector, 'last_confidences'):
            confidences = self.detector.last_confidences
            ax3.hist(confidences, bins=10)
            ax3.set_title('Detection Confidence Distribution')

        # Plot 4: Time Series (if available)
        if hasattr(self.detector, 'occupancy_history'):
            history = self.detector.occupancy_history
            ax4.plot(history)
            ax4.set_title('Occupancy Over Time')

        self.figure.tight_layout()
        self.canvas.draw()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.toggle_detection()
        elif event.key() == Qt.Key_Escape:
            self.close()
        elif event.key() == Qt.Key_R:
            self.reset_layout()
        elif event.key() == Qt.Key_Z:
            self.undo_last_selection()
        elif event.key() == Qt.Key_D:
            if self.detector:
                self.detector.process_image()
        elif event.key() == Qt.Key_S:
            self.save_layout()

    def closeEvent(self, event):
        if self.detector:
            self.detector.cap.release()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = ParkingDetectionGUI()
    gui.show()
    sys.exit(app.exec_()) 