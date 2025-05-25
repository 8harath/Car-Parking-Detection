import sys
from PyQt5.QtWidgets import QApplication
from parking_detection_gui import ParkingDetectionGUI

def main():
    app = QApplication(sys.argv)
    gui = ParkingDetectionGUI()
    gui.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main() 