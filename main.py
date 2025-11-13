import cv2
import pickle
import cvzone
import numpy as np
import config
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Video feed
try:
    if not os.path.exists(config.DEFAULT_VIDEO_PATH):
        raise FileNotFoundError(f"Video file not found: {config.DEFAULT_VIDEO_PATH}")
    cap = cv2.VideoCapture(config.DEFAULT_VIDEO_PATH)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {config.DEFAULT_VIDEO_PATH}")
    logger.info(f"Video loaded: {config.DEFAULT_VIDEO_PATH}")
except Exception as e:
    logger.error(f"Error loading video: {e}")
    raise

# Load parking positions
try:
    with open(config.POSITION_FILE, 'rb') as f:
        posList = pickle.load(f)
    logger.info(f"Loaded {len(posList)} parking positions")
except FileNotFoundError:
    logger.error(f"Parking positions file not found: {config.POSITION_FILE}")
    raise
except Exception as e:
    logger.error(f"Error loading positions: {e}")
    raise

width, height = config.PARKING_WIDTH, config.PARKING_HEIGHT


def checkParkingSpace(imgPro):
    """Check each parking space and count available slots"""
    spaceCounter = 0

    for pos in posList:
        x, y = pos

        imgCrop = imgPro[y:y + height, x:x + width]
        # cv2.imshow(str(x * y), imgCrop)
        count = cv2.countNonZero(imgCrop)

        if count < config.OCCUPANCY_THRESHOLD:
            color = (0, 255, 0)
            thickness = 5
            spaceCounter += 1
        else:
            color = (0, 0, 255)
            thickness = 2

        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, thickness)
        cvzone.putTextRect(img, str(count), (x, y + height - 3), scale=1,
                           thickness=2, offset=0, colorR=color)

    cvzone.putTextRect(img, f'Free: {spaceCounter}/{len(posList)}', (100, 50), scale=3,
                           thickness=5, offset=20, colorR=(0,200,0))

try:
    while True:
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        success, img = cap.read()
        if not success:
            logger.warning("Failed to read frame")
            break

        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
        imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY_INV, 25, 16)
        imgMedian = cv2.medianBlur(imgThreshold, 5)
        kernel = np.ones((3, 3), np.uint8)
        imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)

        checkParkingSpace(imgDilate)
        cv2.imshow("Image", img)
        # cv2.imshow("ImageBlur", imgBlur)
        # cv2.imshow("ImageThres", imgMedian)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    logger.info("Application closed")