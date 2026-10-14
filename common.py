"""Shared configuration and helpers used by all the scripts."""

import pickle

import cv2

IMAGE_X, IMAGE_Y = 50, 50           # size gesture images are saved/resized to
GESTURE_SAMPLE_ID = 0               # folder used by get_image_size() to probe image dims
CAPTURE_BOX = (300, 100, 300, 300)  # x, y, w, h of the capture rectangle
TOTAL_PICS_PER_GESTURE = 1200


def open_camera():
    """Try camera 1 first (external cam), fall back to 0 (default webcam)."""
    cam = cv2.VideoCapture(1)
    if cam.read()[0] is False:
        cam = cv2.VideoCapture(0)
    return cam


def get_hand_hist():
    with open("hist", "rb") as f:
        return pickle.load(f)


def get_image_size():
    img = cv2.imread("gestures/%d/100.jpg" % GESTURE_SAMPLE_ID, 0)
    if img is None:
        raise FileNotFoundError(
            "Couldn't find gestures/%d/100.jpg -- capture gesture data first "
            "(run create_gestures.py)." % GESTURE_SAMPLE_ID
        )
    return img.shape
