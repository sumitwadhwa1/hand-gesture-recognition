"""Stage 1 - Capture the skin-colour histogram used to segment your hand."""

import pickle

import cv2
import numpy as np

from common import CAPTURE_BOX
from common import open_camera


def build_squares(img):
    x, y, w, h = 420, 140, 10, 10
    d = 10
    imgCrop = None
    crop = None
    for i in range(10):
        for j in range(5):
            if imgCrop is None:
                imgCrop = img[y:y + h, x:x + w]
            else:
                imgCrop = np.hstack((imgCrop, img[y:y + h, x:x + w]))
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
            x += w + d
        if crop is None:
            crop = imgCrop
        else:
            crop = np.vstack((crop, imgCrop))
        imgCrop = None
        x = 420
        y += h + d
    return crop


def stage_set_hand_histogram():
    """Show a grid of squares over your hand, press 'c' to capture the
    histogram, 's' to save it to the 'hist' file and exit."""
    cam = open_camera()
    x, y, w, h = CAPTURE_BOX
    flagPressedC, flagPressedS = False, False
    imgCrop = None
    hist = None

    while True:
        img = cam.read()[1]
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (640, 480))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        keypress = cv2.waitKey(1)
        if keypress == ord('c'):
            hsvCrop = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2HSV)
            flagPressedC = True
            hist = cv2.calcHist([hsvCrop], [0, 1], None, [180, 256], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        elif keypress == ord('s'):
            flagPressedS = True
            break

        if flagPressedC:
            dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
            disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
            cv2.filter2D(dst, -1, disc, dst)
            blur = cv2.GaussianBlur(dst, (11, 11), 0)
            blur = cv2.medianBlur(blur, 15)
            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thresh = cv2.merge((thresh, thresh, thresh))
            cv2.imshow("Thresh", thresh)

        if not flagPressedS:
            imgCrop = build_squares(img)

        cv2.imshow("Set hand histogram", img)

    cam.release()
    cv2.destroyAllWindows()

    if hist is None:
        print("No histogram captured (you exited before pressing 'c'). Nothing saved.")
        return

    with open("hist", "wb") as f:
        pickle.dump(hist, f)
    print("Saved histogram to 'hist'.")


if __name__ == "__main__":
    stage_set_hand_histogram()
