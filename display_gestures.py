"""Stage 6 - Show a grid with one random sample image per gesture."""

import os
import random

import cv2
import numpy as np

from common import TOTAL_PICS_PER_GESTURE
from common import get_image_size


def stage_display_gestures():
    image_x, image_y = get_image_size()
    gestures = os.listdir('gestures/')
    gestures.sort(key=int)
    begin_index, end_index = 0, 5

    rows = len(gestures) // 5 + (1 if len(gestures) % 5 != 0 else 0)

    full_img = None
    for _ in range(rows):
        col_img = None
        for j in range(begin_index, end_index):
            img_path = "gestures/%s/%d.jpg" % (j, random.randint(1, TOTAL_PICS_PER_GESTURE))
            img = cv2.imread(img_path, 0)
            if img is None:
                img = np.zeros((image_y, image_x), dtype=np.uint8)
            col_img = img if col_img is None else np.hstack((col_img, img))
        begin_index += 5
        end_index += 5
        full_img = col_img if full_img is None else np.vstack((full_img, col_img))

    cv2.imshow("gestures", full_img)
    cv2.imwrite('full_img.jpg', full_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    stage_display_gestures()
