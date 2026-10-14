"""Stage 3 - Augment the dataset by horizontally flipping every image."""

import os

import cv2

from common import TOTAL_PICS_PER_GESTURE


def stage_rotate_images():
    gest_folder = "gestures"
    for g_id in os.listdir(gest_folder):
        for i in range(TOTAL_PICS_PER_GESTURE):
            path = "%s/%s/%d.jpg" % (gest_folder, g_id, i + 1)
            new_path = "%s/%s/%d.jpg" % (gest_folder, g_id, i + 1 + TOTAL_PICS_PER_GESTURE)
            img = cv2.imread(path, 0)
            if img is None:
                continue
            img = cv2.flip(img, 1)
            cv2.imwrite(new_path, img)
    print("Done flipping/augmenting images.")


if __name__ == "__main__":
    stage_rotate_images()
