"""Stage 4 - Load all gesture images and split them into train/test/val."""

import os
import pickle
from glob import glob

import cv2
import numpy as np


def stage_load_images():
    from sklearn.utils import shuffle

    def pickle_images_labels():
        images_labels = []
        images = glob("gestures/*/*.jpg")
        images.sort()
        for image in images:
            label = image[image.find(os.sep) + 1: image.rfind(os.sep)]
            img = cv2.imread(image, 0)
            images_labels.append((np.array(img, dtype=np.uint8), int(label)))
        return images_labels

    images_labels = pickle_images_labels()
    images_labels = shuffle(images_labels)
    images, labels = zip(*images_labels)
    print("Length of images_labels", len(images_labels))

    def save_split(name_images, name_labels, imgs, lbls):
        print("Length of %s" % name_images, len(imgs))
        with open(name_images, "wb") as f:
            pickle.dump(imgs, f)
        with open(name_labels, "wb") as f:
            pickle.dump(lbls, f)

    n = len(images)
    save_split("train_images", "train_labels", images[:int(5 / 6 * n)], labels[:int(5 / 6 * n)])
    save_split("test_images", "test_labels",
                images[int(5 / 6 * n):int(11 / 12 * n)], labels[int(5 / 6 * n):int(11 / 12 * n)])
    save_split("val_images", "val_labels", images[int(11 / 12 * n):], labels[int(11 / 12 * n):])
    print("Dataset split into train/test/val pickle files.")


if __name__ == "__main__":
    stage_load_images()
