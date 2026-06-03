import os
import cv2
import numpy as np

def load_train_images(base_dataset_path, category):

    train_good_path = os.path.join(base_dataset_path, category, "train", "good")
    print(f"Loading images from: {train_good_path}")
    images = []

    for filename in sorted(os.listdir(train_good_path)):
        if filename.endswith(".png"):
            full_path = os.path.join(train_good_path, filename)
            img = cv2.imread(full_path, cv2.IMREAD_COLOR)
            if img is not None:
                images.append(img)
                
    return np.array(images)


def load_test_images(base_dataset_path, category, defect_type):
    test_path = os.path.join(base_dataset_path, category, "test", defect_type)
    print(f"Loading test images from: {test_path}")
    images = []

    if not os.path.exists(test_path):
        print(f"Directory does not exist: {test_path}")
        return np.array([])

    for filename in sorted(os.listdir(test_path)):
        if filename.endswith(".png"):
            full_path = os.path.join(test_path, filename)
            img = cv2.imread(full_path, cv2.IMREAD_COLOR)
            if img is not None:
                images.append(img)

    return np.array(images)