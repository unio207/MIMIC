import cv2
import numpy as np
import os

def load_hsv_thresholds(txt_path):
    """
    Loads HSV thresholds from a text file.
    Returns a dict: {image_name: (lower, upper)}
    """
    thresholds = {}
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 7:
                img_name = parts[0]
                low = tuple(map(int, parts[1:4]))
                up = tuple(map(int, parts[4:7]))
                thresholds[img_name] = (low, up)
    return thresholds

# Example usage:
# hsv_dict = load_hsv_thresholds('C:/Users/henry/Desktop/isef2026/data/masks/lesion_real/gls/manual/hsv_thresholds.txt')
# for img_name, (low, up) in hsv_dict.items():
#     img_path = os.path.join(image_folder, img_name)
#     img = cv2.imread(img_path)
#     hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     mask = cv2.inRange(hsv_img, np.array(low), np.array(up))
#     # ... process mask ...
