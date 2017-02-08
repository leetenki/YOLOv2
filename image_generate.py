import cv2
import numpy as np
from lib.image_generator import *

print("loading image generator...")
input_width = 448
input_height = 448
item_path = "./items"
background_path = "./backgrounds"
generator = ImageGenerator(item_path, background_path)

# generate random sample
x, t = generator.generate_samples(
    n_samples=64,
    n_items=3,
    crop_width=input_width,
    crop_height=input_height,
    min_item_scale=0.5,
    max_item_scale=2,
    rand_angle=30,
    minimum_crop=0.85,
    delta_hue=0.01,
    delta_sat_scale=0.5,
    delta_val_scale=0.5
)

for i, image in enumerate(x):
    image = np.transpose(image, (1, 2, 0)).copy()
    width, height, _ = image.shape
    for truth_box in t[i]:
        box_x, box_y, box_w, box_h = truth_box['x']*width, truth_box['y']*height, truth_box['w']*width, truth_box['h']*height
        image = cv2.rectangle(image.copy(), (int(box_x-box_w/2), int(box_y-box_h/2)), (int(box_x+box_w/2), int(box_y+box_h/2)), (0, 0, 255), 3)

    print(t[i])
    cv2.imshow("w", image)
    cv2.waitKey(500)
