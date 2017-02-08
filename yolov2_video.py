import time
import cv2
import numpy as np
from chainer import serializers, Variable
import chainer.functions as F
import argparse
from yolov2 import *
from lib.image_generator import *
from yolov2_predict import *

item_path = "./items"
background_path = "./backgrounds"
input_width, input_height = (416, 416)
loop = 10

# load image generator
print("loading image generator...")
generator = ImageGenerator(item_path, background_path)
animation = generator.generate_random_animation(loop=loop, bg_index=4, crop_width=input_width, crop_height=input_height, min_item_scale=1.0, max_item_scale=2.0)

for i in animation:
    cv2.imshow("w", i)
    cv2.waitKey(1)

# init video writer
codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
video_writer = cv2.VideoWriter('output.avi', codec, 25.0, (input_width, input_height)) 

# load predictor
predictor = AnimalPredictor()

for frame in animation:
    orig_img = frame.copy()
    nms_results = predictor(orig_img)

    # draw result
    for result in nms_results:
        left, top = result["box"].int_left_top()
        cv2.rectangle(
            orig_img,
            result["box"].int_left_top(), result["box"].int_right_bottom(),
            (0, 255, 0),
            3
        )
        text = '%s(%2d%%)' % (result["label"], result["probs"].max()*result["conf"]*100)
        cv2.putText(orig_img, text, (left, top-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        print(text)

    cv2.imshow("w", orig_img)
    cv2.waitKey(1)

    video_writer.write(orig_img)
video_writer.release()
