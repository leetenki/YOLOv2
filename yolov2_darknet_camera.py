import cv2
import numpy as np
from chainer import serializers, Variable
import chainer.functions as F
from yolov2_darknet_predict import CocoPredictor

cap = cv2.VideoCapture(0)
coco_predictor = CocoPredictor()

while(True):
    ret, orig_img = cap.read()
    nms_results = coco_predictor(orig_img)

    # draw result
    for result in nms_results:
        left, top = result["box"].int_left_top()
        right, bottom = result["box"].int_right_bottom()
        cv2.rectangle(orig_img, (left, top), (right, bottom), (255, 0, 255), 3)
        text = '%s(%2d%%)' % (result["label"], result["probs"].max()*result["conf"]*100)
        cv2.putText(orig_img, text, (left, top-7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        print(text)

    cv2.imshow("w", orig_img)
    cv2.waitKey(1)
