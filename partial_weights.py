import time
import cv2
import numpy as np
from chainer import serializers, Variable
import chainer.functions as F
import argparse
from darknet19 import *
from yolov2 import *
from yolov2_grid_prob import *
from yolov2_bbox import *

n_classes = 10
n_boxes = 5
partial_layer = 18

def copy_conv_layer(src, dst, layers):
    for i in layers:
        src_layer = eval("src.conv%d" % i)
        dst_layer = eval("dst.conv%d" % i)        
        dst_layer.W = src_layer.W
        dst_layer.b = src_layer.b

def copy_bias_layer(src, dst, layers):
    for i in layers:
        src_layer = eval("src.bias%d" % i)
        dst_layer = eval("dst.bias%d" % i)        
        dst_layer.b = src_layer.b

def copy_bn_layer(src, dst, layers):
    for i in layers:
        src_layer = eval("src.bn%d" % i)
        dst_layer = eval("dst.bn%d" % i)        
        dst_layer.N = src_layer.N
        dst_layer.avg_var = src_layer.avg_var
        dst_layer.avg_mean = src_layer.avg_mean
        dst_layer.gamma = src_layer.gamma
        dst_layer.eps = src_layer.eps

# load model
print("loading original model...")
input_weight_file = "./backup/darknet19_448_final.model"
output_weight_file = "./backup/partial.model"

model = Darknet19Predictor(Darknet19())
serializers.load_hdf5(input_weight_file, model) # load saved model

yolov2 = YOLOv2(n_classes=n_classes, n_boxes=n_boxes)
copy_conv_layer(model.predictor, yolov2, range(1, partial_layer+1))
copy_bias_layer(model.predictor, yolov2, range(1, partial_layer+1))
copy_bn_layer(model.predictor, yolov2, range(1, partial_layer+1))
model = YOLOv2Predictor(yolov2)

print("saving model to %s" % (output_weight_file))
serializers.save_hdf5("%s" % (output_weight_file), model)
