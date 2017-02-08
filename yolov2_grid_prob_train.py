import time
import cv2
import numpy as np
import chainer
import glob
import os
from chainer import serializers, optimizers, Variable, cuda
import chainer.functions as F
from yolov2_grid_prob import *
from lib.utils import *
from lib.image_generator import *

# hyper parameters
input_height, input_width = (416, 416)
item_path = "./items"
background_path = "./backgrounds"
label_file = "../dataset/yolov2_fruits_dataset/label.txt"
initial_weight_file = "./backup/partial.model"
backup_path = "backup"
backup_file = "%s/backup.model" % (backup_path)
batch_size = 16
max_batches = 10000
learning_rate = 1e-5
learning_schedules = { 
    "0"    : 1e-5,
    "500"  : 1e-4,
    "10000": 1e-5,
    "20000": 1e-6 
}
lr_decay_power = 4
momentum = 0.9
weight_decay = 0.005
n_classes = 10
n_boxes = 5

# load image generator
print("loading image generator...")
generator = ImageGenerator(item_path, background_path)

# load dataset
with open(label_file, "r") as f:
    labels = f.read().strip().split("\n")

# load model
print("loading initial model...")
yolov2_grid_prob = YOLOv2GridProb(n_classes=n_classes, n_boxes=n_boxes)
model = YOLOv2GridProbPredictor(yolov2_grid_prob)
serializers.load_hdf5(initial_weight_file, model)

model.predictor.train = True
model.predictor.finetune = False
cuda.get_device(0).use()
model.to_gpu() # for gpu

optimizer = optimizers.MomentumSGD(lr=learning_rate, momentum=momentum)
optimizer.use_cleargrads()
optimizer.setup(model)
#optimizer.add_hook(chainer.optimizer.WeightDecay(weight_decay))

# start to train
print("start training")
for batch in range(max_batches):
    if str(batch) in learning_schedules:
        optimizer.lr = learning_schedules[str(batch)]

    # generate sample
    x, t = generator.generate_samples(
        n_samples=batch_size,
        n_items=1,
        crop_width=416,
        crop_height=416,
        min_item_scale=1,
        max_item_scale=3,
        rand_angle=15,
        minimum_crop=0.8,
        delta_hue=0.01,
        delta_sat_scale=0.5,
        delta_val_scale=0.5
    )
    x = Variable(x)
    x.to_gpu()

    # forward
    loss = model(x, t)
    print(batch, optimizer.lr, loss.data)
    print("///////////////////////////")

    optimizer.zero_grads()
    loss.backward()

    optimizer.update()

    # save model
    if (batch+1) % 500 == 0:
        model_file = "%s/%s.model" % (backup_path, batch+1)
        print("saving model to %s" % (model_file))
        serializers.save_hdf5(model_file, model)
        serializers.save_hdf5(backup_file, model)

print("saving model to %s/yolov2_grid_prob_final.model" % (backup_path))
serializers.save_hdf5("%s/yolov2_grid_prob_final.model" % (backup_path), model)
