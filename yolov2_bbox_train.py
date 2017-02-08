import time
import cv2
import numpy as np
import chainer
import glob
import os
from chainer import serializers, optimizers, Variable, cuda
import chainer.functions as F
from yolov2_bbox import *
from lib.utils import *

# hyper parameters
input_height, input_width = (416, 416)
train_file = "../dataset/yolov2_fruits_dataset/train.txt"
label_file = "../dataset/yolov2_fruits_dataset/label.txt"
truth_path = "../dataset/yolov2_fruits_dataset/labels/"
initial_weight_file = "./backup/partial.model"
backup_path = "backup"
backup_file = "%s/backup.model" % (backup_path)
batch_size = 16
max_batches = 30000
learning_rate = 1e-5
learning_schedules = { "0": 1e-5, "500": 1e-4, "10000": 1e-5, "20000": 1e-6 }
lr_decay_power = 4
momentum = 0.9
weight_decay = 0.005
n_classes = 10
n_boxes = 5

# load dataset
with open(train_file, "r") as f:
    image_files = f.read().strip().split("\n")

with open(label_file, "r") as f:
    labels = f.read().strip().split("\n")

x_train = []
t_train = [] # normal label
print("loading image datasets...")
for image_file in image_files:
    img = cv2.imread(image_file)
    img = cv2.resize(img, (input_height, input_width))
    img = np.asarray(img, dtype=np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    x_train.append(img)

    groundtruth_file = truth_path + image_file.split("/")[-1].split(".")[0] + ".txt"
    with open(groundtruth_file, "r") as f:
        truth_objects = f.read().strip().split("\n")
        truth_info = []
        for truth_object in truth_objects:
            one_hot_label = np.zeros(len(labels)) # one hot label
            label, x, y, w, h = truth_object.split(" ")
            one_hot_label[label] = 1
            truth_info.append({"label": label, "one_hot_label": one_hot_label, "x": x, "y": y, "h": h, "w": w})
        t_train.append(truth_info)

x_train = np.array(x_train)

# load model
print("loading initial model...")
yolov2_bbox = YOLOv2Bbox(n_classes=n_classes, n_boxes=n_boxes)
model = YOLOv2BboxPredictor(yolov2_bbox)
serializers.load_hdf5(initial_weight_file, model)

model.predictor.train = True
model.predictor.finetune = False

if hasattr(cuda, "cupy"):
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

    batch_mask = np.random.choice(len(x_train), batch_size)
    x = Variable(x_train[batch_mask])
    t = np.array(t_train)[batch_mask]
    if hasattr(cuda, "cupy"):
        x.to_gpu() # for gpu

    # forward
    loss = model(x, t)
    print(batch, optimizer.lr, loss.data)
    print("/////////////////////////////////////")

    optimizer.zero_grads()
    loss.backward()

    optimizer.update()

    # save model
    if (batch+1) % 500 == 0:
        model_file = "%s/%s.model" % (backup_path, batch+1)
        print("saving model to %s" % (model_file))
        serializers.save_hdf5(model_file, model)
        serializers.save_hdf5(backup_file, model)

print("saving model to %s/yolov2_bbox_final.model" % (backup_path))
serializers.save_hdf5("%s/yolov2_bbox_final.model" % (backup_path), model)
