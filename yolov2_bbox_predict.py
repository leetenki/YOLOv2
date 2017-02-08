import time
import cv2
import numpy as np
from chainer import serializers, Variable
import chainer.functions as F
import argparse
from yolov2_bbox import *

# argument parse
parser = argparse.ArgumentParser(description="指定したパスの画像を読み込み、bboxの予測を行う")
parser.add_argument('path', help="画像ファイルへのパスを指定")
args = parser.parse_args()

# hyper parameters
input_height, input_width = (416, 416)
weight_file = "./backup/yolov2_bbox_final.model"
label_file = "../dataset/yolov2_fruits_dataset/label.txt"
image_file = args.path
n_classes = 10
n_boxes = 5

# read labels
with open(label_file, "r") as f:
    labels = f.read().strip().split("\n")

# read image
print("loading image...")
orig_img = cv2.imread(image_file)
orig_img = cv2.resize(orig_img, (input_height, input_width))
img = np.asarray(orig_img, dtype=np.float32) / 255.0
img = img.transpose(2, 0, 1)

# read label
truth_file = image_file.replace("images", "labels").replace(".jpg", ".txt")
with open(truth_file, "r") as f:
    truth_class, truth_x, truth_y, truth_w, truth_h = f.read().split(" ")

# load model
print("loading model...")
model = YOLOv2BboxPredictor(YOLOv2Bbox(n_classes=n_classes, n_boxes=n_boxes))
serializers.load_hdf5(weight_file, model) # load saved model
model.predictor.train = False
model.predictor.finetune = False

# forward
x_data = img[np.newaxis, :, :, :]
x = Variable(x_data)
if hasattr(cuda, "cupy"):
    cuda.get_device(0).use()
    model.to_gpu()
    x.to_gpu()

x, y, w, h, conf = model.predict(x)
_, _, _, grid_h, grid_w = x.shape
x = F.reshape(x, (n_boxes, grid_h, grid_w)).data.get()
y = F.reshape(y, (n_boxes, grid_h, grid_w)).data.get()
w = F.reshape(w, (n_boxes, grid_h, grid_w)).data.get()
h = F.reshape(h, (n_boxes, grid_h, grid_w)).data.get()
conf = F.reshape(conf, (n_boxes, grid_h, grid_w)).data.get()

best_iou = 0
best_box = None
truth_box = Box(float(truth_x), float(truth_y), float(truth_w), float(truth_h))
for h_index in range(grid_h):
    for w_index in range(grid_w):
        for b_index in range(n_boxes):
            box = Box(
                (w_index + x[b_index][h_index][w_index]) / grid_w,
                (h_index + y[b_index][h_index][w_index]) / grid_h,
                np.exp(w[b_index][h_index][w_index]) * model.anchors[b_index][0] / grid_w,
                np.exp(h[b_index][h_index][w_index]) * model.anchors[b_index][1] / grid_h,
            )
            iou = box_iou(box, truth_box)
            if iou > best_iou:
                best_box = box
                best_iou = iou

print("best confidences of each grid:")
for i in range(grid_h):
    for j in range(grid_w):
        print("%2d" % (int(conf[:, i, j].max() * 100)), end=" ")
    print()

print("best anchor index of each grid:")
for i in range(grid_h):
    for j in range(grid_w):
        print("%2d" % (conf[:, i, j].argmax()), end=" ")
    print()

b_index, h_index, w_index = np.where(conf==conf.max())
b_index, h_index, w_index = int(b_index), int(h_index), int(w_index)
predicted_box = Box(
    (w_index + x[b_index][h_index][w_index]) / grid_w,
    (h_index + y[b_index][h_index][w_index]) / grid_h,
    np.exp(w[b_index][h_index][w_index]) * model.anchors[b_index][0] / grid_w,
    np.exp(h[b_index][h_index][w_index]) * model.anchors[b_index][1] / grid_h,
)
predicted_iou = box_iou(predicted_box, truth_box)

print("iou of confident bbox: %f" % predicted_iou)
print("iou of best bbox: %f" % best_iou)

# write truth box to image
truth_box.x *= input_width
truth_box.y *= input_height
truth_box.w *= input_width
truth_box.h *= input_height
cv2.rectangle(
    orig_img,
    (int(truth_box.x-truth_box.w/2), int(truth_box.y-truth_box.h/2)), (int(truth_box.x+truth_box.w/2), int(truth_box.y+truth_box.h/2)),
    (0, 255, 0),
    3
)

# write best box to image
best_box.x *= input_width
best_box.y *= input_height
best_box.w *= input_width
best_box.h *= input_height
cv2.rectangle(
    orig_img,
    (int(best_box.x-best_box.w/2), int(best_box.y-best_box.h/2)), (int(best_box.x+best_box.w/2), int(best_box.y+best_box.h/2)),
    (255, 255, 255),
    2
)

# write predicted box to image
predicted_box.x *= input_width
predicted_box.y *= input_height
predicted_box.w *= input_width
predicted_box.h *= input_height
cv2.rectangle(
    orig_img, 
    (int(predicted_box.x-predicted_box.w/2), int(predicted_box.y-predicted_box.h/2)), (int(predicted_box.x+predicted_box.w/2), int(predicted_box.y+predicted_box.h/2)),
    (0, 0, 255),
    1
)

print("save result to ./bbox_result.jpg")
cv2.imwrite("bbox_result.jpg", orig_img)
