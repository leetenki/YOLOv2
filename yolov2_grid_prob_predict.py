import time
import cv2
import numpy as np
from chainer import serializers, Variable
import chainer.functions as F
import argparse
from yolov2_grid_prob import *

# argument parse
parser = argparse.ArgumentParser(description="指定したパスの画像を読み込み、各gridの条件付き確率を求める")
parser.add_argument('path', help="画像ファイルへのパスを指定")
args = parser.parse_args()

# hyper parameters
input_height, input_width = (416, 416)
weight_file = "./backup/yolov2_grid_prob_final.model"
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
img = cv2.resize(orig_img, (input_height, input_width))
img = np.asarray(img, dtype=np.float32) / 255.0
img = img.transpose(2, 0, 1)

# load model
print("loading model...")
model = YOLOv2GridProbPredictor(YOLOv2GridProb(n_classes=n_classes, n_boxes=n_boxes))
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

y = model.predict(x).data
if hasattr(cuda, "cupy"):
    y = y.get()

batch_size, _, n_boxes, grid_h, grid_w = y.shape
grid_height = int(orig_img.shape[0] / grid_h)
grid_width = int(orig_img.shape[1] / grid_w)
colors = [(255, 0, 0), (0, 125, 125), (0, 255, 0), (125, 0, 255), (125, 125, 0), (125, 0, 125), (0, 0, 255), (0, 255, 125), (125, 125, 125), (255, 125, 0)]
y = F.transpose(y, (0, 3, 4, 2, 1)).data.reshape(grid_h, grid_w, n_boxes, -1)
overlay = orig_img.copy()
output = orig_img.copy()
for h_index, h in enumerate(y):
    for w_index, w in enumerate(h):
        prediction = w[int(w.max(axis=1).argmax())].argmax()
        cv2.rectangle(overlay, (grid_width*w_index, grid_height*h_index), (grid_width*(w_index+1), grid_height*(h_index+1)), colors[prediction], -1)

        print(prediction, end=' ')
    print("")
print(labels)

cv2.addWeighted(overlay, 0.3, output, 0.7, 0, output)
cv2.imwrite("grid_prob_result.jpg", output)
