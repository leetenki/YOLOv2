import time
import cv2
import numpy as np
from chainer import serializers, Variable
import chainer.functions as F
import argparse
from darknet19 import *

# argument parse
parser = argparse.ArgumentParser(description="指定したパスの画像を読み込み、darknet19でカテゴリ分類を行う")
parser.add_argument('path', help="クラス分類する画像へのパスを指定")
args = parser.parse_args()

# hyper parameters
input_height, input_width = (224, 224)
weight_file = "./backup/darknet19_final.model"
label_file = "./data/label.txt"
image_file = args.path

# read labels
with open(label_file, "r") as f:
    labels = f.read().strip().split("\n")

# read image
print("loading image...")
img = cv2.imread(image_file)
img = cv2.resize(img, (input_height, input_width))
img = np.asarray(img, dtype=np.float32) / 255.0
img = img.transpose(2, 0, 1)

# load model
print("loading model...")
model = Darknet19Predictor(Darknet19())
serializers.load_hdf5(weight_file, model) # load saved model
model.predictor.train = False

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

predicted_order = np.argsort(-y.flatten())
for index in predicted_order:
    cls = labels[index]
    prob = y.flatten()[index] * 100
    print("%16s : %.2f%%" % (cls, prob))
