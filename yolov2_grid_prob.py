import numpy as np
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.links as L
import chainer.functions as F
from lib.utils import *
from lib.functions import *

class YOLOv2GridProb(Chain):

    """
    YOLOv2GridProb
    - It takes (416, 416, 3) sized image as input
    """

    def __init__(self, n_classes, n_boxes):
        super(YOLOv2GridProb, self).__init__(
            ##### common layers for both pretrained layers and yolov2 #####
            conv1  = L.Convolution2D(3, 32, ksize=3, stride=1, pad=1, nobias=True),
            bn1    = L.BatchNormalization(32, use_beta=False),
            bias1  = L.Bias(shape=(32,)),
            conv2  = L.Convolution2D(32, 64, ksize=3, stride=1, pad=1, nobias=True),
            bn2    = L.BatchNormalization(64, use_beta=False),
            bias2  = L.Bias(shape=(64,)),
            conv3  = L.Convolution2D(64, 128, ksize=3, stride=1, pad=1, nobias=True),
            bn3    = L.BatchNormalization(128, use_beta=False),
            bias3  = L.Bias(shape=(128,)),
            conv4  = L.Convolution2D(128, 64, ksize=1, stride=1, pad=0, nobias=True),
            bn4    = L.BatchNormalization(64, use_beta=False),
            bias4  = L.Bias(shape=(64,)),
            conv5  = L.Convolution2D(64, 128, ksize=3, stride=1, pad=1, nobias=True),
            bn5    = L.BatchNormalization(128, use_beta=False),
            bias5  = L.Bias(shape=(128,)),
            conv6  = L.Convolution2D(128, 256, ksize=3, stride=1, pad=1, nobias=True),
            bn6    = L.BatchNormalization(256, use_beta=False),
            bias6  = L.Bias(shape=(256,)),
            conv7  = L.Convolution2D(256, 128, ksize=1, stride=1, pad=0, nobias=True),
            bn7    = L.BatchNormalization(128, use_beta=False),
            bias7  = L.Bias(shape=(128,)),
            conv8  = L.Convolution2D(128, 256, ksize=3, stride=1, pad=1, nobias=True),
            bn8    = L.BatchNormalization(256, use_beta=False),
            bias8  = L.Bias(shape=(256,)),
            conv9  = L.Convolution2D(256, 512, ksize=3, stride=1, pad=1, nobias=True),
            bn9    = L.BatchNormalization(512, use_beta=False),
            bias9  = L.Bias(shape=(512,)),
            conv10 = L.Convolution2D(512, 256, ksize=1, stride=1, pad=0, nobias=True),
            bn10   = L.BatchNormalization(256, use_beta=False),
            bias10 = L.Bias(shape=(256,)),
            conv11 = L.Convolution2D(256, 512, ksize=3, stride=1, pad=1, nobias=True),
            bn11   = L.BatchNormalization(512, use_beta=False),
            bias11 = L.Bias(shape=(512,)),
            conv12 = L.Convolution2D(512, 256, ksize=1, stride=1, pad=0, nobias=True),
            bn12   = L.BatchNormalization(256, use_beta=False),
            bias12 = L.Bias(shape=(256,)),
            conv13 = L.Convolution2D(256, 512, ksize=3, stride=1, pad=1, nobias=True),
            bn13   = L.BatchNormalization(512, use_beta=False),
            bias13 = L.Bias(shape=(512,)),
            conv14 = L.Convolution2D(512, 1024, ksize=3, stride=1, pad=1, nobias=True),
            bn14   = L.BatchNormalization(1024, use_beta=False),
            bias14 = L.Bias(shape=(1024,)),
            conv15 = L.Convolution2D(1024, 512, ksize=1, stride=1, pad=0, nobias=True),
            bn15   = L.BatchNormalization(512, use_beta=False),
            bias15 = L.Bias(shape=(512,)),
            conv16 = L.Convolution2D(512, 1024, ksize=3, stride=1, pad=1, nobias=True),
            bn16   = L.BatchNormalization(1024, use_beta=False),
            bias16 = L.Bias(shape=(1024,)),
            conv17 = L.Convolution2D(1024, 512, ksize=1, stride=1, pad=0, nobias=True),
            bn17   = L.BatchNormalization(512, use_beta=False),
            bias17 = L.Bias(shape=(512,)),
            conv18 = L.Convolution2D(512, 1024, ksize=3, stride=1, pad=1, nobias=True),
            bn18   = L.BatchNormalization(1024, use_beta=False),
            bias18 = L.Bias(shape=(1024,)),

            ###### new layer
            conv19 = L.Convolution2D(1024, 1024, ksize=3, stride=1, pad=1, nobias=True),
            bn19   = L.BatchNormalization(1024, use_beta=False),
            bias19 = L.Bias(shape=(1024,)),
            conv20 = L.Convolution2D(1024, 1024, ksize=3, stride=1, pad=1, nobias=True),
            bn20   = L.BatchNormalization(1024, use_beta=False),
            bias20 = L.Bias(shape=(1024,)),
            conv21 = L.Convolution2D(3072, 1024, ksize=3, stride=1, pad=1, nobias=True),
            bn21   = L.BatchNormalization(1024, use_beta=False),
            bias21 = L.Bias(shape=(1024,)),
            conv22 = L.Convolution2D(1024, n_boxes * n_classes, ksize=1, stride=1, pad=0),
        )
        self.train = False
        self.finetune = False
        self.n_boxes = n_boxes
        self.n_classes = n_classes

    def __call__(self, x):
        ##### common layer
        h = F.leaky_relu(self.bias1(self.bn1(self.conv1(x), test=not self.train, finetune=self.finetune)), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = F.leaky_relu(self.bias2(self.bn2(self.conv2(h), test=not self.train, finetune=self.finetune)), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = F.leaky_relu(self.bias3(self.bn3(self.conv3(h), test=not self.train, finetune=self.finetune)), slope=0.1)
        h = F.leaky_relu(self.bias4(self.bn4(self.conv4(h), test=not self.train, finetune=self.finetune)), slope=0.1)
        h = F.leaky_relu(self.bias5(self.bn5(self.conv5(h), test=not self.train, finetune=self.finetune)), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = F.leaky_relu(self.bias6(self.bn6(self.conv6(h), test=not self.train, finetune=self.finetune)), slope=0.1)
        h = F.leaky_relu(self.bias7(self.bn7(self.conv7(h), test=not self.train, finetune=self.finetune)), slope=0.1)
        h = F.leaky_relu(self.bias8(self.bn8(self.conv8(h), test=not self.train, finetune=self.finetune)), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = F.leaky_relu(self.bias9(self.bn9(self.conv9(h), test=not self.train, finetune=self.finetune)), slope=0.1)
        h = F.leaky_relu(self.bias10(self.bn10(self.conv10(h), test=not self.train, finetune=self.finetune)), slope=0.1)
        h = F.leaky_relu(self.bias11(self.bn11(self.conv11(h), test=not self.train, finetune=self.finetune)), slope=0.1)
        h = F.leaky_relu(self.bias12(self.bn12(self.conv12(h), test=not self.train, finetune=self.finetune)), slope=0.1)
        h = F.leaky_relu(self.bias13(self.bn13(self.conv13(h), test=not self.train, finetune=self.finetune)), slope=0.1)
        high_resolution_feature = reorg(h) # 高解像度特徴量をreorgでサイズ落として保存しておく
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = F.leaky_relu(self.bias14(self.bn14(self.conv14(h), test=not self.train, finetune=self.finetune)), slope=0.1)
        h = F.leaky_relu(self.bias15(self.bn15(self.conv15(h), test=not self.train, finetune=self.finetune)), slope=0.1)
        h = F.leaky_relu(self.bias16(self.bn16(self.conv16(h), test=not self.train, finetune=self.finetune)), slope=0.1)
        h = F.leaky_relu(self.bias17(self.bn17(self.conv17(h), test=not self.train, finetune=self.finetune)), slope=0.1)
        h = F.leaky_relu(self.bias18(self.bn18(self.conv18(h), test=not self.train, finetune=self.finetune)), slope=0.1)

        ###### new layer
        h = F.leaky_relu(self.bias19(self.bn19(self.conv19(h), test=not self.train, finetune=self.finetune)), slope=0.1)
        h = F.leaky_relu(self.bias20(self.bn20(self.conv20(h), test=not self.train, finetune=self.finetune)), slope=0.1)
        h = F.concat((h, high_resolution_feature), axis=1) # output concatnation
        h = F.leaky_relu(self.bias21(self.bn21(self.conv21(h), test=not self.train, finetune=self.finetune)), slope=0.1)
        h = self.conv22(h)

        return h

class YOLOv2GridProbPredictor(Chain):
    def __init__(self, predictor):
        super(YOLOv2GridProbPredictor, self).__init__(predictor=predictor)
        self.anchors = [[5.375, 5.03125], [5.40625, 4.6875], [2.96875, 2.53125], [2.59375, 2.78125], [1.9375, 3.25]]

    def __call__(self, x, t):
        y = self.predictor(x)
        batch_size, _, grid_h, grid_w = y.shape
        y = F.transpose(F.reshape(y, (batch_size, self.predictor.n_boxes, self.predictor.n_classes, grid_h, grid_w)), (0, 2, 1, 3, 4))
        y = F.softmax(y)

        self.abs_anchors = self.anchors / np.array([grid_w, grid_h])
        t_categories = y.data.copy()
        for batch in range(batch_size):
            truth_box = t[batch][0]
            truth_w = int(float(truth_box["x"]) * grid_w)
            truth_h = int(float(truth_box["y"]) * grid_h)
            truth_n = 0
            best_iou = 0.0
            for anchor_index, abs_anchor in enumerate(self.abs_anchors):
                iou = box_iou(Box(0, 0, float(truth_box["w"]), float(truth_box["h"])), Box(0, 0, abs_anchor[0], abs_anchor[1]))
                if best_iou < iou:
                    best_iou = iou
                    truth_n = anchor_index

            print(truth_n, truth_box["w"], truth_box["h"], abs_anchor[0], abs_anchor[1])
            t_categories[batch, :, truth_n, truth_h, truth_w] = 0
            t_categories[batch, int(truth_box["label"]), truth_n, truth_h, truth_w] = 1

            # debug prints
            maps = F.transpose(y[batch], (2, 3, 1, 0)).data
            print("best conditional probability and class for each grid:")
            for maps_h in maps:
                for maps_w in maps_h:
                    print("%2d" % (maps_w[int(maps_w.max(axis=1).argmax())].argmax()), end=' ')
                print("     ", end="")
                for maps_w in maps_h:
                    print("%2d" % (maps_w[int(maps_w.max(axis=1).argmax())].max()*99), end=' ')
                print("")

            print(y[batch, :, truth_n, truth_h, truth_w].data)
            print(truth_box["one_hot_label"])
            print("-------------------------------")

        t_categories = Variable(t_categories)
        t_categories.to_gpu()

        #loss = F.mean_squared_error(y, t_categories)
        loss = sum_of_squared_error(y, t_categories)
        return loss

    def predict(self, x):
        y = self.predictor(x)
        batch_size, _, grid_h, grid_w = y.shape
        y = F.transpose(F.reshape(y, (batch_size, self.predictor.n_boxes, self.predictor.n_classes, grid_h, grid_w)), (0, 2, 1, 3, 4))
        return F.softmax(y)
