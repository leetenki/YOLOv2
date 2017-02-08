import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
import argparse
from lib.utils import *
from lib.image_generator import *
from yolov2 import *

parser = argparse.ArgumentParser(description="指定したパスのweightsファイルを読み込み、chainerモデルへ変換する")
parser.add_argument('file', help="オリジナルのyolov2のweightsファイルへのパスを指定")
args = parser.parse_args()

print("loading", args.file)
file = open(args.file, "rb")
dat=np.fromfile(file, dtype=np.float32)[4:] # skip header(4xint)

# load model
print("loading initial model...")
n_classes = 80
n_boxes = 5
last_out = (n_classes + 5) * n_boxes

yolov2 = YOLOv2(n_classes=n_classes, n_boxes=n_boxes)
yolov2.train = True
yolov2.finetune = False

layers=[
    [3, 32, 3], 
    [32, 64, 3], 
    [64, 128, 3], 
    [128, 64, 1], 
    [64, 128, 3], 
    [128, 256, 3], 
    [256, 128, 1], 
    [128, 256, 3], 
    [256, 512, 3], 
    [512, 256, 1], 
    [256, 512, 3], 
    [512, 256, 1], 
    [256, 512, 3], 
    [512, 1024, 3], 
    [1024, 512, 1], 
    [512, 1024, 3], 
    [1024, 512, 1], 
    [512, 1024, 3], 
    [1024, 1024, 3], 
    [1024, 1024, 3], 
    [3072, 1024, 3], 
]

offset=0
for i, l in enumerate(layers):
    in_ch = l[0]
    out_ch = l[1]
    ksize = l[2]

    # load bias(Bias.bはout_chと同じサイズ)
    txt = "yolov2.bias%d.b.data = dat[%d:%d]" % (i+1, offset, offset+out_ch)
    offset+=out_ch
    exec(txt)

    # load bn(BatchNormalization.gammaはout_chと同じサイズ)
    txt = "yolov2.bn%d.gamma.data = dat[%d:%d]" % (i+1, offset, offset+out_ch)
    offset+=out_ch
    exec(txt)

    # (BatchNormalization.avg_meanはout_chと同じサイズ)
    txt = "yolov2.bn%d.avg_mean = dat[%d:%d]" % (i+1, offset, offset+out_ch)
    offset+=out_ch
    exec(txt)

    # (BatchNormalization.avg_varはout_chと同じサイズ)
    txt = "yolov2.bn%d.avg_var = dat[%d:%d]" % (i+1, offset, offset+out_ch)
    offset+=out_ch
    exec(txt)

    # load convolution weight(Convolution2D.Wは、outch * in_ch * フィルタサイズ。これを(out_ch, in_ch, 3, 3)にreshapeする)
    txt = "yolov2.conv%d.W.data = dat[%d:%d].reshape(%d, %d, %d, %d)" % (i+1, offset, offset+(out_ch*in_ch*ksize*ksize), out_ch, in_ch, ksize, ksize)
    offset+= (out_ch*in_ch*ksize*ksize)
    exec(txt)
    print(i+1, offset)

# load last convolution weight(BiasとConvolution2Dのみロードする)
in_ch = 1024
out_ch = last_out
ksize = 1

txt = "yolov2.bias%d.b.data = dat[%d:%d]" % (i+2, offset, offset+out_ch)
offset+=out_ch
exec(txt)

txt = "yolov2.conv%d.W.data = dat[%d:%d].reshape(%d, %d, %d, %d)" % (i+2, offset, offset+(out_ch*in_ch*ksize*ksize), out_ch, in_ch, ksize, ksize)
offset+=out_ch*in_ch*ksize*ksize
exec(txt)
print(i+2, offset)

print("save weights file to yolov2_darknet.model")
serializers.save_hdf5("yolov2_darknet.model", yolov2)
