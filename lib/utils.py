import chainer.functions as F
import numpy as np
import cv2
from chainer import Variable

def print_cnn_info(name, link, shape_before, shape_after, time):
    n_stride = (
        int((shape_before[2] + link.pad[0] * 2 - link.ksize) / link.stride[0]) + 1,
        int((shape_before[3] + link.pad[1] * 2 - link.ksize) / link.stride[1]) + 1
    )

    cost = n_stride[0] * n_stride[1] * shape_before[1] * link.ksize * link.ksize * link.out_channels

    print('%s(%d × %d, stride=%d, pad=%d) (%d x %d x %d) -> (%d x %d x %d) (cost=%d): %.6f[sec]' % 
        (
            name, link.W.shape[2], link.W.shape[3], link.stride[0], link.pad[0],
            shape_before[2], shape_before[3], shape_before[1], shape_after[2], shape_after[3], shape_after[1],
            cost, time
        )
    )

    return cost

def print_pooling_info(name, filter_size, stride, pad, shape_before, shape_after, time):
    n_stride = (
        int((shape_before[2] - filter_size) / stride) + 1,
        int((shape_before[3] - filter_size) / stride) + 1
    )
    cost = n_stride[0] * n_stride[1] * shape_before[1] * filter_size * filter_size * shape_after[1]

    print('%s(%d × %d, stride=%d, pad=%d) (%d x %d x %d) -> (%d x %d x %d) (cost=%d): %.6f[sec]' % 
        (name, filter_size, filter_size, stride, pad, shape_before[2], shape_before[3], shape_before[1], shape_after[2], shape_after[3], shape_after[1], cost, time)
    )

    return cost

def print_fc_info(name, link, time):
    import pdb
    cost = link.W.shape[0] * link.W.shape[1]
    print('%s %d -> %d (cost = %d): %.6f[sec]' % (name, link.W.shape[1], link.W.shape[0], cost, time))

    return cost

# x, y, w, hの4パラメータを保持するだけのクラス
class Box():
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def int_left_top(self):
        half_width = self.w / 2
        half_height = self.h / 2
        return (int(round(self.x - half_width)), int(round(self.y - half_height)))

    def left_top(self):
        half_width = self.w / 2
        half_height = self.h / 2
        return [self.x - half_width, self.y - half_height]

    def int_right_bottom(self):
        half_width = self.w / 2
        half_height = self.h / 2
        return (int(round(self.x + half_width)), int(round(self.y + half_height)))

    def right_bottom(self):
        half_width = self.w / 2
        half_height = self.h / 2
        return [self.x + half_width, self.y + half_height]

    def crop_region(self, h, w):
        left, top = self.left_top()
        right, bottom = self.right_bottom()
        left = max(0, left)
        top = max(0, top)
        right = min(w, right)
        bottom = min(h, bottom)
        self.w = right - left
        self.h = bottom - top
        self.x = (right + left) / 2
        self.y = (bottom + top) / 2
        return self

# 2本の線の情報を受取り、被ってる線分の長さを返す。あくまで線分
def overlap(x1, len1, x2, len2):
    len1_half = len1/2
    len2_half = len2/2

    left = max(x1 - len1_half, x2 - len2_half)
    right = min(x1 + len1_half, x2 + len2_half)

    return right - left

# chainerのVariable用のoverlap
def multi_overlap(x1, len1, x2, len2):
    len1_half = len1/2
    len2_half = len2/2

    left = F.maximum(x1 - len1_half, x2 - len2_half)
    right = F.minimum(x1 + len1_half, x2 + len2_half)

    return right - left

# 2つのboxを受け取り、被ってる面積を返す(intersection of 2 boxes)
def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w)
    h = overlap(a.y, a.h, b.y, b.h)
    if w < 0 or h < 0:
        return 0

    area = w * h
    return area

# chainer用
def multi_box_intersection(a, b):
    w = multi_overlap(a.x, a.w, b.x, b.w)
    h = multi_overlap(a.y, a.h, b.y, b.h)
    zeros = Variable(np.zeros(w.shape, dtype=w.data.dtype))
    zeros.to_gpu()

    w = F.maximum(w, zeros)
    h = F.maximum(h, zeros)

    area = w * h
    return area

# 2つのboxを受け取り、合計面積を返す。(union of 2 boxes)
def box_union(a, b):
    i = box_intersection(a, b)
    u = a.w * a.h + b.w * b.h - i
    return u

# chianer用
def multi_box_union(a, b):
    i = multi_box_intersection(a, b)
    u = a.w * a.h + b.w * b.h - i
    return u

# compute iou
def box_iou(a, b):
    return box_intersection(a, b) / box_union(a, b)

# chainer用
def multi_box_iou(a, b):
    return multi_box_intersection(a, b) / multi_box_union(a, b)


# 画像を読み込んで、hue, sat, val空間でランダム変換を加える関数
def random_hsv_image(bgr_image, delta_hue, delta_sat_scale, delta_val_scale):
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV).astype(np.float32)

    # hue
    hsv_image[:, :, 0] += int((np.random.rand() * delta_hue * 2 - delta_hue) * 255)

    # sat
    sat_scale = 1 + np.random.rand() * delta_sat_scale * 2 - delta_sat_scale
    hsv_image[:, :, 1] *= sat_scale

    # val
    val_scale = 1 + np.random.rand() * delta_val_scale * 2 - delta_val_scale
    hsv_image[:, :, 2] *= val_scale

    hsv_image[hsv_image < 0] = 0 
    hsv_image[hsv_image > 255] = 255 
    hsv_image = hsv_image.astype(np.uint8)
    bgr_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return bgr_image

# non maximum suppression
def nms(predicted_results, iou_thresh):
    nms_results = []
    for i in range(len(predicted_results)):
        overlapped = False
        for j in range(i+1, len(predicted_results)):
            if box_iou(predicted_results[i]["box"], predicted_results[j]["box"]) > iou_thresh:
                overlapped = True
                if predicted_results[i]["objectness"] > predicted_results[j]["objectness"]:
                    temp = predicted_results[i]
                    predicted_results[i] = predicted_results[j]
                    predicted_results[j] = temp
        if not overlapped:
            nms_results.append(predicted_results[i])
    return nms_results

# reshape to yolo size
def reshape_to_yolo_size(img):
    input_height, input_width, _ = img.shape
    min_pixel = 320
    #max_pixel = 608
    max_pixel = 448

    min_edge = np.minimum(input_width, input_height)
    if min_edge < min_pixel:
        input_width *= min_pixel / min_edge
        input_height *= min_pixel / min_edge
    max_edge = np.maximum(input_width, input_height)
    if max_edge > max_pixel:
        input_width *= max_pixel / max_edge
        input_height *= max_pixel / max_edge

    input_width = int(input_width / 32 + round(input_width % 32 / 32)) * 32
    input_height = int(input_height / 32 + round(input_height % 32 / 32)) * 32
    img = cv2.resize(img, (input_width, input_height))

    return img
