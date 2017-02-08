import numpy as np
from lib.utils import *
import glob

# hyper parameters
label_path = "../dataset/yolov2_fruits_dataset/labels/"
n_anchors = 5
loss_convergence = 1e-5
image_width = 416
image_height = 416
grid_width = 13
grid_height = 13

boxes = []
label_files = glob.glob("%s/*" % label_path)
for label_file in label_files:
    with open(label_file, "r") as f:
        label, x, y, w, h = f.read().strip().split(" ")
        boxes.append(Box(0, 0, float(w), float(h)))

# initial centroids
centroid_indices = np.random.choice(len(boxes), n_anchors)
centroids = []
for centroid_index in centroid_indices:
    centroids.append(boxes[centroid_index])

# do k-means
def do_kmeans(n_anchors, boxes, centroids):
    loss = 0
    groups = []
    new_centroids = []
    for i in range(n_anchors):
        groups.append([])
        new_centroids.append(Box(0, 0, 0, 0))

    for box in boxes:
        min_distance = 1
        group_index = 0
        for centroid_index, centroid in enumerate(centroids):
            distance = (1 - box_iou(box, centroid))
            if distance < min_distance:
                min_distance = distance
                group_index = centroid_index
        groups[group_index].append(box)
        loss += min_distance
        new_centroids[group_index].w += box.w
        new_centroids[group_index].h += box.h

    for i in range(n_anchors):
        new_centroids[i].w /= len(groups[i])
        new_centroids[i].h /= len(groups[i])

    return new_centroids, groups, loss

# iterate k-means
new_centroids, groups, old_loss = do_kmeans(n_anchors, boxes, centroids)
while(True):
    new_centroids, groups, loss = do_kmeans(n_anchors, boxes, new_centroids)
    print("loss = %f" % loss)
    if abs(old_loss - loss) < loss_convergence:
        break
    old_loss = loss

# print result
for centroid in centroids:
    print(centroid.w * grid_width, centroid.h * grid_height)
