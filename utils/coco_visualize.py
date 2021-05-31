import json
import cv2
import os
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Sequence, Callable
import numpy as np


with open('../coco/annotations/instances_train2014.json', 'r') as json_file:
    json2py = json.load(json_file)

print(json2py['annotations'][0]['bbox'])
print(json2py['annotations'][0]['image_id'])
print(json2py['annotations'][0]['segmentation'][0])

targets = []
for idx in range(len(json2py['annotations'])):
    if json2py['annotations'][idx]['image_id'] == 480023:
        targets.append(json2py['annotations'][idx])

print('targets:', targets)

categories = set(target['category_id'] for target in targets)
print('categories:', categories)

root = os.path.join('..', 'coco')
image_path = os.path.join(root, 'train2014', 'COCO_train2014_000000480023.jpg')
image = cv2.imread(image_path)
print(type(image))

def make_bbox(target, image, color):
    bbox = target['bbox']
    x, y, w, h = list(map(round, bbox))

    image = cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)
    

for target in targets:
    bbox = target['bbox']
    x, y, w, h = list(map(round, bbox))
    
    color = list(np.random.random(size=3) * 256)
    image = cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)

    x_segs = []
    y_segs = []
    for i, loc in enumerate(target['segmentation'][0]):
        if i % 2 == 0:
            x_segs.append(loc)
        else:
            y_segs.append(loc)
    segs = list(zip(x_segs, y_segs))

    color = list(np.random.random(size=3) * 256)
    for i, seg_loc in enumerate(segs):
        if i == 0:
            start_loc = tuple(map(round, seg_loc))
        elif i == 1:
            end_loc = tuple(map(round, seg_loc))
            cv2.line(image, start_loc, end_loc, color, 3)
            start_loc = end_loc
        else:
            end_loc = tuple(map(round, seg_loc))
            cv2.line(image, start_loc, end_loc, color, 3)
            start_loc = end_loc
    start_loc = tuple(map(round, segs[-1]))
    end_loc = tuple(map(round, segs[0]))
    cv2.line(image, start_loc, end_loc, color, 3)

fig, ax = plt.subplots(dpi=200)
ax.imshow(image)
ax.axis('off')
fig.savefig('coco_train_86.png')
plt.show()