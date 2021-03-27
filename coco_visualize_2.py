import json
import cv2
import os
from collections import defaultdict

import matplotlib.pyplot as plt
from typing import Dict, Tuple, Any, List, Union
import numpy as np
import random

from pycocotools import mask
from skimage import measure


class CocoDataset:
    def __init__(self, path: os.PathLike) -> None:
        with open(path, 'r') as json_file:
            data = json.load(json_file)
        
        self.images = [{'id': v['id'], 'file_name': v['file_name']} for v in data['images']]
        self.categories = {v['id']: v['name'] for v in data['categories']}
        
        self.annotations = defaultdict(list)
        for v in data['annotations']:
            bbox = np.array(v['bbox'])
            self.annotations[v['image_id']].append(
                {
                    'id': v['id'],
                    'category_id': self.categories.get(v['category_id']),
                    'bbox': bbox,
                    'segmentation': np.array(v['segmentation'], dtype=object),
                    'iscrowd': v['iscrowd'],
                }
            )
    def __getitem__(self, index: int) -> Tuple[str, List[Dict[str, Any]]]:
        values = self.images[index]
        annotations = self.annotations.get(values['id'])
        file_name = values['file_name']

        return file_name, annotations

    def __len__(self) -> int:
        return len(self.images)


def draw_bbox(
    image: np.ndarray,
    annotations: List[Dict[str, Any]],
    colors: Dict[str, Tuple[int]]
    ) -> np.ndarray:
    for v in annotations:
        bbox = v['bbox']
        x, y, w, h = list(map(round, bbox))
        cv2.rectangle(image, (x, y), (x + w, y + h), colors[v['category_id']], 2)
        cv2.putText(image, v['category_id'], (x + 1, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    return image

def draw_masks(
    image: np.ndarray,
    annotations: List[Dict[str, Any]],
    colors: Dict[str, Tuple[int]]
    ) -> np.ndarray:
    for v in annotations:
        # print('\nannotations:', annotations)
        # print('\nv:', v)
        # print("\nv['segmentation']:", v['segmentation'])
        for seg in v['segmentation']:
            x_segs = [round(loc) for i, loc in enumerate(seg)
                        if i % 2 == 0]
            y_segs = [round(loc) for i, loc in enumerate(seg)
                        if i % 2 != 0]
            zip_segs = list(zip(x_segs, y_segs))
            zip_segs = np.array(zip_segs)

            mask = np.zeros_like(image)
            mask = cv2.fillPoly(mask, [zip_segs], colors[v['category_id']])
            image = cv2.addWeighted(image, 1, mask, 0.8, 0)
            
    return image

def get_colors(annotations: List[Dict[str, Any]]) -> Dict[str, Tuple[int]]:
    return {v['category_id']: tuple(map(int, np.random.randint(0, 255, 3))) for v in annotations}

def show_fig_1x1(image: np.ndarray, file_name: str) -> None:
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(file_name)
    plt.axis('off')
    plt.show()
    plt.savefig('coco_random_idx_1x1.png')

def show_fig_4x4(images: List[np.ndarray], file_names: List[str]) -> None:
    fig = plt.figure()
    rows = 2
    cols = 2
    i = 1

    for image, file_name in list(zip(images, file_names)):
        ax = fig.add_subplot(rows, cols, i)
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title(file_name, fontsize=10)
        ax.axis('off')
        i += 1
    
    plt.show()
    fig.savefig('coco_random_idx_4x4.png')

def rle2loc(segmentation: np.ndarray) -> List[float]:
    segmentation = segmentation.tolist()
    compressed_rle = mask.frPyObjects(
        segmentation, 
        segmentation.get('size')[0], 
        segmentation.get('size')[1]
        )
    ground_truth_binary_mask = mask.decode(compressed_rle)
    contours = measure.find_contours(ground_truth_binary_mask, 0.5)
    segmentations = []
    for contour in contours:
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        segmentations.append(segmentation)

    return segmentations
        

if __name__ == '__main__':
    coco = CocoDataset('../coco/annotations/instances_train2014.json')
    
    num_images = list(range(len(coco)))
    random.shuffle(num_images)
    random_idxs = num_images[:4]

    for i, idx in enumerate(random_idxs):
        print(f'{i + 1}. type of annotations:', type(coco[idx][1]))
        for j, annotation in enumerate(coco[idx][1]):
            #print('annotation:', annotation)
            #print('segmenation:', annotation['segmentation'])
            print('category_id:', annotation['category_id'])
            print('type of segmentation:', type(annotation['segmentation']))
            print('type of bbox:', type(annotation['bbox']))
            print('iscrowd:', annotation['iscrowd'])
            #print('type of iscrowd:', type(annotation['iscrowd']))
            print()
            
            if annotation['iscrowd'] == 1:
                coco[idx][1][j]['segmentation'] = rle2loc(annotation['segmentation'])
                print('new:', coco[idx][1][j]['segmentation'])

    image_list = []
    file_names = []
    for idx in random_idxs:
        file_name = coco[idx][0]
        annotations = coco[idx][1]
        
        root = os.path.join('..', 'coco')
        image_path = os.path.join(root, 'train2014', file_name)

        image = cv2.imread(image_path)
        
        if type(annotations) is None:
            print('There is a NoneType in annotations')
            continue
        else:
            colors = get_colors(annotations)
            image = draw_masks(image, annotations, colors)
            image = draw_bbox(image, annotations, colors)

        image_list.append(image)
        file_names.append(file_name)
        
        colors = None

    show_fig_4x4(image_list, file_names)