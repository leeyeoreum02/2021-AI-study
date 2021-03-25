import json
import cv2
import os
from collections import defaultdict

import matplotlib.pyplot as plt
from typing import Dict, Tuple, Any, List, Union
import numpy as np


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
                    'segmentation': np.array(v['segmentation'])
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
    colors: Dict[str, Tuple[np.ndarray]]
    ) -> np.ndarray:
    for v in annotations:
        bbox = v['bbox']
        x, y, w, h = list(map(round, bbox))
        cv2.rectangle(image, (x, y), (x + w, y + h), colors[v['category_id']], 3)
        cv2.putText(image, v['category_id'], (x - 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    return image

def draw_masks(
    image: np.ndarray,
    annotations: List[Dict[str, Any]],
    colors: Dict[str, Tuple[np.ndarray]]
    ) -> np.ndarray:
    for v in annotations:
        for seg in v['segmentation']:
            x_segs = [round(loc) for i, loc in enumerate(seg)
                        if i % 2 == 0]
            y_segs = [round(loc) for i, loc in enumerate(seg)
                        if i % 2 != 0]
            zip_segs = list(zip(x_segs, y_segs))
            zip_segs = np.array(zip_segs)

            cv2.fillConvexPoly(image, zip_segs, colors[v['category_id']])
            cv2.addWeighted(image, 1, target, 0.5, 0)
            

    return image

def get_colors(annotations: List[Dict[str, Any]]) -> Dict[str, Tuple[np.ndarray]]:
    return {v['category_id']: tuple(map(int, np.random.randint(0, 255, 3))) for v in annotations}
    

if __name__ == '__main__':
    coco = CocoDataset('../coco/annotations/instances_train2014.json')
    print(coco[0])

    root = os.path.join('..', 'coco')
    image_path = os.path.join(root, 'train2014', coco[0][0])

    image = cv2.imread(image_path)
    colors = get_colors(coco[0][1])
    image = draw_bbox(image, coco[0][1], colors)
    image = draw_masks(image, coco[0][1], colors)

    fig, ax = plt.subplots(dpi=200)
    ax.imshow(image)
    ax.axis('off')
    plt.show()
    fig.savefig('coco[0].png')