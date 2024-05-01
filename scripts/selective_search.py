# Author: Abdurahman Ali Mohammed (Github: abdumhmd)

import os
import matplotlib.pyplot as plt
import cv2
import glob2
import numpy as np
from PIL import Image,ImageOps
from tqdm import tqdm

all_images=glob2.glob('IDCIA/images/DAPI/' + "*.tiff")
print(all_images)


def get_bbox(im_path):
    # print(f"Image path: {im_path}")
    im = Image.open(im_path).convert('RGB')


    im = ImageOps.autocontrast(im)

    img = np.asarray(im)

   
    img_copy = img.copy()

 
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()


    ss.setBaseImage(img)

    ss.switchToSelectiveSearchQuality()

    rects = ss.process()


    numShowRects = 500

    # print(f"Total Number of Region Proposals: {len(rects)}")
   

    def iou(rect1, rect2):
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        x1_max, y1_max = x1+w1, y1+h1
        x2_max, y2_max = x2+w2, y2+h2
        x_overlap = max(0, min(x1_max, x2_max) - max(x1, x2))
        y_overlap = max(0, min(y1_max, y2_max) - max(y1, y2))
        intersection = x_overlap * y_overlap
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        return intersection / union

    def dice(rect1, rect2):
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        x1_max, y1_max = x1+w1, y1+h1
        x2_max, y2_max = x2+w2, y2+h2
        x_overlap = max(0, min(x1_max, x2_max) - max(x1, x2))
        y_overlap = max(0, min(y1_max, y2_max) - max(y1, y2))
        intersection = x_overlap * y_overlap
        area1 = w1 * h1
        area2 = w2 * h2
        return 2 * intersection / (area1 + area2)


    new_rects = []
    for i, rect in enumerate(rects):
        x, y, w, h = rect
        # check if the rectangle is already in the new list
        if not any([dice(rect, new_rect) > 0.50 for new_rect in new_rects]):
            new_rects.append(rect)  


    # print(f"Total Number of Region Proposals after merging: {len(new_rects)}")
    # print(f"Total Number of Region Proposals before merging: {len(rects)}")

    # check if a box contains more than one boxes inside it
    def contains(rect1, rect2):
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        x1_max, y1_max = x1+w1, y1+h1
        x2_max, y2_max = x2+w2, y2+h2
        return x1 <= x2 and y1 <= y2 and x1_max >= x2_max and y1_max >= y2_max


    new_rects2 = []
    for i, rect in enumerate(new_rects):
        x, y, w, h = rect

        if not any([contains(rect, new_rect) for new_rect in new_rects2]):
            new_rects2.append(rect)


    # print(f"Total Number of Region Proposals after removing boxes that contain more than one boxes inside it: {len(new_rects2)}")


    with open(im_path.replace('.tiff', '.csv'), 'a') as f:
        
        f.write(f"x1,y1,x2,y2\n")

        for rect in new_rects2:
           
            x, y, w, h = rect
            f.write(f"{x},{y},{w+x},{h+y}\n")

for im_path in all_images:
    get_bbox(im_path)