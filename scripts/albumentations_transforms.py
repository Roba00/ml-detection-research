import torch
import albumentations as A
from PIL import Image, ImageDraw, ImageColor
import numpy as np
from src.datasets.cell_dataset import CellDataset

# Request image from dataset
full_dataset = CellDataset("C:/Users/abbaj/Desktop/Research/IDCIA v2/images/", "C:/Users/abbaj/Desktop/Research/IDCIA v2/bounding_boxes_faster_rcnn/")
idx = int(input(f"Index (0 to {len(full_dataset) - 1})? "))

# Get image and bounding boxes
batch = full_dataset.__getitem__(idx)
image : np.ndarray = torch.Tensor(batch[0]).numpy()[0]
bboxes : np.ndarray = torch.Tensor(batch[1]).numpy()

# Show image and bounding boxes without augmentations
pillow_base_image_array = ((image - image.min()) / image.max()) * 255 # [0, 255] Color Range
pillow_base_image = Image.fromarray(pillow_base_image_array)
draw = ImageDraw.Draw(pillow_base_image)
for bbox in bboxes:
    (x_min, y_min, x_max, y_max) = bbox
    draw.rectangle(xy=[x_min, y_min, x_max, y_max], outline="red")
pillow_base_image.show()

# Composition of different transforms to try
transform = A.Compose([
    A.RandomCrop(width=256, height=256, p=0.75), # Think about doing random width * heights
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomResizedCrop(width=256, height=256, p=0.5)
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=[]))

# Show image with augmentations
transformed = transform(image=image, bboxes=bboxes)
transformed_image: np.ndarray = transformed['image']
transformed_bboxes = transformed['bboxes']
pillow_image_array = ((transformed_image - transformed_image.min()) / transformed_image.max()) * 255 # [0, 255] Color Range
pillow_image = Image.fromarray(pillow_image_array)
draw = ImageDraw.Draw(pillow_image)
for bbox in transformed_bboxes:
    (x_min, y_min, x_max, y_max) = bbox
    draw.rectangle(xy=[x_min, y_min, x_max, y_max], outline="red")
pillow_image.show()
