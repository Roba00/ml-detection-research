import glob2
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from src.datasets.cell_dataset import CellDataset
from torchvision.transforms import ConvertImageDtype
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_bounding_boxes


def visualize_predictions(img: torch.Tensor, 
                          boxes: torch.Tensor) -> None:
    img = ConvertImageDtype(dtype=torch.uint8)(img)
    box = draw_bounding_boxes(img, boxes=boxes, width=4)
    im = to_pil_image(box.detach())
    im.show()

full_dataset = CellDataset("C:/Users/abbaj/Desktop/Research/IDCIAv2/images/", "C:/Users/abbaj/Desktop/Research/IDCIAv2/bounding_boxes_faster_rcnn_larger/")
idx = int(input(f"Index (0 to {len(full_dataset) - 1})? "))
batch = full_dataset.__getitem__(idx)
image = torch.Tensor(batch[1])
annotation = torch.Tensor(batch[2])
visualize_predictions(image, annotation)
