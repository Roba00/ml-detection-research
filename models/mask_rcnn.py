from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models.detection.mask_rcnn import (
    MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn_v2)
from torchvision.transforms import ConvertImageDtype
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_bounding_boxes


def init() -> None:
    global model, params, optimizer
    model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=5e-3, momentum=0.9, weight_decay=5e-4)
    global initialized
    initialized = True

def loss_fn(loss_dict : dict[str, torch.Tensor]) -> torch.Tensor | Literal[0]:
    return sum(loss for loss in loss_dict.values())

def eval_loss_fn(loss_dict : dict[str, torch.Tensor]) -> torch.Tensor | Literal[0]:
    return sum(loss for loss in loss_dict['scores'].values())

# One training step
def train_step(images: torch.Tensor, 
               imageAnnotations: torch.Tensor) -> dict[str, torch.Tensor]:
    targets = []
    
    for i in range(len(images)):
        d = {}
        d['boxes'] = imageAnnotations[i]

        # Number of elements
        n = imageAnnotations[i].size(axis=0)

        # Note: Labels are not to be used in this model, since only one class is used.
        d['labels'] = torch.zeros(n, dtype=int)

        d['masks'] = torch.rand(n, 10, 10)

        targets.append(d)

    model.train()
    loss_dict = model(images, targets)
    # print("Training output:", loss_dict)
    return loss_dict


# One testing step
def test_step(images: list[torch.Tensor], 
              imageAnnotations: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    targets = []
    for i in range(len(images)):
        d = {}
        d['boxes'] = imageAnnotations[i]

        # Number of elements
        n = imageAnnotations[i].size(axis=0)

        # Note: Labels are not to be used in this model, since only one class is used.
        d['labels'] = torch.zeros(n, dtype=int)

        d['masks'] = torch.rand(n, 10, 10)

        targets.append(d)
    
    model.eval()
    loss_dict = model(images, targets)
    # print('Testing output:', predictions)
    return loss_dict

# Train the model
def train_loop(train_loader: DataLoader, num_epochs: int) -> None:
    print("\nBeginning Training...")

    loss_hist = 0
    
    for epoch in range(num_epochs):
        loss_hist = 0

        for (itr, batch) in enumerate(train_loader):
            images = torch.Tensor(batch[0])
            annotations = torch.Tensor(batch[1])
            loss_dict = train_step(images, annotations)

            losses = loss_fn(loss_dict)
            loss_val = losses.item()

            loss_hist += loss_val

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if itr % 50 == 0:
                print(f"Iteration {itr} loss: {loss_val}")
        
        print(f"Epoch {epoch} loss: {loss_hist}")


# Test the model
def test_loop(test_loader: DataLoader) -> None:
    print("\nBeginning Testing...")
    
    total_loss = 0.0

    for (itr, batch) in enumerate(test_loader):
        print("\nBatch ", itr)
        images = torch.Tensor(batch[0])
        annotations = torch.Tensor(batch[1])
        loss_dict = test_step(images, annotations)
        # visualize_predictions(img=images[0], boxes=loss_dict[0]['boxes'])

        losses = eval_loss_fn(loss_dict)
        loss_val = losses.item()
        total_loss += loss_val

        print(f"Iteration {itr} loss: {loss_val}")
    
    print(f"Total loss: {total_loss}")


# Visualize the predictions
def visualize_predictions(img: torch.Tensor, boxes: torch.Tensor) -> None:
    img = ConvertImageDtype(dtype=torch.uint8)(img)
    box = draw_bounding_boxes(img, boxes=boxes, width=4)
    im = to_pil_image(box.detach())
    im.show()


# Save model
def save_model(model_path: str = 'src/weights/mask_rcnn/rename_weight.pt'):
    torch.save(model.state_dict(), model_path)


# Load model
def load_model(model_path: str = 'src/weights/mask_rcnn/rename_weight.pt'):
    model.load_state_dict(torch.load(model_path))