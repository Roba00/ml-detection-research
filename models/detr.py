import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import dataloader
from transformers import DetrForObjectDetection
from torchvision.transforms import ConvertImageDtype
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_bounding_boxes
from tqdm import tqdm
from tqdm.contrib import tenumerate

def init() -> None:
    global model, params, optimizer
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=5e-3, momentum=0.9, weight_decay=5e-4)
    global initialized
    initialized = True


def loss_fn(loss_dict : dict) -> float:
    return sum(loss for loss in loss_dict.values())

# One training step
def train_step(images: torch.Tensor, 
               imageAnnotations: torch.Tensor) -> dict[str, torch.Tensor]:
    targets = []
    
    for i in range(len(images)):
        d = {}
        d['boxes'] = imageAnnotations[i]

        # Note: Labels are not to be used in this model, since only one class is used.
        d['labels'] = torch.zeros(imageAnnotations[i].size(axis=0), dtype=int)

        targets.append(d)

    # print(images)

    # TODO: Run this on a better computer, gets stuck here.
    print("Train step start")
    loss_dict = model(images, targets)
    print("Train step end")
    # print("Training output:", loss_dict)
    return loss_dict


# One testing step
def test_step(images: list[torch.Tensor], 
              imageAnnotations: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    targets = []
    for i in range(len(images)):
        d = {}
        d['boxes'] = imageAnnotations[i]

        # Note: Labels are not to be used in this model, since only one class is used.
        d['labels'] = torch.zeros(imageAnnotations[i].size(axis=0), dtype=int)

        targets.append(d)
    
    loss_dict = model(images, targets)
    # print('Testing output:', predictions)
    return loss_dict


# Train the model
def train_loop(train_loader: DataLoader, num_epochs: int) -> None:
    print("\nBeginning Training...")

    loss_hist = 0
    model.train()
    
    for epoch in tqdm(range(num_epochs)):
        loss_hist = 0
        # print(train_loader.dataset.__getitem__(0))
        for (itr, batch) in tenumerate(train_loader):
            images = batch[0]
            annotations = batch[1]
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
    model.eval()

    for (itr, batch) in tenumerate(test_loader):
        print("\nBatch ", itr)
        images = torch.Tensor(batch[0])
        annotations = torch.Tensor(batch[1])
        loss_dict = test_step(images, annotations)
        # visualize_predictions(img=images[0], boxes=loss_dict[0]['boxes'])

        losses = loss_fn(loss_dict)
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
def save_model(model_path: str = 'src/weights/detr/rename_weight.pt'):
    torch.save(model.state_dict(), model_path)


# Load model
def load_model(model_path: str = 'src/weights/detr/rename_weight.pt'):
    model.load_state_dict(torch.load(model_path))
