import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import ConvertImageDtype, ToTensor
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_bounding_boxes
from PIL import Image
from tqdm.auto import tqdm
import glob2

# Requirements for Model
# ----------------------------------------------------------------------------------------------------------------------
# URL: https://docs.ultralytics.com/yolov5/tutorials/pytorch_hub_model_loading/#load-yolov5-with-pytorch-hub
# - Images: List of tensors, each of shape [C, H, W], one for each image, and should be in 0-1 range.
# - Target: Contains Boxes and Labels
#   - Boxes (FloatTensor[N, 4]): The ground-truth boxes in [x1, y1, x2, y2] format
#       - 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.
#   - Labels (Int64Tensor[N]): The class label for each ground-truth box
#       - Should be set to 0 for all

def init() -> None:
    global model, params, optimizer
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', autoshape=False, pretrained=True)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    global initialized
    initialized = True

def loss_fn(loss_dict : dict) -> float:
    return sum(loss for loss in loss_dict.values())



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
        d['labels'] = torch.ones(imageAnnotations[i].size(axis=0), dtype=int)

        targets.append(d)

    loss_dict = model(images, targets)
    #print("Training output:", loss_dict)
    return loss_dict


# One testing step
def test_step(images: list[torch.Tensor], 
              imageAnnotations: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    targets = []
    for i in range(len(images)):
        d = {}
        d['boxes'] = imageAnnotations[i]

        # Note: Labels are not to be used in this model, since only one class is used.
        d['labels'] = torch.ones(imageAnnotations[i].size(axis=0), dtype=int)

        targets.append(d)
    
    loss_dict = model(images, targets)
    print('Testing output:', loss_dict)
    return loss_dict[0]


# Train the model
def train_loop(train_loader: DataLoader, num_epochs: int, num_batches: int) -> None:
    print("\nBeginning Training...")

    loss_hist = 0
    model.train()
    
    for epoch in tqdm(range(num_epochs)):
        loss_hist = 0
        # print(train_loader.dataset.__getitem__(0))
        for (itr, batch) in enumerate(tqdm(train_loader, total=num_batches)):
            if num_batches and itr > num_batches:
                break
            images = batch[0]
            annotations = batch[1]
            loss_dict = train_step(images, annotations)

            losses = loss_fn(loss_dict)
            loss_val = losses.item()

            loss_hist += loss_val

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            #if itr % 50 == 0:
            print(f"Epoch {epoch}, Iteration {itr} loss: {loss_val}")
        
        print(f"Epoch {epoch} loss: {loss_hist}")


# Test the model
def test_loop(test_loader: DataLoader) -> None:
    print("\nBeginning Testing...")
    
    total_loss = 0.0
    model.eval()

    pbar = tqdm(test_loader)
    for (itr, batch) in enumerate(pbar):
        pbar.set_description(f"Processing batch {itr}")
        images = batch[0]
        annotations = batch[1]
        loss_dict = test_step(images, annotations)
        # visualize_predictions(img=images[0], boxes=loss_dict[0]['boxes'])

        #losses = loss_fn(loss_dict)
        #loss_val = losses.item()
        #total_loss += loss_val

        predicted_count = len(loss_dict['boxes'])
        actual_count = annotations[0].size(dim=0)
        loss_val = abs(actual_count - predicted_count)
        total_loss += loss_val
        print(f"Acutal count: {actual_count}, Predicted count: {predicted_count}, Loss: {loss_val}.")

        print(f"Iteration {itr} loss: {loss_val} cells")
    
    print(f"Total loss: {total_loss}")


# Visualize the predictions
def visualize_predictions(img: torch.Tensor, boxes: torch.Tensor) -> None:
    img = ConvertImageDtype(dtype=torch.uint8)(img)
    box = draw_bounding_boxes(img, boxes=boxes, width=3)
    im = to_pil_image(box.detach())
    im.show()


# View Result of Model
def view_result(image_idx: int, img_dir=r"C:/Users/abbaj/Desktop/Research/IDCIA v2/images/") -> dict[str, torch.Tensor]:
    model.eval()
    image_files: list = glob2.glob(img_dir + "*.tiff")
    image = ToTensor()(Image.open(image_files[image_idx]))
    outputs = model([image])
    print("Image:", outputs)
    img = ConvertImageDtype(dtype=torch.uint8)(image)
    box = draw_bounding_boxes(img, boxes=outputs[0]['boxes'], width=3, colors=["red"] * len(outputs[0]['boxes']))
    im = to_pil_image(box.detach())
    im.show()


# Save model
def save_model(model_path: str = 'src/weights/faster_rcnn/rename_weight.pt'):
    torch.save(model.state_dict(), model_path)


# Load model
def load_model(model_path: str = 'src/weights/faster_rcnn/rename_weight.pt'):
    model.load_state_dict(torch.load(model_path))