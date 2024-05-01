from datetime import datetime
from json import dump

import glob2
import roifile
import torch
from matplotlib.pylab import plt
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.models.detection.retinanet import (
    RetinaNet_ResNet50_FPN_V2_Weights,
    retinanet_resnet50_fpn_v2
)
from torchvision.transforms import ConvertImageDtype, ToTensor
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_bounding_boxes
from tqdm.auto import tqdm

from src.losses.retinanet_loss import eval_forward

# Requirements for Model
# ----------------------------------------------------------------------------------------------------------------------
# URL: https://pytorch.org/vision/main/models/generated/torchvision.models.detection.retinanet_resnet50_fpn.html
# - Images: List of tensors, each of shape [C, H, W], one for each image, and should be in 0-1 range.
# - Target: Contains Boxes and Labels
#   - Boxes (FloatTensor[N, 4]): The ground-truth boxes in [x1, y1, x2, y2] format
#       - 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.
#   - Labels (Int64Tensor[N]): The class label for each ground-truth box
#       - Should be set to 1 for all
# Loss Type: {'classification': tensor(0-1, grad_fn=<DivBackward0>), 'bbox_regression': tensor(0-1, grad_fn=<DivBackward0>)}

image_width = 800
image_height = 600
model_weight_save_itr = 5


class RetinaNet:
    def __init__(self) -> None:
        self.model = retinanet_resnet50_fpn_v2(
            weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT,
            #num_classes = 2,
            detections_per_img=1000,
            nms_thresh=0.2,
        )
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(
            self.params, lr=1e-3, momentum=0.9, weight_decay=5e-4
        )
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=3, gamma=0.1
        )
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.initialized = True

    # One training step
    def train_one_epoch(
        self, train_loader: DataLoader, num_batches: int, epoch: int
    ) -> float:
        # Handles, calculates, and returns the losses for one batch
        def calculate_loss_one_batch(
            images: torch.Tensor, annotations: torch.Tensor
        ) -> dict[str, torch.Tensor]:
            targets = []
            for i in range(len(images)):
                d = {}
                d["boxes"] = annotations[i]

                # Note: Labels are not to be used in this model, since only one class is used.
                d["labels"] = torch.ones(annotations[i].size(axis=0), dtype=int)

                targets.append(d)
            return self.model(images, targets)

        # Loss history of each batch in the epoch
        loss_hist = []

        # Train one epoch on multiple batches
        with tqdm(total=num_batches) as pbar:
            for batch_idx, batch in enumerate(train_loader):
                pbar.set_description(f"Training epoch {epoch+1}, batch {batch_idx+1}")

                if batch_idx >= num_batches:
                    break

                # Get images and annotations in batch
                images = batch[1]
                annotations = batch[2]

                # Make predictions for this batch and calculate the loss
                loss_dict = calculate_loss_one_batch(images, annotations)
                losses = sum(loss for loss in loss_dict.values())
                loss_val = losses.item()
                loss_hist.append(loss_val)

                # Compute the loss and its gradients
                self.optimizer.zero_grad()
                losses.backward()

                # Adjusts model parameters
                self.optimizer.step()

                # Log batch errors
                # print(f"Train Epoch {epoch+1}, Batch {batch_idx+1} loss: {loss_val}")

                pbar.update()

        # Adjusts learning rate
        self.lr_scheduler.step()

        # Calculate epoch loss and returns it
        epoch_loss: float = sum(loss_hist) / len(loss_hist)
        return epoch_loss

    # One test step
    def test_one_epoch(
        self,
        eval_loader: DataLoader,
        num_batches: int,
    ) -> tuple[float, float, float, float, float]:
        # Handles, calculates, and returns the predictions for one batch
        def get_preds_one_batch(images: torch.Tensor) -> dict[str, torch.Tensor]:
            targets = []
            for i in range(len(images)):
                d = {}
                d["boxes"] = annotations[i]

                # Note: Labels are not to be used in this model, since only one class is used.
                d["labels"] = torch.ones(annotations[i].size(axis=0), dtype=int)

                targets.append(d)
            return self.model(images)

        # Handles, calculates, and returns the losses for one batch
        def evaluate_loss(
            images: torch.Tensor, annotations: torch.Tensor
        ) -> dict[str, torch.Tensor]:
            targets = []
            for i in range(len(images)):
                d = {}
                d["boxes"] = annotations[i]

                # Note: Labels are not to be used in this model, since only one class is used.
                d["labels"] = torch.ones(annotations[i].size(axis=0), dtype=int)

                targets.append(d)
            losses = eval_forward(self.model, images, targets)
            return losses
        
        # Partial Credit: https://medium.com/analytics-vidhya/iou-intersection-over-union-705a39e7acef
        def verify_in_box(boxes_pred: torch.Tensor, boxes_gt: torch.Tensor, iou_threshold: int = 0.5) -> torch.Tensor:
            iou_boxes_pred = torch.Tensor([])
            for box_pred in boxes_pred:
                x1 = box_pred[0].item(); y1 = box_pred[1].item(); x2 = box_pred[2].item(); y2 = box_pred[3].item()
                for box_gt in boxes_gt:
                    x3 = box_gt[0].item(); y3 = box_gt[1].item(); x4 = box_gt[2].item(); y4 = box_gt[3].item()
                    x_inter1 = max(x1, x3)
                    y_inter1 = max(y1, y3)
                    x_inter2 = min(x2, x4)
                    y_inter2 = min(y2, y4)
                    area_inter = abs(x_inter2 - x_inter1) * abs(y_inter2 - y_inter1)
                    area_box1 = abs(x2 - x1) * abs(y2 - y1)
                    area_box2 = abs(x4 - x3) * abs(y4 - y3)
                    area_union = area_box1 + area_box2 - area_inter
                    iou = area_inter / area_union
                    if (iou > iou_threshold):
                        iou_boxes_pred = torch.cat([iou_boxes_pred[:0], torch.tensor([x_inter1, y_inter1, x_inter2, y_inter2]), iou_boxes_pred[0:]], 0)
                
            return iou_boxes_pred
                    

        # Loss history of each batch in the epoch, with scores
        loss_hist = []
        RawMAE_Sum = 0
        RawACP_Count = 0
        BoxMAE_Sum = 0
        BoxACP_Count = 0

        # Train one epoch on multiple batches
        with tqdm(total=num_batches) as pbar:
            for batch_idx, batch in enumerate(eval_loader):
                pbar.set_description(f"Testing batch {batch_idx+1}")

                if batch_idx >= num_batches:
                    break

                # Get images and annotations in batch
                images = batch[1]
                annotations = batch[2]

                # Evaluate losses for this batch
                # preds = get_preds_one_batch(images)[0]['boxes']
                loss_dict = evaluate_loss(images, annotations)
                # print(loss_dict)
                losses = sum(loss for loss in loss_dict.values())
                loss_val = losses.item()
                loss_hist.append(loss_val)

                # Evaluate criteria score for this batch
                box_gt = annotations[0]
                box_preds = get_preds_one_batch(images)[0]['boxes']
                # print(box_gt); print(box_preds)
                RawMAE_Sum += abs(len(box_preds) - len(box_gt))
                if abs(len(box_preds) - len(box_gt)) / len(box_gt) <= 0.05:
                    RawACP_Count += 1
                box_preds_ious = verify_in_box(box_preds, box_gt)
                BoxMAE_Sum += abs(len(box_preds_ious) - len(box_gt))
                if abs(len(box_preds_ious) - len(box_gt)) / len(box_gt) <= 0.05:
                    BoxACP_Count += 1

                # Log batch errors
                # print(f"Val Epoch {epoch+1}, Batch {batch_idx+1} loss: {loss_val}")

                pbar.update()

        # epoch_loss = sum(loss_hist) / len(loss_hist)
        RawMAE = RawMAE_Sum / num_batches
        RawACP = RawACP_Count / num_batches
        BoxMAE = BoxMAE_Sum / num_batches
        BoxACP = BoxACP_Count / num_batches
        return loss_hist, RawMAE, RawACP, BoxMAE, BoxACP

    # One validation step
    def valid_one_epoch(
        self, eval_loader: DataLoader, num_batches: int, epoch: int
    ) -> float:
        # Handles, calculates, and returns the predictions for one batch
        def get_preds_one_batch(images: torch.Tensor) -> dict[str, torch.Tensor]:
            targets = []
            for i in range(len(images)):
                d = {}
                d["boxes"] = annotations[i]

                # Note: Labels are not to be used in this model, since only one class is used.
                d["labels"] = torch.ones(annotations[i].size(axis=0), dtype=int)

                targets.append(d)
            return self.model(images)

        # Handles, calculates, and returns the losses for one batch
        def evaluate_loss(
            images: torch.Tensor, annotations: torch.Tensor
        ) -> dict[str, torch.Tensor]:
            targets = []
            for i in range(len(images)):
                d = {}
                d["boxes"] = annotations[i]

                # Note: Labels are not to be used in this model, since only one class is used.
                d["labels"] = torch.ones(annotations[i].size(axis=0), dtype=int)

                targets.append(d)
            losses = eval_forward(self.model, images, targets)
            return losses

        # Loss history of each batch in the epoch
        loss_hist = []

        # Train one epoch on multiple batches
        with tqdm(total=num_batches) as pbar:
            for batch_idx, batch in enumerate(eval_loader):
                pbar.set_description(f"Validating epoch {epoch+1}, batch {batch_idx+1}")

                if batch_idx >= num_batches:
                    break

                # Get images and annotations in batch
                images = batch[1]
                annotations = batch[2]

                # Make predictions for this batch and calculate the loss
                # preds = get_preds_one_batch(images)[0]['boxes']
                loss_dict = evaluate_loss(images, annotations)
                # print(loss_dict)
                losses = sum(loss for loss in loss_dict.values())
                loss_val = losses.item()
                loss_hist.append(loss_val)

                # Log batch errors
                # print(f"Val Epoch {epoch+1}, Batch {batch_idx+1} loss: {loss_val}")

                pbar.update()

        epoch_loss = sum(loss_hist) / len(loss_hist)
        return epoch_loss

    # Train the model
    def train_loop(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        num_epochs: int,
        num_batches: int,
        logResults: bool = True,
        plotResults: bool = True,
        savePath: str = "",
    ) -> None:
        print("\nBeginning Training...")

        train_loss_hist = []
        valid_loss_hist = []

        for epoch in range(num_epochs):
            # Calculate training loss of one epoch
            self.model.train()
            train_epoch_loss: float = self.train_one_epoch(
                train_loader, num_batches, epoch
            )
            train_loss_hist.append(train_epoch_loss)
            print(f"Train Epoch {epoch+1} loss: {train_epoch_loss}")

            # Save model weights
            if epoch % model_weight_save_itr == 0 or epoch == num_epochs - 1:
                torch.save(
                    self.model.state_dict(),
                    savePath
                    + datetime.now().strftime(
                        f"Weights({epoch+1},{num_batches})_%d-%m-%y-%H-%M-%S"
                    )
                    + ".pt",
                )

            # Validate one epoch on the validation set
            self.model.eval()
            valid_epoch_loss: float = self.valid_one_epoch(
                valid_loader, len(valid_loader), epoch
            )
            valid_loss_hist.append(valid_epoch_loss)
            print(f"Valid Epoch {epoch+1} loss: {valid_epoch_loss}")

        # Save the training loss values
        if logResults:
            with open(
                savePath
                + datetime.now().strftime(
                    f"TrainLosses({num_epochs},{num_batches})_%d-%m-%y-%H-%M-%S.json"
                ),
                "w",
            ) as file:
                dump(
                    {"train_lossses": train_loss_hist, "valid_losses": valid_loss_hist},
                    file,
                )

        # Plot the training loss values
        if plotResults:
            #print(range(1, num_epochs + 1))
            plt.plot(range(1, num_epochs + 1), train_loss_hist, label="Training Loss")
            plt.plot(range(1, num_epochs + 1), valid_loss_hist, label="Validation Loss")
            plt.title("Training and Validation Loss")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend(loc="best")
            plt.savefig(
                savePath
                + datetime.now().strftime(
                    f"TrainLosses({num_epochs},{num_batches})_%d-%m-%y-%H-%M-%S"
                )
                + ".png"
            )
            # plt.show()

    # Test the model
    def test_loop(
        self,
        test_loader: DataLoader,
        num_batches: int,
        logResults: bool = True,
        savePath: str = "",
    ) -> None:
        print("\nBeginning Testing...")

        self.model.eval()

        # Evaluate one epoch on the validation set
        test_loss_hist, RawMAE, RawACP, BoxMAE, BoxACP = self.test_one_epoch(
            test_loader, num_batches
        )
        avg_test_loss = sum(test_loss_hist) / len(test_loss_hist)
        print(f"Average Test loss: {avg_test_loss}")
        print(f"Raw MAE: {RawMAE}")
        print(f"Raw ACP: {RawACP}")
        print(f"Box IoU MAE: {BoxMAE}")
        print(f"Box IoU ACP: {BoxACP}")
        print(f"Loss Hist: {test_loss_hist}")

        # Save the test loss values
        if logResults:
            with open(
                savePath
                + datetime.now().strftime(
                    f"TestLosses({num_batches})_%d-%m-%y-%H-%M-%S.json"
                ),
                "w",
            ) as file:
                dump(
                    {
                        "Raw MAE": RawMAE,
                        "Raw ACP": RawACP,
                        "Box IoU MAE": BoxMAE,
                        "Box IoU ACP": BoxACP,
                        "Loss Hist": test_loss_hist,
                    },
                    file,
                )

    # New: Converting boxes dictionary from inference to boxes csv file
    def generate_boxes_csv(self, boxes_dict: dict, filename: str) -> None:

        for boxIndex in len(boxes_dict["boxes"]):
            X1 = boxes_dict["boxes"][boxIndex][0]
            Y1 = boxes_dict["boxes"][boxIndex][1]
            X2 = boxes_dict["boxes"][boxIndex][2]
            Y2 = boxes_dict["boxes"][boxIndex][3]
            width = X2 - X1
            height = Y2 - Y1

            if (
                X1 >= X2
                or Y1 >= Y2
                or X1 < 0
                or Y1 < 0
                or X1 + width >= image_width
                or Y1 + height >= image_height
            ):
                continue
            roi = roifile.ImagejRoi.frompoints([[X1, Y1], [X2, Y2], [X2, Y1], [X2, Y2]])
            roi.roitype = roifile.ROI_TYPE.RECT
            roi.options |= roifile.ROI_OPTIONS.SHOW_LABELS

            # Save the ROI coordinates to zip file
            roi.tofile(filename)

    # Visualize the predictions
    def visualize_predictions(self, img: torch.Tensor, boxes: torch.Tensor) -> None:
        img = ConvertImageDtype(dtype=torch.uint8)(img)
        box = draw_bounding_boxes(img, boxes=boxes, width=3)
        im = to_pil_image(box.detach())
        im.show()

    # View Result of Model
    def view_result(
        self,
        image_idx: int,
        img_dir=r"C:/Users/abbaj/Desktop/Research/IDCIAv2/images/",
    ) -> dict[str, torch.Tensor]:
        self.model.eval()
        image_files: list = glob2.glob(img_dir + "*.tiff")
        image = ToTensor()(Image.open(image_files[image_idx]))
        outputs = self.model([image])
        print(f"Image: {image_files[image_idx]} loaded.")
        img = ConvertImageDtype(dtype=torch.uint8)(image)
        box = draw_bounding_boxes(
            img,
            boxes=outputs[0]["boxes"],
            width=3,
            colors=["red"] * len(outputs[0]["boxes"]),
        )
        im = to_pil_image(box.detach())
        im.show()

    # Save model
    def save_model(
        self,
        model_path: str = "/content/drive/My Drive/weights/retinanet/",
    ):
        torch.save(self.model.state_dict(), model_path + "rename_weight.pt")

    # Load model
    def load_model(
        self,
        model_path: str = "/content/drive/My Drive/weights/retinanet/rename_weight.pt",
    ):
        self.model.load_state_dict(torch.load(model_path))
