import torch
from torchvision.transforms import v2
import src.data_loaders.data_loader as data_loader


preproc = v2.Compose([
    v2.PILToTensor(),
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32),  # to float32 in [0, 1]
    v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # typically from ImageNet
])

cutmix = v2.CutMix(num_classes=1)
mixup = v2.MixUp(num_classes=1)
cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

data_loader.init(images_path="IDCIA/images/DAPI/", 
                 annotations_path="IDCIA/annotations/DAPI/", 
                 batch_size=32)

for images, labels in data_loader.test_loader:
    print(f"Before CutMix/MixUp: {images.shape = }, {labels.shape = }")
    images, labels = cutmix_or_mixup(images, labels)
    print(f"After CutMix/MixUp: {images.shape = }, {labels.shape = }")
    # <rest of the training loop here>
    break