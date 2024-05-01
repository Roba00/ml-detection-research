from enum import Enum
import glob2
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

class AnnotationFileType(Enum):
      CSV = 1
      TXT = 2

class CellDataset(Dataset):
    def __init__(self, images, annotations, isSplit=False, file_type=AnnotationFileType.CSV, transform=None, target_transform=None, shouldLog=True):
        if isSplit:
            self.image_files: list[str] = images
            self.annotations_files: list[str] = annotations
        else:
            self.image_files: list[str] = glob2.glob(images + "*.tiff")
            if (file_type == AnnotationFileType.CSV):
                self.annotations_files: list[str] = glob2.glob(annotations + "*.csv")
            elif (file_type == AnnotationFileType.TXT):
                self.annotations_files: list[str] = glob2.glob(annotations + "*.txt")
            
        self.file_type: AnnotationFileType = file_type 
        if shouldLog:   
            print("Num Images:", len(self.image_files))
            print("Num Image Annotations:", len(self.annotations_files))
        
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # print(self.image_files[idx], self.annotations_files[idx])
        imageName: str = self.image_files[idx].rstrip(".tiff")
        image = Image.open(self.image_files[idx])
        imageTensor = ToTensor()(image)
        if (self.file_type == AnnotationFileType.CSV):
            annotationsDf: pd.DataFrame = pd.read_csv(self.annotations_files[idx])
            annotationsTensor = torch.Tensor(annotationsDf.values)
        else:
            annotationsTensor = self.annotations_files

        if self.transform:
            imageTensor: torch.Tensor = self.transform(imageTensor)
        if self.target_transform:
            if (self.file_type == AnnotationFileType.CSV):
                annotationsTensor: torch.Tensor = self.target_transform(annotationsTensor)
        
        return imageName, imageTensor, annotationsTensor
    
