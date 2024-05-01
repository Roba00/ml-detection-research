from datetime import datetime
from json import dump
from random import shuffle, seed

import pandas as pd
from torch.utils.data import DataLoader, random_split

from src.datasets.cell_dataset import AnnotationFileType, CellDataset


class Data_Loader:
    def collate_fn(self, batch):
        return tuple(zip(*batch))

    def __init__(
        self,
        images_path="IDCIA/images/DAPI/",
        annotations_path="IDCIA/annotations/DAPI/",
        batch_size=1,
        file_type=AnnotationFileType.CSV,
        savePath=""
    ) -> None:
        # Obtain full dataset
        full_dataset = CellDataset(
            images_path, annotations_path, isSplit=False, file_type=file_type
        )
        print("\nFull Dataset Size:", len(full_dataset))

        # Split dataset into training and testing
        train_dataset, valid_dataset, test_dataset = self.stratified_split(
            dataset=full_dataset, train_percentage=0.8, randomSeed=1, savePath=savePath
        )
        print("Training Dataset Size:", len(train_dataset))
        print("Valid Dataset Size:", len(valid_dataset))
        print("Test Dataset Size:", len(test_dataset))
        # print("Training Image 0:", len(train_dataset.__getitem__(0)[0]))

        # Setup data loaders for the models
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )
        self.valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

        self.initialized = True

    def stratified_split(
        self, dataset: CellDataset, train_percentage: float, shouldLog=False, randomSeed=None, savePath=""
    ):
        trainDatasetImages = []
        validDatasetImages = []
        testDatasetImages = []
        trainDatasetBoxes = []
        validDatasetBoxes = []
        testDatasetBoxes = []

        image_dict = {"0-59": [], "60-299": [], "300-999": []}
        boxes_dict = {"0-59": [], "60-299": [], "300-999": []}

        datasetSize: int = len(dataset.image_files)

        if randomSeed != None:
            seed(randomSeed)

        for i in range(datasetSize):
            image_file = dataset.image_files[i]
            boxes_file = dataset.annotations_files[i]
            boxes_df: pd.DataFrame = pd.read_csv(boxes_file)
            nrow: int = len(boxes_df)

            # Create stratified dictionary
            if nrow < 60:
                image_dict["0-59"].append(image_file)
                boxes_dict["0-59"].append(boxes_file)
            elif nrow < 300:
                image_dict["60-299"].append(image_file)
                boxes_dict["60-299"].append(boxes_file)
            elif nrow < 1000:
                image_dict["300-999"].append(image_file)
                boxes_dict["300-999"].append(boxes_file)

        for key in boxes_dict:
            if shouldLog:
                print(f"Cell counts {key} set (size={len(boxes_dict[key])}):")

            if len(boxes_dict[key]) == 0:
                continue

            # Shuffle both images and boxes in same order
            tempForShuffle = list(zip(image_dict[key], boxes_dict[key]))
            shuffle(tempForShuffle)
            image_dict[key], boxes_dict[key] = zip(*tempForShuffle)

            allLeft_images = image_dict[key]
            trainSet_images = allLeft_images[
                : int(train_percentage * len(allLeft_images))
            ]
            trainLeft_images = [a for a in allLeft_images if a not in trainSet_images]
            validSet_images = trainLeft_images[: int(0.5 * len(trainLeft_images))]
            validLeft_images = [a for a in trainLeft_images if a not in validSet_images]
            testSet_images = validLeft_images

            allLeft_boxes = boxes_dict[key]
            trainSet_boxes = allLeft_boxes[: int(train_percentage * len(allLeft_boxes))]
            trainLeft_boxes = [a for a in allLeft_boxes if a not in trainSet_boxes]
            validSet_boxes = trainLeft_boxes[: int(0.5 * len(trainLeft_boxes))]
            validLeft_boxes = [a for a in trainLeft_boxes if a not in validSet_boxes]
            testSet_boxes = validLeft_boxes

            if shouldLog:
                print(
                    "Train/Valid/Test Set Sizes:",
                    len(trainSet_images),
                    len(validSet_images),
                    len(testSet_images),
                )

            trainDatasetImages += trainSet_images
            validDatasetImages += validSet_images
            testDatasetImages += testSet_images

            trainDatasetBoxes += trainSet_boxes
            validDatasetBoxes += validSet_boxes
            testDatasetBoxes += testSet_boxes

        if shouldLog:
            print(
                "Train/Valid/Test Set Sizes:",
                len(trainDatasetImages),
                len(validDatasetImages),
                len(testDatasetImages),
            )

        train_dataset = CellDataset(
            trainDatasetImages, trainDatasetBoxes, isSplit=True, shouldLog=False
        )
        valid_dataset = CellDataset(
            validDatasetImages, validDatasetBoxes, isSplit=True, shouldLog=False
        )
        test_dataset = CellDataset(
            testDatasetImages, testDatasetBoxes, isSplit=True, shouldLog=False
        )

        with open(
            savePath
            + datetime.now().strftime(
                f"DatasetSplit_%d-%m-%y-%H-%M-%S.json"
            ),
            "w",
        ) as file:
            dump({"Train Dataset": trainDatasetImages, 
                  "Valid Dataset": validDatasetImages, 
                  "Test Dataset": testDatasetImages}, file)

        return train_dataset, valid_dataset, test_dataset
