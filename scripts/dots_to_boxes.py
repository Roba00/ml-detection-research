"""
Summary: Converts dot annotations into box annotations.
Author: Roba Abbajabal (robaa@iastate.edu)
"""

import json
import sys
from random import randrange

import cv2
import glob2
import pandas as pd
import roifile
from tqdm.auto import tqdm

# Directories for dot annotations, box annotations (of different formats), and images
dot_dir = r"C:/Users/abbaj/Desktop/Research/IDCIAv2/ground_truth/"
faster_rcnn_boxes_dir = r"C:/Users/abbaj/Desktop/Research/IDCIAv2/bounding_boxes_20/"
imagej_boxes_dir = r"C:/Users/abbaj/Desktop/Research/IDCIAv2/bounding_boxes_imagej/"
yolov5_boxes_dir = r"C:/Users/abbaj/Desktop/Research/IDCIAv2/bounding_boxes_yolov5/"
img_dir = r"C:/Users/abbaj/Desktop/Research/IDCIAv2/images/"

# Image dimensions
image_width = 800
image_height = 600

# Cell dimensions
cell_width_min = 20 # 20 before
cell_width_max = 20 # 25 before
cell_height_min = 20 # 20 before
cell_height_max = 20 # 25 before


# Integer clamp function
def clamp(value, min_val, max_val):
    return int(min(max_val, max(min_val, value)))


match input(
    "Option (F = FasterRCNN Box Generation, I = ImageJ ROI Box Generation, V = View Box Results, C = COCO Box Format Generation, Y = YOLOv5 Box Generation? "
):
    case "F":
        # Get the file names of the dot annotations as a list
        dot_files: list = glob2.glob(dot_dir + "*.csv")

        pbar = tqdm(dot_files)
        for fileIdx, dot_file in enumerate(pbar):
            pbar.set_description(f"Processing file: {dot_file}")

            # Reads dataframe of dot annotations from csv, creates dataframe for box annotations
            dotDf: pd.DataFrame = pd.read_csv(dot_file)
            boxDf: pd.DataFrame = pd.DataFrame(columns=["X1", "Y1", "X2", "Y2"])

            # For each row entry in the dot annotations, calculate the box dimensions,
            # and add the dimensions to the box dataframe
            for rowIdx, row in dotDf.iterrows():
                X = row["X"]
                Y = row["Y"]

                X1 = clamp(
                    X - (randrange(cell_width_min, cell_width_max + 1) / 2),
                    0,
                    image_width,
                )
                X2 = clamp(
                    X + (randrange(cell_width_min, cell_width_max + 1) / 2),
                    0,
                    image_width,
                )
                Y1 = clamp(
                    Y - (randrange(cell_height_min, cell_height_max + 1) / 2),
                    0,
                    image_height,
                )
                Y2 = clamp(
                    Y + (randrange(cell_height_min, cell_height_max + 1) / 2),
                    0,
                    image_height,
                )

                boxDf = pd.concat(
                    [pd.DataFrame([[X1, Y1, X2, Y2]], columns=boxDf.columns), boxDf],
                    ignore_index=True,
                )

            # Save the box dataframe as a csv file
            boxDf.to_csv(faster_rcnn_boxes_dir + dot_file.lstrip(dot_dir), index=False)

    case "I":
        # Get the file names of the dot annotations as a list
        dot_files: list = glob2.glob(dot_dir + "*.csv")

        pbar = tqdm(dot_files)
        for fileIdx, dot_file in enumerate(pbar):
            pbar.set_description(f"Processing file: {dot_file}")

            # Reads dataframe of dot annotations from csv, creates dataframe for box annotations
            dotDf: pd.DataFrame = pd.read_csv(dot_file)

            # For each row entry in the dot annotations, calculate the box dimensions,
            # and add the dimensions to the ROI zip file
            for rowIdx, row in dotDf.iterrows():
                X = row["X"]
                Y = row["Y"]

                X1 = clamp(
                    X - (randrange(cell_width_min, cell_width_max + 1) / 2),
                    0,
                    image_width,
                )
                X2 = clamp(
                    X + (randrange(cell_width_min, cell_width_max + 1) / 2),
                    0,
                    image_width,
                )
                Y1 = clamp(
                    Y - (randrange(cell_height_min, cell_height_max + 1) / 2),
                    0,
                    image_height,
                )
                Y2 = clamp(
                    Y + (randrange(cell_height_min, cell_height_max + 1) / 2),
                    0,
                    image_height,
                )

                roi = roifile.ImagejRoi.frompoints(
                    [[X1, Y1], [X2, Y2], [X2, Y1], [X2, Y2]]
                )
                roi.roitype = roifile.ROI_TYPE.RECT
                roi.options |= roifile.ROI_OPTIONS.SHOW_LABELS

                # Save the ROI coordinates to zip file
                roi.tofile(
                    imagej_boxes_dir + dot_file.lstrip(dot_dir).rstrip(".csv") + ".zip"
                )

    case "V":
        # Get the file names of the images and box annotations as a list
        image_files: list = glob2.glob(img_dir + "*.tiff")
        boxes_files: list = glob2.glob(imagej_boxes_dir + "*.csv")

        # Query wanted image file with boxes to display
        idx = int(input("What Image Index (0-249)? "))
        image_file = image_files[idx]
        boxes_file = boxes_files[idx]
        print("Image File:", image_file)
        print("Box Annoations File:", boxes_file)

        # Read the image file, add rectangles around each box annotation, and display it in cv2
        img = cv2.imread(image_file)
        scale = 3
        img = img * scale  # Set a Max on Brightness
        annotationsDf: pd.DataFrame = pd.read_csv(boxes_file)
        for i, row in annotationsDf.iterrows():
            cv2.rectangle(
                img,
                (row["BX"], row["BY"]),
                (row["BX"] + row["Width"], row["BY"] + row["Height"]),
                (0, 255, 0),
                2,
            )
        cv2.imshow(image_file, img)
        cv2.setWindowProperty(image_file, cv2.WND_PROP_TOPMOST, 1)
        cv2.waitKey()
        cv2.destroyAllWindows()

    case "C":
        # Get the file names of the images and dot annotations as a list
        image_files: list = glob2.glob(img_dir + "*.tiff")
        dot_files: list = glob2.glob(dot_dir + "*.csv")

        with open("ICDIA.json", "w", encoding="utf-8") as f:
            data = {
                "images": [],
                "annotations": [],
                "categories": [{"id": 0, "name": "AHPC"}],
            }
            pbar = tqdm(range(len(image_files)))
            annotation_idx: int = 0
            for fileIdx in pbar:
                image_file = image_files[fileIdx]
                dot_file = dot_files[fileIdx]
                pbar.set_description(f"Processing file: {dot_file}")

                data.get("images").append(
                    {
                        "file_name": image_file,
                        "height": image_height,
                        "width": image_width,
                        "id": fileIdx,
                    }
                )

                # Reads dataframe of dot annotations from csv, creates dataframe for box annotations
                dotDf: pd.DataFrame = pd.read_csv(dot_file)

                for rowIdx, row in dotDf.iterrows():
                    X = row["X"] / image_width
                    Y = row["Y"] / image_height
                    width = randrange(cell_width_min, cell_width_max + 1) / image_width
                    height = (
                        randrange(cell_height_min, cell_height_max + 1) / image_height
                    )
                    data.get("annotations").append(
                        {
                            "image_id": fileIdx,
                            "bbox": [X, Y, width, height],
                            "category_id": 0,
                            "id": annotation_idx,
                        }
                    )
                    annotation_idx += 1

            json.dump(data, f, ensure_ascii=False, indent=4)

    case "Y":
        # Get the file names of the dot annotations as a list
        dot_files: list = glob2.glob(dot_dir + "*.csv")

        pbar = tqdm(dot_files)
        for fileIdx, dot_file in enumerate(pbar):
            pbar.set_description(f"Processing file: {dot_file}")

            # Reads dataframe of dot annotations from csv, creates dataframe for box annotations
            dotDf: pd.DataFrame = pd.read_csv(dot_file)
            boxFile = open(
                yolov5_boxes_dir + dot_file.lstrip(dot_dir).rstrip(".csv") + ".txt", "x"
            )

            # For each row entry in the dot annotations, calculate the box dimensions,
            # and add the dimensions to the box dataframe
            for rowIdx, row in dotDf.iterrows():
                X = row["X"] / image_width
                Y = row["Y"] / image_height
                width = randrange(cell_width_min, cell_width_max + 1) / image_width
                height = randrange(cell_height_min, cell_height_max + 1) / image_height
                boxFile.write(f"0 {X} {Y} {width} {height}\n")

            # Save the box dataframe as a csv file
            boxFile.close()
    case _:
        sys.exit("Invalid model option")
