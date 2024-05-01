import json
import sys
from random import randrange

import cv2
import glob2
import pandas as pd
import roifile
from tqdm.auto import tqdm

box_dir=r"C:/Users/abbaj/Desktop/Research/test_pred_box/"
result_dir=r"C:/Users/abbaj/Desktop/Research/test_pred_box_roi/"

box_files: list = glob2.glob(box_dir + "*.csv")

pbar = tqdm(box_files)
for fileIdx, box_file in enumerate(pbar):
    pbar.set_description(f"Processing file: {box_file}")

    # Reads dataframe of dot annotations from csv, creates dataframe for box annotations
    dotDf: pd.DataFrame = pd.read_csv(box_file)

    # For each row entry in the dot annotations, calculate the box dimensions,
    # and add the dimensions to the ROI zip file
    for rowIdx, row in dotDf.iterrows():
        X1 = row["Xmin"]
        Y1 = row["Ymin"]
        X2 = row["Xmax"]
        Y2 = row["Ymax"]

        roi = roifile.ImagejRoi.frompoints(
            [[X1, Y1], [X2, Y2], [X2, Y1], [X2, Y2]]
        )
        roi.roitype = roifile.ROI_TYPE.RECT
        roi.options |= roifile.ROI_OPTIONS.SHOW_LABELS

        # Save the ROI coordinates to zip file
        roi.tofile(
            result_dir + box_file.lstrip(box_dir).rstrip(".csv") + ".zip"
        )