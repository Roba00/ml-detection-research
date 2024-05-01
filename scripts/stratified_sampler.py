from random import randrange, shuffle

import glob2
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# Constants
TRAINING_SET_PERCENTAGE = .75

# Directories for images and box annotations
img_dir = r"C:/Users/abbaj/Desktop/Research/IDCIAv2/images/"
faster_rcnn_boxes_dir = r"C:/Users/abbaj/Desktop/Research/IDCIAv2/bounding_boxes_faster_rcnn/"
img_files: list = glob2.glob(img_dir + "*.tiff")
boxes_files: list = glob2.glob(faster_rcnn_boxes_dir + "*.csv")

boxes_dict = {'0-59': [], '60-299': [], '300-999': []}

for boxes_file in tqdm(boxes_files):
    boxes_df: pd.DataFrame = pd.read_csv(boxes_file)
    nrow: int = len(boxes_df)

    if nrow < 60:
        boxes_dict['0-59'].append(boxes_file)
    elif nrow < 300:
        boxes_dict['60-299'].append(boxes_file)
    elif nrow < 1000:
        boxes_dict['300-999'].append(boxes_file)
    
counts = []
for key in boxes_dict:
    counts.append(len(boxes_dict[key]))
plt.rc('axes', titlesize=15)
plt.rc('axes', labelsize=15)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.bar(boxes_dict.keys(), counts)
plt.xlabel("Cell Count Range")
plt.ylabel("Amount of Cells")
plt.show()

'''
testSetPercent = .10
validationSetPercent = .10

trainSetLabels = []
validSetLabels = []
testSetLabels = []


for key in boxes_dict:
    print(f'Cell counts {key} set (size={len(boxes_dict[key])}):')
    shuffle(boxes_dict[key])

    allLeft = boxes_dict[key]
    trainSet = allLeft[:int(TRAINING_SET_PERCENTAGE * len(allLeft))]
    trainLeft = [a for a in allLeft if a not in trainSet]
    validSet = trainLeft[:int(.5*len(trainLeft))]
    validLeft = [a for a in trainLeft if a not in validSet]
    testSet = validLeft

    print('Train/Valid/Test Set Sizes:', len(trainSet), len(validSet), len(testSet))
    print('\n')

    trainSetLabels += trainSet
    validSetLabels += validSet
    testSetLabels += testSet

print('Overall Train/Valid/Test Set Sizes:', len(trainSetLabels), len(validSetLabels), len(testSetLabels))
print('\n')
'''