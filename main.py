import argparse
import os
import sys

from src.data_loaders.data_loader import Data_Loader
from src.models.faster_rcnn import Faster_RCNN
from src.models.retinanet import RetinaNet
from src.models.ssd import SSD

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--runOption", required=True)
parser.add_argument("--imagesPath", required=False)
parser.add_argument("--annotationsPath", required=False)
parser.add_argument("--exportBoxesPath", required=False)
parser.add_argument("--numEpochs", required=False)
parser.add_argument("--numBatches", required=False)
parser.add_argument("--loadFile", required=False)
parser.add_argument("--savePath", required=False)
args = parser.parse_args()
print(args)
modelOption = args.model
runOption = args.runOption
imagesPath = (
    "C:/Users/abbaj/Desktop/Research/IDCIAv2/images/"
    if args.imagesPath is None
    else args.imagesPath
)
annotationsPath = (
    "C:/Users/abbaj/Desktop/Research/IDCIAv2/bounding_boxes_20/"
    if args.annotationsPath is None
    else args.annotationsPath
)
exportBoxesPath = (
    "C:/Users/abbaj/Desktop/Research/IDCIAv2/faster_rcnn_predictions/"
    if args.exportBoxesPath is None
    else args.exporBoxestPath
)
runOption = args.runOption
numEpochs = 0 if args.numEpochs is None else int(args.numEpochs)
numBatches = 0 if args.numBatches is None else int(args.numBatches)
loadFile = "" if args.loadFile is None else args.loadFile
savePath = "" if args.savePath is None else args.savePath

if modelOption == "faster-rcnn":
    data_loader = Data_Loader(
        images_path=imagesPath,
        annotations_path=annotationsPath,
        batch_size=1,
        savePath=savePath,
    )
    rcnn = Faster_RCNN()

    if data_loader.initialized == False or rcnn.initialized == False:
        sys.exit("Error: Data loader or RCNN model not initialized.")

    if runOption == "all":
        rcnn.train_loop(
            data_loader.train_loader,
            data_loader.valid_loader,
            num_epochs=numEpochs,
            num_batches=numBatches,
            savePath=savePath,
        )
        rcnn.save_model(model_path=savePath)
        rcnn.test_loop(
            data_loader.test_loader,
            num_batches=len(data_loader.test_loader),
            savePath=savePath,
        )
        # rcnn.view_result(0)
    elif runOption == "train":
        rcnn.train_loop(
            data_loader.train_loader,
            data_loader.valid_loader,
            num_epochs=numEpochs,
            num_batches=numBatches,
            savePath=savePath,
        )
        rcnn.save_model(model_path=savePath)
        # rcnn.view_result(0)
    elif runOption == "test":
        if not os.path.isfile(loadFile):
            print("Invalid Load Path")
            sys.exit(-1)
        rcnn.load_model(model_path=loadFile)
        rcnn.test_loop(
            data_loader.test_loader,
            num_batches=len(data_loader.test_loader),
            savePath=savePath,
        )
    elif runOption == "experiment":
        if not os.path.isfile(loadFile):
            print("Invalid Load Path")
            sys.exit(-1)
        idx = int(input("What Image Index (0-249)? "))
        rcnn.load_model(model_path=loadFile)
        rcnn.view_result(idx, img_dir=imagesPath)
    else:
        sys.exit("Invalid run option")

elif modelOption == "ssd":
    data_loader = Data_Loader(
        images_path=imagesPath,
        annotations_path=annotationsPath,
        batch_size=16,
        savePath=savePath,
    )
    ssd = SSD()

    if data_loader.initialized == False or ssd.initialized == False:
        sys.exit("Error: Data loader or SSD model not initialized.")

    if runOption == "all":
        ssd.train_loop(
            data_loader.train_loader,
            data_loader.valid_loader,
            num_epochs=numEpochs,
            num_batches=numBatches,
            savePath=savePath,
        )
        ssd.save_model(model_path=savePath)
        ssd.test_loop(
            data_loader.test_loader,
            num_batches=len(data_loader.test_loader),
            savePath=savePath,
        )
        # ssd.view_result(0)
    elif runOption == "train":
        ssd.train_loop(
            data_loader.train_loader,
            data_loader.valid_loader,
            num_epochs=numEpochs,
            num_batches=numBatches,
            savePath=savePath,
        )
        ssd.save_model(model_path=savePath)
        # ssd.view_result(0)
    elif runOption == "test":
        if not os.path.isfile(loadFile):
            print("Invalid Load Path")
            sys.exit(-1)
        ssd.load_model(model_path=loadFile)
        ssd.test_loop(
            data_loader.test_loader,
            num_batches=len(data_loader.test_loader),
            savePath=savePath,
        )
    elif runOption == "experiment":
        if loadFile != "":
            if not os.path.isfile(loadFile):
                print("Invalid Load Path")
                sys.exit(-1)
            ssd.load_model(model_path=loadFile)
        idx = int(input("What Image Index (0-249)? "))
        ssd.view_result(idx)
    else:
        sys.exit("Invalid run option")

elif modelOption == "retinanet":
    data_loader = Data_Loader(
        images_path=imagesPath,
        annotations_path=annotationsPath,
        batch_size=1,
    )
    retinanet = RetinaNet()

    if data_loader.initialized == False or retinanet.initialized == False:
        sys.exit("Error: Data loader or RetinaNet model not initialized.")

    if runOption == "all":
        retinanet.train_loop(
            data_loader.train_loader,
            data_loader.valid_loader,
            num_epochs=numEpochs,
            num_batches=numBatches,
            savePath=savePath,
        )
        retinanet.save_model(model_path=savePath)
        retinanet.test_loop(
            data_loader.test_loader,
            num_batches=len(data_loader.test_loader),
            savePath=savePath,
        )
        # ssd.view_result(0)
    elif runOption == "train":
        retinanet.train_loop(
            data_loader.train_loader,
            data_loader.valid_loader,
            num_epochs=numEpochs,
            num_batches=numBatches,
            savePath=savePath,
        )
        retinanet.save_model(model_path=savePath)
        # retinanet.view_result(0)
    elif runOption == "test":
        if loadFile != "":
            if not os.path.isfile(loadFile):
                print("Invalid Load Path")
                sys.exit(-1)
            retinanet.load_model(model_path=loadFile)
        retinanet.test_loop(
            data_loader.test_loader,
            num_batches=len(data_loader.test_loader),
            savePath=savePath,
        )
    elif runOption == "experiment":
        if loadFile != "":
            if not os.path.isfile(loadFile):
                print("Invalid Load Path")
                sys.exit(-1)
            retinanet.load_model(model_path=loadFile)
        idx = int(input("What Image Index (0-249)? "))
        retinanet.view_result(idx)
    else:
        sys.exit("Invalid run option")

else:
    sys.exit("Invalid model option")
