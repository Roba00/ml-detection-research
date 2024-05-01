# Training Faster R-CNN
C:/Users/abbaj/anaconda3/envs/cell_counting_v2/python.exe c:/Users/abbaj/Desktop/Research/rcnn/src/main.py --model faster-rcnn --runOption train --numEpochs 2 --numBatches 1 --savePath c:\\Users\\abbaj\\Desktop\\Research\\rcnn\\src\\weights\\faster_rcnn\\

# Experimenting Faster R-CNN
C:/Users/abbaj/anaconda3/envs/cell_counting_v2/python.exe c:/Users/abbaj/Desktop/Research/rcnn/src/main.py --model faster-rcnn --runOption experiment --savePath c:/Users/abbaj/Desktop/Research/rcnn/src/weights/faster_rcnn/ --loadFile c:/Users/abbaj/Desktop/Research/rcnn/src/weights/faster_rcnn/Weights(1,199)_31-03-24-17-23-55.pt

# Testing Faster R-CNN
C:/Users/abbaj/anaconda3/envs/cell_counting_v2/python.exe c:/Users/abbaj/Desktop/Research/rcnn/src/main.py --model faster-rcnn --runOption test --savePath c:\\Users\\abbaj\\Desktop\\Research\\rcnn\\src\\weights\\ssd\\ --loadFile c:/Users/abbaj/Desktop/Research/rcnn/src/weights/faster_rcnn_new/Weights(50,199)_23-04-24-01-10-13.pt

# All SSD
C:/Users/abbaj/anaconda3/envs/cell_counting_v2/python.exe c:/Users/abbaj/Desktop/Research/rcnn/src/main.py --model ssd --runOption all --numEpochs 2 --numBatches 1 --savePath c:/Users/abbaj/Desktop/Research/rcnn/src/weights/ssd/

# Training SSD
C:/Users/abbaj/anaconda3/envs/cell_counting_v2/python.exe c:/Users/abbaj/Desktop/Research/rcnn/src/main.py --model ssd --runOption train --numEpochs 2 --numBatches 1 --savePath c:/Users/abbaj/Desktop/Research/rcnn/src/weights/ssd/

# Experimenting SSD
C:/Users/abbaj/anaconda3/envs/cell_counting_v2/python.exe c:/Users/abbaj/Desktop/Research/rcnn/src/main.py --model ssd --runOption experiment --savePath c:\\Users\\abbaj\\Desktop\\Research\\rcnn\\src\\weights\\ssd\\ --loadFile c:/Users/abbaj/Desktop/Research/rcnn/src/weights/ssd/ssd-batch16/Weights(100,199)_16-04-24-18-54-15.pt

# Testing SSD
C:/Users/abbaj/anaconda3/envs/cell_counting_v2/python.exe c:/Users/abbaj/Desktop/Research/rcnn/src/main.py --model ssd --runOption test --savePath c:\\Users\\abbaj\\Desktop\\Research\\rcnn\\src\\weights\\ssd\\ --loadFile c:/Users/abbaj/Desktop/Research/rcnn/src/weights/ssd/ssd-batch16/Weights(100,199)_16-04-24-18-54-15.pt

# All RetinaNet
C:/Users/abbaj/anaconda3/envs/cell_counting_v2/python.exe c:/Users/abbaj/Desktop/Research/rcnn/src/main.py --model retinanet --runOption all --numEpochs 2 --numBatches 1 --savePath c:/Users/abbaj/Desktop/Research/rcnn/src/weights/retinanet/

# Training RetinaNet
C:/Users/abbaj/anaconda3/envs/cell_counting_v2/python.exe c:/Users/abbaj/Desktop/Research/rcnn/src/main.py --model retinanet --runOption train --numEpochs 5 --numBatches 3 --savePath c:/Users/abbaj/Desktop/Research/rcnn/src/weights/retinanet/

# Experimenting RetinaNet
C:/Users/abbaj/anaconda3/envs/cell_counting_v2/python.exe c:/Users/abbaj/Desktop/Research/rcnn/src/main.py --model retinanet --runOption experiment --savePath c:\\Users\\abbaj\\Desktop\\Research\\rcnn\\src\\weights\\ssd\\ --loadFile c:/Users/abbaj/Desktop/Research/rcnn/src/weights/retinanet/Weights(16,199)_18-04-24-08-03-23.pt