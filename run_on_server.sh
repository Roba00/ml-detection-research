cd /work/LAS/tavanapo-lab/robaa/code
module spider python/3.10.10-zwlkg4l
module load python/3.10.10-zwlkg4l
python -m venv newmlenv
source newmlenv/bin/activate
export PYTHONPATH="${PYTHONPATH}:/work/LAS/tavanapo-lab/robaa/code/src/"
python src/main.py --model faster-rcnn --runOption test --numEpochs 50 --numBatches 199 --savePath /work/LAS/tavanapo-lab/robaa/code/src/weights/faster_rcnn_new/ --imagesPath /work/LAS/tavanapo-lab/robaa/datasets/images/ --annotationsPath /work/LAS/tavanapo-lab/robaa/datasets/bounding_boxes_20/ --loadFile /work/LAS/tavanapo-lab/robaa/code/src/weights/faster_rcnn_new/Weights_50-199_23-04-24-01-10-13.pt
