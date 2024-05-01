from ultralytics import YOLO

# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolov8n.pt')

# Train the model using the 'coco128.yaml' dataset for 3 epochs
results = model.train(data='src/extras/IDCIAv2.yaml', epochs=1)

# Perform object detection on an image using the model
results = model(r'C:\Users\abbaj\Desktop\Research\IDCIAv2\datasets\IDCIAv2\images\test\220815_GFP-AHPC_A_Ki67_F5_DAPI_ND1_20x.tiff')
print(results)

model.export(format='onnx')

'''
def init() -> None:
    global model, params, optimizer
    model = YOLO('yolov8n.pt')
    global initialized
    initialized = True

def train(num_epochs: int) -> None:
    results = model.train(data='coco128.yaml', epochs=num_epochs)

def view_result(image_url: str):
    model.eval()
    results = model(image_url)
    print("Results:", results)
'''