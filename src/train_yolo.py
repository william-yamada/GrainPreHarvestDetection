from ultralytics import YOLO

model = YOLO('yolov8n.yaml').load("yolov8n-oiv7.pt")

model.train(
    data='/usr/src/ultralytics/data/Wheat/data_docker.yaml',  # Path to the dataset configuration file
    epochs=100,                   # Number of training epochs
    batch=32,                     # Batch size
    imgsz=640,                    # Image size for training
    workers=0,                    # Number of data loader workers
    device=0                      # CUDA device, 0 for CPU, 0,1,2,3 for multiple GPUs
)