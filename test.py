from ultralytics import YOLO 
import torch
import cv2

# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = YOLO("yolov8s.pt").to(device)

results = model.predict(source ="0",show = True)

print(results)
