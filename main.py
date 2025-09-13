from ultralytics import YOLO
import torch
from multiprocessing import freeze_support

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = YOLO("yolo11n.pt")  

data_path = r"/Users/berry/Desktop/demo/dangerous-objects-dataset/data.yaml"

def train():
    model.train(
        data=data_path, 
        epochs=10, 
        imgsz=640,
        device=device,
        patience=10,
        
    )

if __name__ == '__main__':
    freeze_support()
    train()