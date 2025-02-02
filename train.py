from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO("yolo11n-cls.pt")  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(data="mnist", epochs=100, imgsz=32)