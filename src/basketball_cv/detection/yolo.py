from ultralytics import YOLO


def load_yolo_model(weights: str) -> YOLO:
    return YOLO(weights)
