import torch
import cv2
import time
import numpy as np
from ultralytics import YOLO
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from kcf import Tracker

def load_detection_model():
    """
    Loads a trained Faster R-CNN model for object detection.

    Returns:
        detection_model (nn.Module): Loaded PyTorch detection model.
        device (torch.device): Device (CPU/GPU) where model is loaded.
    """
    detection_model_path = "/home/adamslab/Documents/trained_detection_models/fasterrcnn_model.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    detection_model = fasterrcnn_resnet50_fpn(weights=None, num_classes=2)
    state_dict = torch.load(detection_model_path, map_location=device)
    detection_model.load_state_dict(state_dict)
    detection_model.to(device)
    detection_model.eval()
    return detection_model, device

def load_yolo_model():
    """
    Loads a YOLOv8 model for detection.

    Returns:
        yolo_model (YOLO): Loaded YOLO detection model.
        device (torch.device): Device (CPU/GPU) where model is loaded.
    """
    yolo_model = YOLO("yolo11_trained.pt")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    yolo_model.to(device)
    return yolo_model, device

def detect(model, rgb_image, depth_image, device, detection_threshold):
    """
    Performs object detection using Faster R-CNN and estimates depth at the object's center.

    Args:
        model (nn.Module): Detection model.
        rgb_image (np.array): RGB input image.
        depth_image (np.array): Depth map corresponding to RGB image.
        device (torch.device): Device to run inference.
        detection_threshold (float): Confidence threshold for detection.

    Returns:
        bbox (list): Bounding box [x_min, y_min, x_max, y_max].
        distance (float): Depth value at center of bounding box.
        no_detection (bool): True if no object was detected.
    """
    img_tensor = F.to_tensor(rgb_image).unsqueeze_(0).to(device)
    with torch.no_grad():
        results = model(img_tensor)

    bboxes = results[0]['boxes'].cpu().numpy()
    scores = results[0]['scores'].cpu().numpy()
    best_bboxes = bboxes[scores > detection_threshold]

    if len(best_bboxes) > 0:
        bbox = best_bboxes[0]
        x_min, y_min, x_max, y_max = map(int, bbox)
        x_center = int((x_min + x_max) / 2)
        y_center = int((y_min + y_max) / 2)
        distance = depth_image[y_center, x_center]
        no_detection = False
    else:
        bbox = [0, 0, 0, 0]
        distance = 1e4
        no_detection = True

    return bbox, distance, no_detection

kcf_tracker = None
tracking = False

def detect_yolo(model, rgb_image, confidence_threshold=0.7, mode='kcf'):
    """
    Detects object using YOLO or a hybrid YOLO + KCF pipeline.

    Args:
        model (YOLO): YOLO model for detection.
        rgb_image (np.array): RGB frame for detection/tracking.
        confidence_threshold (float): Minimum confidence for detection.
        mode (str): 'yolo' or 'kcf' for direct or hybrid detection.

    Returns:
        bbox (list): Detected/tracked bounding box [x1, y1, x2, y2].
        no_detection (bool): True if no object detected/tracked.
        frame (np.array): Annotated output frame.
        use_yolo (bool): True if YOLO was triggered this frame.
    """
    global kcf_tracker, tracking
    frame = rgb_image.copy()
    bbox = [0, 0, 0, 0]
    no_detection = True
    status = ""
    use_yolo = False

    if mode == 'yolo':
        use_yolo = True
        results = model(frame)
        if results[0] and results[0].boxes:
            for box in results[0].boxes:
                confidence = float(box.conf[0])
                if confidence >= confidence_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    bbox = [x1, y1, x2, y2]
                    no_detection = False
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"YOLO: {confidence:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    break

    elif mode == 'kcf':
        use_yolo = not tracking
        if tracking:
            try:
                (x, y, w, h), apce = kcf_tracker.update(frame)
                if w <= 0 or h <= 0 or apce < 0.009:
                    raise ValueError(f"APCE too low: {apce:.4f}")
                bbox = [x, y, x + w, y + h]
                no_detection = False
                status = f"KCF Tracking | APCE: {apce:.4f}"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, status, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            except Exception as e:
                use_yolo = True
                tracking = False

        if use_yolo:
            results = model(frame)
            detections = results[0].boxes.data
            if len(detections) > 0:
                x1, y1, x2, y2 = map(int, detections[0][:4])
                bbox = [x1, y1, x2, y2]
                box_w, box_h = x2 - x1, y2 - y1
                kcf_tracker = Tracker()
                kcf_tracker.init(frame, (x1, y1, box_w, box_h))
                tracking = True
                no_detection = False
                status = "YOLO Detection (Reinit)"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, status, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            else:
                status = "YOLO: No Detection"
                tracking = False
                no_detection = True

    cv2.putText(frame, mode.upper(), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("Tracking", frame)
    cv2.waitKey(1)

    return bbox, no_detection, frame, use_yolo
