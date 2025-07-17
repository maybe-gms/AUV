import cv2
import numpy as np

# Load labels
with open("data_items.names", "r") as f:
    obj_names = f.read().strip().split("\n")

# Load model
model = cv2.dnn_DetectionModel("frozen_inference_graph.pb", "Pretrained_vectors_mobile_net.pbtxt")
model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

# Targets and depth
targets = ["Human", "Plastic Bottle"]
BASELINE_CM = 12.0
FOCAL_LENGTH_PX = 630.0
depth_history = {}

def smoothed_depth(label, raw_depth, window=5):
    if label not in depth_history:
        depth_history[label] = []
    history = depth_history[label]
    history.append(raw_depth)
    if len(history) > window:
        history.pop(0)
    return round(sum(history) / len(history), 2)

def detect(frame):
    results = model.detect(frame, confThreshold=0.45, nmsThreshold=0.4)
    detections = []
    if len(results) == 3:
        class_ids, confidences, boxes = results
        for class_id, confidence, box in zip(class_ids, confidences, boxes):
            label = obj_names[class_id - 1]
            if label.lower() in [t.lower() for t in targets] and confidence > 0.45:
                detections.append((label, box))
    return detections

def compute_depth(center_left, center_right):
    disparity = abs(center_left[0] - center_right[0])
    if disparity < 1:
        return None
    return round((FOCAL_LENGTH_PX * BASELINE_CM) / disparity, 2)

def match_detections(dets0, dets1):
    matched = []
    for label1, box1 in dets0:
        c1 = (box1[0] + box1[2] // 2, box1[1] + box1[3] // 2)
        for label2, box2 in dets1:
            if label1 == label2:
                c2 = (box2[0] + box2[2] // 2, box2[1] + box2[3] // 2)
                matched.append((label1, box1, box2, c1, c2))
                break
    return matched
