import cv2
import numpy as np
from picamera2 import Picamera2
import time

# === Load labels ===
with open("data_items.names", "r") as f:
    obj_names = f.read().strip().split("\n")

# === Load DNN model ===
model = cv2.dnn_DetectionModel("frozen_inference_graph.pb", "Pretrained_vectors_mobile_net.pbtxt")
model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

# === Target objects and camera setup ===
targets = ["Human", "Plastic Bottle"]
BASELINE_CM = 12.0
FOCAL_LENGTH_PX = 620.0

# === Init cameras ===
picam0 = Picamera2(0)
picam1 = Picamera2(1)

config = {
    "main": {"size": (640, 480)},
    "controls": {
        "FrameDurationLimits": (33333, 33333),  # ~30fps
        "AeEnable": True,
        "AwbEnable": True
    }
}
picam0.configure(picam0.create_video_configuration(**config))
picam1.configure(picam1.create_video_configuration(**config))
picam0.start()
picam1.start()
time.sleep(2)

# === Detection and depth ===
def detect(frame):
    # Ensure RGB → BGR before feeding into model
    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    else:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    results = model.detect(frame, confThreshold=0.45, nmsThreshold=0.4)
    detections = []
    if len(results) == 3:
        class_ids, confidences, boxes = results
        for class_id, confidence, box in zip(class_ids, confidences, boxes):
            label = obj_names[class_id - 1]
            if label.lower() in [t.lower() for t in targets]:
                detections.append((label, box))
    return detections

def compute_depth(center_left, center_right):
    disparity = abs(center_left[0] - center_right[0])
    if disparity < 1:
        return None
    return round((FOCAL_LENGTH_PX * BASELINE_CM) / disparity, 2)

def match_detections(dets0, dets1):
    matches = []
    for label0, box0 in dets0:
        c0 = (box0[0] + box0[2] // 2, box0[1] + box0[3] // 2)
        for label1, box1 in dets1:
            if label0 == label1:
                c1 = (box1[0] + box1[2] // 2, box1[1] + box1[3] // 2)
                matches.append((label0, c0, c1))
                break
    return matches

# === Loop ===
print("Running... Press Ctrl+C to stop.")
try:
    while True:
        f0 = cv2.flip(picam0.capture_array(), 0)
        f1 = cv2.flip(picam1.capture_array(), 0)

        d0 = detect(f0)
        d1 = detect(f1)

        matches = match_detections(d0, d1)
        for label, c0, c1 in matches:
            depth = compute_depth(c0, c1)
            if depth:
                print(f"{label} Depth: {depth} cm")

except KeyboardInterrupt:
    print("\nStopped.")
