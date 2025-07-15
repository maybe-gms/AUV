import cv2
import numpy as np
from picamera2 import Picamera2
import threading
import time

# === Load Labels ===
with open("data_items.names", "r") as f:
    obj_names = f.read().strip().split("\n")

# === Load DNN Model ===
model = cv2.dnn_DetectionModel("frozen_inference_graph.pb", "Pretrained_vectors_mobile_net.pbtxt")
model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

# === Target Classes ===
targets = ["Human", "Plastic Bottle"]
BASELINE_CM = 8.0         # Adjust based on camera mount
FOCAL_LENGTH_PX = 630.0   # Pre-calibrated

# === Thread Lock for Buffer ===
frame_lock = threading.Lock()
latest_frame = None

# === Depth History Smoothing ===
depth_history = {}

def smoothed_depth(label, raw_depth, window=5):
    if label not in depth_history:
        depth_history[label] = []
    history = depth_history[label]
    history.append(raw_depth)
    if len(history) > window:
        history.pop(0)
    return round(sum(history) / len(history), 2)

# === Camera Setup ===
picam0 = Picamera2(0)
picam1 = Picamera2(1)

config0 = picam0.create_video_configuration(
    main={"size": (640, 480)},
    controls={
        "FrameDurationLimits": (33333, 33333),
        "AeEnable": True,
        "AwbEnable": True
    }
)

config1 = picam1.create_video_configuration(
    main={"size": (640, 480)},
    controls={
        "FrameDurationLimits": (33333, 33333),
        "AeEnable": True,
        "AwbEnable": True
    }
)

picam0.configure(config0)
picam1.configure(config1)
picam0.start()
picam1.start()
time.sleep(2)  # Warm-up

# === Object Detection ===
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

# === Depth Calculation ===
def compute_depth(center_left, center_right):
    disparity = abs(center_left[0] - center_right[0])
    if disparity < 1:
        return None
    depth_cm = (FOCAL_LENGTH_PX * BASELINE_CM) / disparity
    return round(depth_cm, 2)

# === Detection Matching (Left <-> Right Camera) ===
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

# === Main Loop: Capture, Detect, Decide ===
def update_frames():
    global latest_frame
    while True:
        frame0 = picam0.capture_array()
        frame1 = picam1.capture_array()

        frame0 = cv2.flip(frame0, 0)
        frame1 = cv2.flip(frame1, 0)

        if frame0.shape[2] == 4:
            frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGRA2BGR)
        if frame1.shape[2] == 4:
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGRA2BGR)

        dets0 = detect(frame0)
        dets1 = detect(frame1)
        matches = match_detections(dets0, dets1)

        human_data = []
        bottle_data = []

        for label, box0, box1, center0, center1 in matches:
            raw_depth = compute_depth(center0, center1)
            if not raw_depth:
                continue
            depth = smoothed_depth(label, raw_depth)

            if label.lower() == "human":
                human_data.append((depth, center0[0]))  # (depth, x)
            elif label.lower() == "plastic bottle":
                bottle_data.append((depth, center0[0]))  # (depth, x)

        # === Movement Logic ===
        if human_data:
            # Closest human
            human_data.sort()
            human_depth, human_x = human_data[0]

            if human_depth > 200:
                print("Move Forward")
            elif human_depth > 50:
                if human_x < 200:
                    print("Turn Left Slightly")
                elif human_x > 440:
                    print("Turn Right Slightly")
                else:
                    print("Approaching Human")
            else:
                print("Stop (Human Very Close)")
        else:
            print("No Human Detected. Stop or Rotate.")

        # === Obstacle Avoidance ===
        for bottle_depth, bottle_x in bottle_data:
            if bottle_depth < 70:
                if 200 < bottle_x < 440:
                    print("Obstacle Ahead (Bottle). Stop or Avoid.")
                elif bottle_x <= 200:
                    print("Bottle on Left. Turn Right.")
                else:
                    print("Bottle on Right. Turn Left.")

        time.sleep(0.1)  # Adjust for performance
