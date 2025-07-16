import cv2
import numpy as np
from picamera2 import Picamera2
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
BASELINE_CM = 12.0        # Distance between cameras
FOCAL_LENGTH_PX = 630.0   # Calibrated focal length

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

config0 = picam0.create_video_configuration(main={"size": (640, 480)})
config1 = picam1.create_video_configuration(main={"size": (640, 480)})

picam0.configure(config0)
picam1.configure(config1)
picam0.start()
picam1.start()
time.sleep(2)

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
    return round((FOCAL_LENGTH_PX * BASELINE_CM) / disparity, 2)

# === Match objects by label ===
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

# === Main Loop ===
print("Running... Press Ctrl+C to stop.")
try:
    while True:
        frame0 = cv2.flip(picam0.capture_array(), 0)
        frame1 = cv2.flip(picam1.capture_array(), 0)

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
                human_data.append((depth, center0[0]))
            elif label.lower() == "plastic bottle":
                bottle_data.append((depth, center0[0]))

        # === Movement Decision ===
        def get_zone(x):
            if x < 213:
                return "right"
            elif x > 426:
                return "left"
            else:
                return "center"

        movement_decision = ""
        obstacle_blocking = False

        for bottle_depth, bottle_x in bottle_data:
            zone = get_zone(bottle_x)
            if bottle_depth < 100 and zone == "center":
                obstacle_blocking = True
                print(f"Obstacle (Bottle) Ahead at {bottle_depth} cm, Zone: {zone}")
                movement_decision = "Bottle Ahead. Rerouting..."
                if bottle_x < 320:
                    movement_decision += " Turn RIGHT."
                else:
                    movement_decision += " Turn LEFT."
                break

        if human_data:
            human_data.sort()
            human_depth, human_x = human_data[0]
            human_zone = get_zone(human_x)
            print(f"Human Detected at {human_depth} cm, Zone: {human_zone}")

            if obstacle_blocking:
                print(f"Decision: {movement_decision}\n")
            else:
                if human_zone == "center":
                    if human_depth > 200:
                        print("Decision: Human Centered. MOVE FORWARD FAST.\n")
                    elif human_depth > 50:
                        print("Decision: Human Centered. Approaching.\n")
                    else:
                        print("Decision: Human Very Close. STOP.\n")
                elif human_zone == "left":
                    print("Decision: Human on Left. TURN RIGHT.\n")  # Robot's POV
                else:
                    print("Decision: Human on Right. TURN LEFT.\n")  # Robot's POV
        else:
            if not obstacle_blocking:
                print("Decision: No Human. Rotate to Search.\n")
            else:
                print(f"Decision: {movement_decision}\n")

        time.sleep(0.1)

except KeyboardInterrupt:
    print("Stopped.")
