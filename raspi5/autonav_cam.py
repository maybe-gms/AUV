from flask import Flask, Response
from picamera2 import Picamera2
import cv2
import threading

# === Stereo Camera Setup ===
picam0 = Picamera2(0)
picam1 = Picamera2(1)
picam0.configure(picam0.create_video_configuration(main={"format":"RGB888","size": (640, 480)}))
picam1.configure(picam1.create_video_configuration(main={"format":"RGB888","size": (640, 480)}))
picam0.start()
picam1.start()

# === Load YOLO or MobileNet SSD Model ===
with open("data_items.names", "r") as f:
    obj_names = f.read().strip().split("\n")

model = cv2.dnn_DetectionModel("frozen_inference_graph.pb", "Pretrained_vectors_mobile_net.pbtxt")
model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)
# === Stereo and Depth Constants ===
BASELINE_CM = 12.0
FOCAL_LENGTH_PIXELS = 640 / (2 * 0.035)  # approx for 35mm FoV on 640px

app = Flask(__name__)

def calculate_depth(x_left, x_right):
    disparity = abs(x_left - x_right)
    if disparity == 0:
        return None
    return round((BASELINE_CM * FOCAL_LENGTH_PIXELS) / disparity, 2)

def get_centroids_and_boxes(frame):
    class_ids, confidences, boxes = model.detect(frame, confThreshold=0.45, nmsThreshold=0.4)
    detections = []
    for class_id, conf, box in zip(class_ids, confidences, boxes):
        label = obj_names[class_id - 1]
        if label in ["Human", "Plastic Bottle"]:
            x, y, w, h = box
            cx = x + w // 2
            detections.append((label, cx, box))
    return detections

def decision_logic(depths):
    if not depths:
        return "No Detection"

    human_depths = [d for l, d in depths if l == "Human"]
    bottle_depths = [d for l, d in depths if l == "Plastic Bottle"]

    if not human_depths:
        return "Searching for Human"

    nearest_human = min(human_depths)
    nearest_bottle = min(bottle_depths) if bottle_depths else float("inf")

    if nearest_bottle < nearest_human and nearest_bottle < 80:
        return "Avoid Obstacle"
    elif nearest_human < 50:
        return "Stop (Human Very Close)"
    elif nearest_human < 90:
        return "Approaching Human"
    else:
        return "Move Forward"

def gen():
    while True:
        left = cv2.flip(picam0.capture_array(), 0)
        right = cv2.flip(picam1.capture_array(), 0)

        if left.shape[2] == 4:
            left = cv2.cvtColor(left, cv2.COLOR_BGRA2BGR)
        if right.shape[2] == 4:
            right = cv2.cvtColor(right, cv2.COLOR_BGRA2BGR)

        det_left = get_centroids_and_boxes(left)
        det_right = get_centroids_and_boxes(right)

        depths = []
        for label_l, cx_l, box_l in det_left:
            for label_r, cx_r, box_r in det_right:
                if label_l == label_r and abs(cx_l - cx_r) < 40:
                    depth = calculate_depth(cx_l, cx_r)
                    if depth:
                        depths.append((label_l, depth))
                        # Draw bounding box and depth
                        x, y, w, h = box_l
                        cv2.rectangle(left, (x, y), (x+w, y+h), (0,255,0), 2)
                        cv2.putText(left, f"{label_l} Depth: {depth}cm", (x, y-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        decision = decision_logic(depths)
        cv2.putText(left, f"Decision: {decision}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        combined = cv2.hconcat([left, right])
        _, jpeg = cv2.imencode('.jpg', combined)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

@app.route('/')
def index():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
