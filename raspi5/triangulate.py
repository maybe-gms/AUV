import cv2
import numpy as np
from flask import Flask, Response
from picamera2 import Picamera2
import threading
import time

# === Flask App ===
app = Flask(__name__)

# === Load Labels ===
with open("data_items.names", "r") as f:
    obj_names = f.read().strip().split("\n")

# === Load DNN Model ===
model = cv2.dnn_DetectionModel("frozen_inference_graph.pb", "Pretrained_vectors_mobile_net.pbtxt")
model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

# === Target Classes and Camera Constants ===
targets = ["Human", "Plastic Bottle"]
BASELINE_CM = 12.0           # Distance between cameras in cm did change to 8.0
FOCAL_LENGTH_PX = 625.0      # Approx focal length in pixels for OV5647 at 640x480

# === Shared Frame Storage ===
frame_lock = threading.Lock()
latest_frame = None

# === Camera Setup ===
picam0 = Picamera2(0)
picam1 = Picamera2(1)

config0 = picam0.create_video_configuration(main={"size": (640, 480)})
config1 = picam1.create_video_configuration(main={"size": (640, 480)})

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

# === Triangulation ===
def compute_depth(center_left, center_right):
    disparity = abs(center_left[0] - center_right[0])
    if disparity == 0:
        return None
    depth_cm = (FOCAL_LENGTH_PX * BASELINE_CM) / disparity
    return round(depth_cm, 2)

# === Match Detected Objects ===
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

# === Frame Update Thread ===
def update_frames():
    global latest_frame
    while True:
        frame0 = picam0.capture_array()
        frame1 = picam1.capture_array()

        # Convert 4-channel to 3-channel if needed
        if frame0.shape[2] == 4:
            frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGRA2BGR)
        if frame1.shape[2] == 4:
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGRA2BGR)

        dets0 = detect(frame0)
        dets1 = detect(frame1)
        matches = match_detections(dets0, dets1)

        for label, box0, box1, center0, center1 in matches:
            depth = compute_depth(center0, center1)
            if depth:
                print(f"{label} Depth: {depth} cm")
                cv2.rectangle(frame0, box0, (0, 255, 0), 1)
                cv2.putText(frame0, f"{label} {depth}cm", (box0[0], box0[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.rectangle(frame1, box1, (255, 0, 0), 1)
                cv2.putText(frame1, f"{label} {depth}cm", (box1[0], box1[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        stacked = np.hstack((frame0, frame1))
        ret, buffer = cv2.imencode('.jpg', stacked, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ret:
            continue
        with frame_lock:
            latest_frame = buffer.tobytes()

# === MJPEG Streaming Route ===
@app.route('/video')
def video():
    def generate():
        while True:
            with frame_lock:
                if latest_frame:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + latest_frame + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# === Basic HTML Frontend ===
@app.route('/')
def index():
    return '''
    <html>
        <head><title>Stereo CSI Depth</title></head>
        <body style="background-color:#000;">
            <h1 style="text-align:center; color:white;">CSI0 | CSI1 Stereo View with Depth</h1>
            <div style="display:flex; justify-content:center;">
                <img src="/video" style="width:95vw; height:auto; border:2px solid white;">
            </div>
        </body>
    </html>
    '''

# === Start Everything ===
if __name__ == '__main__':
    threading.Thread(target=update_frames, daemon=True).start()
    app.run(host='0.0.0.0', port=5000)
