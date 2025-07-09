import cv2
from picamera2 import Picamera2
import numpy as np
from flask import Flask, Response
import threading

# === Load Labels ===
with open("data_items.names", "r") as f:
    obj_names = f.read().strip().split("\n")

# === Load DNN Model ===
model = cv2.dnn_DetectionModel("frozen_inference_graph.pb", "Pretrained_vectors_mobile_net.pbtxt")
model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

# === Set Target Classes ===
targets = ["Human", "Plastic Bottle"]

# === Constants for Triangulation ===
BASELINE_CM = 12.0      # Distance between CSI and USB cameras
FOCAL_LENGTH_PX = 615.0 # Approx focal length in pixels for 640x480 (adjust if needed)

# === Flask App ===
app = Flask(__name__)

# === Camera Setup ===
picamera = Picamera2()
picamera.configure(picamera.create_video_configuration(main={"format": "RGB888", "size": (640, 480)}))
picamera.start()

usb_camera = cv2.VideoCapture(1)
usb_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
usb_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_lock = threading.Lock()
latest_frame = None

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

# === Match Detections by Label ===
def match_detections(csi_dets, usb_dets):
    matched = []
    for label1, box1 in csi_dets:
        c1 = (box1[0] + box1[2] // 2, box1[1] + box1[3] // 2)
        for label2, box2 in usb_dets:
            if label1 == label2:
                c2 = (box2[0] + box2[2] // 2, box2[1] + box2[3] // 2)
                matched.append((label1, box1, box2, c1, c2))
                break
    return matched

# === Frame Producer ===
def update_frames():
    global latest_frame
    while True:
        frame_csi = picamera.capture_array("main")
        frame_csi = cv2.cvtColor(frame_csi, cv2.COLOR_RGB2BGR)

        ret, frame_usb = usb_camera.read()
        if not ret:
            continue

        detections_csi = detect(frame_csi)
        detections_usb = detect(frame_usb)

        matches = match_detections(detections_csi, detections_usb)

        for label, box1, box2, center1, center2 in matches:
            depth = compute_depth(center1, center2)
            if depth:
                print(f"{label} Depth: {depth} cm")
                # Draw CSI view
                cv2.rectangle(frame_csi, box1, (0, 255, 0), 1)
                cv2.putText(frame_csi, f"{label} {depth}cm", (box1[0], box1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                # Draw USB view
                cv2.rectangle(frame_usb, box2, (255, 0, 0), 1)
                cv2.putText(frame_usb, f"{label} {depth}cm", (box2[0], box2[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

        stacked = np.hstack((frame_csi, frame_usb))
        ret, buffer = cv2.imencode('.jpg', stacked, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ret:
            continue
        with frame_lock:
            latest_frame = buffer.tobytes()

# === MJPEG Streaming ===
@app.route('/video')
def video():
    def generate():
        while True:
            with frame_lock:
                if latest_frame:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + latest_frame + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return '''
    <html>
        <head><title>Dual Cam Detection + Depth</title></head>
        <body style="margin:0; background-color:#000;">
            <h1 style="text-align:center; color:white;"> CSI | USB Feed with Detection + Depth (cm)</h1>
            <div style="display:flex; justify-content:center;">
                <img src="/video" style="width:95vw; height:auto; border:2px solid white;">
            </div>
        </body>
    </html>
    '''

if __name__ == '__main__':
    threading.Thread(target=update_frames, daemon=True).start()
    app.run(host='0.0.0.0', port=5000)
