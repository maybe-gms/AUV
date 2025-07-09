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

# === Flask App ===
app = Flask(__name__)

# === Camera Setup ===
picamera = Picamera2()
picamera.configure(picamera.create_video_configuration(main={"format": "RGB888", "size": (640, 480)}))
picamera.start()

usb_camera = cv2.VideoCapture(1)
usb_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
usb_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
usb_camera.set(cv2.CAP_PROP_FPS, 30)

frame_lock = threading.Lock()
latest_frame = None

# === Object Detection ===
def detect_and_draw(frame, color=(0, 255, 0)):
    result = model.detect(frame, confThreshold=0.45, nmsThreshold=0.4)
    if len(result) == 3:
        class_ids, confidences, boxes = result
        for class_id, confidence, box in zip(class_ids, confidences, boxes):
            if class_id <= len(obj_names):
                label = obj_names[class_id - 1]
                if label.lower() in [t.lower() for t in targets]:
                    cv2.rectangle(frame, box, color, 2)
                    cv2.putText(frame, f"{label} {round(confidence * 100)}%", 
                                (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return frame

# === Frame Producer Thread ===
def update_frames():
    global latest_frame
    while True:
        # Get frames
        csi_frame = picamera.capture_array("main")
        csi_frame = cv2.cvtColor(csi_frame, cv2.COLOR_RGB2BGR)

        ret, usb_frame = usb_camera.read()
        if not ret:
            continue

        # Detect and draw
        detect_and_draw(csi_frame, color=(0, 255, 0))    # Green for CSI
        detect_and_draw(usb_frame, color=(255, 0, 0))    # Blue for USB

        # Combine horizontally
        combined = np.hstack((csi_frame, usb_frame))
        ret, buffer = cv2.imencode('.jpg', combined, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ret:
            continue

        with frame_lock:
            latest_frame = buffer.tobytes()

# === Stream Route ===
@app.route('/video')
def video():
    def generate():
        while True:
            with frame_lock:
                if latest_frame:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + latest_frame + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# === Main Page ===
@app.route('/')
def index():
    return '''
    <html>
    <head><title>Dual Camera Detection</title></head>
    <body style="margin:0; background:#000; color:white; text-align:center;">
        <h1>🔍 CSI | USB Feed with Detection</h1>
        <div style="display:flex; justify-content:center;">
            <img src="/video" style="width:95vw; border:2px solid white;">
        </div>
    </body>
    </html>
    '''

# === Launch ===
if __name__ == '__main__':
    threading.Thread(target=update_frames, daemon=True).start()
    app.run(host='0.0.0.0', port=5000)
