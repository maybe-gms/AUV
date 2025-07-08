from flask import Flask, Response
from picamera2 import Picamera2
import cv2

# === Load object class names ===
with open("data_items.names", "r") as f:
    obj_names = f.read().strip().split("\n")

# === Load DNN model ===
model = cv2.dnn_DetectionModel("frozen_inference_graph.pb", "Pretrained_vectors_mobile_net.pbtxt")
model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

# === Only detect these objects ===
target_classes = ["Human", "Plastic Bottle"]

# === Flask app ===
app = Flask(__name__)

# === Set up CSI cam ===
camera_csi = Picamera2()
config = camera_csi.create_video_configuration(main={"format": "RGB888", "size": (640, 480)})
camera_csi.configure(config)
camera_csi.start()

# === Set up USB cam (Logitech C270) ===
camera_usb = cv2.VideoCapture(1)  # Adjust index if needed
camera_usb.set(3, 640)
camera_usb.set(4, 480)

# === Crop center of frame to match USB zoom level ===
def crop_center(frame, width=480, height=360):
    h, w, _ = frame.shape
    x = w // 2 - width // 2
    y = h // 2 - height // 2
    return frame[y:y+height, x:x+width]

# === Object detection function ===
def detect_objects(frame, threshold=0.45, nms_thresh=0.6, draw=True, targets=[]):# Oginally 0.1
    result = model.detect(frame, confThreshold=threshold, nmsThreshold=nms_thresh)

    # Unpack results safely
    if len(result) == 3:
        class_ids, confidences, boxes = result
    else:
        class_ids, boxes = result
        confidences = [1.0] * len(class_ids)

    detected_objects = []
    if len(class_ids) != 0:
        for class_id, confidence, box in zip(class_ids, confidences, boxes):
            obj_name = obj_names[class_id - 1]
            if obj_name.lower() in [t.lower() for t in targets] and confidence > threshold:
                detected_objects.append([box, obj_name])
                if draw:
                    color = (0, 0, 255) if "bottle" in obj_name.lower() else (255, 255, 0)
                    cv2.rectangle(frame, box, color, 1)
                    cv2.putText(frame, obj_name.upper(), (box[0]+10, box[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
                    cv2.putText(frame, f'{round(confidence*100,1)}%', (box[0]+10, box[1]+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return frame, detected_objects

# === CSI frame stream ===
def gen_csi():
    while True:
        frame = camera_csi.capture_array("main")
        frame = crop_center(frame, width=480, height=360)
        frame = cv2.resize(frame, (640, 480))
        frame, _ = detect_objects(frame, targets=target_classes)
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# === USB frame stream ===
def gen_usb():
    while True:
        ret, frame = camera_usb.read()
        if not ret:
            continue
        frame, _ = detect_objects(frame, targets=target_classes)
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# === HTML UI ===
@app.route('/')
def index():
    return """
    <html>
    <head><title>Dual Camera Detection</title></head>
    <body style="background:#000; color:white; text-align:center;">
        <h1>🎥 CSI + USB Live Detection</h1>
        <div style="display:flex; justify-content:center; gap:30px;">
            <div>
                <h3>CSI Camera</h3>
                <img src="/video_csi" style="width:480px; border:2px solid white;">
            </div>
            <div>
                <h3>USB Camera</h3>
                <img src="/video_usb" style="width:480px; border:2px solid white;">
            </div>
        </div>
    </body>
    </html>
    """

@app.route('/video_csi')
def video_csi():
    return Response(gen_csi(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_usb')
def video_usb():
    return Response(gen_usb(), mimetype='multipart/x-mixed-replace; boundary=frame')

# === Run the server ===
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
