from flask import Flask, Response
from picamera2 import Picamera2
import cv2

# Load class names
with open("data_items.names", "r") as f:
    obj_names = f.read().strip().split("\n")

# Load DNN model
model = cv2.dnn_DetectionModel("frozen_inference_graph.pb", "Pretrained_vectors_mobile_net.pbtxt")
model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

# Only detect these objects
target_classes = ["Human", "Plastic Bottle"]

# Flask setup
app = Flask(__name__)

# Camera setup
camera = Picamera2()
config = camera.create_video_configuration(main={"format": "RGB888", "size": (320, 240)})#should probably be 640x480
camera.configure(config)
camera.start()

# Object detection and frame generation
def detect_objects(frame, threshold=0.45, nms_thresh=0.1, draw=True, targets=[]):
    result = model.detect(frame, confThreshold=threshold, nmsThreshold=nms_thresh)

    # Handle OpenCV versions that return 2 or 3 items
    if len(result) == 3:
        class_ids, confidences, boxes = result
    else:
        class_ids, boxes = result
        confidences = [1.0] * len(class_ids)  # assume 100% confidence if not returned

    detected_objects = []
    if len(class_ids) != 0:
        for class_id, confidence, box in zip(class_ids, confidences, boxes):
            obj_name = obj_names[class_id - 1]
            if obj_name.lower() in [t.lower() for t in targets]:
                detected_objects.append([box, obj_name])
                if draw:
                    color = (0, 0, 255) if obj_name.lower() == "bottle" else (255, 255, 0)
                    cv2.rectangle(frame, box, color, 1)
                    cv2.putText(frame, obj_name.upper(), (box[0]+10, box[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
                    cv2.putText(frame, f'{round(confidence*100,1)}%', (box[0]+10, box[1]+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return frame, detected_objects


def gen_frames():
    while True:
        frame = camera.capture_array("main")
        frame, _ = detect_objects(frame, targets=target_classes)
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return """
        <html>
        <head><title>Live Object Detection</title></head>
        <body style="margin:0; background-color:#000;">
            <h1 style="text-align:center; color:white;">Detect CSI</h1>
            <div style="display:flex; justify-content:center;">
                <img src="/video" style="width:480px; height:auto; border:2px solid white;">
            </div>
        </body>
        </html>
    """

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
