from flask import Flask, Response
from picamera2 import Picamera2
import cv2
import numpy as np

# Load COCO labels
with open('coco-labels.txt') as f:
    classes = f.read().splitlines()

# Load Caffe model
net = cv2.dnn.readNetFromCaffe(
    'deploy.prototxt',
    'mobilenet_iter_73000.caffemodel'
)

# Flask app
app = Flask(__name__)

# Setup Picamera2 with lower resolution
camera = Picamera2()
config = camera.create_video_configuration(
    main={"format": "RGB888", "size": (320, 240)}
)
camera.configure(config)
camera.start()

# Target classes and confidence threshold
target_classes = {"person", "bottle", "car"}
confidence_thresh = 0.7

def gen_frames():
    while True:
        frame = camera.capture_array("main")
        (h, w) = frame.shape[:2]

        # Prepare blob
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5
        )
        net.setInput(blob)
        detections = net.forward()

        # Draw boxes
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            class_id = int(detections[0, 0, i, 1])
            label = classes[class_id] if class_id < len(classes) else str(class_id)

            if confidence > confidence_thresh and label in target_classes:
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                (startX, startY, endX, endY) = box.astype('int')
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(
                    frame, f"{label}: {confidence:.2f}",
                    (startX, max(startY - 10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )

        # Encode JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return """
    <html>
    <head><title>Live Object Detection</title></head>
    <body style="background-color:#000; color:#fff; text-align:center; margin:0; padding:0;">
      <h1 style="margin:10px;">📷 Live Object Detection Feed</h1>
      <img src="/video" style="width:60vw; height:auto; border:3px solid white;">
    </body>
    </html>
    """

@app.route('/video')
def video():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
