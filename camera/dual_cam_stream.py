from flask import Flask, Response
from picamera2 import Picamera2
import cv2

app = Flask(__name__)

# ✅ Set up CSI cam (Picamera2)
picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"format": "RGB888", "size": (640, 480)}
)
picam2.configure(config)
picam2.start()

# ✅ Set up USB cam (adjust index if needed)
usb_cam = cv2.VideoCapture(0)  # usually 0 or 1 depending on device

def gen_csi_frames():
    """Generate MJPEG frames from CSI cam."""
    while True:
        frame = picam2.capture_array()
        ret, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ret:
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

def gen_usb_frames():
    """Generate MJPEG frames from USB cam."""
    while True:
        success, frame = usb_cam.read()
        if not success:
            continue
        ret, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ret:
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

@app.route('/')
def index():
    return """
    <html>
    <head><title>Dual Camera Feed</title></head>
    <body style="margin:0; background:#000; color:#fff; text-align:center;">
        <h1 style="padding:10px;">📷 Dual Camera Feed</h1>
        <div style="display:flex; justify-content:center; gap:10px; flex-wrap:wrap;">
            <img src="/csi" style="width:45vw; border:3px solid #fff;" />
            <img src="/usb" style="width:45vw; border:3px solid #fff;" />
        </div>
    </body>
    </html>
    """

@app.route('/csi')
def csi_feed():
    return Response(gen_csi_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/usb')
def usb_feed():
    return Response(gen_usb_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
