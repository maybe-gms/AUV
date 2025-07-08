from flask import Flask, Response
from picamera2 import Picamera2
import cv2

app = Flask(__name__)

# Setup camera in RGB mode
camera = Picamera2()
config = camera.create_video_configuration(main={"format": "RGB888", "size": (1280, 720)})
camera.configure(config)
camera.start()

def gen_frames():
    while True:
        frame = camera.capture_array("main")  # Already in RGB
        # DO NOT convert to BGR — avoid color swap
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return """
        <html>
        <head><title>Live Camera Feed</title></head>
        <body style="margin:0; background-color:#000;">
            <h1 style="text-align:center; color:white;">Camera Feed</h1>
            <div style="display:flex; justify-content:center;">
                <img src="/video" style="width:90vw; height:auto; border:5px solid white;">
            </div>
        </body>
        </html>
    """

@app.route('/video')
def video():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
