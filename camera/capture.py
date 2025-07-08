from picamera2 import Picamera2
import time

picam2 = Picamera2()
picam2.start_preview()
time.sleep(2)  # Let camera warm up
picam2.capture_file("photo.jpg")
picam2.stop_preview()

print("Photo saved as photo.jpg")
