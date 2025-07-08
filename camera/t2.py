from picamera2 import Picamera2
picam2 = Picamera2()
picam2.start_preview()
import time; time.sleep(5)
picam2.stop_preview()
