import time
import cv2
from picamera2 import Picamera2

from detection import detect, match_detections, compute_depth, smoothed_depth
import motor_control as motor

# Camera Setup
picam0 = Picamera2(0)
picam1 = Picamera2(1)
config0 = picam0.create_video_configuration(main={"size": (640, 480)})
config1 = picam1.create_video_configuration(main={"size": (640, 480)})
picam0.configure(config0)
picam1.configure(config1)
picam0.start()
picam1.start()
time.sleep(2)

motor.setup()

def get_zone(x):
    if x < 213:
        return "right"
    elif x > 426:
        return "left"
    else:
        return "center"

print("Running... Press Ctrl+C to stop.")
try:
    while True:
        frame0 = cv2.flip(picam0.capture_array(), 0)
        frame1 = cv2.flip(picam1.capture_array(), 0)

        if frame0.shape[2] == 4:
            frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGRA2BGR)
        if frame1.shape[2] == 4:
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGRA2BGR)

        dets0 = detect(frame0)
        dets1 = detect(frame1)
        matches = match_detections(dets0, dets1)

        human_data = []
        bottle_data = []

        for label, box0, box1, center0, center1 in matches:
            raw_depth = compute_depth(center0, center1)
            if not raw_depth:
                continue
            depth = smoothed_depth(label, raw_depth)
            if label.lower() == "human":
                human_data.append((depth, center0[0]))
            elif label.lower() == "plastic bottle":
                bottle_data.append((depth, center0[0]))

        movement_decision = ""
        obstacle_blocking = False

        for bottle_depth, bottle_x in bottle_data:
            zone = get_zone(bottle_x)
            if bottle_depth < 100 and zone == "center":
                obstacle_blocking = True
                print(f"Obstacle (Bottle) Ahead at {bottle_depth} cm, Zone: {zone}")
                movement_decision = "Bottle Ahead. Rerouting..."
                if bottle_x < 320:
                    movement_decision += " Turn RIGHT."
                    motor.turn_right()
                else:
                    movement_decision += " Turn LEFT."
                    motor.turn_left()
                break

        if human_data:
            human_data.sort()
            human_depth, human_x = human_data[0]
            human_zone = get_zone(human_x)
            print(f"Human Detected at {human_depth} cm, Zone: {human_zone}")

            if obstacle_blocking:
                print(f"Decision: {movement_decision}\n")
            else:
                if human_zone == "center":
                    if human_depth > 200:
                        print("Decision: Human Centered. MOVE FORWARD FAST.\n")
                        motor.move_forward()
                    elif human_depth > 50:
                        print("Decision: Human Centered. Approaching.\n")
                        motor.move_forward()
                    else:
                        print("Decision: Human Very Close. STOP.\n")
                        motor.stop()
                elif human_zone == "left":
                    print("Decision: Human on Left. TURN RIGHT.\n")
                    motor.turn_right()
                else:
                    print("Decision: Human on Right. TURN LEFT.\n")
                    motor.turn_left()
        else:
            if not obstacle_blocking:
                print("Decision: No Human. Rotate to Search.\n")
                motor.turn_right()
            else:
                print(f"Decision: {movement_decision}\n")

        time.sleep(0.1)

except KeyboardInterrupt:
    motor.cleanup()
    print("Stopped.")
