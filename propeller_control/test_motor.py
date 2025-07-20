import motor_control as motor
import time

print("Moving forward")
motor.move_forward()
time.sleep(3)
motor.stop()

print("Turning right")
motor.turn_right()
time.sleep(3)
motor.stop()

print("Turning left")
motor.turn_left()
time.sleep(3)
motor.stop()