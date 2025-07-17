import motor_control as motor
import time

motor.setup()
motor.move_forward()
time.sleep(2)
motor.turn_left()
time.sleep(2)
motor.turn_right()
time.sleep(2)
motor.stop()
motor.cleanup()
