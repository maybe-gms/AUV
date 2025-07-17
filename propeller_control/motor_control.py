import RPi.GPIO as GPIO

# Define GPIO pins
LEFT_MOTOR_IN1 = 17
LEFT_MOTOR_IN2 = 27
RIGHT_MOTOR_IN3 = 23
RIGHT_MOTOR_IN4 = 24

motor_pins = [LEFT_MOTOR_IN1, LEFT_MOTOR_IN2, RIGHT_MOTOR_IN3, RIGHT_MOTOR_IN4]

def setup():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    for pin in motor_pins:
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)

def move_forward():
    GPIO.output(LEFT_MOTOR_IN1, GPIO.HIGH)
    GPIO.output(LEFT_MOTOR_IN2, GPIO.LOW)
    GPIO.output(RIGHT_MOTOR_IN3, GPIO.HIGH)
    GPIO.output(RIGHT_MOTOR_IN4, GPIO.LOW)

def turn_left():
    GPIO.output(LEFT_MOTOR_IN1, GPIO.LOW)
    GPIO.output(LEFT_MOTOR_IN2, GPIO.LOW)
    GPIO.output(RIGHT_MOTOR_IN3, GPIO.HIGH)
    GPIO.output(RIGHT_MOTOR_IN4, GPIO.LOW)

def turn_right():
    GPIO.output(LEFT_MOTOR_IN1, GPIO.HIGH)
    GPIO.output(LEFT_MOTOR_IN2, GPIO.LOW)
    GPIO.output(RIGHT_MOTOR_IN3, GPIO.LOW)
    GPIO.output(RIGHT_MOTOR_IN4, GPIO.LOW)

def stop():
    for pin in motor_pins:
        GPIO.output(pin, GPIO.LOW)

def cleanup():
    stop()
    GPIO.cleanup()
