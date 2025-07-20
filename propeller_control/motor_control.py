from gpiozero import Motor#,Robot
from time import sleep

left = Motor(forward=17, backward=27)     
right = Motor(forward=23, backward=24)
#both = Robot(left=(17, 27), right=(23, 24))

def move_forward():
    #both.forward()
    left.forward()
    #print("Moving left")
    right.forward()
    #print("Moving right")     

def turn_right():
    left.forward()     
    right.stop()  

def turn_left():
    left.stop()
    right.forward()      

def stop():
    left.stop()
    right.stop()
