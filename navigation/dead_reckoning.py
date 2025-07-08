import smbus2
import time
import math
import numpy as np

# Constants
MPU_ADDR = 0x68  # MPU-9250 I2C address
DEG_TO_RAD = math.pi / 180
TIME_STEP = 0.1  # Time step in seconds
ALPHA = 0.9  # Complementary filter coefficient

# Initialize variables
x, y = 0.0, 0.0  # Position
velocity_x, velocity_y = 0.0, 0.0  # Velocity
heading = 0.0  # Current heading
accel_filtered_x, accel_filtered_y = 0.0, 0.0  # Filtered acceleration

# Open file for logging data
file = open("dead_reckoning_log.txt", "w")
file.write("Time,X,Y,Heading\n")  # Write CSV header

# Initialize I2C bus
bus = smbus2.SMBus(1)

# Initialize MPU-9250
def initialize_mpu9250():
    bus.write_byte_data(MPU_ADDR, 0x6B, 0)  # Wake up MPU-9250
    time.sleep(0.1)
    print("MPU-9250 Initialized.")

# Read 16-bit word from sensor
def read_word_2c(addr, reg):
    high = bus.read_byte_data(addr, reg)
    low = bus.read_byte_data(addr, reg + 1)
    value = (high << 8) + low
    if value >= 0x8000:
        value = -((65535 - value) + 1)
    return value

# Get acceleration (m/s²) with filtering
def get_acceleration():
    global accel_filtered_x, accel_filtered_y
    raw_ax = read_word_2c(MPU_ADDR, 0x3B) / 16384.0  # X-axis
    raw_ay = read_word_2c(MPU_ADDR, 0x3D) / 16384.0  # Y-axis

    # Apply Low-Pass Filter
    accel_filtered_x = ALPHA * accel_filtered_x + (1 - ALPHA) * raw_ax
    accel_filtered_y = ALPHA * accel_filtered_y + (1 - ALPHA) * raw_ay

    return accel_filtered_x, accel_filtered_y

# Get angular velocity (Yaw rate)
def get_angular_velocity():
    gyro_z = read_word_2c(MPU_ADDR, 0x47) / 131.0  # Gyro Z-axis (Yaw rate)
    return gyro_z

# Dead reckoning loop
def dead_reckoning():
    global x, y, velocity_x, velocity_y, heading

    try:
        while True:
            start_time = time.time()

            # Read sensor data
            ax, ay = get_acceleration()
            angular_velocity = get_angular_velocity()
            heading += angular_velocity * TIME_STEP
            heading_rad = heading * DEG_TO_RAD

            # Update velocity (Integrate acceleration)
            velocity_x += ax * TIME_STEP
            velocity_y += ay * TIME_STEP

            # Update position (Integrate velocity)
            x += velocity_x * math.cos(heading_rad) * TIME_STEP
            y += velocity_y * math.sin(heading_rad) * TIME_STEP

            # Write data to file
            file.write(f"{time.time()},{x:.2f},{y:.2f},{heading:.2f}\n")

            # Print results
            print(f"X: {x:.2f}, Y: {y:.2f}, Heading: {heading:.2f}°")

            # Wait for next time step
            time.sleep(max(0, TIME_STEP - (time.time() - start_time)))

    except KeyboardInterrupt:
        print("Stopping dead reckoning...")
        file.close()  # Ensure file is properly closed

# Initialize sensors
initialize_mpu9250()

# Start dead reckoning
dead_reckoning()
