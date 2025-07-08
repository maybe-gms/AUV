import smbus2
import time

# MPU9250 Registers
MPU9250_ADDR = 0x68
PWR_MGMT_1 = 0x6B
ACCEL_XOUT_H = 0x3B
GYRO_XOUT_H = 0x43

bus = smbus2.SMBus(1)

def read_word(reg):
    high = bus.read_byte_data(MPU9250_ADDR, reg)
    low = bus.read_byte_data(MPU9250_ADDR, reg+1)
    val = (high << 8) + low
    if val > 32767:
        val -= 65536
    return val

def init_mpu():
    bus.write_byte_data(MPU9250_ADDR, PWR_MGMT_1, 0x00)
    time.sleep(0.1)

def read_sensor_data():
    accel = {
        'x': read_word(ACCEL_XOUT_H),
        'y': read_word(ACCEL_XOUT_H + 2),
        'z': read_word(ACCEL_XOUT_H + 4)
    }
    gyro = {
        'x': read_word(GYRO_XOUT_H),
        'y': read_word(GYRO_XOUT_H + 2),
        'z': read_word(GYRO_XOUT_H + 4)
    }
    return accel, gyro

def interpret_movement(accel, gyro):
    # Convert to g and °/s
    ax, ay, az = accel['x'] / 16384.0, accel['y'] / 16384.0, accel['z'] / 16384.0
    gx, gy, gz = gyro['x'] / 131.0, gyro['y'] / 131.0, gyro['z'] / 131.0

    print(f"Accelerometer (g): x={ax:.2f}, y={ay:.2f}, z={az:.2f}")
    print(f"Gyroscope (°/s): x={gx:.2f}, y={gy:.2f}, z={gz:.2f}")

    # Interpret basic tilt
    if ax > 0.5:
        print("↖ Tilted Left")
    elif ax < -0.5:
        print("↗ Tilted Right")

    if ay > 0.5:
        print("↑ Tilted Forward")
    elif ay < -0.5:
        print("↓ Tilted Backward")

    if az > 0.9:
        print("✔ Lying Flat (Facing Up)")
    elif az < -0.9:
        print("✘ Upside Down")

    # Interpret rotation
    if abs(gx) > 100:
        print("↺ Spinning Side-to-Side")
    if abs(gy) > 100:
        print("↻ Spinning Forward/Backward")
    if abs(gz) > 100:
        print("⟳ Rotating (Yaw)")

    print("-" * 50)

init_mpu()

while True:
    accel_data, gyro_data = read_sensor_data()
    interpret_movement(accel_data, gyro_data)
    time.sleep(1)
