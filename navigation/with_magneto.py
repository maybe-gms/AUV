import smbus2
import time

bus = smbus2.SMBus(1)
MPU_ADDR = 0x68  # MPU9250 default I2C address

# MPU9250 register addresses
PWR_MGMT_1 = 0x6B
ACCEL_XOUT_H = 0x3B
GYRO_XOUT_H = 0x43
MAG_ADDR = 0x0C
MAG_XOUT_L = 0x03

# Wake up MPU9250
bus.write_byte_data(MPU_ADDR, PWR_MGMT_1, 0)

def read_raw_data(addr):
    high = bus.read_byte_data(MPU_ADDR, addr)
    low = bus.read_byte_data(MPU_ADDR, addr+1)
    value = (high << 8) | low
    if value > 32768:
        value -= 65536
    return value

# Enable magnetometer passthrough
bus.write_byte_data(MPU_ADDR, 0x37, 0x02)  # INT_PIN_CFG - Bypass enable
bus.write_byte_data(MAG_ADDR, 0x0A, 0x16)  # 100Hz continuous measurement mode

while True:
    # ACCELEROMETER
    acc_x = read_raw_data(ACCEL_XOUT_H) / 16384.0
    acc_y = read_raw_data(ACCEL_XOUT_H + 2) / 16384.0
    acc_z = read_raw_data(ACCEL_XOUT_H + 4) / 16384.0

    # GYROSCOPE
    gyro_x = read_raw_data(GYRO_XOUT_H) / 131.0
    gyro_y = read_raw_data(GYRO_XOUT_H + 2) / 131.0
    gyro_z = read_raw_data(GYRO_XOUT_H + 4) / 131.0

    # MAGNETOMETER (AK8963)
    try:
        # Check data ready bit
        if bus.read_byte_data(MAG_ADDR, 0x02) & 0x01:
            mag_x = (bus.read_byte_data(MAG_ADDR, MAG_XOUT_L + 1) << 8 | bus.read_byte_data(MAG_ADDR, MAG_XOUT_L)) / 0.6
            mag_y = (bus.read_byte_data(MAG_ADDR, MAG_XOUT_L + 3) << 8 | bus.read_byte_data(MAG_ADDR, MAG_XOUT_L + 2)) / 0.6
            mag_z = (bus.read_byte_data(MAG_ADDR, MAG_XOUT_L + 5) << 8 | bus.read_byte_data(MAG_ADDR, MAG_XOUT_L + 4)) / 0.6
        else:
            mag_x = mag_y = mag_z = 0.0
    except:
        mag_x = mag_y = mag_z = 0.0

    # Display in layman format
    print("\n📈 Accelerometer (tilt/movement):")
    print(f"X: {acc_x:.2f} g, Y: {acc_y:.2f} g, Z: {acc_z:.2f} g")

    print("🌀 Gyroscope (rotation):")
    print(f"X: {gyro_x:.2f} °/s, Y: {gyro_y:.2f} °/s, Z: {gyro_z:.2f} °/s")

    print("🧲 Magnetometer (direction):")
    print(f"X: {mag_x:.2f} µT, Y: {mag_y:.2f} µT, Z: {mag_z:.2f} µT")

    time.sleep(1)
