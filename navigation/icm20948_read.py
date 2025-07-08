import smbus2
import time

bus = smbus2.SMBus(1)
ICM_ADDR = 0x69

REG_BANK_SEL = 0x7F
PWR_MGMT_1   = 0x06
WHO_AM_I     = 0x00
ACCEL_XOUT_H = 0x2D
GYRO_XOUT_H  = 0x33

def select_bank(bank):
    bus.write_byte_data(ICM_ADDR, REG_BANK_SEL, bank << 4)
    time.sleep(0.01)

def read_i2c_word(reg):
    high = bus.read_byte_data(ICM_ADDR, reg)
    low = bus.read_byte_data(ICM_ADDR, reg + 1)
    val = (high << 8) | low
    return val - 65536 if val & 0x8000 else val

# Initialize
select_bank(0)
bus.write_byte_data(ICM_ADDR, PWR_MGMT_1, 0x01)
time.sleep(0.1)

whoami = bus.read_byte_data(ICM_ADDR, WHO_AM_I)
print(f"WHO_AM_I = 0x{whoami:X} (should be 0xEA)")

# Sensitivity
ACCEL_SENS = 16384.0  # For ±2g
GYRO_SENS  = 131.0    # For ±250 dps

while True:
    select_bank(0)

    # Read and convert
    ax = read_i2c_word(ACCEL_XOUT_H) / ACCEL_SENS
    ay = read_i2c_word(ACCEL_XOUT_H + 2) / ACCEL_SENS
    az = read_i2c_word(ACCEL_XOUT_H + 4) / ACCEL_SENS

    gx = read_i2c_word(GYRO_XOUT_H) / GYRO_SENS
    gy = read_i2c_word(GYRO_XOUT_H + 2) / GYRO_SENS
    gz = read_i2c_word(GYRO_XOUT_H + 4) / GYRO_SENS

    print(f"Accel (g):     X={ax:.2f}, Y={ay:.2f}, Z={az:.2f}")
    print(f"Gyro  (°/s):   X={gx:.2f}, Y={gy:.2f}, Z={gz:.2f}")
    print("-" * 35)
    time.sleep(0.5)
