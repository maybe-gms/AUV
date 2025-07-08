import smbus2

bus = smbus2.SMBus(1)
MPU_ADDR = 0x68
WHO_AM_I = 0x75

who_am_i = bus.read_byte_data(MPU_ADDR, WHO_AM_I)
print(f"WHO_AM_I register value: 0x{who_am_i:X}")
