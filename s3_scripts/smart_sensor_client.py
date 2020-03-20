#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Smart Sensor Client: ASHRAE Environmental Monitoring System.
usage: sudo ./smart_sensor_client.py
Python version used: 3.6.8
See requirements.txt for additional dependencies
Styling guide: PEP 8 -- Style Guide for Python Code
    (https://www.python.org/dev/peps/pep-0008/) and
    PEP 257 -- Docstring Conventions
    (https://www.python.org/dev/peps/pep-0257/)
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import ctypes
import smbus
import socket
import time

# Module metadata dunders
__author__ = "Rob Garcia"
__copyright__ = "Copyright 2019-2020, Rob Garcia"
__email__ = "rgarcia@rgprogramming.com"
__license__ = "MIT"

# Waveshare Sense HAT (B) information
HOST = "192.168.1.12"
PORT = 333

# I2C address
LPS22HB_I2C_ADDRESS = 0x5C

LPS_ID = 0xB1
# Register
# Interrupt register
LPS_INT_CFG = 0x0B
# Pressure threshold registers
LPS_THS_P_L = 0x0C
LPS_THS_P_H = 0x0D
# Who am I
LPS_WHO_AM_I = 0x0F
# Control registers
LPS_CTRL_REG1 = 0x10
LPS_CTRL_REG2 = 0x11
LPS_CTRL_REG3 = 0x12
# FIFO configuration register
LPS_FIFO_CTRL = 0x14
# Reference pressure registers
LPS_REF_P_XL = 0x15
LPS_REF_P_L = 0x16
LPS_REF_P_H = 0x17
# Pressure offset registers
LPS_RPDS_L = 0x18
LPS_RPDS_H = 0x19
# Resolution register
LPS_RES_CONF = 0x1A
# Interrupt register
LPS_INT_SOURCE = 0x25
# FIFO status register
LPS_FIFO_STATUS = 0x26
# Status register
LPS_STATUS = 0x27
# Pressure output registers
LPS_PRESS_OUT_XL = 0x28
LPS_PRESS_OUT_L = 0x29
LPS_PRESS_OUT_H = 0x2A
# Temperature output registers
LPS_TEMP_OUT_L = 0x2B
LPS_TEMP_OUT_H = 0x2C
# Filter reset register
LPS_RES = 0x33


class SHTC3:
    """Class for SHTC3: Temperature and Humidity sensor."""

    def __init__(self):
        """Constructor"""
        self.dll = ctypes.CDLL("./SHTC3.so")
        init = self.dll.init
        init.restype = ctypes.c_int
        init.argtypes = [ctypes.c_void_p]
        init(None)

    def SHTC3_Read_Temperature(self):
        """Get the temperature reading in Celsius
        :return: The temperature data collected.
        :rtype: float
        """
        temperature = self.dll.SHTC3_Read_TH
        temperature.restype = ctypes.c_float
        temperature.argtypes = [ctypes.c_void_p]
        return temperature(None)

    def SHTC3_Read_Humidity(self):
        """Get the relative humidity reading in percentage
        :return: The humidity data collected.
        :rtype: float
        """
        humidity = self.dll.SHTC3_Read_RH
        humidity.restype = ctypes.c_float
        humidity.argtypes = [ctypes.c_void_p]
        return humidity(None)


class LPS22HB(object):
    """Class for SHTC3: Barometric Pressure and Temperature sensor."""

    def __init__(self, address=LPS22HB_I2C_ADDRESS):
        """Constructor"""
        self._address = address
        self._bus = smbus.SMBus(1)
        # Wait for reset to complete
        self.LPS22HB_RESET()
        # Low-pass filter disabled, output registers not updated until
        # MSB and LSB have been read,
        # Enable Block Data Update, Set Output Data Rate to 0
        self._write_byte(LPS_CTRL_REG1, 0x02)

    def LPS22HB_RESET(self):
        Buf = self._read_u16(LPS_CTRL_REG2)
        Buf |= 0x04
        # SWRESET Set 1
        self._write_byte(LPS_CTRL_REG2, Buf)
        while Buf:
            Buf = self._read_u16(LPS_CTRL_REG2)
            Buf &= 0x04

    def LPS22HB_START_ONESHOT(self):
        Buf = self._read_u16(LPS_CTRL_REG2)
        # ONE_SHOT Set 1
        Buf |= 0x01
        self._write_byte(LPS_CTRL_REG2, Buf)

    def _read_byte(self, cmd):
        return self._bus.read_byte_data(self._address, cmd)

    def _read_u16(self, cmd):
        LSB = self._bus.read_byte_data(self._address, cmd)
        MSB = self._bus.read_byte_data(self._address, cmd+1)
        return (MSB << 8) + LSB

    def _write_byte(self, cmd, val):
        self._bus.write_byte_data(self._address, cmd, val)


def collect_sensor_data():
    """Collects temperature and humidity data
    :return: The temperature and humidity data collected, formated with
             atmospheric pressure, air speed, metabolic rate, and clothing level.
    :rtype: list of tuples
    """
    # Instantiate the classes of the sensors
    lps22hb = LPS22HB()
    shtc3 = SHTC3()
    temp = 0.0
    humid = 0.0
    pressure = 0.0
    u8Buf = [0, 0, 0]
    sensor_data = []
    try:
        while True:
            time.sleep(0.1)
            lps22hb.LPS22HB_START_ONESHOT()
            if (lps22hb._read_byte(LPS_STATUS) & 0x01) == 0x01:  # a new pressure data is generated
                u8Buf[0] = lps22hb._read_byte(LPS_PRESS_OUT_XL)
                u8Buf[1] = lps22hb._read_byte(LPS_PRESS_OUT_L)
                u8Buf[2] = lps22hb._read_byte(LPS_PRESS_OUT_H)
                pressure = ((u8Buf[2] << 16) +
                            (u8Buf[1] << 8) + u8Buf[0]) / 4096.0
                temp = shtc3.SHTC3_Read_Temperature()
                humid = shtc3.SHTC3_Read_Humidity()
                break
    except (IOError, TypeError) as ex:
        print("Error: {}".format(str(ex)))
    except KeyboardInterrupt:
        pass
    return pressure, humid, temp


def main():
    """Application entry point."""
    print("Starting Smart Sensor Client...")
    response = ""
    sensor_data = []
    while True:
        # Collect three samples
        for y in range(3):
            print(
                "Collecting atmospheric pressure, temperature, and relative humidity...")
            pressure, humid, temp = collect_sensor_data()
            sensor_data.append(
                ([[round(pressure, 2), 0.1, round(humid, 2), 1.0, 0.61, round(temp, 2)]]))
        for index, data in enumerate(sensor_data):
            print("Sample #{0} collected: {1}".format(index + 1, data))
        print("Sending data...")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            # Convert to byte literal
            data = str(sensor_data).encode("utf-8")
            s.sendall(data)
            # Get a response
            response = s.recv(1024)
            response = response.decode("utf-8")
            if response == "terminate":
                break
            else:
                print("Received command for {} second delay from server.".format(response))
            sensor_data = []
            # Delay to allow serer to process data
            time.sleep(int(response))
    print("Shutting down client. Have an excellent day.")


if __name__ == "__main__":
    main()
