#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Sensor and LCD output test.
Python version used: 3.6.8
See requirements.txt for additional dependencies
Styling guide: PEP 8 -- Style Guide for Python Code
    (https://www.python.org/dev/peps/pep-0008/) and
    PEP 257 -- Docstring Conventions
    (https://www.python.org/dev/peps/pep-0257/)
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import math
import time

import grovepi
import grove_rgb_lcd

# Module metadata dunders
__author__ = "Rob Garcia"
__copyright__ = "Copyright 2019-2020, Rob Garcia"
__email__ = "rgarcia@rgcoding.com"
__license__ = "MIT"

BLUE_DHT = 0         # For DHT11
# WHITE_DHT = 1       # For DHT22
GREEN_LED = 5        # Digital port 5
RED_LED = 6          # Digital port 6
DHT_SENSOR_PORT = 7  # Digital port 7
ON = 1
OFF = 0


def sensor_test():
    # type: () -> None
    """Collects and displays temperature and humidity data"""
    sensor_data = []

    for _ in range(10):
        try:
            for _ in range(3):
                [temp, humid] = grovepi.dht(DHT_SENSOR_PORT, BLUE_DHT)
                if not math.isnan(temp) and not math.isnan(humid):
                    if temp >= 20.0:
                        # Temperature is good: Everything is green.
                        grove_rgb_lcd.setRGB(0, 255, 0)
                        grovepi.digitalWrite(GREEN_LED, ON)
                        time.sleep(2)
                        grovepi.digitalWrite(GREEN_LED, OFF)
                    else:
                        # Temperature is too low: Turn to red.
                        grove_rgb_lcd.setRGB(255, 0, 0)
                        grovepi.digitalWrite(RED_LED, ON)
                        time.sleep(2)
                        grovepi.digitalWrite(RED_LED, OFF)
                    t_str = str(temp)
                    h_str = str(humid)
                    print("Temperature: {}C | Humidity: {}%".format(t_str, h_str))
                    grove_rgb_lcd.setText_norefresh(
                        "T: {} C\nH: {} %".format(temp, humid))
                    sensor_data.append([temp, humid])
                # For DHT11, wait three seconds before next reading
                time.sleep(3)
        except (IOError, TypeError) as ex:
            print("Error: {}".format(str(ex)))
            shutdown_board()
        except KeyboardInterrupt:
            shutdown_board()
        print(sensor_data)
        # Wait 30 seconds before collecting next set of data
        grove_rgb_lcd.setRGB(0, 0, 0)
        time.sleep(30)
    shutdown_board()


def shutdown_board():
    # type: () -> None
    """Turns off LEDs and clears LCD screen"""
    grovepi.digitalWrite(RED_LED, OFF)
    grovepi.digitalWrite(GREEN_LED, OFF)
    grove_rgb_lcd.setRGB(0, 0, 0)
    grove_rgb_lcd.setText("")
    print("Job complete. Have an excellent day.")


def main():
    """Application entry point."""
    print("Test of the GrovePi DHT sensor, LEDs and RGB LCD screen.\n")
    sensor_test()


if __name__ == "__main__":
    main()
