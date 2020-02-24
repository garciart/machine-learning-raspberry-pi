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

import pandas as pd
from grovepi import *
from grove_rgb_lcd import *

# Module metadata dunders
__author__ = "Rob Garcia"
__copyright__ = "Copyright 2019-2020, Rob Garcia"
__email__ = "rgarcia@rgprogramming.com"
__license__ = "MIT"

def sensors_only_test():
	"""Collects and displays temperature and humidity data"""
	blue_dht = 0        # For DHT11
	white_dht = 1       # For DHT22
	red_led = 5         # Digital port 5
	green_led = 6       # Digital port 6
	dht_sensor_port = 7 # Digital port 7

	sensor_data = []

	for x in range(10):
		try:
			for y in range(3):
				[temp,humid] = dht(dht_sensor_port, blue_dht)
				if math.isnan(temp) == False and math.isnan(humid) == False:
					if temp >= 20.0:
						# Temperature is good: Everything is green.
						setRGB(0, 255, 0)
						digitalWrite(6, 1)
						time.sleep(2)
						digitalWrite(6, 0)
					else:
						# Temperature is too low: Turn to red.
						setRGB(255, 0, 0)
						digitalWrite(5, 1)
						time.sleep(2)
						digitalWrite(5, 0)
					t = str(temp)
					h = str(humid)
					print("Temperature:{}C | Humidity:{}%".format(t, h))
					setText_norefresh("Temp:{} C\nHumidity: {}%".format(t, h))
					sensor_data.append([temp, humid]);
				# For DHT22, wait three seconds before next reading
				time.sleep(3)
		except (IOError, TypeError) as ex:
			print("Error: {}".format(str(ex)))
			shutdown_board()
		except KeyboardInterrupt:
			shutdown_board()
		print(sensor_data)
		# Wait 30 seconds before collecting next set of data
		setRGB(0, 0, 0)
		time.sleep(30)
	shutdown_board()

def shutdown_board():
	"""Turns off LEDs and clears LCD screen"""
	digitalWrite(5, 0)
	digitalWrite(6, 0)
	setRGB(0, 0, 0)
	setText_norefresh("")
	print("Job complete. Have an excellent day.")

def main():
    """Application entry point."""
    print("Sensors Only Test.\n")
    sensors_only_test()

if __name__ == "__main__":
    main()