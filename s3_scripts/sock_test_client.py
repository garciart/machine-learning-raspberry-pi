#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""TCP Socket Client Test
https://stackoverflow.com/questions/34548037/two-raspberries-communicating-each-other
usage: sudo ./sock_test_client.py
Python version used: 3.6.8
See requirements.txt for additional dependencies
Styling guide: PEP 8 -- Style Guide for Python Code
    (https://www.python.org/dev/peps/pep-0008/) and
    PEP 257 -- Docstring Conventions
    (https://www.python.org/dev/peps/pep-0257/)
"""

import socket
import time

# Module metadata dunders
__author__ = "Rob Garcia"
__copyright__ = "Copyright 2019-2020, Rob Garcia"
__email__ = "rgarcia@rgcoding.com"
__license__ = "MIT"

# Waveshare Sense HAT (B) information
HOST = "192.168.1.12"
PORT = 333


def main():
    """Application entry point."""
    for i in range(3):
        print("Sending data...")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            if i < 2:
                data = b"Hello, friend."
                delay = 10
            else:
                data = b"Good-bye!"
                delay = 0
            s.sendall(data)
            data = s.recv(1024)
            print("Received {} from server!".format(repr(data.decode("utf-8"))))
            time.sleep(delay)
    print("Signing off: Good-bye.")


if __name__ == "__main__":
    main()
