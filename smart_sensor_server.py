#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
TCP Socket Server Test
https://stackoverflow.com/questions/34548037/two-raspberries-communicating-each-other
usage: sudo ./sock_test_server.py
Python version used: 3.6.8
See requirements.txt for additional dependencies
Styling guide: PEP 8 -- Style Guide for Python Code
    (https://www.python.org/dev/peps/pep-0008/) and
    PEP 257 -- Docstring Conventions
    (https://www.python.org/dev/peps/pep-0257/)
"""

import socket

# Module metadata dunders
__author__ = "Rob Garcia"
__copyright__ = "Copyright 2019-2020, Rob Garcia"
__email__ = "rgarcia@rgprogramming.com"
__license__ = "MIT"

# GrovePi information
HOST = "192.168.1.12"
PORT = 333

def main():
    """Application entry point."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        while True:
            print("Waiting for data...")
            s.listen(1)
            connection, client_address = s.accept()
            data = connection.recv(1024)
            if data:
                data = repr(data.decode("ascii"))
                print("Received {} from client!".format(data))
                if data == "'Good-bye!'":
                    data = b"Good-bye!"
                    connection.sendall(data)
                    print("The client has signed off.")
                    break
                else:
                    data = b"Hello back!"
                    connection.sendall(data)


if __name__ == "__main__":
    main()
