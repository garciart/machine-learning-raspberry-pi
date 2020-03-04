#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
TCP Socket Client Test
https://stackoverflow.com/questions/34548037/two-raspberries-communicating-each-other
"""

import socket
import time

# GrovePi information
HOST = "192.168.1.12"
PORT = 333

for i in range(20):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(b"Hello, world")
        data = s.recv(1024)
    print("Received {}".format(repr(data.decode("ascii"))))
    time.sleep(10)