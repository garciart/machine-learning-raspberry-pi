#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
TCP Socket Server Test
https://stackoverflow.com/questions/34548037/two-raspberries-communicating-each-other
"""

import socket

# GrovePi information
HOST = "192.168.1.12"
PORT = 333

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    print("Waiting for data...")
    s.bind((HOST, PORT))
    s.listen(1)
    connection, client_address = s.accept()
    while True:
        data = connection.recv(1024)
        if data:
            print("Data received!")
            data = b"Hello back!"
            connection.sendall(data)
        else:
            break
