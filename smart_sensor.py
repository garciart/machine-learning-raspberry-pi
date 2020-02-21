#!python
# -*- coding: utf-8 -*-

"""Thermal comfort predictor using Scikit Learn and TensorFlow machine learning.
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
"""
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
"""
import grovepi

# Module metadata dunders
__author__ = "Rob Garcia"
__copyright__ = "Copyright 2019-2020, Rob Garcia"
__email__ = "rgarcia@rgprogramming.com"
__license__ = "MIT"

dht_sensor_port = 5 # Digital port 5
blue_dht = 0        # For DHT11
white_dht = 1       # For DHT22

sensor_data = []

for x in range(100):
    try:
        for y in range(3):
            [temp,humid] = grovepi.dht(dht_sensor_port, blue_dht)
            if math.isnan(temp) == False and math.isnan(humid) == False:
                print("Temperature:{}C | Humidity:{}%".format(temp, humid))
            sensor_data.append([temp, humid]);
            # Wait three seconds before next reading
            time.sleep(3)
    except (IOError, TypeError) as ex:
        print("Error: {}".format(str(ex)))
    print(sensor_data)
    # Wait 30 seconds before collecting next set of data
    time.sleep(30)