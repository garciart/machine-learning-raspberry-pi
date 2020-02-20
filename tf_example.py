#!python
# -*- coding: utf-8 -*-

"""Thermal comfort predictor using TensorFlow machine learning.
Python version used: 3.6.8
See requirements.txt for additional dependencies
Styling guide: PEP 8 -- Style Guide for Python Code
    (https://www.python.org/dev/peps/pep-0008/) and
    PEP 257 -- Docstring Conventions
    (https://www.python.org/dev/peps/pep-0257/)
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os

import numpy as np
import pandas as pd
import tensorflow as tf

# Module metadata dunders
__author__ = "Rob Garcia"
__copyright__ = "Copyright 2019-2020, Rob Garcia"
__email__ = "rgarcia@rgprogramming.com"
__license__ = "MIT"

# Read CSV data into dataset
CSV_COLUMN_NAMES = ["air_speed", "rel_humid",
                    "meta_rate", "cloth_lvl", "oper_temp", "sens_desc"]
dataframe = pd.read_csv("thermal_comfort.csv", names=CSV_COLUMN_NAMES, header=0)

# Name classes and features
CLASSES = ["Cold", "Cool", "Slightly Cool",
           "Neutral", "Slightly Warm", "Warm", "Hot"]
FEATURES = ["air_speed", "rel_humid", "meta_rate", "cloth_lvl", "oper_temp"]

"""
# Utility functions to verify dataset
print("Dataset shape:", dataframe.shape)
print("Data check (first 20 rows:):")
print(dataframe.head(20))
print("Dataset feature descriptions:")
print(dataframe.describe())
print("Dataset class distribution:")
print(dataframe.groupby("sens_desc").size())
"""

# Split the dataframe into a training set and test set (80:20 ratio)
msk = np.random.rand(len(dataframe)) < 0.8
train = dataframe[msk]
test = dataframe[~msk]

# Removethe sensation description from dataframe (i.e., the CLASSES)
train_y = train.pop("sens_desc")
test_y = test.pop("sens_desc")

print(train.head())



# https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough