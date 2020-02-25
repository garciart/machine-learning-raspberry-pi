#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Thermal comfort predictor using Keras (TensorFlow backend).
Python version used: 3.6.8
See requirements.txt for additional dependencies
Styling guide: PEP 8 -- Style Guide for Python Code
    (https://www.python.org/dev/peps/pep-0008/) and
    PEP 257 -- Docstring Conventions
    (https://www.python.org/dev/peps/pep-0257/)
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import time
import numpy as np
import pandas as pd

from sklearn.utils import Bunch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import *
from keras.losses import *

import tensorflow as tf

# Module metadata dunders
__author__ = "Rob Garcia"
__copyright__ = "Copyright 2019-2020, Rob Garcia"
__email__ = "rgarcia@rgprogramming.com"
__license__ = "MIT"


def load_local_dataset(csv_file_name):
    with open(csv_file_name) as csv_file:
        # Broken down for tutorial. Can be optimized into fewer lines.
        dataframe = pd.read_csv(csv_file_name, header=0)
        column_names = dataframe.columns.values
        feature_names = column_names[:-1]
        # label_name = column_names[-1]
        num_inputs = len(feature_names)
        array = dataframe.values
        feature_values = array[:, 0:num_inputs]
        label_values = array[:, num_inputs]
    return Bunch(data=feature_values, target=label_values), num_inputs


def tensorflow_classification_test(file_name, class_names, predict_x, expected):
    """Tensorflow processor"""
    print("TensorFlow version: {}".format(tf.__version__))
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)

    dataset, num_inputs = load_local_dataset(file_name)

    x = dataset.data
    y_ = dataset.target.reshape(-1, 1) # Convert data to a single column

    # One Hot encode the class labels
    encoder = OneHotEncoder(sparse=False)
    y = encoder.fit_transform(y_)

    # Split the data for training and testing
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.20)

    # Build the model
    model = Sequential()

    model.add(Dense(10, input_shape=(num_inputs,), activation='relu', name='fc1'))
    model.add(Dense(10, activation='relu', name='fc2'))
    model.add(Dense(len(class_names), activation='softmax', name='output'))

    """
    If your loss and accuracy do not change, try changing the optimizer or losses;

    Keras optimizers (https://keras.io/optimizers/)
    SGD - Stochastic gradient descent optimizer.
    RMSprop - RMSProp optimizer.
    Adagrad - Adagrad optimizer.
    Adadelta - Adadelta optimizer.
    Adam - Adam optimizer.
    Adamax - Adamax optimizer from Adam paper's Section 7.
    Nadam - Nesterov Adam optimizer.

    Keras losses (https://keras.io/losses/)
    mean_squared_error
    mean_absolute_error
    mean_absolute_percentage_error
    mean_squared_logarithmic_error
    squared_hinge
    hinge
    categorical_hinge
    logcosh
    huber_loss
    categorical_crossentropy
    sparse_categorical_crossentropy
    binary_crossentropy
    kullback_leibler_divergence
    poisson
    cosine_proximity
    is_categorical_crossentropy
    """

    optimizer = 'RMSprop'
    loss = 'mean_squared_error'
    model.compile(optimizer, loss, metrics=['accuracy'])

    print('Neural Network Model Summary: ')
    print(model.summary())

    # Train the model
    model.fit(train_x, train_y, verbose=2, batch_size=32, epochs=200)

    # Test on unseen data

    results = model.evaluate(test_x, test_y)

    print('Final test set loss: {:4f}'.format(results[0]))
    print('Final test set accuracy: {:4f}'.format(results[1]))

    predictions = model.predict_classes(predict_x, verbose=1)

    for i in range(len(predict_x)):
        print("X={}, Predicted: {} ({}), Expected {}".format(predict_x[i], class_names[predictions[i]], predictions[i], expected[i]))


def main():
    """Application entry point."""
    start_time = time.time()

    # Sample data to be evaluated

    # file_name = "iris.csv"
    file_name = "thermal_comfort.csv"

    # class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']
    class_names = ["Cold", "Cool", "Slightly Cool", "Neutral", "Slightly Warm", "Warm", "Hot"]

    """
    predict_x = np.array([
        [5.9, 3.0, 4.2, 1.5,],
        [5.1, 3.3, 1.7, 0.5,],
        [6.9, 3.1, 5.4, 2.1]
    ])
    """
    predict_x = np.array([
        [1013.25, 0.1, 50.0, 1.0, 0.5, 23.0],
        [1013.25, 0.1, 60.0, 1.0, 0.6, 26.0],
        [1013.25, 0.1, 76.0, 1.0, 0.6, 28.0]
    ])

    # expected = ['Iris versicolor', 'Iris setosa', 'Iris virginica']
    expected = ["Slightly Cool", "Neutral", "Slightly Warm"]

    print()
    print("TensorFlow Classification Test.\n")
    tensorflow_classification_test(file_name, class_names, predict_x, expected)
    print("Elapsed time: {} seconds.".format((time.time() - start_time)))
    print("Job complete. Have an excellent day.")


if __name__ == "__main__":
    main()