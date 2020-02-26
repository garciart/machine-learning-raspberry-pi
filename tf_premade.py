#!python
# -*- coding: utf-8 -*-

"""Thermal comfort predictor using TensorFlow machine learning.
#!/usr/bin/python3
Python version used: 3.6.8
TensorFlow version used: 1.15.2
See requirements.txt for additional dependencies
Styling guide: PEP 8 -- Style Guide for Python Code
    (https://www.python.org/dev/peps/pep-0008/) and
    PEP 257 -- Docstring Conventions
    (https://www.python.org/dev/peps/pep-0257/)
Ref: https://www.tensorflow.org/tutorials/estimator/premade
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import math
import time

import numpy as np
import pandas as pd
import tensorflow as tf

# Module metadata dunders
__author__ = "Rob Garcia"
__copyright__ = "Copyright 2019-2020, Rob Garcia"
__email__ = "rgarcia@rgprogramming.com"
__license__ = "MIT"


def input_fn(feature_dict, label_values=None, training=True, batch_size=256):
    """An overloaded input function for training, evaluating, and prediction

    :param feature_dict: A Python dictionary in which each key is the name of
                         a feature and each value is an array containing all
                         of that feature's values.
    :type feature_dict: dict
    :param label_values: An array containing the values of the label for every example.
    :type label_values: array
    :param training: Shuffle data set if training.
    :type training: bool
    :param batch_size: The number of training examples utilized in one iteration.
    :type batch_size: int
    :return: A batched dataset
    :rtype: list
    """
    if label_values is None:
        # Convert the inputs to a Dataset without labels.
        return tf.data.Dataset.from_tensor_slices(dict(feature_dict)).batch(batch_size)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(
        (dict(feature_dict), label_values))
    # Shuffle and repeat if you are in training mode.
    if training:
        dataset = dataset.shuffle(1000).repeat()
    return dataset.batch(batch_size)


def tensorflow_classification_test(file_name, label_names, unlabeled_x):
    """Implementation of TensorFlow premade estimators:
    https://www.tensorflow.org/tutorials/estimator/premade

    :param file_name: The name of the csv file with the data.
    :type file_name: str
    :param label_names: The list of labels.
    :type label_names: list
    :param unlabeled_x: Unlabeled data to be classified.
    :type unlabeled_x: tuple
    :param expected_y: The expected results of classifying the unlabeled data.
    :type expected_y: list
    :return: Numpy array of predicted labels.
    :rtype: list
    """
    print("TensorFlow version: {}".format(tf.__version__))
    # Limit decimal places to three and do not use scientific notation
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)

    # Import and parse the training dataset
    # Broken down for tutorial. Can be optimized into fewer lines.
    column_titles = []
    feature_titles = []
    label_title = ""
    num_of_inputs = 0
    num_of_outputs = len(label_names)
    # feature_values = []
    # label_values = []

    with open(file_name) as csv_file:
        dataframe = pd.read_csv(csv_file, header=0)
        column_titles = dataframe.columns.values
        feature_titles = column_titles[:-1]
        label_title = column_titles[-1]
        num_of_inputs = len(feature_titles)
        # array = dataframe.values
        # feature_values = array[:, 0:num_of_inputs]
        # label_values = array[:, num_of_inputs]

    """
    # Utility functions to verify dataset
    print("Dataset shape:", dataframe.shape)
    print("Data check (first 5 rows:):")
    print(dataframe.head())
    print("Dataset feature descriptions:")
    print(dataframe.describe())
    print("Dataset class distribution:")
    print(dataframe.groupby(label_title).size())
    """

    # Split the dataframe into a training set and test set (80:20 ratio)
    msk = np.random.rand(len(dataframe)) < 0.8
    train = dataframe[msk]
    test = dataframe[~msk]

    # Remove the sensation description from dataframe (i.e., the CLASSES)
    train_y = train.pop(label_title)
    test_y = test.pop(label_title)

    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in train.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # Build a DNN with 2 hidden layers with hidden nodes sized according to
    # Guang-Bin Huang. (2003). Learning capability and storage capacity of
    # two-hidden-layer feedforward networks. IEEE Transactions on Neural Networks,
    # 14(2), 274–281. doi:10.1109/tnn.2003.809401
    # Thanks to Arvis Sulovari of the University of Washington Seattle
    hidden_layer_1 = int(round(math.sqrt(((num_of_outputs + 2) * num_of_inputs)) +
                               (2 * (math.sqrt(num_of_inputs / (num_of_outputs + 2))))))
    hidden_layer_2 = int(
        round(num_of_outputs * (math.sqrt(num_of_inputs / (num_of_outputs + 2)))))

    """
    # A classifier for TensorFlow DNN models.
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        hidden_units=[hidden_layer_1, hidden_layer_2],
        n_classes=num_of_outputs)

    # An estimator for TensorFlow Linear and DNN joined classification models.
    classifier = tf.estimator.DNNLinearCombinedClassifier(
        dnn_feature_columns=my_feature_columns,
        dnn_hidden_units=[hidden_layer_1, hidden_layer_2],
        n_classes=num_of_outputs)
    """
    # Linear classifier model.
    classifier = tf.estimator.LinearClassifier(
        feature_columns=my_feature_columns,
        n_classes=num_of_outputs)

    # Train the Model.
    classifier.train(
        input_fn=lambda: input_fn(train, train_y, training=True), steps=5000)

    eval_result = classifier.evaluate(
        input_fn=lambda: input_fn(test, test_y, training=False))

    print("\nTest set accuracy: {accuracy:0.3f}\n".format(**eval_result))

    return classifier


def main():
    """Application entry point."""
    start_time = time.time()
    print("TensorFlow Classification Test using Premade Estimators.\n")
    # Sample data to be evaluated

    # Iris test
    file_name = "iris.csv"
    label_names = ["Iris setosa", "Iris versicolor", "Iris virginica"]
    expected_y = ["Iris setosa", "Iris versicolor", "Iris virginica"]
    unlabeled_x = {
        "sepal_length": [5.1, 5.9, 6.9],
        "sepal_width": [3.3, 3.0, 3.1],
        "petal_length": [1.7, 4.2, 5.4],
        "petal_width": [0.5, 1.5, 2.1],
    }
    """
    file_name = "thermal_comfort.csv"
    label_names = ["Cold", "Cool", "Slightly Cool",
                   "Neutral", "Slightly Warm", "Warm", "Hot"]

    unlabeled_x = {
        "atmo_pres": [1013.25, 1013.25, 1013.25],
        "air_speed": [0.1, 0.1, 0.1],
        "rel_humid": [50.0, 60.0, 76.0],
        "meta_rate": [1.0, 1.0, 1.0],
        "cloth_lvl": [0.61, 0.61, 0.61],
        "oper_temp": [23.0, 26.0, 28.0],
    }
    expected_y = ["Slightly Cool", "Neutral", "Slightly Warm"]
    """
    classifier = tensorflow_classification_test(
        file_name, label_names, unlabeled_x)
    # Generate predictions from the model
    predictions = classifier.predict(
        input_fn=lambda: input_fn(unlabeled_x))
    print()
    for pred_dict, expec in zip(predictions, expected_y):
        class_id = pred_dict["class_ids"][0]
        probability = pred_dict["probabilities"][class_id]
        print("Prediction is \"{}\" ({:.1f}%), expected \"{}\"".format(
            label_names[class_id], 100 * probability, expec))
    print("Elapsed time: {} seconds.".format((time.time() - start_time)))
    print("Job complete. Have an excellent day.")


if __name__ == "__main__":
    main()
