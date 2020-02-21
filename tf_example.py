#!python
# -*- coding: utf-8 -*-

"""Thermal comfort predictor using TensorFlow machine learning.
Python version used: 3.6.8
TensorFlow version used: 1.15.2
See requirements.txt for additional dependencies
Styling guide: PEP 8 -- Style Guide for Python Code
    (https://www.python.org/dev/peps/pep-0008/) and
    PEP 257 -- Docstring Conventions
    (https://www.python.org/dev/peps/pep-0257/)
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import math

import numpy as np
import pandas as pd
import tensorflow as tf

# Module metadata dunders
__author__ = "Rob Garcia"
__copyright__ = "Copyright 2019-2020, Rob Garcia"
__email__ = "rgarcia@rgprogramming.com"
__license__ = "MIT"


def input_fn(features, labels=None, training=True, batch_size=1024):
    """An overloaded input function for training, evaluating, and prediction"""
    if labels is None:
        # Convert the inputs to a Dataset without labels.
        return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    # Shuffle and repeat if you are in training mode.
    if training:
        dataset = dataset.shuffle(1000).repeat()
    return dataset.batch(batch_size)


def tensorflow_classification_test():
    """
    Implementation of TensorFlow premade estimators:
    https://www.tensorflow.org/tutorials/estimator/premade
    """
    # Read CSV data into dataset
    csv_column_names = ["air_speed", "rel_humid",
                        "meta_rate", "cloth_lvl", "oper_temp", "sens_desc"]
    dataframe = pd.read_csv("thermal_comfort.csv",
                            names=csv_column_names, header=0)

    # Name classes and features
    class_names = ["Cold", "Cool", "Slightly Cool",
                   "Neutral", "Slightly Warm", "Warm", "Hot"]
    # "air_speed", "rel_humid", "meta_rate", "cloth_lvl", "oper_temp"
    # feature_names = csv_column_names[:-1]
    # "sens_desc"
    label_name = csv_column_names[-1]

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

    # Remove the sensation description from dataframe (i.e., the CLASSES)
    train_y = train.pop("sens_desc")
    test_y = test.pop("sens_desc")

    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in train.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # Build a DNN with 2 hidden layers with hidden nodes sized according to
    # Guang-Bin Huang. (2003). Learning capability and storage capacity of
    # two-hidden-layer feedforward networks. IEEE Transactions on Neural Networks,
    # 14(2), 274–281. doi:10.1109/tnn.2003.809401
    hidden_layer_1 = int(math.sqrt(((len(label_name) + 2) * len(dataframe))) +
                         (2 * math.sqrt(len(dataframe) / (len(label_name) + 2))))
    hidden_layer_2 = int(
        len(label_name) * (math.sqrt((len(dataframe) / (len(label_name) + 2)))))

    # A classifier for TensorFlow DNN models.
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        hidden_units=[hidden_layer_1, hidden_layer_2],
        n_classes=7)

    """
    # An estimator for TensorFlow Linear and DNN joined classification models.
    classifier = tf.estimator.DNNLinearCombinedClassifier(
        dnn_feature_columns=my_feature_columns,
        dnn_hidden_units=[hidden_layer_1, hidden_layer_2],
        n_classes=7)
    # Linear classifier model.
    classifier = tf.estimator.LinearClassifier(
        feature_columns=my_feature_columns,
        n_classes=7)
    """

    # Train the Model.
    classifier.train(
        input_fn=lambda: input_fn(train, train_y, training=True), steps=5000)

    eval_result = classifier.evaluate(
        input_fn=lambda: input_fn(test, test_y, training=False))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # Generate predictions from the model
    expected = ["Slightly Cool", "Neutral", "Slightly Warm"]
    predict_x = {
        "air_speed": [0.1, 0.1, 0.1],
        "rel_humid": [50.0, 60.0, 76.0],
        "meta_rate": [1.0, 1.0, 1.0],
        "cloth_lvl": [0.5, 0.6, 0.6],
        "oper_temp": [23.0, 26.0, 28.0],
    }

    predictions = classifier.predict(
        input_fn=lambda: input_fn(predict_x))

    for pred_dict, expec in zip(predictions, expected):
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print('Prediction is "{}" ({:.1f}%), expected "{}"'.format(
            class_names[class_id], 100 * probability, expec))


def main():
    """Application entry point."""
    print("scikit-learn Classification Test.\n")
    tensorflow_classification_test()
    print("Job complete. Have an excellent day.")


if __name__ == "__main__":
    main()