#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Thermal comfort predictor using TensorFlow machine learning.
Python version used: 3.6.8
See requirements.txt for additional dependencies
Styling guide: PEP 8 -- Style Guide for Python Code
    (https://www.python.org/dev/peps/pep-0008/) and
    PEP 257 -- Docstring Conventions
    (https://www.python.org/dev/peps/pep-0257/)
For details, see https://github.com/tensorflow/docs/blob/master/site/en/r1/tutorials/eager/custom_training_walkthrough.ipynb
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


# Module metadata dunders
__author__ = "Rob Garcia"
__copyright__ = "Copyright 2019-2020, Rob Garcia"
__email__ = "rgarcia@rgprogramming.com"
__license__ = "MIT"


def pack_features_vector(features, labels):
    """Pack the features into a single array."""
    features = tf.stack(list(features.values()), axis=1)
    return features, labels


def loss(model, x, y):
    y_ = model(x)
    return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def tensorflow_classification_test(expected, predict_x):
    """
    Implementation of TensorFlow:
    https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough
    """
    # ---------
    tf.compat.v1.enable_eager_execution()
    print("TensorFlow version: {}".format(tf.__version__))
    ## print("Eager execution enabled: {}".format(tf.executing_eagerly()))

    # Read CSV data into dataset
    train_dataset_fp = "iris.csv"
    ## print("Local copy of the dataset file: {}".format(train_dataset_fp))

    # Read feature names and class names from column headers:
    # ["atmo_pres", "air_speed", "rel_humid", "meta_rate", "cloth_lvl", "oper_temp", "sens_desc"]
    # Do not hardcode feature names or label names; this way, you can add or subtract columns later.
    dataframe = pd.read_csv(train_dataset_fp)
    num_of_rows = dataframe.shape[0]
    print("Number of rows: {}".format(num_of_rows))
    column_names = dataframe.columns.values
    ## print("Column names: {}".format(column_names))
    feature_names = column_names[:-1]
    label_name = column_names[-1]
    ## print("Features: {}".format(feature_names))
    ## print("Label: {}".format(label_name))
    # Drop the column headers
    # train_dataset_fp.drop([0, 0])
    # Assign names to the labels/classes
    class_names = ["Cold", "Cool", "Slightly Cool",
                   "Neutral", "Slightly Warm", "Warm", "Hot"]
    ## print("Class names: {}".format(class_names))

    batch_size = 32

    dataset = tf.data.experimental.make_csv_dataset(
        train_dataset_fp,
        batch_size,
        column_names=column_names,
        label_name=label_name,
        num_epochs=5)

    the_split = int(0.8 * num_of_rows)
    print("The Split: {}".format(the_split))

    train_dataset = dataset.take(the_split)
    test_dataset = dataset.skip(the_split)
    # train_dataset = dataset
    # test_dataset = dataset.take(num_of_rows - the_split)

    





    # features, labels = next(iter(train_dataset))
    # print("\nFeatures:")
    # print(features)

    train_dataset = train_dataset.map(pack_features_vector)
    features, labels = next(iter(train_dataset))
    print("\nFeatures values (rows 1-5):")
    print(features[:5])

    c = len(class_names)
    f = len(feature_names)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(f,)),  # input shape required
        tf.keras.layers.Dense(10, activation=tf.nn.relu),
        tf.keras.layers.Dense(c)
    ])

    predictions = model(features)
    print("\nLogits (rows 1-5):")
    print(predictions[:5])
    print("\nLogits converted to probabilities using softmax (rows 1-5):")
    print(tf.nn.softmax(predictions[:5]))

    # Use argmax to get the index with the largest value across axes of a tensor.
    print("Index of the largest value across axes of the tensors (untrained).")
    print("\nPrediction: {}".format(tf.argmax(predictions, axis=1)))
    print("    Labels: {}".format(labels))

    l = loss(model, features, labels)
    print("Loss test: {}".format(l))

    # Test out other optimization algorithms later
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    global_step = tf.Variable(0)

    loss_value, grads = grad(model, features, labels)
    print("Step: {}, Initial Loss: {}".format(global_step.numpy(), loss_value.numpy()))
    optimizer.apply_gradients(zip(grads, model.trainable_variables), global_step)
    print("Step: {},         Loss: {}".format(global_step.numpy(), loss(model, features, labels).numpy()))


    # keep results for plotting
    train_loss_results = []
    train_accuracy_results = []
    num_epochs = 201
    for epoch in range(num_epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.Accuracy()
        # Training loop - using batches of 32
        for x, y in train_dataset:
            # Optimize the model
            loss_value, grads = grad(model, x, y)
            optimizer.apply_gradients(zip(grads, model.trainable_variables),
                                      global_step)
            # Track progress
            epoch_loss_avg(loss_value)  # add current batch loss
            # compare predicted label to actual label
            epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)
        # end epoch
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())
        if epoch % 50 == 0:
            print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                        epoch_loss_avg.result(),
                                                                        epoch_accuracy.result()))

    test_dataset = test_dataset.map(pack_features_vector)
    test_accuracy = tf.keras.metrics.Accuracy()
    for (x, y) in test_dataset:
        logits = model(x)
        prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
        test_accuracy(prediction, y)
    print("Test set accuracy: {:.3%}".format(test_accuracy.result()))

    predict_dataset = tf.convert_to_tensor([
        [1013.25, 0.1, 50.0, 1.0, 0.5, 23.0],
        [1013.25, 0.1, 60.0, 1.0, 0.6, 26.0],
        [1013.25, 0.1, 76.0, 1.0, 0.6, 28.0]
    ])
    
    predictions = model(predict_dataset)

    for i, logits in enumerate(predictions):
        class_idx = tf.argmax(logits).numpy()
        p = tf.nn.softmax(logits)[class_idx]
        name = class_names[class_idx]
        print("Example {} prediction: {} ({:4.1f}%)".format(i, name, 100*p))
    

def main():
    """Application entry point."""
    start_time = time.time()
    # Sample data to be evaluated
    expected = ["Slightly Cool", "Neutral", "Slightly Warm"]
    predict_x = {
        "atmo_pres": [1013.25, 1013.25, 1013.25],
        "air_speed": [0.1, 0.1, 0.1],
        "rel_humid": [50.0, 60.0, 76.0],
        "meta_rate": [1.0, 1.0, 1.0],
        "cloth_lvl": [0.5, 0.6, 0.6],
        "oper_temp": [23.0, 26.0, 28.0],
    }
    print()
    print("TensorFlow Classification Test.\n")
    tensorflow_classification_test(expected, predict_x)
    print("Elapsed time: {} seconds.".format((time.time() - start_time)))
    print("Job complete. Have an excellent day.")


if __name__ == "__main__":
    main()