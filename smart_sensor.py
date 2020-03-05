#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Smart Sensor: ASHRAE Environmental Monitoring System.
usage: sudo ./smart_sensor.py
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

import numpy as np
import pandas as pd
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

import grovepi
import grove_rgb_lcd

# Module metadata dunders
__author__ = "Rob Garcia"
__copyright__ = "Copyright 2019-2020, Rob Garcia"
__email__ = "rgarcia@rgprogramming.com"
__license__ = "MIT"

BLUE_DHT = 0         # For DHT11
# WHITE_DHT = 1       # For DHT22
GREEN_LED = 5        # Digital port 5
RED_LED = 6          # Digital port 6
DHT_SENSOR_PORT = 7  # Digital port 7
ON = 1
OFF = 0

def sensors_test():
    """Test the GrovePi DHT sensor, LEDs and RGB LCD screen."""
    try:
        for y in range(3):
            # Get sensor data and actuate LEDs and RGB LCD screen
            [temp, humid] = grovepi.dht(DHT_SENSOR_PORT, BLUE_DHT)
            if not math.isnan(temp) and not math.isnan(humid):
                if temp >= 20.0:
                    # Temperature is good: Everything is green.
                    grove_rgb_lcd.setRGB(0, 255, 0)
                    grovepi.digitalWrite(GREEN_LED, ON)
                    time.sleep(2)
                    grovepi.digitalWrite(GREEN_LED, OFF)
                else:
                    # Temperature is too low: Turn to red.
                    grove_rgb_lcd.setRGB(255, 0, 0)
                    grovepi.digitalWrite(RED_LED, ON)
                    time.sleep(2)
                    grovepi.digitalWrite(RED_LED, OFF)
                t_str = str(temp)
                h_str = str(humid)
                print("Temperature: {}C | Humidity: {}%".format(t_str, h_str))
                grove_rgb_lcd.setText_norefresh("Tmp: {} C\nHmd: {} %".format(t_str, h_str))
            # For DHT22, wait three seconds before next reading
            time.sleep(3)
            grove_rgb_lcd.setRGB(0, 0, 0)
    except (IOError, TypeError) as ex:
        print("Error: {}".format(str(ex)))
        shutdown_board()
    except KeyboardInterrupt:
        shutdown_board()
    shutdown_board()


def prepare_model(file_name, label_names):
    """Train, test, and run different classification estimators and select
    the best classifier.
    (see
    https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html
    for a list of classifiers)

    :param file_name: The name of the csv file with the data.
    :type file_name: str
    :param label_names: The list of labels.
    :type label_names: list
    :return: The most accurate classifer model in the form of a history object.
    :rtype: object
    """
    # Limit decimal places to three and do not use scientific notation
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)

    # Import and parse the training dataset
    # Broken down for tutorial. Can be optimized into fewer lines.
    column_titles = []
    feature_titles = []
    # label_title = ""
    num_of_inputs = 0
    # num_of_outputs = len(label_names)
    feature_values = []
    label_values = []

    with open(file_name) as csv_file:
        dataframe = pd.read_csv(csv_file, header=0)
        column_titles = dataframe.columns.values
        feature_titles = column_titles[:-1]
        label_title = column_titles[-1]
        num_of_inputs = len(feature_titles)
        values = dataframe.values
        feature_values = values[:, 0:num_of_inputs]
        label_values = values[:, num_of_inputs]

    """Utility functions to verify dataframe
    print("Dataframe shape:", dataframe.shape)
    print("Data check (first 20 rows:):")
    print(dataframe.head(20))
    print("Dataframe feature descriptions:")
    print(dataframe.describe().round(3))
    print("Dataframe class distribution:")
    print(dataframe.groupby("sens_desc").size())
    """

    # Split the dataframe, then use 80% for training and 20% for testing
    # (as determined by test_size). Remove random_state for production.
    x_train, x_validation, y_train, y_validation = train_test_split(
        feature_values, label_values, test_size=0.20, random_state=1)

    # Create tuple of estimators
    estimators = []
    estimators.append(("Logistic Regression", LogisticRegression(
        solver="liblinear", multi_class="ovr")))
    estimators.append(
        ("Linear Support Vector Classification (LinearSVC)", LinearSVC(dual=False)))
    estimators.append(("Stochastic Gradient Descent (SGD)", SGDClassifier()))
    estimators.append(
        ("k-Nearest Neighbors Classifier (k-NN)", KNeighborsClassifier()))
    estimators.append(("Support Vector Classification (SVC)",
                       SVC(kernel="linear", C=1.0)))
    estimators.append(("Gaussian Naive Bayes (GaussianNB)", GaussianNB()))
    estimators.append(("Random Forest Classifier", RandomForestClassifier()))
    estimators.append(("Extra Trees Classifier", ExtraTreesClassifier()))
    estimators.append(("Decision Tree Classifier", DecisionTreeClassifier()))
    estimators.append(("AdaBoost Classifier", AdaBoostClassifier()))
    estimators.append(("Gradient Boosting Classifier",
                       GradientBoostingClassifier()))
    estimators.append(("Linear Discriminant Analysis (LDA)",
                       LinearDiscriminantAnalysis()))

    # Evaluate the accuracy of each estimator (limited to classifiers)
    results = []
    for name, classifier in estimators:
        training_model = classifier.fit(x_train, y_train)
        prediction = training_model.predict(x_validation)
        score = accuracy_score(y_validation, prediction)
        results.append((name, training_model, score))

    # Get and sort the results
    results.sort(key=lambda r: r[2], reverse=True)

    # Select the highest scoring training model and refit to the entire dataset
    top_model_name = results[0][0]
    top_model_refit = results[0][1].fit(feature_values, label_values)
    top_model_score = results[0][2]

    # Package and return
    selected_model = [top_model_name, top_model_refit, top_model_score]
    return selected_model


def check_model(label_names, selected_model):
    """Check selected model against unlabeled data (with expected predictions).

    :param label_names: The list of labels.
    :type label_names: list
    :param selected_model: The most accurate classifer model in the form of a history object.
    :type selected_model: object
    """
    print("Data to be evaluated:")
    unlabeled_x = []
    unlabeled_x.append(([[1013.25, 0.1, 50.0, 1.0, 0.61, 23.0]], "Slightly Cool"))
    unlabeled_x.append(([[1013.25, 0.1, 60.0, 1.0, 0.61, 26.0]], "Neutral"))
    unlabeled_x.append(([[1013.25, 0.1, 76.0, 1.0, 0.61, 28.0]], "Slightly Warm"))
    # expected_y = ["Slightly Cool", "Neutral", "Slightly Warm"]

    for i, (data, expected_label) in enumerate(unlabeled_x, start=1):
        print("Sample #{}: {} = {}".format(i, data, expected_label))

    # Run unlabeled data against selected model and check accuracy
    print("Prediction(s):")
    for j, (data, expected_label) in enumerate(unlabeled_x, start=1):
        print("Sample #{}: Prediction: {} (expected {})".format(
            j, label_names[int(selected_model[1].predict(data))], expected_label))


def collect_sensor_data():
    """Collects temperature and humidity data
    :return: The temperature and humidity data collected, formated with
             atmospheric pressure, air speed, metabolic rate, and clothing level.
    :rtype: list of tuples
    """
    sensor_data = []
    try:
        for y in range(3):
            [temp, humid] = grovepi.dht(DHT_SENSOR_PORT, BLUE_DHT)
            if not math.isnan(temp) and not math.isnan(humid):
                t_str = str(temp)
                h_str = str(humid)
                grove_rgb_lcd.setRGB(127, 127, 127)
                grove_rgb_lcd.setText_norefresh("Tmp: {} C\nHmd: {} %".format(t_str, h_str))
                sensor_data.append(([[1013.25, 0.1, humid, 1.0, 0.61, temp]]))
            # For DHT11, wait three seconds before next reading
            time.sleep(3)
            grove_rgb_lcd.setRGB(0, 0, 0)
    except (IOError, TypeError) as ex:
        print("Error: {}".format(str(ex)))
        shutdown_board()
    except KeyboardInterrupt:
        shutdown_board()
    for index, data in enumerate(sensor_data):
        print("Sample #{0} collected: {1}".format(index + 1, data))
    shutdown_board()
    return sensor_data


def process_sensor_data(label_names, selected_model, sensor_data):
    """Classify unlabeled sensor data against the selected model and get a prediction.

    :param label_names: The list of labels.
    :type label_names: list
    :param selected_model: The most accurate classifer model in the form of a history object.
    :type selected_model: object
    :param sensor_data: The temperature and humidity data collected, formated with
                        atmospheric pressure, air speed, metabolic rate, and clothing level.
    :type sensor_data: list of tuples
    """
    try:
        sensation = 0
        for j, data in enumerate(sensor_data, start=1):
            print("Sensor data #{}: Prediction: {}".format(
            j, label_names[int(selected_model[1].predict(data))]))
            sensation += int(selected_model[1].predict(data))
        sensation = int(sensation / len(sensor_data))
        print("Overall sensation: {} ({})".format(sensation, label_names[sensation]))
        grove_rgb_lcd.setText_norefresh("Sensation:\n{} ({})".format(sensation, label_names[sensation]))
        if sensation == 3:
            # Temperature is good: Everything is green.
            grove_rgb_lcd.setRGB(0, 255, 0)
            grovepi.digitalWrite(GREEN_LED, ON)
            time.sleep(2)
            grovepi.digitalWrite(GREEN_LED, OFF)
            grove_rgb_lcd.setRGB(0, 0, 0)
            grove_rgb_lcd.setText_norefresh("")
        else:
            # Temperature is too low: Turn to red.
            grove_rgb_lcd.setRGB(255, 0, 0)
            grovepi.digitalWrite(RED_LED, ON)
            time.sleep(2)
            grovepi.digitalWrite(RED_LED, OFF)
    except (IOError, TypeError) as ex:
        print("Error: {}".format(str(ex)))
        shutdown_board()
    except KeyboardInterrupt:
        shutdown_board()
    time.sleep(3)
    shutdown_board()


def shutdown_board():
    """Turns off LEDs and clears LCD screen"""
    grovepi.digitalWrite(RED_LED, OFF)
    grovepi.digitalWrite(GREEN_LED, OFF)
    grove_rgb_lcd.setRGB(0, 0, 0)
    grove_rgb_lcd.setText_norefresh("")


def main():
    """Application entry point."""
    print("Smart Sensor Application.\n")

    # Test sensors first (optional)
    print("Testing sensors...")
    sensors_test()
    print("Test complete.\n")

    file_name = "thermal_comfort.csv"
    label_names = ["Cold", "Cool", "Slightly Cool",
                   "Neutral", "Slightly Warm", "Warm", "Hot"]
    start_time = time.time()

    print("Training and testing model...")
    selected_model = prepare_model(file_name, label_names)
    print("Model selected: {0} ({1:.2f}%).".format(selected_model[0], selected_model[2]))
    print("Training and testing model complete.")
    print("Elapsed time: {} seconds.\n".format((time.time() - start_time)))

    print("Checking model against unlabeled data...")
    check_model(label_names, selected_model)
    print()

    print("Collecting sensor data...")
    sensor_data = collect_sensor_data()
    print("Collection complete.\n")

    print("Processing sensor data...")
    process_sensor_data(label_names, selected_model, sensor_data)
    print()

    print("Shutting down board...")
    shutdown_board()
    print("Job complete. Have an excellent day.")


if __name__ == "__main__":
    main()
