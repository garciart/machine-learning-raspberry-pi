# Smart Sensor

> ***DISCLAIMER** - This repository and its contents are not endorsed by, sponsored by, affiliated with, nor associated with the National Aeronautics and Space Administration (NASA). This repository is only a demonstration of some of the concepts that I explored while I was an intern at NASA's Langley Research Center.*

- [Introduction](#introduction)
- [Parts](#parts)
- [Key Terms](#key-terms)
- [How It Works](#how-it-works)
- [Steps](#introduction)
- [Additional Information](#introduction)
- [Extra Credit](#extra-credit)
- [References](#references)

## Introduction

While I was an intern at NASA, I noticed a lot of interest in "smart" technologies, e.g., "Edge" computing and Internet of Things (IoT) devices; Augmented and virtual reality; Machine learning; etc.

Since there's no MicroCenter or Home Depot in space (and space delivery is not included in Amazon Prime), one of the things I thought about was a portable, customizable, plug-and-play data analyzer, aka a smart sensor suite (S3). For less than $100, the S3, with a set of basic sensors, could replace the following devices:

- Gas Detector (O2, LEL, CO-H2S, NH3, NO2): $2571.91
- Light Meter $94.95
- Sound Level Meter: $76.68
- Temperature and Humidity Meter: $19.99
- Universal Remote Control: $9.99

In addition, users could add or create ad hoc sensors and actuators from simple components using wires, resistors, breadboards, etc. As a bonus, the S3 could communicate over Bluetooth, RF, and WiFi wireless networks.

This device would also use machine learning, as well as rule-based, conditional logic, to analyze data on the "edge" and act on the results of its analyses through actuators. This system would complement or replace rule-based, conditional logic systems, such as engine control units (i.e., ECU's, which use lookup tables), electronic flight control systems (i.e., EFCS's, which use flight control laws), heating, ventilation, and air conditioning (HVAC) control systems (which use target states), etc.

In this demo, we will create a device that collects environmental data to determine if the thermal comfort level of a room complies with the American Society of Heating, Refrigerating and Air-Conditioning Engineers (ASHRAE) Standard 55-2017. This standard helps engineers design effective HVAC systems.

As a bonus, we will demonstrate how to collect and process data over-the-air (OTA) from satellite devices using the S3.

## Parts

For this demo, we used:

- One (1) Raspberry Pi 3 Model B+ with 5V/2.5A power source as the S3
- One (1) GrovePi+ HAT (hardware attached on top) add-on board
- One (1) temperature and humidity sensor with a Grove connector (we used a simple DHT11)
- One (1) red and one (1) green LED's with Grove connectors
- One (1) 16 x 2 LCD RGB Backlight with Grove connector
- Four (4) 4-wire cables with four-pin female-to-female Grove connectors

This is not the only configuration the S3 can use; for our bonus demo, instead of connected sensors, the S3 uses a remote Raspberry Pi Zero W with a Waveshare Sense HAT (B).

## Key Terms

- Features (aka the x's, input variable names) - The names of each value in a vector of values (e.g., temperature, humidity, etc.). Together, they imply a relationship (e.g., hot or cold, etc.).
- Labels (aka the y's, classes, targets, output variable names) - The name of the relationship between the values (e.g., hot or cold, etc.).
- Weights (aka the thetas) - The actual values in the vector of features (e.g., 37 deg C, 100%, etc.).
- Dataset - The collection of all the feature weights and label values.
- Classifier - An algorithm that tries to predict the relationship between features from unlabeled data.
- Model - An algorithm that accounts for the relationship between features and the correct label.

## How it Works

1. The application loads and parses the dataset and tests all classifiers against the dataset.
2. The application creates a model using the highest scoring classifier.
3. The application then requests data from the sensors.
4. The application runs the data against the model using the selected classifier.
5. The application returns the results:
   - If the result is "Neutral", it complies with ASHRAE Standard 55-2017. The application displays the data on its LCD screen and indicates compliance by lighting a green LED. The application can also forward this data to a supervisory control and data acquisition (SCADA) system via JSON.
   - If the result is anything but "Neutral), it does NOT comply with ASHRAE Standard 55-2017. The application displays the data on its LCD screen and indicates non-compliance by lighting a red LED. The application can also forward this data to a supervisory control and data acquisition (SCADA) system via JSON and/or adjust the HVAC system autonomously.
6. The application repeats steps 3 through 7 until it is shut off.

For this demo, we will only measure operative temperature and relative humidity. However, the S3 can also process atmospheric pressure, air speed, metabolic rate, and the insulating effects of clothing level, to predict if the occupants will feel cold, cool, slightly cool, neutral, slightly warm, warm, or hot.

| atmo_pres | air_speed | rel_humid | meta_rate | cloth_lvl | oper_temp | sens_desc |
|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
|   1013.25 |       0.1 |      50.0 |       1.0 |      0.61 |      23.0 |         2 |
|   1013.25 |       0.1 |      60.0 |       1.0 |      0.61 |      26.0 |         3 |
|   1013.25 |       0.1 |      70.0 |       1.0 |      0.61 |      28.0 |         4 |

While this task would be easy to accomplish with simple if-else-then program, we also wanted to demonstrate the capabilities of machine learning.

## Steps

1. Install Raspbian for Robots on an SD card for the Raspberry Pi by following the instructions at [https://www.dexterindustries.com/howto/install-raspbian-for-robots-image-on-an-sd-card/](https://www.dexterindustries.com/howto/install-raspbian-for-robots-image-on-an-sd-card/) to install Dexter Industries' version of Raspbian on your Raspberry Pi. 

2. Set up the Raspberry Pi with the GrovePi HAT:

   - Follow the instructions at [https://www.dexterindustries.com/GrovePi/get-started-with-the-grovepi/](https://www.dexterindustries.com/GrovePi/get-started-with-the-grovepi/) to setup and update your system:

     - For Step 2, use the custom software you installed on your card earlier.
     - For Step 3, use "pi" for the username and "robots1234" for the password.
     - For Step 4, don't forget to execute the DI Software Update, as well update and upgrade your software.

   - If you prefer to SSH, follow these instructions instead: [https://www.dexterindustries.com/GrovePi/get-started-with-the-grovepi/setting-software/](https://www.dexterindustries.com/GrovePi/get-started-with-the-grovepi/setting-software/).

3. Ensure that Python 3 and PIP are installed:

   - Run the following commands to verify Python 3 and PIP are installed:

     ```linux
     pi@dex:~ $ python3 --version
     pi@dex:~ $ pip3 --version
     ```

   - If they are not installed, follow the instructions at [https://www.raspberrypi.org/documentation/linux/software/python.md]( https://www.raspberrypi.org/documentation/linux/software/python.md).

4. In the terminal, clone the Smart Sensor repository:

   ```linux
   pi@dex:~ $ sudo git clone https://github.com/garciart/SmartSensor
   pi@dex:~ $ cd SmartSensor
   pi@dex:~/SmartSensor $ sudo chmod +x permissions.sh
   pi@dex:~/SmartSensor $ sudo ./permissions.sh
   pi@dex:~/SmartSensor $ sudo python3 setup.py install
   ```

5. Install the required dependencies using the below command. While we tried to ensure the requirements file is correct, if you run into an issue, install the problem package separately, and then re-run the requirements.txt file:

   ```linux
   pi@dex:~/SmartSensor $ pip3 install --no-cache-dir -r requirements.txt
   ```

6. Once complete, verify scikit-learn, TensorFlow, and the sensors work (disregard any TensorFlow warnings for now):

   ```linux
   pi@dex:~ /SmartSensor $ ./sensors_only_test.py
   pi@dex:~ /SmartSensor $ ./tf_premade.py
   pi@dex:~ /SmartSensor $ ./sl_example.py
   pi@dex:~ /SmartSensor $ ./tf_example.py
   ```

7. Notice that the scikit-learn example ran faster (4.6 sec vs 14.4 and 13.5 sec) and was much more accurate than the TensorFlow scripts. We believe this is due to the dataset having six features, but with variations in only two of those features. If you run the scripts against the Iris dataset (remove the comments surrounding the Iris test and place them around the Thermal Comfort test), the accuracy of the TensorFlow scripts increases dramatically. However, for the S3, we will use the scikit classifiers.

```linux
scikit-learn Classification Test.
...
Running samples using Gradient Boosting Classifier
(test score was 1.00%)...
Sample #1: Prediction: Slightly Cool (expected Slightly Cool)
Sample #2: Prediction: Neutral (expected Neutral)
Sample #3: Prediction: Slightly Warm (expected Slightly Warm)

Running samples using Random Forest Classifier
(test score was 1.00%)...
Sample #1: Prediction: Slightly Cool (expected Slightly Cool)
Sample #2: Prediction: Neutral (expected Neutral)
Sample #3: Prediction: Slightly Warm (expected Slightly Warm)
...
Elapsed time: 4.682486534118652 seconds.
Job complete. Have an excellent day.
```

```linux
TensorFlow Classification Test using Premade Estimators.
...
Prediction is "Slightly Cool" (32.4%), expected "Slightly Cool"
Prediction is "Warm" (51.3%), expected "Neutral"
Prediction is "Warm" (85.8%), expected "Slightly Warm"
Elapsed time: 14.417850971221924 seconds.
Job complete. Have an excellent day.
```

```linux
TensorFlow Classification Test using Keras.
...
X=[1013.25    0.1    50.      1.      0.61   23.  ], Predicted: Cool (1), Expected Slightly Cool
X=[1013.25    0.1    60.      1.      0.61   26.  ], Predicted: Cool (1), Expected Neutral
X=[1013.25    0.1    76.      1.      0.61   28.  ], Predicted: Cool (1), Expected Slightly Warm
Elapsed time: 13.566766738891602 seconds.
Job complete. Have an excellent day.
```

## Additional Information

### Data Source

Data verified using the [Thermal Comfort Tool](https://comfort.cbe.berkeley.edu/), Center for the Built Environment, University of California Berkeley.

### Features

- Atmospheric Pressure(atmo_pres)
- Air Speed (air_speed)
- Relative Humidity (rel_humid)
- Metabolic Rate (meta_rate)
- Clothing Level (cloth_lvl)
- Operative Temperature (oper_temp)

### Labels

- Label: Sensation (sens_desc)
- Label descriptions and value
  - Cold (0)
  - Cool (1)
  - Slightly Cool (2)
  - Neutral (3)
  - Slightly Warm (4)
  - Warm (5)
  - Hot (6)

## Extra Credit

## References

ASHRAE. (2017). ANSI/ASHRAE standard 55-2017; Thermal environmental conditions for human occupancy (55). Atlanta, GA: Author.

Engineering ToolBox. (2004). Illuminance - recommended light level. Retrieved February 18, 2020, from [https://www.engineeringtoolbox.com/light-level-rooms-d_708.html](https://www.engineeringtoolbox.com/light-level-rooms-d_708.html)

Guenther, S. (2019, November 7). What Is PMV? What Is PPD? The basics of thermal comfort. Retrieved from [https://www.simscale.com/blog/2019/09/what-is-pmv-ppd/](https://www.simscale.com/blog/2019/09/what-is-pmv-ppd/)

Hoyt, T., Schiavon, S., Tartarini, F., Cheung, T., Steinfeld, K., Piccioli, A., & Moon, D. (2019). CBE Thermal Comfort Tool. Retrieved from [https://comfort.cbe.berkeley.edu/](https://comfort.cbe.berkeley.edu/)
Center for the Built Environment, University of California Berkeley
