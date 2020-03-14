# Smart Sensor

![Smart Sensor Animation](README/smart_sensor.gif "Smart Sensor Animation")

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

While I was an intern at NASA, I noticed a lot of interest in "smart" technologies, e.g., "Edge" computing and Internet of Things (IoT) devices; augmented and virtual reality; machine learning; etc.

Since there's no MicroCenter or Home Depot in space (and space delivery is not included in Amazon Prime), one of the things I thought about was a portable, customizable, plug-and-play data analyzer, aka a smart sensor suite (S3). For less than $100, the S3, with a set of basic sensors, could replace the following devices:

- Gas Detector (O2, LEL, CO-H2S, NH3, NO2): $2571.91
- Light Meter $94.95
- Sound Level Meter: $76.68
- Temperature and Humidity Meter: $19.99
- Universal Remote Control: $9.99

In addition, users could add or create ad hoc sensors and actuators from simple components using wires, resistors, breadboards, etc. As a bonus, the S3 could communicate over Bluetooth, RF, and WiFi wireless networks.

This device would also use machine learning, as well as rule-based, conditional logic, to analyze data on the "edge" and act on the results through actuators. This system would complement or replace rule-based, conditional logic systems, such as engine control units (i.e., ECU's, which use lookup tables), electronic flight control systems (i.e., EFCS's, which use flight control laws), heating, ventilation, and air conditioning (HVAC) control systems (which use target states), etc.

In this demo, we will create a device that collects and analyzes environmental data to determine if the thermal comfort level of a room complies with the American Society of Heating, Refrigerating and Air-Conditioning Engineers (ASHRAE) Standard 55-2017. This standard helps engineers design effective HVAC systems.

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

   - If they are not installed, follow the instructions at [https://www.raspberrypi.org/documentation/linux/software/python.md]( https://www.raspberrypi.org/documentation/linux/software/python.md). You may have to install Python 3.6 using the instructions found at [https://installvirtual.com/install-python-3-on-raspberry-pi-raspbian/](https://installvirtual.com/install-python-3-on-raspberry-pi-raspbian/).

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
   pi@dex:~/SmartSensor $ pip3 install --no-cache-dir -r --upgrade requirements.txt
   ```

6. Our next step is to test the sensors:

   - Connect them as follows:

     - DHT (11 or 22) to digital port 7
     - RGB LCD to I2C port 2
     - Green LED to digital port 5
     - Red LED to digital port 6

   ![GrovePi Connections](README/smart_sensor_01.png "GrovePi Connections")

   - Once they are connected, run the following command:

   ```linux
   pi@dex:~ /SmartSensor $ cd s3_scripts
   pi@dex:~ /SmartSensor/s3_scripts $ ./sensors_test.py
   ```

   - We should see results similar to the following, but with different values for temperature and humidity:

   ```linux
   pi@dex:~ /SmartSensor/s3_scripts $ Temperature: 20.5C | Humidity: 30%
   ```

7. Once the sensor test is complete, verify scikit-learn, TensorFlow, and the sensors work (disregard any TensorFlow warnings for now). We recommend examining the code as it runs:

   ```linux
   pi@dex:~ /SmartSensor/s3_scripts $ cd ..
   pi@dex:~ /SmartSensor $ cd ml_scripts
   pi@dex:~ /SmartSensor/ml_scripts $ ./tf_premade.py
   pi@dex:~ /SmartSensor/ml_scripts $ ./sl_example.py
   pi@dex:~ /SmartSensor/ml_scripts $ ./tf_example.py
   ```

8. Notice that the scikit-learn example (sl_example.py) ran faster (4.6 sec vs 14.4 and 13.5 sec) and was much more accurate than the TensorFlow scripts (tf_premade.py and tf_example.py). We believe this is due to the dataset having six features, but with variations in only two of those features. If you run the scripts against the Iris dataset (remove the comments surrounding the Iris test and place them around the Thermal Comfort test), the accuracy of the TensorFlow scripts increases dramatically. However, for the S3, we will use the scikit classifiers.

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

9. Finally, we'll put everything together in smart_sensor.py:

   ![Smart Sensor Animation](README/smart_sensor.gif "Smart Sensor Animation")

   - Test the Sensors (Optional) - First, we will make sure all the sensors and actuators work. In our case, the sensor is the DHT-11, and the actuators are the two LEDs and the RGB LCD. For production, you may remove this code if you like.

   - Prepare the Model: Next, we will train all the classifiers as we did in sl_example.py. We will then pick the most accurate classifier for our model and retrain it against the whole data set. You will not get the same classifier all the time; during our test runs, we used the Extra Trees, the Gradient Boosting, and the Random Forest classifiers.

   - Check the Model (Optional): This is another optional step; We will check the accuracy of the selected model against some sample data. By the way, even though the data is unlabeled, we know what the resulting predictions should be. Once again, for production, you may remove this code if you like.

   - Collect Sensor Data: This is GrovePi specific code. We collect three samples of temperature and humidity from the DHT sensor, reformat it into a list of tuples, and send it for processing.

   - Process Sensor Data: Here, we run the collected data against the selected model. We collect the results, average them together, and make a determination of the conditions in the room based on the ASHRAE 7-point scale for thermal comfort.

   - Finally, we shutdown the sensors and actuators to extend their working life.

   - Here are the results of a sample run:

   ```linux
   pi@dex:~ /SmartSensor/ml_scripts $ cd ..
   pi@dex:~ /SmartSensor $ cd s3_scripts
   pi@dex:~ /SmartSensor/s3_scripts $ ./smart_sensor.py
   Smart Sensor Application.

   Testing sensors...
   Temperature: 22.0C | Humidity: 45.0%
   Temperature: 22.0C | Humidity: 44.0%
   Temperature: 22.0C | Humidity: 44.0%
   Test complete.

   Training and testing model...
   Model selected: Extra Trees Classifier (1.00%).
   Training and testing model complete.
   Elapsed time: 15.283817291259766 seconds.

   Checking model against unlabeled data...
   Data to be evaluated:
   Sample #1: [[1013.25, 0.1, 50.0, 1.0, 0.61, 23.0]] = Slightly Cool
   Sample #2: [[1013.25, 0.1, 60.0, 1.0, 0.61, 26.0]] = Neutral
   Sample #3: [[1013.25, 0.1, 76.0, 1.0, 0.61, 28.0]] = Slightly Warm
   Prediction(s):
   Sample #1: Prediction: Slightly Cool (expected Slightly Cool)
   Sample #2: Prediction: Neutral (expected Neutral)
   Sample #3: Prediction: Slightly Warm (expected Slightly Warm)

   Collecting sensor data...
   Sample #1 collected: [[1013.25, 0.1, 44.0, 1.0, 0.61, 22.0]]
   Sample #2 collected: [[1013.25, 0.1, 44.0, 1.0, 0.61, 22.0]]
   Sample #3 collected: [[1013.25, 0.1, 44.0, 1.0, 0.61, 22.0]]
   Collection complete.

   Processing sensor data...
   Sensor data #1: Prediction: Slightly Cool
   Sensor data #2: Prediction: Slightly Cool
   Sensor data #3: Prediction: Slightly Cool
   Overall sensation: 2 (Slightly Cool)

   Shutting down board...
   Job complete. Have an excellent day.
   pi@dex:~ /SmartSensor/s3_scripts $
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

Like we stated earlier, for extra credit, we will demonstrate how to collect and process data over-the-air (OTA) from satellite devices using the S3 and a remote Raspberry Pi Zero W with a Waveshare Sense HAT (B). The Sense HAT is much more accurate than our DHT-11 and also provides us with barometric pressure readings.

> *If you are going to use the Sense HAT (B), you need to install the BCM2835 library. For instructions, check out [https://www.waveshare.com/wiki/Sense_HAT_(B)](https://www.waveshare.com/wiki/Sense_HAT_(B))*

1. Turn on both the Raspberry Pi 3 Model B+ and the Pi Zero W.

> *Note - By the way, for this demo, you do not need the GrovePi HAT; you can remove all the GrovePi code. We just like the lights (and the board we printed out on our Ender 3. Thanks Chris Cirone at https://www.thingiverse.com/thing:2161971!)*

2. Make and test the connection:

   - Once they are started, make sure sock_test_server.py is on the S3 and sock_test_client.py on the Pi Zero.

   - Get the IP address of each device (while there are many ways of doing this, we found the easiest way was to execute "hostname -I" at the command line)

   - Open both scripts and replace the HOST value with the IP address of the device (leave PORT 333 the same, unless you are already using it).

   - Test the connection by running sock_test_server.py on the S3 first, and then running sock_test_client.py on the Pi Zero (you may have to use sudo to run the scripts). You should get results similar to the following:

     - S3:

     ```linux
     pi@dex:~ /SmartSensor/s3_scripts $ sudo ./sock_test_server.py
     Waiting for data...
     Received 'Hello, friend.' from client!
     Waiting for data...
     Received 'Hello, friend.' from client!
     Waiting for data...
     Received 'Good-bye!' from client!
     The client has signed off.
     pi@dex:~ /SmartSensor/s3_scripts $
     ```

     - Pi Zero W:

     ```linux
     pi@raspberrypi:~ /SmartSensor $ sudo ./sock_test_client.py
     Sending data...
     Received 'Hello back!' from server!
     Sending data...
     Received 'Hello back!' from server!
     Sending data...
     Received 'Good-bye!' from server!
     Signing off: Good-bye.
     pi@raspberrypi:~ /SmartSensor $
     ```



## References

ASHRAE. (2017). ANSI/ASHRAE standard 55-2017; Thermal environmental conditions for human occupancy (55). Atlanta, GA: Author.

Engineering ToolBox. (2004). Illuminance - recommended light level. Retrieved February 18, 2020, from [https://www.engineeringtoolbox.com/light-level-rooms-d_708.html](https://www.engineeringtoolbox.com/light-level-rooms-d_708.html)

Guenther, S. (2019, November 7). What Is PMV? What Is PPD? The basics of thermal comfort. Retrieved from [https://www.simscale.com/blog/2019/09/what-is-pmv-ppd/](https://www.simscale.com/blog/2019/09/what-is-pmv-ppd/)

Hoyt, T., Schiavon, S., Tartarini, F., Cheung, T., Steinfeld, K., Piccioli, A., & Moon, D. (2019). CBE Thermal Comfort Tool. Retrieved from [https://comfort.cbe.berkeley.edu/](https://comfort.cbe.berkeley.edu/)
Center for the Built Environment, University of California Berkeley
