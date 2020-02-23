# Smart Sensor

> ***DISCLAIMER** - This repository and its contents are not endorsed by, sponsored by, affiliated with, nor associated with the National Aeronautics and Space Administration (NASA). This repository is only a demonstration of some of the concepts that I explored while I was an intern at NASA's Langley Research Center.*

## Introduction

While I was an intern at NASA, I noticed a lot of interest in "smart buildings", i.e., buildings or public areas with built-in support for Internet of Things (IoT) devices.

Smart Sensor is a customizable, plug-and-play analyzer, written in Python, which uses machine learning instead of a rule-based, conditional logic system to process data. It can also act on the results of that data on the "edge". The Smart Sensor can accept static and streaming input from multiple sources within the Internet of Things (IoT), such as data files, sensors, audio and video feeds, etc. It can accept this data through both over-the-air and hard-wired connections.

The data, in the form of a numerical vector, consists of readings from the system to be analyzed, such as speed, temperature, vibration, etc. These vectors can scale from one reading to multiple sets with hundreds of simultaneous readings. Once the Smart Sensor receives the data, it runs it through a local machine learning classifier engine, such as scikit learn or TensorFlow, which allows it to verify performance, detect anomalies, or identify trends. Afterwards, it can display the results; it can alert the user; or it can take corrective action on its own through an actuator. This system is designed to complement or replace rule-based, conditional logic systems, such as engine control units (i.e., ECU's, which use lookup tables), electronic flight control systems (i.e., EFCS's, which use flight control laws), HVAC control systems (which use target states), etc.

In this demo, we will collect environmental data to determine if the thermal comfort level of a room complies with the American Society of Heating, Refrigerating and Air-Conditioning Engineers (ASHRAE) Standard 55-2017. This standard helps engineers design effective heating, ventilation, and air conditioning (HVAC) systems.

## Key Terms

- Features (aka the x's, input variable names) - The names of each value in a vector of values (e.g., temperature, humidity). Together, they imply a relationship (e.g., hot or cold, etc.).
- Labels (aka the y's, classes, targets, output variable names) - The name of the relationship between the values (e.g., hot or cold, etc.).
- Weights (aka the thetas) - The actual values in the vector of features (e.g., 37 deg C, 100%, etc.).
- Dataset (aka dataframe) - The collection of all the feature weights and label values.
- Classifier - An algorithm that tries to predict the relationship between features from unlabeled data.
- Model - An algorithm that accounts for the relationship between features and the correct label.

## Steps

While the system will only measure operative temperature and relative humidity, it can also process atmospheric pressure, air speed, metabolic rate, and the insulating effects of clothing level. It will predict if the occupants feel cold, cool, slightly cool, neutral, slightly warm, warm, or hot.

1. The application loads and parses the dataset, and tests all classifiers against the dataset.
2. The application creates a model using the highest scoring classifier.
3. The application then requests data from the sensors.
4. The application runs the data against the model using the selected classifier.
5. If the result is "Neutral", it complies with ASHRAE Standard 55-2017. The application displays the data on its LCD screen and indicates compliance by lighting a green LED. The application can also forward this data to a supervisory control and data acquisition (SCADA) system via JSON.
6. If the result is anything but "Neutral), it does NOT comply with ASHRAE Standard 55-2017. The application displays the data on its LCD screen and indicates non-compliance by lighting a red LED. The application can also forward this data to a supervisory control and data acquisition (SCADA) system via JSON and/or adjust the HVAC system autonomously.
7. The application repeats steps 3 through 7 until it is shut off.

Even though the dataset contains six features, we will limit our data collection to temperature and humidity, using a Seeed DHT11 sensor connected to a Raspberry Pi via a GrovePi HAT (Hardware Attached on Top). While this task would be easy to accomplish with simple if-else-then program, we wanted to demonstrate the capabilities of a machine learning system.

In addition, an if-else-then system is not be as scalable as a machine learning system. For instance, by adding a barometer and a pitot tube with a MPXV7002 transducer, we could also measure air pressure and speed without modifying the code. Also, we could add an additional column for outside air temperature and feed data into our system by an ESP8266/ESP32 with a DHT11 or 22 without modifying the evaluation code. The cost effectiveness of machine learning systems becomes apparent as the number of inputs and the size of the data increases.

| atmo_pres | air_speed | rel_humid | meta_rate | cloth_lvl | oper_temp | sens_desc |
|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
|   1013.25 |       0.1 |      50.0 |       1.0 |      0.61 |      23.0 |         2 |
|   1013.25 |       0.1 |      60.0 |       1.0 |      0.61 |      26.0 |         3 |
|   1013.25 |       0.1 |      70.0 |       1.0 |      0.61 |      28.0 |         4 |

## Data Source

Data verified using the [Thermal Comfort Tool](https://comfort.cbe.berkeley.edu/), Center for the Built Environment, University of California Berkeley.

## Features

- Atmospheric Pressure
- Air Speed
- Relative Humidity
- Metabolic Rate
- Clothing Level
- Operative Temperature

## Classes/Labels

- Cold (0)
- Cool (1)
- Slightly Cool (2)
- Neutral (3)
- Slightly Warm (4)
- Warm (5)
- Hot (6)

## References

ASHRAE. (2017). ANSI/ASHRAE standard 55-2017; Thermal environmental conditions for human occupancy (55). Atlanta, GA: Author.

Engineering ToolBox. (2004). Illuminance - recommended light level. Retrieved February 18, 2020, from [https://www.engineeringtoolbox.com/light-level-rooms-d_708.html](https://www.engineeringtoolbox.com/light-level-rooms-d_708.html)

Guenther, S. (2019, November 7). What Is PMV? What Is PPD? The basics of thermal comfort. Retrieved from [https://www.simscale.com/blog/2019/09/what-is-pmv-ppd/](https://www.simscale.com/blog/2019/09/what-is-pmv-ppd/)

Hoyt, T., Schiavon, S., Tartarini, F., Cheung, T., Steinfeld, K., Piccioli, A., & Moon, D. (2019). CBE Thermal Comfort Tool. Retrieved from [https://comfort.cbe.berkeley.edu/](https://comfort.cbe.berkeley.edu/)
Center for the Built Environment, University of California Berkeley