# Smart Sensor

## Introduction

Smart Sensor is a customizable, plug-and-play analyzer, which uses TensorFlow instead of a rule-based, conditional logic system to process data.

In this demo, we will use temperature and humidity data collected by sensors, connected to a Raspberry Pi via a GrovePi HAT (Hardware Attached on Top), to determine if the thermal comfort level of a room complies with the American Society of Heating, Refrigerating and Air-Conditioning Engineers (ASHRAE) Standard 55-2017. While this would be easy to accomplish with simple if-else-then program, we wanted to demonstrate the capabilities and scalability of machine learning systems.

## Assumtions

- There is a heating, ventilation, and air conditioning system in place that can manipulate the operative temperature, air speed, and relative humidity of the room.
- The thermal comfort level is based on the predicted mean vote (PMV).

Data from [Thermal Comfort Tool](https://comfort.cbe.berkeley.edu/), Center for the Built Environment, University of California Berkeley.

## References

ASHRAE. (2017). ANSI/ASHRAE standard 55-2017; Thermal environmental conditions for human occupancy (55). Atlanta, GA: Author.

Engineering ToolBox. (2004). Illuminance - recommended light level. Retrieved February 18, 2020, from [https://www.engineeringtoolbox.com/light-level-rooms-d_708.html](https://www.engineeringtoolbox.com/light-level-rooms-d_708.html)

Guenther, S. (2019, November 7). What Is PMV? What Is PPD? The basics of thermal comfort. Retrieved from [https://www.simscale.com/blog/2019/09/what-is-pmv-ppd/](https://www.simscale.com/blog/2019/09/what-is-pmv-ppd/)

Hoyt, T., Schiavon, S., Tartarini, F., Cheung, T., Steinfeld, K., Piccioli, A., & Moon, D. (2019). CBE Thermal Comfort Tool. Retrieved from [https://comfort.cbe.berkeley.edu/](https://comfort.cbe.berkeley.edu/)
Center for the Built Environment, University of California Berkeley