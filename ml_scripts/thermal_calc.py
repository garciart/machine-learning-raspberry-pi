#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Thermal data generator.
Python version used: 3.6.8
Required package(s): pythermalcomfort
Styling guide: PEP 8 -- Style Guide for Python Code
    (https://www.python.org/dev/peps/pep-0008/) and
    PEP 257 -- Docstring Conventions
    (https://www.python.org/dev/peps/pep-0257/)
"""

from pythermalcomfort.models import pmv_ppd
from pythermalcomfort.utilities import v_relative

LABEL_NAMES = ["Cold", "Cool", "Slightly Cool",
               "Neutral", "Slightly Warm", "Warm", "Hot"]


def main():
    """Application entry point."""
    # measured air velocity
    v_rel = v_relative(v=0.1, met=1.0)
    # Temperature in Celsius (10C (50F) to 36C (96.8))
    for temp in range(10, 37):
        # Relative humidity (0 to 100%)
        for humid in range(0, 101):
            # Determine the PMV index per the ASHRAE 55 2017
            results = pmv_ppd(tdb=temp, tr=temp, vr=v_rel, rh=humid,
                              met=1.0, clo=0.61, wme=0, standard="ASHRAE")
            pmv_index = 0
            if results["pmv"] < -2.5:
                # Cold
                pmv_index = 0
            elif -2.5 <= results["pmv"] < -1.5:
                # Cool
                pmv_index = 1
            elif -1.5 <= results["pmv"] < -0.5:
                # Slightly Cool
                pmv_index = 2
            elif -0.5 <= results["pmv"] < 0.5:
                # Neutral
                pmv_index = 3
            elif 0.5 <= results["pmv"] < 1.5:
                # Slightly Warm
                pmv_index = 4
            elif 1.5 <= results["pmv"] < 2.5:
                # Warm
                pmv_index = 5
            elif results["pmv"] >= 2.5:
                # Hot
                pmv_index = 6

            # print PMV value and classification
            # print(
            # "Temp: {0:.2f}C | Humid: {1:.2f}% | PMV: {2:.2f} ({3})".format(
            # temp, humid, results["pmv"], LABEL_NAMES[pmv_index]))

            # print the entry for the CSV file
            print("1013.25,0.1,{0:.1f},1.0,0.61,{1:.1f},{2}".format(
                humid, temp, pmv_index))


if __name__ == "__main__":
    main()
