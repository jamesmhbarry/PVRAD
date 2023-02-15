#!/bin/sh
#This is a shell script to run through the three python scripts for PV2RAD

#Perform calibration using clear sky days
python pvcal_inversion.py -s $1

#Invert PV power or current onto irradiance in the plane-of-array
python pv2poarad.py -s $1

#Invert onto GHI using LUT
python pvpyr2ghi_lut.py -s $1


