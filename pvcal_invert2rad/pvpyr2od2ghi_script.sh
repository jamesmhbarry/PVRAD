#!/bin/sh
#This is a shell script to run through the three python scripts for PV2RAD

#Extract AOD from DISORT LUT
python pvpyr2aod2rad_interpolate_fit.py -s $1

#Extract COD from DISORT LUT
python pvpyr2cod2rad_interpolate_fit.py -s $1



