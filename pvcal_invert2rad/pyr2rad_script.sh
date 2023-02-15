#!/bin/sh
#This is a shell script to run through the three python scripts for PYR2RAD

#Perform calibration using clear sky days
python pyrcal_inversion.py -f $1 -s $2

#Get cloud fraction by using clear sky simulation
python pyr2cloudfraction.py -f $1 -s $2


