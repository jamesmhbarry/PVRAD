#!/bin/sh
#This is a shell script to run through the analysis scripts

#COD analysis plots
python pvpyr2cod_analysis.py -s $1

#Irradiance analysis plots
python pvpyr2ghi_analysis.py -s $1



