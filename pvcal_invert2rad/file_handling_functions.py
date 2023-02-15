#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 13:38:37 2021

@author: james

These are different function used to handle files, commonly used in all the 
PVCAL and PVRAD programs

"""

import os
import yaml
import datetime

def list_dirs(path):
    """
    list all directories in a given directory
    
    args:
    :param path: string with the path to the search directory
    
    out:
    :return: all directories within the search directory
    """
    dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path,d))]
    return sorted(dirs)

def list_files(path):
    """
    lists all filenames in a given directory
    args:
    :param path: string with the path to the search directory
    
    out:
    :return: all files within the search directory
    """
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path,f))]
    return sorted(files)        

def load_yaml_configfile(fname):
    """
    load yaml config file
    
    args:
    :param fname: string, complete name of config file
    
    out:
    :return: config dictionary
    """
    with open(fname, 'r') as ds:
        try:
            config = yaml.load(ds,Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)
    return config     

def create_daterange(dates):
    """
    :param dates: list of 2 element list with
    tin and tend of datetrange. tin/tend are expected
    to be of type datetime.date or datetime.datetime.
    """
    if type(dates[0]) == list:
        datelist = []
        for tin, tend in dates:
            tin  = datetime.datetime(tin.year, tin.month, tin.day)
            tend = datetime.datetime(tend.year, tend.month, tend.day)
            dt = tend - tin
            drange = [tin+datetime.timedelta(days=i) for i in range(dt.days+1)]
            datelist += drange
            
        dates = sorted(datelist)
        
    dates_str = [datetime.datetime.strftime(d, "%Y-%m-%d") for d in dates] 

    return dates_str

def merge_two_dicts(x, y):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z  


