#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 10:13:58 2020

@author: james
"""

import os
import yaml
import pickle
import pandas as pd

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

def load_resampled_data(timeres,config,home):
    """
    Load data that has already been resampled to a specified time resolution
    
    args:    
    :param timeres: string, timeresolution of the data
    :param config: dictionary with paths for loading data
    :param home: string, homepath
    
    out:
    :return pv_systems: dictionary of PV systems with dataframes and other information
    :return sys_info: table with information about each station
    """

    savedir = os.path.join(home,config["paths"]["savedata"])
    files = list_files(savedir)
        
    #Choose which stations to load
    if config["stations"] == "all":
        #get system info from Excel table
        sys_info  = pd.read_excel(os.path.join(savedir,config["configtable"]),index_col=0)                
        stations = sys_info.index
    else:
        sys_info = pd.DataFrame()
        stations = config["stations"]
        if type(stations) != list:
            stations = [stations]
        
    pv_systems = {}    
    
    for station in stations:
        filename = config["description"] + '_' + station + "_" + timeres + ".data"
        if filename in files:        
            with open(os.path.join(savedir,filename), 'rb') as filehandle:  
                (pvstat, info) = pickle.load(filehandle)            
            pv_systems.update({station:pvstat})
            sys_info = pd.concat([sys_info,info],axis=0)
            print('Data for %s loaded from %s' % (station,filename))
        else:
            print('Required file not found')

    return pv_systems, sys_info   

def save_to_text(pv_systems,config,description,timeres,home):
    """
    

    Parameters
    ----------
    pv_systems : dictionary
        dictionary with all information about the PV systems, and dataframes
    config : dictionary
        dictionary with information from config file
    description : string
        description of current campaign
    timeres : string
        string with time resolution
    home : string
        homepath as a string

    Returns
    -------
    None.

    """
    
    path = os.path.join(home,config["paths"]["savedata"])
    
    print("Please wait while data is saved to file")
       
    description = description[:-8]
    for key in pv_systems:
        filename = os.path.join(path, 'irrad_' + description + '_' + key + '_' + 
                                timeres + '.dat')
        f = open(filename, 'w')
        f.write('#Irradiance data from MS_01, %s\n' % description)
       
#        if key in config["data_processing"]["module_temp"]:
#            shift = config["data_processing"]["module_temp"][key]["slope"]
#            f.write('#Time shift of %.3e seconds per second has been applied\n' % shift)
        
        pv_systems[key]['df'].to_csv(f,sep=';',float_format='%.6f', 
                  na_rep='nan')
        
        f.close()
        
        print('Data saved to %s' % filename)


#%%  
#######################################################################
###                         MAIN PROGRAM                            ###
#######################################################################
#def main():
#    import argparse
#    
#    parser = argparse.ArgumentParser()
#    parser.add_argument("configfile", help="yaml file containing config")
#    parser.add_argument("-d","--description",help="description of the data")    
#    args = parser.parse_args()

config_filename = "/home/james/MetPVNet/Code/Current/pvcal/config_data_2019_messkampagne.yaml" #os.path.abspath(args.configfile) #

#Read in values from configuration file
data_config = load_yaml_configfile(config_filename)

#Description used for saving outpout files
#    if args.description is not None:
#        sim_info = args.description
#    else:
sim_info = data_config["description"]
           
homepath = os.path.expanduser('~')

#Choose which stations to extract
stations = data_config["stations"]

time_res = '15min' #min' #pvtemp_config["timeres"]

print("Loading resampled data")
pvsys, select_system_info = load_resampled_data(time_res,data_config,homepath)

save_to_text(pvsys,data_config,sim_info,time_res,homepath)