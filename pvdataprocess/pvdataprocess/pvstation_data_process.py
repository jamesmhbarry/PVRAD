#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 10:36:54 2018

@author: james
"""

import os
import yaml
import time
import pickle
from .data_process_functions import *
from .plotting_functions import *
#from copy import deepcopy


#%%Functions
###############################################################
###   functions 
###############################################################

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

def save_imported_data(pv_station,key,sys_info,description,binarypath,csvpath,timeres,datatypes):
    """
    Save imported data in a .data file using pickle, for further use in Python
    Also save data as .dat file, in CSV format
        
    args:
    :param pv_station: dictionary of one PV system with info and data
    :param key: string, name of PV system
    :param info: dictionary with information about PV systems
    :param description: string describing current simulation
    :param binarypath: string with path to save pickle files
    :param csvpath: string with path to save CSV files
    :param timeres: string giving time resolution
    :param datatypes: list of datatypes from config file
    
    """    
    print("Please wait while data is saved to file")
    
    start = time.time()
    
    #Remove raw data for resampled data        
    if timeres != "raw":            
        print('Removing original high frequency data')
        if "raw_data" in pv_station:                    
            del pv_station["raw_data"]    

    #Save in binary format
    filename = os.path.join(binarypath, description + '_' + key + '_' + 
                            timeres + '.data')
    with open(filename, 'wb') as filehandle:  
        # store the data as binary data stream
        pickle.dump((pv_station, sys_info.loc[[key]]), filehandle)
        
    print(('Data saved to %s' % filename))

    #Save resampled data in CSV format        
    if timeres != "raw":            
        filename = os.path.join(csvpath, description + '_' + key + '_' + 
                                timeres + '.dat')
        f = open(filename, 'w')
        f.write('#Measurement data from %s, %s, resampled to %s\n' % (key,description,timeres))
        f.write('#Multi-index: first line ("variable") refers to measured quantity\n')
        f.write('#second line ("substat") refers to measurement device\n')
        f.write('\n')                    
       
#        if key in config["data_processing"]["module_temp"]:
#            shift = config["data_processing"]["module_temp"][key]["slope"]
#            f.write('#Time shift of %.3e seconds per second has been applied\n' % shift)
        
        pv_station['df'].to_csv(f,sep=';',float_format='%.6f', 
                  na_rep='nan')
        
        f.close()
        
        print(('Resampled data saved as CSV to %s' % filename))
        
        #Temporary bit to save data for temperature model paper
        if (key == "MS_02" and "2018" in description) or\
            (key == "PV_11" and "2018" in description) or\
            (key == "PV_11" and "2019" in description):
            
            filename = os.path.join(csvpath, 'Temperature_model_data_'  +
                                description + '_' + key + '_' + timeres + '.dat')
            
            f = open(filename, 'w')                
            
            if key == "MS_02":
                f.write('#Measurement data from System 1, 2018, resampled to %s\n' % timeres)                    
                data_write = pv_station['df'].loc[:,[('T_module_C','PVTemp_1'),('T_ambient_pyr_C','Pyr012'),
                                ('Etotpoa_pyr_Wm2','Pyr012'),('v_wind_mast_ms','Windmast')]]                                
                error_etot = 5
            elif key == "PV_11":
                if "2018" in description:
                    f.write('#Measurement data from System 2A, 2018, resampled to %s\n' % timeres)
                    data_write = pv_station['df'].loc[:,[('T_module_C','PVTemp_1'),
                            ('T_ambient_pyr_C','Pyr001'),('Etotpoa_pyr_Wm2','Pyr001'),
                            ('v_wind_mast_ms','Windmast')]]
                    error_etot = 5
                elif "2019" in description:
                    f.write('#Measurement data from System 2B, 2019, resampled to %s\n' % timeres)
                    data_write = pv_station['df'].loc[:,[('T_module_C','RT1'),
                            ('T_ambient_pyr_C','Pyr038'),('Etotpoa_RT1_Wm2','RT1'),
                            ('v_wind_mast_ms','Windmast')]]
                    error_etot = 3
            
            data_write.dropna(axis='index',how='all',inplace=True)
            f.write('#PV module temperature: T_module_C, units: degrees C, uncertainty: 1 C\n')
            f.write('#Ambient temperature: T_ambient_C, units: degrees C, uncertainty: 1 C\n')
            f.write('#Plane-of-array irradiance: Gtotpoa_Wm2, units: W/m^2, uncertainty: {}%\n'.format(error_etot))
            f.write('#Windspeed: v_wind_ms, units: m/s, uncertainty 0.15 m/s\n')                                            
            f.write('\n')                   
            
            data_write.to_csv(f,sep=';',float_format='%.6f',header=['T_module_C',
                                'T_ambient_C','Gtotpoa_Wm2','v_wind_ms'],
                                index_label='Timestamp_UTC',na_rep='nan')
        
            f.close()
        
        if key == "MS_01":
           filename = os.path.join(csvpath, 'Longwave_thermal_emission_data_'  +
                                description + '_' + key + '_' + timeres + '.dat') 
           f = open(filename, 'w')
           f.write('#Measurement data from master station 01, resampled to %s\n' % timeres)
           f.write('#Global longwave irradiance from 2nd standard Pyrgeometer: Etotdownlw_Wm2, units: W/m^2, uncertainty: 2%\n')                                           
           f.write('\n')       
           data_write = pv_station['df'].loc[:,[('Etotdownlw_CGR4_Wm2','mordor')]]
           
           data_write.dropna(axis='index',how='all',inplace=True)

           data_write.to_csv(f,sep=';',float_format='%.6f', 
                  na_rep='nan',header=['Etotdownlw_Wm2'],
                  index_label='Timestamp_UTC')
           
           f.close()
                       
    
    end = time.time()
    savetime = end - start
        
    return savetime

#%%Main Program  
#######################################################################
###                         MAIN PROGRAM                            ###
#######################################################################
# This program imports all the data from the MetPVNet Messkampagne and saves
# it to both binary streams (with pickle) and to CSV
# The data is checked for duplicates, errors and also resampled to different resolutions
# that can be specified in the config files

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("configfile", help="yaml file containing config")
    parser.add_argument("-d","--description",help="description of the data")    
    parser.add_argument("-s","--stations",nargs='+',help="stations to process")
    args = parser.parse_args()

    config_filename = os.path.abspath(args.configfile) #"../../pvcal/config_data_2018_messkampagne.yaml" #"../../pvcal/config_data_2018_messkampagne.yaml" #
    
    #Read in values from configuration file
    config = load_yaml_configfile(config_filename)
    
    ##Description used for saving outpout files
    if args.description is not None:
        sim_info = args.description
    else:
        sim_info = config["description"]
               
    print(('Importing data for %s' % sim_info))
    
    homepath = os.path.expanduser('~')
    
    #Choose which stations to extract
    if args.stations:
        stations = args.stations        
        if stations[0] == 'all':
            stations = 'all'
    else:        
        stations = config["stations"]
    
    #Data types from config file
    data_types = config['data_types']
    
    #Define saving paths
    savebinarypath = os.path.join(homepath,config["paths"]["savedata"]["main"],
                       config["paths"]["savedata"]["binary"])
    savecsvpath = os.path.join(homepath,config["paths"]["savedata"]["main"],
                      config["paths"]["savedata"]["csv"])    
    
    #Extract config and load data
    pvsys, select_system_info, loadtime, paths = extract_config_load_data(config,stations,homepath,sim_info)
    
    #Resampling section
    time_res = config["time_resolution"]
    
    savetime_total = 0.
    for key in pvsys:        
        #group raw data together
        pvsys[key].update({"raw_data":{}})
        for idata in data_types:
            if idata in pvsys[key]:
                pvsys[key]["raw_data"].update({idata:pvsys[key][idata]})                                        
                del pvsys[key][idata]
        
        #resample data            
        savetime_res = 0.
        if config["resample"]:
            for res in time_res:                
                print(('Resampling data to %s resolution' % res))                            
                df_rs, raw_data_new = resample_interpolate_merge(pvsys[key]["raw_data"],
                              key,res,config["data_processing"],data_types)             
                pvstat_rs = {}                
                pvstat_rs.update({'lat_lon':pvsys[key]['lat_lon']})                
                pvstat_rs.update({'df':df_rs})            
                pvstat_rs.update({"raw_data":raw_data_new})
    
                if config["save_flag"]["resampled"]:        
                    savetime_res = savetime_res + save_imported_data(pvstat_rs,key,select_system_info,
                                              sim_info,savebinarypath,savecsvpath,res,data_types)        
            
        #Save data to file
        savetime = 0.
        if config["save_flag"]["raw"]:
            #Here we don't resample but just remove duplicates
            df_rs, raw_data_new = resample_interpolate_merge(pvsys[key]["raw_data"],
                              key,"raw",config["data_processing"],data_types)  
    
            pvsys[key]["raw_data"] = raw_data_new           
            print('Saving data to file....')
            savetime = savetime + save_imported_data(pvsys[key],key,select_system_info,sim_info,savebinarypath
                                          ,savecsvpath,"raw",data_types)
        else:
            savetime = 0.        
    
        savetime_total = savetime_total + savetime + savetime_res
        
    #Plotting section
    start = time.time()
    
    plot_flags = config["plot_flags"]
    
    plot_styles = config["plot_styles"]
    
    #plot PV data
    if plot_flags['pv']:
        plot_pv_data(pvsys,homepath,paths,select_system_info,plot_styles,sim_info)
    
    #plot radiation data
    if plot_flags['irrad']:
        plot_rad_data(pvsys,homepath,paths,select_system_info,plot_styles,sim_info)
        
    #plot temperature data
    if plot_flags['temp']:
        plot_temp_data(pvsys,homepath,paths,select_system_info,plot_styles,sim_info)
    
    #plot combined data
    if plot_flags['combo_p_rad']:
        pvsys = plot_combined_data_power_rad(pvsys,homepath,paths,select_system_info,plot_styles,sim_info)
        
    if plot_flags['combo_p_temp']:
        pvsys = plot_combined_data_power_temp(pvsys,homepath,paths,select_system_info,plot_styles,sim_info)
    
    end = time.time()
    plottime = end - start
    
    print(("Loading data took %g seconds" % loadtime))
    print(("Saving data took %g seconds" % savetime))
    if any(vals for vals in list(plot_flags.values())):
        print(("Plotting data took %g seconds" % plottime))
        totaltime = loadtime + savetime + plottime
    else:
        print("No plots were made")
        totaltime = loadtime + savetime
    print(("Total processing time was %g seconds" % totaltime))    
 
if __name__ == "__main__":
    main()
