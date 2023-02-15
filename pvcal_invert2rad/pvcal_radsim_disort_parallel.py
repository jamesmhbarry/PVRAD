#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 10:36:54 2018

@author: james
"""

import os
import tempfile
import numpy as np
import pandas as pd
import ephem
import subprocess
import time
import datetime
import pickle
from file_handling_functions import *
import data_process_functions as dpf
from rt_functions import *
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from geopy import distance


###############################################################
###   functions 
###############################################################



def select_simulation_days(sys_info,pv_systems,days,sens_type):
    """
    Select only the days we want, defined by the simulation days in config file
    Check if days we want are contained in the data!
    
    args:
    :param sys_info: dataframe with station information from table
    :param pv_systems: dictionary of PV systems with dataframes
    :param days: list of clear sky days
    :param sens_type: type of sensor for simulation
    
    out:
    :return pv_systems: dictionary of PV systems with dataframe containing clear sky days
    :return sys_info: dataframe with station information from table
    """    
    
    #check if there are any empty dataframes or ones with no PV data, delete from dictionary if empty
    empty_stations = []
    for key in pv_systems:        
        if pv_systems[key]['df'].empty:
            empty_stations.append(key)
            print('Station %s has no data' % key)
        if sens_type == "pv":
            if 'P_kW' not in pv_systems[key]['df'].columns.levels[0] and\
                'Pdc1_1' not in pv_systems[key]['df'].columns.levels[0]:            
                empty_stations.append(key)
                print('Station %s has no PV data' % key)
        elif sens_type == "pyranometer":
            if 'Etotpoa_pyr_Wm2' not in pv_systems[key]['df'].columns.levels[0] and\
                'Etotdown_pyr_Wm2' not in pv_systems[key]['df'].columns.levels[0]:
                empty_stations.append(key)
                print('Station %s has no pyranometer data' % key)
            
    for key in empty_stations:
        del pv_systems[key]            
        sys_info.drop(key,axis=0,inplace=True)        
             
    #Select only the data we want
    for key in pv_systems:
        sim_days_index = pd.DatetimeIndex(days)
        pv_systems[key]['df'] = pv_systems[key]['df'].loc[np.isin(pv_systems[key]['df'].index.date,
                           sim_days_index.date)]
        pv_systems[key]['df'].dropna(axis=0,how='all',inplace=True)
    
        new_days_index = sim_days_index[np.isin(sim_days_index.date,
                                                pv_systems[key]['df'].index.date)]        
        pv_systems[key]['sim_days'] = new_days_index.strftime('%Y-%m-%d').tolist()
                
    return pv_systems, sys_info  
        
def sun_position(pv_systems,sza_limit):
    """
    Using PyEphem to calculate the sun position
    
    args:    
    :param pv_systems: dictionary of PV systems
    :param sza_limit: float defining maximum solar zenith angle for simulation
    
    out:
    :return: dictionary of PV systems
    
    """
    
    for key in pv_systems:
        print('Calculating sun position for %s' % key)
        dataframe = pv_systems[key]['df']
        len_time = len(dataframe)
        index_time = dataframe.index

        # initalize observer object
        observer = ephem.Observer()
        observer.lat = np.deg2rad(pv_systems[key]['lat_lon'][0])
        observer.lon = np.deg2rad(pv_systems[key]['lat_lon'][1])
    
        # define arrays for storing the angles
        sza = np.zeros(len_time)
        phi0 = np.zeros(len_time)
        
        for it in range(len_time):
            #Set time for observer
            observer.date = index_time[it]
            
            # decide which planet/star/moon...you want to observe
            sun = ephem.Sun(observer)
            
            # sun position:
            sza[it] = 90 - np.rad2deg(sun.alt) # convert to degrees, change to sza
            phi0[it]   = np.fmod(np.rad2deg(sun.az) + 180,360) # change convention for lrt
        
        #Add values to the correct dataframs    
        df_sun = pd.DataFrame({'sza':sza,'phi0':phi0},index=index_time)                
        df_sun.columns = pd.MultiIndex.from_product([df_sun.columns.values.tolist(),['sun']],
                                                                       names=['variable','substat']) 
        dataframe = pd.merge(dataframe,df_sun,left_index=True,right_index=True)
        
        #throw away values with SZA greater than 90        
        dataframe = dataframe[dataframe[('sza','sun')] <= sza_limit]        
        
        pv_systems[key]['df'] = dataframe
        
    return pv_systems

def import_aerosol_to_dataframe(station,filename,timeres,path):
    """
    Import aerosol data from aeronetmystic to dataframe
    
    args:
    :param station: dictionary with information about Aeronet stration and data
    :param filename: string with filename containing output of aeronetmystic
    :param timeres: string with time resolution 'day' or 'all'
    :param path: string, path where Aeronet fit data is stored
        
    out:
    :return: dataframe containing fitted parameters from aeronetmystic
    
    """
    
    df_name = 'df_' + timeres
    station[df_name] = pd.DataFrame()
    station[df_name] = pd.read_csv(os.path.join(path,filename),sep=' ',
           index_col=0,comment='#')
    station[df_name].index = pd.to_datetime(station[df_name].index,format='%Y-%m-%dT%H:%M:%S')
    station[df_name].index = station[df_name].index.tz_localize(tz='UTC',ambiguous='NaT')
    
    return station[df_name]

def import_aerosol_data(config_file,station,home,info):
    """
    Import data that has been processed by aerosol program
    
    args:
    :param config_file: dictionary of config information, paths etc
    :param station: dictionary with information about Aeronet station and data
    :param home: string, homepath
    :param info: string giving description of simulation
    
    out:
    :return: dictionary of aerosol stations
    """
        
    config = load_yaml_configfile(config_file)
    fit_path = os.path.join(home,config["path_fit_data"])
    filelist = os.path.join(fit_path,"aerosol_filelist_" + info + ".dat")    
        
    df_filelist = pd.read_csv(filelist, sep= ' ',index_col="Station")    
    
    station_dict = config["aeronet_stations"]
    #Import fitted parameters from aeronetmystic to dataframe
    for key in station_dict:
        filename_all = df_filelist.loc[key,"all"]
        filename_day = df_filelist.loc[key,"day_ave"]        
        station_dict[key]['df_all'] = import_aerosol_to_dataframe(station_dict[key],filename_all,'all',fit_path)
        station_dict[key]['df_day'] = import_aerosol_to_dataframe(station_dict[key],filename_day,'day',fit_path)     
        
    #Create list of only one element if there is only one source station
    if type(station) != list:
        station = [station]
        
    station_dict.update({"mean":{'df_all':pd.DataFrame(),'df_day':pd.DataFrame()}})
    
    if len(station) > 1:
        #Load data from aeronet2mystic, i.e. Fit parameters and SSA_vis
        filename_all_mean = df_filelist.loc["mean","all"]
        filename_day_mean = df_filelist.loc["mean","day_ave"]
        station_dict['mean']['df_all'] = import_aerosol_to_dataframe(station_dict['mean'],filename_all_mean,'all',fit_path)
        station_dict['mean']['df_day'] = import_aerosol_to_dataframe(station_dict['mean'],filename_day_mean,'day',fit_path)
        
    #If there is only one station then the mean is the same as that station    
    elif len(station) == 1:
        station_dict['mean']['df_all'] = station_dict[station[0]]['df_all']  
        station_dict['mean']['df_day'] = station_dict[station[0]]['df_day']
    
    return station_dict 

def find_nearest_cosmo_grid_folder(configfile,pv_systems,datatype):
    """
    Search through the output of cosmomystic or cosmopvcal to find the 
    gridpoint (folder) corresponding to the location of each PV station
    
    args:
    :param configfile: string, configfile for cosmomystic
    :param pv_systems: dictionary of pv_systems
    :param datatype: string, either surf or atmo
    
    out:
    :return pv_systems: dictionary updated with path to cosmo atmfiles
    """
    
    config = load_yaml_configfile(configfile)
    
    if datatype == "atmo":
        path = config["path_atmofiles"]
    elif datatype == "surf":
        path = config["path_surface_files"]
    
    cosmo_folders = list_dirs(path)
    
    #Define paths for COSMO-modified atmosphere files
    for key in pv_systems:
        for folder in cosmo_folders:
            fname = "known_stations.dat"
            ds = pd.read_csv(os.path.join(path,folder,fname),comment='#',names=['name','lat','lon'],sep=' ',
                             index_col=0)
            for station in ds.index:
                if station == key:
                    if datatype == "atmo":
                        pv_systems[key]['path_cosmo_lrt'] = os.path.join(path,folder)
                    elif datatype == "surf":
                        pv_systems[key]['path_cosmo_surface'] = os.path.join(path,folder)
    
    return pv_systems

def read_spectral_albedo(key,pv_station,folder,days):
    """
    Read in spectral Albedo from MODIS file (for record keeping only, we can implement
    the BRDF in libradtran directly) and save to dataframe
    
    args:
    :param key: string, name of PV station
    :param pv_station: dictionary of information and data about PV station
    :param folder: string, folder where files are stored
    :param days: list of days to consider    
    
    out:
    :return: series containing spectral albedo dataframe
    """
    
# Here we would have different Albedo files per station
#    dirs_exist = list_dirs(folder)
#    
#    key_folder = os.path.join(folder,key)
#    if key not in dirs_exist:
#        os.mkdir(key_folder)
        
    key_folder = folder
    files = list_files(key_folder)
        
    #Tuple of lists, days plus spectral albedo
    dfs = ([f.split('_')[-1].split('.')[0] for f in files],
            [pd.read_csv(os.path.join(key_folder,f),comment='#',sep=' ',index_col=0,
                       names=['Albedo']) for f in files])
    [df.index.rename('Wavelength_nm',inplace=True) for df in dfs[1]]
    
    albedo_series = ([iday for iday in days],['']*len(days),[pd.DataFrame()]*len(days))
    for ix, iday in enumerate(days):
        if ''.join(iday.split('-')) in dfs[0]:   
            new_ix = dfs[0].index(''.join(iday.split('-')))
            albedo_series[1][ix] = files[new_ix]
            albedo_series[2][ix] = dfs[1][new_ix]
    
    return albedo_series

def prepare_atmosphere_aerosol_albedo(config,pv_systems,station_info,home,info):
    """
    Set up atmosphere (call COSMO2MYSTIC), set up aerosol (call AERONET2MYSTIC)
    set up albedo (include MODIS data in the future)
    
    args:
    :param config: dictionary of configuration details from config file
    :param pv_systems: dictionary of pv systems    
    :param station_info: dataframe with station info from table
    :param home: string with homepath
    :param info: string giving description of simulation
    
    out:
    :return pv_systems: dictionary of pv systems  
    :return aeronet_stats: dictionary of aeronet stations
    """
    
    year = "mk_" + info.split('_')[1]
    #Set up atmosphere
    atm_source = config["atmosphere"]
    
    #If not default, call cosmo2mystic to create COSMO-modified atmosphere file
    if atm_source == 'cosmo':
        cosmo_configfile = os.path.join(home,config["cosmo_configfile"][year])
        cosmo_config = load_yaml_configfile(cosmo_configfile)
        finp = open(cosmo_configfile,'a')
        if "stations_lat_lon" not in cosmo_config:
            finp.write('# latitude, longitude of PV stations within the COSMO grid)\n')
            finp.write('stations_lat_lon:\n')
        
            for key in pv_systems:
                finp.write('    %s: [%5f,%5f]\n' % (key,pv_systems[key]['lat_lon'][0],pv_systems[key]['lat_lon'][1]))
        else:
            for key in pv_systems:
                if not cosmo_config["stations_lat_lon"]:                
                    finp.write('    %s: [%.5f,%.5f]\n' % (key,pv_systems[key]['lat_lon'][0],
                           pv_systems[key]['lat_lon'][1]))
                elif key not in cosmo_config["stations_lat_lon"]:
                    finp.write('    %s: [%.5f,%.5f]\n' % (key,pv_systems[key]['lat_lon'][0],
                           pv_systems[key]['lat_lon'][1]))
        finp.close()
        if config["cosmo_sim"]:
            # call cosmo2mystic
            child = subprocess.Popen('cosmo2mystic ' + cosmo_configfile, shell=True)
            child.wait()
            print('Running cosmo2mystic to create libRadtran atmosphere files')
        else:
            print('cosmo2mystic already run, read in atmosphere files')
        
        pv_systems = find_nearest_cosmo_grid_folder(cosmo_configfile,pv_systems,'atmo')
        
    #Import aerosol data from AERONET 
    #Configuration of aerosols
    aerosol_info = config["aerosol"]
    asl_station = aerosol_info["station"]
    
    #If Aeronet data is to be used, call the aeronet2mystic program to import aerosols
    if aerosol_info["source"] != 'default':
        aerosol_configfile = os.path.join(home,aerosol_info["configfile"][year])
        # call aeronet2mystic
        if config["aeronet_sim"]:
            child = subprocess.Popen('aeronet2mystic ' + aerosol_configfile, shell=True)
            child.wait()
            print('Running aeronet2mystic to extract Aerosol information')
        else:
            print('aeronet2mystic already run, read in Angstrom parameters')
        
        #Import data from aeronetmystic fitting procedure to dataframe
        aeronet_stats = import_aerosol_data(aerosol_configfile,asl_station,home,
                                            info)
        
        #put aerosol and albedo data into the PV systems dataframe
        if type(asl_station) == list:
            asl_station = "mean"
                        
        for key in pv_systems:            
            aeronet_distance = distance.distance(aeronet_stats[asl_station]\
                         ['lat_lon'],pv_systems[key]['lat_lon']).km
                        
            if aeronet_distance > aerosol_info["aeronet_distance_limit"]:
                aerosol_info["data_res"] = 'day'
                print(f'{key} more than {aerosol_info["aeronet_distance_limit"]} km from Aeronet station {aeronet_stats[asl_station]["name"]}, using daily averages')    
            
            #If day average then reindex            
            if aerosol_info["data_res"] == 'all':
                df_asl = aeronet_stats[asl_station]['df_all']
            elif aerosol_info["data_res"] == 'day':
                asl_days = pd.to_datetime(aeronet_stats[asl_station]['df_day'].\
                                          index.date).unique().strftime('%Y-%m-%d')  
                pv_days = pd.to_datetime(pv_systems[key]['df'].\
                                          index.date).unique().strftime('%Y-%m-%d')  
                combine_days = asl_days.intersection(pv_days)
                #Some data is not localized
                if not pv_systems[key]['df'].index.tzinfo:
                    aeronet_stats[asl_station]['df_day'].index = \
                        aeronet_stats[asl_station]['df_day'].index.tz_localize(None)
                df_asl = pd.concat([aeronet_stats[asl_station]['df_day'].loc[[day]].reindex(\
                         pv_systems[key]['df'].loc[day].index,method='pad')\
                           for day in combine_days],axis=0)
                    
            #Create multiindex
            df_asl.columns = pd.MultiIndex.from_product([df_asl.columns.values.tolist(),['Aeronet']],
                                                                   names=['variable','substat']) 
                       
            #Some data is not localized
            if not pv_systems[key]['df'].index.tzinfo:
                df_asl.index = df_asl.index.tz_localize(None)
                
            #Merge Aerosol values into the pv systems dataframe, this only finds intersection
            pv_systems[key]['df'] = pd.merge(pv_systems[key]['df'],df_asl,
                      left_index=True,right_index=True)
            
            #Check which days are still in the dataset
            sim_days_index = pd.DatetimeIndex(pv_systems[key]['sim_days'])
            new_days_index = sim_days_index[np.isin(sim_days_index.date,
                                                pv_systems[key]['df'].index.date)]        
            pv_systems[key]['sim_days'] = new_days_index.strftime('%Y-%m-%d').tolist()
            
    #If default aerosol to be used, return empty dictionary
    else:
        aeronet_stats = {}
    
    #Albedo
    #Setup albedo
    albedo_config = rt_config["albedo"]  
    
    #Read in constant Albedo even if MODIS is selected, in case of errors or missing data
    albedo = albedo_config['constant']
    print('Constant albedo set to %s' % albedo)  
    print('Looking for MODIS Albedo files')
    
    #Set up Albedo for each system
    for key in pv_systems:        
        #Days still left in the dataframe
        days = pv_systems[key]['sim_days']
        #Setup spectral albedo from MODIS     
        pv_systems[key]['albedo_series'] = read_spectral_albedo(key,pv_systems[key],
                 os.path.join(home,albedo_config['brdf_folder']),days)    
        
        pv_systems[key]['df']['albedo','MODIS_500nm'] = pd.Series(np.nan,index=pv_systems[key]['df'].index)
        for ix, iday in enumerate(days):
            if not pv_systems[key]['albedo_series'][2][ix].empty:
                pv_systems[key]['df']['albedo','MODIS_500nm'].loc[iday] = \
                pv_systems[key]['albedo_series'][2][ix].Albedo.loc[500]
                
        #Setup daily variable albedo (constant values)
        pv_systems[key]['df'][('albedo','constant')] = pd.Series(np.nan,index=pv_systems[key]['df'].index)
        for ix, iday in enumerate(days):
            pv_systems[key]['df'].loc[iday,('albedo','constant')] = albedo #[ix]

    return pv_systems, aeronet_stats 
 
def generate_folders(config,sens_type,path):
    """
    Generate folders for results dependent on choice of input parameters
    
    args:
    :param config: dictionary with DISORT config
    :param sens_type: string with type of sensor (pv or pyranometer)
    :param path: main path for saving files or plots    
    
    out:
    :return fullpath: string with label for saving folders
    :return res_string_label: string with label for saving files
    
    """

    #atmosphere model
    atm_geom_config = config["disort_base"]["pseudospherical"]
    
    if atm_geom_config == True:
        atm_geom_folder = "Pseudospherical"
    else:
        atm_geom_folder = "Plane-parallel"
        
    dirs_exist = list_dirs(path)
    fullpath = os.path.join(path,atm_geom_folder)
    if atm_geom_folder not in dirs_exist:
        os.mkdir(fullpath)
    
    #Wavelength range of simulation
    wvl_config = config["common_base"]["wavelength"][sens_type]
    
    if type(wvl_config) == list:
        wvl_folder_label = "Wvl_" + str(int(wvl_config[0])) + "_" + str(int(wvl_config[1]))
    elif wvl_config == "all":
        wvl_folder_label = "Wvl_Full"

    dirs_exist = list_dirs(fullpath)
    fullpath = os.path.join(fullpath,wvl_folder_label)        
    if wvl_folder_label not in dirs_exist:
        os.mkdir(fullpath)
    
    #Disort grid resolution
    disort_config = config["disort_rad_res"]
    theta_res = str(disort_config["theta"]).replace('.','-')
    phi_res = str(disort_config["phi"]).replace('.','-')
    
    disort_folder_label = 'Zen_' + theta_res + '_Azi_' + phi_res
    res_string_label = 'disortres_' + theta_res + '_' + phi_res + '_'
    
    dirs_exist = list_dirs(fullpath)
    fullpath = os.path.join(fullpath,disort_folder_label)
    if disort_folder_label not in dirs_exist:
        os.mkdir(fullpath)
    
    #Atmosphere input (COSMO or default)
    dirs_exist = list_dirs(fullpath)
    if config["atmosphere"] == "default":
        atm_folder_label = "Atmos_Default"    
    elif config["atmosphere"] == "cosmo":
        atm_folder_label = "Atmos_COSMO"
    
    fullpath = os.path.join(fullpath,atm_folder_label)
    if atm_folder_label not in dirs_exist:
        os.mkdir(fullpath)
    
    #Aerosol input (Aeronet or default)
    dirs_exist = list_dirs(fullpath)
    if config["aerosol"]["source"] == "default":
        aero_folder_label = "Aerosol_Default"
    elif config["aerosol"]["source"] == "aeronet":
        aero_folder_label = "Aeronet_" + config["aerosol"]["station"]
    
    fullpath = os.path.join(fullpath,aero_folder_label)
    if aero_folder_label not in dirs_exist:
        os.mkdir(fullpath)    
    
    return fullpath, res_string_label
    
def write_comments_itime(file_handle,station,lat,lon,config,disort_res,sens_type,datarow):
    """
    Generate comments for the results files
    
    args:
    :param file_handle: file object
    :param station: string, station name
    :param lat: float, latitude
    :param lon: float, longitude
    :param config: dictionary, configuration of DISORT simulation
    :param disort_res: dictionary, disort angular resolution
    :param sens_type: string with type of sensor (pv or pyranometer)
    :param datarow: row of pandas dataframe
        
    out:
    :return: file handle object
    """
    
    file_handle.write('# DISORT simulation results for %s \n' % station)
    file_handle.write('# Sun position: sza %.6f, phi0 %.6f\n' % (datarow[('sza','sun')],
                      datarow[('phi0','sun')]))
    file_handle.write('# latitude: %g, longitude: %g\n' % (lat,lon))
    file_handle.write('# wavelength range: %s\n' % config["common_base"]["wavelength"][sens_type])
    file_handle.write('# atmosphere: %s \n' % config['atmosphere'])
    file_handle.write('# DISORT resolution, time: %s, angular: %s \n' % 
            (config["timeres"],disort_res))
    file_handle.write('# maximum SZA: %g degrees \n' % config["sza_max"]["clear_sky"])
    file_handle.write('# pseudospherical: %s \n' % config["disort_base"]["pseudospherical"])      
    file_handle.write('# aerosol source: %s\n' % config["aerosol"]["source"])
    if config["aerosol"]["source"] != "default" and pd.notna(datarow[('AOD_500', 'Aeronet')]):
        file_handle.write('# aeronet station: %s\n' % config["aerosol"]["station"])
        file_handle.write('# aerosol_angstrom: %g %g \n' % (datarow[('alpha','Aeronet')],
                          datarow[('beta','Aeronet')]))
    if "aerosol_species_library" in config["common_base"]:
        file_handle.write('# aerosol species library: %s\n' % 
            config["common_base"]["aerosol_species_library"])
    if "aerosol_species_file" in config["common_base"]:        
        file_handle.write('# aerosol species file: %s\n' %
            config["common_base"]["aerosol_species_file"])
    if config["albedo"]["choice"] == "MODIS" and pd.notna(datarow[('albedo','MODIS_500nm')]): 
        file_handle.write('# albedo MODIS spectral: %g\n' % datarow[('albedo','MODIS_500nm')])
    else:
        file_handle.write('# albedo constant: %g\n' % datarow[('albedo','constant')])
        
    file_handle.write('# mol_abs_param: %s\n' % config["common_base"]["mol_abs_param"])
    file_handle.write('# output_process: %s\n' % config["common_base"]["output_process"])
                      
    return file_handle              
    
def save_radsim_itime(key,data,df_Idiff,description,rt_config,itime,lat_lon,sens_type,home,grid):
    """
    Save results of the DISORT simulation to file for a single time stamp
    
    args:
    :param key: string, code name for PV system
    :param data: dataframe to save
    :param df_Idiff: dataframe with diffuse radiance field
    :param description: string describing simulation    
    :param rt_config: dictionary, configuration of DISORT simulation from config file    
    :param itime: timestamp, time point to save
    :param lat_lon: list of coordinates (latitute and longited) of system
    :param sens_type: string with type of sensor (pv or pyranometer)
    :param home: string, home path
    :param grid: dictionary of grid for libradtran    
    
    """
    mainpath = os.path.join(home,rt_config["save_path"]["disort"],
                            rt_config["save_path"]["clear_sky"])
    res = rt_config["disort_rad_res"]
    latitude = lat_lon[0]
    longitude = lat_lon[1]      
    
    #Generate folders for saving results
    folder_label, res_string_label = generate_folders(rt_config,sens_type,mainpath)
            
    stat_dirs_exist = list_dirs(folder_label)
    
    #Save results to individual dat files        
    if key not in stat_dirs_exist:
        os.mkdir(os.path.join(folder_label,key))    

    #Save dataframe results
    filename = 'lrt_sim_' + description + '_' + itime.strftime('%Y%m%dT%H%M%S') + '.dat'        
    f = open(os.path.join(folder_label,key,filename), 'w')
    #Write comments to the header of the file            
    f = write_comments_itime(f,key,latitude,longitude,rt_config,res,sens_type,data.loc[itime])
    
    #Write input data
    f.write('\n# Input data:\n') 
    data_write = pd.concat([data.loc[[itime],pd.IndexSlice[:,['sun','Aeronet']]]
    ,data.loc[[itime],pd.IndexSlice['albedo',:]]],axis=1)
    data_write.to_csv(f, sep=';',
           float_format='%.6f', na_rep='nan',header=True)
    #Write dataframe to file
    f.write('\n# Irradiance data:\n')
    data.xs('libradtran',level='substat',axis=1).loc[[itime]].to_csv(f, sep=';',
           float_format='%.6f', na_rep='nan',header=True,index_label='Timestamp')
    f.write('\n# Radiance distribution:\n')
    #Write radiance distribution to file                    
    df_Idiff.to_csv(f, sep = ',', float_format='%.9f',
              index_label='mu/phi', na_rep='nan', 
              header=[str(phi) for phi in grid['phi']])
    f.close()            
    print("Simulation results written to %s" % filename)             
    
def write_comments_all(file_handle,station,lat,lon,config,disort_res,sens_type):
    """
    Generate comments for the results files
    
    args:
    :param file_handle: file object
    :param station: string, station name
    :param lat: float, latitude
    :param lon: float, longitude
    :param config: dictionary, configuration of DISORT simulation
    :param disort_res: dictionary, disort angular resolution    
    :param sens_type: string with type of sensor (pv or pyranometer)

        
    out:
    :return: file handle object
    """
    
    file_handle.write('# DISORT simulation results for %s \n' % station)
    file_handle.write('# latitude: %g, longitude: %g\n' % (lat,lon))
    file_handle.write('# wavelength range: %s\n' % config["common_base"]["wavelength"][sens_type])
    file_handle.write('# atmosphere: %s \n' % config['atmosphere'])
    file_handle.write('# DISORT resolution, time: %s, angular: %s \n' % 
            (config["timeres"],disort_res))
    file_handle.write('# maximum SZA: %g degrees \n' % config["sza_max"]["clear_sky"])
    file_handle.write('# pseudospherical: %s \n' % config["disort_base"]["pseudospherical"])      
    file_handle.write('# aerosol source: %s\n' % config["aerosol"]["source"])
    if config["aerosol"]["source"] != "default":
        file_handle.write('# aeronet station: %s\n' % config["aerosol"]["station"])       
    if "aerosol_species_library" in config["common_base"]:
        file_handle.write('# aerosol species library: %s\n' % 
            config["common_base"]["aerosol_species_library"])
    if "aerosol_species_file" in config["common_base"]:        
        file_handle.write('# aerosol species file: %s\n' %
            config["common_base"]["aerosol_species_file"])
    if config["albedo"]["choice"] == "MODIS": 
        file_handle.write('# albedo MODIS spectral\n')
    else:
        file_handle.write('# albedo constant\n')
        
    file_handle.write('# mol_abs_param: %s\n' % config["common_base"]["mol_abs_param"])
    file_handle.write('# output_process: %s\n' % config["common_base"]["output_process"])
                      
    return file_handle              

  
def save_radsim_all_binary(pv_station,key,description,rt_config,sim_time,folder,
                           res_string,sens_type):
    """
    Save results of the DISORT simulation to a single binary file with pickle
    
    args:
    :param pv_station: dictionary of information and data from one PV system
    :param key: string, code name for PV system
    :param description: string describing simulation
    :param rt_config: dictionary, configuration of DISORT simulation from config file    
    :param sim_time: float, time simulation took to complete
    :param folder: string, folder to save results
    :param res_string: string to label file with DISORT resolution
    :param sens_type: string with type of sensor (pv or pyranometer)

    """    
    res = rt_config["disort_rad_res"]
    latitude = pv_station['lat_lon'][0]
    longitude = pv_station['lat_lon'][1]  
    
    #Save entire simulation to a pickle file (Python binary format)
    filename = 'lrt_sim_results_'
    if rt_config["atmosphere"] == 'cosmo':
        filename = filename + 'atm_'
    if rt_config["aerosol"]["source"] != 'default':
        filename = filename + 'asl_' +  rt_config["aerosol"]["data_res"] + '_'       
    
    filename = filename + description + '_' + res_string
        
    #Save as binary files...
    filename = filename + key + '.data'        
    with open(os.path.join(folder,filename), 'wb') as filehandle:          
        pickle.dump((pv_station, rt_config), filehandle)
    print('Simulation results written to %s' % filename)
    
    #Save a log file with the simulation configuration
    current_time = datetime.datetime.now()
    log_file = 'log_lrt_sim_' + description + '_' + key + '_' + current_time.strftime('%Y%m%dT%H%M%S')\
                + '.dat'
    f = open(os.path.join(folder,log_file),'w')
    f = write_comments_all(f,key,latitude,longitude,rt_config,res,sens_type)
    f.write('Simulation took %g seconds to complete' % sum(sim_time))
    f.close()
    
    return 
              
def read_disort_output(key,itime,dataframe,rt_config,description,grid,output_file,lat_lon,sens_type,home):
    """
    Read disort output file for one time stamp
    
    args:
    :param key: string, name of PV system to be simulated
    :param itime: timestamp to be simulated
    :param dataframe: dataframe to be filled with simulated values
    :param rt_config: dictionary of config details for simulation
    :param description: string describing simulation    
    :param grid: dictionary of grid for libradtran    
    :param output_file: string, name of DISORT output file
    :param lat_lon: list of coordinates (latitute and longited) of system
    :param sens_type: string with type of sensor (pv or pyranometer)
    :param home: string, home path
    
    out:
    :return dataframe: dataframe with results of simulation added as extra columns
    :return Idiff: dataframe with radiance field distribution for one time step
    """    
    
    # read in irradiance values calculated in libradtran
    df_irrad_out = pd.read_csv(output_file,sep='\s+',
                                 header=None,nrows=1,usecols=(1,2))
    dataframe.loc[itime,('Edirdown','libradtran')] = df_irrad_out.values[0,0]
    dataframe.loc[itime,('Ediffdown','libradtran')] = df_irrad_out.values[0,1]
    dataframe.loc[itime,('Etotdown','libradtran')] = df_irrad_out.values[0,0] + df_irrad_out.values[0,1]
    
    # read in individual radiance values as per the defined umu, phi grid                
    Idiff = pd.read_csv(output_file,sep='\s+',
                    header=None,skiprows=2,usecols=np.arange(2,len(grid["phi"]) + 2,1))
    Idiff.index = grid["umu"]
    Idiff.columns = grid["phi"]    
    
    #Integrate radiance field as a check, taking only values from "above" (no reflection)
    dataframe.loc[itime,('Ediffdown_calc','libradtran')] = int_2d_diff(Idiff.iloc[0:int(len(grid["umu"])/2)+1,:].values,
                            grid["umu"][0:int(len(grid["umu"])/2)+1],
                            grid["umu"][0:int(len(grid["umu"])/2)+1],np.deg2rad(grid["phi"]))
    
    #Save results for one time stamp to file
    save_radsim_itime(key,dataframe,Idiff,description,rt_config,itime,lat_lon,sens_type,home,grid)
        
    return dataframe, Idiff
    
def generate_disort_files(rt_config,sens_type,altitude,home,cosmo_path,modis_path,
                          itime,grid,data,albedo,n_cpu):
    """
    Generate input files for libradtran based on time series data, one for each time
    step defined by itime
    
    args:
    :param rt_config: dictionary of config details for simulation
    :param sens_type: string, PV or pyranometer
    :param altitude: float, altitude of station, if necessary
    :param home: string, home path
    :param cosmo_path: string with path for cosmo files
    :param modis_path: string with path for spectral albedo files from MODIS
    :param itime: timestamp to be simulated
    :param grid: dictionary with grid for diffuse radiance field
    :param data: data point (row of dataframe) for which file should be generated
    :param albedo: tuple of lists with albedo files from MODIS
    :param n_cpu: integer, which cpu to use
    
    out:
    :return inputfile: string, name of input file
    :return outputfile: string, name of output file
       
    """
    lrt_dir = os.path.join(home,rt_config["working_directory"])
    
    base_config = rt_config["common_base"]
    for key, value in rt_config["disort_base"].items():
        base_config.update({key:value})
    base_config["data_files_path"] = os.path.join(home,base_config["data_files_path"])
            
    #prepare input file for libradtran
    if rt_config["atmosphere"] == 'cosmo': #include COSMO-modified atmosphere
        atm_path_temp = os.path.join(home,cosmo_path,
        str(itime.date()).replace('-',''),str(itime.time()).replace(':','') +
        '_atmofile.dat')
    else:
        atm_path_temp = os.path.join(home,rt_config["atmosphere_file"])
                    
    inp = open(os.path.join(lrt_dir,'disort_run_' + str(n_cpu) + '.inp'),'w+')
    inp.write('# Input file for DISORT simulation for PV calibration\n\n')
    inp.write('# Basis configuration (same for all points)\n\n')
    #Write all configuration steps to file (independent of time stamp)
    for key in base_config:
        if base_config[key]:
            if key == 'running_mode' or key == "aerosol": 
                inp.write(base_config[key] + '\n')
            elif key == 'wavelength':    
                if base_config[key][sens_type] != "all":
                    inp.write(key + ' ' + str(base_config[key][sens_type][0]) + 
                              ' ' + str(base_config[key][sens_type][1]) + '\n')                
            elif type(base_config[key]) == bool:
                if base_config[key]:
                    inp.write(key + '\n')
            else:
                inp.write(key + ' ' + base_config[key] + '\n')      
                
    inp.write('umu %s \n' % grid["umustring"])
    inp.write('phi %s \n\n' % grid["phistring"])
    if altitude == 0:
        inp.write('zout 0.0015\n')
    else:
        inp.write('altitude %g\n' % altitude)
              
    #Write time-varying part (atmosphere and albedo, day of year, sun position)
    inp.write('# Time-varying part\n\n')
    inp.write('atmosphere_file ' + atm_path_temp + ' \n')                
        
    if rt_config["aerosol"]["source"] != 'default' and not np.isnan(data[('AOD_500', 'Aeronet')].values):
        inp.write('aerosol_angstrom %g %g \n' % (data[('alpha','Aeronet')],data[('beta','Aeronet')]))
    
    #SSA scale removed for the moment
    #                if asl_config["source"] != 'default' and not np.isnan(dataframe.loc[itime].ssa_vis):
    #                    inp.write('aerosol_modify ssa scale %g \n' % dataframe.loc[itime].ssa_vis)
    
    #Constant albedo from dataframe    
    if rt_config['albedo']['choice'] == 'constant':
        inp.write('albedo %g \n' % data[('albedo','constant')])
    #Spectral Albedo from MODIS data
    elif rt_config['albedo']['choice'] == 'MODIS':
        ix = albedo[0].index(str(itime.date()))
        modis_file = albedo[1][ix]
        if not modis_file:
            inp.write('albedo %g \n' % data[('albedo','constant')])
        else:
            brdf_path = os.path.join(modis_path,modis_file)
            inp.write('albedo_file %s \n' % brdf_path)
        
    #Don't need time and lat lon, day of year and sun position is enough
    inp.write('day_of_year %s \n' % itime.strftime('%j'))
    inp.write('phi0 %g \n' % data[('phi0','sun')])
    inp.write('sza %g \n' % data[('sza','sun')])                
    inp.close()
    
    inputfile = inp.name
    outputfile = os.path.join(lrt_dir,'disort_' + str(n_cpu) + '.out')
    
    return inputfile, outputfile
    
def plot_results(pv_station,key,days,folder,styles,home):
    """
    Plot libradtran simulation results
    
    args:    
    :param pv_station: dictionary with one PV system
    :param key: name of PV station currently under consideration
    :param days: list of clear sky days that were simulated    
    :param folder: dictionary with paths to save plots
    :param styles: dictionary with plot styles
    :param home: string with homepath
    """
    
    plt.ioff()
    plt.style.use(styles["single_small"])           
    
    station_folders = list_dirs(folder)
    
    save_folder = os.path.join(folder,key)
    if key not in station_folders:        
        os.mkdir(save_folder)
        
    for iday in days:        
        print("Plotting simulation vs. measurement for %s on %s" %(key,iday))
        #Slice all values coming from libradtran
        df_simulation_day = pv_station['df'].xs('libradtran',level='substat',axis=1).loc[iday]
        
        fig,ax = plt.subplots(figsize=(9,8))            
        ax.set_prop_cycle(color=['g','k','r','b','m'])
        
        ax.plot(df_simulation_day.index,df_simulation_day.Etotdown)
        max_df = np.max(df_simulation_day.Etotdown)
        max_rad = int(np.ceil(max_df/100)*100)
        
        #df_simulation_day.Edirdown.plot(color='k',style = '-',legend=True)
        #df_simulation_day.Ediffdown.plot(color='b',style = '-',legend=True)
        
        legend = [r'$G_{\rm tot,sim}^{\downarrow}$']
        
        for name, df in pv_station['irrad'].items():
            df_day = df.loc[iday]
            if 'Pyr' in name:
                df_day.Etotdown_pyr_Wm2.plot(ax=ax,legend=True)                
                legend.append(r'$G_{\rm tot,' + name.split('_')[1] + '}^{\downarrow}$')
        
            if 'suntracker' in name:
                df_day.Etotdown_CMP11_Wm2.plot(ax=ax,legend=True)                
                legend.append(r'$G_{\rm tot,CMP11}^{\downarrow}$')
        
        plt.legend(legend)                
        
        # Make the y-axis label, ticks and tick labels match the line color.
        plt.ylabel('Irradiance ($W/m^2$)')#, color='b')
        plt.xlabel('Time (UTC)')
        plt.title('Irradiance simulation and measurements for ' + key  + ' on ' + iday)
        #plt.axis('square')
        #plt.ylim([0,1000])
        datemin = np.datetime64(iday + ' 04:00:00')
        datemax = np.datetime64(iday + ' 18:00:00')  
        ax.set_xlim([datemin, datemax])
        ax.set_ylim([0,max_rad])
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        fig.autofmt_xdate(rotation=0,ha='center') 
                
        fig.tight_layout()
        plt.savefig(os.path.join(save_folder,'disort_simulation_' + key + '_' + iday + '.png'))        
        plt.close(fig)
        
    return

def disort_simulation(pv_systems,grid,rt_config,pv_config,sens_type,description,
                      config,home,plot_styles):
    """
    Run disort simulation for the pv systems and clear sky days, save results to file
    
    args:    
    :param pv_systems: dictionary of PV systems    
    :param grid: dictionary of grid for libradtran    
    :param rt_config: dictionary of config details for simulation        
    :param pv_config: dictionary of config details of stations
    :param sens_type: string, PV or pyranometer
    :param description: string describing simulation        
    :param config: dictionary, configuration of PVCAL run
    :param home: string, home path
    :param plot_styles: dictionary with plot style file names
    
    """        
    lrt_dir = os.path.join(home,rt_config["working_directory"])
    
    cols_rad = ['Edirdown','Ediffdown','Etotdown','Ediffdown_calc']
    
    for key in pv_systems:   
        print('Days with data are %s' % pv_systems[key]['sim_days'])
        print('Simulating radiative transfer for %s using wavelength range %s' 
              % (key,rt_config['common_base']['wavelength'][sens_type]))
        if rt_config["disort_base"]["pseudospherical"]:
            print('Results are with pseudospherical atmosphere')
        else:
            print('Results are with plane-parallel atmosphere')
        print('Molecular absorption is calculated using %s' % rt_config["common_base"]["mol_abs_param"])
        #Create dataframe columns for irradiance values
        multi_ind = pd.MultiIndex.from_product([cols_rad,['libradtran']],names=['variable','substat'])
        df_lrt = pd.DataFrame(dtype=np.float64,columns=multi_ind,index=pv_systems[key]['df'].index)
        pv_systems[key]['df'] = pd.merge(pv_systems[key]['df'],df_lrt,left_index=True,right_index=True)
                
        #Copy dataframe from dictionary
        dataframe = pv_systems[key]['df']#.iloc[0:1,:]
        if rt_config['atmosphere'] == 'cosmo':
            cosmo_path = pv_systems[key]['path_cosmo_lrt']
        else:
            cosmo_path = ''
        lat_lon = pv_systems[key]['lat_lon']
        modis_path = os.path.join(home,rt_config['albedo']['brdf_folder']) #,key)  #Turned off for now - same MODIS file for all!      
        if rt_config['albedo']['choice'] == 'MODIS':
            albedo_series = pv_systems[key]['albedo_series']
        else:
            albedo_series = ()
            
        if "altitude" in pv_config["pv_stations"][key]:
            altitude = pv_config["pv_stations"][key]["altitude"]/1000.
        else:
            altitude = 0.
        
        #Create list of dataframes for radiance distribution
        Idiff = [pd.DataFrame(dtype=np.float64)]*len(dataframe)        
        
        #List of simulation calculation times
        total_time = []
        
        #Get number of processors
        num_cores = mp.cpu_count()
        
        #List of CPUs
        n_time_vec = np.arange(0,num_cores)
        
        #List of disort files, tuple (input, output)
        disort_files = []
        
        #List of processes
        processes = []
        
        while True:                                            
            #Check whether we have gone too far
            if np.max(n_time_vec) >= len(dataframe):
                n_time_vec = n_time_vec[n_time_vec < len(dataframe)]
                if len(n_time_vec) == 0:
                    break
                else:
                    continue                  
            
            #Start timer
            start = time.time()
            
            #Get the timestamps corresponding to the points to be simulated
            itime_vec = dataframe.index[n_time_vec]
            
            #Prepare the input and output files, depending on number of cores
            for n in range(len(n_time_vec)):
                itime = itime_vec[n]            
            
                #Generate input files to match number of cores                
                disort_input, disort_output = generate_disort_files(rt_config,sens_type,altitude,home,cosmo_path,modis_path,itime,
                                                                    grid,dataframe.loc[[itime]],albedo_series,n)
                        
                disort_files.append((disort_input,disort_output))                        
            
            #Start subprocesses with libradtran simulation
            for (input_file,output_file) in disort_files:
                logfile = tempfile.TemporaryFile()
                p = subprocess.Popen(['uvspec < ' + input_file + ' > ' + 
                                      output_file,input_file,output_file],
                                    stdout=logfile,shell=True,cwd=lrt_dir)
                processes.append((p, output_file, logfile))
                
            #Collect results
            for n, (p, output, log) in enumerate(processes):
                p.wait()
                log.close()
                #Read output files for the specified time step and save the results
                itime = itime_vec[n]
                dataframe, Idiff[n_time_vec[n]] = read_disort_output(key,itime,
                                dataframe,rt_config,description,grid,
                                output,lat_lon,sens_type,home)
                #print output
                print('%d: %s, %s: SZA %g, phi0 %g, Edirdown %g, Ediffdown %g, Ediffdown_calc %g' 
                      % (n_time_vec[n],itime,key,
                         dataframe.loc[itime,'sza'],
                         dataframe.loc[itime,'phi0'],
                         dataframe.loc[itime,('Edirdown','libradtran')],
                         dataframe.loc[itime,('Ediffdown','libradtran')],
                         dataframe.loc[itime,('Ediffdown_calc','libradtran')]))
            
            #stop timer and calculate runtime and total time
            end = time.time()
            runtime = end - start   
            total_time.append(runtime)
            
            print('Simulation of %d time steps took %g seconds' %(num_cores,runtime))
            
            #Clear lists of files
            del processes[:]
            del disort_files[:]
            
            #Move to the next group of times
            n_time_vec = n_time_vec + num_cores      
        
        #Assign radiance field to the main dataframe as a series of dataframes
        dataframe[('Idiff','libradtran')] = pd.Series(Idiff,index=dataframe.index)

        #Generate folders with results
        mainpath = os.path.join(home,rt_config["save_path"]["disort"],
                                rt_config["save_path"]["clear_sky"])
        save_folder, res_string_label = generate_folders(rt_config,sens_type,mainpath)
        
        if rt_config['plot_flag'] and rt_config['common_base']['wavelength'][sens_type] == 'all':
            #Plot simulation results to compare with measurement
            plot_results(pv_systems[key],key,pv_systems[key]['sim_days'],
                         save_folder,plot_styles,homepath)        
        
        datatypes = ["pv","irrad","temp","wind"]
        #Remove separate dictionaries of long dataframes before saving results
        for idata in datatypes:   
            if idata in pv_systems[key]:             
                del pv_systems[key][idata]
        
        #Save all results for one station to a single binary file               
        save_radsim_all_binary(pv_systems[key],key,description,rt_config,total_time,
                               save_folder,res_string_label,sens_type)
        
        print('Simulation for %s took %g secs ' % (key,sum(total_time)))        
    
    return
                                              

#%%Main program                 
#######################################################################
###                         MAIN PROGRAM                            ###
#######################################################################
#def main():
import argparse
   
parser = argparse.ArgumentParser()
parser.add_argument("-f","--configfile", help="yaml file containing config")
parser.add_argument("-s","--station",nargs='+',help="station for which to perform simulation")
parser.add_argument("-c","--campaign",nargs='+',help="measurement campaign for which to perform simulation")
args = parser.parse_args()
    
#Main configuration file
if args.configfile:
    config_filename = os.path.abspath(args.configfile) #"config_PYRCAL_2018_messkampagne.yaml" #
else:
    config_filename = "config_PVCAL_MetPVNet_messkampagne.yaml"  #

#Read in values from configuration file
config = load_yaml_configfile(config_filename)

if "PVCAL" in config_filename:
    sensor_type = "pv"
    pv_config = load_yaml_configfile(config["pv_configfile"])
elif "PYRCAL" in config_filename:
    sensor_type = "pyranometer"

#Load radiative transfer configuration
rt_config = load_yaml_configfile(config["rt_configfile"])

#These are plotting styles
plot_styles = config["plot_styles"]

#Get home directory
homepath = os.path.expanduser('~')

if args.campaign:
    campaigns = args.campaign
else:
    campaigns = rt_config["simulation_source"]

print('Simulation will be run for %s' % campaigns)
#Load data configuration
for measurement in campaigns:
    year = "mk_" + measurement.split('_')[1]
    data_config = load_yaml_configfile(config["data_configfile"][year])

    #Which days to consider
    test_days = create_daterange(rt_config["test_days"][year]["all"])

    if args.station:
        data_config["stations"] = args.station
    else:
        data_config["stations"] = "PV_17" #rt_config["stations"]
    
    print('%s calibration: running DISORT simulation for %s and %s stations\n\
          Simulation days are %s' % (sensor_type,measurement,data_config["stations"],test_days))
        
    #Load data from measurements
    print('Loading resampled data for %s resolution, this could take a while' % rt_config["timeres"])
    pvsys, select_system_info = dpf.load_resampled_data(rt_config["timeres"],data_config,homepath)
    
    #Select days for simulation, check if there is PV data!
    pvsys, select_system_info = select_simulation_days(select_system_info,pvsys,
                                                       test_days,sensor_type)
    
    if pvsys:
        #Resolution for diffuse radiance simulation
        disort_res = rt_config["disort_rad_res"]
        grid_dict = define_disort_grid(disort_res)
        
        #Calculate sun position
        pvsys = sun_position(pvsys,rt_config['sza_max']['clear_sky'])
            
        #Prepare atmosphere and aerosol inputs
        pvsys, aeronet_stats = prepare_atmosphere_aerosol_albedo(rt_config,pvsys,
                                select_system_info,homepath,measurement)
              
        #Run DISORT simulation        
        print('Starting DISORT simulation with %s, this could take a while....' % disort_res)
        disort_simulation(pvsys,grid_dict,rt_config,pv_config,sensor_type,
                                  measurement,config,homepath,plot_styles)
     
#if __name__ == "__main__":
#    main()
