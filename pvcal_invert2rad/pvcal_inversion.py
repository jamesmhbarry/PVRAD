#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 15:02:41 2018

@author: james

Inversion using results of DISORT simulation for irradiance input and power data as 
measurement vector
"""

import os
import numpy as np
from numpy import deg2rad, rad2deg, nan
import pandas as pd
import pickle
import collections
from file_handling_functions import *
from rt_functions import *
from pvcal_forward_model import P_mod_simple_cal, azi_shift, I_mod_simple_diode_cal
from vorwaertsmodell_reg_new_ohne_Matrizen_schalter_JB import F_temp_model_dynamic, F_temp_model_static
import inversion_functions as inv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import subprocess
import data_process_functions as dpf
import scipy.constants as const


###############################################################
###   general functions to load and process data    ###
###############################################################

def generate_folder_names_disort(mainpath,config,sens_type):
    """
    Generate folder structure to retrieve DISORT simulation results
    
    args:
    :param mainpath: string with parent folder
    :param config: dictionary with RT configuration
    :param sens_type: string specifying type of sensor 
    
    out:
    :return folder_label: string with complete folder path
    :return filename: string with name of file (prefix)
    :return theta_res, phi_res: tuple of string with DISORT grid resolution

    """
    
    #geometry model
    atm_geom_config = config["disort_base"]["pseudospherical"]
    if atm_geom_config == True:
        atm_geom_folder = "Pseudospherical"
    else:
        atm_geom_folder = "Plane-parallel"
    
    #Get wavelength folder label
    wvl_config = config["common_base"]["wavelength"][sens_type]
    if type(wvl_config) == list:
        wvl_folder_label = "Wvl_" + str(int(wvl_config[0])) + "_" + str(int(wvl_config[1]))
    elif wvl_config == "all":
        wvl_folder_label = "Wvl_Full"
    
    #Get DISORT resolution folder label
    disort_config = config["disort_rad_res"]   
    theta_res = str(disort_config["theta"]).replace('.','-')
    phi_res = str(disort_config["phi"]).replace('.','-')
    
    disort_folder_label = 'Zen_' + theta_res + '_Azi_' + phi_res
    filename = 'lrt_sim_results_'
    
    if config["atmosphere"] == "default":
        atm_folder_label = "Atmos_Default"    
    elif config["atmosphere"] == "cosmo":
        atm_folder_label = "Atmos_COSMO"
        filename = filename + 'atm_'
        
    if config["aerosol"]["source"] == "default":
        aero_folder_label = "Aerosol_Default"
    elif config["aerosol"]["source"] == "aeronet":
        aero_folder_label = "Aeronet_" + config["aerosol"]["station"]
        filename = filename + 'asl_' + config["aerosol"]["data_res"] + '_'

    folder_label = os.path.join(mainpath,atm_geom_folder,wvl_folder_label,disort_folder_label,
                                atm_folder_label,aero_folder_label)
    
    return folder_label, filename, (theta_res,phi_res)
    

def load_data_radsim_results(config,inv_config,rt_config,station_list,home):
    """
    Load results from measurement and from DISORT radiation simulation
    
    args:
    :param config: dictionary with main configuration
    :param inv_config: dictionary with inversion config
    :param rt_config: dictionary with current RT configuration
    :param station_list: list of stations for which simulation was run
    :param home: string with homepath
    
    out:
    :return pv_systems: dictionary of PV systems with data
    """

    #Get correct path for DISORT simulation            
    mainpath_disort = os.path.join(home,rt_config['save_path']['disort'],
                                   rt_config['save_path']['clear_sky'])        
    folder_label_pv, filename_pv, (theta_res,phi_res) = \
    generate_folder_names_disort(mainpath_disort,rt_config,"pv")
    
    folder_label_pyr, filename_pyr, (theta_res,phi_res) = \
    generate_folder_names_disort(mainpath_disort,rt_config,"pyranometer")
        
    #Choose which stations to load
    if type(station_list) != list:
        station_list = [station_list]
        if station_list[0] == "all":
            station_list = inv_config["pv_stations"]
            #select_system_info = system_info                
    
    #Define empty dictionary of PV systems
    pv_systems = {}
    calibration_source = inv_config["calibration_source"]
    
    #Import data for each station, from each campaign
    for station in station_list:        
        for measurement in calibration_source:
            year = "mk_" + measurement.split('_')[1]
            #if inv_config["pv_stations"][station]["calibration_days"][year]:
            new_df_data = 'df_' + year.split('_')[-1]    
            new_df_sim_pv = 'df_sim_pv_' + year.split('_')[-1]    
            #Load data configuration (different for each campaign)
            data_config = load_yaml_configfile(config["data_configfile"][year])
            mainpath_data = os.path.join(home,data_config["paths"]["savedata"]
            ["main"],data_config["paths"]["savedata"]["binary"])
            #Define filename
            filename_data = measurement + '_' + station + '_'\
            + rt_config["timeres"] + '.data'
            
            #Load data from measurement campaign "measurement"
            data_types = data_config["data_types"]
            pvstat, info = dpf.load_station_data(mainpath_data,filename_data,data_types,False)
            
            if pvstat:
                #Rename dataframe to reflect which campaign
                pvstat[new_df_data] = pvstat['df']
                del pvstat['df']                        
            
                print('Data for %s, %s loaded from %s' %(station,year,filename_data))
            
            #Load results from DISORT simulation
            filename_sim = filename_pv + measurement + '_disortres_' + theta_res\
            + '_' + phi_res + '_' + station + '.data'

            try:
                with open(os.path.join(folder_label_pv,filename_sim), 'rb') as filehandle:  
                    # read the data as binary data stream
                    (pvstat_sim, temp) = pd.read_pickle(filehandle)    
                
                #If the data file is available, extract only RT sim from pvstat_sim
                if pvstat:
                    #Get only columns from libradtran, sun position, aerosol, albedo
                    pvstat_sim[new_df_sim_pv] = pd.concat([pvstat_sim['df'].loc[:,pd.IndexSlice[:,['sun','Aeronet']]],
                    pvstat_sim['df'].loc[:,pd.IndexSlice['albedo',:]],pvstat_sim['df'].loc[:,pd.IndexSlice[:,'libradtran']]],axis=1)
                    del pvstat_sim['df']
                        
                    #Merge dictionaries
                    pvstat = merge_two_dicts(pvstat,pvstat_sim)
                    del pvstat_sim
                    
                    #Merge dataframes
                    pvstat[new_df_sim_pv] = pd.merge(pvstat[new_df_data],pvstat[new_df_sim_pv],
                          left_index=True,right_index=True)
                    del pvstat[new_df_data]
                else:
                    #If datafile was not available, use all info from RT sim (data included)
                    #This should normally be the case
                    pvstat = pvstat_sim
                    pvstat[new_df_sim_pv] = pvstat['df']
                    del pvstat['df']      
                    
                #Rename it for PV simulation
                pvstat[new_df_sim_pv].columns = pd.MultiIndex.from_tuples([(var,'libradtran_pv') 
                       if substat == 'libradtran' else (var,substat) for (var,substat) in 
                       pvstat[new_df_sim_pv].columns],names=pvstat[new_df_sim_pv].columns.names)
                                

                print('Data for %s from %s loaded from %s' % (station,year,filename_sim))
            except IOError:
                print('There is no PV simulation for %s, %s' % (station,year))
                
             #Load results from DISORT simulation for pyranometers
            filename_sim = filename_pyr + measurement + '_disortres_' + theta_res\
            + '_' + phi_res + '_' + station + '.data'
            new_df_sim_pyr = 'df_sim_pyr_' + year.split('_')[-1]    
            try:
                with open(os.path.join(folder_label_pyr,filename_sim), 'rb') as filehandle:  
                    # read the data as binary data stream
                    (pvstat_sim, temp) = pd.read_pickle(filehandle)   
               
                pvstat_sim[new_df_sim_pyr] = pd.concat([pvstat_sim['df'].loc[:,pd.IndexSlice[:,['sun','Aeronet','libradtran']]],
                pvstat_sim['df'].loc[:,pd.IndexSlice['albedo',:]]],axis=1)
                del pvstat_sim['df']
                
                #Rename it for pyranometer simulation (broadband)
                pvstat_sim[new_df_sim_pyr].columns = pd.MultiIndex.from_tuples([(var,'libradtran_pyr') 
                       if substat == 'libradtran' else (var,substat) for (var,substat) in 
                       pvstat_sim[new_df_sim_pyr].columns],names=pvstat_sim[new_df_sim_pyr].columns.names)
                
                #Merge dictionaries
                pvstat = merge_two_dicts(pvstat,pvstat_sim)
                del pvstat_sim
                
                new_df_sim = f"df_sim_{year.split('_')[-1]}"
                if new_df_sim_pv in pvstat and new_df_sim_pyr in pvstat:
                    pvstat[new_df_sim] = pd.concat([pvstat[new_df_sim_pv],pvstat[new_df_sim_pyr]],axis=1)
                    del pvstat[new_df_sim_pv], pvstat[new_df_sim_pyr]
                elif new_df_sim_pv in pvstat:
                    pvstat[new_df_sim] = pvstat[new_df_sim_pv]
                
                if new_df_sim in pvstat:
                    pvstat[new_df_sim] = pvstat[new_df_sim].loc[:,~pvstat[new_df_sim].columns.duplicated()] 
                
                
                #Update dictionary with current station
                if station not in pv_systems:
                    pv_systems.update({station:pvstat})
                    sim_days = pv_systems[station]["sim_days"]
                    del pv_systems[station]["sim_days"]
                    pv_systems[station]["sim_days"] = {}
                    pv_systems[station]["sim_days"].update({year:sim_days})
                else:                    
                    pv_systems[station].update({new_df_sim:pvstat[new_df_sim]})                    
                    pv_systems[station]["sim_days"].update({year:pvstat["sim_days"]})
                
                print('Data for %s from %s loaded from %s' % (station,year,filename_sim))    
            except IOError:
                print('There is no pyranometer simulation for %s, %s' % (station,year))
        
        #Check if there is data for the required calibration source
        pv_systems[station].update({"cal_source":[]})
        for measurement in calibration_source:
            dfname = 'df_sim_' + measurement.split('_')[1]
            if dfname in pv_systems[station]:    
                pv_systems[station]["cal_source"].append(measurement)
                print(f"Using {measurement} for calibration")
    
    return pv_systems

def load_resampled_data(station,timeres,measurement,config,home):
    """
    Load data that has already been resampled to a specified time resolution
    
    args:    
    :param station: string, name of PV station to load
    :param timeres: string, timeresolution of the data
    :param measurement: string, year of measurements
    :param config: dictionary with paths for loading data
    :param home: string, homepath    
    
    out:
    :return pv_stat: dictionary of information and data from PV station

    """

    savedir = os.path.join(home,config["paths"]["savedata"]["main"])    
    
    binarypath = os.path.join(savedir,config["paths"]["savedata"]["binary"])
    files = list_files(binarypath)    
    
    filename = measurement + '_' + station + "_" + timeres + ".data"
    if filename in files:        
        with open(os.path.join(binarypath,filename), 'rb') as filehandle:  
            (pvstat, info) = pd.read_pickle(filehandle)          
                
        print('Data for %s loaded from %s' % (station,filename))
        return pvstat
    else:
        print('Required file not found')
        return None
    
def load_lw_data(station,timeres,description,path):
    """
    
    Load the LW downward welling irradiance measured at MS01
    

    Parameters
    ----------
    station : string, name of station
    timeres : string, time resolution of data
    description : string, description of current campaign
    path : string, path to load data from

    Returns
    -------
    dataframe / series with longwave data

    """
    
    filename = description + '_' + station[0] + '_' + timeres + ".data"
    
    files = list_files(path)    
        
    if filename in files:        
        with open(os.path.join(path,filename), 'rb') as filehandle:  
            (lw_pvstat, info) = pd.read_pickle(filehandle)  
        
        print('LW data from ' + station[0] + ' loaded from %s' % filename)
        return lw_pvstat["df"][[(station[2],station[1])]]
    else:
        print('Required file not found')
        return None   

def load_spectral_mismatch_fit(config,home):
    """
    Load dataframe for spectral mismatch fit

    Parameters
    ----------
    config : dictionary with configuration for inversion
    home : string with home path

    Returns
    -------
    dataframe with spectral mismatch fit

    """
    
    folder = os.path.join(home,config["spectral_mismatch_lut"]["clear_sky"])
    
    file = list_files(folder)[0]
    
    df = pd.read_csv(os.path.join(folder,file),sep=',',comment='#')
    
    return df

###############################################################
###   Preparation for inversion                             ###
###############################################################
def find_nearest_cosmo_grid_folder(configfile,key,pv_station,datatype,home):
    """
    Search through the output of cosmomystic or cosmopvcal to find the 
    gridpoint (folder) corresponding to the location of each PV station
    
    args:
    :param configfile: string, configfile for cosmomystic
    :param key: string, name of station
    :param pv_station: dictionary with all info and data of pv station
    :param datatype: string, either surf or atmo
    :param home: string, home path
    
    out:
    :return pv_station: dictionary with updated information on COSMO files
    """
    
    config = load_yaml_configfile(configfile)
    
    if datatype == "atmo":
        path = os.path.join(home,config["path_atmofiles"])
    elif datatype == "surf":
        path = os.path.join(home,config["path_surface_files"])
    elif datatype == "irrad":
        path = os.path.join(home,config["path_irradiance_files"])
    
    cosmo_folders = list_dirs(path)
    
    #Define paths for COSMO-modified atmosphere files
    
    for folder in cosmo_folders:
        fname = "known_stations.dat"
        ds = pd.read_csv(os.path.join(path,folder,fname),comment='#',names=['name','lat','lon'],sep=' ',
                         index_col=0)
        for station in ds.index:
            if station == key:
                if datatype == "atmo":
                    pv_station['path_cosmo_lrt'] = os.path.join(path,folder)
                elif datatype == "surf":
                    pv_station['path_cosmo_surface'] = os.path.join(path,folder)
                elif datatype == "irrad":
                    pv_station['path_cosmo_irrad'] = os.path.join(path,folder)

    return pv_station
 
def import_cosmo_surf_data(key,pv_station,days,year):
    """
    Import surface data from cosmo2pvcal
    
    args:
    :param key: string, name of station
    :param pv_station: dictionary with all info and data of pv station
    :param days: list of days to consider
    :param year: string describing the data source
    
    out:
    :return pv_station: dictionary with updated COSMO surface data
    """
    df_name = 'df_sim_' + year.split('_')[-1]
        
    #Extract data from COSMO files
    dataframe = pd.DataFrame()
    dfs = [pd.read_csv(os.path.join(pv_station['path_cosmo_surface'],\
            iday.replace('-','') + '_surface_props.dat'),sep='\s+',index_col=0,skiprows=2,
            header=None,names=['v_wind_10M','dir_wind_10M','T_ambient_2M_C']) for iday in days]
     
    dataframe = pd.concat(dfs,axis=0)
    dataframe.index = pd.to_datetime(dataframe.index,format='%d.%m.%Y;%H:%M:%S')    
        
    dataframe.T_ambient_2M_C = dataframe.T_ambient_2M_C - 273.15   
    
    dfs = [pd.read_csv(os.path.join(pv_station['path_cosmo_irrad'],\
            iday.replace('-','') + '_irradiance.dat'),sep='\s+',index_col=0,skiprows=2,
            header=None,names=['Edirdown_Wm2','Edirdown_mean_Wm2','Edirdown_iqr_Wm2','Ediffdown_Wm2',
                               'Ediffdown_mean_Wm2','Ediffdown_iqr_Wm2']) for iday in days]
     
    dataframe2 = pd.concat(dfs,axis=0)
    dataframe2.index = pd.to_datetime(dataframe2.index,format='%d.%m.%Y;%H:%M:%S')    
    
    rs_dataframe2 = dataframe2.resample('15T').interpolate('linear')
    
    dataframe = pd.concat([dataframe,rs_dataframe2],axis=1)
    
    #Create Multi-Index for cosmo data
    dataframe.columns = pd.MultiIndex.from_product([dataframe.columns.values.tolist(),['cosmo']],
                                                                   names=['substat','variable'])       
    
    #If the simulation etc has timezone info then we should keep it!
    #Need this for backward compatibility               
    if pv_station[df_name].index.tzinfo:
        dataframe.index = dataframe.index.tz_localize(tz='UTC',ambiguous='NaT')
    
    #Assign to special cosmo dataframe, and join with main dataframe
    pv_station['df_cosmo_' + year.split('_')[-1]] = dataframe
    
    pv_station[df_name] = pd.concat([pv_station[df_name],dataframe],axis=1,join='inner')
       
    return pv_station

def prepare_surface_data(inv_config,key,pv_station,days,home):
    """
    args:
    :param inv_config: dictionary of inversion configuration
    :param key: string, name of station
    :param pv_station: dictionary with all info and data of pv station
    :param days: dictionary of list of days to consider
    :param home: string, home path
    
    out:
    :return pv_station: dictionary updated with COSMO surface and irradiance data
    """
    
    for measurement in pv_station["cal_source"]:
        year = "mk_" + measurement.split('_')[1]
        configfile = os.path.join(home,inv_config["cosmopvcal_configfile"][year])
        cosmo_config = load_yaml_configfile(configfile)
        finp = open(configfile,'a')
        if "stations_lat_lon" not in cosmo_config:
            finp.write('# latitude, longitude of PV stations within the COSMO grid\n')
            finp.write('stations_lat_lon:\n')
        
            #Write lat lon into config file                
            finp.write('    %s: [%.5f,%.5f]\n' % (key,pv_station['lat_lon'][0],
                           pv_station['lat_lon'][1]))
        else:
            if not cosmo_config["stations_lat_lon"]:                
                finp.write('    %s: [%.5f,%.5f]\n' % (key,pv_station['lat_lon'][0],
                           pv_station['lat_lon'][1]))
            elif key not in cosmo_config["stations_lat_lon"]:
                finp.write('    %s: [%.5f,%.5f]\n' % (key,pv_station['lat_lon'][0],
                           pv_station['lat_lon'][1]))
        
        finp.close()    
        
        #Which days to consider
        test_days = create_daterange(days[year]["all"])
        
        #Prepare surface data from COSMO
        if inv_config["cosmo_sim"]:
            # call cosmo2pvcal 
            print('Running cosmo2pvcal to extract surface properties')
            child = subprocess.Popen('cosmo2pvcal ' + configfile, shell=True)
            child.wait()
        else:
            print(f'cosmo2pvcal already run, read in surface files for {year}')

        pv_station = find_nearest_cosmo_grid_folder(configfile,key,pv_station,'surf',home)   
        pv_station = find_nearest_cosmo_grid_folder(configfile,key,pv_station,'irrad',home)   
        pv_station = import_cosmo_surf_data(key,pv_station,test_days,year)
    
    return pv_station

def import_cosmo_water_vapour_data(key,pv_station,days,year):
    """
    Import COSMO water vapour data
    
    args:
    :param key: string, name of station
    :param pv_station: dictionary with all info and data of pv station
    :param days: list of days to consider
    :param year: string describing the data source
    
    out:
    :return pv_station: dictionary with updated information on COSMO water vapour
    """
    df_sim_name = 'df_sim_' + year.split('_')[-1]
    df_cosmo_name = 'df_cosmo_' + year.split('_')[-1]
        
    #Extract data from COSMO files, calculate precipitable water
    dfs = []
    print(f'Calculating precipitable water from COSMO atmosphere for {year}')
    for iday in days:
        filepath = os.path.join(pv_station['path_cosmo_lrt'],\
             iday.replace('-',''))
        dfs_lrt_atm = [(file,read_lrt_atmosphere(os.path.join(filepath,file),skiprows=3)) \
                       for file in list_files(filepath)]
            
        h2o = [calc_precipitable_water(df[1]["h2o(cm-3)"].values[::-1], 
                                  df[1]["z(km)"].values[::-1]) for df in dfs_lrt_atm]
        df_index =  pd.DatetimeIndex([pd.to_datetime(f"{iday} {df[0][0:2]}:{df[0][2:4]}:{df[0][4:6]}")\
                     for df in dfs_lrt_atm])
        
        dfs.append(pd.DataFrame(h2o,index=df_index))
    
    dataframe = pd.concat(dfs,axis=0)
    
    #Create Multi-Index for cosmo data
    dataframe.columns = pd.MultiIndex.from_product([['n_h2o_mm'],['cosmo']],
                                                                   names=['substat','variable'])       
    
    #If the simulation etc has timezone info then we should keep it!
    #Need this for backward compatibility               
    if pv_station[df_sim_name].index.tzinfo:
        dataframe.index = dataframe.index.tz_localize(tz='UTC',ambiguous='NaT')        
    
    for df_name in [df_sim_name,df_cosmo_name]:
        pv_station[df_name] = pd.concat([pv_station[df_name],dataframe],axis=1,join='inner')
       
    return pv_station

def prepare_water_vapour(config,key,pv_station,days,home):
    """
    Set up atmosphere (call COSMO2MYSTIC), and extract precipitable water
    
    args:
    :param config: dictionary of configuration details from config file
    :param key: string, name of PV station
    :param pv_station: dictionary information and data from one PV station  
    :param days: list of days
    :param home: string with homepath    
    
    out:
    :return pv_station: dictionary with updated information on COSMO water vapour
    """
    
    for measurement in pv_station["cal_source"]:
        year = "mk_" + measurement.split('_')[1]
    
        #Set up atmosphere
        atm_source = config["atmosphere"]
        
        #If not default, call cosmo2mystic to create COSMO-modified atmosphere file
        if atm_source == 'cosmo':
            cosmo_configfile = os.path.join(home,config["cosmo_configfile"][year])
            cosmo_config = load_yaml_configfile(cosmo_configfile)
            finp = open(cosmo_configfile,'a')
            if "stations_lat_lon" not in cosmo_config:
                finp.write('# latitude, longitude of PV stations within the COSMO grid\n')
                finp.write('stations_lat_lon:\n')
            
                #Write lat lon into config file                
                finp.write('    %s: [%.5f,%.5f]\n' % (key,pv_station['lat_lon'][0],
                               pv_station['lat_lon'][1]))
            else:
                if not cosmo_config["stations_lat_lon"]:                
                    finp.write('    %s: [%.5f,%.5f]\n' % (key,pv_station['lat_lon'][0],
                               pv_station['lat_lon'][1]))
                elif key not in cosmo_config["stations_lat_lon"]:
                    finp.write('    %s: [%.5f,%.5f]\n' % (key,pv_station['lat_lon'][0],
                               pv_station['lat_lon'][1]))
            
            finp.close()    
            
        #Which days to consider
        test_days = create_daterange(days[year]["all"])
        
        if config["cosmo_sim"]:
            # call cosmo2mystic
            child = subprocess.Popen('cosmo2mystic ' + cosmo_configfile, shell=True)
            child.wait()
            print('Running cosmo2mystic to create libRadtran atmosphere files')
        else:
            print('cosmo2mystic already run, read in atmosphere files')
        
        #if "path_cosmo_lrt" not in pv_station:
        pv_station = find_nearest_cosmo_grid_folder(cosmo_configfile,key,pv_station,'atmo',home)
        
        pv_station = import_cosmo_water_vapour_data(key,pv_station,test_days,year)
        
    #Import aerosol data from AERONET     

    return pv_station

def load_temp_model_results(station,pv_station,pvcal_config,home):
    """
    Load results from dynamic temperature temperature model
    
    args:
    :param station: string, name of PV station
    :param pv_station: dictionary with info and dataframes    
    :param pvcal_config: dictionary with configuration for calibration
    :param home: string, homepath
    
    out:
    :return pv_station: dictionary with temperature model results loaded
    """
    
    temppath = pvcal_config["results_path"]["temp_model"]    
    
    for substat in pvcal_config["pv_stations"][station]["substat"]:
        mount_type = pvcal_config["pv_stations"][station]["substat"][substat]["mount"]
        station_temp_model = pvcal_config["T_model_mount_type"][mount_type]
        loadpath = os.path.join(home,temppath,station_temp_model)
            
        if f"temp_model_{mount_type}" not in pv_station:   
            if len(station_temp_model.split('_')) != 2:
                station_temp_model = '_'.join(station_temp_model.split('_')[0:2])
            filename = "pvtemp_inversion_results_" + station_temp_model + ".data"
            
            try:
                with open(os.path.join(loadpath,filename), 'rb') as filehandle:  
                        # read the data as binary data stream
                    (pvstat, pvtemp_config) = pd.read_pickle(filehandle)
                            
                pv_station[f"temp_model_{mount_type}"] = pvstat["inversion"]["all_sky"] #["dynamic"]
                pv_station[f"pvtemp_config_{mount_type}"] = pvtemp_config                
                pv_station[f"pvtemp_config_{mount_type}"]["name"] = station_temp_model
                pv_station["t_res_lw"] = pvtemp_config["timeres"]
                                
            except IOError:
                print("No dynamic temperature model results for %s" % station)
                return pv_station, None
            
    return pv_station

def calculate_temp_module(station,pv_station,pv_config,timeres_substat,home):
    """
    Calculate module temperature with dynamic model, using atmospheric conditions
    
    args:
    :param station: string, name of PV station    
    :param pv_station: dictionary with all information and data from PV station    
    :param pv_config: dictionary with configuration for inversion    
    :param timeres_substat: timedelta with resolution of substation data    
    :param home: string, homepath
    
    out:
    :return pv_station: dictionary with updated module temperature
    
    """
    
        
    for measurement in pv_station["cal_source"]:
        year = "mk_" + measurement.split('_')[1]        
                
        source = pv_config["pv_stations"][key]["input_data"]
        T_source = source["temp_module"][year]
                                
        if "model" in T_source:                                       
            T_model = T_source.split('_')[0]
            #Check data - can only use static model with coarse data
            if "cosmo" in source["temp_amb"][year]:
                print("Cannot use dynamic model with COSMO data - use static model")
                T_model == "static"
                T_amb_name = "T_ambient_2M_C"                
            else:
                T_amb_name = 'T_ambient_pyr_C'
                                
            if "cosmo" in source["wind"][year]:
                print("Cannot use dynamic model with COSMO data - use static model")
                T_model == "static"
                v_wind_name = "v_wind_10M"
            else:                
                v_wind_name = 'v_wind_mast_ms'
                        
            #Put together data for temperature model
            print(f"Using results of {T_model} temperature model")
            
            dfs = []
            for substat in pv_config["pv_stations"][key]["substat"]:
                
                mount_type = pv_config["pv_stations"][key]["substat"][substat]["mount"]
                dataframe = pd.DataFrame()      
                
                temp_model = pv_station[f"temp_model_{mount_type}"][T_model]
                pvtemp_config= pv_station[f"pvtemp_config_{mount_type}"]
                
                #temp_station = pvtemp_config["name"]
                df_temp = pv_station[f"df_sim_{year.split('_')[1]}"]
                        
                if station == "PV_11" or (station == "PV_12" and "2019" in year):
                    dataframe["Gtotpoa"] = df_temp[('Etotpoa_RT1_Wm2',
                         source["irrad"][year])]
                # elif station == "MS_02":
                #     dataframe["Gtotpoa"] = df_temp[('Etotpoa_CMP11_Wm2',
                #          source["irrad"][year])]
                else:
                    dataframe["Gtotpoa"] = df_temp[('Etotpoa_pyr_Wm2',
                         source["irrad"][year])]            
                                
                dataframe["T_ambient"] = df_temp[(T_amb_name,source["temp_amb"][year])]
                
                dataframe["v_wind"] = df_temp[(v_wind_name,source["wind"][year])]
                
                dataframe["G_lw_down"] = pv_station[f"df_lw_{year.split('_')[1]}"]\
                .loc[:,(source["longwave"][year][2],source["longwave"][year][1])]
                
                dataframe.dropna(axis=0,inplace=True)
                
                #Get unique days
                days = pd.to_datetime(dataframe.index.date).unique().strftime('%Y-%m-%d')    
                    
                time_res_temp = dataframe.index.to_series().diff().min()
                
                #kontrolle_zeitaufloesung(dataframe,time_res_temp,days)
                
                number_days = len(days)
                
                time_res_temp_sec = time_res_temp/np.timedelta64(1, 's')
                day_length = np.zeros(number_days, dtype=int)
                for i, iday in enumerate(days):
                    day_length[i] = len(dataframe.loc[iday]) 
            
                n_past = pvtemp_config["inversion"]["inv_param"]["n_past"]
                schalter = pvtemp_config["inversion"]["inv_param"]["all_points"]        
                x_hat = temp_model["x_dach"]
                g = dataframe['Gtotpoa'].values
                t_amb = dataframe['T_ambient'].values
                v = dataframe['v_wind'].values
                g_lw = dataframe["G_lw_down"].values
                
                #Run Dirk's temperature model
                if T_model == "dynamic":
                    dataframe["T_module_C"] = F_temp_model_dynamic(x_hat, t_amb, g, v, g_lw,\
                            n_past, number_days, day_length, time_res_temp_sec, schalter)    
                elif T_model == "static":
                    dataframe["T_module_C"] = F_temp_model_static(x_hat, t_amb, g, v, g_lw)
                
                #Resample results to the correct resolution    
                time_res_substat = pd.to_timedelta(timeres_substat)    
                dfs_rs = [pd.DataFrame()]*len(days)
                for i, iday in enumerate(days):
                    df_day = dataframe.loc[iday]
                    if time_res_temp != time_res_substat:
                        try:
                            if time_res_temp < time_res_substat:
                                try: 
                                    df_day_rs = dpf.downsample(df_day,time_res_temp,time_res_substat)
                                except:
                                    print('error')#print('error in data from %s, %s on %s' % (station,substat,iday))
                            elif time_res_temp > time_res_substat:
                                df_day_rs = dpf.interpolate(df_day,time_res_substat)        
                        except:
                            print('error in resampling')
                    else:
                        df_day_rs = df_day
                                    
                    dfs_rs[i] = df_day_rs
                    
                df_rs = pd.concat(dfs_rs,axis=0)

                df_rs = df_rs[["T_module_C"]]
                
                #Extract just the modelled temperature and create Multiindex
                df_rs.columns = pd.MultiIndex.from_product([df_rs.columns.values.tolist(),[f"{substat}_{T_model}"]],
                                                                               names=['variable','substat'])                    
                
                dfs.append(df_rs)
                
            df_temp_result = pd.concat(dfs,axis=1)
                           
            pv_station[f"df_sim_{year.split('_')[1]}"] = pd.concat([pv_station[f"df_sim_{year.split('_')[1]}"],
                                                                            df_temp_result],axis=1)    
            pv_station[f"df_sim_{year.split('_')[1]}"].sort_index(axis=1,level=1,inplace=True)
                    
        else:
            print(f"Using measured module temperature data from {T_source} for {year}")      
            
    return pv_station

def calculate_sky_temperature(G_lw,emissivity):
    """
    Calculate sky temperature using measured longwave irradiance and emissivity

    Parameters
    ----------
    G_lw : vector of longwave irradiance 
    emissivity : float, emissivity of the atmosphere

    Returns
    -------
    vector of sky temperature in Celsius

    """
    
    return np.power(G_lw/const.sigma/emissivity,1./4) - 273.15
    
def prepare_longwave_data(key,pv_station,timeres_sim,pv_config,home):
    """
    

    Parameters
    ----------
    key : string, name of pv station
    pv_station : dictionary with information and data from PV station
    timeres_sim : string, time resolution of simulated data
    pv_config : dictionary with configuration for inversion
    home : string, home path

    Returns
    -------
    pv_station : dictionary with PV station data updated with sky temperature

    """
    
    for measurement in pv_station["cal_source"]:
        year = "mk_" + measurement.split('_')[1]        
                        
        data_config = load_yaml_configfile(config["data_configfile"][year])

        #Get configuration           
        lw_station = pv_config["pv_stations"][key]["input_data"]["longwave"][year]                
        loadpath_lw = os.path.join(home,data_config["paths"]["savedata"]["main"],
                                data_config["paths"]["savedata"]["binary"])                                                    

        #Load data for temperature model
        df_lw = load_lw_data(lw_station,pv_station["t_res_lw"],measurement,loadpath_lw)
        
        df_lw[("T_sky_C",lw_station[1])] = calculate_sky_temperature(df_lw,pv_config["atm_emissivity"])
        
        tres_lw = pd.Timedelta(pv_station["t_res_lw"])
        tres_sim = pd.Timedelta(timeres_sim)
        
        if tres_lw < tres_sim:
            lw_days = pd.to_datetime(df_lw.index.date).unique().strftime('%Y-%m-%d')    
            dfs_rs = []
            for day in lw_days:
                df_lw_day = df_lw.loc[day]
                try:
                    df_rs_day = dpf.downsample(df_lw_day,tres_lw, tres_sim)
                    dfs_rs.append(df_rs_day)
                except:
                    print(f'error in resampling on {day}')                    
            df_rs = pd.concat(dfs_rs,axis=0)
        else:
            df_rs = df_lw
            
        pv_station[f"df_lw_{year.split('_')[1]}"] = df_rs
        
        #combined_index = pv_station[f"df_sim_{year.split('_')[1]}"].index.intersection(df_rs.index)
        pv_station[f"df_sim_{year.split('_')[1]}"] = pd.concat([pv_station[f"df_sim_{year.split('_')[1]}"],
                                                df_rs],axis=1,join='inner')
    return pv_station

def generate_folders(rt_config,pv_config,path,model,power_model,eff_model,T_model):
    """
    Generate folders for results
    
    args:
    :param rt_config: dictionary with DISORT config
    :param pv_config: dictionary with inversion config
    :param path: main path for saving files or plotsintersection
    :param model: name of model used for inversion (power or current)
    :param power_model: name of power model
    :param eff_model: name of efficiency model used for inversion
    :param T_model: name of temperature model used for inversion
    
    
    out:
    :return fullpath: string with label for saving folders
    :return res_string_label: string with label for saving files
    
    """  
    
    #atmosphere model
    atm_geom_config = rt_config["disort_base"]["pseudospherical"]
    
    if atm_geom_config == True:
        atm_geom_folder = "Pseudospherical"
    else:
        atm_geom_folder = "Plane-parallel"
        
    dirs_exist = list_dirs(path)
    fullpath = os.path.join(path,atm_geom_folder)
    if atm_geom_folder not in dirs_exist:
        os.mkdir(fullpath)

    #Wavelength range of simulation
    wvl_config = rt_config["common_base"]["wavelength"]["pv"]
    
    if type(wvl_config) == list:
        wvl_folder_label = "Wvl_" + str(int(wvl_config[0])) + "_" + str(int(wvl_config[1]))
    elif wvl_config == "all":
        wvl_folder_label = "Wvl_Full"

    dirs_exist = list_dirs(fullpath)
    fullpath = os.path.join(fullpath,wvl_folder_label)        
    if wvl_folder_label not in dirs_exist:
        os.mkdir(fullpath)
        
    #DISORT Resolution    
    disort_config = rt_config["disort_rad_res"]
    theta_res = str(disort_config["theta"]).replace('.','-')
    phi_res = str(disort_config["phi"]).replace('.','-')
    
    disort_folder_label = 'Zen_' + theta_res + '_Azi_' + phi_res
    res_string_label = 'disortres_' + theta_res + '_' + phi_res + '_'
    
    dirs_exist = list_dirs(fullpath)
    fullpath = os.path.join(fullpath,disort_folder_label)

    if disort_folder_label not in dirs_exist:
        os.mkdir(fullpath)
        
    if rt_config["atmosphere"] == "default":
        atm_folder_label = "Atmos_Default"    
    elif rt_config["atmosphere"] == "cosmo":
        atm_folder_label = "Atmos_COSMO"
        
    dirs_exist = list_dirs(fullpath)  
    fullpath = os.path.join(fullpath,atm_folder_label)        
    if atm_folder_label not in dirs_exist:
        os.mkdir(fullpath)
    
    if rt_config["aerosol"]["source"] == "default":
        aero_folder_label = "Aerosol_Default"
    elif rt_config["aerosol"]["source"] == "aeronet":
        aero_folder_label = "Aeronet_" + rt_config["aerosol"]["station"]

    dirs_exist = list_dirs(fullpath)
    fullpath = os.path.join(fullpath,aero_folder_label)  
    if aero_folder_label not in dirs_exist:
        os.mkdir(fullpath)
        
    dirs_exist = list_dirs(fullpath)
    if model == "current":
        fullpath = os.path.join(fullpath,"Diode_Model")  
        if "Diode_Model" not in dirs_exist:
            os.mkdir(fullpath)
    elif model == "power":
        fullpath = os.path.join(fullpath,power_model)  
        if power_model not in dirs_exist:
            os.mkdir(fullpath)
    
        dirs_exist = list_dirs(fullpath)
        fullpath = os.path.join(fullpath,eff_model)  
        if eff_model not in dirs_exist:
            os.mkdir(fullpath)            
        
    dirs_exist = list_dirs(fullpath)
    fullpath = os.path.join(fullpath,T_model)  
    if T_model not in dirs_exist:
        os.mkdir(fullpath)    
        
    sza_label = "SZA_" + str(int(pv_config["sza_max"]["disort"]))
    
    dirs_exist = list_dirs(fullpath)
    fullpath = os.path.join(fullpath,sza_label)
    if sza_label not in dirs_exist:
        os.mkdir(fullpath)
    
    return fullpath, res_string_label

def select_days_calibration(station,pvstation,inv_config):
    """
    Select the correct days for calibration inversion from the dataframes, make sure that
    there is PV power data, otherwise inversion cannot continue!
    
    args:
    :param station: string, name of station
    :param pvstation: dictionary with data and info for one PV station
    :param inv_config: dictionary with inversion configuration
    
    out:
    :return pvstation: dictionary of PV station data with updated dataframes
    :return flag: list of booleans for measurement campaigns
    :return total_days: list of days used for calibration
    """
    
    df_total = []
    flag = [False]*len(inv_config["calibration_source"])
    cal_days_index = []    
    
    for ix, measurement in enumerate(pvstation["cal_source"]):
        year = "mk_" + measurement.split('_')[1]
        dfname = 'df_sim_' + year.split('_')[-1]
        if 'P_kW' in pvsys[key][dfname]:
            flag[ix] = True
            cal_days_index.append(pd.DatetimeIndex(inv_config["pv_stations"]
            [station]["calibration_days"][year]))
            
            df_total.append(pd.DataFrame())  
            df_total[ix] = pvstation[dfname].loc[np.isin(pvstation[dfname].index.date,
                               cal_days_index[ix].date)]                                       
        else:
            print('There is no PV data for station %s from %s \n' % (key,measurement))

    flag = all(flag)
    pvstation['df_cal'] = pd.concat(df_total,axis=0) 
    #Flatten list and get days back to string format
    total_days = [index.strftime('%Y-%m-%d') for index in 
                  [y for x in cal_days_index for y in x]]               
    
    return pvstation, flag, total_days

def prepare_calibration_data(key,dataframe,data_source,cal_source,substat_inv,
                             T_model,ap_substat,model):
    """
    Prepare dataframe for inversion depending on source config
    Check for NaNs
    Check for zero power values
    Create inversion dataframe with
    PV power, Irradiance, Temperature (Ambient and Module), Wind
    Drop multiindex
    
    args:    
    :param key: string, name of station
    :param dataframe: dataframe for calibration "df_cal"
    :param data_source: dictionary with config for data sources
    :param cal_source: list of campaigns to use for calibration
    :param substat_inv: substation to use for inversion
    :param T_model: string, temperature model to use for inversion
    :param ap_substat: dictionary with a-priori values for substation
    :param model: string, name of model used 
    
    out:
    :return: dictionary with new inversion dataframe
    """
    
    if model == "power":
        #Put together values for inversion
        #Measurement vector (y)    
        dataframe[('P_meas_W',substat_inv)] = dataframe[('P_kW',substat_inv)]*1000.
        
        #This is not a generalised piece of code!
        #For PV_11 we need to substract the power of the small system from the total for AÜW_1, only for MK 2018
        # if key == 'PV_11' and substat_inv == 'auew_1' and 'egrid_1' in dataframe.columns.get_level_values('substat'):        
        #     dataframe[('P_meas_W',substat_inv)] = dataframe[('P_meas_W',substat_inv)] -\
        #     dataframe[('P_kW','egrid_1')]*1000    
        
        #Measurement uncertainty of 1% (set in config file) but not less than 100W (alsoset in config file)
        dataframe[('P_error_meas_W',substat_inv)] = dataframe[('P_meas_W',substat_inv)]*ap_substat["p_err_rel"]    
        dataframe[('P_error_meas_W',substat_inv)].where(dataframe
            [('P_error_meas_W',substat_inv)] > ap_substat["p_err_min"],ap_substat["p_err_min"],inplace=True)
        
    elif model =="current":
        if "inverter_strings" in ap_substat:
            dataframe[('I_meas_A',substat_inv)] = pd.concat([dataframe[(f'Idc_{n}',substat_inv)] for n in 
                                                   ap_substat["inverter_strings"]],axis=1).mean(axis=1)        
        else:
            dataframe[('I_meas_A',substat_inv)] = dataframe[('Idc_A',substat_inv)]
        
        #Measurement uncertainty of 1% (set in config file) but not less than 100W (alsoset in config file)
        dataframe[('I_error_meas_A',substat_inv)] = dataframe[('I_meas_A',substat_inv)]*ap_substat["i_err_rel"]    
        dataframe[('I_error_meas_A',substat_inv)].where(dataframe
            [('I_error_meas_A',substat_inv)] > ap_substat["i_err_min"],ap_substat["i_err_min"],inplace=True)
    
    #Get the solar angles in radians
    dataframe[('theta0rad','sun')] = deg2rad(dataframe.sza)
    dataframe[('phi0rad','sun')] = deg2rad(dataframe.phi0)    

    #Get Values for irradiance calculation from simulation, plus albedo and sun angles
    df_cal_notnan = dataframe.loc[:,[('Edirdown','libradtran_pv'),('Idiff','libradtran_pv'),
                                     ('Edirdown','libradtran_pyr'),('Idiff','libradtran_pyr'),
                                      ('theta0rad','sun'),('phi0rad','sun'),('albedo',
                                      'constant')]]
    df_cal_notnan.dropna(axis=0,inplace=True)
    #Drop multiindex
    df_cal_notnan.columns = pd.MultiIndex.from_tuples([(f"{var}_pv",substat) 
        if "pv" in substat else (f"{var}_pyr",substat) 
        if 'pyr' in substat else (var,substat) for (var,substat) in 
        df_cal_notnan.columns],names=df_cal_notnan.columns.names)
                
    df_cal_notnan.columns = df_cal_notnan.columns.droplevel(level='substat')                            
    
    #Drop days with errors
    if 'error_days' in ap_substat:
        for mk in sorted(ap_substat['error_days']):
            if mk.split('_')[1] in "_".join(cal_source):
                for day in ap_substat['error_days'][mk]:
                    df_cal_notnan.drop(index=df_cal_notnan.loc[day.strftime('%Y-%m-%d')].index,inplace=True)
                    print('Leaving out %s' % day)
    
    if model == "power":
        #Find all valid power values
        P_meas_W = dataframe.loc[dataframe
        [('P_meas_W',substat_inv)].notna(), # & dataframe[('P_meas_W',substat_inv)] != 0.
        ('P_meas_W',substat_inv)].rename('P_meas_W',inplace=True)    
        
        df_cal_notnan = pd.merge(df_cal_notnan,P_meas_W,left_index=True,right_index=True)
        
        #Get all errors
        P_error_meas_W = dataframe.loc[dataframe
        [('P_error_meas_W',substat_inv)].notna(), # & dataframe[('P_meas_W',substat_inv)] != 0.
        ('P_error_meas_W',substat_inv)].rename('P_error_meas_W',inplace=True)    
        
        df_cal_notnan = pd.merge(df_cal_notnan,P_error_meas_W,left_index=True,right_index=True)
    elif model == "current":
        #Find all valid power values
        I_meas_A = dataframe.loc[dataframe
        [('I_meas_A',substat_inv)].notna(), # & dataframe[('P_meas_W',substat_inv)] != 0.
        ('I_meas_A',substat_inv)].rename('I_meas_A',inplace=True)    
        
        df_cal_notnan = pd.merge(df_cal_notnan,I_meas_A,left_index=True,right_index=True)
        
        #Get all errors
        I_error_meas_A = dataframe.loc[dataframe
        [('I_error_meas_A',substat_inv)].notna(), # & dataframe[('P_meas_W',substat_inv)] != 0.
        ('I_error_meas_A',substat_inv)].rename('I_error_meas_A',inplace=True)    
        
        df_cal_notnan = pd.merge(df_cal_notnan,I_error_meas_A,left_index=True,right_index=True)
    
    n_h2o_mm = dataframe.loc[dataframe
        [('n_h2o_mm',"cosmo")].notna(), # & dataframe[('P_meas_W',substat_inv)] != 0.
        ('n_h2o_mm',"cosmo")].rename('n_h2o_mm',inplace=True)    
    
    df_cal_notnan = pd.merge(df_cal_notnan,n_h2o_mm,left_index=True,right_index=True)        
    
    #Time series for temperature model
    if T_model == "Dynamic_or_Measured":                         
        T_model_C = pd.Series(dtype='float64')
        del_vals = []
        for mk in data_source["temp_module"]:                        
            if mk.split('_')[1] not in "_".join(cal_source):
                del_vals.append(mk)
        for val in del_vals:
            del data_source["temp_module"][val]
            
        for sensor in sorted(set(sorted(data_source["temp_module"].values()))):                        
            if "model" in sensor:
                T_substat_name = f"{substat}_{sensor.split('_')[0]}"
            else:
                T_substat_name = sensor
            #T_substat
            T_model_C = pd.concat([T_model_C,dataframe.loc[dataframe[("T_module_C",
                        T_substat_name)].notna(),
                        ("T_module_C",T_substat_name)]],axis=0).\
                         rename('T_module_C',inplace=True)
            
        df_cal_notnan = pd.merge(df_cal_notnan,T_model_C,left_index=True,right_index=True)

    else:    
        T_ambient_C = pd.Series(dtype='float64')
        #Check for missing temperature values
        del_vals = []
        for mk in data_source["temp_amb"]:            
            if mk.split('_')[1] not in "_".join(cal_source):
                del_vals.append(mk)
        for val in del_vals:
            del data_source["temp_amb"][val]
        
        for sensor in sorted(set(sorted(data_source["temp_amb"].values()))):                    
            if sensor == "cosmo":
                Tname = "T_ambient_2M_C"
            elif "Pyr" in sensor:
                Tname = "T_ambient_pyr_C"
            elif "Windmast" in sensor:
                Tname = "T_ambient_C"
            
            T_ambient_C = pd.concat([T_ambient_C,dataframe.loc[dataframe[(Tname,
                      sensor)].notna(),(Tname,sensor)]],axis=0).\
                      rename('T_ambient_C',inplace=True)              
            
        df_cal_notnan = pd.merge(df_cal_notnan,T_ambient_C,left_index=True,right_index=True)
                
        v_wind_ms = pd.Series(dtype='float64')
        #Check for missing wind values
        del_vals = []
        for mk in data_source["wind"]:
            if mk.split('_')[1] not in "_".join(cal_source):
                del_vals.append(mk)
        for val in del_vals:
            del data_source["wind"][val]
                
        for sensor in sorted(set(sorted(data_source["wind"].values()))):            
            if sensor == "cosmo":
                vname = "v_wind_10M"
            elif sensor == "Windmast":
                vname = "v_wind_mast_ms"
                            
            v_wind_ms = pd.concat([v_wind_ms,dataframe.loc[dataframe[(vname,
                    sensor)].notna(),(vname,sensor)]],axis=0).\
                    rename('v_wind_ms',inplace=True)            
    
        #Final dataframe
        df_cal_notnan = pd.merge(df_cal_notnan,v_wind_ms,left_index=True,right_index=True)
        
        T_sky_C = pd.Series(dtype='float64')
        #Check for missing wind values
        del_vals = []
        for mk in data_source["longwave"]:
            if mk.split('_')[1] not in "_".join(cal_source):
                del_vals.append(mk)
        for val in del_vals:
            del data_source["longwave"][val]
            
        for lw_sensor in sorted(set(sorted([','.join(val) \
                    for val in sorted(data_source["longwave"].values())]))): 
            sensor = lw_sensor.split(',')
            T_sky_C = pd.concat([T_sky_C,dataframe.loc[dataframe[("T_sky_C",
                    sensor[1])].notna(),("T_sky_C",sensor[1])]],axis=0).\
                    rename('T_sky_C',inplace=True)         
                    
        df_cal_notnan = pd.merge(df_cal_notnan,T_sky_C,left_index=True,right_index=True)
        
    return df_cal_notnan
    
###############################################################
###   Plotting results                                      ###
###############################################################

def power_plot(key,substat,day,df_cal,df_sim,sza_index,ymax,sza,pars,opt_pars,
               eff_model,T_model):
    """
    

    Parameters
    ----------
    key : string, name of PV station
    substat : string, name of substat
    day : string, day for which plot is made
    df_cal : dataframe, calibration dataframe
    df_sim : dataframe, simulation dataframe
    sza_index : index with times within SZA limit, 
    ymax : float, maximum for y axis
    sza : float, SZA limit
    pars : dictionary with model parameters
    opt_pars : dictionary with optimisation parameters
    eff_model : string with name of efficiency model
    T_model : string with temperature model

    Returns
    -------
    fig : matplotlib figure with power plot

    """
    
    fig, ax = plt.subplots(figsize=(9,8))                    

    ax.plot(df_cal.index,df_cal.P_meas_W/1000, color='r',linestyle = '-')
    ax.fill_between(df_cal.index, df_cal.P_meas_W/1000 - df_cal.P_error_meas_W/1000,
                     df_cal.P_meas_W/1000 + df_cal.P_error_meas_W/1000,color='r', alpha=0.3)
    
    ax.plot(df_cal.index,df_cal.P_MAP/1000, color='b',linestyle = '-')
    
    ax.fill_between(df_cal.index, df_cal.P_MAP/1000 - df_cal.error_power_fit/1000,
                     df_cal.P_MAP/1000 + df_cal.error_power_fit/1000,color='b', alpha=0.3)
    
    ax.plot(df_cal.index,df_cal.P_MAP_extra/1000,color='b',linestyle = '--')        
    
    plt.legend((r'$P_{\rm AC,meas}$',r'$P_{\rm AC,mod,SZA \leq' + sza + '^{\circ}}$',
                r'$P_{\rm AC,mod,SZA > ' + sza + '^{\circ}}$'),loc='upper right')
    
    # Make the y-axis label, ticks and tick labels match the line color.
    plt.ylabel('Power (kW)')#, color='b')
    plt.xlabel('Time (UTC)')
    plt.title('MAP solution for ' + key + ', ' + substat + ' on ' + day)
    #plt.ylim([0,50000])
    
    datemin = pd.Timestamp(df_cal.index[0])
    datemax = pd.Timestamp(df_cal.index[-1])      
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))      
    ax.axvspan(datemin,sza_index[0], alpha=0.2, color='gray')                
    ax.axvspan(sza_index[-1],datemax, alpha=0.2, color='gray')
    ax.set_xlim([datemin, datemax])
    ax.set_ylim([0,ymax])
    #plt.axis('square')
    start_y = 0.6
    
    datemid = df_cal.iloc[df_cal["P_MAP"].argmax()].name
    start_x = (datemid - datemin)/(datemax - datemin) - 0.05  
    
    #print parameters on plot                
    c_par = 0
    units = [r'$^\circ$',r'$^\circ$','','kW','$^{\circ}C^{-1}$']
    for i, par in enumerate(pars[0:5]):
        if par[2] != 0:
            if par[0] == 'theta':
                parstring = par[0] + ' = ' + str(np.round(np.rad2deg(opt_pars['x_min'][c_par]),2)) + '$\pm$' +\
                     str(np.round(np.rad2deg(opt_pars['s_vec'][c_par]),2)) + units[i]
            elif par[0] == 'phi':
                parstring = par[0] + ' = ' + str(np.round(np.rad2deg(azi_shift(opt_pars['x_min'][c_par])),2)) + '$\pm$' +\
                     str(np.round(np.rad2deg(opt_pars['s_vec'][c_par]),2)) + units[i]
            elif par[0] == "zeta":                            
                parstring = par[0] + ' = ' + str(np.round(opt_pars['x_min'][c_par]*100,3)) + '$\pm$' +\
                     str(np.round(opt_pars['s_vec'][c_par]*100,3)) + ' %' + units[i] 
            else:                                 
                 parstring = par[0] + ' = ' + str(np.round(opt_pars['x_min'][c_par],2)) + '$\pm$' +\
                     str(np.round(opt_pars['s_vec'][c_par],2)) + ' ' + units[i]
            c_par = c_par + 1
        else:
            if par[0] == 'theta':
                parstring = par[0] + ' = ' + str(np.round(np.rad2deg(par[1]),2)) + units[i] + ' (fixed)'
            elif par[0] == 'phi':
                parstring = par[0] + ' = ' + str(np.round(np.rad2deg(azi_shift(par[1])),2)) + units[i] + ' (fixed)'
            else:
                parstring = par[0] + ' = ' + str(par[1]) + ' ' + units[i] + ' (fixed)'
            
        plt.annotate(parstring,xy=(start_x,start_y - 0.05*i),xycoords='figure fraction',fontsize=14)
                    
    if eff_model == "Ransome":
        units = ['','$W\,m^{-2}$']
    elif eff_model == "Beyer":
        units = ['','$m^2\, W^{-1}$','']
    else:
        units = []
    
    if T_model == "Tamizhmani":
        units.extend(['','$^{\circ}C\, m^2\, W^{-1}$','$^{\circ}C\,m^{-1}\,s$','$^{\circ}C$'])
    elif T_model == "King":
        units.extend(['','$m^{-1}\,s$','$^{\circ}C$'])
    elif T_model == "Faiman" or T_model == "Barry":
        units.extend(['$W\,m^{-2}\,^{\circ}C^{-1}$','$W\,s\,m^{-3}\,^{\circ}C^{-1}$','$^{\circ}C$'])
        
    for i, par in enumerate(pars[5:]):
        if T_model != "King":
            if par[2] != 0:
                parstring = '$' + par[0][0] + '_' + par[0][1] + '$ = ' + str(np.round(opt_pars['x_min'][c_par],3)) + '$\pm$' +\
                     str(np.round(opt_pars['s_vec'][c_par],3)) + ' ' + units[i]
                c_par = c_par + 1
            else:
                parstring = '$' + par[0][0] + '_' + par[0][1] + '$ = ' + str(par[1]) + ' (fixed)'
        else:
            if par[2] != 0:
                parstring = '$' + par[0][0] + '$ = ' + str(np.round(opt_pars['x_min'][c_par],3)) + '$\pm$' +\
                     str(np.round(opt_pars['s_vec'][c_par],3)) + ' ' + units[i]
                c_par = c_par + 1
            else:
                parstring = '$' + par[0][0] + '$ = ' + str(par[1]) + ' (fixed)'
            
        plt.annotate(parstring,xy=(start_x,start_y - 0.05*(i + 5)),
                     xycoords='figure fraction',fontsize=14)
        
    
    plt.annotate('$\chi^2$ = ' + str(np.round(opt_pars['min_chisq'],2)) 
                 ,xy=(start_x,0.15),xycoords='figure fraction',fontsize=14)
    
    fig.tight_layout()
    
    return fig

def current_plot(key,substat,day,df_cal,df_sim,sza_index,ymax,sza,pars,opt_pars,
                T_model):
    """
    

    
    Parameters
    ----------
    key : string, name of PV station
    substat : string, name of substat
    day : string, day for which plot is made
    df_cal : dataframe, calibration dataframe
    df_sim : dataframe, simulation dataframe
    sza_index : index with times within SZA limit, 
    ymax : float, maximum for y axis
    sza : float, SZA limit
    pars : dictionary with model parameters
    opt_pars : dictionary with optimisation parameters
    eff_model : string with name of efficiency model
    T_model : string with temperature model

    Returns
    -------
    fig : matplotlib figure with current plot

    """
    
    fig, ax = plt.subplots(figsize=(9,8))                    

    ax.plot(df_cal.index,df_cal.I_meas_A, color='r',linestyle = '-')
    ax.fill_between(df_cal.index, df_cal.I_meas_A - df_cal.I_error_meas_A,
                      df_cal.I_meas_A + df_cal.I_error_meas_A,color='r', alpha=0.3)
    
    ax.plot(df_cal.index,df_cal.I_MAP, color='b',linestyle = '-')
    
    ax.fill_between(df_cal.index, df_cal.I_MAP - df_cal.error_current_fit,
                      df_cal.I_MAP + df_cal.error_current_fit,color='b', alpha=0.3)
    
    ax.plot(df_cal.index,df_cal.I_MAP_extra,color='b',linestyle = '--')        
    
    plt.legend((r'$I_{\rm DC,meas}$',r'$I_{\rm DC,mod,SZA \leq' + sza + '^{\circ}}$',
                r'$I_{\rm DC,mod,SZA > ' + sza + '^{\circ}}$'),loc='upper right')
    
    # Make the y-axis label, ticks and tick labels match the line color.
    plt.ylabel('Current (A)')#, color='b')
    plt.xlabel('Time (UTC)')
    plt.title('MAP solution for ' + key + ', ' + substat + ' on ' + day)
    #plt.ylim([0,50000])
    
    datemin = pd.Timestamp(df_cal.index[0])
    datemax = pd.Timestamp(df_cal.index[-1])      
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))      
    ax.axvspan(datemin,sza_index[0], alpha=0.2, color='gray')                
    ax.axvspan(sza_index[-1],datemax, alpha=0.2, color='gray')
    ax.set_xlim([datemin, datemax])
    ax.set_ylim([0,ymax])
    #plt.axis('square')
    start_y = 0.7
    
    datemid = df_cal.iloc[df_cal["I_MAP"].argmax()].name
    start_x = (datemid - datemin)/(datemax - datemin) - 0.05   

    #print parameters on plot                
    c_par = 0
    #Units for theta, phi, n, Impn, Npp, Ki
    units = [r'$^\circ$',r'$^\circ$','','A','','$^{\circ}C^{-1}$']
    for i, par in enumerate(pars[0:6]):
        if par[2] != 0:
            if par[0] == 'theta':
                parstring = par[0] + ' = ' + str(np.round(np.rad2deg(opt_pars['x_min'][c_par]),2)) + '$\pm$' +\
                      str(np.round(np.rad2deg(opt_pars['s_vec'][c_par]),2)) + units[i]
            elif par[0] == 'phi':
                parstring = par[0] + ' = ' + str(np.round(np.rad2deg(azi_shift(opt_pars['x_min'][c_par])),2)) + '$\pm$' +\
                      str(np.round(np.rad2deg(opt_pars['s_vec'][c_par]),2)) + units[i]
            elif par[0] == "ki":                            
                parstring = par[0] + ' = ' + str(np.round(opt_pars['x_min'][c_par]*100,3)) + '$\pm$' +\
                      str(np.round(opt_pars['s_vec'][c_par]*100,3)) + ' %' + units[i] 
            else:                                 
                  parstring = par[0] + ' = ' + str(np.round(opt_pars['x_min'][c_par],2)) + '$\pm$' +\
                      str(np.round(opt_pars['s_vec'][c_par],2)) + ' ' + units[i]
            c_par = c_par + 1
        else:
            if par[0] == 'theta':
                parstring = par[0] + ' = ' + str(np.round(np.rad2deg(par[1]),2)) + units[i] + ' (fixed)'
            elif par[0] == 'phi':
                parstring = par[0] + ' = ' + str(np.round(np.rad2deg(azi_shift(par[1])),2)) + units[i] + ' (fixed)'
            else:
                parstring = par[0] + ' = ' + str(par[1]) + ' ' + units[i] + ' (fixed)'
            
        plt.annotate(parstring,xy=(start_x,start_y - 0.05*i),xycoords='figure fraction',fontsize=14)
                    
    
    units = []
    
    if T_model == "Tamizhmani":
        units.extend(['','$^{\circ}C\, m^2\, W^{-1}$','$^{\circ}C\,m^{-1}\,s$','$^{\circ}C$'])
    elif T_model == "King":
        units.extend(['','$m^{-1}\,s$','$^{\circ}C$'])
    elif T_model == "Faiman":
        units.extend(['$W\,m^{-2}\,^{\circ}C^{-1}$','$W\,s\,m^{-3}\,^{\circ}C^{-1}$','$^{\circ}C$'])
        
    for i, par in enumerate(pars[6:]):
        if T_model != "King":
            if par[2] != 0:
                parstring = '$' + par[0][0] + '_' + par[0][1] + '$ = ' + str(np.round(opt_pars['x_min'][c_par],3)) + '$\pm$' +\
                      str(np.round(opt_pars['s_vec'][c_par],3)) + ' ' + units[i]
                c_par = c_par + 1
            else:
                parstring = '$' + par[0][0] + '_' + par[0][1] + '$ = ' + str(par[1]) + ' (fixed)'
        else:
            if par[2] != 0:
                parstring = '$' + par[0][0] + '$ = ' + str(np.round(opt_pars['x_min'][c_par],3)) + '$\pm$' +\
                      str(np.round(opt_pars['s_vec'][c_par],3)) + ' ' + units[i]
                c_par = c_par + 1
            else:
                parstring = '$' + par[0][0] + '$ = ' + str(par[1]) + ' (fixed)'
            
        plt.annotate(parstring,xy=(start_x,start_y - 0.05*(i + 6)),
                      xycoords='figure fraction',fontsize=14)
        
    
    plt.annotate('$\chi^2$ = ' + str(np.round(opt_pars['min_chisq'],2)) 
                  ,xy=(start_x,0.15),xycoords='figure fraction',fontsize=14)
    
    fig.tight_layout()
    
    return fig


def irradiance_plot(key,day,df_cal,df_sim,pars,opt_pars):
    """
    

    Parameters
    ----------
    key : string, name of PV station    
    day : string, day for which plot is made
    df_cal : dataframe, calibration dataframe
    df_sim : dataframe, simulation dataframe    
    pars : dictionary with model parameters
    opt_pars : dictionary with optimisation parameters
    

    Returns
    -------
    fig : matplotlib figure with irradiance plot
  

    """
    
    #Irradiance plot for each day
    fig, ax = plt.subplots(figsize=(9,8))                    
    
    df_sim.Etotdown.plot(color='g',style = '-',legend=True)               
    df_cal.Etotpoa.plot(color='r',style = '-',legend=True)    
    df_cal.Edirpoa.plot(color='k',style = '-',legend=True)
    df_cal.Ediffpoa.plot(color='b',style = '-',legend=True)    
    df_sim.Ediffdown.plot(color='b',style = '--',legend=True)
        
    plt.legend((r'$G_{\rm tot,[0.3,1.2]\mu m}^{\downarrow}$',r'$G_{\rm tot,[0.3,1.2]\mu m}^{\angle}$',
        r'$G_{\rm dir,[0.3,1.2]\mu m}^{\angle}$',r'$G_{\rm diff,[0.3,1.2]\mu m}^{\angle}$',        
        r'$G_{\rm diff,[0.3,1.2]\mu m}^{\downarrow}$'),loc='upper right')
    
    # Make the y-axis label, ticks and tick labels match the line color.
    plt.ylabel(r'Irradiance, ($W/m^2$)')#, color='b')
    plt.xlabel('Time (UTC)')
    plt.title('Irradiance components after calibration for ' + key  + ' on ' + day)
    #plt.axis('square')
    #plt.ylim([0,1000])
    datemin = pd.Timestamp(df_cal.index[0])
    datemax = pd.Timestamp(df_cal.index[-1])      
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  
    plt.xlim([datemin, datemax])
    
    c_par = 0
    for par in pars[0:2]:
        if par[0] == 'theta':
            if par[2] == 0:
                theta_val = np.rad2deg(par[1])                            
            else:
                theta_val = np.rad2deg(opt_pars['x_min'][c_par])
                c_par = c_par + 1
        if par[0] == 'phi':
            if par[2] == 0:
                phi_val = np.rad2deg(azi_shift(par[1]))
            else:
                phi_val = np.rad2deg(azi_shift(opt_pars['x_min'][c_par]))
                c_par = c_par + 1   
    
    plt.annotate(r'$\theta$ (tilt) = ' + str(np.round(theta_val,2)) + '$^\circ$',
                 xy=(0.15,0.9),xycoords='figure fraction',fontsize=14)     
    plt.annotate('$\phi$ (azimuth) = ' + str(np.round(phi_val,2)) + 
                 '$^\circ$',xy=(0.15,0.85),xycoords='figure fraction',fontsize=14)     
    
    fig.tight_layout()
        
    return fig

def plot_fit_results(key,pv_station,substat_inv,rt_config,pv_config,styles,home,
                      model_dict,flag,pars):
    """
    Plot results from the non-linear inversion
    
    args:
    :param key: string, name of current pv station
    :param pv_station: dictionary of one PV system given by key
    :param substat_inv: string, name of substation used for inversion
    :param rt_config: dictionary with disort configuration for saving
    :param pv_config: dictionary with inversion configuration for saving    
    :param styles: dictionary with plot styles
    :param home: string with homepath
    :param model_dict: dictionary with different models used for inversion
    :param flag: boolean, whether solution has been found or not
    :param pars: list of tuples with parameters (name,ap_value,ap_error)
    
    out:
    :return folder_label: string with folder path for saving plots & results
    
    """
    
    
    plt.ioff()
    plt.style.use(styles["single_small"])

    cal_model = model_dict["cal_model"]    
    power_model = model_dict['power_model']
    T_model = model_dict["T_model"]
    eff_model = model_dict["eff_model"]
    mainpath = pv_config['results_path']
    sza_limit = pv_config["sza_max"]["disort"]
    str_sza = str(int(sza_limit))
    
    savepath = os.path.join(home,mainpath["main"])
    
    folder_label, res_string_label = generate_folders(rt_config,pv_config,
                                      savepath,cal_model,power_model,eff_model,T_model)    
    
    if mainpath["plots"] not in list_dirs(folder_label):
        os.mkdir(os.path.join(folder_label,mainpath["plots"]))
    
    plotpath = os.path.join(folder_label,mainpath["plots"])
    
    station_folders = list_dirs(plotpath)
    if key not in station_folders:        
        os.mkdir(os.path.join(plotpath,key))
        
    substat_folders = list_dirs(os.path.join(plotpath,key))
    if substat_inv not in substat_folders:
        os.mkdir(os.path.join(plotpath,key,substat_inv))
    
    save_folder = os.path.join(plotpath,key,substat_inv)
        
    opt_pars = pv_station['substations'][substat_inv]['opt']
    
    if flag:
        if cal_model == "power" and 'P_MAP' in pv_station['substations'][substat_inv]['df_cal']:            
            #Get dataframe from calibration
            df_cal = pv_station['substations'][substat_inv]['df_cal'].xs(substat_inv,level='substat',axis=1)
            #Get dataframe from RT simulation
            df_rt_sim = pv_station['df_cal'].xs('libradtran_pv',level='substat',axis=1)
            
            max_df = np.max(np.max(df_cal.loc[pv_station['df_cal'][("sza","sun")] <= sza_limit][['P_meas_W','P_MAP']]))/1000    
            if max_df > 50:
                max_pv = np.ceil(max_df/10)*10
            elif max_df > 25:
                max_pv = np.ceil(max_df/5)*5
            elif max_df > 10:
                max_pv = np.ceil(max_df/2)*2
            else:
                max_pv = np.ceil(max_df)
            
            for iday in pv_station['substations'][substat_inv]['cal_days']:                
                #Slice all values from iday
                df_cal_day = df_cal.loc[iday]
                sza_index_day = df_cal_day.loc[df_cal_day.theta0rad <= np.deg2rad(sza_limit)].index    
                df_rt_sim_day = df_rt_sim.loc[iday]
                
                #Fit plot for each day
                fig = power_plot(key,substat_inv,iday,df_cal_day,df_rt_sim_day,
                                  sza_index_day,max_pv,str_sza,
                                  pars,opt_pars,eff_model,T_model)
                
                plt.savefig(os.path.join(save_folder,'chi_sq_fit_' + key + '_' + iday + '_' + 
                              res_string_label + 'power.png'))
                #plt.savefig('chi_sq_fit_' + pv_station['code'] + '_' + test_days[iday] + '_power.eps')
                plt.close(fig)
                
                #Plot irradiance components for each day
                fig = irradiance_plot(key,iday,df_cal_day,df_rt_sim_day,pars,opt_pars)
                
                plt.savefig(os.path.join(save_folder,'chi_sq_fit_' + key + '_' + iday + '_' + res_string_label
                                          + 'irradiance.png'))
                #plt.savefig('chi_sq_fit_' + pv_station['code'] + '_' + test_days[iday] + '_irradiance.eps')
                plt.close(fig)
                
            plot_grid_power_fit(key,pv_station,substat_inv,styles,save_folder,res_string_label,sza_limit,pars)
        
        #Current plots
        elif cal_model == "current" and 'I_MAP' in pv_station['substations'][substat_inv]['df_cal']:  
            #Get dataframe from calibration
            df_cal = pv_station['substations'][substat_inv]['df_cal'].xs(substat_inv,level='substat',axis=1)
            #Get dataframe from RT simulation
            df_rt_sim = pv_station['df_cal'].xs('libradtran_pv',level='substat',axis=1)
            
            max_df = np.max(np.max(df_cal.loc[pv_station['df_cal'][("sza","sun")] <= sza_limit][['I_meas_A','I_MAP']]))            
            if max_df > 50:
                max_pv = np.ceil(max_df/10)*10
            elif max_df > 25:
                max_pv = np.ceil(max_df/5)*5
            elif max_df > 10:
                max_pv = np.ceil(max_df/2)*2
            else:
                max_pv = np.ceil(max_df)
            
            for iday in pv_station['substations'][substat_inv]['cal_days']:                
                #Slice all values from iday
                df_cal_day = df_cal.loc[iday]
                sza_index_day = df_cal_day.loc[df_cal_day.theta0rad <= np.deg2rad(sza_limit)].index    
                df_rt_sim_day = df_rt_sim.loc[iday]
                
                #Fit plot for each day
                fig = current_plot(key,substat_inv,iday,df_cal_day,df_rt_sim_day,
                                  sza_index_day,max_pv,str_sza,
                                  pars,opt_pars,T_model)
                
                plt.savefig(os.path.join(save_folder,'chi_sq_fit_' + key + '_' + iday + '_' + 
                              res_string_label + 'current.png'))
                #plt.savefig('chi_sq_fit_' + pv_station['code'] + '_' + test_days[iday] + '_power.eps')
                plt.close(fig)
                
                #Plot irradiance components for each day
                fig = irradiance_plot(key,iday,df_cal_day,df_rt_sim_day,pars,opt_pars)
                
                plt.savefig(os.path.join(save_folder,'chi_sq_fit_' + key + '_' + iday + '_' + res_string_label
                                          + 'irradiance.png'))
                #plt.savefig('chi_sq_fit_' + pv_station['code'] + '_' + test_days[iday] + '_irradiance.eps')
                plt.close(fig)
            
            
    return folder_label

def plot_grid_power_fit(key,pv_station,substat_inv,styles,savepath,res_string_label,sza_limit,pars):
    """
    Plots in a grid of subplots
    
    args:
    :param key: string, name of current pv station
    :param pv_station: dictionary of one PV system given by key
    :param substat_inv: string, name of substation used for inversion    
    :param styles: dictionary with plot styles
    :param savepath: string with path for saving plot
    :param res_string_label: string with label for DISORT resolution 
    :param sza_limit: float, limit for SZA    
    :param pars: list of tuples with parameters (name,ap_value,ap_error)
    
    """
    
    plt.ioff()     
    plt.close()
    plt.style.use(styles["combo_small"])   
        
    opt_pars = pv_station['substations'][substat_inv]['opt']
    num_plots = len(pv_station['substations'][substat_inv]['cal_days'])
    if num_plots >= 17:
        fig, axs = plt.subplots(4, 5, sharey='row')
    elif num_plots >= 13:
        fig, axs = plt.subplots(4, 4, sharey='row')
    elif num_plots >= 10:
        fig, axs = plt.subplots(4, 3, sharey='row')
    elif num_plots <= 9:
        fig, axs = plt.subplots(3, 3, sharey='row', figsize=(10,9))    
    
    axvec = axs.flatten()
    c_par = 0
    for par in pars:
        if par[0] == 'theta':
            if par[2] == 0:
                theta_val = np.rad2deg(par[1])
                theta_err = 0
            else:
                theta_val = np.rad2deg(opt_pars['x_min'][c_par])
                theta_err = np.rad2deg(opt_pars['s_vec'][c_par])
                c_par = c_par + 1
        if par[0] == 'phi':
            if par[2] == 0:
                phi_val = np.rad2deg(azi_shift(par[1]))
                phi_err = 0
            else:
                phi_val = np.rad2deg(azi_shift(opt_pars['x_min'][c_par]))
                phi_err = np.rad2deg(opt_pars['s_vec'][c_par])
                c_par = c_par + 1        
        
    opt_string = r'$\theta$ (tilt) = ' + "{:.2f}".format(np.round(theta_val,2)) + '$\pm $' +\
                  "{:.2f}".format(np.round(theta_err,2)) + '$^\circ$, $\phi$ (azimuth) = '\
                  + "{:.2f}".format(np.round(phi_val,2)) + '$\pm $'\
                  + "{:.2f}".format(np.round(phi_err,2)) + '$^\circ$'
    
    fig.suptitle(key + ', ' + substat_inv + ' : ' + opt_string)    
        
    #Slice all values coming from inversion
    df_cal = pv_station['substations'][substat_inv]['df_cal'].xs(substat_inv,level='substat',axis=1)
    
    max_df = np.max(np.max(df_cal.loc[pv_station['df_cal'][("sza","sun")] <= sza_limit][['P_meas_W','P_MAP']]))/1000    
    if max_df > 50:
        max_pv = np.ceil(max_df/10)*10
    elif max_df > 25:
        max_pv = np.ceil(max_df/5)*5
    elif max_df > 10:
        max_pv = np.ceil(max_df/2)*2
    else:
        max_pv = np.ceil(max_df)

    for ix, iday in enumerate(pv_station['substations'][substat_inv]['cal_days']):
        #Slice all values coming from inversion
        df_cal_day = df_cal.loc[iday]
        
        #Slice all values coming from libradtran
        #df_simulation_day = pv_station['df_cal'].xs('libradtran',level='substat',axis=1).loc[iday]
        df_sun_day = pv_station['df_cal'].xs('sun',level='substat',axis=1).loc[iday]
        sza_index_day = df_sun_day.loc[df_sun_day.sza <= sza_limit].index   
    
        axvec[ix].plot(df_cal_day.index,df_cal_day.P_meas_W/1000,color='r')
        axvec[ix].fill_between(df_cal_day.index, df_cal_day.P_meas_W/1000 - df_cal_day.P_error_meas_W/1000,
                          df_cal_day.P_meas_W/1000 + df_cal_day.P_error_meas_W/1000,color='r', alpha=0.3)
        
        axvec[ix].plot(df_cal_day.index,df_cal_day.P_MAP/1000,color='b')        
        axvec[ix].fill_between(df_cal_day.index, df_cal_day.P_MAP/1000 - df_cal_day.error_power_fit/1000,
                          df_cal_day.P_MAP/1000 + df_cal_day.error_power_fit/1000,color='b', alpha=0.3)
       
        axvec[ix].plot(df_cal_day.index,df_cal_day.P_MAP_extra/1000,color='b',linestyle = '--')        
        #axvec[ix].set_title(iday + " ",fontsize=14,loc="right",pad=-16)
        axvec[ix].text(0.5, 0.01, iday + " ",verticalalignment='bottom', horizontalalignment='center',
        transform=axvec[ix].transAxes, fontsize=14)
        datemin = pd.Timestamp(df_cal_day.index[0])
        datemax = pd.Timestamp(df_cal_day.index[-1])        
        
        axvec[ix].xaxis.set_major_locator(mdates.HourLocator(interval=6))
        axvec[ix].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        axvec[ix].set_xlim([datemin, datemax])
        axvec[ix].set_xlabel("")
        axvec[ix].set_ylim([0,max_pv])

        axvec[ix].axvspan(datemin,sza_index_day[0],alpha=0.2,color='gray')
        axvec[ix].axvspan(sza_index_day[-1],datemax,alpha=0.2,color='gray')
    
    if len(axvec) - len(pv_station['substations'][substat_inv]['cal_days']) != 0:
        for ix_extra in range(ix + 1,len(axvec)):
            axvec[ix_extra].xaxis.set_major_locator(mdates.HourLocator(interval=6))
            axvec[ix_extra].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            axvec[ix_extra].set_xlim([datemin, datemax])        
            axvec[ix_extra].set_ylim([0,max_pv])
            
    for ax in axvec:
        ax.label_outer()
    
#    plt.annotate(r'$\theta$ (tilt) = ' + "{:.2f}".format(np.round(np.rad2deg(opt_pars['x_min'][0]),2)) + '$\pm $' + 
#                 "{:.2f}".format(np.round(np.rad2deg(opt_pars['s_vec'][0]),2)) + '$^\circ$, $\phi$ (azimuth) = ' 
#                 + "{:.2f}".format(np.round(np.rad2deg(azi_shift(opt_pars['x_min'][1])),2)) + '$\pm $'
#                 + "{:.2f}".format(np.round(np.rad2deg(opt_pars['s_vec'][1]),2)) + '$^\circ$',
#                 xy=(0.2,0.95),xycoords='figure fraction',fontsize=16)     
#    
#    fig.text(0.5, 0.04, 'Time (UTC)', ha='center',fontsize=20)
#    fig.text(0.05, 0.5, 'Power (kW)', va='center', rotation='vertical',fontsize=20)
    
    #fig.autofmt_xdate(rotation=45,ha='center') 
    
    fig.legend((r'$P_{\rm AC,meas}$',r'$P_{\rm AC,mod}$'),bbox_to_anchor=(1., 0.58), loc='upper right')
    #fig.legend((r'$P_{\rm AC,meas}$',r'$P_{\rm AC,mod}$'),loc='upper center',bbox_to_anchor=(0.51, 0.45))#loc=[0.43,0.38])
    # hide tick and tick label of the big axes
    
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel("Time (UTC)")
    plt.ylabel("Power (kW)")
        
    fig.subplots_adjust(hspace=0.07,wspace=0.05)
    #fig.tight_layout()
    #fig.autolayout = False
    fig.subplots_adjust(top=0.94)
    plt.savefig(os.path.join(savepath,'chi_sq_fit_grid_' + key + '_' + substat_inv + '_' + res_string_label + 'power.png'))

def save_calibration_results(key,pv_station,substat_inv,config,pv_config,rt_config,path,home):
    """
    Save inversion results to a binary file
    
    args:
    :param key: string, name of current pv station
    :param pv_station: dictionary of one PV system given by key
    :param substat_inv: string, name of substation used for inversion
    :param config: dictionary with main configuration from config file
    :param pv_config: dictionary with inversion configuration for saving    
    :param rt_config: dictionary with disort configuration for saving
    :param path: string, path for saving results    
    :param home: string with homepath
    
    
    """
    
    model = pv_config["inversion"]["power_model"]
    T_model = pv_config["T_model"]
    eff_model = pv_config["eff_model"]
    sza_limit = pv_config["sza_max"]["disort"]
    
    #get description/s
    if len(pv_config["calibration_source"]) > 1:
        infos = '_'.join(pv_config["calibration_source"])
    else:
        infos = pv_config["calibration_source"][0]
        
    atm_source = rt_config["atmosphere"]
    asl_source = rt_config["aerosol"]["source"]
    asl_res = rt_config["aerosol"]["data_res"]
    res = rt_config["disort_rad_res"]    
    
    filename = 'calibration_results_'
    if atm_source == 'cosmo':
        filename = filename + 'atm_'
    if asl_source != 'default':
        filename = filename + 'asl_' + asl_res + '_'
    
    filename = filename + infos + '_disortres_' + str(res["theta"]).replace('.','-')\
                    + '_' + str(res["phi"]).replace('.','-') + '_'
    
    filename_stat = filename + key + '.data'
    with open(os.path.join(path,filename_stat), 'wb') as filehandle:  
        # store the data as binary data stream
        pickle.dump((pv_station, rt_config, pv_config), filehandle)

    print('Results written to file %s\n' % filename_stat)

    #Write to CSV file        
    filename_csv = filename_stat.split('.')[0] + '_' + substat_inv + '.dat'
        
    f = open(os.path.join(path,filename_csv), 'w')
    f.write('#Calibration results for %s, %s\n' % (key,substat_inv))
    f.write('#Data from %s was used for calibration\n' % infos)
    f.write('#Time resolution is %s\n' % rt_config["timeres"])    
    f.write('#Data used up to SZA = %g\n' % sza_limit)
    f.write('#PV model: %s, efficiency model: %s, temperature model: %s\n' % (model,eff_model,T_model))            
            
    ap_pars = pv_station["substations"][substat_inv]["ap_pars"]
    f.write('#A-priori parameters (name,value,error):\n')
    for par in ap_pars:
        f.write('%s, %g, %g\n' % par)              
    if "opt_pars" in pv_station["substations"][substat_inv]:
        opt_pars = pv_station["substations"][substat_inv]["opt_pars"]
        f.write('#Optimisation parameters (name,value,error):\n')
        for par in opt_pars:
            f.write('%s, %g, %g\n' % par)                        
    else:
        f.write('No solution found by the optimisation routine\n')

# f.write('\n#Multi-index: first line ("variable") refers to measured quantity\n')
# f.write('#second line ("substat") refers to measurement device\n')
# f.write('\n')                   
# dataframe.to_csv(f,sep=';',float_format='%.6f', 
#                   na_rep='nan')
    f.close()    
    print('Results written to file %s\n' % filename_csv)

def inversion_setup(key,pv_station,substat_inv,pv_config,resolution,modeltype):
    """
    Setup the various quantities to run the non-linear inversion
    
    args:
    :param key: Code of PV system
    :param pv_station: dictionary with information and data on "key" PV systems
    :param substat_inv: substation to use for inversion
    :param pv_config: dictionary with inversion parameters from config file
    :param resolution: dictionary with resolution of DISORT simulation
    :param modeltype: string, power or current model    
    
    out:
    :return pv_systems: dictionary of PV systems with quantities for inverison added
    :return invdict: dictionary with parameters for inversion
    """      

    invpars = pv_config["inversion"]

    #These are the increments to use for numerical differentiation
    diff_theta = deg2rad(resolution["theta"]/2)
    diff_phi = deg2rad(resolution["phi"]/2)
    diff_n = invpars["n_diff"]

    diffs = collections.namedtuple('diffs', 'theta phi n')
    diffs = diffs(diff_theta,diff_phi,diff_n)

    #Parameters for optimisation
    invdict = {}
    invdict['diffs'] = diffs
    invdict['max_iterations'] = invpars["max_iterations"]
    invdict['gamma'] = invpars["gamma"] #Parameter for Levenberg-Marquardt
    invdict['n_range'] = invpars["n_range"] #Define allowed ranges for parameters    
    invdict['tilt_range'] = invpars["theta_range"]
    invdict['converge'] = invpars["convergence_limit"]
    #invdict["method"] = invpars["method"]    
    #invdict['zeta_ap'] = invpars['zeta_ap']    
    
    #Set up quantities for inversion
    #Apriori values
    apriori = pv_station["substations"][substat_inv]
    
    #Set up list of parameters
    theta_ap = deg2rad(apriori['tilt_ap'][0])
    phi_ap = azi_shift(deg2rad(apriori['azimuth_ap'][0]))
    theta_err = deg2rad(apriori['tilt_ap'][1])
    phi_err = deg2rad(apriori['azimuth_ap'][1])
    
    #This is a list of tuples with
    #(Name,apriori value,apriori error)
    #If apriori error is zero then this parameter will be fixed!
    invdict['pars'] = [('theta',theta_ap,theta_err),
                   ('phi',phi_ap,phi_err)]
    
    #Error in refractive index
    if "n_ap" in apriori:
        invdict['n_ap'] = apriori['n_ap']    
    else:
        invdict['n_ap'] = invpars["n_ap"]
    n_err = invdict['n_ap'][1]*invdict['n_ap'][0]
    invdict['pars'].append(('n',invdict['n_ap'][0],n_err))
    
    opt_dict = {}    
    T_model = pv_config["T_model"]["model"]
    T_model_type = pv_config["T_model"]["type"]
    
    if modeltype == "power":        
        eff_model = pv_config["eff_model"]    
            
        if type(apriori["pdcn_ap"]) == list:
            pdcn_err = apriori['pdcn_ap'][1]*apriori['pdcn_ap'][0]        
            invdict['pars'].append(('pdcn',apriori['pdcn_ap'][0],pdcn_err))
        else:
            pdcn_err = invpars['pdcn_ap_err']*apriori['pdcn_ap']        
            invdict['pars'].append(('pdcn',apriori['pdcn_ap'],pdcn_err))
            
        zeta_err = apriori['zeta_ap'][1]*apriori['zeta_ap'][0]
        invdict['pars'].append(('zeta',apriori['zeta_ap'][0],zeta_err))
                
        #Set up a-priori state vector and covariance matrix
            
        #Efficiency parameters
        if eff_model == "Beyer" or eff_model == "Ransome":
            params_eff_model = invpars["eff_model"][eff_model]
    
            for i, par in enumerate(sorted(params_eff_model)):
                invdict['pars'].append((par.split('_')[0],params_eff_model[par][0],
                       np.abs(params_eff_model[par][1]*params_eff_model[par][0])))                
                
    elif modeltype == "current":
        if "diode_model" in apriori:
            diode_model_aps = apriori["diode_model"]
        else:
            diode_model_aps = pv_config["pv_stations"][key]["diode_model"]
            
        Isc_err = diode_model_aps["Impn"][1]*diode_model_aps["Impn"][0]
        invdict['pars'].append(('impn',diode_model_aps["Impn"][0],Isc_err))
        
        Npp_err = diode_model_aps["Npp"][1]*diode_model_aps["Npp"][0]
        invdict['pars'].append(('npp',diode_model_aps["Npp"][0],Npp_err))

        Ki_err = diode_model_aps["Ki"][1]*diode_model_aps["Ki"][0]
        invdict['pars'].append(('ki',diode_model_aps["Ki"][0],Ki_err))
    
    #Temperature model parameters
    if T_model != "Dynamic_or_Measured":        
        if T_model == "Barry":
            mount_type = pv_config["pv_stations"][key]["substat"][substat_inv]["mount"]
            params_t_model = invpars["temp_model"][T_model][T_model_type][mount_type]
        else:
            params_t_model = invpars["temp_model"][T_model]
        
        for i, par in enumerate(sorted(params_t_model)):
            invdict['pars'].append((par.split('_')[0],params_t_model[par][0],
                   np.abs(params_t_model[par][1]*params_t_model[par][0])))
    
    #Define a-priori state vector
    opt_dict['x_a'] = np.array([invdict['pars'][i][1] for i 
                in range(len(invdict['pars'])) if invdict['pars'][i][2] != 0])
    
    #Define covariance matrix
    opt_dict['S_a'] = np.diag(np.array([invdict['pars'][i][2] for i 
                in range(len(invdict['pars'])) if invdict['pars'][i][2] != 0]))**2        
    
    #Hard constraints
    opt_dict['x_constr'] = [2,invdict['n_range']]
    
    #Define parameter space x
    opt_dict['x'] = np.zeros((invdict['max_iterations'] + 1,len(opt_dict['x_a'])))
    
    pv_station['substations'][substat_inv]['opt'] = opt_dict
            
#    if "p_dc_nom" in apriori:
#        pv_station['df_cal'][('P_meas_norm',substat_inv)] = pv_station['df_cal']\
#        [('P_kW',substat_inv)]/apriori["p_dc_nom"]
        
    return pv_station, invdict

def non_linear_optimisation(model_dict,meas_dict,opt_dict,n_index,angles,const_opt,gamma,
                            invdict,found=False,switch=False):
    """
    

    Parameters
    ----------
    meas_dict : dictionary with models used for inversion
    opt_dict : dictionary with optimisation parameters and results
    n_index : integer, index of refractive index parameter
    gamma : vector of integers for LM parameter gamma        
    invdict : dictionary of inversion parameters
    found : flag for showing whether solution is found or not
        The default is False.
    switch : flag for ending optimisation routine
        The default is False.

    Returns
    -------
    found : flag showing whether solution is found
    opt_dict : dictionary with optimisation parameters and results 
    k : index of solution

    """
    
    #power_model = model_dict["power_model"]
    T_model = model_dict["T_model"]
    eff_model = model_dict["eff_model"]    
    
    x = opt_dict["x"]
    x_a = opt_dict["x_a"]
    S_a = opt_dict["S_a"]
    S_eps = opt_dict["S_eps"]
    chi_sq = opt_dict["chi_sq"]
    chi_sq_retr = opt_dict["chi_sq_retr"]
    delta_chisq = opt_dict["delta_chisq"]
    grad_chisq = opt_dict["grad_chisq"]
    di_sq = opt_dict["di_sq"]
    K_matrix = opt_dict["K_matrix"]

    #Start by using simple Gauss-Newton
    method = 'lm' #'gn' 
    
    if model == "power":
        Ymeas = meas_dict["Pmeas"]
    elif model == "current":
        Ymeas = meas_dict["Imeas"]
    Edirdown_pv = meas_dict["Edirdown_pv"]
    Edirdown_pyr = meas_dict["Edirdown_pyr"]
    diff_field_pv = meas_dict["diff_field_pv"]
    diff_field_pyr = meas_dict["diff_field_pyr"]
    albedo = meas_dict["albedo"]
    TmodC = meas_dict["TmodC"]
    TambC = meas_dict["TambC"]
    vwind = meas_dict["vwind"]
    TskyC = meas_dict["TskyC"]
    sun_pos = meas_dict["sun_pos"]
    n_h2o = meas_dict["n_h2o_mm"]
    spectral_lut = meas_dict["spectral_lut"]
    
    while not switch:
        k = 0           
        if method == 'lm':
            #Set the intial value of gamma    
            gamma[k] = invdict["gamma"]["initial_value"]    
            print('Using the Levenberg-Marquardt method with gamma_0 = %g' % gamma[k])
            print('Convergence criteria is di_sq < %g' % invdict["converge"]["disort"])
        elif method == 'gn':
            print('Using the Gauss-Newton method (gamma = 0)')
            gamma[k] = 0.0
            
        while True:
            #Start at a-priori values
            if k == 0:
                x[k,:] = opt_dict["x_a"]
            
            if model == "power":
                #Calculate modelled power, POA irradiance and dP/dE                                
                dict_Pmodel = P_mod_simple_cal(x[k,:],Edirdown_pv,Edirdown_pyr,diff_field_pv,diff_field_pyr,
                           albedo,n_h2o,spectral_lut,TmodC,TambC,vwind,TskyC,sun_pos,angles,const_opt,invdict,
                         K_matrix,T_model,eff_model)
                #Calculate columns of K matrix
                K_matrix = dict_Pmodel['K_mat']
                Fmod = dict_Pmodel["P_mod"]
            elif model == "current":
                dict_Imodel = I_mod_simple_diode_cal(x[k,:],Edirdown_pv,Edirdown_pyr,diff_field_pv,
                                 diff_field_pyr,albedo,TmodC,TambC,vwind,TskyC,sun_pos,angles,const_opt,invdict,
                         K_matrix,T_model)
                #Calculate columns of K matrix
                K_matrix = dict_Imodel['K_mat']
                Fmod = dict_Imodel["I_mod"]        
            
            
            #Calculate chi squared function for current iteration
            chi_sq[k] = inv.chi_square_fun(x[k,:],x_a,Ymeas,Fmod,S_a,S_eps)
            chi_sq_retr[k] = inv.chi_sq_retrieval(Ymeas,Fmod,K_matrix,S_a,S_eps)
            
            if k < invdict['max_iterations']:
                if k > 0:
                    #Change in chi squared function
                    delta_chisq[k] = chi_sq[k] - chi_sq[k - 1]
                                
                    #Levenberg-Marquardt
                    if method == 'lm':
                        if delta_chisq[k] > 0:
                            gamma[k + 1] = gamma[k]*invdict["gamma"]["factor"]
                            x[k + 1,:] = inv.x_non_lin(x[k,:],x_a,Ymeas,Fmod,
                                                       K_matrix,S_a,S_eps,gamma[k + 1])                    
                        else:
                             #Iterate to find better solution
                            x[k + 1,:] = inv.x_non_lin(x[k,:],x_a,Ymeas,Fmod,
                                                       K_matrix,S_a,S_eps,gamma[k])
                            gamma[k + 1] = gamma[k]/invdict["gamma"]["factor"]
                    else:
                        x[k + 1,:] = inv.x_non_lin(x[k,:],x_a,Ymeas,Fmod,
                                                   K_matrix,S_a,S_eps,gamma[k])
                        gamma[k + 1] = gamma[k]
                        
                    grad_chisq[k,:] = inv.grad_chi_square(x[k,:],x_a,Ymeas,
                                          Fmod,K_matrix,S_a,S_eps)
                    di_sq[k] = inv.d_i_sq(x[k,:],x[k + 1,:],x_a,Ymeas,
                                          Fmod,K_matrix,S_a,S_eps)
    
                else:
                    #First iteration
                    x[k + 1,:] = inv.x_non_lin(x[k,:],x_a,Ymeas,Fmod,
                                               K_matrix,S_a,S_eps,gamma[k])
                    delta_chisq[k] = nan
                    di_sq[k] = nan
                    gamma[k + 1] = gamma[k]
                
                #Write out parameters during iteration steps
                if np.fmod(k,10) == 0 and k > 0:
                    print('%d: x: %s chi_sq: %.6f, delta chi_sq: %.6f, d_i_sq: %.6f'\
                          % (k,x[k,:],chi_sq[k],delta_chisq[k],di_sq[k]))                        
                
                #Check that variables lie within the constraints, (only for n)
                if invdict['pars'][2][2] != 0:
                    if x[k + 1,n_index] < invdict['n_range'][0]:
                        x[k + 1,n_index] = invdict['n_range'][0]                 
                    if x[k + 1,n_index] > invdict['n_range'][1]:
                        x[k + 1,n_index] = invdict['n_range'][1]
                        
#                if T_model == "Tamizhmani":
#                    if invdict['pars'][6][2] != 0:
#                        if x[k + 1,u1_index] < 0:
#                            x[k + 1,u1_index] = 0.01                                         
                
                #Check whether minimum has been found
                if k > 0:
                    #Check whether iteration converges using di_sq from Rodgers
                    if di_sq[k] > invdict["converge"]["disort"] or di_sq[k] < 0.0: # or abs(delta_chisq[k]) > 0.1:
                        if k == len(chi_sq) - 1:
                            print('Minimum not found within tolerance')
                            break
                        else:
                            #Continue to next iteration
                            k = k + 1
                            continue
                    else:
                        #If di_sq small enough, check to see whether current 
                        #iteration gives the smallest chi-squared
                        if k > 1:
                            if chi_sq[k] < np.min(chi_sq[0:k-1]):
                                x_min = x[k,:]
                                min_chisq = chi_sq[k]
                                min_dsq = di_sq[k]
                                min_ix = k
                            else:
                                min_chisq = np.min(chi_sq[0:k-1])
                                min_ix = np.argmin(chi_sq[0:k-1])
                                x_min = x[min_ix,:]
                                min_dsq = di_sq[min_ix]
                        else:
                            x_min = x[k,:]
                            min_chisq = chi_sq[k]
                            min_dsq = di_sq[k]
                            min_ix = k
                            
                        found = True
                        switch = True
                        
                        #Calculate further quantities for the solution
                        S_hat = inv.S_post(S_a,K_matrix,S_eps)                        
                        A = inv.A_kernel(S_a,K_matrix,S_eps)
                        d_s = np.trace(A)
                        info = inv.H_info(A)
                        eigs_S_hat, L_S_hat = np.linalg.eig(S_hat)
                        S_dely = inv.S_del_y(K_matrix,S_a,S_eps)
                        s_vec = np.sqrt(np.diag(S_hat))  
                                                     
                        c_par = 0
                        for par in invdict["pars"]:
                            if par[2] != 0:
                                if par[0] == "theta":
                                    value = rad2deg(x_min[c_par])
                                elif par[0] == "phi":
                                    value = rad2deg(azi_shift(x_min[c_par]))
                                else:
                                    value = x_min[c_par]
                                print('%s: %g' % (par[0],value))
                                c_par = c_par + 1
                            else:
                                if par[0] == "theta":
                                    value = rad2deg(par[1])
                                elif par[0] == "phi":
                                    value = rad2deg(azi_shift(par[1]))
                                else:
                                    value = par[1]
                                print('%s: %g (fixed)' % (par[0],value))
                            
                        break
                else:
                    k = k + 1
            else:                
                if method == "gn":
                    print(f'Could not converge to a solution after {k} iterations')
                    print('Trying with Levenberg-Marquardt method')      
                    method = "lm"
                    
                elif method == "lm":
                    print(f'Could not converge to a solution after {k} iterations')
                    x_min = nan
                    min_ix = nan
                    min_chisq = nan
                    min_dsq = nan
                    d_s = nan
                    S_hat = nan
                    s_vec = nan                    
                    A = nan
                    info = nan
                    S_dely = nan
                    switch = True
                    
                break
    
    #Save optimisation parameters to dictionary
    opt_dict['x_min'] = x_min
    opt_dict["min_ix"] = min_ix
    opt_dict['chi_sq'] = chi_sq
    opt_dict['min_chisq'] = min_chisq
    opt_dict['delta_chisq'] = delta_chisq
    opt_dict['K_mat'] = K_matrix
    opt_dict['S_eps'] = S_eps
    opt_dict['S_hat'] = S_hat
    opt_dict['s_vec'] = s_vec      
    opt_dict['chi_sq_retr'] = chi_sq_retr
    opt_dict['A'] = A
    opt_dict['d_s'] = d_s
    opt_dict['info'] = info
    opt_dict['S_dely'] = S_dely
    opt_dict['d_isq'] = di_sq
    opt_dict["min_dsq"] = min_dsq
    opt_dict['gamma'] = gamma
    
    return found, opt_dict, k

def pv_calibration(key,pv_station,substat_inv,invdict,const_opt,angles,
                         plot_styles,homepath,rt_config,pv_config,model):
    """
    Perform non-linear inversion to calibrate PV system
    
    args:
    :param key: 
    :param pv_station: dictionary with PV systems, info and dataframe
    :param substat_inv: name of substation to use for inversion
    :param invdict: dictionary with parameters for inversion procedure
    :param const_opt: named tuple with optical constants
    :param angles: named tuple with angle grid for DISORT
    :param plot_styles: dictionary with plot styles
    :param homepath: string with home path
    :param rt_config: dictionary with radiative transfer configuration
    :param pv_config: dictionary with calibration configuration 
    :param model: string with name of model used
    
    out:
    :return pv_station: dictionary with info and data from PV system
    :return folder_label: string with folder label
    """
    sun_position = collections.namedtuple('sun_position','sza azimuth')   
    
    model_dict = {}
    model_dict["cal_model"] = model
    model_dict["power_model"] = pv_config["inversion"]["power_model"]
    model_dict["T_model"] = pv_config["T_model"]["model"]
    model_dict["T_model_type"] = pv_config["T_model"]["type"]
    model_dict["eff_model"] = pv_config["eff_model"]
    sza_limit = pv_config["sza_max"]["disort"]
                
    #Get data source from config file
    data_source = pv_config["pv_stations"][key]["input_data"]
    
    #Get a-priori measurement error
    ap_substat = pv_station["substations"][substat_inv]    
    
    if model == "current":
        if "diode_model" in ap_substat:
            diode_model_aps = ap_substat["diode_model"]
        else:
            diode_model_aps = pv_config["pv_stations"][key]["diode_model"]
    
    #Assign a-priori parameters to station dictionary
    pv_station['substations'][substat_inv]['ap_pars'] = invdict['pars']
    
    #Get index of values
    df_cal_notnan = prepare_calibration_data(key,pv_station['df_cal'],data_source,
                                          pv_station['cal_source'],substat_inv,
                                          model_dict["T_model"],ap_substat,model)        
    
    #Use only values up to defined SZA limit    
    sza_index = df_cal_notnan.loc[df_cal_notnan['theta0rad'] 
    <= np.deg2rad(sza_limit)].index  
    
    #Create list for each substation
    pv_station['substations'][substat_inv]['cal_days'] = \
    pd.to_datetime(df_cal_notnan.index.date).unique().strftime('%Y-%m-%d')    
    
    #This is the new index of valid values up to SZA limit
    new_index = sza_index.intersection(df_cal_notnan.index)
    
    #This is the final dataframe for inversion    
    dataframe = df_cal_notnan.loc[new_index]        
    
    #Optimisation parameters
    opt_dict = pv_station['substations'][substat_inv]['opt']
    
    #x_constr = opt_dict['x_constr']    
    opt_dict["chi_sq"] = np.zeros(invdict['max_iterations'] + 1)
    opt_dict["chi_sq_retr"] = np.zeros(invdict['max_iterations'] + 1)
    opt_dict["delta_chisq"] = np.zeros(invdict['max_iterations'])
    opt_dict["grad_chisq"] = np.zeros((invdict['max_iterations'],len(opt_dict["x_a"])))
    opt_dict["di_sq"] = np.zeros(invdict['max_iterations'] + 1)
    gamma = np.zeros(invdict['max_iterations'] + 1)
            
    #Convert all pandas series into numpy arrays for calculation speed
    meas_dict = {}
    meas_dict["Edirdown_pv"] = dataframe.Edirdown_pv.values.flatten()
    meas_dict["Edirdown_pyr"] = dataframe.Edirdown_pyr.values.flatten()
    meas_dict["sun_pos"] = sun_position(dataframe.theta0rad.values.flatten(),
                           dataframe.phi0rad.values.flatten())        
    meas_dict["albedo"] = dataframe.albedo.values.flatten()
    meas_dict["n_h2o_mm"] = dataframe.n_h2o_mm.values.flatten()
    
    meas_dict["spectral_lut"] = pv_station["df_ghi_spectral_mismatch"]
    
    #Measurement covariance matrix
    if model == "power":
        meas_dict["Pmeas"] = dataframe.P_meas_W.values.flatten()
        meas_dict["P_error_meas"] = dataframe.P_error_meas_W.values.flatten()
        opt_dict["S_eps"] = np.identity(len(new_index))*meas_dict["P_error_meas"]**2 # Measurement error in W^2
    elif model =="current":
        meas_dict["Imeas"] = dataframe.I_meas_A.values.flatten()
        meas_dict["I_error_meas"] = dataframe.I_error_meas_A.values.flatten()
        opt_dict["S_eps"] = np.identity(len(new_index))*meas_dict["I_error_meas"]**2 # Measurement error in W^2
    
    #Extract the different values for temperature model
    if model_dict["T_model"] == "Dynamic_or_Measured":
        meas_dict["TmodC"] = dataframe.T_module_C.values.flatten()
        meas_dict["TambC"] = np.nan*np.ones(len(dataframe))
        meas_dict["vwind"] = np.nan*np.ones(len(dataframe))
        meas_dict["TskyC"] = np.nan*np.ones(len(dataframe))
    else:
        meas_dict["TmodC"] = np.nan*np.ones(len(dataframe))
        meas_dict["TambC"] = dataframe.T_ambient_C.values.flatten()
        meas_dict["vwind"] = dataframe.v_wind_ms.values.flatten()
        meas_dict["TskyC"] = dataframe.T_sky_C.values.flatten()
        
    #Get the diffuse radiance field
    meas_dict["diff_field_pv"] = np.zeros((len(new_index),len(angles.theta),len(angles.phi)))                
    meas_dict["diff_field_pyr"] = np.zeros((len(new_index),len(angles.theta),len(angles.phi)))                
    for i, itime in enumerate(new_index):
        meas_dict["diff_field_pv"][i,:,:] = dataframe.loc[itime,'Idiff_pv'].values #.flatten()
        meas_dict["diff_field_pyr"][i,:,:] = dataframe.loc[itime,'Idiff_pyr'].values #.flatten()
    
    #Define Jacobian matrix
    opt_dict["K_matrix"] = np.zeros((len(new_index),len(opt_dict["x_a"])))
    
    print('\n%s, %s calibration, clear sky days are: %s' % (key,substat_inv,[day for day in 
                            pv_station['substations'][substat_inv]["cal_days"]]))
    if model == "power":
        print('Running power calibration with %s, %s, %s model for %s, substation %s for SZA_max = %d and a-priori values:' 
          % (model_dict["power_model"],model_dict["eff_model"],model_dict["T_model"],
             key,substat_inv,sza_limit))
    elif model == "current":
        print('Running photocurrent calibration with %s temperature model for %s, substation %s for SZA_max = %d and a-priori values:' 
          % (model_dict["T_model"],
             key,substat_inv,sza_limit))
        
    print('theta: %s, phi: %s n: %s'
          % (ap_substat["tilt_ap"],ap_substat["azimuth_ap"],inv_dict["n_ap"]))
    if model == "power":
        print('Pdcn: %s, zeta: %s'
          % (ap_substat["pdcn_ap"],ap_substat["zeta_ap"]))
        print('Error on power measurement is %s (relative) and at least %s W'
          % (ap_substat["p_err_rel"],ap_substat["p_err_min"]))
    elif model == "current":
        print('Impn: [%s,%s], Npp: [%s,%s], Ki: [%s,%s]'
              % (diode_model_aps["Impn"][0],diode_model_aps["Impn"][1],
                 diode_model_aps["Npp"][0],diode_model_aps["Npp"][1],
                 diode_model_aps["Ki"][0],diode_model_aps["Ki"][1]))
        print('Error on current measurement is %s (relative) and at least %s A'
          % (ap_substat["i_err_rel"],ap_substat["i_err_min"]))
    
        
    #Print out apriori values
    if model_dict["eff_model"] != "Evans":
        for par,value in sorted(pv_config["inversion"]["eff_model"][model_dict["eff_model"]].items()):
            print("{}: {}".format(par,value))            
    
    if model_dict["T_model"] != "Dynamic_or_Measured":
        if model_dict["T_model"] == "Barry":
            mount_type = pv_config["pv_stations"][key]["substat"][substat]["mount"]
            t_mod_pars = pv_config["inversion"]["temp_model"][model_dict["T_model"]]\
            [model_dict["T_model_type"]][mount_type]
        else:
            t_mod_pars = pv_config["inversion"]["temp_model"][model_dict["T_model"]]
            
        for par,value in sorted(t_mod_pars.items()):
            print("{}: {}".format(par,value))            
        
    print("Input data:")
    for data_type in data_source:
        for mk in data_source[data_type]:
            print(f'{data_type}, {mk}: {data_source[data_type][mk]}')
    #Get limits for n, find the index for n
    n_index = 0    
    for i, par in enumerate(invdict['pars']):
        if par[0] != 'n':
            if par[2] != 0:
                n_index = n_index + 1
        else:
            break        
    u1_index = 0
    for i, par in enumerate(invdict['pars']):
        if par[0] != 'u1':
            if par[2] != 0:
                u1_index = u1_index + 1
        else:
            break        

    #Perform non-linear optimisation            
    found, opt_dict, k = non_linear_optimisation(model_dict,meas_dict,opt_dict,n_index,angles,
                                                 const_opt,gamma,invdict)
       
    #Calculate power, irradiance etc at the solution x_min
    if found:
        print('There were %d measurements used, retrieval chi_sq is %.3f' % (len(opt_dict["S_eps"]),opt_dict["chi_sq_retr"][k]))
        print('MAP Solution for %s, %s after %d iterations: Chi_sq: %.6f, delta Chi_sq: %.6f, di_sq: %.6f, chi_sq_ret: %.6f'
              % (key,substat_inv,opt_dict["min_ix"], opt_dict["min_chisq"], opt_dict["delta_chisq"][k], 
                 opt_dict["min_dsq"], opt_dict["chi_sq_retr"][k]))
        print('Degrees of freedom, d_s: %.3f out of %d parameters' % (opt_dict["d_s"],len(opt_dict["x_a"])))
                                        
        #Go back to the entire dataframe (all SZA from simulation)
        Edirdown_pv = df_cal_notnan.Edirdown_pv.values.flatten()
        Edirdown_pyr = df_cal_notnan.Edirdown_pyr.values.flatten()
        sun_pos = sun_position(df_cal_notnan.theta0rad.values.flatten(),
                           df_cal_notnan.phi0rad.values.flatten())

        if model == "power":
            Pmeas = df_cal_notnan.P_meas_W.values.flatten()
            P_error_meas = df_cal_notnan.P_error_meas_W.values.flatten()
        elif model == "current":
            Imeas = df_cal_notnan.I_meas_A.values.flatten()
            I_error_meas = df_cal_notnan.I_error_meas_A.values.flatten()
            
        alb = df_cal_notnan.albedo.values.flatten()
        n_h2o = df_cal_notnan.n_h2o_mm.values.flatten()
    
        if model_dict["T_model"] == "Dynamic_or_Measured":
            TmodC = df_cal_notnan.T_module_C.values.flatten()
            TambC = np.nan*np.ones(len(df_cal_notnan))
            vwind = np.nan*np.ones(len(df_cal_notnan))
            TskyC = np.nan*np.ones(len(df_cal_notnan))
        else:
            TmodC = np.nan*np.ones(len(df_cal_notnan))
            TambC = df_cal_notnan.T_ambient_C.values.flatten()
            vwind = df_cal_notnan.v_wind_ms.values.flatten()
            TskyC = df_cal_notnan.T_sky_C.values.flatten()

        diff_field_pv = np.zeros((len(df_cal_notnan),len(angles.theta),len(angles.phi)))                
        diff_field_pyr = np.zeros((len(df_cal_notnan),len(angles.theta),len(angles.phi)))                
        for i, itime in enumerate(df_cal_notnan.index):
            diff_field_pv[i,:,:] = df_cal_notnan.loc[itime,'Idiff_pv'].values
            diff_field_pyr[i,:,:] = df_cal_notnan.loc[itime,'Idiff_pyr'].values
        
        K_matrix = np.zeros((len(df_cal_notnan.index),len(opt_dict["x_a"])))

        if model == "power":
            model_solution = \
                P_mod_simple_cal(opt_dict["x_min"],Edirdown_pv,Edirdown_pyr,diff_field_pv,
                                 diff_field_pyr,alb,n_h2o,meas_dict["spectral_lut"],
                 TmodC,TambC,vwind,TskyC,sun_pos,angles,const_opt,invdict,K_matrix,model_dict["T_model"],
                model_dict["eff_model"])
                
            #Solution for whole SZA range
            df_cal_notnan['P_MAP'] = model_solution['P_mod']
            
            #Extra part of solution (include endpoints for pretty plots)
            df_cal_notnan['P_MAP_extra'] = df_cal_notnan['P_MAP']
            sza_index_extra = df_cal_notnan.loc[df_cal_notnan['theta0rad'] < 
                                                np.deg2rad(sza_limit - 5.0)].index    
            df_cal_notnan.loc[sza_index_extra,'P_MAP_extra'] = np.nan
            
            #Real part of solution (up to SZA limit)
            df_cal_notnan['P_MAP'] = df_cal_notnan.loc[sza_index,'P_MAP']\
                                                .reindex(df_cal_notnan.index)
                                                
            df_cal_notnan['eff_temp_inv'] = model_solution['eff_temp']
            
        if model == "current":
            model_solution = \
                I_mod_simple_diode_cal(opt_dict["x_min"],Edirdown_pv,Edirdown_pyr,
               diff_field_pv,diff_field_pyr,alb,TmodC,TambC,vwind,TskyC,sun_pos,
               angles,const_opt,invdict,K_matrix,model_dict["T_model"])
                
            #Solution for whole SZA range
            df_cal_notnan['I_MAP'] = model_solution['I_mod']
            
            #Extra part of solution (include endpoints for pretty plots)
            df_cal_notnan['I_MAP_extra'] = df_cal_notnan['I_MAP']
            sza_index_extra = df_cal_notnan.loc[df_cal_notnan['theta0rad'] < 
                                                np.deg2rad(sza_limit - 5.0)].index    
            df_cal_notnan.loc[sza_index_extra,'I_MAP_extra'] = np.nan
            
            #Real part of solution (up to SZA limit)
            df_cal_notnan['I_MAP'] = df_cal_notnan.loc[sza_index,'I_MAP']\
                                                .reindex(df_cal_notnan.index)
        
        #Optimal parameters - both those varied and fixed!
        optpars = []    
        c_par = 0
        for i, par in enumerate(invdict['pars']):
            if par[2] != 0:
                optpars.append((par[0],model_solution["pars"][i],opt_dict["s_vec"][c_par]))
                c_par = c_par + 1
            else:
                optpars.append(par)
                        
        pv_station['substations'][substat_inv]['opt_pars'] = optpars
        
        df_cal_notnan['Etotpoa'] = model_solution['Etotpoa']
        df_cal_notnan['Edirpoa'] = model_solution['Edirpoa']
        df_cal_notnan['Ediffpoa'] = model_solution['Ediffpoa']
        df_cal_notnan['Etotpoa_pv'] = model_solution['Etotpoa_pv']
        df_cal_notnan['Edirpoa_pv'] = model_solution['Edirpoa_pv']
        df_cal_notnan['Ediffpoa_pv'] = model_solution['Ediffpoa_pv']
        #df_cal_notnan['Ereflpoa'] = model_solution['Ereflpoa']
        df_cal_notnan['T_module_inv_C'] = model_solution['T_module']        

        
        df_cal_notnan[f'error_{model}_fit'] = np.nan
        df_cal_notnan.loc[new_index,f'error_{model}_fit'] = np.sqrt(np.diagonal(opt_dict['S_dely']))
        
        df_cal_notnan.columns = pd.MultiIndex.from_product([df_cal_notnan.columns.
                                values.tolist(),[substat_inv]],names=['variable','substat']) 
        
        #Puts the inversion results back into the dataframe (keep all days)
        pv_station['substations'][substat_inv]['df_cal'] = df_cal_notnan
    
    #Plot results for key
    print('Plotting results for %s, %s\n' % (key,substat_inv))
    folder_label = plot_fit_results(key,pv_station,substat_inv,rt_config,pv_config,
                                    plot_styles,homepath,model_dict,found,
                                    invdict["pars"])    
    
    return pv_station, folder_label
    
#%%Main Program
#######################################################################
###                         MAIN PROGRAM                            ###
#######################################################################
#def main():
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f","--configfile", help="yaml file containing config")
parser.add_argument("-s","--station",nargs='+',help="station for which to perform inversion")
parser.add_argument("-c","--campaign",nargs='+',help="measurement campaign for which to perform simulation")
args = parser.parse_args()

#Main configuration file
if args.configfile:
    config_filename = os.path.abspath(args.configfile) #"config_PYRCAL_2018_messkampagne.yaml" #
else:
    config_filename = "config_PVCAL_MetPVNet_messkampagne.yaml" #os.path.abspath(args.configfile)2021_solarwatt.yaml
 
config = load_yaml_configfile(config_filename)

#Load PV configuration
pv_config = load_yaml_configfile(config["pv_configfile"])

#Load radiative transfer configuration
rt_config = load_yaml_configfile(config["rt_configfile"])

homepath = os.path.expanduser('~') #"/media/luke/" #

if args.station:
    stations = args.station
    if stations[0] == 'all':
        stations = 'all'
else:
    #Stations for which to perform inversion
    stations = "PV_12" #pv_config["stations"]

if args.campaign:
    pv_config["calibration_source"] = args.campaign

print('\nPV calibration with Bayesian inversion for %s stations' % stations)
print('Using data from %s' % pv_config["calibration_source"])
print('Loading results from radiative transfer simulation')
if rt_config["disort_base"]["pseudospherical"]:
    print('Results are with pseudospherical atmosphere')
else:
    print('Results are with plane-parallel atmosphere')
print('Wavelength range is set to %s nm' % rt_config["common_base"]["wavelength"]["pv"])
print('Molecular absorption is calculated using %s' % rt_config["common_base"]["mol_abs_param"])

pvsys = load_data_radsim_results(config,pv_config,rt_config,stations,homepath)

sza_max = pv_config["sza_max"]["disort"]   

disort_res = rt_config["disort_rad_res"]
grid_dict = define_disort_grid(disort_res)

optics_dict = pv_config["optics"]
optics = collections.namedtuple('optics', 'kappa L')
const_opt = optics(optics_dict["kappa"],optics_dict["L"])

angle_grid = collections.namedtuple('angle_grid', 'theta phi umu')
angle_arrays = angle_grid(grid_dict["theta"],deg2rad(grid_dict["phi"]),grid_dict["umu"])

for key in pvsys:   
    print(f'Performing PV calibration for {key}')    
    #Prepare surface inputs
    if pv_config["surface_data"] == "cosmo":
        pvsys[key] = prepare_surface_data(pv_config,key,pvsys[key],rt_config["test_days"],homepath)
        
    #Calculate precipitable water
    if rt_config["atmosphere"] == "cosmo":
        pvsys[key] = prepare_water_vapour(rt_config,key,pvsys[key],rt_config["test_days"],homepath)                

    #Load temperature model results and model temperature if no measurement is available        
    pvsys[key] = load_temp_model_results(key,pvsys[key],pv_config,homepath)
    
    if pv_config["longwave_data"]:        
        pvsys[key] = prepare_longwave_data(key, pvsys[key], rt_config["timeres"],
                                           pv_config,homepath)
    
    if pv_config["T_model"]["model"] == "Dynamic_or_Measured":  
        pvsys[key] = calculate_temp_module(key,pvsys[key],pv_config,
                                           rt_config["timeres"],homepath)                      
    
    #Load spectral mismatch lookup table
    pvsys[key]["df_ghi_spectral_mismatch"] = load_spectral_mismatch_fit(pv_config, homepath)
    
    #Define days to use for inversion (the same for all substations at a particular station), extract
    #relevant days from the simulation dataframe and rename it df_cal
    #Later we will check if each of these days has data for each substation
    pvsys[key], flag, pvsys[key]['cal_days'] = select_days_calibration(key,pvsys[key],pv_config)

    #Get substation for inversion            
    pvsys[key]['substations'] = pv_config["pv_stations"][key]["substat"]        
        
    for substat in pvsys[key]['substations']:        
        for model in pvsys[key]['substations'][substat]['model']:        
            #Set up inversion depending on the model chosen
            pvsys[key], inv_dict = inversion_setup(key,pvsys[key],substat,
                 pv_config,disort_res,model)    
            
            #Perform inversion
            pvsys[key], savepath = pv_calibration(key,pvsys[key],substat,
                 inv_dict,const_opt,angle_arrays,config["plot_styles"],
                 homepath,rt_config,pv_config,model)
        
            # Save solution for key to file
            save_calibration_results(key,pvsys[key],substat,config,pv_config,rt_config,savepath,homepath)
    
#if __name__ == "__main__":
#    main()
