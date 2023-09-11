#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 09:56:29 2021

@author: james
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import ephem
from copy import deepcopy

from file_handling_functions import *
from plotting_functions import confidence_band
from data_process_functions import downsample
import pickle
import subprocess
from matplotlib.gridspec import GridSpec
from pvcal_forward_model import azi_shift
from scipy.stats import gaussian_kde
import seaborn as sns

#%%Functions

def generate_folder_names_pvpyr2od(rt_config,pvcal_config,od_type):
    """
    Generate folder structure to retrieve PYR2CF simulation results
    
    args:    
    :param rt_config: dictionary with RT configuration    
    :param pvcal_config: dictionary with PV calibration configuration
    :param od_type: string with OD type - AOD or COD
    
    out:
    :return folder_label: string with complete folder path
    """        
    
    #geometry model
    atm_geom_config = rt_config["disort_base"]["pseudospherical"]
    if atm_geom_config == True:
        atm_geom_folder = "Pseudospherical"
    else:
        atm_geom_folder = "Plane-parallel"    
    
    #Get wavelength folder label
    wvl_config = rt_config["common_base"]["wavelength"]["pyranometer"]
    if type(wvl_config) == list:
        wvl_folder_label = "Wvl_" + str(int(wvl_config[0])) + "_" + str(int(wvl_config[1]))
    elif wvl_config == "all":
        wvl_folder_label = "Wvl_Full"
    
    #Get DISORT resolution folder label
    if od_type == "cod":
        disort_config = rt_config["clouds"]["disort_rad_res"]   
    elif od_type == "aod":
        disort_config = rt_config["aerosol"]["disort_rad_res"]   
        
    theta_res = str(disort_config["theta"]).replace('.','-')
    phi_res = str(disort_config["phi"]).replace('.','-')
    
    disort_folder_label = 'Zen_' + theta_res + '_Azi_' + phi_res
        
    if rt_config["atmosphere"] == "default":
        atm_folder_label = "Atmos_Default"    
    elif rt_config["atmosphere"] == "cosmo":
        atm_folder_label = "Atmos_COSMO"        
        
    if rt_config["aerosol"]["source"] == "default":
        aero_folder_label = "Aerosol_Default"
    elif rt_config["aerosol"]["source"] == "aeronet":
        aero_folder_label = "Aeronet_" + rt_config["aerosol"]["station"]
            
    sza_label = "SZA_" + str(int(rt_config["sza_max"]["lut"]))
    
    
    model = pvcal_config["inversion"]["power_model"]
    eff_model = pvcal_config["eff_model"]
    
    T_model = pvcal_config["T_model"]["model"]

    folder_label = os.path.join(atm_geom_folder,wvl_folder_label,disort_folder_label,
                                atm_folder_label,aero_folder_label,sza_label,
                                model,eff_model,T_model)
        
    return folder_label

def generate_folder_names_pvpyr2ghi(rt_config,pvcal_config):
    """
    Generate folder structure to retrieve GHI results from LUT
    
    args:    
    :param rt_config: dictionary with RT configuration    
    :param pvcal_config: dictionary with PV calibration configuration

    out:
    :return folder_label: string with complete folder path
    """        
    
    #geometry model
    atm_geom_config = rt_config["disort_base"]["pseudospherical"]
    if atm_geom_config == True:
        atm_geom_folder = "Pseudospherical"
    else:
        atm_geom_folder = "Plane-parallel"    
        
    disort_config = rt_config["clouds"]["disort_rad_res"]   
    theta_res = str(disort_config["theta"]).replace('.','-')
    phi_res = str(disort_config["phi"]).replace('.','-')
    
    disort_folder_label = 'Zen_' + theta_res + '_Azi_' + phi_res
        
    if rt_config["atmosphere"] == "default":
        atm_folder_label = "Atmos_Default"    
    elif rt_config["atmosphere"] == "cosmo":
        atm_folder_label = "Atmos_COSMO"        
        
    if rt_config["aerosol"]["source"] == "default":
        aero_folder_label = "Aerosol_Default"
    elif rt_config["aerosol"]["source"] == "aeronet":
        aero_folder_label = "Aeronet_" + rt_config["aerosol"]["station"]
            
    sza_label = "SZA_" + str(int(rt_config["sza_max"]["lut"]))

    model = pvcal_config["inversion"]["power_model"]
    eff_model = pvcal_config["eff_model"]
    
    T_model = pvcal_config["T_model"]["model"]    

    folder_label = os.path.join(atm_geom_folder,disort_folder_label,
                                atm_folder_label,aero_folder_label,sza_label,
                                model,eff_model,T_model)
        
    return folder_label

def load_resampled_data(station,timeres,measurement,config,home):
    """
    Load data that has already been resampled to a specified time resolution
    
    args:    
    :param station: string, name of PV station to load
    :param timeres: string, timeresolution of the data
    :param measurements: string, description of measurement campaign
    :param config: dictionary with paths for loading data
    :param home: string, homepath    
    
    out:
    :return pv_stat: dictionary of PV station dataframes and other information    
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

def load_pvpyr2od_fit_ghilut_results(rt_config, pyr_config, pvcal_config, pvrad_config, 
                                  info, station_list, od_types, home):
    """
    
    args:    
    :param rt_config: dictionary with current RT configuration
    :param pyr_config: dictionary with current RT configuration
    :param pvcal_config: dictionary with current calibration configuration    
    :param pvrad_config: dictionry with pv2rad configuration
    :param info: string with name of campaign
    :param station_list: list of stations
    :param od_types: list of OD types
    :param home: string with homepath

    Returns
    -------
    pv_systems: dictionary with PV systems and all information / data
    station_list: list of stations that contain data

    """
    
    results_path_od = os.path.join(home,pyr_config["results_path"]["main"],
                                pyr_config["results_path"]["optical_depth"])    
    results_path_irrad = os.path.join(home,pyr_config["results_path"]["main"],
                                pyr_config["results_path"]["irradiance"]) 
        
    timeres = pyr_config["t_res_inversion"]
    
    year = info.split('_')[1]
    
    filename_lut = f"ghi_lut_results_{info}"
    
    #Define new dictionary to contain all information, data etc
    pv_systems = {}    
        
    #Choose which stations to load    
    if type(station_list) != list:
        station_list = [station_list]
        if station_list[0] == "all":
            station_list = list(pvrad_config["pv_stations"].keys())
            # station_list.extend(list(pvrad_config["pv_stations"].keys()))
            # station_list = list(set(station_list))
            # station_list.sort()     
    
    for station in station_list:                    
        for od in od_types:
            folder_label = generate_folder_names_pvpyr2od(rt_config,pvcal_config,od)
            for t_res in timeres:                
                filename = f"{od}_fit_results_{info}_"
        
                #Read in binary file that was saved from pvcal_radsim_disort
                filename_stat = filename + station + '_' + f'{t_res}.data'
                try:
                    with open(os.path.join(results_path_od,folder_label,filename_stat), 'rb') as filehandle:  
                        # read the data as binary data stream
                        pvstat = pd.read_pickle(filehandle)                        
                        
                    if station in pv_systems:
                        if type(pv_systems[station]["timeres"]) != list:
                            pv_systems[station]["timeres"] = [pv_systems[station]["timeres"]]
                        if pvstat["timeres"][0] not in pv_systems[station]["timeres"]:
                            pv_systems[station]["timeres"].extend(pvstat["timeres"])
                            
                        pv_systems[station] = merge_two_dicts(pvstat, pv_systems[station])
                    else:
                        pv_systems.update({station:pvstat})

                    print('Loaded data for %s %s retrieval, at %s for %s' % (od,t_res,station,year))
                except IOError:
                    print(f'There are no {od} results at {t_res} resolution for %s in {year}' % station)                   
                
        pvstat_raw = load_resampled_data(station,"raw",info,data_config,homepath)            
        if pvstat_raw:
            print(f'Loaded original measurement data for {station}, {info}')
            pv_systems[station] = merge_two_dicts(pv_systems[station],pvstat_raw)
        else:
            print('No original data found')
            
        folder_label = generate_folder_names_pvpyr2ghi(rt_config,pvcal_config)                
        filename_stat = f"{filename_lut}_{station}.data"
        
        try:
            with open(os.path.join(results_path_irrad,folder_label,filename_stat), 'rb') as filehandle:  
                # read the data as binary data stream
                (pvstat, dummy, dummy, dummy) = pd.read_pickle(filehandle)                        
                
            if pv_systems:     
                if type(pv_systems[station]["timeres"]) != list:
                    pv_systems[station]["timeres"] = [pv_systems[station]["timeres"]]
                for t_res in pvstat["timeres"]:
                    if t_res not in pv_systems[station]["timeres"]:
                        pv_systems[station]["timeres"].extend([t_res])
                pv_systems[station] = merge_two_dicts(pvstat, pv_systems[station])
            else:
                pv_systems.update({station:pvstat})      
                
            print('Loaded inverted GHI via LUT data for %s in %s\n' % (station,year))
        except IOError:
            print('There are no LUT irradiance results for %s\n' % station)     
                                
    return pv_systems, station_list

def get_sun_position(dataframe,lat_lon,sza_limit):
    """
    Using PyEphem to calculate the sun position
    
    args:    
    :param dataframe: pandas dataframe for which to calculate sun position
    :param lat_lon: list of floats, coordinates
    :param sza_limit: float defining maximum solar zenith angle for simulation
    
    out:
    :return: dataframe with sun position
    
    """        
    
    len_time = len(dataframe)
    index_time = dataframe.index

    # initalize observer object
    observer = ephem.Observer()
    observer.lat = np.deg2rad(lat_lon[0])
    observer.lon = np.deg2rad(lat_lon[1])

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
    if sza_limit:    
        dataframe = dataframe.loc[dataframe[('sza','sun')] <= sza_limit]        
        
    return dataframe

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
    :return pv_station: dictionary with PV station info updated with path to cosmo atmfiles
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
                    
                print(f"Nearest COSMO grid point to {key} is at {folder[4:9]}, {folder[14:19]}")

    return pv_station
 
def import_cosmo_irrad_data(key,pv_station,year):
    """
    Import irradiance data from cosmo2pvcal
    
    args:
    :param key: string, name of station
    :param pv_station: dictionary with all info and data of pv station    
    :param year: string describing the data source
    
    out:
    :return pv_station: dictionary with PV station data, including COSMO irradiance data
    """
    
    dfs = [pd.read_csv(os.path.join(pv_station['path_cosmo_irrad'],filename),sep='\s+',index_col=0,skiprows=2,
            header=None,names=['Edirdown_Wm2','Edirdown_mean_Wm2','Edirdown_iqr_Wm2','Ediffdown_Wm2',
                               'Ediffdown_mean_Wm2','Ediffdown_iqr_Wm2']) 
           for filename in list_files(pv_station['path_cosmo_irrad']) if '_irradiance.dat' 
           in filename and year.split('_')[1] in filename]
     
    dataframe = pd.concat(dfs,axis=0)
    
    #Shift data to the left, since COSMO irradiance is integrated over the last hour
    dataframe.index = pd.to_datetime(dataframe.index,format='%d.%m.%Y;%H:%M:%S')
    
    #Create new index to take averaging into account
    shifted_index = dataframe.index - pd.Timedelta('30T')
    df_shifted = dataframe.reindex(shifted_index,method='bfill')
        
    #Resample data at double frequency and linearly interpolate
    df_rs = df_shifted.resample('30T').interpolate('linear')
                            
    #Put new values into dataframe                    
    df_new = df_rs.reindex(dataframe.index)

    for col in df_new.columns:
        if "iqr" not in col:
            df_new[col].loc[df_new[col] < 0.] = 0.
    
    #Create Multi-Index for cosmo data
    df_new.columns = pd.MultiIndex.from_product([df_new.columns.values.tolist(),['cosmo']],
                                                                   names=['substat','variable'])       
    
    #Get sun position and throw away night time values
    df_new = get_sun_position(df_new, pv_station["lat_lon"], 90.)   

    #quickfix, calculate DNI for COSMO
    df_new[("Edirnorm_mean_Wm2","cosmo")] = df_new[("Edirdown_mean_Wm2","cosmo")]/\
                           np.abs(np.cos(np.deg2rad(df_new[("sza","sun")])))
    df_new[("Edirnorm_iqr_Wm2","cosmo")] = df_new[("Edirdown_iqr_Wm2","cosmo")]/\
                           np.abs(np.cos(np.deg2rad(df_new[("sza","sun")])))
                           
   #quickfix, calculate GHI for COSMO
    df_new[("Etotdown_mean_Wm2","cosmo")] = df_new[("Edirdown_mean_Wm2","cosmo")]+\
                                            df_new[("Ediffdown_mean_Wm2","cosmo")]
    df_new[("Etotdown_iqr_Wm2","cosmo")] = df_new[("Edirdown_iqr_Wm2","cosmo")]+\
                                            df_new[("Ediffdown_iqr_Wm2","cosmo")]                                   

    df_new.sort_index(axis=1,level=1,inplace=True)
    
    #Assign to special cosmo dataframe, and join with main dataframe
    pv_station['df_cosmo_ghi_' + year.split('_')[-1]] = df_new
       
    return pv_station

def prepare_cosmo_data(year,inv_config,key,pv_station,home):
    """
    args:
    :param year: string, year of campaign
    :param inv_config: dictionary of inversion configuration
    :param key: string, name of station
    :param pv_station: dictionary with all info and data of pv station    
    :param home: string, home path
    
    out:
    :return pv_station: dictionary of information and data for PV station including COSMO data
    """
       
    
    configfile = os.path.join(home,inv_config["cosmopvcal_configfile"][year])
    input_config = load_yaml_configfile(configfile)
    finp = open(configfile,'a')
    if "stations_lat_lon" not in input_config:
        finp.write('# latitude, longitude of PV stations within the COSMO grid\n')
        finp.write('stations_lat_lon:\n')
    
        #Write lat lon into config file                
        finp.write('    %s: [%.5f,%.5f]\n' % (key,pv_station['lat_lon'][0],
                       pv_station['lat_lon'][1]))
    else:
        if not input_config["stations_lat_lon"]:                
            finp.write('    %s: [%.5f,%.5f]\n' % (key,pv_station['lat_lon'][0],
                       pv_station['lat_lon'][1]))
        elif key not in input_config["stations_lat_lon"]:
            finp.write('    %s: [%.5f,%.5f]\n' % (key,pv_station['lat_lon'][0],
                       pv_station['lat_lon'][1]))
    
    finp.close()    
    
    #Which days to consider
    #test_days = create_daterange(days[year]["all"])
    
    #Prepare surface data from COSMO
    if inv_config["cosmo_sim"]:
        # call cosmo2pvcal 
        print('Running cosmo2pvcal to extract irradiance at the surface')
        child = subprocess.Popen('cosmo2pvcal ' + configfile, shell=True)
        child.wait()
    else:
        print('cosmo2pvcal already run, read in irradiance files')
    
    pv_station = find_nearest_cosmo_grid_folder(configfile,key,pv_station,'irrad',home)   
    pv_station = import_cosmo_irrad_data(key,pv_station,year)    
    
    return pv_station


def convert_cams_time_stamp (input_value):
    """
    Convert cams observation range to timestamp
    """
    output_value = input_value.split('/')[1]
        
    return output_value

def convert_cams_Whm2_to_Wm2 (input_value):
    """
    Convert cams observation range to timestamp
    
    args:
    :param input_value: input float or integer
    
    out:
    :return output_value: modified float value
    """
    output_value = float(input_value)*60
        
    return output_value

def import_cams_data(year,pv_station,config,homepath):
    """
    Import CAMS irradiance data

    Parameters
    ----------
    year : string, year under consideration
    pv_station : dictionary with PV station information and data
    config : dictionary with PV inversion configuration
    homepath : string with homepath

    Returns
    -------
    pv_station: dictionary with information and data, including CAMS data

    """    
    
    cams_path = os.path.join(homepath,config["cams_data_path"][year])
    
    col_names = ["Timestamp","Etotdown_Wm2","Ediffdown_Wm2","Edirnorm_Wm2"]
    
    #CAMS integrated data needs to be converted back to Wm2
    convert_list = list(zip(col_names[1:],[convert_cams_Whm2_to_Wm2]*3))
    convert_list.append(("Timestamp",convert_cams_time_stamp))
    
    dfs = [pd.read_csv(os.path.join(cams_path,filename),sep=';',
           comment='#',header=None,na_values="nan",usecols=[0,6,8,9],
           names=col_names,converters=dict(convert_list)) for filename in 
           list_files(cams_path) if key.replace('_','') in filename]
        
    dataframe = pd.concat(dfs,axis=0)
    dataframe.index = pd.to_datetime(dataframe["Timestamp"],format='%Y-%m-%dT%H:%M') 
    dataframe.drop(columns="Timestamp",inplace=True)
    
    dataframe.sort_index(axis=0,inplace=True)       
    
    #Create Multi-Index 
    dataframe.columns = pd.MultiIndex.from_product([dataframe.columns.values.tolist(),
                            ['cams']],names=['variable','substat'])       
    
    #Assign to special cosmo dataframe, and join with main dataframe
    pv_station['df_cams_' + year.split('_')[-1]] = dataframe
    
    return pv_station

def moving_average_std(input_series,data_freq,window_avg):
    """
    Calculate moving average and standard deviation of input series

    Parameters
    ----------
    input_series : series with input data
    data_freq : timedelta, data frequency
    window_avg : timedelta, width of averaging window   
    
    Returns
    -------
    dataframe with average and standard deviation

    """        
    
    window_size = int(window_avg/data_freq) 

    #Calculate moving average with astropy convolution       
    # avg = pd.Series(data=convolve(input_series,Box1DKernel(window_size),nan_treatment='interpolate'),
    #                 index=input_series.index,name='avg_conv')
    
    # #Calculate standard deviation
    # std = pd.Series(np.sqrt(convolve(input_series.values**2,Box1DKernel(window_size),
    #              nan_treatment='interpolate') - avg**2),index=input_series.index,name='std_conv')
    edge = int(window_size/2.)
    # avg  = avg[edge:-edge]
    # std  = std[edge:-edge]
    
    #alternative method with pandas
    avg_alt = input_series.interpolate(method='linear',limit=int(edge/2)).\
        rolling(window=window_avg,min_periods=edge).\
            mean().shift(-edge).rename('avg_pd')        
    std_alt = input_series.interpolate(method='linear',limit=int(edge/2)).\
        rolling(window=window_avg,min_periods=edge).\
        std().shift(-edge).rename('std_pd')
    
    dataframe = pd.concat([avg_alt,std_alt],axis=1) #avg,std,
    
    return dataframe

def combine_cams_hires(year,pv_station):
    """
    
    Combine CAMS with high resolution measured irradiance
    
    Parameters
    ----------
    year : string, year under consideration 
    pv_station : dictionary with information and data from PV station

    Returns
    -------
    None.

    """
    
    data_freq_cams = pd.Timedelta('1T')        
    df_cams = pv_station[f"df_cams_{year}"]    
    
    #Go through the timeres list and add CAMS to dataframes if appropriate
    for timeres in pv_station["timeres"]:
        window_avg = pd.Timedelta(timeres)
        if f"df_pyr_pv_ghi_{year}_{timeres}" in pv_station:
            if window_avg == data_freq_cams:
                pv_station[f"df_pyr_pv_ghi_{year}_{timeres}"] = pd.concat([pv_station[f"df_pyr_pv_ghi_{year}_{timeres}"],
                              df_cams.loc[pv_station[f"df_pyr_pv_ghi_{year}_{timeres}"].index]],axis=1)
                if f"df_aodfit_pyr_pv_{year}_{timeres}" in pv_station:
                    pv_station[f"df_aodfit_pyr_pv_{year}_{timeres}"] = pd.concat([pv_station[f"df_aodfit_pyr_pv_{year}_{timeres}"],
                              df_cams.loc[pv_station[f"df_aodfit_pyr_pv_{year}_{timeres}"].index]],axis=1)
                if f"df_codfit_pyr_pv_{year}_{timeres}" in pv_station:
                    pv_station[f"df_codfit_pyr_pv_{year}_{timeres}"] = pd.concat([pv_station[f"df_codfit_pyr_pv_{year}_{timeres}"],
                              df_cams.loc[pv_station[f"df_codfit_pyr_pv_{year}_{timeres}"].index]],axis=1)
                
            elif window_avg > data_freq_cams:                
                #List for concatenating days
                dfs_Edir = []
                dfs_Ediff = []
                dfs_Etot = []
                
                #List for moving averages
                dfs_Edir_avg_std = []
                dfs_Ediff_avg_std = []
                dfs_Etot_avg_std = []
                
                
                #1. Calculate moving average of CAMS
                for day in pd.to_datetime(df_cams.index.date).unique():                                            
                    if len(df_cams.loc[day.strftime("%Y-%m-%d")]) > 1:
                        df_Edir_avg_std = moving_average_std(df_cams.loc[day.strftime("%Y-%m-%d"),
                                        ("Edirnorm_Wm2","cams")],data_freq_cams, window_avg) 
                        df_Ediff_avg_std = moving_average_std(df_cams.loc[day.strftime("%Y-%m-%d"),
                                      ("Ediffdown_Wm2","cams")],data_freq_cams, window_avg)
                        df_Etot_avg_std = moving_average_std(df_cams.loc[day.strftime("%Y-%m-%d"),
                                      ("Etotdown_Wm2","cams")],data_freq_cams, window_avg)
                        
                        dfs_Edir_avg_std.append(df_Edir_avg_std)
                        dfs_Ediff_avg_std.append(df_Ediff_avg_std)
                        dfs_Etot_avg_std.append(df_Etot_avg_std)
                        
                        df_Edir_cams_reindex = df_Edir_avg_std.reindex(pd.date_range(start=df_Edir_avg_std.index[0].round(window_avg),
                                                                 end=df_Edir_avg_std.index[-1].round(window_avg),freq=window_avg),
                                                                 method='nearest',tolerance='5T').loc[day.strftime("%Y-%m-%d")]        
                        df_Ediff_cams_reindex = df_Ediff_avg_std.reindex(pd.date_range(start=df_Ediff_avg_std.index[0].round(window_avg),
                                                                 end=df_Ediff_avg_std.index[-1].round(window_avg),freq=window_avg),
                                                                 method='nearest',tolerance='5T').loc[day.strftime("%Y-%m-%d")]        
                        df_Etot_cams_reindex = df_Etot_avg_std.reindex(pd.date_range(start=df_Etot_avg_std.index[0].round(window_avg),
                                                                 end=df_Etot_avg_std.index[-1].round(window_avg),freq=window_avg),
                                                                 method='nearest',tolerance='5T').loc[day.strftime("%Y-%m-%d")]        
                        
                        dfs_Edir.append(df_Edir_cams_reindex)
                        dfs_Ediff.append(df_Ediff_cams_reindex)
                        dfs_Etot.append(df_Etot_cams_reindex)
                
                #Assign average and std to cams dataframe
                df_total = pd.concat([pd.concat(dfs_Edir_avg_std,axis=0),pd.concat(dfs_Ediff_avg_std,axis=0),
                                        pd.concat(dfs_Etot_avg_std,axis=0)],axis=1)
                df_cams[(f"Edirnorm_{timeres}_avg","cams")] = df_total.iloc[:,0]
                df_cams[(f"Edirnorm_{timeres}_std","cams")] = df_total.iloc[:,1]
                df_cams[(f"Ediffdown_{timeres}_avg","cams")] = df_total.iloc[:,2]    
                df_cams[(f"Ediffdown_{timeres}_std","cams")] = df_total.iloc[:,3]    
                df_cams[(f"Etotdown_{timeres}_avg","cams")] = df_total.iloc[:,4]    
                df_cams[(f"Etotdown_{timeres}_std","cams")] = df_total.iloc[:,5]                
                
                df_cams.sort_index(axis=1,level=1,inplace=True)        
                
                #Assign reindexed values to comparison list
                df_compare_tres = pd.concat([pd.concat(dfs_Edir,axis=0),pd.concat(dfs_Ediff,axis=0),
                                        pd.concat(dfs_Etot,axis=0)],axis=1)
                df_compare_tres.columns = pd.MultiIndex.from_product([[f"Edirnorm_{timeres}_avg",f"Edirnorm_{timeres}_std",
                                    f"Ediffdown_{timeres}_avg",f"Ediffdown_{timeres}_std",
                                    f"Etotdown_{timeres}_avg",f"Etotdown_{timeres}_std"]
                                                                ,['cams']],names=['variable','substat'])                       
                pv_station[f"df_pyr_pv_ghi_{year}_{timeres}"] = pd.concat([pv_station[f"df_pyr_pv_ghi_{year}_{timeres}"],
                              df_compare_tres.loc[pv_station[f"df_pyr_pv_ghi_{year}_{timeres}"].index]],axis=1)                        
                if f"df_aodfit_pyr_pv_{year}_{timeres}" in pv_station:
                    pv_station[f"df_aodfit_pyr_pv_{year}_{timeres}"] = pd.concat([pv_station[f"df_aodfit_pyr_pv_{year}_{timeres}"],
                              df_compare_tres.loc[pv_station[f"df_aodfit_pyr_pv_{year}_{timeres}"].index]],axis=1)
                if f"df_codfit_pyr_pv_{year}_{timeres}" in pv_station:
                    pv_station[f"df_codfit_pyr_pv_{year}_{timeres}"] = pd.concat([pv_station[f"df_codfit_pyr_pv_{year}_{timeres}"],
                              df_compare_tres.loc[pv_station[f"df_codfit_pyr_pv_{year}_{timeres}"].index]],axis=1)

def combine_cosmo_cams(year,pv_station,str_window_avg):
    """
    

    Parameters
    ----------
    year : string, year under consideration
    pv_station : dictionary with information and data from PV station
    str_window_avg : string with width of averaging window

    Returns
    -------
    df_compare_list : dataframe with all data averaged and combined

    """
                 
    #Size of moving average window
    window_avg = pd.Timedelta(str_window_avg)  
    
    df_cosmo = pv_station[f"df_cosmo_ghi_{year}"]
    df_cams = pv_station[f"df_cams_{year}"]    
    
    #List for comparison of different CODs
    dfs_compare_list = []
    
    #List for concatenating days
    dfs_Edir = []
    dfs_Ediff = []
    dfs_Etot = []
    
    #List for moving averages
    dfs_Edir_avg_std = []
    dfs_Ediff_avg_std = []
    dfs_Etot_avg_std = []
    data_freq_cams = pd.Timedelta('1T')        
    #1. Calculate moving average of CAMS
    for day in pd.to_datetime(df_cams.index.date).unique():                                            
        if len(df_cams.loc[day.strftime("%Y-%m-%d")]) > 1:
            df_Edir_avg_std = moving_average_std(df_cams.loc[day.strftime("%Y-%m-%d"),
                            ("Edirnorm_Wm2","cams")],data_freq_cams, window_avg) 
            df_Ediff_avg_std = moving_average_std(df_cams.loc[day.strftime("%Y-%m-%d"),
                          ("Ediffdown_Wm2","cams")],data_freq_cams, window_avg)
            df_Etot_avg_std = moving_average_std(df_cams.loc[day.strftime("%Y-%m-%d"),
                          ("Etotdown_Wm2","cams")],data_freq_cams, window_avg)
            
            dfs_Edir_avg_std.append(df_Edir_avg_std)
            dfs_Ediff_avg_std.append(df_Ediff_avg_std)
            dfs_Etot_avg_std.append(df_Etot_avg_std)
            
            df_Edir_cams_reindex_60 = df_Edir_avg_std.reindex(pd.date_range(start=df_Edir_avg_std.index[0].round(window_avg),
                                                     end=df_Edir_avg_std.index[-1].round(window_avg),freq=window_avg),
                                                     method='nearest',tolerance='5T').loc[day.strftime("%Y-%m-%d")]        
            df_Ediff_cams_reindex_60 = df_Ediff_avg_std.reindex(pd.date_range(start=df_Ediff_avg_std.index[0].round(window_avg),
                                                     end=df_Ediff_avg_std.index[-1].round(window_avg),freq=window_avg),
                                                     method='nearest',tolerance='5T').loc[day.strftime("%Y-%m-%d")]        
            df_Etot_cams_reindex_60 = df_Etot_avg_std.reindex(pd.date_range(start=df_Etot_avg_std.index[0].round(window_avg),
                                                     end=df_Etot_avg_std.index[-1].round(window_avg),freq=window_avg),
                                                     method='nearest',tolerance='5T').loc[day.strftime("%Y-%m-%d")]        
            
            dfs_Edir.append(df_Edir_cams_reindex_60)
            dfs_Ediff.append(df_Ediff_cams_reindex_60)
            dfs_Etot.append(df_Etot_cams_reindex_60)
    
    #Assign reindexed values to comparison list
    df_compare = pd.concat([pd.concat(dfs_Edir,axis=0),pd.concat(dfs_Ediff,axis=0),
                            pd.concat(dfs_Etot,axis=0)],axis=1)
    df_compare.columns = pd.MultiIndex.from_product([[f"Edirnorm_{str_window_avg}_avg",f"Edirnorm_{str_window_avg}_std",
                        f"Ediffdown_{str_window_avg}_avg",f"Ediffdown_{str_window_avg}_std",
                        f"Etotdown_{str_window_avg}_avg",f"Etotdown_{str_window_avg}_std"]
                                                    ,['cams']],names=['variable','substat'])                       
    dfs_compare_list.append(df_compare)        
    
    #Assign average and std to cams dataframe
    df_total = pd.concat([pd.concat(dfs_Edir_avg_std,axis=0),pd.concat(dfs_Ediff_avg_std,axis=0),
                            pd.concat(dfs_Etot_avg_std,axis=0)],axis=1)
    df_cams[(f"Edirnorm_{str_window_avg}_avg","cams")] = df_total.iloc[:,0]
    df_cams[(f"Edirnorm_{str_window_avg}_std","cams")] = df_total.iloc[:,1]
    df_cams[(f"Ediffdown_{str_window_avg}_avg","cams")] = df_total.iloc[:,2]    
    df_cams[(f"Ediffdown_{str_window_avg}_std","cams")] = df_total.iloc[:,3]    
    df_cams[(f"Etotdown_{str_window_avg}_avg","cams")] = df_total.iloc[:,4]    
    df_cams[(f"Etotdown_{str_window_avg}_std","cams")] = df_total.iloc[:,5]                
    
    df_cams.sort_index(axis=1,level=1,inplace=True)        
            
    #Cosmo dataframe is already at one hour    
    data_freq_cosmo = pd.Timedelta('60T')
    
    #List for concatenating days
    dfs_Edir = []
    dfs_Ediff = []    
    dfs_Etot
    
    #List for moving averages
    dfs_Edir_avg_std = []
    dfs_Ediff_avg_std = []    
    dfs_Etot_avg_std = []
    
    if window_avg == data_freq_cosmo:
        dfs_compare_list.append(df_cosmo)
    elif window_avg > data_freq_cosmo:
        for day in pd.to_datetime(df_cosmo.index.date).unique():  
            df_Edir_avg_std = moving_average_std(df_cosmo.loc[day.strftime("%Y-%m-%d"),
                            ('Edirnorm_mean_Wm2',"cosmo")],data_freq_cosmo, window_avg) 
            df_Ediff_avg_std = moving_average_std(df_cosmo.loc[day.strftime("%Y-%m-%d"),
                          ("Ediffdown_mean_Wm2","cosmo")],data_freq_cosmo, window_avg)            
            df_Etot_avg_std = moving_average_std(df_cosmo.loc[day.strftime("%Y-%m-%d"),
                          ("Etotdown_mean_Wm2","cosmo")],data_freq_cosmo, window_avg)            
            
            dfs_Edir_avg_std.append(df_Edir_avg_std)
            dfs_Ediff_avg_std.append(df_Ediff_avg_std)            
            dfs_Etot_avg_std.append(df_Etot_avg_std) 
            
            df_Edir_cosmo_reindex_60 = df_Edir_avg_std.reindex(pd.date_range(start=df_Edir_avg_std.index[0].round(window_avg),
                                                     end=df_Edir_avg_std.index[-1].round(window_avg),freq=window_avg),
                                                     method='nearest',tolerance='5T').loc[day.strftime("%Y-%m-%d")]        
            df_Ediff_cosmo_reindex_60 = df_Ediff_avg_std.reindex(pd.date_range(start=df_Ediff_avg_std.index[0].round(window_avg),
                                                     end=df_Ediff_avg_std.index[-1].round(window_avg),freq=window_avg),
                                                     method='nearest',tolerance='5T').loc[day.strftime("%Y-%m-%d")]                    
            df_Etot_cosmo_reindex_60 = df_Etot_avg_std.reindex(pd.date_range(start=df_Etot_avg_std.index[0].round(window_avg),
                                                     end=df_Etot_avg_std.index[-1].round(window_avg),freq=window_avg),
                                                     method='nearest',tolerance='5T').loc[day.strftime("%Y-%m-%d")]                    
            
            dfs_Edir.append(df_Edir_cosmo_reindex_60)
            dfs_Ediff.append(df_Ediff_cosmo_reindex_60)                                        
            dfs_Etot.append(df_Etot_cosmo_reindex_60)   
    
        #Assign reindexed values to comparison list
        df_compare = pd.concat([pd.concat(dfs_Edir,axis=0),pd.concat(dfs_Ediff,axis=0),pd.concat(dfs_Ediff,axis=0)],axis=1)
        df_compare.columns = pd.MultiIndex.from_product([[f"Edirnorm_mean_{str_window_avg}_avg",f"Edirnorm_mean_{str_window_avg}_std",
                            f"Ediffdown_mean_{str_window_avg}_avg",f"Ediffdown_mean_{str_window_avg}_std",
                            f"Etotdown_mean_{str_window_avg}_avg",f"Etotdown_mean_{str_window_avg}_std"]
                                                        ,['cosmo']],names=['variable','substat'])                       
        dfs_compare_list.append(df_compare)        
        
        #Assign average and std to cams dataframe
        df_total = pd.concat([pd.concat(dfs_Edir_avg_std,axis=0),pd.concat(dfs_Ediff_avg_std,axis=0),
                              pd.concat(dfs_Ediff_avg_std,axis=0)],axis=1)
        df_cosmo[(f"Edirnorm_mean_{str_window_avg}_avg","cosmo")] = df_total.iloc[:,0]
        df_cosmo[(f"Edirnorm_mean_{str_window_avg}_std","cosmo")] = df_total.iloc[:,1]
        df_cosmo[(f"Ediffdown_mean_{str_window_avg}_avg","cosmo")] = df_total.iloc[:,2]    
        df_cosmo[(f"Ediffdown_mean_{str_window_avg}_std","cosmo")] = df_total.iloc[:,3]    
        df_cosmo[(f"Etotdown_mean_{str_window_avg}_avg","cosmo")] = df_total.iloc[:,4]    
        df_cosmo[(f"Etotdown_mean_{str_window_avg}_std","cosmo")] = df_total.iloc[:,5]    
        
        df_cosmo.sort_index(axis=1,level=1,inplace=True)   
    
    #Combine all into one
    df_compare_list = pd.concat(dfs_compare_list,axis=1)    
                
    return df_compare_list

def combine_raw_data(year,data_dict,str_window_avg,config):
    """
    
    Average raw data and combine with other data

    Parameters
    ----------
    year : string with year under consideration
    data_dict : dictionary with raw data
    str_window_avg : string with window for averaging
    config : dictionary with configuration for specific station

    Returns
    -------
    dataframe with all data combined

    """
    
    #Size of moving average window
    window_avg = pd.Timedelta(str_window_avg)  
    
    irrad_names = config["irrad_names"]
    rad_substats = irrad_names["substat"]
    
    if type(rad_substats) != list:
        rad_substats = [rad_substats]
    
    dfs_compare = []
    for rad_substat in rad_substats:
    
        df_raw = data_dict["irrad"][f"df_{rad_substat}"]
        days_raw = pd.to_datetime(df_raw.index.date).unique().strftime('%Y-%m-%d')    
         
        dfs_raw_new = []  
    
        if "DNI" in irrad_names:
            dfs_Edir = []
            dfs_Ediff = []         
        dfs_Etot = []
        
        for day in days_raw:
            df_raw_day = df_raw.loc[day]        
            if not df_raw_day.empty:
                if len(df_raw_day) > 10:
                    if df_raw_day.index.duplicated().any():
                        df_raw_day = df_raw_day[~df_raw_day.index.duplicated()]
                    t_delta = df_raw_day.index.to_series().diff()
                    if t_delta.max() != t_delta.min():                
                        df_raw_day = df_raw_day.resample(t_delta.min()).interpolate('linear')
                                
                    if "DNI" in irrad_names:
                        df_Edir_avg_std = moving_average_std(df_raw_day[irrad_names["DNI"]],t_delta.min(),
                                                             pd.Timedelta(str_window_avg))    
                        df_Ediff_avg_std = moving_average_std(df_raw_day[irrad_names["DHI"]],t_delta.min(),
                                                             pd.Timedelta(str_window_avg)) 
                        
                        df_raw_day[f"{irrad_names['DNI']}_{str_window_avg}_avg"] = df_Edir_avg_std["avg_pd"]
                        df_raw_day[f"{irrad_names['DNI']}_{str_window_avg}_std"] = df_Edir_avg_std["std_pd"]
                        df_raw_day[f"{irrad_names['DHI']}_{str_window_avg}_avg"] = df_Ediff_avg_std["avg_pd"]
                        df_raw_day[f"{irrad_names['DHI']}_{str_window_avg}_std"] = df_Ediff_avg_std["std_pd"]
                        
                        dfs_Edir.append(df_Edir_avg_std.reindex(pd.date_range(start=df_Edir_avg_std.index[0].round(window_avg),
                                                             end=df_Edir_avg_std.index[-1].round(window_avg),freq=window_avg),
                                                             method='nearest',tolerance='5T').loc[day])        
                        dfs_Ediff.append(df_Ediff_avg_std.reindex(pd.date_range(start=df_Ediff_avg_std.index[0].round(window_avg),
                                                             end=df_Ediff_avg_std.index[-1].round(window_avg),freq=window_avg),
                                                             method='nearest',tolerance='5T').loc[day])    
                    
                    df_Etot_avg_std = moving_average_std(df_raw_day[irrad_names["GHI"]],t_delta.min(),
                                                             pd.Timedelta(str_window_avg))    
                    
                    df_raw_day[f"{irrad_names['GHI']}_{str_window_avg}_avg"] = df_Etot_avg_std["avg_pd"]
                    df_raw_day[f"{irrad_names['GHI']}_{str_window_avg}_std"] = df_Etot_avg_std["std_pd"]
                        
                    dfs_Etot.append(df_Etot_avg_std.reindex(pd.date_range(start=df_Etot_avg_std.index[0].round(window_avg),
                                                             end=df_Etot_avg_std.index[-1].round(window_avg),freq=window_avg),
                                                             method='nearest',tolerance='5T').loc[day])        
                                
            dfs_raw_new.append(df_raw_day)
            
        if "DNI" in irrad_names:
            df_compare = pd.concat([pd.concat(dfs_Edir,axis=0),pd.concat(dfs_Ediff,axis=0),pd.concat(dfs_Etot,axis=0)],axis=1)
            df_compare.columns = pd.MultiIndex.from_product([[f"Edirnorm_Wm2_{str_window_avg}_avg",f"Edirnorm_Wm2_{str_window_avg}_std",
                            f"Ediffdown_Wm2_{str_window_avg}_avg",f"Ediffdown_Wm2_{str_window_avg}_std",
                            f"Etotdown_Wm2_{str_window_avg}_avg",f"Etotdown_Wm2_{str_window_avg}_std"]
                            ,[rad_substat]],names=['variable','substat'])   
        else:
            df_compare = pd.concat(dfs_Etot,axis=0)
            df_compare.columns = pd.MultiIndex.from_product([[f"Etotdown_Wm2_{str_window_avg}_avg",f"Etotdown_Wm2_{str_window_avg}_std"]
                            ,[rad_substat]],names=['variable','substat'])               
        
        dfs_compare.append(df_compare)
        if len(dfs_raw_new) > 0:
            data_dict["irrad"][f"df_{rad_substat}"] = pd.concat(dfs_raw_new,axis=0)
        
    return pd.concat(dfs_compare,axis=1)

def downsample_pyranometer_data(dataframe,substat,timeres_old,timeres_new):
    """
    Downsample pyranometer data to coarser data

    Parameters
    ----------
    dataframe : dataframe with high resolution data
    substat : string, name of substation
    timeres_old : string, old time resolution
    timeres_new : string, desired time resolution

    Returns
    -------
    dataframe with downsampled pyranometer data

    """
    
    #Convert time resolutions
    timeres_old = pd.to_timedelta(timeres_old)
    timeres_new = pd.to_timedelta(timeres_new)
    
    #Get correct names for irradiance columns
    if "Pyr" in substat:
        radtypes = ["down","poa"]
    elif "mordor" in substat or "CMP11_Horiz" in substat or "suntracker" == substat:
        radtypes = ["down"]
    else:
        radtypes = ["poa"]
            
    radnames_inversion = [f'E{component}down_{radtype}_inv' for radtype in radtypes 
                          for component in ["dir","diff"]]        
    
    #Select correct columns
    df_old = dataframe.loc[:,pd.IndexSlice[radnames_inversion,substat]]
    
    dfs_rs = []
    for day in pd.to_datetime(df_old.index.date).unique().strftime('%Y-%m-%d'):
        #Downsample data
        dfs_rs.append(downsample(df_old.loc[day], timeres_old, timeres_new))
    
    df_rs = pd.concat(dfs_rs,axis=0)
        
    return df_rs
    
def avg_std_irradiance_retrieval(day,df_day_result,df_day_lut,substat,radtype,timeres,
                       str_window_avg,odtype):
    """
    
    Calculate average and standard deviation of irradiance retrieval

    Parameters
    ----------
    day : string, day under consideration
    df_day_result : dataframe with DISORT OD LUT results for specific day
    df_day_lut : dataframe with MYSTIC LUT results for specific day
    substat : string, name of substation
    radtype : string, type of irradiance (poa or down)    
    timeres : string, time resolution of data 
    str_window_avg : string, width of window for moving average
    odtype : string with OD type (AOD or COD)
    
    
    Returns
    -------
    dataframe with combined and averaged results

    """        
    
    #Original time resolution
    data_freq = pd.Timedelta(timeres)
    #Size of moving average window
    window_avg = pd.Timedelta(str_window_avg)    
    
    #1. Calculate moving average of irradiance retrieval        
    df_Edir_avg_std = moving_average_std(df_day_result[(f"Edirdown_{radtype}_inv",substat)],data_freq,window_avg)    
    df_Ediff_avg_std = moving_average_std(df_day_result[(f"Ediffdown_{radtype}_inv",substat)],data_freq,window_avg)
    if odtype == "cod":
        df_Edir_eff_avg_std = moving_average_std(df_day_result[(f"Edirdown_eff_{radtype}_inv",substat)],data_freq,window_avg)    
        df_Ediff_eff_avg_std = moving_average_std(df_day_result[(f"Ediffdown_eff_{radtype}_inv",substat)],data_freq,window_avg)
    
    if "Etotdown_lut_Wm2" in df_day_lut.columns.levels[0] and radtype == "poa":
        df_Etotlut_avg_std = moving_average_std(df_day_lut[("Etotdown_lut_Wm2",substat)],data_freq,window_avg)
            
    #Calculate moving average of cloud fraction
    df_cf_day = deepcopy(df_day_result[(f"cloud_fraction_{radtype}",substat)])
    df_cf_day.loc[df_cf_day < 0] = np.nan #0.5
    df_cf_avg_std = moving_average_std(df_cf_day,data_freq,window_avg)
    
    #Assign to dataframe    
    for vartype in ["avg","std"]:
        df_day_result[(f"Edirdown_{radtype}_inv_{str_window_avg}_{vartype}",substat)] = df_Edir_avg_std[f"{vartype}_pd"]        
        df_day_result[(f"Ediffdown_{radtype}_inv_{str_window_avg}_{vartype}",substat)] = df_Ediff_avg_std[f"{vartype}_pd"]        
        
        #Calculate DNI
        df_day_result[(f"Edirnorm_{radtype}_inv_{str_window_avg}_{vartype}",substat)] = \
            df_day_result[(f"Edirdown_{radtype}_inv_{str_window_avg}_{vartype}",substat)]/\
                np.cos(np.deg2rad(df_day_result[('sza','sun')]))
        #Calculate GHI
        df_day_result[(f"Etotdown_{radtype}_inv_{str_window_avg}_{vartype}",substat)] = \
            df_day_result[(f"Edirdown_{radtype}_inv_{str_window_avg}_{vartype}",substat)]+\
            df_day_result[(f"Ediffdown_{radtype}_inv_{str_window_avg}_{vartype}",substat)]    
            
        if odtype =="cod":
            df_day_result[(f"Edirdown_eff_{radtype}_inv_{str_window_avg}_{vartype}",substat)] = df_Edir_eff_avg_std[f"{vartype}_pd"]        
            df_day_result[(f"Ediffdown_eff_{radtype}_inv_{str_window_avg}_{vartype}",substat)] = df_Ediff_eff_avg_std[f"{vartype}_pd"]        
        
            df_day_result[(f"Edirnorm_eff_{radtype}_inv_{str_window_avg}_{vartype}",substat)] = \
                df_day_result[(f"Edirdown_eff_{radtype}_inv_{str_window_avg}_{vartype}",substat)]/\
                    np.cos(np.deg2rad(df_day_result[('sza','sun')]))
                    
            df_day_result[(f"Etotdown_eff_{radtype}_inv_{str_window_avg}_{vartype}",substat)] = \
                df_day_result[(f"Edirdown_eff_{radtype}_inv_{str_window_avg}_{vartype}",substat)]+\
                df_day_result[(f"Ediffdown_eff_{radtype}_inv_{str_window_avg}_{vartype}",substat)]
        
        if radtype == "poa":
            df_day_lut[(f"Etotdown_lut_{str_window_avg}_{vartype}",substat)] = df_Etotlut_avg_std[f"{vartype}_pd"]                
        
    if (f"cf_{radtype}_{str_window_avg}_avg",substat) not in df_day_result.columns:
        df_day_result[(f"cf_{radtype}_{str_window_avg}_avg",substat)] = df_cf_avg_std["avg_pd"]                  
    
    #Reindex the moving average to the nearest hour
    dfs_reindex = []
    reindex_names = [f"Edirdown_{radtype}_inv_{str_window_avg}_avg",f"Edirdown_{radtype}_inv_{str_window_avg}_std",                   
                   f"Ediffdown_{radtype}_inv_{str_window_avg}_avg",f"Ediffdown_{radtype}_inv_{str_window_avg}_std",
                   f"Edirnorm_{radtype}_inv_{str_window_avg}_avg",f"Edirnorm_{radtype}_inv_{str_window_avg}_std",
                   f"Etotdown_{radtype}_inv_{str_window_avg}_avg",f"Etotdown_{radtype}_inv_{str_window_avg}_std",
                   f"cf_{radtype}_{str_window_avg}_avg"]
    
    if odtype == "cod":
        reindex_names.extend([f"Edirdown_eff_{radtype}_inv_{str_window_avg}_avg",f"Edirdown_eff_{radtype}_inv_{str_window_avg}_std",
                              f"Ediffdown_eff_{radtype}_inv_{str_window_avg}_avg",f"Ediffdown_eff_{radtype}_inv_{str_window_avg}_std",
                              f"Edirnorm_eff_{radtype}_inv_{str_window_avg}_avg",f"Edirnorm_eff_{radtype}_inv_{str_window_avg}_std",
                              f"Etotdown_eff_{radtype}_inv_{str_window_avg}_avg",f"Etotdown_eff_{radtype}_inv_{str_window_avg}_std"])
    
    for reindex_name in reindex_names:
        df_reindex = df_day_result[(reindex_name,substat)].reindex(pd.date_range(start=df_Edir_avg_std.index[0].round(window_avg),
                        end=df_Edir_avg_std.index[-1].round(window_avg),freq=window_avg)
                          ,method='nearest',tolerance='5T')
        df_reindex.rename((reindex_name,substat),inplace=True) #,names=['variable','substat'])   
        dfs_reindex.append(df_reindex)
    
    if radtype == "poa":
        reindex_names = [f"Etotdown_lut_{str_window_avg}_avg",f"Etotdown_lut_{str_window_avg}_std"]
        for reindex_name in reindex_names:
            dflut_reindex = df_day_lut[(reindex_name,substat)].reindex(pd.date_range(start=df_Etotlut_avg_std.index[0].round(window_avg),
                            end=df_Etotlut_avg_std.index[-1].round(window_avg),freq=window_avg)
                              ,method='nearest',tolerance='5T')
            dflut_reindex = dflut_reindex.loc[day]
            dflut_reindex.rename((reindex_name,substat),inplace=True) #,names=['variable','substat'])   
            dfs_reindex.append(dflut_reindex)
            
    df_reindex_combine = pd.concat(dfs_reindex,axis=1)
    df_reindex_combine.rename_axis(['variable','substat'],axis=1,inplace=True) 
    
    return df_reindex_combine
#df_Edir_reindex_60,df_Ediff_reindex_60,df_Edirnorm_reindex_60,df_Etotdown_reindex_60,cf_avg_reindex_60

def irradiance_analysis_plots(name,pv_station,substat_pars,od_types,year,sza_limit_aod,
                      sza_limit_cod,str_window_avg,window_avg_cf,meas_names,pyr_down_name,
                      savepath,flags):
    """
    

    Parameters
    ----------
    name : string, name of PV station
    pv_station : dictionary with information and data from PV station
    substat_pars : dictionary with substation parameters
    od_types : list of OD types (AOD or COD)
    year : string with year under consideration
    sza_limit_aod : float with SZA limit of simulation and results for AOD
    sza_limit_cod : float with SZA limit of simulation and results for COD
    str_window_avg : string with width of averaging window for COD
    window_avg_cf : string with width of averaging window for cloud fraction
    meas_names : dictionary with variable names for measured data
    pyr_down_name : string with name of pyranometer for validation
    savepath : string with path to save plots    
    flags : dictionary of booleans for plotting
    
    Returns
    -------
    dataframe with combined and averaged results

    """    
    
    plt.close('all')
    
    res_dirs = list_dirs(savepath)
    savepath = os.path.join(savepath,'DNI_DHI_GHI_Plots')
    if 'DNI_DHI_GHI_Plots' not in res_dirs:
        os.mkdir(savepath)                
            
    stat_dirs = list_dirs(savepath)
    savepath = os.path.join(savepath,name)
    if name not in stat_dirs:
        os.mkdir(savepath)            
        
    stat_dirs = list_dirs(savepath)
    savepath = os.path.join(savepath,"Comparison")
    if "Comparison" not in stat_dirs:
        os.mkdir(savepath)      
        
    res_dirs = list_dirs(savepath)
    savepath = os.path.join(savepath,str_window_avg)
    if str_window_avg not in res_dirs:
        os.mkdir(savepath)

    df_cams = pv_station[f"df_cams_{year}"]    
    df_cosmo = pv_station[f"df_cosmo_ghi_{year}"]    
    
    dfs_substat_combine = [pv_station[f"df_compare_ghi_{year}_{str_window_avg}"]]
    
    for substat in substat_pars:
        substat_dirs = list_dirs(savepath)        
        plotpath = os.path.join(savepath,substat)
        if substat not in substat_dirs:
            os.mkdir(plotpath)                                
                    
        timeres = substat_pars[substat]["t_res_inv"]
        
        df_lut_result = pv_station[f"df_pyr_pv_ghi_{year}_{timeres}"]
        if substat in df_lut_result.columns.levels[1]:
            days_lut = pd.to_datetime(df_lut_result.index.date).unique().strftime('%Y-%m-%d')                
        else: days_lut = []
        
        if meas_names:
            if type(meas_names["substat"]) != list:
                df_raw = pv_station["raw_data"]["irrad"][f"df_{meas_names['substat']}"]
            else:
                df_raw = pd.concat([pv_station["raw_data"]["irrad"]
                            [f"df_{rad_substat}"] for rad_substat in meas_names['substat']],axis=1)
            days_raw = pd.to_datetime(df_raw.index.date).unique().strftime('%Y-%m-%d')                
        else:
            days_raw = pd.DataFrame()
        
        #Lists for combination of ODs
        dfs_final_results = []
        dfs_final_combine = []
        
        for od in od_types:
            if od == "aod":
                odstring = "AOD_500"
                sza_limit = sza_limit_aod
            elif od == "cod":
                odstring = "COD_550"
                sza_limit = sza_limit_cod
            df_od_result = pv_station[f"df_{od}fit_pyr_pv_{year.split('_')[-1]}_{timeres}"]
        
            if "Pyr" in substat:
                radtypes = ["down","poa"]
            elif "mordor" in substat or "CMP11_Horiz" in substat or "suntracker" == substat:
                radtypes = ["down"]
            else:
                radtypes = ["poa"]
                            
            #Lists for combination of radtypes
            dfs_results = []
            dfs_combine = []    
        
            days_result = pd.to_datetime(df_od_result.index.date).unique().strftime('%Y-%m-%d')
            empty_radtypes = []
            
            for radtype in radtypes:                                
                if radtype == "poa":                
                    radname = substat_pars[substat]["name"]
                    if "pv" not in radname:
                        errorname = "error_" + '_'.join([radname.split('_')[0],radname.split('_')[-1]])
                    else:
                        errorname = "error_" + radname
                    
                elif radtype == "down":
                    radname = substat_pars[substat]["name"].replace("poa","down")            
                    errorname = "error_" + '_'.join([radname.split('_')[0],radname.split('_')[-1]])                                               
                                
                if (radname,substat) in df_od_result.columns:
                    print("Irradiance analysis and plots for %s, %s for irradiance from %s, from %s measurement at %s" % (name,substat,od,radtype,timeres))            
                    #Plotlabels
                    if radtype == "poa":
                        radlabel = "\\angle"
                        titlelabel = "tilted"
                    elif radtype == "down":
                        radlabel = "\\downarrow"
                        titlelabel = "downward"
                    
                    dfs_radtype_combine = []   
                    dfs_radtype_results = []
                    dfs_raw_new = []
                    # loop over the dates and plot for each day
                    for day in days_result:
                        df_day_result = df_od_result.loc[day,pd.IndexSlice[:,[substat,"sun"]]]
                        df_day_lut = df_lut_result.loc[day,pd.IndexSlice[:,substat]]
                            
                        #Calculate moving average and standard deviation for COD retrieval
                        dfs_radtype_combine.append(avg_std_irradiance_retrieval(day,df_day_result,df_day_lut,
                                                        substat,radtype,timeres,str_window_avg,od))   
                        
                        if radtype == "poa":
                            df_day_result = pd.concat([df_day_result,df_day_lut[[\
                                                    (f"Etotdown_lut_{str_window_avg}_avg",substat),
                                                    (f"Etotdown_lut_{str_window_avg}_std",substat)]]],axis=1)
                        dfs_radtype_results.append(df_day_result)                                                            
                            
                            
                    if len(dfs_radtype_combine) > 0:
                        df_radtype_combine = pd.concat(dfs_radtype_combine,axis=0)        
                        dfs_combine.append(df_radtype_combine)
                    
                        df_radtype_results = pd.concat(dfs_radtype_results,axis=0)
                        dfs_results.append(df_radtype_results)    
                else:
                    print(f"There is no data for {radname}")
                    empty_radtypes.append(radtype)
                    
            #Combine all results after calculating moving averages
            df_results_combine = pd.concat(dfs_results,axis=1).filter(regex='cf|E.*avg|std', axis=1) 
            #Added this to make up for the fact that cloud fraction was calculated in previous script
            df_results_combine = df_results_combine.loc[:,~df_results_combine.columns.duplicated()]               
                        
            if ("day_type","sun") not in df_results_combine.columns:
                if od == "aod":
                    df_results_combine[("day_type","sun")] = 0
                elif od == "cod":
                    df_results_combine[("day_type","sun")] = 1
                
            df_results_combine.sort_index(axis=1,level=1,inplace=True)        
            dfs_final_results.append(df_results_combine)                           
                        
            
            #Combine results for statistics
            df_combine = pd.concat(dfs_combine,axis=1)
            
            if ("day_type","sun") not in df_combine.columns:
                if od == "aod":
                    df_combine[("day_type","sun")] = 0
                elif od == "cod":
                    df_combine[("day_type","sun")] = 1            
            
            dfs_final_combine.append(df_combine)
            
            # if flags["compare"]:
            #     for day in days_result:
            #         df_day_result = df_od_result.loc[day] #,pd.IndexSlice[:,[substat,'sun']]]                    
            #         df_day_avg = df_results_combine.loc[day,pd.IndexSlice[:,substat]]                    
                    
            #         if day in days_lut:
            #             df_day_lut = df_lut_result.loc[day]
            #             #df_day_lut.sort_index(axis=1,inplace=True) #,pd.IndexSlice[:,[substat,'sun']]]
            #         else:
            #             df_day_lut = pd.DataFrame()
                    
            #         df_cams_day = df_cams.loc[day]
            #         df_cosmo_day = df_cosmo.loc[day]
            #         if not days_raw.empty and day in days_raw:
            #             df_raw_day = df_raw.loc[day]
            #             #df_raw_day = get_sun_position(df_raw_day, pvstation["lat_lon"], 80)
            #         else:
            #             df_raw_day = pd.DataFrame()
                        
            #         if df_day_result[(radname,substat)].dropna(how="all").empty:
            #             print(f"No irradiance data from {substat}, {radtype} on {day}")
            #         else:                                                    
            #             print(f"Plotting {timeres} irradiance from {od} for {substat} on {day}")
            #             if str_window_avg == "60min":
            #                 plot_irradiance_comparison(name,substat,df_day_result,df_raw_day,day,radtypes,
            #                       df_cams_day,df_cosmo_day,odstring,meas_names,plotpath)   
                                                
            #             for radtype in radtypes:
            #                 if radtype not in empty_radtypes:
            #                     plot_dni_dhi_comparison_grid(name,substat,df_day_result,df_day_avg,day,df_raw_day,days_raw,
            #                                          str_window_avg,window_avg_cf,sza_limit,radtype,df_cams_day,df_cosmo_day,
            #                                          odstring,meas_names,plotpath)
                                                                                        
            #                     if radtype == "poa":   
            #                         if (not df_day_result[(f"{odstring}_{radtype}_inv",substat)].dropna().empty or\
            #                         not df_day_lut[("Etotdown_lut_Wm2",substat)].dropna().empty) and str_window_avg == "60min":
            #                             plot_ghi_comparison(name, substat, df_day_lut, df_day_result, day, 
            #                                             radtype, odstring, str_window_avg, window_avg_cf, pyr_down_name, plotpath)
                                    
            #                         plot_ghi_comparison_grid(name,substat,df_day_result,df_day_avg,day,df_raw_day,days_raw,meas_names,
            #                                          str_window_avg,window_avg_cf,sza_limit,radtype,df_cams_day,df_cosmo_day,odstring,
            #                                          plotpath)
                         
        df_final_results = pd.concat(dfs_final_results,axis=0)
        df_final_results.sort_index(axis=0,inplace=True)
        
        pv_station[f"df_pyr_pv_ghi_{year}_{timeres}"] = pd.concat([pv_station[f"df_pyr_pv_ghi_{year}_{timeres}"],
                                  df_final_results],axis=1)
        pv_station[f"df_pyr_pv_ghi_{year}_{timeres}"] = pv_station[f"df_pyr_pv_ghi_{year}_{timeres}"]\
            .loc[:,~pv_station[f"df_pyr_pv_ghi_{year}_{timeres}"].columns.duplicated()] 
        pv_station[f"df_pyr_pv_ghi_{year}_{timeres}"].sort_index(axis=1,level=1,inplace=True)        
      
        #Combine OD types, for the averaged data
        df_final_combine = pd.concat(dfs_final_combine,axis=0)
        # df_finals_combine.columns = pd.MultiIndex.from_product([df_results_combine.columns.to_list(),[substat]],
        #                        names=['variable','substat'])   
        df_final_combine = df_final_combine[~df_final_combine.index.duplicated()]
        df_final_combine.sort_index(axis=0,inplace=True)
        dfs_substat_combine.append(df_final_combine)
    
    #This is to combine substats, for the averaged data            
    df_substat_combine = pd.concat(dfs_substat_combine,axis=1)  
    df_substat_combine = df_substat_combine.loc[:,~df_substat_combine.columns.duplicated()]
    df_substat_combine.sort_index(axis=1,level=1,inplace=True)        

    return df_substat_combine    

        
def generate_results_folders(rt_config,pyr_config,pvcal_config,path):
    """
    Generate folders for results
    
    args:
    :param rt_config: dictionary with DISORT config
    :param pyr_config: dictionary with pyranometer configuration
    :param pvcal_config: dictioanry with PV calibration configuration
    :param path: main path for saving files or plots        
    
    out:
    :return fullpath: string with label for saving folders    
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
    
    disort_config = rt_config["aerosol"]["disort_rad_res"]
    theta_res = str(disort_config["theta"]).replace('.','-')
    phi_res = str(disort_config["phi"]).replace('.','-')
    
    disort_folder_label = 'Zen_' + theta_res + '_Azi_' + phi_res    
    
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
        
    sza_label = "SZA_" + str(int(pyr_config["sza_max"]["inversion"]))
    
    dirs_exist = list_dirs(fullpath)
    fullpath = os.path.join(fullpath,sza_label)
    if sza_label not in dirs_exist:
        os.mkdir(fullpath)
                    
    model = pvcal_config["inversion"]["power_model"]
    dirs_exist = list_dirs(fullpath)
    fullpath = os.path.join(fullpath,model)  
    if model not in dirs_exist:
        os.mkdir(fullpath)
    
    eff_model = pvcal_config["eff_model"]
    dirs_exist = list_dirs(fullpath)
    fullpath = os.path.join(fullpath,eff_model)  
    if eff_model not in dirs_exist:
        os.mkdir(fullpath)
        
    T_model = pvcal_config["T_model"]["model"]
    dirs_exist = list_dirs(fullpath)
    fullpath = os.path.join(fullpath,T_model)  
    if T_model not in dirs_exist:
        os.mkdir(fullpath)        
        
    return fullpath        

def irradiance_stats_plots(name,dataframe,pyr_substats,pv_substats,year,
                        str_window_avg,num_classes,styles,flags,plotpath):
    """
    

    Parameters
    ----------
    name : string, name of PV station
    dataframe : dataframe with relevant data
    pyr_substats : dictionary with pyranometer substations
    pv_substats : dictionary with pv substations
    year : string with year under consideration    
    str_window_avg : string, width of averaging window
    num_classes : integer, number of classes
    styles : dictionary with plot styles
    flags : dictionary with booleans for plots
    plotpath : string with plot path

    Returns
    -------
    dataframe with statistics
    dataframe with deviations
    dictionary with variance classes

    """  
    
    if str_window_avg == "60min":
        name_cosmo_avg = "mean_Wm2"
        name_cosmo_std = "iqr_Wm2"
    else:
        name_cosmo_avg = f"mean_{str_window_avg}_avg"
        name_cosmo_std = f"mean_{str_window_avg}_std"    
    
    radcomps = [("DNI","Edirnorm"),("DHI","Ediffdown"),("GHI","Etotdown")]
        
    df_stats_index = dataframe.dropna(how='any',axis=0).index
                
    for radcomp in radcomps:
        
        for substat in pyr_substats:
            #Get radnames
            radname = pyr_substat_pars[substat]["name"]                    
            if "pyr" in radname:
                radnames = [radname,radname.replace('poa','down')]
            else:
                radnames = [radname]                    
            
            radtypes = []        
            for radname in radnames:
                if "poa" in radname:
                    radtypes.append("poa")                
                elif "down" in radname:
                    radtypes.append("down")                                   
                
        dfs_delta = []            
        for substat_type in pv_substats:
            for substat in pv_substats[substat_type]["data"]:
                if year in pv_substats[substat_type]["source"]:  
                    #Get radnames
                    radtype = "poa"
                                       
                    #Calculate variance classes for PV stations                                                            
                    if radcomp[0] == "GHI":
                        
                        pyrname = pv_substats[substat_type]["pyr_down"][year][1]   
                        # pyr_station = pv_substats[substat_type]["pyr_down"][year][0]
                        # radname = pyr_substats[pyr_station]["substat"][pyrname]["name"]  
                        # radname = radname.replace('poa','down')
                        #Calculate delta for LUT
                        dataframe[(f"delta_GHI_lut_pyr_{str_window_avg}_Wm2",substat)] = \
                            dataframe[(f"{radcomp[1]}_lut_{str_window_avg}_avg",substat)] - \
                            dataframe[(f'{radcomp[1]}_Wm2_{str_window_avg}_avg', pyrname)]
                            
                        dataframe[(f"delta_GHI_lut_sat_{str_window_avg}_Wm2",substat)] = \
                            dataframe[(f"{radcomp[1]}_lut_{str_window_avg}_avg",substat)] - \
                            dataframe[(f'{radcomp[1]}_{str_window_avg}_avg', 'cams')]
                            
                        dataframe[(f"delta_GHI_lut_cosmo_{str_window_avg}_Wm2",substat)] = \
                            dataframe[(f"{radcomp[1]}_lut_{str_window_avg}_avg",substat)] - \
                            dataframe[(f'{radcomp[1]}_{name_cosmo_avg}', 'cosmo')]
                            
                        #Calculate delta for OD method
                        dataframe[(f"delta_GHI_aod_pyr_{str_window_avg}_Wm2",substat)] = \
                            dataframe.loc[dataframe["day_type","sun"]==0][(f"{radcomp[1]}_{radtype}_inv_{str_window_avg}_avg",substat)] - \
                            dataframe.loc[dataframe["day_type","sun"]==0][(f'{radcomp[1]}_Wm2_{str_window_avg}_avg', pyrname)]
                            
                        dataframe[(f"delta_GHI_cod_pyr_{str_window_avg}_Wm2",substat)] = \
                            dataframe.loc[dataframe["day_type","sun"]==1][(f"{radcomp[1]}_{radtype}_inv_{str_window_avg}_avg",substat)] - \
                            dataframe.loc[dataframe["day_type","sun"]==1][(f'{radcomp[1]}_Wm2_{str_window_avg}_avg', pyrname)]
                            
                        dataframe[(f"delta_GHI_aod_sat_{str_window_avg}_Wm2",substat)] = \
                            dataframe.loc[dataframe["day_type","sun"]==0][(f"{radcomp[1]}_{radtype}_inv_{str_window_avg}_avg",substat)] - \
                            dataframe.loc[dataframe["day_type","sun"]==0][(f'{radcomp[1]}_{str_window_avg}_avg', 'cams')]
                            
                        dataframe[(f"delta_GHI_cod_sat_{str_window_avg}_Wm2",substat)] = \
                            dataframe.loc[dataframe["day_type","sun"]==1][(f"{radcomp[1]}_{radtype}_inv_{str_window_avg}_avg",substat)] - \
                            dataframe.loc[dataframe["day_type","sun"]==1][(f'{radcomp[1]}_{str_window_avg}_avg', 'cams')]
                            
                        dataframe[(f"delta_GHI_aod_cosmo_{str_window_avg}_Wm2",substat)] = \
                            dataframe.loc[dataframe["day_type","sun"]==0][(f"{radcomp[1]}_{radtype}_inv_{str_window_avg}_avg",substat)] - \
                            dataframe.loc[dataframe["day_type","sun"]==0][(f'{radcomp[1]}_{name_cosmo_avg}', 'cosmo')]
                            
                        dataframe[(f"delta_GHI_cod_cosmo_{str_window_avg}_Wm2",substat)] = \
                            dataframe.loc[dataframe["day_type","sun"]==1][(f"{radcomp[1]}_{radtype}_inv_{str_window_avg}_avg",substat)] - \
                            dataframe.loc[dataframe["day_type","sun"]==1][(f'{radcomp[1]}_{name_cosmo_avg}', 'cosmo')]
                            
                        dfs_delta.append(dataframe[[(f"delta_GHI_lut_pyr_{str_window_avg}_Wm2",substat),
                                                       (f"{radcomp[1]}_lut_{str_window_avg}_avg",substat),
                                                       (f'{radcomp[1]}_Wm2_{str_window_avg}_avg', pyrname),
                                                       (f"delta_GHI_lut_sat_{str_window_avg}_Wm2",substat),
                                                       (f'{radcomp[1]}_{str_window_avg}_avg', 'cams'),
                                                       (f"delta_GHI_lut_cosmo_{str_window_avg}_Wm2",substat),
                                                       (f'{radcomp[1]}_{name_cosmo_avg}', 'cosmo'),
                                                       (f"delta_GHI_aod_pyr_{str_window_avg}_Wm2",substat),
                                                       (f"delta_GHI_cod_pyr_{str_window_avg}_Wm2",substat),
                                                       (f"{radcomp[1]}_{radtype}_inv_{str_window_avg}_avg",substat),
                                                       (f"delta_GHI_aod_sat_{str_window_avg}_Wm2",substat),
                                                       (f"delta_GHI_cod_sat_{str_window_avg}_Wm2",substat),
                                                       (f"delta_GHI_aod_cosmo_{str_window_avg}_Wm2",substat),
                                                       (f"delta_GHI_cod_cosmo_{str_window_avg}_Wm2",substat),
                                                       ("day_type","sun")]])
                                                                                               
        dataframe.sort_index(axis=1,level=1,inplace=True)                                        
            
    if len(dfs_delta) != 0:
        df_delta = pd.concat(dfs_delta,axis=1)
        idx = df_delta.columns.to_frame()
        idx.insert(2, 'station', name)
        df_delta.columns = pd.MultiIndex.from_frame(idx) 
        
        new_cols = [col for col in df_delta.columns.levels[1]\
                            if ("Pyr" not in col) and ("CMP" not in col) 
                            and ("RT" not in col) and ("suntracker" not in col)
                            and (col != "cams") and (col != "cosmo") and (col != "sun")]            
        # if name != "MS_02":
        #     new_cols = [col.split('_')[0] for col in new_cols if ("egrid" in col)]
        # elif timeres == "15min":
        #     new_cols = [col for col in new_cols if "auew" in col]
        df_delta.columns = pd.MultiIndex.from_product(
            [new_cols,['delta_GHI_lut_pyr_Wm2','GHI_PV_lut_inv','GHI_Pyr_ref','delta_GHI_lut_sat_Wm2',
                       "GHI_sat_ref","delta_GHI_lut_cosmo_Wm2","GHI_cosmo_ref","delta_GHI_aod_pyr_Wm2",
                       "delta_GHI_cod_pyr_Wm2","GHI_PV_od_inv","delta_GHI_aod_sat_Wm2","delta_GHI_cod_sat_Wm2",
                       "delta_GHI_aod_cosmo_Wm2","delta_GHI_cod_cosmo_Wm2","day_type"],[key]], #,
            names=['substat','variable','station']).swaplevel(0,1)
    else:
        df_delta = pd.DataFrame()

    return dataframe.loc[df_stats_index], df_delta #, variance_class_dict    

# def calc_hires_lut_statistics(year,key,pv_station,substat_pars,savepath,flags,styles):
#     """
    

#     Parameters
#     ----------
#     year : TYPE
#         DESCRIPTION.
#     key : TYPE
#         DESCRIPTION.
#     pv_station : TYPE
#         DESCRIPTION.
#     substat_pars : TYPE
#         DESCRIPTION.

#     Returns
#     -------
#     None.

#     """

#     plt.ioff()
#     plt.style.use(styles["single_small"])        
    
#     res_dirs = list_dirs(savepath)
#     savepath = os.path.join(savepath,'GHI_GTI_Plots')
#     if 'GHI_GTI_Plots' not in res_dirs:
#         os.mkdir(savepath)  
        
#     stat_dirs = list_dirs(savepath)
#     savepath = os.path.join(savepath,key)
#     if key not in stat_dirs:
#         os.mkdir(savepath)       
        
#     res_dirs = list_dirs(savepath)
#     savepath = os.path.join(savepath,'Stats')        
#     if 'Stats' not in res_dirs:
#         os.mkdir(savepath)
        
#     for timeres in pv_station["timeres"]:
#         dataframe = pv_station[f"df_pyr_pv_ghi_{year.split('_')[1]}_{timeres}"]
        
#         histmin = np.floor(np.min(dataframe.Delta_Etotdown_lut_Wm2.min())/10)*10
#         histmax = np.ceil(np.min(dataframe.Delta_Etotdown_lut_Wm2.max())/10)*10
        
#         bins = np.arange(histmin,histmax,10)
    
#         fig, ax = plt.subplots()
        
#         for substat in dataframe["Delta_Etotdown_lut_Wm2"].columns:        
#             ax.hist(dataframe["Delta_Etotdown_lut_Wm2",substat],bins=bins,label=substat,alpha=0.3)
            
#         ax.legend()
#         ax.set_xlabel(r"$\Delta$GHI (W/m$^2$)")
#         ax.set_ylabel("Frequency")
        
#         if flags["titles"]:
#             ax.set_title(f"Deviation between inferred and measured GHI at {timeres} resolution, {key}, {year.split('_')[1]}")
            
#         plt.savefig(os.path.join(savepath,f"GHI_LUT_histogram_{timeres}_{key}_{year.split('_')[1]}.png"),
#                     bbox_inches = 'tight')
    
def write_results_table(key,substat,stats,pyrname,year,path):
    """
    Write results to a table

    Parameters
    ----------
    key : string, name of PV station
    string, name of substation
    stats : dictionary with statistics
    pyrname : string with name of validation measurement
    year : string with year under consideration
    path : string with path to save results

    Returns
    -------
    None.

    """
    
    #Save 
    savepath = "/" + "/".join([p for p in path.split('/')[1:-1]])
    
    filename = f"results_table_{year}.txt"
    
    model = path.split('/')[-1]
    
    if filename not in list_files(savepath):
        f = open(os.path.join(savepath,filename),'w')
        header = ["Station", "Substat", "Model", "RMSE", "MBE", "MAE", "max_Delta_plus", "max_Delta_minus", "n_delta"]
        
        header_new = "{} {:>8} {:>8} {:>9} {:>9} {:>9} {:>16} {:>16} {:>8}".format(header[0],header[1],header[2],
                                                                                      header[3],header[4],header[5],
                                                                                      header[6],header[7],header[8])
        f.write(f'{header_new}\n')
        f.close()
    
    f = open(os.path.join(savepath,filename),'a')
    f.write(f"{key} {substat:>8} {model:>10} {stats[f'RMSE_GHI_lut_{pyrname}_Wm2']:10.3f} "\
            f"{stats[f'MBE_GHI_lut_{pyrname}_Wm2']:10.3f} {stats[f'MAD_GHI_lut_{pyrname}_Wm2']:10.3f} "\
                f"{stats[f'max_Delta_GHI_plus_lut_{pyrname}_Wm2']:13.3f} {stats[f'max_Delta_GHI_minus_lut_{pyrname}_Wm2']:13.3f} "\
                    f"{stats['n_delta_lut_pyr']:10.0f}\n")
    f.close()

def calc_statistics_irradiance(key,pv_station,year,pvrad_config,pyrcal_config,folder):
    """
    

    Parameters
    ----------
    key : string with name of PV station
    pv_station : dictionary with information and data from PV station
    year : string with year under consideration        
    pvrad_config : dictionary with PV inversion configuration
    pyrcal_config : dictionary with pyranometer calibration configuration
    folder : string with folder to save results

    Returns
    -------
    dataframe with final combined statistics

    """
    
    if "stats" not in pv_station:
        pv_station.update({"stats":{}})
    pv_station["stats"].update({year:{}})
    stats = pv_station["stats"][year]
    
    dfs_stats = []
    for substat_type in pv_station["substations_pv"]:    
        for substat in pv_station["substations_pv"][substat_type]["data"]:
            if year in pv_station["substations_pv"][substat_type]["source"]:                                
                
                yrname = year.split('_')[-1]
                timeres = pv_station["substations_pv"][substat_type]["t_res_inv"]                                
                
                if timeres == "1min":
                    cams_name = "Etotdown_Wm2"
                    
                elif timeres == "15min":
                    cams_name = "Etotdown_15min_avg"
                
                pyrname = pvrad_config["pv_stations"][key][substat_type]["pyr_down"][year][1]                
                    
                pyr_station = pvrad_config["pv_stations"][key][substat_type]["pyr_down"][year][0]
                radname = pyrcal_config["pv_stations"][pyr_station]["substat"][pyrname]["name"]  
                radname = radname.replace('poa','down')
                
                #First the GHI inverted from AOD and COD - clear and cloudy separate!
                
                for odtype in ["aod","cod"]:
                    dfname = f"df_{odtype}fit_pyr_pv_{yrname}_{timeres}"
                    
                    dataframe = pv_station[dfname].loc[(pv_station[dfname][("sza","sun")]\
                          <= pvrad_config["sza_max"]["inversion"])  &\
                            (pv_station[dfname][("theta_IA",substat)] <= \
                              pvrad_config["sza_max"]["inversion"])]
                    
                    delta_GHI_od_pyr = dataframe[("Edirdown_poa_inv",substat)]\
                                + dataframe[("Ediffdown_poa_inv",substat)]\
                                - pv_station[f'df_pyr_pv_ghi_{yrname}_{timeres}']\
                                .loc[dataframe.index,(radname,pyrname)]
                                
                    delta_GHI_od_sat = dataframe[("Edirdown_poa_inv",substat)]\
                                + dataframe[("Ediffdown_poa_inv",substat)]\
                                - dataframe[(cams_name,"cams")]
                                
                    GHI_ref = pv_station[f'df_pyr_pv_ghi_{yrname}_{timeres}']\
                                 .loc[dataframe.index,(radname,pyrname)]                                                        
                    
                    rmse = (((delta_GHI_od_pyr)**2).mean())**0.5
                    mad = abs(delta_GHI_od_pyr).mean()
                    mbe = delta_GHI_od_pyr.mean()
                    
                    #Make sure that mean is using only the data for which inversion exists!!
                    mean_obs = GHI_ref.loc[~np.isnan(delta_GHI_od_pyr)].mean(axis=0)
                                        
                    rmbe = mbe/mean_obs*100
                    rrmse = rmse/mean_obs*100
                    
                    delta_max_plus = delta_GHI_od_pyr.max()
                    delta_max_minus = delta_GHI_od_pyr.min()                
                    
                    n_delta_pyr = len(delta_GHI_od_pyr.dropna())
                    
                    stats.update({substat:{}})
                    
                    stats[substat].update({f"n_delta_{odtype}_pyr":n_delta_pyr})
                    stats[substat].update({f"RMSE_GHI_{odtype}_{pyrname}_Wm2":rmse})
                    stats[substat].update({f"rRMSE_GHI_{odtype}_{pyrname}_%":rrmse})
                    stats[substat].update({f"MAD_GHI_{odtype}_{pyrname}_Wm2":mad})
                    stats[substat].update({f"MBE_GHI_{odtype}_{pyrname}_Wm2":mbe})
                    stats[substat].update({f"rMBE_GHI_{odtype}_{pyrname}_%":rmbe})
                    stats[substat].update({f"mean_GHI_{odtype}_{pyrname}_Wm2":mean_obs})
                    stats[substat].update({f"max_Delta_GHI_plus_{odtype}_{pyrname}_Wm2":delta_max_plus})
                    stats[substat].update({f"max_Delta_GHI_minus_{odtype}_{pyrname}_Wm2":delta_max_minus})
                    
                    print(f"\n{key}, {yrname}: statistics at {timeres} calculated with {n_delta_pyr} measurements")
                    print(f"RMSE for GHI inverted from {substat} via {odtype} compared to {pyrname} is {rmse:.2f} ({rrmse:.1f} %)")
                    print(f"MAE for GHI inverted from {substat} via {odtype} compared to {pyrname} is {mad:.2f}")
                    print(f"MBE for GHI inverted from {substat} via {odtype} compared to {pyrname} is {mbe:.2f} ({rmbe:.1f} %)")
                    print(f"Average pyranometer GHI was {mean_obs:.2f} W/m^2 calculated\n") #" from {GHI_ref.index}")
                                        
                    rmse = (((delta_GHI_od_sat)**2).mean())**0.5
                    mad = abs(delta_GHI_od_sat).mean()
                    mbe = delta_GHI_od_sat.mean()
                    
                    mean_obs = dataframe[(cams_name,"cams")].loc[~np.isnan(delta_GHI_od_sat)].mean(axis=0)
                    rmbe = mbe/mean_obs*100
                    rrmse = rmse/mean_obs*100
                    
                    delta_max_plus = delta_GHI_od_sat.max()
                    delta_max_minus = delta_GHI_od_sat.min()                
                    
                    n_delta_sat = len(delta_GHI_od_sat.dropna())
                    
                    stats[substat].update({"n_delta_{odtype}_sat":n_delta_pyr})
                    stats[substat].update({"RMSE_GHI_{odtype}_CAMS_Wm2":rmse})
                    stats[substat].update({"rRMSE_GHI_{odtype}_CAMS_%":rmse})
                    stats[substat].update({"MAD_GHI_{odtype}_CAMS_Wm2":mad})
                    stats[substat].update({"MBE_GHI_{odtype}_CAMS_Wm2":mbe})
                    stats[substat].update({"rMBE_GHI_{odtype}_CAMS_%":rmbe})
                    stats[substat].update({f"mean_GHI_{odtype}_CAMS_Wm2":mean_obs})
                    stats[substat].update({"max_Delta_GHI_plus_{odtype}_CAMS_Wm2":delta_max_plus})
                    stats[substat].update({"max_Delta_GHI_minus_{odtype}_CAMS_Wm2":delta_max_minus})
                    
                    print(f"{key}, {yrname}: statistics at {timeres} calculated with {n_delta_pyr} measurements")
                    print(f"RMSE for GHI inverted from {substat} via {odtype} compared to CAMS is {rmse:.2f} ({rrmse:.1f} %)")
                    print(f"MAE for GHI inverted from {substat} via {odtype} compared to CAMS is {mad}")
                    print(f"MBE for GHI inverted from {substat} via {odtype} compared to CAMS is {mbe:.2f} ({rmbe:.1f} %)")
                    print(f"Average CAMS GHI was {mean_obs:.2f} W/m^2\n")

                #for odtype in ["aod","cod"]:
                    #dfname = f"df_{odtype}fit_pyr_pv_{yrname}_{timeres}"
                    # dataframe = pv_station[dfname].loc[(pv_station[dfname][("sza","sun")]\
                    #       <= pvrad_config["sza_max"]["inversion"])]#  &\
                           # (pv_station[dfname][("theta_IA",substat)] <= \
                           #  pvrad_config["sza_max"]["inversion"])]
                    pv_station[dfname].loc[dataframe.index,(f"delta_GHI_{odtype}_pyr_Wm2",substat)] = \
                        delta_GHI_od_pyr.loc[dataframe.index]
                    pv_station[dfname].loc[dataframe.index,(f"delta_GHI_{odtype}_sat_Wm2",substat)] = \
                        delta_GHI_od_sat.loc[dataframe.index]
                
                #Second the GHI inverted directly from irradiance via MYSTIC LUT
                dfname = f'df_pyr_pv_ghi_{yrname}_{timeres}'
                
                dataframe = pv_station[dfname].loc[(pv_station[dfname][("sza","sun")]\
                          <= pvrad_config["sza_max"]["inversion"])  &\
                             (pv_station[dfname][("theta_IA",substat)] <= \
                              pvrad_config["sza_max"]["inversion"])]
                                
                delta_GHI_pyr = dataframe[("Etotdown_lut_Wm2",substat)]\
                    - dataframe[(radname,pyrname)]
                    
                #GHI_ref = dataframe[(radname,pyrname)]
                    
                rmse = (((delta_GHI_pyr)**2).mean())**0.5
                mad = abs(delta_GHI_pyr).mean()
                mbe = delta_GHI_pyr.mean()
                
                mean_obs = dataframe[(radname,pyrname)].loc[~np.isnan(delta_GHI_pyr)].mean(axis=0)
                rmbe = mbe/mean_obs*100
                rrmse = rmse/mean_obs*100
                                
                delta_max_plus = delta_GHI_pyr.max()
                delta_max_minus = delta_GHI_pyr.min()                
                
                n_delta_pyr = len(delta_GHI_pyr.dropna())                                
                
                stats[substat].update({"n_delta_lut_pyr":n_delta_pyr})
                stats[substat].update({f"RMSE_GHI_lut_{pyrname}_Wm2":rmse})
                stats[substat].update({f"rRMSE_GHI_lut_{pyrname}_%":rrmse})
                stats[substat].update({f"MAD_GHI_lut_{pyrname}_Wm2":mad})
                stats[substat].update({f"MBE_GHI_lut_{pyrname}_Wm2":mbe})
                stats[substat].update({f"rMBE_GHI_lut_{pyrname}_%":rmbe})
                stats[substat].update({f"mean_GHI_lut_{pyrname}_Wm2":mean_obs})
                stats[substat].update({f"max_Delta_GHI_plus_lut_{pyrname}_Wm2":delta_max_plus})
                stats[substat].update({f"max_Delta_GHI_minus_lut_{pyrname}_Wm2":delta_max_minus})
                
                print(f"{key}, {yrname}: statistics at {timeres} calculated with {n_delta_pyr} measurements")
                print(f"RMSE for GHI inverted from {substat} via LUT compared to {pyrname} is {rmse:.2f} ({rrmse:.1f} %)")
                print(f"MAE for GHI inverted from {substat} via LUT compared to {pyrname} is {mad}")
                print(f"MBE for GHI inverted from {substat} via LUT compared to {pyrname} is {mbe:.2f} ({rmbe:.1f} %)")
                print(f"Average pyranometer GHI was {mean_obs:.2f} W/m^2\n")

                
                #Assign delta to the dataframe
                pv_station[dfname].loc[dataframe.index,("delta_GHI_lut_pyr_Wm2",substat)] = delta_GHI_pyr                                
                
                delta_GHI_sat = dataframe[("Etotdown_lut_Wm2",substat)]\
                    - dataframe[(cams_name,"cams")]
                    
                rmse = (((delta_GHI_sat)**2).mean())**0.5
                mad = abs(delta_GHI_sat).mean()
                mbe = delta_GHI_sat.mean()
                
                mean_obs = dataframe[(cams_name,"cams")].loc[~np.isnan(delta_GHI_sat)].mean(axis=0)
                rmbe = mbe/mean_obs*100
                rrmse = rmse/mean_obs*100                
                
                delta_max_plus = delta_GHI_sat.max()
                delta_max_minus = delta_GHI_sat.min()                
                
                n_delta_sat = len(delta_GHI_sat.dropna())                                
                                
                stats[substat].update({"n_delta_lut_sat":n_delta_sat})
                stats[substat].update({"RMSE_GHI_lut_CAMS_Wm2":rmse})
                stats[substat].update({"rRMSE_GHI_lut_CAMS_%":rrmse})
                stats[substat].update({"MAD_GHI_lut_CAMS_Wm2":mad})
                stats[substat].update({"MBE_GHI_lut_CAMS_Wm2":mbe})
                stats[substat].update({"rMBE_GHI_lut_CAMS_%":rmbe})
                stats[substat].update({"mean_GHI_lut_CAMS_Wm2":mean_obs})
                stats[substat].update({"max_Delta_lut_GHI_plus_CAMS_Wm2":delta_max_plus})
                stats[substat].update({"max_Delta_lut_GHI_minus_CAMS_Wm2":delta_max_minus})
                                
                print(f"{key}, {yrname}: statistics at {timeres} calculated with {n_delta_sat} measurements")
                print(f"RMSE for GHI inverted from {substat} via LUT compared to CAMS is {rmse:.2f} ({rrmse:.1f} %)")
                print(f"MAE for GHI inverted from {substat} via LUT compared to CAMS is {mad}")
                print(f"MBE for GHI inverted from {substat} via LUT compared to CAMS is {mbe:.2f} ({rmbe:.1f} %)")
                print(f"Average CAMS GHI was {mean_obs:.2f} W/m^2\n")
                
                #Assign delta to the dataframe
                pv_station[dfname].loc[dataframe.index,("delta_GHI_lut_sat_Wm2",substat)] = delta_GHI_sat
                
                if f"df_stats_{timeres}_{yrname}" not in pv_station:
                    pv_station[f"df_stats_{timeres}_{yrname}"] = pv_station[dfname].loc[dataframe.index,
                        [("delta_GHI_lut_pyr_Wm2",substat),("Etotdown_lut_Wm2",substat),(radname,pyrname),
                         ("delta_GHI_lut_sat_Wm2",substat),(cams_name,"cams")]]
                else:
                    pv_station[f"df_stats_{timeres}_{yrname}"] = pd.concat([pv_station[f"df_stats_{timeres}_{yrname}"],
                            pv_station[dfname].loc[dataframe.index,[("delta_GHI_lut_pyr_Wm2",substat),
                                ("Etotdown_lut_Wm2",substat),(radname,pyrname),
                                ("delta_GHI_lut_sat_Wm2",substat),(cams_name,"cams")]]],axis=1)            
                
                #Assign OD-based deltas to dataframe
                for odtype in ["aod","cod"]:
                    dfname = f"df_{odtype}fit_pyr_pv_{yrname}_{timeres}"
                    dataframe = pv_station[dfname].loc[(pv_station[dfname][("sza","sun")]\
                          <= pvrad_config["sza_max"]["inversion"])   &\
                             (pv_station[dfname][("theta_IA",substat)] <= \
                              pvrad_config["sza_max"]["inversion"])]
                    
                    Etot = pd.concat([pv_station[dfname].loc[dataframe.index,("Edirdown_poa_inv",substat)] 
                                    + pv_station[dfname].loc[dataframe.index,("Ediffdown_poa_inv",substat)],
                                    pv_station[dfname].loc[dataframe.index,[(f"delta_GHI_{odtype}_pyr_Wm2",substat),                                
                                (f"delta_GHI_{odtype}_sat_Wm2",substat)]]],axis=1)
                    
                    # pv_station[f"df_stats_{timeres}_{yrname}"] = pd.concat([pv_station[f"df_stats_{timeres}_{yrname}"],
                    #         pv_station[dfname].loc[dataframe.index,[("delta_GHI_od_pyr_Wm2",substat),                                
                    #             ("delta_GHI_od_sat_Wm2",substat)]]],axis=1) 
                    
                
                    Etot.rename(columns={0:(f"Etotdown_{odtype}_inv",substat)},inplace=True)
                    pv_station[f"df_stats_{timeres}_{yrname}"] = pd.concat([pv_station[f"df_stats_{timeres}_{yrname}"],
                            Etot],axis=1) 
                                
                #Write stats results to text file
                write_results_table(key,substat,stats[substat],pyrname,year,folder)
                
                df_stats = pd.DataFrame(pv_station["stats"][year][substat],index=[key])
                new_columns = ["n_delta_od_pyr"]
                new_columns.extend(["_".join([val for i, val in enumerate(col.split('_')) 
                             if i != len(col.split('_')) - 2]) for col in df_stats.columns[1:6]])
                new_columns.extend(df_stats.columns[6:])
                if "egrid" in substat and key != "MS_02":
                    new_substat = substat.split('_')[0]
                else:
                    new_substat = substat
                df_stats.columns = pd.MultiIndex.from_product([new_columns,[new_substat]],
                             names=['variable', 'substat'])
                df_stats.index.rename('station',inplace=True)
                dfs_stats.append(df_stats)
            #else: return pd.DataFrame() #concat(dfs_stats,axis=1)
                
    if len(dfs_stats) != 0:
        df_stats_final = pd.concat(dfs_stats,axis=1)
        for timeres in pvrad_config["timeres_comparison"]:
            if f"df_stats_{timeres}_{yrname}" in pv_station:
                pv_station[f"df_stats_{timeres}_{yrname}"].columns.rename(["variable","substat"],inplace=True)                    
                idx = pv_station[f"df_stats_{timeres}_{yrname}"].columns.to_frame()
                idx.insert(2, 'station', key)
                pv_station[f"df_stats_{timeres}_{yrname}"].columns = pd.MultiIndex.from_frame(idx) 
                
                new_cols = [col for col in pv_station[f"df_stats_{timeres}_{yrname}"].columns.levels[1]\
                            if ("Pyr" not in col) and ("CMP" not in col) 
                            and ("RT" not in col) and (col != "suntracker") and (col != "cams")]            
                if timeres == "1min" and key != "MS_02":
                    new_cols = [col.split('_')[0] for col in new_cols if ("egrid" in col)]
                # elif timeres == "15min":
                #     new_cols = [col for col in new_cols if "auew" in col]
                pv_station[f"df_stats_{timeres}_{yrname}"].columns = pd.MultiIndex.from_product(
                    [new_cols,['delta_GHI_lut_pyr_Wm2','GHI_PV_lut_inv','GHI_Pyr_ref',
                               'delta_GHI_lut_sat_Wm2',"GHI_sat_ref","GHI_PV_aod_inv",
                               "delta_GHI_aod_pyr_Wm2","delta_GHI_aod_sat_Wm2",
                               "GHI_PV_cod_inv","delta_GHI_cod_pyr_Wm2","delta_GHI_cod_sat_Wm2"],[key]], #,
                    names=['substat','variable','station']).swaplevel(0,1)
    else: df_stats_final = pd.DataFrame()
    
        
    return df_stats_final

def combined_stats(dict_combo_stats,year,timeres_list,window_avgs):
    """

    Calculate combined stats for different averaging times    

    Parameters
    ----------
    dict_combo_stats : dictionary with combined stats for different averaging times
    year : string, year under consideration
    timeres_list : list of data resolutions
    window_avgs : list of averaging window times

    Returns
    -------
    None.

    """
    for timeres in timeres_list:        
        if f"df_delta_all_{timeres}" in dict_combo_stats:
            #Stack all values on top of each other for combined stats
            df_delta_all = dict_combo_stats[f"df_delta_all_{timeres}"].stack(dropna=True)
            
            dict_combo_stats.update({timeres:{}})            
            for inv_type in ["aod","cod","lut"]:
                rmse = ((((df_delta_all[f"delta_GHI_{inv_type}_pyr_Wm2"].stack())**2).mean())**0.5)
                mad = abs(df_delta_all[f"delta_GHI_{inv_type}_pyr_Wm2"].stack()).mean()
                mbe = df_delta_all[f"delta_GHI_{inv_type}_pyr_Wm2"].stack().mean()
                
                mean_obs = df_delta_all.loc[df_delta_all[f"delta_GHI_{inv_type}_pyr_Wm2"].stack().index].GHI_Pyr_ref.stack().mean()
                rmbe = mbe/mean_obs*100
                rrmse = rmse/mean_obs*100                   
                
                delta_max_plus = df_delta_all[f"delta_GHI_{inv_type}_pyr_Wm2"].stack().max()
                delta_max_minus = df_delta_all[f"delta_GHI_{inv_type}_pyr_Wm2"].stack().min()
                
                n_delta_pyr = len(df_delta_all[f"delta_GHI_{inv_type}_pyr_Wm2"].stack().dropna())
                                
                dict_combo_stats[timeres].update({f"n_delta_{inv_type}_pyr":n_delta_pyr})
                dict_combo_stats[timeres].update({f"RMSE_GHI_{inv_type}_pyr_Wm2":rmse})
                dict_combo_stats[timeres].update({f"rRMSE_GHI_{inv_type}_pyr_%":rrmse})
                dict_combo_stats[timeres].update({f"MAD_GHI_{inv_type}_pyr_Wm2":mad})
                dict_combo_stats[timeres].update({f"MBE_GHI_{inv_type}_pyr_Wm2":mbe})
                dict_combo_stats[timeres].update({f"rMBE_GHI_{inv_type}_pyr_%":rmbe})
                dict_combo_stats[timeres].update({f"mean_GHI_{inv_type}_pyr_Wm2":mean_obs})
                dict_combo_stats[timeres].update({f"max_Delta_GHI_{inv_type}_pyr_plus_Wm2":delta_max_plus})
                dict_combo_stats[timeres].update({f"max_Delta_GHI_{inv_type}_pyr_minus_Wm2":delta_max_minus})
                
                print(f"\n{year}: combined statistics at {timeres} from "\
                      f"{dict_combo_stats[f'df_delta_all_{timeres}'].columns.levels[2].to_list()}"\
                      f" calculated with {n_delta_pyr} measurements")
                print(f"Combined RMSE for GHI via {inv_type} in {year} is {rmse:.2f} ({rrmse:.1f}%)")
                print(f"Combined MAE for GHI via {inv_type} in {year} is {mad:.2f}")
                print(f"Combined MBE for GHI via {inv_type} in {year} is {mbe:.2f} ({rmbe:.1f}%)")
                print(f"Average pyranometer GHI was {mean_obs:.2f} W/m^2") #" from {df_delta_all.loc[~np.isnan(df_delta_all[f'delta_GHI_{inv_type}_pyr_Wm2']).stack().values].index}")
                
                rmse = ((((df_delta_all[f"delta_GHI_{inv_type}_sat_Wm2"].stack())**2).mean())**0.5)
                mad = abs(df_delta_all[f"delta_GHI_{inv_type}_sat_Wm2"].stack()).mean()
                mbe = df_delta_all[f"delta_GHI_{inv_type}_sat_Wm2"].stack().mean()
                
                mean_obs = df_delta_all.loc[df_delta_all[f"delta_GHI_{inv_type}_sat_Wm2"].stack().index].GHI_sat_ref.stack().mean()
                rmbe = mbe/mean_obs*100
                rrmse = rmse/mean_obs*100  
                
                delta_max_plus = df_delta_all[f"delta_GHI_{inv_type}_sat_Wm2"].stack().max()
                delta_max_minus = df_delta_all[f"delta_GHI_{inv_type}_sat_Wm2"].stack().min()
                
                n_delta_sat = len(df_delta_all[f"delta_GHI_{inv_type}_sat_Wm2"].stack().dropna())
                            
                dict_combo_stats[timeres].update({f"n_delta_{inv_type}_sat":n_delta_sat})
                dict_combo_stats[timeres].update({f"RMSE_GHI_{inv_type}_sat_Wm2":rmse})
                dict_combo_stats[timeres].update({f"rRMSE_GHI_{inv_type}_sat_%":rrmse})
                dict_combo_stats[timeres].update({f"MAD_GHI_{inv_type}_sat_Wm2":mad})
                dict_combo_stats[timeres].update({f"MBE_GHI_{inv_type}_sat_Wm2":mbe})
                dict_combo_stats[timeres].update({f"rMBE_GHI_{inv_type}_sat_%":rmbe})
                dict_combo_stats[timeres].update({f"mean_GHI_{inv_type}_sat_Wm2":mean_obs})
                dict_combo_stats[timeres].update({f"max_Delta_GHI_{inv_type}_sat_plus_Wm2":delta_max_plus})
                dict_combo_stats[timeres].update({f"max_Delta_GHI_{inv_type}_sat_minus_Wm2":delta_max_minus})                    
                
                print(f"Combined RMSE for GHI via {inv_type} vs. CAMS in {year} is {rmse:.2f} ({rrmse:.1f}%)")
                print(f"Combined MAE for GHI via {inv_type} vs. CAMS in {year} is {mad:.2f}")
                print(f"Combined MBE for GHI via {inv_type} vs. CAMS in {year} is {mbe:.2f} ({rmbe:.1f}%)")    
                print(f"Average CAMS GHI was {mean_obs:.2f} W/m^2")                        
            
    for window_avg in window_avgs:
        df_delta_all = dict_combo_stats[f"df_delta_all_{window_avg}"].stack(dropna=True)
        dict_combo_stats.update({window_avg:{}})
        for inv_type in ["aod","cod","lut"]:            
            rmse = ((((df_delta_all[f"delta_GHI_{inv_type}_pyr_Wm2"].stack())**2).mean())**0.5)
            mad = abs(df_delta_all[f"delta_GHI_{inv_type}_pyr_Wm2"].stack()).mean()
            mbe = df_delta_all[f"delta_GHI_{inv_type}_pyr_Wm2"].stack().mean()
            
            mean_obs = df_delta_all.GHI_Pyr_ref.stack().loc[~np.isnan(df_delta_all[f"delta_GHI_{inv_type}_pyr_Wm2"]).stack()].mean()
            rmbe = mbe/mean_obs*100
            rrmse = rmse/mean_obs*100 
            
            delta_max_plus = df_delta_all[f"delta_GHI_{inv_type}_pyr_Wm2"].stack().max()
            delta_max_minus = df_delta_all[f"delta_GHI_{inv_type}_pyr_Wm2"].stack().min()
            
            n_delta_pyr = len(df_delta_all[f"delta_GHI_{inv_type}_pyr_Wm2"].stack().dropna())
            
            dict_combo_stats[window_avg].update({f"n_delta_{inv_type}_pyr":n_delta_pyr})
            dict_combo_stats[window_avg].update({f"RMSE_GHI_{inv_type}_pyr_Wm2":rmse})
            dict_combo_stats[window_avg].update({f"rRMSE_GHI_{inv_type}_pyr_%":rrmse})
            dict_combo_stats[window_avg].update({f"MAD_GHI_{inv_type}_pyr_Wm2":mad})
            dict_combo_stats[window_avg].update({f"MBE_GHI_{inv_type}_pyr_Wm2":mbe})
            dict_combo_stats[window_avg].update({f"rMBE_GHI_{inv_type}_pyr_%":rmbe})
            dict_combo_stats[window_avg].update({f"mean_GHI_{inv_type}_pyr_Wm2":mean_obs})
            dict_combo_stats[window_avg].update({f"max_Delta_GHI_{inv_type}_pyr_plus_Wm2":delta_max_plus})
            dict_combo_stats[window_avg].update({f"max_Delta_GHI_{inv_type}_pyr_minus_Wm2":delta_max_minus})
            
            print(f"\n{year}: combined {inv_type} statistics at {window_avg} from "\
                  f"{dict_combo_stats[f'df_delta_all_{window_avg}'].columns.levels[2].to_list()}")
            print(f"Pyranometer stats calculated with {n_delta_pyr} measurements")
            print(f"Combined RMSE for GHI in {year} is {rmse:.2f} ({rrmse:.1f}%)")
            print(f"Combined MAE for GHI in {year} is {mad:.2f}")
            print(f"Combined MBE for GHI in {year} is {mbe:.2f} ({rmbe:.1f}%)")
            print(f"Average pyranometer GHI was {mean_obs:.2f} W/m^2")
            
            rmse = ((((df_delta_all[f"delta_GHI_{inv_type}_sat_Wm2"].stack())**2).mean())**0.5)
            mad = abs(df_delta_all[f"delta_GHI_{inv_type}_sat_Wm2"].stack()).mean()
            mbe = df_delta_all[f"delta_GHI_{inv_type}_sat_Wm2"].stack().mean()
            
            mean_obs = df_delta_all.GHI_sat_ref.stack().loc[~np.isnan(df_delta_all[f"delta_GHI_{inv_type}_sat_Wm2"]).stack()].mean()
            rmbe = mbe/mean_obs*100
            rrmse = rmse/mean_obs*100 
            
            delta_max_plus = df_delta_all[f"delta_GHI_{inv_type}_sat_Wm2"].stack().max()
            delta_max_minus = df_delta_all[f"delta_GHI_{inv_type}_sat_Wm2"].stack().min()
            
            n_delta_sat = len(df_delta_all[f"delta_GHI_{inv_type}_sat_Wm2"].stack().dropna())
                        
            dict_combo_stats[window_avg].update({f"n_delta_{inv_type}_sat":n_delta_sat})
            dict_combo_stats[window_avg].update({f"RMSE_GHI_{inv_type}_sat_Wm2":rmse})
            dict_combo_stats[window_avg].update({f"rRMSE_GHI_{inv_type}_sat_%":rrmse})
            dict_combo_stats[window_avg].update({f"MAD_GHI_{inv_type}_sat_Wm2":mad})
            dict_combo_stats[window_avg].update({f"MBE_GHI_{inv_type}_sat_Wm2":mbe})
            dict_combo_stats[window_avg].update({f"rMBE_GHI_{inv_type}_sat_%":rmbe})
            dict_combo_stats[window_avg].update({f"mean_GHI_{inv_type}_sat_Wm2":mean_obs})
            dict_combo_stats[window_avg].update({f"max_Delta_GHI_{inv_type}_sat_plus_Wm2":delta_max_plus})
            dict_combo_stats[window_avg].update({f"max_Delta_GHI_{inv_type}_sat_minus_Wm2":delta_max_minus})                    
            
            print(f"CAMS stats calculated with {n_delta_sat} measurements")
            print(f"Combined RMSE for GHI vs. CAMS in {year} is {rmse:.2f} ({rrmse:.1f}%)")
            print(f"Combined MAE for GHI vs. CAMS in {year} is {mad:.2f}")
            print(f"Combined MBE for GHI vs. CAMS in {year} is {mbe:.2f} ({rmbe:.1f}%)")  
            print(f"Average CAMS GHI was {mean_obs:.2f} W/m^2")
            
            rmse = ((((df_delta_all[f"delta_GHI_{inv_type}_cosmo_Wm2"].stack())**2).mean())**0.5)
            mad = abs(df_delta_all[f"delta_GHI_{inv_type}_cosmo_Wm2"].stack()).mean()
            mbe = df_delta_all[f"delta_GHI_{inv_type}_cosmo_Wm2"].stack().mean()
            
            mean_obs = df_delta_all.GHI_cosmo_ref.stack().loc[~np.isnan(df_delta_all[f"delta_GHI_{inv_type}_cosmo_Wm2"]).stack()].mean()
            rmbe = mbe/mean_obs*100
            rrmse = rmse/mean_obs*100 
            
            delta_max_plus = df_delta_all[f"delta_GHI_{inv_type}_sat_Wm2"].stack().max()
            delta_max_minus = df_delta_all[f"delta_GHI_{inv_type}_sat_Wm2"].stack().min()
            
            n_delta_cosmo = len(df_delta_all[f"delta_GHI_{inv_type}_cosmo_Wm2"].stack().dropna())
                        
            dict_combo_stats[window_avg].update({f"n_delta_{inv_type}_cosmo":n_delta_cosmo})
            dict_combo_stats[window_avg].update({f"RMSE_GHI_{inv_type}_cosmo_Wm2":rmse})
            dict_combo_stats[window_avg].update({f"rRMSE_GHI_{inv_type}_cosmo_%":rrmse})
            dict_combo_stats[window_avg].update({f"MAD_GHI_{inv_type}_cosmo_Wm2":mad})
            dict_combo_stats[window_avg].update({f"MBE_GHI_{inv_type}_cosmo_Wm2":mbe})
            dict_combo_stats[window_avg].update({f"rMBE_GHI_{inv_type}_cosmo_%":rmbe})
            dict_combo_stats[window_avg].update({f"mean_GHI_{inv_type}_cosmo_Wm2":mean_obs})
            dict_combo_stats[window_avg].update({f"max_Delta_GHI_{inv_type}_cosmo_plus_Wm2":delta_max_plus})
            dict_combo_stats[window_avg].update({f"max_Delta_GHI_{inv_type}_cosmo_minus_Wm2":delta_max_minus})                    
            
            print(f"COSMO stats calculated with {n_delta_cosmo} measurements")
            print(f"Combined RMSE for GHI vs. COSMO in {year} is {rmse:.2f} ({rrmse:.1f}%)")
            print(f"Combined MAE for GHI vs. COSMO in {year} is {mad:.2f}")
            print(f"Combined MBE for GHI vs. COSMO in {year} is {mbe:.2f} ({rmbe:.1f}%)")  
            print(f"Average COSMO GHI was {mean_obs:.2f} W/m^2")

def plot_all_ghi_combined_scatter(dict_stats,list_stations,pvrad_config,T_model,folder,title_flag,window_avgs):
    """
    Plot combined scatter plots for all stations

    Parameters
    ----------
    dict_stats : dictionary with combined statistics for different averaging times
    list_stations : list of PV stations
    pvrad_config : dictionary with PV inversion configuration
    T_model : string, temperature model used
    folder : string, folder to save plots
    title_flag : boolean, whether to add title to plots
    window_avgs : list of averaging windows

    Returns
    -------
    None.
    """
    
    plt.ioff()
    plt.style.use('my_presi_grid')        
    
    res_dirs = list_dirs(folder)
    savepath = os.path.join(folder,'GHI_GTI_Plots')
    if 'GHI_GTI_Plots' not in res_dirs:
        os.mkdir(savepath)    
        
    # res_dirs = list_dirs(savepath)
    # savepath = os.path.join(savepath,'Scatter')
    # if 'Scatter' not in res_dirs:
    #     os.mkdir(savepath)
        
    years = ["mk_" + campaign.split('_')[1] for campaign in pvrad_config["calibration_source"]]    
    stations_label = '_'.join(["".join(s.split('_')) for s in list_stations])
    
    tres_list = pvrad_config["timeres_comparison"] + window_avgs
    data_types = [("Pyr","pyr","pyranometer"),("sat","sat","CAMS")]        
    inv_types = ["aod","cod","lut"]   
    
    #Combined plot with all three time resolutions
    for data_type, data_type_short, data_label in data_types:
        for inv_type in inv_types:
        
            #1. Plot comparing inverted irradiance with that from pyranometers
            fig, axs = plt.subplots(len(tres_list),len(years),sharex='all',sharey='all')
            #cbar_ax = fig.add_axes([.76, .25, .015, .45])                    
                                                
            print(f"Plotting combined frequency scatter plot for {inv_type}, {data_label}...please wait....")
            # plot_data = []
            # norm = []
            max_ghi = 0 #; min_z = 500.; max_z = 0.
            for i, ax in enumerate(axs.flatten()):            
                year = years[np.fmod(i,2)]
                timeres = tres_list[int((i - np.fmod(i,2))/2)]
                                
                if timeres in window_avgs and "od" in inv_type:
                    if inv_type == "aod":
                        day_type = 0
                    elif inv_type == "cod":
                        day_type = 1
                    stacked_data = dict_stats[year][f"df_delta_all_{timeres}"].stack()\
                        .loc[:,["GHI_PV_od_inv",f"GHI_{data_type}_ref","day_type"]].stack()
                    ghi_data = stacked_data.loc[stacked_data["day_type"] == day_type].dropna()
                    
                    ghi_ref = ghi_data[f"GHI_{data_type}_ref"].values.flatten()
                    ghi_inv = ghi_data["GHI_PV_od_inv"].values.flatten()
                else:                    
                    ghi_data = dict_stats[year][f"df_delta_all_{timeres}"].stack()\
                        .loc[:,[f"GHI_PV_{inv_type}_inv",f"GHI_{data_type}_ref"]].stack().dropna(how='any')
                    
                    ghi_ref = ghi_data[f"GHI_{data_type}_ref"].values.flatten()
                    ghi_inv = ghi_data[f"GHI_PV_{inv_type}_inv"].values.flatten()
                    
                xy = np.vstack([ghi_ref,ghi_inv])
                z = gaussian_kde(xy)(xy)
                idx = z.argsort()
                
            #     # sc = sns.kdeplot(x=ghi_ref,y=ghi_inv,cbar=i==0,ax=ax,cmap="icefire",fill=True,#bins=100,
            #     #   cbar_kws={'label': 'PDF'},
            #     #   cbar_ax=None if i else cbar_ax)
                
            #     # sc = sns.histplot(x=ghi_ref,y=ghi_inv,cbar=i==0,ax=ax,bins=100,cmap="icefire",
            #     #       cbar_ax=None if i else cbar_ax,cbar_kws={'label': 'Frequency'})
                                
            #     #if i == 0:
            #         #sc.figure.axes[-1].yaxis.set_ticks(sc.figure.axes[-1].get_ylim())                    
            #         #sc.figure.axes[-1].yaxis.set_ticklabels(["Low", "High"])
            #         #sc.figure.axes[-1].set_ylabel("PDF",size=16,labelpad=-18)
                    
                max_ghi = np.max([max_ghi,ghi_ref.max(),ghi_inv.max()])
                max_ghi = np.ceil(max_ghi/100)*100    
                # print(f"{np.min(z)},{np.max(z)}")
                # max_z = np.max([max_z,np.max(z)])
                # min_z = np.min([min_z,np.min(z)])
                
                sc = ax.scatter(ghi_ref[idx],ghi_inv[idx], s=5, c=z[idx], 
                                cmap="plasma")
                
                ax.plot([0, 1], [0, 1], ls = '--', transform=ax.transAxes,c='k')
                #ax.set_title(f"{year},{timeres}",fontsize=14)
                                
                print(f"Using {dict_stats[year][timeres][f'n_delta_{inv_type}_{data_type_short}']} data points for {timeres}, {year} plot")
                ax.annotate(rf"MBE = {dict_stats[year][timeres][f'MBE_GHI_{inv_type}_{data_type_short}_Wm2']:.2f} W m$^{{-2}}$" \
                            rf" ({dict_stats[year][timeres][f'rMBE_GHI_{inv_type}_{data_type_short}_%']:.1f} %)" "\n" \
                            rf"RMSE = {dict_stats[year][timeres][f'RMSE_GHI_{inv_type}_{data_type_short}_Wm2']:.2f} W m$^{{-2}}$" \
                                rf" ({dict_stats[year][timeres][f'rRMSE_GHI_{inv_type}_{data_type_short}_%']:.1f} %)" "\n"\
                                    r"$\langle G_\mathrm{ref} \rangle$ ="\
                                    rf" {dict_stats[year][timeres][f'mean_GHI_{inv_type}_{data_type_short}_Wm2']:.2f} W m$^{{-2}}$" "\n"\
                            rf"n = ${dict_stats[year][timeres][f'n_delta_{inv_type}_{data_type_short}']:.0f}$",
                          xy=(0.05,0.63),xycoords='axes fraction',fontsize=7,color='k',
                          bbox = dict(facecolor='lightgrey',alpha=0.3),
                          horizontalalignment='left',multialignment='left')                     
                    
            fig.subplots_adjust(wspace=-0.4,hspace=0.15)    
            cb = fig.colorbar(sc,ticks=[np.min(z),np.max(z)], 
                              ax=axs[:3], shrink=0.6,
                                aspect=20) 
            #print(f"{min_z},{max_z}")

            #cb.set_ticks()
            #cb.set_ticklabels([f"{val:.2f}" for val in cb.get_ticks()*1e5])
            cb.set_ticklabels(["Low", "High"])  # horizontal colorbar
            cb.ax.tick_params(labelsize=14) 
            cb.set_label("PDF", labelpad=-18, 
                         fontsize=14)
            
            #Set axis limits
            for i, ax in enumerate(axs.flatten()):
                ax.set_xlim([0.,max_ghi])
                ax.set_ylim([0.,max_ghi])
                ax.set_aspect('equal')
                ax.grid(False)
                # if max_gti < 1400:
                ax.set_xticks([0,500,1000])
                # else:
                #     ax.set_xticks([0,400,800,1200,1400])
                if i == 4:
                    ax.set_xlabel(rf"$G_\mathrm{{tot,{data_label}}}^{{\downarrow}}$ (W m$^{{-2}}$)",position=(1.1,0))
                if i == 2:
                    ax.set_ylabel(r"$G_\mathrm{{tot,PV,inv}}^{{\downarrow}}$ (W m$^{-2}$)")
                            
            # fig.add_subplot(111, frameon=False)
            # plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            # plt.grid(False)
            
            plt.savefig(os.path.join(savepath,f"ghi_scatter_hist_combo_all_{inv_type}_{data_label}_{T_model['model']}_"\
                      f"{T_model['type']}_{stations_label}.png"),bbox_inches = 'tight')  
                
            plt.close(fig)            
            
            
    for inv_type in inv_types:
        for timeres in window_avgs:
            print(f"Plotting combined frequency scatter plot for {inv_type}, {timeres}, COSMO...please wait....")

            #3. Plot comparing inverted irradiance with that from cosmo
            fig, axs = plt.subplots(1,len(years),sharex='all',sharey='all')
            #cbar_ax = fig.add_axes([.27, .75, .43,.015]) 
                                
            max_ghi = 0. #; min_z = 500.; max_z = 0.
            for i, ax in enumerate(axs.flatten()):            
                year = years[i]
                
                if "od" in inv_type:
                    if inv_type == "aod":
                        day_type = 0
                    elif inv_type == "cod":
                        day_type = 1
                    stacked_data = dict_stats[year][f"df_delta_all_{timeres}"].stack()\
                        .loc[:,["GHI_PV_od_inv","GHI_cosmo_ref","day_type"]].stack()
                    ghi_data = stacked_data.loc[stacked_data["day_type"] == day_type].dropna()
                    
                    ghi_ref = ghi_data["GHI_cosmo_ref"].values.flatten()
                    ghi_inv = ghi_data["GHI_PV_od_inv"].values.flatten()
                else:                    
                    ghi_data = dict_stats[year][f"df_delta_all_{timeres}"].stack()\
                        .loc[:,[f"GHI_PV_{inv_type}_inv","GHI_cosmo_ref"]].stack().dropna(how='any')
                    
                    ghi_ref = ghi_data["GHI_cosmo_ref"].values.flatten()
                    ghi_inv = ghi_data[f"GHI_PV_{inv_type}_inv"].values.flatten()
                xy = np.vstack([ghi_ref,ghi_inv])
                z = gaussian_kde(xy)(xy)
                idx = z.argsort()
                
                max_ghi = np.max([max_ghi,ghi_ref.max(),ghi_inv.max()])
                max_ghi = np.ceil(max_ghi/100)*100  
                # print(f"{np.min(z)},{np.max(z)}")
                # max_z = np.max([max_z,np.max(z)])
                # min_z = np.min([min_z,np.min(z)])                            
                
                sc = ax.scatter(ghi_ref[idx],ghi_inv[idx], s=8, c=z[idx], 
                                cmap="plasma") #,norm=norm)
                
                ax.plot([0, 1], [0, 1], ls = '--', transform=ax.transAxes,c='k')
                #ax.set_title(f"{station_label}, {year_label}",fontsize=14)
                
                print(f"Using {dict_stats[year][timeres][f'n_delta_{inv_type}_cosmo']} data points for {timeres}, {year} plot")
                ax.annotate(rf"MBE = {dict_stats[year][timeres][f'MBE_GHI_{inv_type}_cosmo_Wm2']:.2f} W m$^{{-2}}$" \
                            rf" ({dict_stats[year][timeres][f'rMBE_GHI_{inv_type}_cosmo_%']:.1f} %)" "\n" \
                            rf"RMSE = {dict_stats[year][timeres][f'RMSE_GHI_{inv_type}_cosmo_Wm2']:.2f} W m$^{{-2}}$" \
                                rf" ({dict_stats[year][timeres][f'rRMSE_GHI_{inv_type}_cosmo_%']:.1f} %)" "\n"\
                                    r"$\langle G_\mathrm{ref} \rangle$ ="\
                            rf" {dict_stats[year][timeres][f'mean_GHI_{inv_type}_cosmo_Wm2']:.2f} W m$^{{-2}}$" "\n"\
                            rf"n = ${dict_stats[year][timeres][f'n_delta_{inv_type}_cosmo']:.0f}$",
                          xy=(0.05,0.7),xycoords='axes fraction',fontsize=10,color='k',
                          bbox = dict(facecolor='lightgrey', alpha=0.3),
                          horizontalalignment='left',multialignment='left')     
                # ax.annotate(rf"RMSE = {dict_stats[year][timeres]['RMSE_GTI_Wm2']:.2f} W/m$^2$",
                #          xy=(0.05,0.85),xycoords='axes fraction',fontsize=10,bbox = dict(facecolor='lightgrey', alpha=0.3))  
                # ax.annotate(rf"n = {dict_stats[year][timeres]['n_delta']:.0f}",
                #          xy=(0.05,0.78),xycoords='axes fraction',fontsize=10,bbox = dict(facecolor='lightgrey', alpha=0.3))                  
                #ax.set_xticks([0,400,800,1200])
                    
            cb = fig.colorbar(sc,ticks=[np.min(z),np.max(z)], ax=axs[:2], shrink=0.6, location = 'top', 
                                aspect=20)    
            #print(f"{min_z},{max_z}")
            cb.set_ticklabels(["Low", "High"])  # horizontal colorbar
            cb.set_label("PDF", labelpad=-10, fontsize=16)
            
            #Set axis limits
            for i, ax in enumerate(axs.flatten()):
                ax.set_xlim([0.,max_ghi])
                ax.set_ylim([0.,max_ghi])
                ax.set_aspect('equal')
                ax.grid(False)
                # if max_gti < 1400:
                #     ax.set_xticks([0,400,800,1200])
                # else:
                #     ax.set_xticks([0,400,800,1200,1400])
                if i == 0:
                    ax.set_xlabel(r"$G_\mathrm{tot,COSMO}^{\downarrow}$ (W m$^{-2}$)",position=(1.1,0))
                    ax.set_ylabel(r"$G_\mathrm{{tot,inv}}^{{\downarrow}}$ (W m$^{-2}$)")
            
            #fig.subplots_adjust(wspace=0.1)    
            # fig.add_subplot(111, frameon=False)
            # plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            # plt.grid(False)
            
            plt.savefig(os.path.join(savepath,f"ghi_scatter_hist_combo_all_{inv_type}_COSMO_{timeres}_{T_model['model']}_"\
                      f"{T_model['type']}_{stations_label}.png"),bbox_inches = 'tight')  
            plt.close(fig)
            
def plot_all_ghi_combined_scatter_new(dict_stats,list_stations,pvrad_config,T_model,folder,title_flag,window_avgs):
    """
    Plot combined scatter plots for all stations

    Parameters
    ----------
    dict_stats : dictionary with combined statistics for different averaging times
    list_stations : list of PV stations
    pvrad_config : dictionary with PV inversion configuration
    T_model : string, temperature model used
    folder : string, folder to save plots
    title_flag : boolean, whether to add title to plots
    window_avgs : list of averaging windows

    Returns
    -------
    None.
    """
    
    plt.ioff()
    plt.style.use('my_presi_grid')        
    
    res_dirs = list_dirs(folder)
    savepath = os.path.join(folder,'GHI_GTI_Plots')
    if 'GHI_GTI_Plots' not in res_dirs:
        os.mkdir(savepath)    
        
    # res_dirs = list_dirs(savepath)
    # savepath = os.path.join(savepath,'Scatter')
    # if 'Scatter' not in res_dirs:
    #     os.mkdir(savepath)
        
    years = ["mk_" + campaign.split('_')[1] for campaign in pvrad_config["calibration_source"]]    
    stations_label = '_'.join(["".join(s.split('_')) for s in list_stations])
    
    tres_list = pvrad_config["timeres_comparison"] + window_avgs
    data_types = [("Pyr","pyr","pyranometer"),("sat","sat","CAMS")]        
    
                    
    #Combined plot with all three time resolutions
    for data_type, data_type_short, data_label in data_types:
        for timeres in tres_list:
            #For pyranometer - use all three types of inversion
            #For CAMS, use only the OD-based type 
            if timeres == "1min" and data_type == "Pyr":
                inv_types = ["aod","cod","lut"]   
                width = -0.4
                height = 0.15
                leg_posy =  0.63
            else:
                inv_types = ["aod","cod"]
                width = 0.15
                height = 0.08
                leg_posy = 0.75
            
            #1. Plot comparing inverted irradiance with that from pyranometers
            fig, axs = plt.subplots(len(inv_types),len(years),sharex='all',sharey='all')
            #cbar_ax = fig.add_axes([.76, .25, .015, .45])                    
                                                
            print(f"Plotting combined scatter plot for {timeres}, {inv_types}, {data_label}...please wait....")
            plot_data = []
            norm = []
            max_ghi = 0. #; min_z = 500.; max_z = 0.
                
            for i, ax in enumerate(axs.flatten()):            
                year = years[np.fmod(i,2)]
                inv_type = inv_types[int((i - np.fmod(i,2))/2)]
                                
                if timeres in window_avgs and "od" in inv_type:
                    if inv_type == "aod":
                        day_type = 0
                    elif inv_type == "cod":
                        day_type = 1
                    stacked_data = dict_stats[year][f"df_delta_all_{timeres}"].stack()\
                        .loc[:,["GHI_PV_od_inv",f"GHI_{data_type}_ref","day_type"]].stack()
                    ghi_data = stacked_data.loc[stacked_data["day_type"] == day_type].dropna()
                                        
                    ghi_ref = ghi_data[f"GHI_{data_type}_ref"].values.flatten()
                    ghi_inv = ghi_data["GHI_PV_od_inv"].values.flatten()
                else:                    
                    ghi_data = dict_stats[year][f"df_delta_all_{timeres}"].stack()\
                        .loc[:,[f"GHI_PV_{inv_type}_inv",f"GHI_{data_type}_ref"]].stack().dropna(how='any')
                    
                    ghi_ref = ghi_data[f"GHI_{data_type}_ref"].values.flatten()
                    ghi_inv = ghi_data[f"GHI_PV_{inv_type}_inv"].values.flatten()
                    
                xy = np.vstack([ghi_ref,ghi_inv])
                z = gaussian_kde(xy)(xy)
                idx = z.argsort()
                
                max_ghi = np.max([max_ghi,ghi_ref.max(),ghi_inv.max()])
                max_ghi = np.ceil(max_ghi/100)*100    
                
                
                sc = ax.scatter(ghi_ref[idx], ghi_inv[idx], s=5, c=z[idx], 
                                cmap="plasma") #,norm=norm[int((i - np.fmod(i,2))/2)])
                
                ax.plot([0, 1], [0, 1], ls = '--', transform=ax.transAxes,c='k')
                #ax.set_title(f"{year},{inv_type}",fontsize=14)
                                
                print(f"Using {dict_stats[year][timeres][f'n_delta_{inv_type}_{data_type_short}']} data points for {timeres}, {year} plot")
                ax.annotate(rf"MBE = {dict_stats[year][timeres][f'MBE_GHI_{inv_type}_{data_type_short}_Wm2']:.2f} W m$^{{-2}}$" \
                            rf" ({dict_stats[year][timeres][f'rMBE_GHI_{inv_type}_{data_type_short}_%']:.1f} %)" "\n" \
                            rf"RMSE = {dict_stats[year][timeres][f'RMSE_GHI_{inv_type}_{data_type_short}_Wm2']:.2f} W m$^{{-2}}$" \
                                rf" ({dict_stats[year][timeres][f'rRMSE_GHI_{inv_type}_{data_type_short}_%']:.1f} %)" "\n"\
                            r"$\langle G_\mathrm{ref} \rangle$ =" \
                                    rf" {dict_stats[year][timeres][f'mean_GHI_{inv_type}_{data_type_short}_Wm2']:.2f} W m$^{{-2}}$" "\n"\
                            rf"n = ${dict_stats[year][timeres][f'n_delta_{inv_type}_{data_type_short}']:.0f}$",
                          xy=(0.05,leg_posy),xycoords='axes fraction',fontsize=7,color='k',
                          bbox = dict(facecolor='lightgrey',alpha=0.3),
                          horizontalalignment='left',multialignment='left')     
                # ax.annotate(rf"RMSE = {dict_stats[year][timeres]['RMSE_GTI_Wm2']:.2f} W/m$^2$",
                #          xy=(0.05,0.85),xycoords='axes fraction',fontsize=10,bbox = dict(facecolor='lightgrey', alpha=0.3))  
                # ax.annotate(rf"n = {dict_stats[year][timeres]['n_delta']:.0f}",
                #          xy=(0.05,0.78),xycoords='axes fraction',fontsize=10,bbox = dict(facecolor='lightgrey', alpha=0.3))                  
                #ax.set_xticks([0,400,800,1200])
                    
            
            fig.subplots_adjust(wspace=width,hspace=height) 
            
            cb = fig.colorbar(sc,ticks=[np.min(z),np.max(z)], 
                              ax=axs[:3], shrink=0.6,
                                aspect=20) 
            #cb.set_ticks()
            #cb.set_ticklabels([f"{val:.2f}" for val in cb.get_ticks()*1e5])
            cb.set_ticklabels(["Low", "High"])  # horizontal colorbar
            cb.ax.tick_params(labelsize=14) 
            cb.set_label("PDF", labelpad=-18, 
                         fontsize=14)
            
            #Set axis limits
            for i, ax in enumerate(axs.flatten()):
                ax.set_xlim([0.,max_ghi])
                ax.set_ylim([0.,max_ghi])
                ax.set_aspect('equal')
                ax.grid(False)
                # if max_ghi > 1400:
                ax.set_xticks([0,500,1000])
                # else:
                #     ax.set_xticks([0,400,800,1200,1400])
                if len(inv_types) == 3:
                    if i == 4:
                        ax.set_xlabel(rf"$G_\mathrm{{tot,{data_label}}}^{{\downarrow}}$ (W m$^{{-2}}$)",position=(1.1,0))
                    if i == 2:
                        ax.set_ylabel(r"$G_\mathrm{{tot,PV,inv}}^{{\downarrow}}$ (W m$^{-2}$)")
                else:
                    if i == 2:
                        ax.set_xlabel(rf"$G_\mathrm{{tot,{data_label}}}^{{\downarrow}}$ (W m$^{{-2}}$)",position=(1.1,0))
                    if i == 2:
                        ax.set_ylabel(r"$G_\mathrm{{tot,PV,inv}}^{{\downarrow}}$ (W m$^{-2}$)",position=(0,1.1))
                    
                            
            # fig.add_subplot(111, frameon=False)
            # plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            # plt.grid(False)
            
            plt.savefig(os.path.join(savepath,f"ghi_scatter_hist_combo_all_{timeres}_{data_label}_{T_model['model']}_"\
                      f"{T_model['type']}_{stations_label}.png"),bbox_inches = 'tight')  
                
            plt.close(fig)            
            
            
    for timeres in window_avgs:
        inv_types = ["aod","cod"]
                    
        print(f"Plotting combined frequency scatter plot for {inv_types}, {timeres}, COSMO...please wait....")
        #3. Plot comparing inverted irradiance with that from cosmo
        fig, axs = plt.subplots(len(inv_types),len(years),sharex='all',sharey='all')
        #cbar_ax = fig.add_axes([.27, .75, .43,.015]) 
                
        # plot_data = []
        max_ghi = 0. #; min_z = 500.; max_z = 0.
        for i, ax in enumerate(axs.flatten()):            
            year = years[np.fmod(i,2)]
            inv_type = inv_types[int((i - np.fmod(i,2))/2)]
            
            if "od" in inv_type:
                if inv_type == "aod":
                    day_type = 0
                elif inv_type == "cod":
                    day_type = 1
                stacked_data = dict_stats[year][f"df_delta_all_{timeres}"].stack()\
                    .loc[:,["GHI_PV_od_inv","GHI_cosmo_ref","day_type"]].stack()
                ghi_data = stacked_data.loc[stacked_data["day_type"] == day_type].dropna()
                
                ghi_ref = ghi_data["GHI_cosmo_ref"].values.flatten()
                ghi_inv = ghi_data["GHI_PV_od_inv"].values.flatten()
            else:                    
                ghi_data = dict_stats[year][f"df_delta_all_{timeres}"].stack()\
                    .loc[:,[f"GHI_PV_{inv_type}_inv","GHI_cosmo_ref"]].stack().dropna(how='any')
                
                ghi_ref = ghi_data["GHI_cosmo_ref"].values.flatten()
                ghi_inv = ghi_data[f"GHI_PV_{inv_type}_inv"].values.flatten()
            xy = np.vstack([ghi_ref,ghi_inv])
            z = gaussian_kde(xy)(xy)
            idx = z.argsort()
            
            #plot_data.append((ghi_ref[idx], ghi_inv[idx], z[idx]))
            
            # sc = sns.kdeplot(x=ghi_ref,y=ghi_inv,cbar=i==0,ax=ax,cmap="icefire",fill=True,#bins=100,
            #  cbar_kws={'orientation': 'horizontal'},
            #  cbar_ax=None if i else cbar_ax)
                            
            # if i == 0:
            #     sc.figure.axes[-1].xaxis.set_ticks(sc.figure.axes[-1].get_xlim())                    
            #     sc.figure.axes[-1].tick_params(axis="x",direction="in", pad=-30)
            #     sc.figure.axes[-1].xaxis.set_ticklabels(["Low", "High"])
            #     sc.figure.axes[-1].set_xlabel("PDF",size=16,labelpad=-30)
                #sc.figure.axes[-1].set_location('top')
            
            max_ghi = np.max([max_ghi,ghi_ref.max(),ghi_inv.max()])
            max_ghi = np.ceil(max_ghi/100)*100    
        #     max_z = np.max([max_z,np.max(z)])
        #     min_z = np.min([min_z,np.min(z)])
            
        # norm = plt.Normalize(min_z,max_z)    
        
        # for i, ax in enumerate(axs.flatten()):
        #     year = years[np.fmod(i,2)]
        #     inv_type = inv_types[int((i - np.fmod(i,2))/2)]
            
            sc = ax.scatter(ghi_ref[idx], ghi_inv[idx],  s=8, c=z[idx],
                            cmap="plasma")#,norm=norm)
            
            ax.plot([0, 1], [0, 1], ls = '--', transform=ax.transAxes,c='k')
            #ax.set_title(f"{year},{inv_type}",fontsize=14)            
            
            print(f"Using {dict_stats[year][timeres][f'n_delta_{inv_type}_cosmo']} data points for {timeres}, {year} plot")
            ax.annotate(rf"MBE = {dict_stats[year][timeres][f'MBE_GHI_{inv_type}_cosmo_Wm2']:.2f} W m$^{{-2}}$" \
                        rf" ({dict_stats[year][timeres][f'rMBE_GHI_{inv_type}_cosmo_%']:.1f} %)" "\n" \
                        rf"RMSE = {dict_stats[year][timeres][f'RMSE_GHI_{inv_type}_cosmo_Wm2']:.2f} W m$^{{-2}}$" \
                            rf" ({dict_stats[year][timeres][f'rRMSE_GHI_{inv_type}_cosmo_%']:.1f} %)" "\n"\
                                r"$\langle G_\mathrm{ref} \rangle$ ="\
                            rf" {dict_stats[year][timeres][f'mean_GHI_{inv_type}_cosmo_Wm2']:.2f} W m$^{{-2}}$" "\n"\
                        rf"n = ${dict_stats[year][timeres][f'n_delta_{inv_type}_cosmo']:.0f}$",
                      xy=(0.05,0.75),xycoords='axes fraction',fontsize=7,color='k',
                      bbox = dict(facecolor='lightgrey', alpha=0.3),
                      horizontalalignment='left',multialignment='left')     
            # ax.annotate(rf"RMSE = {dict_stats[year][timeres]['RMSE_GTI_Wm2']:.2f} W/m$^2$",
            #          xy=(0.05,0.85),xycoords='axes fraction',fontsize=10,bbox = dict(facecolor='lightgrey', alpha=0.3))  
            # ax.annotate(rf"n = {dict_stats[year][timeres]['n_delta']:.0f}",
            #          xy=(0.05,0.78),xycoords='axes fraction',fontsize=10,bbox = dict(facecolor='lightgrey', alpha=0.3))                  
            #ax.set_xticks([0,400,800,1200])
                
        fig.subplots_adjust(wspace=0.15,hspace=0.08) 
            
        cb = fig.colorbar(sc,ticks=[np.min(z),np.max(z)], 
                              ax=axs[:3], shrink=0.6,
                                aspect=20)    
        cb.set_ticklabels(["Low", "High"])  # horizontal colorbar
        cb.set_label("PDF", labelpad=-18, 
                         fontsize=14)
        
        #Set axis limits
        for i, ax in enumerate(axs.flatten()):
            ax.set_xlim([0.,max_ghi])
            ax.set_ylim([0.,max_ghi])
            ax.set_aspect('equal')
            ax.grid(False)
            # if max_gti < 1400:
            #     ax.set_xticks([0,400,800,1200])
            # else:
            #     ax.set_xticks([0,400,800,1200,1400])
            if i == 2:
                ax.set_xlabel(r"$G_\mathrm{tot,COSMO}^{\downarrow}$ (W m$^{-2}$)",position=(1.1,0))                    
                ax.set_ylabel(r"$G_\mathrm{{tot,PV,inv}}^{{\downarrow}}$ (W m$^{-2}$)",position=(0,1.1))
        
        #fig.subplots_adjust(wspace=0.1)    
        # fig.add_subplot(111, frameon=False)
        # plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        # plt.grid(False)
        
        plt.savefig(os.path.join(savepath,f"ghi_scatter_hist_combo_all_COSMO_{timeres}_{T_model['model']}_"\
                  f"{T_model['type']}_{stations_label}.png"),bbox_inches = 'tight')  
        plt.close(fig)

def generate_folders(rt_config,pvcal_config,pvrad_config,home):
    """
    Generate folders for results
    
    args:    
    :param rt_config: dictionary with configuration of RT simulation
    :param pvcal_config: dictionary with configuration for calibration
    :param pvrad_config: dictionary with configuration for inversion
    :param home: string, home directory
    
    out:
    :return fullpath: string with label for saving folders   
    """    

    path = os.path.join(pvrad_config["results_path"]["main"],
                        pvrad_config["results_path"]["irradiance"])
    
    #atmosphere model
    atm_geom_config = rt_config["disort_base"]["pseudospherical"]
    
    if atm_geom_config == True:
        atm_geom_folder = "Pseudospherical"
    else:
        atm_geom_folder = "Plane-parallel"
        
    dirs_exist = list_dirs(os.path.join(home,path))
    fullpath = os.path.join(home,path,atm_geom_folder)
    if atm_geom_folder not in dirs_exist:
        os.mkdir(fullpath)        
    
    disort_config = rt_config["clouds"]["disort_rad_res"]
    theta_res = str(disort_config["theta"]).replace('.','-')
    phi_res = str(disort_config["phi"]).replace('.','-')
    
    disort_folder_label = 'Zen_' + theta_res + '_Azi_' + phi_res    
    
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
        
    sza_label = "SZA_" + str(int(pvrad_config["sza_max"]["inversion"]))
    
    dirs_exist = list_dirs(fullpath)
    fullpath = os.path.join(fullpath,sza_label)    
    if sza_label not in dirs_exist:
        os.mkdir(fullpath)
            
    model = pvcal_config["inversion"]["power_model"]
    dirs_exist = list_dirs(fullpath)
    fullpath = os.path.join(fullpath,model)  
    if model not in dirs_exist:
        os.mkdir(fullpath)
    
    eff_model = pvcal_config["eff_model"]
    dirs_exist = list_dirs(fullpath)
    fullpath = os.path.join(fullpath,eff_model)  
    if eff_model not in dirs_exist:
        os.mkdir(fullpath)
        
    T_model = pvcal_config["T_model"]["model"]
    dirs_exist = list_dirs(fullpath)
    fullpath = os.path.join(fullpath,T_model)  
    if T_model not in dirs_exist:
        os.mkdir(fullpath)        
        
    return fullpath

def save_results(name,pv_station,info,pyr_substats,pv_substats,
                 rt_config,pyr_config,pvcal_config,
                 pvrad_config,config,savepath):
    """
    

    Parameters
    ----------
    name : string, name of PV station
    pv_station : dictionary with information and data from PV station
    info : string, description of current campaign
    pyr_substats : dictionary with pyranometer substations
    pv_substats : dictionary with pv substations
    rt_config : dictionary with radiative transfer configuration 
    pyr_config : dictionary with pyranometer configuration
    pvcal_config : dictionary with PV calibration configuration
    pvrad_config : dictionary with PV inversion configuration
    config : dictionary with overall configuration
    savepath : string with path for saving results

    Returns
    -------
    None.

    """
                 
    pv_station_save = deepcopy(pv_station)
    
    year = info.split('_')[1]
    for timeres in pyr_config["t_res_inversion"]:
        if f"df_codfit_pyr_pv_{year}_{timeres}" in pv_station_save:
            del pv_station_save[f"df_codfit_pyr_pv_{year}_{timeres}"]
    
    filename_stat = f"ghi_analysis_results_{info}_{name}.data"            
    
    with open(os.path.join(savepath,filename_stat), 'wb') as filehandle:  
        # store the data as binary data stream
        pickle.dump((pv_station_save, rt_config, pyr_config, pvcal_config, pvrad_config), filehandle)

    print('Results written to file %s\n' % filename_stat)
    
    #Write GHI data to CSV    
    csvpath = os.path.join(savepath,'CSV_Results')
    if 'CSV_Results' not in list_dirs(savepath):
        os.mkdir(csvpath) 
    
    year = info.split('_')[1]  
    cf_avg_res = pvrad_config["cloud_fraction"]["cf_avg_window"]
    
    substat_cols = list(pyr_substats.keys())
    for timeres in pv_station["timeres"]:        
        if f"df_stats_{timeres}_{year}" in pv_station:            
                        
            dataframe = pv_station[f"df_stats_{timeres}_{year}"].dropna(how='all',axis=0)
    
            #Write all results to CSV file
            filename_csv = f'ghi_results_{name}_{timeres}_{year}.dat'
            f = open(os.path.join(csvpath,filename_csv), 'w')
            f.write('#Station: %s, Global horizontal irradiance inverted from PV data\n' % name)    
            f.write('#GHI from DISORT LUT via AOD (GHI_PV_aod_inv), COD (GHI_PV_cod_inv) and MYSTIC LUT (GHI_PV_lut_inv)\n')
            f.write('#Comparison data from pyranometer (GHI_Pyr_ref) and CAMS (GHI_sat_ref)\n')
            f.write('#See the corresponding files "tilted_irradiance_cloud_fraction..." for calibration parameters\n')
            #f.write('#Tilted irradiance and cloud fraction inferred from %s\n' % substat_type)
            #f.write('#Results up to maximum SZA of %d degrees\n' % sza_limit)
            # if inv_model == "power":
            #     f.write('#PV model: %s, efficiency model: %s, temperature model: %s\n' % (model,eff_model,T_model))        
            # elif inv_model == "current":
            #     f.write('#PV model: %s, temperature model: %s\n' % (model,T_model))        
            # for substat in pv_station["substations"][substat_type]["data"]:                
            #     ap_pars = pv_station["substations"][substat_type]["data"][substat]["ap_pars"]
            #     f.write('#A-priori parameters (value,error):\n')
            #     for par in ap_pars:
            #         f.write('#%s: %g (%g)\n' % par)                        
            #     if "opt_pars" in pv_station["substations"][substat_type]["data"][substat]:
            #         opt_pars = pv_station["substations"][substat_type]["data"][substat]["opt_pars"]
            #         f.write('#Optimisation parameters (value,error):\n')
            #         for par in opt_pars:
            #             f.write('#%s: %g (%g)\n' % par)     
            #     else:
            #         f.write('No solution found by the optimisation routine, using a-priori values\n')
    
            f.write('\n#Multi-index: first line ("variable") refers to measured quantity\n')
            f.write('#second line ("substat") refers to sensor used for inversion onto GHI\n')
            f.write('#third line ("station") refers to PV station\n')

            f.write('#First column is the time stamp, in the format %Y-%m-%d %HH:%MM:%SS\n')
            f.write('\n')                       
            
            dataframe.to_csv(f,sep=';',float_format='%.6f',
                              na_rep='nan')
            
            f.close()    
            print('Results written to file %s\n' % filename_csv)     
            
def save_combo_stats(dict_stats,list_stations,pvrad_config,T_model,folder,window_avgs):
    """
    

    Parameters
    ----------
    dict_stats : dictioanry with combined statistics for different averaging times
    list_stations : list of PV stations
    pvrad_config : dictionary with PV inversion configuration
    T_model : string, temperature model used
    folder : string, folder for saving resul
    window_avgs : list of averaging windows

    Returns
    -------
    None.

    """
        
    stations_label = '_'.join(["".join(s.split('_')) for s in list_stations])
    
    filename = f"ghi_combo_results_stats_{T_model['model']}_{stations_label}.data"
    
    with open(os.path.join(folder,filename), 'wb') as filehandle:  
        # store the data as binary data stream
        pickle.dump((dict_stats, list_stations, pvrad_config, T_model, window_avgs), filehandle)
    
    csvpath = os.path.join(folder,'CSV_Results')
    if 'CSV_Results' not in list_dirs(folder):
        os.mkdir(csvpath)    
    
    #Write combined results to CSV
    for measurement in pvrad_config["inversion_source"]:
        year = f"mk_{measurement.split('_')[1]}"
        
        for window_avg in window_avgs:            
            if f"df_delta_all_{window_avg}" in dict_stats[year]:
                
                dataframe = dict_stats[year][f"df_delta_all_{window_avg}"]
                                                                                                
                stacked_data = dataframe.stack()\
                    .loc[:,["GHI_PV_od_inv","delta_GHI_aod_cosmo_Wm2","day_type"]].stack()
                ghi_data = stacked_data.loc[stacked_data["day_type"] == 0].dropna()                
                ghi_data.rename(columns={'GHI_PV_od_inv':'GHI_PV_aod_inv'}, inplace=True)
                ghi_data.drop(columns="day_type",inplace=True)
                GHI_PV_aod_inv = ghi_data.unstack().unstack()
                
                stacked_data = dataframe.stack()\
                    .loc[:,["GHI_PV_od_inv","delta_GHI_cod_cosmo_Wm2","day_type"]].stack()                
                ghi_data = stacked_data.loc[stacked_data["day_type"] == 1].dropna()                    
                ghi_data.rename(columns={'GHI_PV_od_inv':'GHI_PV_cod_inv'}, inplace=True)
                ghi_data.drop(columns="day_type",inplace=True)
                GHI_PV_cod_inv = ghi_data.unstack().unstack()
                    
                df_write = pd.merge(pd.concat([GHI_PV_aod_inv,GHI_PV_cod_inv],axis=1),
                                    dataframe.loc[:,pd.IndexSlice["GHI_cosmo_ref",:,:]],
                                    how="left",left_index=True,right_index=True)
                df_write.sort_index(axis=1,level=2,inplace=True)
                df_write.dropna(axis=0,how='all',inplace=True)
                
                filename_csv = f'ghi_combo_results_{window_avg}_{year}_{T_model["model"]}.dat'
                f = open(os.path.join(csvpath,filename_csv), 'w')
                f.write(f'#Global horizontal irradiance inverted from PV data combined and averaged to {window_avg}\n')    
                f.write('#GHI from DISORT LUT via AOD (GHI_PV_aod_inv) and COD (GHI_PV_cod_inv)\n')
                f.write('#Comparison data from COSMO (GHI_cosmo_ref)\n')                    
                f.write(f'#Stations considered: {list_stations}\n')                    
                
                f.write('\n#Multi-index: first line ("variable") refers to measured quantity\n')
                f.write('#second line ("substat") refers to sensor used for inversion of GHI\n')
                f.write('#third line ("station") refers to PV station\n')
                f.write('#First column is the time stamp, in the format %Y-%m-%d %HH:%MM:%SS\n')
                f.write('\n')                       
                
                df_write.to_csv(f,sep=';',float_format='%.6f',
                                  na_rep='nan')
                
                f.close()    
                print('Combined results written to file %s\n' % filename_csv)    

#%%Main Program
#######################################################################
###                         MAIN PROGRAM                            ###
#######################################################################
#This program takes makes plots of GHI, DNI and GHI for comparison with measurements
#All data are averaged to 60 minutes, in order for comparison with COSMO
#In addition, different moving averages (1min, 15min) are calculated, dependent on the measurements available
#def main():
import argparse
    
parser = argparse.ArgumentParser()
parser.add_argument("-f","--configfile", help="yaml file containing config")
parser.add_argument("-s","--station",nargs='+',help="station for which to perform inversion")
parser.add_argument("-c","--campaign",nargs='+',help="measurement campaign for which to perform simulation")
args = parser.parse_args()
   
if args.configfile:
    config_filename = os.path.abspath(args.configfile) #"config_PYRCAL_2018_messkampagne.yaml" #
else:
    config_filename = "config_PVPYRODGHI_MetPVNet_messkampagne.yaml"

config = load_yaml_configfile(config_filename)

#Load radiative transfer configuration
rt_config = load_yaml_configfile(config["rt_configfile"])

#Load PVCAL configuration
pvcal_config = load_yaml_configfile(config["pvcal_configfile"])

#Load pv2rad configuration
pvrad_config = load_yaml_configfile(config["pvrad_configfile"])

homepath = os.path.expanduser('~') # #"/media/luke" #
    
if args.campaign:
    campaigns = args.campaign
    if type(campaigns) != list:
        campaigns = [campaigns]
else:
    campaigns = config["description"]  
    
plot_styles = config["plot_styles"]
plot_flags = config["plot_flags"]
sza_limit_cod = rt_config["sza_max"]["cod_cutoff"]
sza_limit_aod = rt_config["sza_max"]["lut"]
window_avgs = config["window_size_moving_average"]
T_model = pvcal_config["T_model"]
                               
#%% Run through campaigns, load OD results and do a DISORT simuation
od_types = ["aod","cod"]
str_window_avg = config["window_size_moving_average"]
num_var_classes = 3

combo_stats = {}
print('Loading irradiance results to perform plots and analysis')
for campaign in campaigns:   
    year = "mk_" + campaign.split('_')[1]
    yrname = year.split('_')[-1]
    #Load PV configuration
    pyr_config = load_yaml_configfile(config["pyrcalod_configfile"][year])
    data_config = load_yaml_configfile(config["data_configfile"][year])
    window_avg_cf = pyr_config["cloud_fraction"]["cf_avg_window"]
    
    if args.station:
        stations = args.station
        if stations[0] == 'all':
            stations = 'all'
    else:
        #Stations for which to perform inversion
        stations = ["PV_11","PV_12"] #["PV_12","PV_15"] #pyr_config["stations"]

    #Load both OD inversion results and LUT results
    print(f'Loading PYR2OD and PV2OD and LUT results for {campaign}')
    pvsys, station_list = \
    load_pvpyr2od_fit_ghilut_results(rt_config, pyr_config, pvcal_config, pvrad_config, 
                              campaign, stations, od_types, homepath)

    pyr_results_path = os.path.join(homepath,pyr_config["results_path"]["main"],
                                    pyr_config["results_path"]["irradiance"])
    
    savepath = generate_results_folders(rt_config,pyr_config,pvcal_config,
                            pyr_results_path)

    #plotpaths = pyr_config["results_path"]["plots"]
    
    dfs_stats_all = []
    var_class_dict = {}
    
    combo_stats.update({year:{}})
    combo_stats[year].update({f"df_{yrname}_stats":pd.DataFrame(index=station_list)})
    dfs_deviations = {}
    for tres in pvrad_config["timeres_comparison"]:
        dfs_deviations.update({tres:[]})    
    for tres in window_avgs:
        dfs_deviations.update({tres:[]})    
        
    
    #Calculate moving averages in order to compare retrieval with CAMS/COSMO
    for key in pvsys:
        print(f'Importing data from CAMS for {key}, {year}')
        pvsys[key] = import_cams_data(year,pvsys[key],pvrad_config,homepath)
        
        print(f'Importing COSMO data for {key}, {year}')
        pvsys[key] = prepare_cosmo_data(year,pvcal_config,key,pvsys[key],homepath)        
        
        combine_cams_hires(year.split('_')[1], pvsys[key])
        
        # print(f'Calculating statistics for LUT results')
        # if key in pyr_config["pv_stations"]:
        #     pyr_substat_pars = pvsys[key]["substations_pyr"]  
        #     calc_hires_lut_statistics(year,key,pvsys[key],pyr_substat_pars,
        #                            savepath,plot_flags,plot_styles)                
        
        #plot_scatter
        
        for str_window_avg in window_avgs:
            print(f'Using {str_window_avg} moving average for comparison')
            print("Combine COSMO and CAMS data")        
            pvsys[key][f"df_compare_ghi_{year.split('_')[1]}_{str_window_avg}"] = combine_cosmo_cams(year.split('_')[1],
                                                        pvsys[key], str_window_avg)        

            #Combine raw data and calculate moving averages        
            if key in pyr_config["pv_stations"]:
                pvsys[key][f"df_compare_ghi_{year.split('_')[1]}_{str_window_avg}"] = pd.concat([
                    pvsys[key][f"df_compare_ghi_{year.split('_')[1]}_{str_window_avg}"],combine_raw_data(year,pvsys[key]["raw_data"],
                      str_window_avg,pyr_config["pv_stations"][key])],axis=1)
        
                measdata_names = pyr_config["pv_stations"][key]["irrad_names"]
            else: measdata_names = {} 
                
            #Go through Pyranometers, calculate moving average, plot      
            if key in pyr_config["pv_stations"]:
                #Get substation parameters
                pyr_substat_pars = pvsys[key]["substations_pyr"]                        
                #Perform analysis and plot
                pvsys[key][f"df_compare_ghi_{year.split('_')[1]}_{str_window_avg}"] = irradiance_analysis_plots(key,
                        pvsys[key],pyr_substat_pars,od_types,year.split('_')[1],sza_limit_aod,sza_limit_cod,
                        str_window_avg,window_avg_cf,measdata_names,"",savepath,plot_flags)                                                
                
                for substat in pyr_substat_pars:
                    timeres = pyr_substat_pars[substat]["t_res_inv"]
                    #radname = pyr_substat_pars[substat]["name"]                    
                    for od in od_types:
                        if f"df_{od}fit_pyr_pv_{year.split('_')[-1]}_{rt_config['timeres']}" in pvsys[key]:
                            pvsys[key][f"df_{od}fit_pyr_pv_{year.split('_')[-1]}_{rt_config['timeres']}"] = \
                            pd.concat([pvsys[key][f"df_{od}fit_pyr_pv_{year.split('_')[-1]}_{rt_config['timeres']}"],
                            downsample_pyranometer_data(pvsys[key][f"df_{od}fit_pyr_pv_{year.split('_')[-1]}_{timeres}"],
                            substat,timeres,rt_config["timeres"])],axis=1)
            
            else:
                pyr_substat_pars = {}  
                                
            #Go through PVs, calculate moving average, plot
            if key in pvrad_config["pv_stations"]:
                pv_substat_pars = pvsys[key]["substations_pv"]
                for substat_type in pv_substat_pars:                
                    pv_substats = pv_substat_pars[substat_type]["data"]
                    if year in pv_substat_pars[substat_type]["source"]:
                        pyr_down_name = pvrad_config["pv_stations"][key][substat_type]["pyr_down"][year]                    
                        if key == pyr_down_name[0]:
                            pyr_down_name = pyr_down_name[1]
                        else:
                            pyr_down_name = ""
                        pvsys[key][f"df_compare_ghi_{year.split('_')[1]}_{str_window_avg}"] = irradiance_analysis_plots(key,
                            pvsys[key],pv_substats,od_types,year.split('_')[1],sza_limit_aod,sza_limit_cod,
                            str_window_avg,window_avg_cf,measdata_names,pyr_down_name,savepath,plot_flags)
            else:
                pv_substat_pars = {}                            
            
            # #Scatter plots            
            if plot_flags["scatter"]:
                print(f"Creating scatter plots with data averaged over {str_window_avg}")
                scatter_plot_irradiance_comparison_grid(key,pvsys[key][f"df_compare_ghi_{year.split('_')[1]}_{str_window_avg}"], 
                              rt_config,pyr_substat_pars,pv_substat_pars,year,plot_styles,
                              savepath, str_window_avg, plot_flags, day_type='all')
                
                # for day_type in pyr_config["test_days"]:                    
                #     df_compare = pd.concat([pvsys[key][f"df_compare_ghi_{year.split('_')[1]}_{str_window_avg}"].loc[day.strftime('%Y-%m-%d')]
                #                             for day in pyr_config["test_days"][day_type]],axis=0)
                    
                #     print(f"Creating scatter plots with data averaged over {str_window_avg} and {day_type} days")
                #     scatter_plot_irradiance_comparison_grid(key,df_compare, 
                #               rt_config,pyr_substat_pars,pv_substat_pars,year,plot_styles,
                #               savepath, str_window_avg, plot_flags,day_type)
            
            #Calculate variance classes
            if key in pvrad_config["pv_stations"]:
                pv_substats = pvsys[key]["substations_pv"]
            else:
                pv_substats = {}
            
            #Stats: calculation and plots per station
            pvsys[key][f"df_stats_ghi_{year.split('_')[1]}_{str_window_avg}"],\
            pvsys[key][f"df_delta_{str_window_avg}_{yrname}"] = \
                    irradiance_stats_plots(key,
                    pvsys[key][f"df_compare_ghi_{year.split('_')[1]}_{str_window_avg}"],
                    pyr_substat_pars,pv_substats,
                    year,str_window_avg,num_var_classes,plot_styles,
                    plot_flags,savepath)                        
            
            #Join all dataframes into one for all stations
            df_stats = pvsys[key][f"df_stats_ghi_{year.split('_')[1]}_{str_window_avg}"]
            idx = df_stats.columns.to_frame()
            idx.insert(2, 'station', key)
            df_stats.columns = pd.MultiIndex.from_frame(idx) 
            
            dfs_stats_all.append(df_stats)            
        
        #Calculate statistics
        if combo_stats[year][f"df_{yrname}_stats"].empty:
            combo_stats[year][f"df_{yrname}_stats"] = calc_statistics_irradiance(
                key,pvsys[key],year,pvrad_config,pyr_config,savepath)                            
        else:
            combo_stats[year][f"df_{yrname}_stats"] = pd.concat([combo_stats[year][f"df_{yrname}_stats"],
                 calc_statistics_irradiance(key,pvsys[key],year,pvrad_config,
                                            pyr_config,savepath)],axis=0)
            
        for timeres in pvrad_config["timeres_comparison"]:
            if f"df_stats_{timeres}_{yrname}" in pvsys[key]:
                dfs_deviations[timeres].append(pvsys[key][f"df_stats_{timeres}_{yrname}"])
            else:
                dfs_deviations[timeres].append(pd.DataFrame())
                
        for str_window_avg in window_avgs:
            dfs_deviations[str_window_avg].append(pvsys[key][f"df_delta_{str_window_avg}_{yrname}"])                            
        
        results_path = generate_folders(rt_config,pvcal_config,pvrad_config,homepath)
        save_results(key, pvsys[key], campaign, pyr_substat_pars, pv_substats, rt_config, pyr_config, 
                     pvcal_config, pvrad_config, config, results_path)
    
    for timeres in pvrad_config["timeres_comparison"] + window_avgs:
        if not combo_stats[year][f"df_{yrname}_stats"].empty:
            combo_stats[year][f"df_delta_all_{timeres}"] = pd.concat(dfs_deviations[timeres],axis=1)
        else:
            combo_stats[year][f"df_delta_all_{timeres}"] = pd.DataFrame()
        if combo_stats[year][f"df_delta_all_{timeres}"].empty:
            del combo_stats[year][f"df_delta_all_{timeres}"]                
        
    if stations == "all":
        df_stats_all = pd.concat(dfs_stats_all,axis=1)
        df_stats_all.sort_index(axis=1,level=[1,2],inplace=True)        
                        
        #Stats for all stations
        # df_mean_stats, var_classes_mean = combined_stats_plots(df_stats_all,
        #             pyr_substat_pars,pv_substats,year,str_window_avg,
        #           num_var_classes,plot_styles,plot_flags,savepath)
            
    combined_stats(combo_stats[year],yrname,pvrad_config["timeres_comparison"],window_avgs)
        
save_combo_stats(combo_stats,station_list,pvrad_config,T_model,
                                  results_path,window_avgs)

if plot_flags["combo_stats"]:
    plot_all_ghi_combined_scatter(combo_stats,station_list,pvrad_config,T_model,
                                  savepath,plot_flags["titles"],window_avgs)
    plot_all_ghi_combined_scatter_new(combo_stats,station_list,pvrad_config,T_model,
                                  savepath,plot_flags["titles"],window_avgs)
        
