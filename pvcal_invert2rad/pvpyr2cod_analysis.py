#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 10:33:16 2021

@author: james
"""

#%% Preamble
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import ScalarFormatter
#from matplotlib.gridspec import GridSpec
from pvcal_forward_model import azi_shift
import pandas as pd
from file_handling_functions import *
from analysis_functions import v_index, overshoot_index
from plotting_functions import confidence_band
import subprocess
import pickle
# from astropy.convolution import convolve, Box1DKernel
from copy import deepcopy
from scipy.stats import gaussian_kde

#%%Functions

def generate_folder_names_pvpyr2cod(rt_config,pvcal_config):
    """
    Generate folder structure to retrieve PV2COD simulation results
    
    args:    
    :param rt_config: dictionary with RT configuration
    :param pvcal_config: dictionary with PYRCAL configuration
    
    out:
    :return folder_label: string with complete folder path
    :return filename: string with name of file (prefix)
    :return theta_res, phi_res: tuple of string with DISORT grid resolution
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
    disort_config = rt_config["clouds"]["disort_rad_res"]   
    theta_res = str(disort_config["theta"]).replace('.','-')
    phi_res = str(disort_config["phi"]).replace('.','-')
    
    disort_folder_label = 'Zen_' + theta_res + '_Azi_' + phi_res
    filename = 'cod_fit_results_'
    
    if rt_config["atmosphere"] == "default":
        atm_folder_label = "Atmos_Default"    
    elif rt_config["atmosphere"] == "cosmo":
        atm_folder_label = "Atmos_COSMO"
        #filename = filename + 'atm_'
        
    if rt_config["aerosol"]["source"] == "default":
        aero_folder_label = "Aerosol_Default"
    elif rt_config["aerosol"]["source"] == "aeronet":
        aero_folder_label = "Aeronet_" + rt_config["aerosol"]["station"]
        #filename = filename + 'asl_' + rt_config["aerosol"]["data_res"] + '_'
            
    sza_label = "SZA_" + str(int(rt_config["sza_max"]["lut"]))         
    
    model = pvcal_config["inversion"]["power_model"]
    eff_model = pvcal_config["eff_model"]
    
    T_model = pvcal_config["T_model"]["model"]

    folder_label = os.path.join(atm_geom_folder,wvl_folder_label,disort_folder_label,
                                atm_folder_label,aero_folder_label,sza_label,
                                model,eff_model,T_model)
        
    return folder_label, filename, (theta_res,phi_res)


def load_pvpyr2cod_results(rt_config,pyr_config,pvcal_config,pvrad_config,
                           info,station_list,home):
    """
    Load results from cloud optical depth simulations with DISORT
    
    args:        
    :param rt_config: dictionary with current RT configuration
    :param pyr_config: dictionary with current pyranometer calibration configuration    
    :param pvcal_config: dictionary with current calibration configuration    
    :param pvrad_config: dictionary with current inversion configuration    
    :param info: string with description of current campaign
    :param station_list: list of stations
    :param home: string with homepath
    
    out:
    :return pv_systems: dictionary of PV systems with data    
    :return pv_folder_label: string with path for PV results
    :return station_list: list of stations
       
    """
    
    mainpath = os.path.join(home,pyr_config['results_path']['main'],
                                   pyr_config['results_path']['optical_depth'])     
    
    folder_label, filename, (theta_res,phi_res) = \
    generate_folder_names_pvpyr2cod(rt_config,pvcal_config)

    pyr_folder_label = os.path.join(mainpath,folder_label)    
    
    filename = filename + info + '_'#+ '_disortres_' + theta_res + '_' + phi_res + '_'
    
    #Choose which stations to load    
    if type(station_list) != list:
        station_list = [station_list]
        if station_list[0] == "all":
            station_list = list(pyr_config["pv_stations"].keys())
            station_list.extend(list(pvrad_config["pv_stations"].keys()))
            station_list = list(set(station_list))
            station_list.sort()    
    
    year = info.split('_')[1]
    
    pv_systems = {}
    
    for station in station_list:     
        if station in pyr_config["pv_stations"]:
            timeres = "1min"
            #Read in binary file that was saved from pvcal_radsim_disort
            filename_stat = filename + station + f'_{timeres}.data'
            try:
                with open(os.path.join(pyr_folder_label,filename_stat), 'rb') as filehandle:  
                    # read the data as binary data stream
                    pvstat = pd.read_pickle(filehandle)            
                
                if station not in pv_systems:
                    pv_systems.update({station:pvstat})
                else:
                    if timeres not in pv_systems[station]["timeres"]:
                        pvstat["timeres"].extend(pv_systems[station]["timeres"])
                        
                    pv_systems[station] = merge_two_dicts(pv_system[station], pvstat)
                
                print('Loaded pyranometer COD retrieval for %s at %s in %s' % (station,timeres,year))
            except IOError:
                print('There is no COD retrieval for pyranometers at %s, %s in %s' % (station,timeres,year))                   
        if station in pvrad_config["pv_stations"]:
            for substat_type in pvrad_config["pv_stations"][station]:
                timeres = pvrad_config["pv_stations"][station][substat_type]["t_res_inv"]
                
                #Read in binary file that was saved from pvcal_radsim_disort
                filename_stat = filename + station + f'_{timeres}.data'
                try:
                    with open(os.path.join(pyr_folder_label,filename_stat), 'rb') as filehandle:  
                        # read the data as binary data stream
                        pvstat = pd.read_pickle(filehandle)  
                        
                    if station not in pv_systems:
                        pv_systems.update({station:pvstat})
                    else:
                        if timeres not in pv_systems[station]["timeres"]:
                            pvstat["timeres"].extend(pv_systems[station]["timeres"])
                        
                        pv_systems[station] = merge_two_dicts(pv_systems[station], pvstat)
                
                    print('Loaded PV COD retrieval for %s at %s in %s' % (station,timeres,year))
                except IOError:
                    print('There is no COD retrieval for PV systems at %s, %s in %s' % (station,timeres,year))                                                 
            
    results_path = os.path.join(home,pvrad_config["results_path"]["main"],
                                pvrad_config["results_path"]["optical_depth"])
    pv_folder_label = os.path.join(results_path,folder_label)
            
    return pv_systems, pv_folder_label, station_list

def find_nearest_grid_folder(configfile,key,pv_station,datatype,home):
    """
    Search through the output of cosmopvcod or msgseviripvcod to find the 
    gridpoint (folder) corresponding to the location of each PV station
    
    args:
    :param configfile: string, configfile
    :param key: string, name of station
    :param pv_station: dictionary with all info and data of pv station
    :param datatype: string, either surf or atmo
    :param home: string, home path
    
    out:
    :return pv_systems: dictionary updated with path to cosmo atmfiles
    :return pathname: string with name of path for station dictionary
    """
    
    config = load_yaml_configfile(configfile)
    
    if datatype == "atmo":
        path = os.path.join(home,config["path_atmofiles"])
    elif datatype == "surf":
        path = os.path.join(home,config["path_surface_files"])
    elif datatype == "irrad":
        path = os.path.join(home,config["path_irradiance_files"])
    elif datatype == "cloud" or datatype == "seviri":
        path = os.path.join(home,config["path_cod_files"])    
    
    cosmo_folders = list_dirs(path)
    
    #Define paths for COSMO-modified atmosphere files, surface files, irradiance files or cloud prop files    
    for folder in cosmo_folders:
        fname = "known_stations.dat"
        ds = pd.read_csv(os.path.join(path,folder,fname),comment='#',names=['name','lat','lon'],sep=' ',
                         index_col=0)
        for station in ds.index:
            if station == key:                
                if datatype == "atmo":
                    pathname = 'path_cosmo_lrt'
                    source = 'COSMO'
                elif datatype == "surf":
                    pathname = 'path_cosmo_surface'                    
                    source = 'COSMO'
                elif datatype == "irrad":
                    pathname = 'path_cosmo_irrad'
                    source = 'COSMO'
                elif datatype == "cloud":
                    pathname = 'path_cosmo_cloudprops'
                    source = 'COSMO'
                elif datatype == "seviri":
                    pathname = "path_seviri_cloudprops"
                    source = 'SEVIRI'
                    
                print(f"Nearest {source} grid point to {key} is at {folder[4:9]}, {folder[14:19]}")
                
                pv_station[pathname] = os.path.join(path,folder)

    return pv_station, pathname

def import_cosmo_cloud_data(key,pv_station,year,pathname):
    """
    Import cloud data from cosmo2pvcod
    
    args:
    :param key: string, name of station
    :param pv_station: dictionary with all info and data of pv station    
    :param year: string describing the data source
    :param pathname: string with pathname for files
    
    out:
    :return pv_station: dictionary with all info and data, including cloud data from COSMO
    """
            
    #Extract data from COSMO files
    dataframe = pd.DataFrame()
    dfs = [pd.read_csv(os.path.join(pv_station[pathname],filename),sep='\s+',
           index_col=0,skiprows=2,header=None,names=['COD_w_600','COD_i_600','COD_tot_600','cf_tot',
                 'COD_tot_600_avg','COD_tot_600_iqr','COD_tot_wi_600_avg','COD_tot_wi_600_iqr','cf_tot_avg'])
           for filename in list_files(pv_station[pathname]) if '_cod.dat' 
           in filename and year.split('_')[1] in filename]
    
    dataframe = pd.concat(dfs,axis=0)
    dataframe.index = pd.to_datetime(dataframe.index,format='%d.%m.%Y;%H:%M:%S')        

    #get total cloud optical depth    
    # dataframe["COD_tot_600"] = dataframe["COD_w_600"] + dataframe["COD_i_600"]
    # dataframe["COD_tot_600_avg"] = dataframe["COD_w_600_avg"] + dataframe["COD_i_600_avg"]
    # dataframe["COD_tot_600_iqr"] = dataframe["COD_w_600_iqr"] + dataframe["COD_i_600_iqr"]
    
    #Create Multi-Index for cosmo data
    dataframe.columns = pd.MultiIndex.from_product([dataframe.columns.values.tolist(),['cosmo']],
                                                                   names=['variable','substat'])       
                   
    #This is temporary due to an error in the cosmo2pvcod script with division by zero
    dataframe.fillna(0., inplace=True)
    
    #Assign to special cosmo dataframe, and join with main dataframe
    pv_station['df_cosmo_cod_' + year.split('_')[-1]] = dataframe
    
#    pv_station[df_name] = pd.concat([pv_station[df_name],dataframe],axis=1,join='inner')
       
    return pv_station

def import_seviri_cloud_data(key,pv_station,year,pathname):
    """
    Import cloud data from seviri2pvcod
    
    args:
    :param key: string, name of station
    :param pv_station: dictionary with all info and data of pv station    
    :param year: string describing the data source
    :param pathname: string with pathname for files
    
    out:
    :return pv_station: dictionary with all info and data, including cloud data from SEVIRI
    """

    #Read in satellite COD values
    dataframe = pd.DataFrame()
    dfs = [pd.read_csv(os.path.join(pv_station[pathname],filename),sep='\s+',
           index_col=0,skiprows=2,header=None,
           names=['COD_500','d_COD_500','mean_COD_500','iqr_COD_500']) 
           for filename in list_files(pv_station[pathname]) if '_seviri_cod.dat' 
           in filename and year.split('_')[1] in filename]
    
    dataframe = pd.concat(dfs,axis=0)
    dataframe.index = pd.to_datetime(dataframe.index,format='%d.%m.%Y;%H:%M:%S')        
    
    #Create Multi-Index for cosmo data
    dataframe.columns = pd.MultiIndex.from_product([dataframe.columns.values.tolist(),['seviri']],
                                                                   names=['variable','substat'])       
                   
    #Assign to special cosmo dataframe, and join with main dataframe
    pv_station['df_seviri_' + year.split('_')[-1]] = dataframe
    
#    pv_station[df_name] = pd.concat([pv_station[df_name],dataframe],axis=1,join='inner')
       
    return pv_station

def import_apollo_cloud_data(key,pv_station,year,configfile,home):
    """
    Import cloud data from APOLLO retrieval, already split into stations
    by DLR
    
    args:
    :param key: string, name of station
    :param pv_station: dictionary with all info and data of pv station    
    :param year: string describing the data source
    :param configfile: string with name of configfile
    :param home: string with home path
    
    out:
    :return pv_station: dictionary with all info and data, including cloud data from APOLLO
    """

    config = load_yaml_configfile(configfile)
    path_apollo = os.path.join(home,config["path_apollo"])

    #Read in satellite COD values    
    dataframe = pd.DataFrame()        
    
    dfs = [pd.read_csv(os.path.join(path_apollo,filename),sep='\s+',
           comment='#',header=None,na_values=-999, parse_dates={'Timestamp':[0,1,2,3]},
           date_parser=lambda a,b,c,d: pd.to_datetime(a+b+c+d,format="%Y%m%d%H%M")) for filename in 
           list_files(path_apollo) if key in filename]
    
    dataframe = pd.concat(dfs,axis=0)
    dataframe.index = dataframe["Timestamp"]
    dataframe.drop(columns="Timestamp",inplace=True)
    
    #Sort index - bug in data found (JB, 20.07.2021)
    dataframe.sort_index(axis=0,inplace=True)
    
    #Add offset to get true observation time
    dataframe.index = dataframe.index + pd.Timedelta("668S")
    
    dataframe.columns = ['type','cov','phase','cot_AP','ctt','scat_mix','snow_prob']
    
    
    dataframe["cot_AP"] = dataframe["cot_AP"]/100
    dataframe["ctt"] = dataframe["ctt"]/10
    
    #Create Multi-Index for cosmo data
    dataframe.columns = pd.MultiIndex.from_product([dataframe.columns.values.tolist(),['apollo']],
                                                                   names=['variable','substat'])       
                   
    #Assign to special cosmo dataframe, and join with main dataframe
    pv_station['df_apollo_' + year.split('_')[-1]] = dataframe
       
    return pv_station

def prepare_cosmo_seviri_data(year,inv_config,key,pv_station,home):
    """
    args:
    :param year: string, year of campaign
    :param inv_config: dictionary of inversion configuration
    :param key: string, name of station
    :param pv_station: dictionary with all info and data of pv station    
    :param home: string, home path
    
    out:
    :return pv_station: dictionary with all info and data, including cloud data from COSMO,
    SEVIRI, APOLLO
    """
       
    list_configfiles = [inv_config["cosmopvcod_configfile"],inv_config["seviripvcod_configfile"]]    
    for filename in list_configfiles:
        configfile = os.path.join(home,filename)
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
    if inv_config["cosmo_input_flag"]["cloud"]:
        # call cosmo2pvcod
        print('Running cosmo2pvcod to extract surface properties')
        child = subprocess.Popen('cosmo2pvcod ' + configfile, shell=True)
        child.wait()
    else:
        print('cosmo2pvcod already run, read in cloud files')
        
    #Prepare surface data from SEVIRI
    if inv_config["seviri_input_flag"]:
        # call seviri2pvcod 
        print('Running seviri2pvcod to extract surface properties')
        child = subprocess.Popen('seviri2pvcod ' + configfile, shell=True)
        child.wait()
    else:
        print('seviri2pvcod already run, read in cloud files')

    for filename in list_configfiles:
        configfile = os.path.join(home,filename)
        if "COSMO" in filename:
            pv_station, pathname = find_nearest_grid_folder(configfile,key,
                                                pv_station,'cloud',home)       
            pv_station = import_cosmo_cloud_data(key,pv_station,year,pathname)
        elif "SEVIRI" in filename:
            pv_station, pathname = find_nearest_grid_folder(configfile,key,
                                                pv_station,'seviri',home)       
            pv_station = import_seviri_cloud_data(key,pv_station,year,pathname)   
            
            print(f'Importing data from APOLLO retrieval for {key}, {year}')
            pv_station = import_apollo_cloud_data(key,pv_station,year,
                                                  configfile,home)
    
    return pv_station

def moving_average_std(input_series,data_freq,window_avg,center=True):
    """
    Calculate moving average and standard deviation of input series

    Parameters
    ----------
    input_series : series with input data
    data_freq : timedelta, data frequency
    window_avg : timedelta, width of averaging window   
    center : boolean, whether the window is centered around old data points     

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
    
    if center:
        shift = -edge
    else:
        shift = 0
    
    #alternative method with pandas
    avg_alt = input_series.interpolate(method='linear',limit=int(np.ceil(edge/2))).\
        rolling(window=window_avg,min_periods=edge).\
            mean().shift(shift).rename('avg_pd')        
    std_alt = input_series.interpolate(method='linear',limit=int(np.ceil(edge/2))).\
        rolling(window=window_avg,min_periods=edge).\
        std().shift(shift).rename('std_pd')
    
    dataframe = pd.concat([avg_alt,std_alt],axis=1) #avg,std,
    
    return dataframe

def combine_cosmo_seviri(year,pv_station,str_window_avg):
    """
    

    Parameters
    ----------
    year : string, year under consideration
    pv_station : dictionary with information and data from PV station
    str_window_avg : string with time resolution for moving averages

    Returns
    -------
    df_compare_list : dataframe with combined COSMO and SEVIRI and APOLLO data

    """
                 
    #Size of moving average window
    window_avg = pd.Timedelta(str_window_avg)  
    
    df_cosmo = pv_station[f"df_cosmo_cod_{year.split('_')[1]}"]
    df_seviri = pv_station[f"df_seviri_{year.split('_')[1]}"]
    df_apollo = pv_station[f"df_apollo_{year.split('_')[1]}"]      
    
    #List for comparison of different CODs
    dfs_compare_list = []
    
    #List for concatenating days
    dfs = []
    
    #List for moving averages
    dfs_avg_std = []
    data_freq_AP = pd.Timedelta('15T')
    #2. Calculate moving average of apollo COD data        
    for day in pd.to_datetime(df_apollo.index.date).unique():                                            
        df_avg_std = moving_average_std(df_apollo.loc[day.strftime("%Y-%m-%d"),("cot_AP","apollo")],
                                  data_freq_AP, window_avg) 
        # df_avg_std = pd.concat([df_avg_std,moving_average_std(df_apollo.loc[day.strftime("%Y-%m-%d"),("cov","apollo")],
        #                           data_freq_AP, window_avg) ],axis=1)                
        dfs_avg_std.append(df_avg_std)
        #Alternative methods
        # cot_AP_rs = cot_AP.rolling(window=window_size_AP,min_periods=int(window_size_AP/2),center=True).mean() #.shift(-int(window_size_AP/2))                
        # cot_AP_rs_std = cot_AP.rolling(window=window_size_AP,min_periods=int(window_size_AP/2),center=True).std() #.shift(-int(window_size_AP/2))        
        
        df_AP_reindex_60 = df_avg_std.reindex(pd.date_range(start=df_avg_std.index[0].round(window_avg),
                                                 end=df_avg_std.index[-1].round(window_avg),freq=window_avg),
                                                 method='nearest',tolerance='5T').loc[day.strftime("%Y-%m-%d")]        
        dfs.append(df_AP_reindex_60)       
    
    #Assign reindexed values to comparison list
    df_compare = pd.concat(dfs,axis=0)
    df_compare.columns = pd.MultiIndex.from_product([[f"cot_AP_{str_window_avg}_avg",f"cot_AP_{str_window_avg}_std"]#,
                                                      #f"cot_AP_{str_window_avg}_avg2",f"cot_AP_{str_window_avg}_std2"]
                                                    ,['apollo']],names=['variable','substat'])                       
    dfs_compare_list.append(df_compare)        
    
    #Assign average and std to Apollo dataframe
    df_total = pd.concat(dfs_avg_std,axis=0)
    # df_apollo[(f"cot_AP_{str_window_avg}_avg","apollo")] = df_total["avg_conv"]
    # df_apollo[(f"cot_AP_{str_window_avg}_std","apollo")] = df_total["std_conv"]    
    df_apollo[(f"cot_AP_{str_window_avg}_avg","apollo")] = df_total["avg_pd"]
    df_apollo[(f"cot_AP_{str_window_avg}_std","apollo")] = df_total["std_pd"]    
        
    #List for concatenating days
    dfs = []
    
    #List for moving averages
    dfs_avg_std = []
    data_freq_hrv = pd.Timedelta('5T')        
    #SEVIRI data
    for day in pd.to_datetime(df_seviri.index.date).unique():                    
        df_avg_std = moving_average_std(df_seviri.loc[day.strftime("%Y-%m-%d"),("mean_COD_500","seviri")],
                                  data_freq_hrv, str_window_avg) 
                
        dfs_avg_std.append(df_avg_std)                
        
        #Alternative methods
        # cod_seviri_rs = cod_hrv.rolling(window=window_size_hrv,min_periods=int(window_size_hrv/2),center=True).mean()                
        # cod_seviri_rs_std = cod_hrv.rolling(window=window_size_hrv,min_periods=int(window_size_hrv/2),center=True).std()        
        
        #Reindex and assign to Apollo dataframe
        df_apollo.loc[day.strftime("%Y-%m-%d"),(f"COD_500_{str_window_avg}_avg","seviri")] = \
            df_avg_std["avg_pd"].reindex(df_apollo.loc[day.strftime("%Y-%m-%d")].index,method='nearest',tolerance='5T')        
        
        df_seviri_reindex_60 = df_avg_std.reindex(pd.date_range(start=df_avg_std.index[0].round(window_avg),
                                                           end=df_avg_std.index[-1].round(window_avg),freq=window_avg),
                                                          method='nearest',tolerance='5T')                        
        dfs.append(df_seviri_reindex_60)
        
    #Assign reindexed values to comparison list
    df_compare = pd.concat(dfs,axis=0)
    df_compare.columns = pd.MultiIndex.from_product([[f"COD_500_{str_window_avg}_avg",f"COD_500_{str_window_avg}_std"]#,
                                                      #f"COD_500_{str_window_avg}_avg2",f"COD_500_{str_window_avg}_std2"]
                                                    ,['seviri']],names=['variable','substat'])                       
    dfs_compare_list.append(df_compare)
    
    #Assign average and std to Seviri dataframe
    df_total = pd.concat(dfs_avg_std,axis=0)
    # df_seviri[(f"COD_500_{str_window_avg}_avg","seviri")] = df_total["avg_conv"]
    # df_seviri[(f"COD_500_{str_window_avg}_std","seviri")] = df_total["std_conv"]
    df_seviri[(f"COD_500_{str_window_avg}_avg","seviri")] = df_total["avg_pd"]
    df_seviri[(f"COD_500_{str_window_avg}_std","seviri")] = df_total["std_pd"]
            
    #COSMO data is already in hourly resolution  
    data_freq_cosmo = pd.Timedelta('60T')
    
    dfs = []    
    
    #List for moving averages
    dfs_avg_std = []
    dfs_cf_avg_std = []
    
    #df_cosmo[("COD_tot_600_std","cosmo")] = df_cosmo[("COD_tot_600","cosmo")].std()
    if window_avg == data_freq_cosmo:
        dfs_compare_list.append(df_cosmo)
    elif window_avg > data_freq_cosmo:
        for day in pd.to_datetime(df_cosmo.index.date).unique():                    
            df_avg_std = moving_average_std(df_cosmo.loc[day.strftime("%Y-%m-%d"),("COD_tot_600_avg","cosmo")],
                                      data_freq_cosmo, str_window_avg) 
                    
            dfs_avg_std.append(df_avg_std)   

            df_cf_avg_std = moving_average_std(df_cosmo.loc[day.strftime("%Y-%m-%d"),("cf_tot_avg","cosmo")],
                                      data_freq_cosmo, str_window_avg)["avg_pd"]
            df_cf_avg_std.rename(f"cf_tot_{str_window_avg}_avg",inplace=True)
            dfs_cf_avg_std.append(df_cf_avg_std)                 
            
            #Alternative methods
            # cod_seviri_rs = cod_hrv.rolling(window=window_size_hrv,min_periods=int(window_size_hrv/2),center=True).mean()                
            # cod_seviri_rs_std = cod_hrv.rolling(window=window_size_hrv,min_periods=int(window_size_hrv/2),center=True).std()        
            
            # #Reindex and assign to Apollo dataframe
            # df_apollo.loc[day.strftime("%Y-%m-%d"),(f"COD_500_{str_window_avg}_avg","seviri")] = \
            #     df_avg_std["avg_pd"].reindex(df_apollo.loc[day.strftime("%Y-%m-%d")].index,method='nearest',tolerance='5T')        
            
            df_cosmo_reindex = pd.concat([df_avg_std.reindex(pd.date_range(start=df_avg_std.index[0].round(window_avg),
                                              end=df_avg_std.index[-1].round(window_avg),freq=window_avg),
                                              method='nearest',tolerance='5T'),
                                          df_cf_avg_std.reindex(pd.date_range(start=df_cf_avg_std.index[0].round(window_avg),
                                              end=df_cf_avg_std.index[-1].round(window_avg),freq=window_avg),
                                              method='nearest',tolerance='5T')],axis=1)                        
            dfs.append(df_cosmo_reindex.loc[day.strftime("%Y-%m-%d")])
        
        #Assign reindexed values to comparison list
        df_compare = pd.concat(dfs,axis=0)
        df_compare.columns = pd.MultiIndex.from_product([[f"COD_tot_600_{str_window_avg}_avg",f"COD_tot_600_{str_window_avg}_std",
                                                          f"cf_tot_{str_window_avg}_avg"]                                                        
                                                        ,['cosmo']],names=['variable','substat'])                       
        dfs_compare_list.append(df_compare)       
        
        #Assign average and std to Seviri dataframe
        df_total = pd.concat(dfs_avg_std,axis=0)        
        df_cosmo[(f"COD_tot_600_{str_window_avg}_avg","cosmo")] = df_total["avg_pd"]
        df_cosmo[(f"COD_tot_600_{str_window_avg}_std","cosmo")] = df_total["std_pd"]
        df_cosmo[(f"cf_tot_{str_window_avg}_avg","cosmo")] = pd.concat(dfs_cf_avg_std,axis=0)        
    
    df_compare_list = pd.concat(dfs_compare_list,axis=1)    

    return df_compare_list

def combine_apollo_seviri_hires(year,pv_station):
    """
    Combine higher resolution (1 min) data from APOLLO and SEVIRI

    Parameters
    ----------
    year : string with year under consideration
    pv_station : dictionary with information and data from PV station

    Returns
    -------
    None.

    """
    
    data_freq_hrv = pd.Timedelta('5T')        
    df_seviri = pv_station[f"df_seviri_{year}"]  
    
    data_freq_AP = pd.Timedelta('15T')
    df_apollo = pv_station[f"df_apollo_{year}"]
    
    #Go through the timeres list and add SEVIRI and APOLLO to dataframes if appropriate
    for timeres in pv_station["timeres"]:
        window_avg = pd.Timedelta(timeres)
        if f"df_codfit_pyr_pv_{year}_{timeres}" in pv_station:
            if window_avg == data_freq_hrv:
                pv_station[f"df_codfit_pyr_pv_{year}_{timeres}"] = pd.concat([pv_station[f"df_codfit_pyr_pv_{year}_{timeres}"],
                              df_seviri.loc[pv_station[f"df_codfit_pyr_pv_{year}_{timeres}"].index]],axis=1)
                
            elif window_avg > data_freq_hrv:                
                #List for concatenating days
                dfs_cod = []
                
                #List for moving averages
                dfs_cod_avg_std = []                
                
                #1. Calculate moving average of HRV
                for day in pd.to_datetime(df_seviri.index.date).unique():                                            
                    if len(df_seviri.loc[day.strftime("%Y-%m-%d")]) > 1:
                        df_cod_avg_std = moving_average_std(df_seviri.loc[day.strftime("%Y-%m-%d"),("mean_COD_500","seviri")],
                                                            data_freq_hrv, window_avg) 
                                                
                        dfs_cod_avg_std.append(df_cod_avg_std)                        
                        
                        df_cod_seviri_reindex = df_cod_avg_std.reindex(pd.date_range(start=df_cod_avg_std.index[0].round(window_avg),
                                                                 end=df_cod_avg_std.index[-1].round(window_avg),freq=window_avg),
                                                                 method='nearest',tolerance='5T').loc[day.strftime("%Y-%m-%d")]        
                        
                        dfs_cod.append(df_cod_seviri_reindex)
                        
                
                #Assign average and std to cams dataframe
                df_total = pd.concat(dfs_cod_avg_std,axis=0)
                df_seviri[(f"mean_COD_500_{timeres}_avg","seviri")] = df_total.iloc[:,0]
                df_seviri[(f"mean_COD_500_{timeres}_std","seviri")] = df_total.iloc[:,1]
                
                df_seviri.sort_index(axis=1,level=1,inplace=True)        
                
                #Assign reindexed values to comparison list
                df_compare_tres = pd.concat(dfs_cod,axis=0)
                df_compare_tres.columns = pd.MultiIndex.from_product([[f"mean_COD_500_{timeres}_avg",
                                                                       f"mean_COD_500_{timeres}_std",]
                                                                ,['seviri']],names=['variable','substat'])                       
                                
                pv_station[f"df_codfit_pyr_pv_{year}_{timeres}"] = pv_station[f"df_codfit_pyr_pv_{year}_{timeres}"]\
                                .join(df_compare_tres,how='left')
                                
            #elif 


def plot_cod_lut(name,substat,df_day_result,day,radtype,
                 radname,errorname,radlabel,titlelabel,df_seviri,df_apollo,
                 df_cosmo,plotpath,titleflag=True):
    """
    

    Parameters
    ----------
    name : string, name of system
    substat : string, name of substation
    df_day_result : dataframe with result from specific day
    day : string, day under consideration
    radtype : string, radiation type (poa or down)
    radname : string, name of irradiance variable
    errorname : string, name of error vector
    radlabel : string, label for latex plot labels
    titlelabel : string, plot title label
    df_seviri : dataframe with SEVIRI data
    df_apollo : dataframe with APOLLO data        
    df_cosmo : dataframe with COSMO data
    plotpath : string, path to save plots
    titleflag : boolean, whether to add title. The default is True.            

    Returns
    -------
    None.

    """
    
    
    #Plot COD LUT  
    plt.ioff()
    plt.close('all')    
    plt.style.use("my_paper")                        
    
    fig = plt.figure(figsize=(15, 7))
    if titleflag:
        fig.suptitle(f"COD retrieved from {titlelabel} irradiance at {name.replace('_','')}, {substat.replace('_',' ')} on {day:%Y-%m-%d}", fontsize=16)
    
    ax = fig.add_subplot(1, 2, 1)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))      

    ## first the simulated
     #COD_range = np.linspace(0.0, 1.0, 5)
    lut_range = df_day_result[f'COD_{radtype}_table'].mean()[:,0]
    for COD_index, COD_value in enumerate(lut_range):
        irrad_simul, d_irrad_simul = [], []
        for time in df_day_result.index:
            if type(df_day_result.loc[time.strftime("%Y-%m-%d %H:%M:%S"),
                              f'COD_{radtype}_table']) != np.ndarray:
                irrad_simul.append(np.nan)
                d_irrad_simul.append(np.nan)
            else:
                # reads out the dataframe containing the COD / irrad lookuptable for each timestep
                OD_array, F_array = df_day_result.loc[time.strftime("%Y-%m-%d %H:%M:%S"),
                                  f'COD_{radtype}_table'].transpose()                
    
                F_value = F_array[COD_index]
                #if abs(F_value) > 1000: F_value = np.nan
                irrad_simul.append(F_value)
                d_irrad_simul.append(F_value*0.02)

        if COD_index%2 == 0 or COD_index == len(lut_range) - 1:
            ax.plot(df_day_result.index, irrad_simul,
                label= r"$\tau_\mathrm{{wc}}={:.2f}$"
                .format(COD_value), ls="--")
            confidence_band(ax,df_day_result.index, irrad_simul, d_irrad_simul)
            #if settings['print_flag'] == True: print(f"plotting the line of simulated values: t, F: \n {timeindex},{irrad_simul}")

    # then the measured

    #dF_meas = 0.02*irradiance_meas
    dF_meas = df_day_result.loc[:,errorname]
    df_plot = df_day_result.loc[:,radname]
    
    ax.plot(df_plot.index,df_plot,label=r"$G^{}_\mathrm{{pyr,{},{}}}$"
             .format(radlabel,name.replace("_",""),substat.replace('_',' ')))
    confidence_band(ax,df_plot.index,df_plot, dF_meas)
    ax.legend(loc = "best")
    ax.set_ylim(0,1200)
    datemin = pd.Timestamp(df_day_result.index[0] - pd.Timedelta('30T'))
    datemax = pd.Timestamp(df_day_result.index[-1] + pd.Timedelta('30T'))                        
    ax.set_xlim([datemin, datemax])            
    ax.set_ylabel("Irradiance {}".format(plotlabels["wmsq"]))
    ax.set_xlabel(r"Time (UTC)")
    ax.legend(loc = "best")
    
    # plot the retrieved COD
    
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    ax2 = fig.add_subplot(1,2,2)            
    ax2.set_xlim([datemin, datemax])
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))      
    # Todo: Do a subplot routine to combine this with the other plot!!
    
    ax2.plot(df_day_result.index , df_day_result[f"COD_550_{radtype}_inv"], 
             label=r'$\tau_\mathrm{{wc,550nm,{},{}}}$'\
                     .format(name.replace("_",""),substat.replace("_",' ')),color=cycle[0])
    confidence_band(ax2,df_day_result.index , df_day_result[f"COD_550_{radtype}_inv"],\
                    df_day_result[f"error_COD_550_{radtype}_inv"],color=cycle[0])    
    ax2.set_ylabel(plotlabels["COD"])
    #ax2.set_ylim([0,150])
    
    #Compare COD retrieval to SEVIRI-APOLLO, SEVIRI-HRV, COSMO
    max_cosmo = 0; max_seviri = 0; max_apollo = 0    
                        
    if day in df_seviri.index.date:    
        df_day_seviri = df_seviri.xs('seviri',level='substat',axis=1).loc[day.strftime("%Y-%m-%d")]        
        max_seviri = df_day_seviri["mean_COD_500"].max()
        ax2.plot(df_day_seviri.index,df_day_seviri["mean_COD_500"],linestyle='--',color=cycle[1],
                 label=r'$\tau_\mathrm{c,500nm,MSG-SEVIRI-HRV}$')
        confidence_band(ax2,df_day_seviri.index,df_day_seviri["mean_COD_500"],
                        df_day_seviri["iqr_COD_500"],color=cycle[1])        
        
    if day in df_apollo.index.date:    
        df_day_apollo = df_apollo.xs('apollo',level='substat',axis=1).loc[day.strftime("%Y-%m-%d")]        
        max_apollo = df_day_apollo["cot_AP"].max()                    
        ax2.plot(df_day_apollo.index,df_day_apollo["cot_AP"],linestyle='None',marker='x',color=cycle[2],
                 label=r'$\tau_\mathrm{c,vis,MSG-SEVIRI-APNG\_1.1}$')        
        
    if day in df_cosmo.index.date:    
        df_day_cosmo = df_cosmo.xs('cosmo',level='substat',axis=1).loc[day.strftime("%Y-%m-%d")]
        max_cosmo = df_day_cosmo["COD_tot_600_avg"].max()        
        ax2.errorbar(df_day_cosmo.index,df_day_cosmo["COD_tot_600_avg"],yerr=df_day_cosmo["COD_tot_600_iqr"],
                     linestyle='None',marker='o',color=cycle[3],label=r'$\tau_\mathrm{600nm,COSMO}$')        
    
    if radtype:
        max_cod = np.max([max_seviri,max_apollo,max_cosmo,
                      df_day_result[f"COD_550_{radtype}_inv"].max()])                
    else:
        max_cod = np.max([max_seviri,max_apollo,max_cosmo,
                      df_day_result[f"COD_550_{radtype}inv"].max()])
    
    max_cod = np.ceil(max_cod/10)*10
    ax2.set_ylim([0,max_cod])
    #ax2.set_yscale("log")
    ax2.set_xlabel(r"Time (UTC)")
    ax2.legend()
        
    fig.tight_layout()
    fig.subplots_adjust(top=0.94)
    
    plt.savefig(os.path.join(plotpath,"COD_lut_from_Etot{}_{}_{}_{}.png".format(
        radtype,name,substat,day.strftime("%Y-%m-%d")))) #, dpi = 300)
    plt.close(fig)
    
def avg_std_cod_retrieval(day,df_day_result,substat,radtype,df_seviri,df_apollo,df_cosmo,timeres,
                       str_window_avg,cs_threshold):
    """
    

    Parameters
    ----------
    day : string, day under consideration
    df_day_result : dataframe with results for specific day
    substat : string, name of substation
    radtype : string, type of irradiance (poa or down)
    df_seviri : dataframe with SEVIRI data
    df_apollo : dataframe with APOLLO data        
    df_cosmo : dataframe with COSMO data
    timeres : string, time resolution of data 
    str_window_avg : string, width of window for moving average
    cs_threshold : float, threshold for clear sky calculation

    Returns
    -------
    dataframe with combined results

    """
    
    #Original time resolution
    data_freq = pd.Timedelta(timeres)
    #Size of moving average window
    window_avg = pd.Timedelta(str_window_avg)    
    
    #1. Calculate moving average of COD retrieval        
    df_cod_avg_std = moving_average_std(df_day_result[f"COD_550_{radtype}_inv"],data_freq,window_avg)
    df_cod_eff_avg_std = moving_average_std(df_day_result[f"COD_eff_550_{radtype}_inv"],data_freq,window_avg)
    
    #Calculate moving average of clearness index    
    df_ki_avg_std = moving_average_std(df_day_result[f"k_index_{radtype}"],data_freq,window_avg)
    
    #Calculate moving average of cloud fraction
    if f"cf_{radtype}_{str_window_avg}_avg" not in df_day_result.columns:
        df_day = deepcopy(df_day_result[f"cloud_fraction_{radtype}"])
        df_day.loc[df_day < 0] = np.nan
        df_cf_avg_std = moving_average_std(df_day,data_freq,window_avg)
        df_day_result[f"cf_{radtype}_{str_window_avg}_avg"] = df_cf_avg_std["avg_pd"]
    
    #Assign to dataframe
    # df_day_result[f"COD_550_{radtype}_inv_{str_window_avg}_avg"] = df_cod_avg_std["avg_conv"]
    # df_day_result[f"COD_550_{radtype}_inv_{str_window_avg}_std"] = df_cod_avg_std["std_conv"]
    df_day_result[f"COD_550_{radtype}_inv_{str_window_avg}_avg"] = df_cod_avg_std["avg_pd"]
    df_day_result[f"COD_550_{radtype}_inv_{str_window_avg}_std"] = df_cod_avg_std["std_pd"]
    df_day_result[f"COD_eff_550_{radtype}_inv_{str_window_avg}_avg"] = df_cod_eff_avg_std["avg_pd"]
    df_day_result[f"COD_eff_550_{radtype}_inv_{str_window_avg}_std"] = df_cod_eff_avg_std["std_pd"]
    df_day_result[f"k_index_{radtype}_{str_window_avg}_avg"] = df_ki_avg_std["avg_pd"]
    
    
    #Calculate the variability index (Coimbra et al.)
    df_day_result[f"v_index_{radtype}_{str_window_avg}_avg"] = df_day_result[f"k_index_{radtype}"].rolling(window_avg).apply(v_index)
    
    #Calculate overshoot index
    df_day_result[f"overshoot_index_{radtype}_{str_window_avg}_avg"] = df_day_result[f"k_index_{radtype}"].rolling(window_avg).\
        apply(overshoot_index,args=(cs_threshold[1],))
        
    
    #Reindex and assign to Apollo dataframe
    df_apollo[(f"COD_550_{radtype}_inv_{str_window_avg}_avg",substat)] = \
        df_cod_avg_std["avg_pd"].reindex(df_apollo.index,method='nearest',tolerance='5T')
        
    #Alternative methods for moving average functions
    # cod_avg_rs5 = cod_retrieval.rolling(window=window_avg,min_periods=int(window_size/2)).mean().shift(-int(window_size/2))
    # cod_std_rs5 = cod_retrieval.rolling(window=window_avg,min_periods=int(window_size/2)).std().shift(-int(window_size/2))    
        
    dfs_reindex = []
    reindex_names = [f"COD_550_{radtype}_inv_{str_window_avg}_avg",f"COD_550_{radtype}_inv_{str_window_avg}_std",
                     f"COD_eff_550_{radtype}_inv_{str_window_avg}_avg",f"COD_eff_550_{radtype}_inv_{str_window_avg}_std",
                     f"k_index_{radtype}_{str_window_avg}_avg",f"cf_{radtype}_{str_window_avg}_avg",
                     f"v_index_{radtype}_{str_window_avg}_avg",f"overshoot_index_{radtype}_{str_window_avg}_avg"]
    
    for reindex_name in reindex_names:
        df_reindex = df_day_result[reindex_name].reindex(pd.date_range(start=df_cod_avg_std.index[0].round(window_avg),
                        end=df_cod_avg_std.index[-1].round(window_avg),freq=window_avg)
                          ,method='nearest',tolerance='5T')
        df_reindex.rename((reindex_name,substat),inplace=True)   
        dfs_reindex.append(df_reindex)              
    
    return pd.concat(dfs_reindex,axis=1) #ki_avg_reindex_60,

def plot_cod_comparison(name,substat,df_day_result,day,sza_index_day,
                        str_window_avg,window_avg_cf,radtype,radname,errorname,radlabel,
                        titlelabel,df_seviri,df_apollo,df_cosmo,plotpath,titleflag=True):
    """
    

    Parameters
    ----------
    name : string, name of system
    substat : string, name of substation
    df_day_result : dataframe with result from specific day
    sza_index_day : index with SZA limits        
    str_window_avg : string with width of averaging window for COD
    window_avg_cf : string with width of averaging window for cloud fraction
    radtype : string, radiation type (poa or down)
    radname : string, name of irradiance variable
    errorname : string, name of error vector
    radlabel : string, label for latex plot labels
    titlelabel : string, plot title label
    df_seviri : dataframe with SEVIRI data
    df_apollo : dataframe with APOLLO data        
    df_cosmo : dataframe with COSMO data    
    plotpath : string, path to save plots
    titleflag : boolean, whether to add title. The default is True.    

    Returns
    -------
    None.

    """
    
    
    #Plot COD comparison with variances
    plt.ioff()
    #plt.close('all')    
    plt.style.use("my_paper")                                
    
    fig,axs = plt.subplots(nrows=2,ncols=2,sharex='all',sharey='all',squeeze=True,figsize=(16,9))
    fig.subplots_adjust(wspace=0.05,hspace=0.05)  
    fig.suptitle(f'COD retrieval comparison at {name} on {day}')
    fig.subplots_adjust(top=0.94)            
    
    ax = axs.flat[0]
    #Plot the original time series
    ax.plot(df_day_result.index,df_day_result[f"COD_550_{radtype}_inv"],label="Retrieval data",
            linestyle='--',color='gray',marker='o')    
    ax.plot(df_day_result.index,df_day_result[f"COD_eff_550_{radtype}_inv"],label="COD eff",
            linestyle=':',color='k')
    ax.plot(df_day_result.index,df_day_result[f"COD_550_{radtype}_inv_{str_window_avg}_avg"],
            label=f'{str_window_avg} avg',color='r')    # (Box1DKernel)
    ax.plot(df_day_result.index,df_day_result[f"COD_eff_550_{radtype}_inv_{str_window_avg}_avg"],
            label=f'COD (eff) {str_window_avg} avg',color='orange')    # (Box1DKernel)
    confidence_band(ax,df_day_result.index,
                    df_day_result[f"COD_550_{radtype}_inv_{str_window_avg}_avg"],
                    df_day_result[f"COD_550_{radtype}_inv_{str_window_avg}_std"],color='r')                        
    # ax.plot(df_day_result.index,df_day_result[f"COD_550_{radtype}_inv_{str_window_avg}_avg2"],
    #         label=f'{str_window_avg} MA alt',color='k')    # (Box1DKernel)
    # confidence_band(ax,df_day_result.index,
    #                 df_day_result[f"COD_550_{radtype}_inv_{str_window_avg}_avg2"],
    #                 df_day_result[f"COD_550_{radtype}_inv_{str_window_avg}_std2"],color='k')                        
    
    axr = ax.twinx()    
    axr.plot(df_day_result.index,df_day_result[f"cf_{radtype}_{window_avg_cf}_avg"],
            label=f'{window_avg_cf} cf',color='c')
    # axr.plot(df_day_result.index,df_day_result[f"cloud_fraction_{radtype}"],
    #          label='cf mask',color='m',linestyle='',marker='x',markersize=4)
    axr.set_ylabel(rf"$<cf>_\mathrm{{{window_avg_cf}}}$")
    axr.set_ylim([-0.05,1.05])
    axr.yaxis.grid(False)
    #ax.yaxis.grid(False)
    
    ax.legend()
    ax.set_ylabel('COD',position=(-0.1,0))            
    ax.set_title(f"{substat}, {radtype}")            
    
    max_apollo = 0; max_seviri = 0; max_cosmo = 0;
    
    ax2 = axs.flat[1]
    if day in df_apollo.index.date:    
        df_day_apollo = df_apollo.xs('apollo',level='substat',axis=1).loc[day.strftime("%Y-%m-%d")]                
        max_apollo = df_day_apollo["cot_AP"].max()
        
        #Masks for cloud type
        lowcloud_mask = df_day_apollo["type"]==5
        mediumcloud_mask = df_day_apollo["type"]==6
        highcloud_mask = df_day_apollo["type"]==7
        thincloud_mask = df_day_apollo["type"]==8
        
        #Masks for scattered mixed flag
        notscat_notmix = df_day_apollo["scat_mix"]==00
        scat_notmix = df_day_apollo["scat_mix"]==10# 10: scattered, but not mixed
        notscat_mix = df_day_apollo["scat_mix"]==1# 01: not scattered, but mixed
        scat_mix = df_day_apollo["scat_mix"]==11# 11: scattered, mixed
        
        cot_AP = df_day_apollo["cot_AP"]
        ax2.plot(cot_AP.loc[lowcloud_mask].index,cot_AP.loc[lowcloud_mask],
                 linestyle='None',marker='o',label='Low',color='k')
        ax2.plot(cot_AP.loc[mediumcloud_mask].index,cot_AP.loc[mediumcloud_mask],
                 linestyle='None',marker='>',label='Medium',color='k')
        ax2.plot(cot_AP.loc[highcloud_mask].index,cot_AP.loc[highcloud_mask],
                 linestyle='None',marker='*',label='High',color='k')
        ax2.plot(cot_AP.loc[thincloud_mask].index,cot_AP.loc[thincloud_mask],
                 linestyle='None',marker='D',label='Thin',color='k')
        
        # ax2.plot(cot_AP.loc[notscat_notmix].index,cot_AP.loc[notscat_notmix],linestyle='None',marker='o',label='not scat not mix',color='k')
        # ax2.plot(cot_AP.loc[scat_notmix].index,cot_AP.loc[scat_notmix],linestyle='None',marker='>',label='scat not mix',color='k')
        # ax2.plot(cot_AP.loc[notscat_mix].index,cot_AP.loc[notscat_mix],linestyle='None',marker='*',label='not scat mix',color='k')
        # ax2.plot(cot_AP.loc[scat_mix].index,cot_AP.loc[scat_mix],linestyle='None',marker='D',label='scat mix',color='k')                
        
        ax2.plot(df_day_apollo.index,df_day_apollo[f"cot_AP_{str_window_avg}_avg"],
                 label=f'{str_window_avg} MA',color='g') # (Box1DKernel)
        confidence_band(ax2, df_day_apollo.index, df_day_apollo[f"cot_AP_{str_window_avg}_avg"], 
                        df_day_apollo[f"cot_AP_{str_window_avg}_std"], color='g')                
        
        ax2.legend()
        
    ax2.set_title('SEVIRI-APNG_1.1')    
    
    #SEVIRI data
    ax3 = axs.flat[2]                
    if day in df_seviri.index.date:    
        df_day_seviri = df_seviri.xs('seviri',level='substat',axis=1).loc[day.strftime("%Y-%m-%d")]        
        max_seviri = df_day_seviri["mean_COD_500"].max()
        
        
        cod_hrv = df_day_seviri["mean_COD_500"]
        ax3.plot(cod_hrv.index,cod_hrv,label='Raw data',linestyle='--',color='gray')
                
        ax3.plot(df_day_seviri.index,df_day_seviri[f"COD_500_{str_window_avg}_avg"],
                 label=f'{str_window_avg} MA',color='b') # (Box1DKernel)
        confidence_band(ax3, df_day_seviri.index, df_day_seviri[f"COD_500_{str_window_avg}_avg"], 
                        df_day_seviri[f"COD_500_{str_window_avg}_std"], color='b')                
        
        ax3.legend()        
        
    ax3.set_title('SEVIRI-HRV')
    
    #4th axis
    ax4 = axs.flat[3]        
    
    ax4.plot(df_day_result.index,df_day_result[f"COD_550_{radtype}_inv_{str_window_avg}_avg"],
             label=f'{substat}, {radtype}',color='r') # (Box1DKernel)
    confidence_band(ax4,df_day_result.index, df_day_result[f"COD_550_{radtype}_inv_{str_window_avg}_avg"], 
                    df_day_result[f"COD_550_{radtype}_inv_{str_window_avg}_std"], color='r')
    
    if day in df_apollo.index.date:
        ax4.plot(df_day_apollo.index,df_day_apollo[f"cot_AP_{str_window_avg}_avg"],
                 label='APNG_1.1',color='g') # (pd rolling) df_day_apollo = df_day_apollo.loc["2019-07-31"]
        confidence_band(ax4,df_day_apollo.index,df_day_apollo[f"cot_AP_{str_window_avg}_avg"],
                        df_day_apollo[f"cot_AP_{str_window_avg}_std"], color='g')
    
    if day in df_seviri.index.date:
        ax4.plot(df_day_seviri.index,df_day_seviri[f"COD_500_{str_window_avg}_avg"],
                 label='HRV',color='b') # (pd rolling)
        confidence_band(ax4,df_day_seviri.index,df_day_seviri[f"COD_500_{str_window_avg}_avg"],
                        df_day_seviri[f"COD_500_{str_window_avg}_std"],color='b')
    
    if day in df_cosmo.index.date:    
        df_day_cosmo = df_cosmo.xs('cosmo',level='substat',axis=1).loc[day.strftime("%Y-%m-%d")]
        max_cosmo = df_day_cosmo["COD_tot_600_avg"].max()
        ax4.errorbar(df_day_cosmo.index,df_day_cosmo["COD_tot_600_avg"],df_day_cosmo["COD_tot_600_iqr"],
                     label='COSMO',linestyle='None',marker = 'x',color='k')        
        
    ax4.legend()
    ax4.set_title('All')
    ax4.set_xlabel("Time (UTC)",position=(0,0))
    
    max_cod = np.ceil(np.max([df_day_result[f"COD_550_{radtype}_inv"].max(),
                              max_seviri,max_apollo,max_cosmo])/10)*10                
    
    for ax in axs.flat:
        datemin = pd.Timestamp(df_day_apollo.index[0] - pd.Timedelta('30T'))
        datemax = pd.Timestamp(df_day_apollo.index[-1] + pd.Timedelta('30T'))    
        ax.set_xlim([datemin, datemax])
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))      
        ax.axvspan(datemin,sza_index_day[0],alpha=0.2,color='gray')
        ax.axvspan(sza_index_day[-1],datemax,alpha=0.2,color='gray')
        if max_cod <= 30:
             ax.set_ylim([0.1,max_cod])
             ax.set_yscale('log')
        else:                
            ax.set_ylim([0,max_cod])
        
    # ax4r = ax4.twinx()
    # ax4r.plot(epyr_test.index,epyr_test,label=r"G^\angle_\mathrm{{{}}}".format(sensor))
    # ax4r.set_ylabel(r"Irradiance (W/m$^2$)")
    
    fig.tight_layout()
    plt.savefig(os.path.join(plotpath,f"COD_retrieval_comparison_{substat}_{radtype}_{str_window_avg}_mvavg_{name.replace('_','')}_{day}.png"))
   
    plt.close(fig)    
    
def plot_cod_overshoot_comparison(name,substat,df_day_result,day,sza_index_day,
                        str_window_avg,window_avg_cf,radtype,radname,errorname,radlabel,
                        titlelabel,df_seviri,df_apollo,df_cosmo,plotpath,titleflag=True):
    """
    
    Parameters
    ----------
    name : string, name of system
    substat : string, name of substation
    df_day_result : dataframe with result from specific day
    sza_index_day : index with SZA limits        
    str_window_avg : string with width of averaging window for COD
    window_avg_cf : string with width of averaging window for cloud fraction
    radtype : string, radiation type (poa or down)
    radname : string, name of irradiance variable
    errorname : string, name of error vector
    radlabel : string, label for latex plot labels
    titlelabel : string, plot title label
    df_seviri : dataframe with SEVIRI data
    df_apollo : dataframe with APOLLO data        
    df_cosmo : dataframe with COSMO data    
    plotpath : string, path to save plots
    titleflag : boolean, whether to add title. The default is True.    

    Returns
    -------
    None.

    """
    
    plt.ioff()
    plt.style.use("my_paper")                                

    fig, axs = plt.subplots(figsize=(15,10),
    nrows=4, ncols=1, sharex=True, sharey=False, 
    gridspec_kw={'height_ratios':[2,2,1,1]})

    ax = axs[0]
    df_day_result.plot(y=radname,ax=ax,label=rf"$G_{{\rm tot}}^{radlabel}$ ({substat})")
    if "pv" not in radname:
        df_day_result.plot(y=f"Etot{radtype}_clear_Wm2",ax=ax,label=rf"$G_{{\rm tot,clear}}^{radlabel}$ ({substat})")
    else:
        df_day_result.plot(y=f"Etot{radtype}_pv_clear_Wm2",ax=ax,label=rf"$G_{{\rm tot,clear}}^{radlabel}$ ({substat})")
    df_day_result.plot(y=f"dE_overshoot_{radtype}_Wm2",ax=ax,label=rf"$\Delta G_{{\rm tot,overshoot}}^{radlabel}$ ({substat})")
    
    ax2 = axs[1]
    df_day_result.plot(y=f"COD_550_{radtype}_inv",ax=ax2,color='m',label=r'COD$_\mathrm{ (cf = 1)}$')
    df_day_result.plot(y=f"COD_550_{radtype}_inv_{str_window_avg}_avg",ax=ax2,color='k',label=rf'$\langle$COD$_\mathrm{{ (cf = 1)}}\rangle_\mathrm{{{str_window_avg}}}$')
    df_day_result.plot(y=f"COD_eff_550_{radtype}_inv",ax=ax2,color='r',label=r'COD$_\mathrm{eff}$')
    df_day_result.plot(y=f"COD_eff_550_{radtype}_inv_{str_window_avg}_avg",ax=ax2,color='gray',label=rf'$\langle$COD$_\mathrm{{eff}}\rangle_\mathrm{{{str_window_avg}}}$')
    if day in df_seviri.index.date:    
        df_seviri.loc[day.strftime("%Y-%m-%d")].plot(y=("mean_COD_500","seviri"),ax=ax2,color='c',label="COD (SEVIRI-HRV)")
    if day in df_apollo.index.date:    
        df_apollo_rs = df_apollo.reindex(pd.date_range(start=df_apollo.index[0].round('15T'),
                                                 end=df_apollo.index[-1].round('15T'),freq='15T'),
                                                 method='nearest',tolerance='5T')
        df_apollo_rs.loc[day.strftime("%Y-%m-%d")].plot(y=('cot_AP',"apollo"),ax=ax2,color='b',label='COD (APNG)')        
    
    ax3 = axs[2]    
    df_day_result.plot(y=f"v_index_{radtype}_{str_window_avg}_avg",label=rf'$\langle V_i\rangle_\mathrm{{{str_window_avg}}}$',ax=ax3)    
    
    ax4 = axs[3]
    cf_nans = deepcopy(df_day_result[f"cloud_fraction_{radtype}"])
    cf_nans[cf_nans < 0] = np.nan 
    cf_nans.plot(ax=ax4,label='cf mask',color='m',linestyle='',marker='x',markersize=4)
    df_day_result.plot(y=f"cf_{radtype}_{window_avg_cf}_avg",ax=ax4,label=rf'$\langle$cf$\rangle_\mathrm{{{window_avg_cf}}}$',color='b')
    df_day_result.plot(y=f"cf_{radtype}_{window_avg_cf}_avg_alt",ax=ax4,label=rf'$\langle$cf$\rangle_\mathrm{{{window_avg_cf}}}$ (right)',linestyle=':',color='b')
    df_day_result.plot(y=f"overshoot_index_{radtype}_{str_window_avg}_avg",ax=ax4,\
                       label=rf"$\langle\Delta G'_i\rangle_\mathrm{{{str_window_avg}}}$",color='g') 
    overshoot_product = df_day_result[f"overshoot_index_{radtype}_{str_window_avg}_avg"]*df_day_result[f"k_index_{radtype}_{str_window_avg}_avg"]
    overshoot_product.plot(ax=ax4,label=rf"$\langle\Delta G'_i\rangle_\mathrm{{{str_window_avg}}}*\langle k_i\rangle_\mathrm{{{str_window_avg}}}$",
                           color='r')    
    ax4.legend()
    
    ax.set_title(f"Measured irradiance and retrieved COD from {substat}, {radtype} on {day}")
    ax.set_ylabel(r'Irradiance (W/m$^2$)')
    ax2.set_ylabel('COD')
    #ax2.set_ylim([0,50])
    ax3.set_ylabel(r'Variability')
    ax4.set_ylabel('cf / Overshoot')
    ax4.set_xlabel('Time (UTC)')
    
    fig.tight_layout()
    plt.savefig(os.path.join(plotpath,f"COD_overshoot_comparison_{substat}_{radtype}_{str_window_avg}_mvavg_{name.replace('_','')}_{day}.png"))
   
    plt.close(fig)        
    
def cod_analysis_plots(name,pv_station,substat_pars,year,
                 savepath,sza_limit,cs_threshold,str_window_avg,
                 window_avg_cf,flags):
    """
    

    Parameters
    ----------
    name : string, name of PV station
    pv_station : dictionary with information and data from PV station
    substat_pars : dictionary with substation parameters
    year : string with year under consideration
    savepath : string with path to save plots
    sza_limit : float with SZA limit of simulation and results
    str_window_avg : string with width of averaging window
    flags : dictionary of booleans for plotting

    Returns
    -------
    df_combine : dataframe with combined data including moving averages

    """
    
    res_dirs = list_dirs(savepath)
    savepath = os.path.join(savepath,'COD_Plots')
    if 'COD_Plots' not in res_dirs:
        os.mkdir(savepath)    
        
    stat_dirs = list_dirs(savepath)
    savepath = os.path.join(savepath,name)
    if name not in stat_dirs:
        os.mkdir(savepath)
    
    res_dirs = list_dirs(savepath)
    savepath_LUT = os.path.join(savepath,'LUT')        
    if 'LUT' not in res_dirs:
        os.mkdir(savepath_LUT)
    savepath_comp = os.path.join(savepath,'Comparison')        
    if 'Comparison' not in res_dirs:
        os.mkdir(savepath_comp)            

    res_dirs = list_dirs(savepath_comp)
    savepath_comp = os.path.join(savepath_comp,str_window_avg)
    if str_window_avg not in res_dirs:
        os.mkdir(savepath_comp)

    dfs_combine = []        
    
    for substat in substat_pars:        
        substat_dirs_LUT = list_dirs(savepath_LUT)
        plotpath_LUT = os.path.join(savepath_LUT,substat)
        if substat not in substat_dirs_LUT:
            os.mkdir(plotpath_LUT)
            
        substat_dirs_comp = list_dirs(savepath_comp)
        plotpath_comp = os.path.join(savepath_comp,substat)
        if substat not in substat_dirs_comp:
            os.mkdir(plotpath_comp)
    
        timeres = substat_pars[substat]["t_res_inv"]
        
        #Get radnames
        radname = substat_pars[substat]["name"]                    
        if "pyr" in radname:
            radnames = [radname,radname.replace('poa','down')]
        else:
            radnames = [radname]
        
        df_result = pv_station[f"df_codfit_pyr_pv_{year}_{timeres}"].loc[:,pd.IndexSlice[:,[substat,"sun"]]]
        
        #If there is no data for a specific radname remove it
        for radname in radnames:
            if (radname,substat) not in df_result.columns:
                radnames.remove(radname)
                print(f"There is no data for {radname}")
        
        radtypes = []        
        for radname in radnames:
            if "poa" in radname:
                radtypes.append("poa")                
            elif "down" in radname:
                radtypes.append("down")                    
        
        dfs_results = []
        
        df_cosmo = pv_station[f"df_cosmo_cod_{year}"]
        df_seviri = pv_station[f"df_seviri_{year}"]
        df_apollo = pv_station[f"df_apollo_{year}"]
        
        day_list = [group[0] for group in df_result
                               .groupby(df_result.index.date)]
        day_index_list = [pd.to_datetime(group[1].index) for group in df_result
                               .groupby(df_result.index.date)]
        
        for radtype in radtypes:
            print("COD analysis and plots for %s, %s for %s irradiance at %s" % (name,substat,radtype,timeres))            
            if radtype == "poa":                
                radname = substat_pars[substat]["name"]
                if "pv" not in radname:
                    errorname = "error_" + '_'.join([radname.split('_')[0],radname.split('_')[-1]])
                else:
                    errorname = "error_" + radname
                
            elif radtype == "down":
                radname = substat_pars[substat]["name"].replace("poa","down")            
                errorname = "error_" + '_'.join([radname.split('_')[0],radname.split('_')[-1]])                                               
            
            #Plotlabels
            if radtype == "poa":
                radlabel = "\\angle"
                titlelabel = "tilted"
            elif radtype == "down":
                radlabel = "\\downarrow"
                titlelabel = "downward"
            # elif radtype == "":
            #     radlabel = "\\angle"
            #     titlelabel = "tilted"
            
            dfs_radtype_combine = []   
            dfs_radtype_results = []
            # loop over the dates and plot for each day
            for day_no, day_index in enumerate(day_index_list):
                day = day_list[day_no] 
            
                df_day_result = df_result.xs(substat,level='substat',axis=1).loc[day_index]
                sza_index_day = df_result.loc[day_index].loc[df_result[("sza","sun")] <= sza_limit].index 
                
                if df_day_result[f"COD_550_{radtype}_inv"].dropna(how="all").empty:
                    print(f"No COD data from {substat}, {radtype} on {day}")
                else:                                                    
                    if flags["lut"]:
                        print(f'Plotting COD LUT for {substat}, {radtype} on {day}')
                        plot_cod_lut(name,substat,df_day_result,day,radtype,
                                  radname,errorname,radlabel,titlelabel,df_seviri,df_apollo,
                                  df_cosmo,plotpath_LUT)
                    
                    #Calculate moving average and standard deviation for COD retrieval
                    dfs_radtype_combine.append(avg_std_cod_retrieval(day,df_day_result,substat,radtype,
                                                          df_seviri,df_apollo,df_cosmo,timeres,
                                                          str_window_avg,cs_threshold))                    
                    
                    #Plot comparison plot for each day if required
                    if flags["compare"]:      
                        print(f'Plotting COD comparison for {substat}, {radtype} on {day}')
                        plot_cod_comparison(name,substat,df_day_result,day,sza_index_day,
                            str_window_avg,window_avg_cf,radtype,radname,errorname,radlabel,
                            titlelabel,df_seviri,df_apollo,df_cosmo,plotpath_comp)
                        
                        plot_cod_overshoot_comparison(name,substat,df_day_result,day,sza_index_day,
                            str_window_avg,window_avg_cf,radtype,radname,errorname,radlabel,
                            titlelabel,df_seviri,df_apollo,df_cosmo,plotpath_comp)
                        
                    dfs_radtype_results.append(df_day_result)
                        
            if len(dfs_radtype_combine) > 0:
                df_radtype_combine = pd.concat(dfs_radtype_combine,axis=0)        
                dfs_combine.append(df_radtype_combine)
            
                df_radtype_results = pd.concat(dfs_radtype_results,axis=0)
                dfs_results.append(df_radtype_results)
            
        #Combine all results after calculating moving averages
        df_results_combine = pd.concat(dfs_results,axis=1).\
        filter(regex=f'COD_.*{str_window_avg}_avg|{str_window_avg}_std|v_index_.*{str_window_avg}|overshoot_index_.*{str_window_avg}|k_index_.*{str_window_avg}', axis=1)
        #Added this to make up for the fact that cloud fraction was calculated in previous script
        # df_results_combine = df_results_combine.loc[:,~df_results_combine.columns.duplicated()]                               
        # df_results_combine.sort_index(axis=1,level=1,inplace=True)   
        df_results_combine.columns = pd.MultiIndex.from_product([df_results_combine.columns.to_list(),[substat]],
                               names=['variable','substat'])   
            
        pv_station[f"df_codfit_pyr_pv_{year}_{timeres}"] = pd.concat([pv_station[f"df_codfit_pyr_pv_{year}_{timeres}"],
                                  df_results_combine],axis=1)
        pv_station[f"df_codfit_pyr_pv_{year}_{timeres}"].sort_index(axis=1,level=1,inplace=True)        
        
        #Combine results for statistics
        df_combine = pd.concat(dfs_combine,axis=1,join='inner')
        df_combine = df_combine.join(pv_station[f"df_compare_cod_{year}_{str_window_avg}"],how='left')
        df_combine.columns.names = ['variable','substat']                
            
    return df_combine            
            
def scatter_plot_cod(xvals,yvals,cvals,labels,titlestring,figname,
                     cod_range,plot_style,title=True,logscale=True):
    """
    
    Parameters
    ----------
    xvals : vector of floats for scatter plot (x)
    yvals : vector of floats for scatter plot (y)
    cvals : vector of floats for scatter plot (z)
    labels : list of labels for plot axes
    titlestring : string with title for plot
    figname : string with name of figure for saving
    cod_range : list with min and max for axes
    plot_style : string, name of plot style
    title : boolean, whether to add title to plots. The default is True
    logscale : boolean, whether to use log scale in plots. The default is True.

    Returns
    -------
    figure handle

    """
        
    plt.ioff()
    #plt.close('all')
    plt.style.use(plot_style)

    fig, ax = plt.subplots(figsize=(9,8))           
            
    sc = ax.scatter(xvals,yvals,c=cvals,cmap='jet')        

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])

    ax.set_xlim(cod_range)
    ax.set_ylim(cod_range)
    if logscale:
        ax.set_yscale('log')
        ax.set_xscale('log')
        for axis in [ax.xaxis, ax.yaxis]:
            axis.set_major_formatter(ScalarFormatter())  
        
    ax.plot([0, 1], [0, 1], ls = '--', transform=ax.transAxes,c='r')
    
    cb = plt.colorbar(sc)
    cb.set_label(labels[2])
    
    if title: ax.set_title(titlestring)
    fig.tight_layout()
    plt.savefig(figname)   
    plt.close(fig)
    
    return fig

def scatter_plot_grid_3(plot_vals_dict,sources,cod_range,plot_style,
                        title=True,logscale=True):
    """
    
    Plot three scatter plots in a grid

    Parameters
    ----------
    plot_vals_dict : dictionary with plot values
    sources : list of sources    
    cod_range : list for min and max of plot
    plot_style : string, plot style
    title : boolean, whether to put title above plot    
    logscale : boolean, whether the plot with logscale. The default is True.    

    Returns
    -------
    None.

    """
    
    plt.ioff()
    plt.close('all')
    plt.style.use(plot_style)
    
    fig, axs = plt.subplots(2, 2,sharey='row',sharex='col',figsize=(10,9))            
        
    #Upper left plot        
    n_ax = 0  
    
    figstring = plot_vals_dict["figstringprefix"]
    titlestring = plot_vals_dict["titlestring"]
    for i, source in enumerate(sources):
        figstring += plot_vals_dict["sources"][source]["figstring"]        
        if i < len(sources) - 1:
            titlestring += f"{source} vs. "
        else:
            titlestring += f"{source}"
    
    sc1 = axs.flat[n_ax].scatter(plot_vals_dict["sources"][sources[0]]["data"],
                                      plot_vals_dict["sources"][sources[1]]["data"],
                                      c=plot_vals_dict["sources"][sources[0]]["colourdata"],
                                      cmap="jet")
    #axs.flat[n_ax].set_xlabel(plot_vals_dict["sources"][sources[0]]["label"])
    axs.flat[n_ax].set_ylabel(plot_vals_dict["sources"][sources[1]]["label"])    
    axs.flat[n_ax].set_xticklabels('')    
    
    # l, b, w, h = axs.flat[n_ax].get_position().bounds
    # axs.flat[n_ax].set_position([-3.,b,w,h])
        
    #Delete upper right plot
    n_ax += 1
    axs.flat[n_ax].set_visible(False)
    
    #Lower left plot
    n_ax += 1
    
    sc2 = axs.flat[n_ax].scatter(plot_vals_dict["sources"][sources[0]]["data"],
                                      plot_vals_dict["sources"][sources[2]]["data"],
                                      c=plot_vals_dict["sources"][sources[0]]["colourdata"],
                                      cmap="jet")
    axs.flat[n_ax].set_xlabel(plot_vals_dict["sources"][sources[0]]["label"])
    axs.flat[n_ax].set_ylabel(plot_vals_dict["sources"][sources[2]]["label"])
    
    # l, b, w, h = axs.flat[n_ax].get_position().bounds
    # axs.flat[n_ax].set_position([-3,b,w,h])
    
    #Lower right plot
    n_ax += 1
    
    sc3 = axs.flat[n_ax].scatter(plot_vals_dict["sources"][sources[1]]["data"],
                                      plot_vals_dict["sources"][sources[2]]["data"],
                                      c=plot_vals_dict["sources"][sources[0]]["colourdata"],
                                      cmap="jet")
    axs.flat[n_ax].set_xlabel(plot_vals_dict["sources"][sources[1]]["label"])
    #axs.flat[n_ax].set_ylabel(plot_vals_dict["sources"][sources[2]]["label"])    
                
    axs.flat[n_ax].set_yticklabels('')
    
    # cb3 = plt.colorbar(sc3)
    # cb3.set_label(plot_vals_dict["sources"][sources[2]]["colourlabel"])
        
    for ax in axs.flat:
        ax.set_xlim(cod_range)
        ax.set_ylim(cod_range)    
        ax.set(adjustable='box', aspect='equal')
        if logscale:
            ax.set_yscale('log')
            ax.set_xscale('log')
            for axis in [ax.xaxis, ax.yaxis]:
                axis.set_major_formatter(ScalarFormatter())            
        ax.plot([0, 1], [0, 1], ls = '--', transform=ax.transAxes,c='r')
            
    # fig.add_subplot(111, frameon=False)
    # plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    # plt.grid(False)

    cb = plt.colorbar(sc3,cax = fig.add_axes([0.89, 0.11, 0.02, 0.81]),pad=0.5)
    cb.set_label(plot_vals_dict["sources"][sources[0]]["colourlabel"])
    
    fig.subplots_adjust(hspace=0.04,wspace=-0.06)
    fig.subplots_adjust(top=0.93)   
    
    if title: fig.suptitle(titlestring)   
    figstring += plot_vals_dict["figstringsuffix"]
    
    plt.savefig(figstring,bbox_inches = 'tight')   
    plt.close(fig)
                
def scatter_plot_cod_comparison_grid(name,df_compare,rt_config,pyr_substats,
                             pv_substats,info,styles,pv_path,pyr_path,
                             avg_window,flags,day_type):
    """
    

    Parameters
    ----------
    name : string, name of station
    df_compare : dataframe with all results 
    rt_config : dictionary with radiative transfer configuration
    pyr_substats : dictionary with pyranometer substations
    pv_substats : dictionary with pv substations
    info : string, description of current campaign    
    styles : dictionary with plot styles
    pv_path : string with path for PV plots
    pyr_path : string with path for pyranometer plots
    avg_window : string with width of window for moving averages
    flags : dictionary of booleans for plotting
    day_type : string describing cloud conditions on specific day

    Returns
    -------
    None.

    """        
    
    year = info.split('_')[1]
    
    #Generate scatter plots
    cod_range = rt_config["clouds"]["lut_config"]["range"]
    cod_range[1] = cod_range[1] + 5
    
    if str_window_avg == "60min":
        name_cosmo = "COD_tot_600_avg"
    else:
        name_cosmo = f"COD_tot_600_{str_window_avg}_avg"
    
    #Paths for PV plots
    res_dirs = list_dirs(pv_path)
    pv_path = os.path.join(pv_path,'COD_Plots')
    if 'COD_Plots' not in res_dirs:
        os.mkdir(pv_path)   
    
    stat_dirs = list_dirs(pv_path)
    pv_path = os.path.join(pv_path,name)
    if name not in stat_dirs:
        os.mkdir(pv_path)
        
    res_dirs = list_dirs(pv_path)
    pv_path = os.path.join(pv_path,'Scatter')        
    if 'Scatter' not in res_dirs:
        os.mkdir(pv_path)
        
    res_dirs = list_dirs(pv_path)
    pv_path = os.path.join(pv_path,day_type)
    if day_type not in res_dirs:
        os.mkdir(pv_path)
    
    res_dirs = list_dirs(pv_path)
    pv_path = os.path.join(pv_path,str_window_avg)
    if str_window_avg not in res_dirs:
        os.mkdir(pv_path)
        
    res_dirs = list_dirs(pyr_path)
    pyr_path = os.path.join(pyr_path,'COD_Plots')
    if 'COD_Plots' not in res_dirs:
        os.mkdir(pyr_path)   
        
    stat_dirs = list_dirs(pyr_path)
    pyr_path = os.path.join(pyr_path,name)
    if name not in stat_dirs:
        os.mkdir(pyr_path)
        
    res_dirs = list_dirs(pyr_path)
    pyr_path = os.path.join(pyr_path,'Scatter')        
    if 'Scatter' not in res_dirs:
        os.mkdir(pyr_path)
        
    res_dirs = list_dirs(pyr_path)
    pyr_path = os.path.join(pyr_path,day_type)
    if day_type not in res_dirs:
        os.mkdir(pyr_path)
    
    res_dirs = list_dirs(pyr_path)
    pyr_path = os.path.join(pyr_path,str_window_avg)
    if str_window_avg not in res_dirs:
        os.mkdir(pyr_path)

    #Generate plots by looping through substations
    print(f"Plotting pyranometer scatter plots for {str_window_avg}")        
    for substat in pyr_substats:
        radtypes = []
        if '_' in substat:
            substat_label = f'{substat.split("_")[0]}_{{{substat.split("_")[1]}}}'
        else: substat_label = substat
        
        colnames = df_compare.xs(substat,level="substat",axis=1).columns
        if f"COD_550_poa_inv_{avg_window}_avg" in colnames:
            radtypes.append("poa")
            
            #Get angle parameters 
            if "opt_pars" in pyr_substats[substat]:
                pyrtilt = np.rad2deg(pyr_substats[substat]["opt_pars"][0][1])
                pyrazimuth = np.rad2deg(azi_shift(pyr_substats[substat]["opt_pars"][1][1]))
            else:
                pyrtilt = np.rad2deg(pyr_substats[substat]["ap_pars"][0][1])
                pyrazimuth = np.rad2deg(azi_shift(pyr_substats[substat]["ap_pars"][1][1]))
                
        if f"COD_550_down_inv_{avg_window}_avg" in colnames:
            radtypes.append("down")
        
        #Plot poa vs. down for pyranometers
        if "poa" in radtypes and "down" in radtypes:
            
            labels = [r"COD 550nm ($G^\downarrow_\mathrm{{{}}}$)".format(substat),
                      r"COD 550nm ($G^\angle_\mathrm{{{}}}, \theta={:.1f}^\circ,\phi={:.1f}^\circ$)"
                      .format(substat_label,pyrtilt,pyrazimuth),
                      r"$<cf>_\mathrm{{{}}} (G^\downarrow_\mathrm{{{}}}$)".format(avg_window,substat_label)]
            
            titlestring = f'COD, {substat} tilted vs. downward irradiance: {name}, {year}, {day_type}'
            figstring = os.path.join(pyr_path,f"COD_poa_vs_downward_{substat}_{name}_{year}_{day_type}.png")
            
            scatter_plot_cod(df_compare[(f"COD_550_down_inv_{avg_window}_avg",substat)],
                            df_compare[(f"COD_550_poa_inv_{avg_window}_avg",substat)],
                            df_compare[(f"cf_down_{avg_window}_avg",substat)],
                            labels, titlestring, figstring, cod_range, 
                            styles["single_small"],flags["titles"])

        plot_dict = {}    
        
        #Compare pyranometer to SEVIRI, APOLLO, COSMO, PV
        for radtype in radtypes:
            if radtype == "poa":
                radlabel = "\\angle"
                anglelabelsmall = "^{{\\theta={:.1f}^\circ}}_{{\phi={:.1f}^\circ}}".format(pyrtilt,pyrazimuth)
                anglelabellarge = ", \\theta={:.1f}^\circ,\phi={:.1f}^\circ".format(pyrtilt,pyrazimuth)
                titlelabel = "tilted"
            elif radtype == "down":
                radlabel = "\\downarrow"
                titlelabel = "downward"
                anglelabelsmall = ""
                anglelabellarge = ""
        
            #Add relevant info to plot dictionary for grid plots
            plot_dict.update({"sources":{}})
            plot_dict["sources"].update({substat:{}})
            plot_dict["sources"][substat].update({"data":
                          df_compare[(f"COD_550_{radtype}_inv_{avg_window}_avg",substat)]})           
            plot_dict["sources"][substat].update({"label":
                          r"COD 550nm (${{G^{}_\mathrm{{{}}}}}\ {}$)".format(radlabel,substat_label,anglelabelsmall)})
            plot_dict["sources"][substat].update({"colourdata":
                          df_compare[(f"cf_{radtype}_{avg_window}_avg",substat)]})           
            plot_dict["sources"][substat].update({"colourlabel":
                          r"$<cf>_\mathrm{{{}}} (G^{}_\mathrm{{{}}}$)".format(avg_window,radlabel,substat_label)})
            plot_dict["sources"][substat].update({"figstring":f"{radtype}_{substat}_"})
                
            plot_dict.update({"titlestring":f"COD at {name}, {year}, {day_type}: "})            
            plot_dict.update({"figstringprefix":os.path.join(pyr_path,"COD_scatter_grid_")})
            plot_dict.update({"figstringsuffix":f"{name}_{year}_{day_type}.png"})
        
            #SEVIRI
            labels = [r"COD 550nm ($G^{}_\mathrm{{{}}}{}$)".format(radlabel,substat_label,anglelabellarge),
                      r"COD 500nm (SEVIRI-HRV)",
                      r"$<cf>_\mathrm{{{}}} (G^\angle_\mathrm{{{}}}$)".format(avg_window,substat_label)]
            
            titlestring = f'COD, {substat} {titlelabel} vs. SEVIRI-HRV: {name}, {year}, {day_type}'
            figstring = os.path.join(pyr_path,f"COD_{radtype}_{substat}_vs_seviri_{name}_{year}_{day_type}.png")
            
            scatter_plot_cod(df_compare[(f"COD_550_{radtype}_inv_{avg_window}_avg",substat)],
                            df_compare[(f"COD_500_{avg_window}_avg","seviri")],
                            df_compare[(f"cf_{radtype}_{avg_window}_avg",substat)],
                            labels, titlestring, figstring, cod_range, 
                            styles["single_small"],flags["titles"])
            
            plot_dict["sources"].update({"SEVIRI-HRV":{}})
            plot_dict["sources"]["SEVIRI-HRV"].update({"data":df_compare[(f"COD_500_{avg_window}_avg","seviri")]})
            plot_dict["sources"]["SEVIRI-HRV"].update({"label":r"COD 500nm (SEVIRI-HRV)"})                        
            plot_dict["sources"]["SEVIRI-HRV"].update({"figstring":"seviri_hrv_"})
            
            #APOLLO
            labels = [r"COD 550nm ($G^{}_\mathrm{{{}}}{}$)".format(radlabel,substat_label,anglelabellarge),
                      r"COD 500nm (SEVIRI-APNG_1.1)",
                      r"$<cf>_\mathrm{{{}}} (G^{}_\mathrm{{{}}}$)".format(avg_window,radlabel,substat_label)]
            
            titlestring = f'COD, {substat} {titlelabel} vs. SEVIRI-APNG_1.1: {name}, {year}, {day_type}'
            figstring = os.path.join(pyr_path,f"COD_{radtype}_{substat}_vs_apollo_{name}_{year}_{day_type}.png")
            
            scatter_plot_cod(df_compare[(f"COD_550_{radtype}_inv_{avg_window}_avg",substat)],
                            df_compare[(f"cot_AP_{avg_window}_avg","apollo")],
                            df_compare[(f"cf_{radtype}_{avg_window}_avg",substat)],
                            labels, titlestring, figstring, cod_range, 
                            styles["single_small"],flags["titles"])
            
            plot_dict["sources"].update({"SEVIRI-APNG_1.1":{}})
            plot_dict["sources"]["SEVIRI-APNG_1.1"].update({"data":df_compare[(f"cot_AP_{avg_window}_avg","apollo")]})
            plot_dict["sources"]["SEVIRI-APNG_1.1"].update({"label":r"COD (SEVIRI-APNG_1.1)"})                        
            plot_dict["sources"]["SEVIRI-APNG_1.1"].update({"figstring":"apollo_"})
            
            #COSMO
            labels = [r"COD 550nm ($G^{}_\mathrm{{{}}}{}$)".format(radlabel,substat_label,anglelabellarge),
                      r"COD 600nm (COSMO)",
                      r"$<cf>_\mathrm{{{}}} (G^{}_\mathrm{{{}}}$)".format(avg_window,radlabel,substat_label)]
            
            titlestring = f'COD, {substat} {titlelabel} vs. COSMO: {name}, {year}, {day_type}'
            figstring = os.path.join(pyr_path,f"COD_{radtype}_{substat}_vs_cosmo_{name}_{year}_{day_type}.png")
            
            scatter_plot_cod(df_compare[(f"COD_550_{radtype}_inv_{avg_window}_avg",substat)],
                            df_compare[(name_cosmo,"cosmo")],
                            df_compare[(f"cf_{radtype}_{avg_window}_avg",substat)],
                            labels, titlestring, figstring, cod_range, 
                            styles["single_small"],flags["titles"])
            
            plot_dict["sources"].update({"COSMO":{}})
            plot_dict["sources"]["COSMO"].update({"data":df_compare[(name_cosmo,"cosmo")]})
            plot_dict["sources"]["COSMO"].update({"label":r"COD 600nm (COSMO)"}) 
            plot_dict["sources"]["COSMO"].update({"figstring":"cosmo_"})
            
            scatter_plot_grid_3(plot_dict,[substat,'SEVIRI-HRV','SEVIRI-APNG_1.1'],
                                                cod_range,styles["combo_small"],flags["titles"])
                                                                  
            scatter_plot_grid_3(plot_dict,[substat,'SEVIRI-APNG_1.1','COSMO'],
                                cod_range,styles["combo_small"],flags["titles"])
            
            scatter_plot_grid_3(plot_dict,[substat,'SEVIRI-HRV','COSMO'],
                                cod_range,styles["combo_small"],flags["titles"])
            
            if pv_substats:
                #Compare to PV
                for substat_type in pv_substats:
                    if info in pv_substats[substat_type]["source"]:
                        for substat_pv in pv_substats[substat_type]["data"]:                
                            #Get angle parameters 
                            if "opt_pars" in pv_substats[substat_type]["data"][substat_pv]:
                                pvtilt = np.rad2deg(pv_substats[substat_type]["data"][substat_pv]["opt_pars"][0][1])
                                pvazimuth = np.rad2deg(azi_shift(pv_substats[substat_type]["data"]
                                                                 [substat_pv]["opt_pars"][1][1]))
                            else:
                                pvtilt = np.rad2deg(pv_substats[substat_type]["data"][substat_pv]["ap_pars"][0][1])
                                pvazimuth = np.rad2deg(azi_shift(pv_substats[substat_type]["data"]
                                                                 [substat_pv]["ap_pars"][1][1]))
                            pvanglelabelsmall = "^{{\\theta={:.1f}^\circ}}_{{\phi={:.1f}^\circ}}".format(pvtilt,pvazimuth)
                            pvanglelabellarge = ", \\theta={:.1f}^\circ,\phi={:.1f}^\circ".format(pvtilt,pvazimuth)
                            if "_" in substat_pv:
                                substat_pv_label = f'{substat_pv.split("_")[0]}_{{{substat_pv.split("_")[1]}}}'
                            else: substat_pv_label = substat_pv
                            
                            labels = [r"COD 550nm ($G^{}_\mathrm{{{}}}{}$)".format(radlabel,substat_label,anglelabellarge),                      
                                  r"COD 550nm ($G^\angle_\mathrm{{{}}}{}$)"
                                  .format(substat_pv_label,pvanglelabellarge),
                                  r"$<cf>_\mathrm{{{}}} (G^{}_\mathrm{{{}}}$)".format(avg_window,radlabel,substat_label)]
                        
                            titlestring = f'COD, {substat} {titlelabel} vs. {substat_pv}: {name}, {year}, {day_type}'
                            figstring = os.path.join(pyr_path,f"COD_{radtype}_{substat}_vs_{substat_pv}_{year}_{day_type}.png")
                        
                            scatter_plot_cod(df_compare[(f"COD_550_{radtype}_inv_{avg_window}_avg",substat)],
                                        df_compare[(f"COD_550_poa_inv_{avg_window}_avg",substat_pv)],
                                        df_compare[(f"cf_{radtype}_{avg_window}_avg",substat)],
                                        labels, titlestring, figstring, cod_range, 
                                        styles["single_small"],flags["titles"])
                            
                            plot_dict["sources"].update({substat_pv:{}})
                            plot_dict["sources"][substat_pv].update({"data":df_compare[(f"COD_550_poa_inv_{avg_window}_avg",substat_pv)]})
                            plot_dict["sources"][substat_pv].update({"label":
                                      r"COD 550nm (${{G^\angle_\mathrm{{{}}}}}\ {}$)".format(substat_pv_label,pvanglelabelsmall)})
                            plot_dict["sources"][substat_pv].update({"colourdata":df_compare[(f"cf_poa_{avg_window}_avg",substat_pv)]})         
                            plot_dict["sources"][substat_pv].update({"colourlabel":
                                      r"$<cf>_\mathrm{{{}}} (G^\angle_\mathrm{{{}}}$)".format(avg_window,substat_pv_label)})
                            plot_dict["sources"][substat_pv].update({"figstring":f"{substat_pv}_"})
                                        
                            #Create grids of scatter plots in different combinations
                            scatter_plot_grid_3(plot_dict,[substat,'SEVIRI-HRV',substat_pv],
                                                cod_range,styles["combo_small"],flags["titles"])
                            
                            scatter_plot_grid_3(plot_dict,[substat,'SEVIRI-APNG_1.1',substat_pv],
                                                cod_range,styles["combo_small"],flags["titles"])
                            
                            scatter_plot_grid_3(plot_dict,[substat,'COSMO',substat_pv],
                                                cod_range,styles["combo_small"],flags["titles"])                                                        
                            
                            # scatter_plot_grid_3(plot_dict,['SEVIRI-APNG_1.1','SEVIRI-HRV','COSMO'],
                            #                     cod_range,styles["combo_small"],flags["titles"])
                        
    if pv_substats:
        #Same for PV            
        print("Plotting PV scatter plots")
        plot_dict = {}    
        plot_dict.update({"sources":{}})
        for substat_type in pv_substats:
            if info in pv_substat_pars[substat_type]["source"]:
                for substat_pv in pv_substats[substat_type]["data"]:                            
                    #Get angle parameters 
                    if "opt_pars" in pv_substats[substat_type]["data"][substat_pv]:
                        pvtilt = np.rad2deg(pv_substats[substat_type]["data"][substat_pv]["opt_pars"][0][1])
                        pvazimuth = np.rad2deg(azi_shift(pv_substats[substat_type]["data"]
                                                         [substat_pv]["opt_pars"][1][1]))
                    else:
                        pvtilt = np.rad2deg(pv_substats[substat_type]["data"]
                                            [substat_pv]["ap_pars"][0][1])
                        pvazimuth = np.rad2deg(azi_shift(pv_substats[substat_type]["data"]
                                                         [substat_pv]["ap_pars"][1][1]))
                    pvanglelabelsmall = "^{{\\theta={:.1f}^\circ}}_{{\phi={:.1f}^\circ}}".format(pvtilt,pvazimuth)
                    pvanglelabellarge = ", \\theta={:.1f}^\circ,\phi={:.1f}^\circ".format(pvtilt,pvazimuth)
                    if '_' in substat_pv:
                        substat_pv_label = f'{substat_pv.split("_")[0]}_{{{substat_pv.split("_")[1]}}}'
                    else: substat_pv_label = substat_pv
                    
                    titlelabel = "tilted"
                    
                    plot_dict["sources"].update({substat_pv:{}})
                    plot_dict["sources"][substat_pv].update({"data":df_compare[(f"COD_550_poa_inv_{avg_window}_avg",substat_pv)]})
                    plot_dict["sources"][substat_pv].update({"label":
                              r"COD 550nm (${{G^\angle_\mathrm{{{}}}}}\ {}$)".format(substat_pv_label,pvanglelabelsmall)})
                    plot_dict["sources"][substat_pv].update({"colourdata":df_compare[(f"cf_poa_{avg_window}_avg",substat_pv)]})         
                    plot_dict["sources"][substat_pv].update({"colourlabel":
                              r"$<cf>_\mathrm{{{}}} (G^\angle_\mathrm{{{}}}$)".format(avg_window,substat_pv_label)})
                    plot_dict["sources"][substat_pv].update({"figstring":f"{substat_pv}_"})
                    
                    plot_dict.update({"titlestring":f"COD retrieval at {name}, {year}, {day_type}: "})            
                    plot_dict.update({"figstringprefix":os.path.join(pv_path,"COD_scatter_grid_")})
                    plot_dict.update({"figstringsuffix":f"{name}_{year}_{day_type}.png"})
                    
                    #SEVIRI
                    labels = [r"COD 550nm ($G^\angle_\mathrm{{{}}}{}$)".format(substat_pv_label,pvanglelabellarge),
                              r"COD 500nm (SEVIRI-HRV)",
                              r"$<cf>_\mathrm{{{}}} (G^\angle_\mathrm{{{}}}$)".format(avg_window,substat_pv_label)]
                    
                    titlestring = f'COD, {substat_pv} {titlelabel} vs. SEVIRI-HRV: {name}, {year}, {day_type}'
                    figstring = os.path.join(pv_path,f"COD_poa_{substat_pv}_vs_seviri_{name}_{year}_{day_type}.png")
                    
                    scatter_plot_cod(df_compare[(f"COD_550_poa_inv_{avg_window}_avg",substat_pv)],
                                    df_compare[(f"COD_500_{avg_window}_avg","seviri")],
                                    df_compare[(f"cf_poa_{avg_window}_avg",substat_pv)],
                                    labels, titlestring, figstring, cod_range, 
                                    styles["single_small"],flags["titles"])
                    
                    plot_dict["sources"].update({"SEVIRI-HRV":{}})
                    plot_dict["sources"]["SEVIRI-HRV"].update({"data":df_compare[(f"COD_500_{avg_window}_avg","seviri")]})
                    plot_dict["sources"]["SEVIRI-HRV"].update({"label":r"COD 500nm (MSG - SEVIRI)"})                        
                    plot_dict["sources"]["SEVIRI-HRV"].update({"figstring":"seviri_hrv_"})
                    
                    #APOLLO
                    labels = [r"COD 550nm ($G^\angle_\mathrm{{{}}}{}$)".format(substat_pv_label,pvanglelabellarge),
                              r"COD (SEVIRI-APNG_1.1)",
                              r"$<cf>_\mathrm{{{}}} (G^\angle_\mathrm{{{}}}$)".format(avg_window,substat_pv)]
                    
                    titlestring = f'COD, {substat_pv} {titlelabel} vs. SEVIRI-APNG_1.1: {name}, {year}, {day_type}'
                    figstring = os.path.join(pv_path,f"COD_poa_{substat_pv}_vs_apollo_{name}_{year}_{day_type}.png")
                    
                    scatter_plot_cod(df_compare[(f"COD_550_poa_inv_{avg_window}_avg",substat_pv)],
                                    df_compare[(f"cot_AP_{avg_window}_avg","apollo")],
                                    df_compare[(f"cf_poa_{avg_window}_avg",substat_pv)],
                                    labels, titlestring, figstring, cod_range, 
                                    styles["single_small"],flags["titles"])
                    
                    plot_dict["sources"].update({"SEVIRI-APNG_1.1":{}})
                    plot_dict["sources"]["SEVIRI-APNG_1.1"].update({"data":df_compare[(f"cot_AP_{avg_window}_avg","apollo")]})
                    plot_dict["sources"]["SEVIRI-APNG_1.1"].update({"label":r"COD (SEVIRI-APNG_1.1)"})                        
                    plot_dict["sources"]["SEVIRI-APNG_1.1"].update({"figstring":"apollo_"})
                    
                    #COSMO
                    labels = [r"COD 550nm ($G^\angle_\mathrm{{{}}}{}$)".format(substat_pv_label,pvanglelabellarge),
                              r"COD 600nm (COSMO)",
                              r"$<cf>_\mathrm{{{}}} (G^\angle_\mathrm{{{}}}$)".format(avg_window,substat_pv_label)]
                    
                    titlestring = f'COD, {substat_pv} {titlelabel} vs. COSMO: {name}, {year}, {day_type}'
                    figstring = os.path.join(pv_path,f"COD_poa_{substat_pv}_vs_cosmo_{name}_{year}_{day_type}.png")
                    
                    scatter_plot_cod(df_compare[(f"COD_550_poa_inv_{avg_window}_avg",substat_pv)],
                                    df_compare[(name_cosmo,"cosmo")],
                                    df_compare[(f"cf_poa_{avg_window}_avg",substat_pv)],
                                    labels, titlestring, figstring, cod_range, 
                                    styles["single_small"],flags["titles"])
                
                     
                    plot_dict["sources"].update({"COSMO":{}})
                    plot_dict["sources"]["COSMO"].update({"data":df_compare[(name_cosmo,"cosmo")]})
                    plot_dict["sources"]["COSMO"].update({"label":r"COD 600nm (COSMO)"}) 
                    plot_dict["sources"]["COSMO"].update({"figstring":"cosmo_"})
                        
                    scatter_plot_grid_3(plot_dict,[substat_pv,'SEVIRI-HRV','COSMO'],
                                        cod_range,styles["combo_small"],flags["titles"])    
                    
                    scatter_plot_grid_3(plot_dict,[substat_pv,'SEVIRI-HRV',"SEVIRI-APNG_1.1"],
                                        cod_range,styles["combo_small"],flags["titles"])    
                    
                    scatter_plot_grid_3(plot_dict,[substat_pv,'SEVIRI-APNG_1.1','COSMO'],
                                        cod_range,styles["combo_small"],flags["titles"])    
                                
    return 

def variance_class(input_series,num_classes):
    """
    Split up an input series into variance classes

    Parameters
    ----------
    input_series : series, input data
    num_classes : integer, number of classes

    Returns
    -------
    var_class_series : series with variance classes
    class_limits : list with class limits

    """
    
    max_std = input_series.max()
    min_std = input_series.min()
        
    bin_size = (max_std - min_std)/num_classes
    
    class_limits = [min_std + i*bin_size for i in range(num_classes)]
        
    var_class_series = pd.Series(dtype=int,index=input_series.index)    
    for i in range(num_classes):
        if i < num_classes - 1: 
            mask = (input_series >= min_std + i*bin_size) &\
            (input_series < min_std + (i+1)*bin_size)            
        else:
            mask = (input_series >= min_std + i*bin_size) &\
            (input_series <= min_std + (i+1)*bin_size)            
        
        var_class_series.loc[mask] = i        
                            
    return var_class_series, class_limits

def box_plots_variance(name,dataframe,year,names,str_window_avg,num_classes,
                       var_class_dict,title,styles,plotpath):
    """
    
    Parameters
    ----------
    name : string, name of PV station
    dataframe : dataframe with relevant data
    year : string, year under consideration
    names : list of tuples with plot names 
    str_window_avg : string, width of averaging window
    num_classes : integer, number of classes
    var_class_dict : dictionary with variance classes
    title : string, plot title
    styles : dictionary with plot styles
    plotpath : string with plot path

    Returns
    -------
    None.    

    """

    plt.ioff()
    plt.style.use(styles["combo_small"])
    
    res_dirs = list_dirs(plotpath)
    plotpath = os.path.join(plotpath,'COD_Plots')
    if 'COD_Plots' not in res_dirs:
        os.mkdir(plotpath)    
        
    stat_dirs = list_dirs(plotpath)
    plotpath = os.path.join(plotpath,name.replace(' ','_'))
    if name.replace(' ','_') not in stat_dirs:
        os.mkdir(plotpath)
    
    res_dirs = list_dirs(plotpath)
    plotpath = os.path.join(plotpath,'Stats')        
    if 'Stats' not in res_dirs:
        os.mkdir(plotpath)
    
    figstring = f'COD_box_grid_{str_window_avg}_avg_{name.replace(" ","_")}_{year}.png'
    
    if num_classes == 2:
        class_labels = ["Low","High"]
    elif num_classes == 3:
        class_labels = ["Low","Medium","High"]
    elif num_classes == 4:
        class_labels = ["Low","Low-Medium","High-Medium","High"]
    
    if num_classes <= 3:
        fig, axs = plt.subplots(num_classes,1,sharey='all',figsize=(10,9))            
    elif num_classes == 4:
        fig, axs = plt.subplots(2, 2,sharey='all',figsize=(10,9))            
                                
    for i in range(num_classes):
        plot_list = []
        label_list = []
        class_list = []
        for avg, var, substat, label in names: 
            if np.isnan(dataframe[(var,substat)]).all():
                plot_list.append(dataframe.loc[:,
                       (avg,substat)])
            else:
                plot_list.append(dataframe.loc[dataframe[(var,substat)]==i,
                       (avg,substat)])
            label_list.append(label)
            class_list.append(var_class_dict[label][i:i+2])
        
        axs.flat[i].boxplot(plot_list)
        
        axs.flat[i].set_ylabel('COD')
        axs.flat[i].set_title(f"{class_labels[i]} variance",fontsize=14)
        axs.flat[i].set_xticklabels([])
        #axs.flat[i].set_yscale('log')                
        #axs.flat[i].set_ylim((-1, 10))
        # axs.flat[i].yaxis.set_ticks([0,5,10])        
        # axs.flat[i].set_yticklabels([0,5,''])        
        
        # divider = make_axes_locatable(axs.flat[i])
        # logAxis = divider.append_axes("top", size=1, pad=0.) #, sharex=axs.flat[i])
        # logAxis.boxplot(plot_list)
        # logAxis.set_yscale('log')
        # logAxis.set_ylim((10, 150));
        # logAxis.set_xticklabels([])
        # logAxis.set_title(f"{class_labels[i]} variance",fontsize=14)
        # if i==1: logAxis.set_ylabel('COD',position=[0,0])
    
    if len(label_list) > 5:                    
        axs.flat[i].set_xticklabels(label_list,rotation=45,fontsize=14)
    else:
        axs.flat[i].set_xticklabels(label_list)
    
    if title:
        fig.suptitle(f'{str_window_avg} average COD comparison at {name}, {year}')
        fig.subplots_adjust(top=0.93)   
        
    fig.tight_layout()
    plt.savefig(os.path.join(plotpath,figstring))   
    plt.close(fig)    

def cod_histograms(name,dataframe,year,names,str_window_avg,num_classes,
                       var_class_dict,title,styles,plotpath):
    """
    

    Parameters
    ----------
    name : string, name of PV station
    dataframe : dataframe with relevant data
    year : string with year under consideration    
    names : list of tuples with plot names 
    str_window_avg : string, width of averaging window
    num_classes : integer, number of classes
    var_class_dict : dictionary with variance classes
    title : string, plot title
    styles : dictionary with plot styles
    plotpath : string with plot path

    Returns
    -------
    None.

    """
    
    plt.ioff()
    plt.style.use(styles["combo_small"])
    
    res_dirs = list_dirs(plotpath)
    plotpath = os.path.join(plotpath,'COD_Plots')
    if 'COD_Plots' not in res_dirs:
        os.mkdir(plotpath)    
        
    stat_dirs = list_dirs(plotpath)
    plotpath = os.path.join(plotpath,name.replace(' ','_'))
    if name.replace(' ','_') not in stat_dirs:
        os.mkdir(plotpath)
    
    res_dirs = list_dirs(plotpath)
    plotpath = os.path.join(plotpath,'Stats')        
    if 'Stats' not in res_dirs:
        os.mkdir(plotpath)
    
    figstring = f'COD_histogram_grid_{str_window_avg}_avg_{name.replace(" ","_")}_{year}.png'
    
    if str_window_avg == "60min":
        name_cf = "cf_tot_avg"
    else:
        name_cf = f"cf_tot_{str_window_avg}_avg"
    
    if num_classes == 2:
        class_labels = ["Low","High"]
    elif num_classes == 3:
        class_labels = ["Low","Medium","High"]
    elif num_classes == 4:
        class_labels = ["Low","Low-Medium","High-Medium","High"]
        
    if len(names) <= 3:
        fig, axs = plt.subplots(len(names),1,sharey='all',figsize=(10,9))            
    elif len(names) == 4:
        fig, axs = plt.subplots(2, 2,sharey='all',sharex='all',figsize=(10,9))     
    elif len(names) <= 6:
        fig, axs = plt.subplots(3, 2,sharey='all',sharex='all',figsize=(10,9))     
    elif len(names) <= 8:
        fig, axs = plt.subplots(4, 2,sharey='all',sharex='all',figsize=(10,9))     
    elif len(names) == 9:
        fig, axs = plt.subplots(3, 3,sharey='all',sharex='all',figsize=(10,9))     
    elif len(names) <= 12:
        fig, axs = plt.subplots(4, 3,sharey='all',sharex='all',figsize=(10,9))
    elif len(names) <= 16:
        fig, axs = plt.subplots(4, 4,sharey='all',sharex='all',figsize=(10,9))    
    elif len(names) <= 20:
        fig, axs = plt.subplots(5, 4,sharey='all',sharex='all',figsize=(10,9))
    else:
        print('Too many plots to fit in')
        
    bins = np.logspace(np.log10(0.5),np.log10(150),20)
    
    for i, (avg, var, substat, label) in enumerate(names):                     
        if substat == "cosmo":
            axs.flat[i].hist(dataframe[(avg,substat)],bins=bins,
                             weights=dataframe[(name_cf,substat)])
        else:
            axs.flat[i].hist(dataframe[(avg,substat)],bins=bins)
        axs.flat[i].set_xscale('log')
        axs.flat[i].set_title(label)
                
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel("COD")
    plt.ylabel("Frequency")    
        
    if title:
        fig.suptitle(f'{str_window_avg} average COD comparison at {name}, {year}')
        fig.subplots_adjust(top=0.93)   
        
    fig.tight_layout()
    plt.savefig(os.path.join(plotpath,figstring),bbox_inches = 'tight')   
    plt.close(fig)    
        

def cod_stats_plots(name,dataframe,pyr_substats,pv_substats,year,
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
    dictionary with variance classes

    """  
    variance_class_dict = {}
    
    if str_window_avg == "60min":
        name_cosmo_avg = "COD_tot_600_avg"
        name_cosmo_std = "COD_tot_600_iqr"
    else:
        name_cosmo_avg = f"COD_tot_600_{str_window_avg}_avg"
        name_cosmo_std = f"COD_tot_600_{str_window_avg}_std"
    
    plot_names = [(f"cot_AP_{str_window_avg}_avg",f"cot_AP_{str_window_avg}_varclass",
                   "apollo","SEVIRI-APNG_1.1"),                  
                  (name_cosmo_avg,"COD_tot_600_varclass",
                    "cosmo","COSMO")]
    
    if year.split('_')[1] != "2018": 
        df_stats_index = dataframe.drop("seviri",level=1,axis=1).dropna(how='any',axis=0).index
    else:
        df_stats_index = dataframe.dropna(how='any',axis=0).index

    #Calculate variance classes            
    var_class, limits = variance_class(dataframe.loc[df_stats_index,(f"cot_AP_{str_window_avg}_std","apollo")],
                       num_classes)
    dataframe.loc[df_stats_index,(f"cot_AP_{str_window_avg}_varclass","apollo")] = var_class
    variance_class_dict.update({"SEVIRI-APNG_1.1":limits})
        
    if year.split('_')[1] == "2018":
        var_class, limits =\
            variance_class(dataframe.loc[df_stats_index,(f"COD_500_{str_window_avg}_std","seviri")],
                            num_classes)     
        dataframe.loc[df_stats_index,(f"COD_500_{str_window_avg}_varclass","seviri")] = var_class
        variance_class_dict.update({"SEVIRI-HRV":limits})
        
        plot_names.append((f"COD_500_{str_window_avg}_avg",f"COD_500_{str_window_avg}_varclass",
                   "seviri","SEVIRI-HRV"))
        
    var_class, limits = variance_class(dataframe.loc[df_stats_index,(name_cosmo_std,"cosmo")],num_classes)    
    dataframe.loc[df_stats_index,("COD_tot_600_varclass","cosmo")] = var_class     
    variance_class_dict.update({"COSMO":limits})                
    
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

        #Calculate variance classes for pyranometers                
        for radtype in radtypes:
            if (f"COD_550_{radtype}_inv_{str_window_avg}_avg",substat) in dataframe.columns:
                var_class, limits = variance_class(dataframe.loc[df_stats_index,(f"COD_550_{radtype}_inv_{str_window_avg}_std",
                                 substat)],num_classes)
                dataframe.loc[df_stats_index,(f"COD_550_{radtype}_inv_{str_window_avg}_varclass",substat)] =\
                    var_class
                variance_class_dict.update({f"{substat}, {radtype}":limits})
                    
                plot_names.append((f"COD_550_{radtype}_inv_{str_window_avg}_avg",
                                   f"COD_550_{radtype}_inv_{str_window_avg}_varclass",
                                   substat,f"{substat}, {radtype}"))
                
    for substat_type in pv_substats:
        for substat in pv_substats[substat_type]["data"]:
            if year in pv_substats[substat_type]["source"]:  
                #Get radnames
                radtype = "poa"
                
                #Calculate variance classes for PV stations
                var_class, limits = variance_class(dataframe.loc[df_stats_index,(f"COD_550_{radtype}_inv_{str_window_avg}_std",
                                 substat)],num_classes)
                dataframe.loc[df_stats_index,(f"COD_550_{radtype}_inv_{str_window_avg}_varclass",substat)] =\
                    var_class
                variance_class_dict.update({f"{substat}":limits})                
                    
                plot_names.append((f"COD_550_{radtype}_inv_{str_window_avg}_avg",
                                   f"COD_550_{radtype}_inv_{str_window_avg}_varclass",
                                   substat,substat))            
                              
    dataframe.sort_index(axis=1,level=1,inplace=True)        
    
    if flags["stats"]:
        print(f'Plotting box-whisker plots for {name}, {year}')
        df_plot = dataframe.loc[df_stats_index]
        box_plots_variance(name,df_plot,year.split('_')[1],plot_names,str_window_avg,num_classes,
                           variance_class_dict,flags["titles"],styles,plotpath)
        
        print(f'Plotting histograms for {name}, {year}')
        cod_histograms(name,df_plot,year.split('_')[1],plot_names,str_window_avg,num_classes,
                           variance_class_dict,flags["titles"],styles,plotpath)
        
    if year.split('_')[1] != "2018":
        df_result = dataframe.drop("seviri",level=1,axis=1).loc[df_stats_index]
    else:
        df_result = dataframe.loc[df_stats_index]
    
    return df_result, variance_class_dict    

def combined_stats_plots(dataframe,pyr_substats,pv_substats,year,
                        str_window_avg,num_classes,
                        styles,flags,plotpath):
    """
    

    Parameters
    ----------
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
    dataframe with mean values
    dictionary with variance classes

    """
    
    variance_class_dict = {}
    
    cod_names = [('COD_tot_600_avg',"cosmo"),('COD_tot_600_iqr',"cosmo"),('cf_tot_avg',"cosmo"),
                 (f'cot_AP_{str_window_avg}_avg',"apollo"),
                 (f'cot_AP_{str_window_avg}_std',"apollo")]
    
    plot_names = [(f"cot_AP_{str_window_avg}_avg",f"cot_AP_{str_window_avg}_varclass",
                   "apollo","SEVIRI-APNG_1.1"),                  
                  ("COD_tot_600_avg","COD_tot_600_varclass",
                    "cosmo","COSMO")]
    
    if year.split('_')[1] == "2018":
        cod_names.extend([(f"COD_500_{str_window_avg}_avg","seviri"),
                          (f"COD_500_{str_window_avg}_std","seviri")])
        plot_names.append((f"COD_500_{str_window_avg}_avg",f"COD_500_{str_window_avg}_varclass",
                    "seviri","SEVIRI-HRV"))
    
    #Combine Pyranometers and PV stations
    for col in dataframe.columns:
        if "Pyr" in col[1]:
            dataframe.rename(columns={col[1]:"Pyr"},level='substat',inplace=True)
            cod_names.append((col[0],"Pyr"))
        if "egrid" in col[1]:
            dataframe.rename(columns={col[1]:"PV_1min"},level='substat',inplace=True)
            cod_names.append((col[0],"PV_1min"))
        if "auew" in col[1]:
            dataframe.rename(columns={col[1]:"PV_15min"},level='substat',inplace=True)
            cod_names.append((col[0],"PV_15min"))
       
    new_names = []
    [new_names.append(x) for x in cod_names if x not in new_names]
    cod_names = [x for x in new_names if "varclass" not in x[0]]# and "cf" not in x[0]]

    #Calculate means        
    df_mean = pd.concat([dataframe.loc[:,pd.IndexSlice[cod_name[0],cod_name[1],:]].mean(axis=1)
                for cod_name in cod_names],axis=1,keys=cod_names,names=['variable','substat'])        

    df_mean.dropna(how='any',axis=0,inplace=True)
    
    #Calculate variance classes            
    var_class, limits = variance_class(df_mean[(f"cot_AP_{str_window_avg}_std","apollo")],
                       num_classes)
    df_mean[(f"cot_AP_{str_window_avg}_varclass","apollo")] = var_class
    variance_class_dict.update({"SEVIRI-APNG_1.1":limits})
        
    if year.split('_')[1] == "2018":    
         var_class, limits =\
            variance_class(df_mean[(f"COD_500_{str_window_avg}_std","seviri")],num_classes)     
         df_mean[(f"COD_500_{str_window_avg}_varclass","seviri")] = var_class
         variance_class_dict.update({"SEVIRI-HRV":limits})
    
    var_class, limits = variance_class(df_mean[("COD_tot_600_iqr","cosmo")],num_classes)
    df_mean[("COD_tot_600_varclass","cosmo")] = var_class     
    variance_class_dict.update({"COSMO":limits})            
        
    radtypes = ["poa","down"]        
    
    #Calculate variance classes for pyranometers                
    for radtype in radtypes:
        if f"COD_550_{radtype}_inv_{str_window_avg}_avg" in df_mean.columns.levels[0]:
            var_class, limits = variance_class(df_mean[(f"COD_550_{radtype}_inv_{str_window_avg}_std","Pyr")],
                                               num_classes)
            df_mean[(f"COD_550_{radtype}_inv_{str_window_avg}_varclass","Pyr")] =\
                var_class
            variance_class_dict.update({f"Pyr, {radtype}":limits})
                
            plot_names.append((f"COD_550_{radtype}_inv_{str_window_avg}_avg",
                               f"COD_550_{radtype}_inv_{str_window_avg}_varclass",
                               "Pyr",f"Pyr, {radtype}"))
    
    #Calculate variance classes for PV
    for pv_type in ["PV_1min","PV_15min"]:           
        if pv_type in df_mean.columns.levels[1]:
            var_class, limits = variance_class(df_mean[(f"COD_550_poa_inv_{str_window_avg}_std",pv_type)],
                                               num_classes)
            df_mean[(f"COD_550_poa_inv_{str_window_avg}_varclass",pv_type)] =\
                var_class
            variance_class_dict.update({pv_type:limits})
                
            plot_names.append((f"COD_550_poa_inv_{str_window_avg}_avg",
                               f"COD_550_poa_inv_{str_window_avg}_varclass",
                               pv_type,pv_type))
    
    df_mean.sort_index(axis=1,level=1,inplace=True)        
    
    
    if flags["combo_stats"]:
        print(f'Plotting box-whisker plots for all stations, {year}')        
        box_plots_variance("all stations",df_mean,year.split('_')[1],plot_names,str_window_avg,num_classes,
                           variance_class_dict,flags["titles"],styles,plotpath)
        
        print(f'Plotting histograms for all stations, {year}')
        cod_histograms("all stations",df_mean,year.split('_')[1],plot_names,str_window_avg,num_classes,
                           variance_class_dict,flags["titles"],styles,plotpath)
    
    return df_mean, variance_class_dict

def calc_statistics_cod(key,pv_station,year,pvrad_config,pyrcal_config,window_avgs,folder):
    """
    

    Parameters
    ----------
    key : string with name of PV station
    pv_station : dictionary with information and data from PV station
    year : string with year under consideration        
    pvrad_config : dictionary with PV inversion configuration
    pyrcal_config : dictionary with pyranometer calibration configuration
    window_avgs : list of averaging windows
    folder : string with folder to save results

    Returns
    -------
    dataframe with final combined statistics

    """
    
    if "stats" not in pv_station:
        pv_station.update({"stats":{}})
    pv_station["stats"].update({year:{}})
    stats = pv_station["stats"][year]
    
    yrname = year.split('_')[-1]                                               
    
    for str_window_avg in window_avgs:
        if str_window_avg == "60min":
            name_cosmo_avg = "COD_tot_600_avg"
            name_cosmo_std = "COD_tot_600_iqr"
        else:
            name_cosmo_avg = f"COD_tot_600_{str_window_avg}_avg"
            name_cosmo_std = f"COD_tot_600_{str_window_avg}_std"
        
        dfs_stats = []        
        dfname = f"df_stats_cod_{yrname}_{str_window_avg}"                                                                                               
        
        for substat in pv_station["substations_pyr"]:    
            radname = pv_station["substations_pyr"][substat]["name"]                    
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
            
            for radtype in radtypes:                
                                                    
                if (f"COD_550_{radtype}_inv_{str_window_avg}_avg",substat,key) in pv_station[dfname].columns:
                    #Apollo
                    delta_cod_pyr_apollo = pv_station[dfname][(f'COD_550_{radtype}_inv_{str_window_avg}_avg',
                                           substat, key)] - pv_station[dfname]\
                                            [(f'cot_AP_{str_window_avg}_avg','apollo', key)]
    
                    rmse = (((delta_cod_pyr_apollo)**2).mean())**0.5
                    mad = abs(delta_cod_pyr_apollo).mean()
                    mbe = delta_cod_pyr_apollo.mean()
                    
                    mean_obs = pv_station[dfname]\
                                            [(f'cot_AP_{str_window_avg}_avg','apollo', key)].mean(axis=0)
                    rrmse = rmse/mean_obs*100
                    rmbe = mbe/mean_obs*100
                    
                    delta_max_plus = delta_cod_pyr_apollo.max()
                    delta_max_minus = delta_cod_pyr_apollo.min()                
                    
                    n_delta_pyr_apollo = len(delta_cod_pyr_apollo.dropna())
                        
                    stats.update({substat:{}})
                    
                    stats[substat].update({f"n_delta_pyr_{radtype}_apollo":n_delta_pyr_apollo})
                    stats[substat].update({f"RMSE_COD_pyr_{radtype}_apollo_Wm2":rmse})
                    stats[substat].update({f"rRMSE_COD_pyr_{radtype}_apollo_%":rrmse})
                    stats[substat].update({f"MAD_COD_pyr_{radtype}_apollo_Wm2":mad})
                    stats[substat].update({f"MBE_COD_pyr_{radtype}_apollo_Wm2":mbe})
                    stats[substat].update({f"rMBE_COD_pyr_{radtype}_apollo_%":rmbe})
                    stats[substat].update({f"max_Delta_COD_plus_pyr_{radtype}_apollo_Wm2":delta_max_plus})
                    stats[substat].update({f"max_Delta_COD_minus_pyr_{radtype}_apollo_Wm2":delta_max_minus})
                    
                    print(f"{key}, {yrname}: statistics at {str_window_avg} calculated with {n_delta_pyr_apollo} measurements")
                    print(f"RMSE for COD inverted from {substat}, {radtype} compared to APOLLO is {rmse:.2f} ({rrmse:.1f}%)")
                    print(f"MAE for COD inverted from {substat}, {radtype} compared to APOLLO is {mad:.2f}")
                    print(f"MBE for COD inverted from {substat}, {radtype} compared to APOLLO is {mbe:.2f} ({rmbe:.1f}%)")
                    
                    #Assign delta to the dataframe
                    pv_station[dfname][(f"delta_COD_pyr_{radtype}_apollo_Wm2",substat,key)] = \
                                        delta_cod_pyr_apollo
                    
                    if f"df_delta_pyr_{str_window_avg}_{yrname}" not in pv_station:
                        pv_station[f"df_delta_pyr_{str_window_avg}_{yrname}"] = pv_station[dfname][
                            [(f"delta_COD_pyr_{radtype}_apollo_Wm2",substat,key),(f'COD_550_{radtype}_inv_{str_window_avg}_avg',
                                   substat, key),(f'cot_AP_{str_window_avg}_avg','apollo', key)]]
                    else:
                        pv_station[f"df_delta_pyr_{str_window_avg}_{yrname}"] = pd.concat([pv_station[f"df_delta_pyr_{str_window_avg}_{yrname}"],
                                pv_station[dfname][[(f"delta_COD_pyr_{radtype}_apollo_Wm2",substat,key),(f'COD_550_{radtype}_inv_{str_window_avg}_avg',
                                   substat, key),(f'cot_AP_{str_window_avg}_avg','apollo', key)]]],axis=1)            
                    
                    if "seviri" in pv_station[dfname].columns.levels[1]:
                    
                        delta_cod_pyr_seviri = pv_station[dfname][(f'COD_550_{radtype}_inv_{str_window_avg}_avg',
                                               substat, key)] - pv_station[dfname]\
                                                [(f'COD_500_{str_window_avg}_avg','seviri', key)]
    
                        rmse = (((delta_cod_pyr_seviri)**2).mean())**0.5
                        mad = abs(delta_cod_pyr_seviri).mean()
                        mbe = delta_cod_pyr_seviri.mean()
                        
                        mean_obs = pv_station[dfname]\
                                            [(f'COD_500_{str_window_avg}_avg','seviri', key)].mean(axis=0)
                        rrmse = rmse/mean_obs*100
                        rmbe = mbe/mean_obs*100
                        
                        delta_max_plus = delta_cod_pyr_seviri.max()
                        delta_max_minus = delta_cod_pyr_seviri.min()                
                        
                        n_delta_pyr_seviri = len(delta_cod_pyr_seviri.dropna())                                        
                        
                        stats[substat].update({f"n_delta_pyr_{radtype}_seviri":n_delta_pyr_seviri})
                        stats[substat].update({f"RMSE_COD_pyr_{radtype}_seviri_Wm2":rmse})
                        stats[substat].update({f"rRMSE_COD_pyr_{radtype}_seviri_%":rrmse})
                        stats[substat].update({f"MAD_COD_pyr_{radtype}_seviri_Wm2":mad})
                        stats[substat].update({f"MBE_COD_pyr_{radtype}_seviri_Wm2":mbe})
                        stats[substat].update({f"rMBE_COD_pyr_{radtype}_seviri_%":rmbe})
                        stats[substat].update({f"max_Delta_COD_plus_pyr_{radtype}_seviri_Wm2":delta_max_plus})
                        stats[substat].update({f"max_Delta_COD_minus_pyr_{radtype}_seviri_Wm2":delta_max_minus})
                        
                        print(f"{key}, {yrname}: statistics at {str_window_avg} calculated with {n_delta_pyr_seviri} measurements")
                        print(f"RMSE for COD inverted from {substat}, {radtype} compared to SEVIRI is {rmse:.2f} ({rrmse:.1f}%)")
                        print(f"MAE for COD inverted from {substat}, {radtype} compared to SEVIRI is {mad:.2f}")
                        print(f"MBE for COD inverted from {substat}, {radtype} compared to SEVIRI is {mbe:.2f} ({rmbe:.1f}%)")
                        
                        #Assign delta to the dataframe
                        pv_station[dfname][(f"delta_COD_pyr_{radtype}_seviri_Wm2",substat,key)] = \
                                            delta_cod_pyr_seviri
                                            
                        pv_station[f"df_delta_pyr_{str_window_avg}_{yrname}"] = pd.concat([pv_station[f"df_delta_pyr_{str_window_avg}_{yrname}"],
                                pv_station[dfname][[(f"delta_COD_pyr_{radtype}_seviri_Wm2",substat,key),
                                                    (f'COD_500_{str_window_avg}_avg','seviri', key)]]],axis=1) 
                        
                    
                    if "cosmo" in pv_station[dfname].columns.levels[1]:
                        delta_cod_pyr_cosmo = pv_station[dfname][(f'COD_550_{radtype}_inv_{str_window_avg}_avg',
                                           substat, key)] - pv_station[dfname]\
                                            [('COD_tot_600_avg','cosmo', key)]
    
                        rmse = (((delta_cod_pyr_cosmo)**2).mean())**0.5
                        mad = abs(delta_cod_pyr_cosmo).mean()
                        mbe = delta_cod_pyr_cosmo.mean()
                        
                        mean_obs = pv_station[dfname]\
                                            [('COD_tot_600_avg','cosmo', key)].mean(axis=0)
                        rrmse = rmse/mean_obs*100
                        rmbe = mbe/mean_obs*100
                        
                        delta_max_plus = delta_cod_pyr_cosmo.max()
                        delta_max_minus = delta_cod_pyr_cosmo.min()                
                        
                        n_delta_pyr_cosmo = len(delta_cod_pyr_cosmo.dropna())                                                
                        
                        stats[substat].update({f"n_delta_pyr_{radtype}_cosmo":n_delta_pyr_cosmo})
                        stats[substat].update({f"RMSE_COD_pyr_{radtype}_cosmo_Wm2":rmse})
                        stats[substat].update({f"rRMSE_COD_pyr_{radtype}_cosmo_%":rrmse})
                        stats[substat].update({f"MAD_COD_pyr_{radtype}_cosmo_Wm2":mad})
                        stats[substat].update({f"MBE_COD_pyr_{radtype}_cosmo_Wm2":mbe})
                        stats[substat].update({f"rMBE_COD_pyr_{radtype}_cosmo_%":rmbe})
                        stats[substat].update({f"max_Delta_COD_plus_pyr_{radtype}_cosmo_Wm2":delta_max_plus})
                        stats[substat].update({f"max_Delta_COD_minus_pyr_{radtype}_cosmo_Wm2":delta_max_minus})
                        
                        print(f"{key}, {yrname}: statistics at {str_window_avg} calculated with {n_delta_pyr_cosmo} measurements")
                        print(f"RMSE for COD inverted from {substat}, {radtype} compared to COSMO is {rmse:.2f} ({rrmse:.1f}%)")
                        print(f"MAE for COD inverted from {substat}, {radtype} compared to COSMO is {mad:.2f}")
                        print(f"MBE for COD inverted from {substat}, {radtype} compared to COSMO is {mbe:.2f} ({rmbe:.1f}%)")
                    
                        #Assign delta to the dataframe
                        pv_station[dfname][(f"delta_COD_pyr_{radtype}_cosmo_Wm2",substat,key)] = \
                                            delta_cod_pyr_cosmo
                            
                        pv_station[f"df_delta_pyr_{str_window_avg}_{yrname}"] = pd.concat([pv_station[f"df_delta_pyr_{str_window_avg}_{yrname}"],
                                pv_station[dfname][[(f"delta_COD_pyr_{radtype}_cosmo_Wm2",substat,key),
                                                    ('COD_tot_600_avg','cosmo', key)]]],axis=1) 
                                                            
                                    
                    #Write stats results to text file
                    #write_results_table(key,substat,stats[substat],pyrname,year,folder)
                    
                    df_stats = pd.DataFrame(pv_station["stats"][year][substat],index=[key])
                    # new_columns = ["n_delta_pv_seviri"]
                    # new_columns.extend(["_".join([val for i, val in enumerate(col.split('_')) 
                    #              if i != len(col.split('_')) - 2]) for col in df_stats.columns[1:]])
                    # new_columns.extend(df_stats.columns[6:])
                    if "egrid" in substat and key != "MS_02":
                        new_substat = substat.split('_')[0]
                    else:
                        new_substat = substat
                    df_stats.columns = pd.MultiIndex.from_product([df_stats.columns,[new_substat]],
                                 names=['variable', 'substat'])
                    df_stats.index.rename('station',inplace=True)
                    dfs_stats.append(df_stats)
        
        #PV systems
        if "substations_pv" in pv_station:
            for substat_type in pv_station["substations_pv"]:    
                for substat in pv_station["substations_pv"][substat_type]["data"]:
                    if year in pv_station["substations_pv"][substat_type]["source"]:                                                                        
                                        
                        dfname = f"df_stats_cod_{yrname}_{str_window_avg}"                                                                                               
                        
                        #Apollo
                        delta_cod_pv_apollo = pv_station[dfname][(f'COD_550_poa_inv_{str_window_avg}_avg',
                                               substat, key)] - pv_station[dfname]\
                                                [(f'cot_AP_{str_window_avg}_avg','apollo', key)]
    
                        rmse = (((delta_cod_pv_apollo)**2).mean())**0.5
                        mad = abs(delta_cod_pv_apollo).mean()
                        mbe = delta_cod_pv_apollo.mean()
                        
                        mean_obs = pv_station[dfname]\
                                                [(f'cot_AP_{str_window_avg}_avg','apollo', key)].mean(axis=0)
                        rrmse = rmse/mean_obs*100
                        rmbe = mbe/mean_obs*100
                        
                        delta_max_plus = delta_cod_pv_apollo.max()
                        delta_max_minus = delta_cod_pv_apollo.min()                
                        
                        n_delta_pv_apollo = len(delta_cod_pv_apollo.dropna())
                            
                        stats.update({substat:{}})
                        
                        stats[substat].update({"n_delta_pv_apollo":n_delta_pv_apollo})
                        stats[substat].update({"RMSE_COD_pv_apollo_Wm2":rmse})
                        stats[substat].update({"rRMSE_COD_pv_apollo_%":rrmse})
                        stats[substat].update({"MAD_COD_pv_apollo_Wm2":mad})
                        stats[substat].update({"MBE_COD_pv_apollo_Wm2":mbe})
                        stats[substat].update({"rMBE_COD_pv_apollo_%":rmbe})
                        stats[substat].update({"max_Delta_COD_plus_pv_apollo_Wm2":delta_max_plus})
                        stats[substat].update({"max_Delta_COD_minus_pv_apollo_Wm2":delta_max_minus})
                        
                        print(f"{key}, {yrname}: statistics at {str_window_avg} calculated with {n_delta_pv_apollo} measurements")
                        print(f"RMSE for COD inverted from {substat} compared to APOLLO is {rmse:.2f} ({rrmse:.1f}%)")
                        print(f"MAE for COD inverted from {substat} compared to APOLLO is {mad:.2f}")
                        print(f"MBE for COD inverted from {substat} compared to APOLLO is {mbe:.2f} ({rmbe:.1f}%)")
                        
                        #Assign delta to the dataframe
                        pv_station[dfname][("delta_COD_pv_apollo_Wm2",substat,key)] = \
                                            delta_cod_pv_apollo
                        
                        if f"df_delta_pv_{str_window_avg}_{yrname}" not in pv_station:
                            pv_station[f"df_delta_pv_{str_window_avg}_{yrname}"] = pv_station[dfname][
                                [("delta_COD_pv_apollo_Wm2",substat,key),(f'COD_550_poa_inv_{str_window_avg}_avg',
                                       substat, key),(f'cot_AP_{str_window_avg}_avg','apollo', key)]]
                        else:
                            pv_station[f"df_delta_pv_{str_window_avg}_{yrname}"] = pd.concat([pv_station[f"df_delta_pv_{str_window_avg}_{yrname}"],
                                    pv_station[dfname][[("delta_COD_pv_apollo_Wm2",substat,key),(f'COD_550_poa_inv_{str_window_avg}_avg',
                                       substat, key),(f'cot_AP_{str_window_avg}_avg','apollo', key)]]],axis=1)            
                        
                        if "seviri" in pv_station[dfname].columns.levels[1]:
                        
                            delta_cod_pv_seviri = pv_station[dfname][(f'COD_550_poa_inv_{str_window_avg}_avg',
                                                   substat, key)] - pv_station[dfname]\
                                                    [(f'COD_500_{str_window_avg}_avg','seviri', key)]
        
                            rmse = (((delta_cod_pv_seviri)**2).mean())**0.5
                            mad = abs(delta_cod_pv_seviri).mean()
                            mbe = delta_cod_pv_seviri.mean()
                            
                            mean_obs = pv_station[dfname]\
                                                    [(f'COD_500_{str_window_avg}_avg','seviri', key)].mean(axis=0)
                            rrmse = rmse/mean_obs*100
                            rmbe = mbe/mean_obs*100
                            
                            delta_max_plus = delta_cod_pv_seviri.max()
                            delta_max_minus = delta_cod_pv_seviri.min()                
                            
                            n_delta_pv_seviri = len(delta_cod_pv_seviri.dropna())                                        
                            
                            stats[substat].update({"n_delta_pv_seviri":n_delta_pv_seviri})
                            stats[substat].update({"RMSE_COD_pv_seviri_Wm2":rmse})
                            stats[substat].update({"rRMSE_COD_pv_seviri_%":rrmse})
                            stats[substat].update({"MAD_COD_pv_seviri_Wm2":mad})
                            stats[substat].update({"MBE_COD_pv_seviri_Wm2":mbe})
                            stats[substat].update({"rMBE_COD_pv_seviri_%":rmbe})
                            stats[substat].update({"max_Delta_COD_plus_pv_seviri_Wm2":delta_max_plus})
                            stats[substat].update({"max_Delta_COD_minus_pv_seviri_Wm2":delta_max_minus})
                            
                            print(f"{key}, {yrname}: statistics at {str_window_avg} calculated with {n_delta_pv_seviri} measurements")
                            print(f"RMSE for COD inverted from {substat} compared to SEVIRI is {rmse:.2f} ({rrmse:.1f}%)")
                            print(f"MAE for COD inverted from {substat} compared to SEVIRI is {mad:.2f}")
                            print(f"MBE for COD inverted from {substat} compared to SEVIRI is {mbe:.2f} ({rmbe:.1f}%)")
                            
                            #Assign delta to the dataframe
                            pv_station[dfname][("delta_COD_pv_seviri_Wm2",substat,key)] = \
                                                delta_cod_pv_seviri
                                                
                            pv_station[f"df_delta_pv_{str_window_avg}_{yrname}"] = pd.concat([pv_station[f"df_delta_pv_{str_window_avg}_{yrname}"],
                                    pv_station[dfname][[("delta_COD_pv_seviri_Wm2",substat,key),
                                                        (f'COD_500_{str_window_avg}_avg','seviri', key)]]],axis=1) 
                            
                        
                        if "cosmo" in pv_station[dfname].columns.levels[1]:
                            delta_cod_pv_cosmo = pv_station[dfname][(f'COD_550_poa_inv_{str_window_avg}_avg',
                                               substat, key)] - pv_station[dfname]\
                                                [('COD_tot_600_avg','cosmo', key)]
    
                            rmse = (((delta_cod_pv_cosmo)**2).mean())**0.5
                            mad = abs(delta_cod_pv_cosmo).mean()
                            mbe = delta_cod_pv_cosmo.mean()
                            
                            mean_obs = pv_station[dfname]\
                                                [('COD_tot_600_avg','cosmo', key)].mean(axis=0)
                            rrmse = rmse/mean_obs*100
                            rmbe = mbe/mean_obs*100
                            
                            delta_max_plus = delta_cod_pv_cosmo.max()
                            delta_max_minus = delta_cod_pv_cosmo.min()                
                            
                            n_delta_pv_cosmo = len(delta_cod_pv_cosmo.dropna())                                                
                            
                            stats[substat].update({"n_delta_pv_cosmo":n_delta_pv_cosmo})
                            stats[substat].update({"RMSE_COD_pv_cosmo_Wm2":rmse})
                            stats[substat].update({"rRMSE_COD_pv_cosmo_%":rrmse})
                            stats[substat].update({"MAD_COD_pv_cosmo_Wm2":mad})
                            stats[substat].update({"MBE_COD_pv_cosmo_Wm2":mbe})
                            stats[substat].update({"rMBE_COD_pv_cosmo_%":rmbe})
                            stats[substat].update({"max_Delta_COD_plus_pv_cosmo_Wm2":delta_max_plus})
                            stats[substat].update({"max_Delta_COD_minus_pv_cosmo_Wm2":delta_max_minus})
                            
                            print(f"{key}, {yrname}: statistics at {str_window_avg} calculated with {n_delta_pv_cosmo} measurements")
                            print(f"RMSE for COD inverted from {substat} compared to COSMO is {rmse:.2f} ({rrmse:.1f}%)")
                            print(f"MAE for COD inverted from {substat} compared to COSMO is {mad:.2f}")
                            print(f"MBE for COD inverted from {substat} compared to COSMO is {mbe:.2f} ({rmbe:.1f}%)")
                        
                            #Assign delta to the dataframe
                            pv_station[dfname][("delta_COD_pv_cosmo_Wm2",substat,key)] = \
                                                delta_cod_pv_cosmo
                                
                            pv_station[f"df_delta_pv_{str_window_avg}_{yrname}"] = pd.concat([pv_station[f"df_delta_pv_{str_window_avg}_{yrname}"],
                                    pv_station[dfname][[("delta_COD_pv_cosmo_Wm2",substat,key),
                                                        ('COD_tot_600_avg','cosmo', key)]]],axis=1) 
                                                                
                                        
                        #Write stats results to text file
                        #write_results_table(key,substat,stats[substat],pyrname,year,folder)
                        
                        df_stats = pd.DataFrame(pv_station["stats"][year][substat],index=[key])
                        # new_columns = ["n_delta_pv_seviri"]
                        # new_columns.extend(["_".join([val for i, val in enumerate(col.split('_')) 
                        #              if i != len(col.split('_')) - 2]) for col in df_stats.columns[1:]])
                        # new_columns.extend(df_stats.columns[6:])
                        if "egrid" in substat and key != "MS_02":
                            new_substat = substat.split('_')[0]
                        else:
                            new_substat = substat
                        df_stats.columns = pd.MultiIndex.from_product([df_stats.columns,[new_substat]],
                                     names=['variable', 'substat'])
                        df_stats.index.rename('station',inplace=True)
                        dfs_stats.append(df_stats)
                    #else: return pd.DataFrame() #concat(dfs_stats,axis=1)
                    
    if len(dfs_stats) != 0:
        df_stats_final = pd.concat(dfs_stats,axis=1)
        for str_window_avg in window_avgs:
            if f"df_delta_pyr_{str_window_avg}_{yrname}" in pv_station:
                pv_station[f"df_delta_pyr_{str_window_avg}_{yrname}"] = pv_station[f"df_delta_pyr_{str_window_avg}_{yrname}"].loc\
                [:,~pv_station[f"df_delta_pyr_{str_window_avg}_{yrname}"].columns.duplicated()].copy()
                #new_var_pyr_cols = []
                
                if radtypes:
                    pyr_final_multiindex = pd.MultiIndex(levels=[[],[],[]],
                             codes=[[],[],[]],
                             names=['substat','variable','station'])
                    
                    for substat in pv_station["substations_pyr"]:
                        radname = pv_station["substations_pyr"][substat]["name"]                    
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
                        var_pyr_cols = []                                        
                        for radtype in radtypes:
                            if (f"COD_550_{radtype}_inv_{str_window_avg}_avg",substat,key) in\
                                pv_station[f"df_delta_pyr_{str_window_avg}_{yrname}"]:
                                var_pyr_cols.extend([f'delta_COD_pyr_{radtype}_apollo_Wm2',f'COD_PYR_{radtype}_inv'])
                                if "COD_apollo_ref" not in pyr_final_multiindex.levels[0]\
                                and "COD_apollo_ref" not in var_pyr_cols:
                                    var_pyr_cols.append("COD_apollo_ref")
                                if "seviri" in pv_station[f"df_delta_pyr_{str_window_avg}_{yrname}"].columns.levels[1]:
                                    var_pyr_cols.extend([f"delta_COD_pyr_{radtype}_seviri_Wm2"])
                                    if "COD_seviri_ref" not in pyr_final_multiindex.levels[0]\
                                        and "COD_seviri_ref" not in var_pyr_cols:
                                        var_pyr_cols.append("COD_seviri_ref")                        
                                if "cosmo" in pv_station[f"df_delta_pyr_{str_window_avg}_{yrname}"].columns.levels[1]:
                                    var_pyr_cols.extend([f"delta_COD_pyr_{radtype}_cosmo_Wm2"])
                                    if "COD_cosmo_ref" not in pyr_final_multiindex.levels[0]\
                                        and "COD_cosmo_ref" not in var_pyr_cols:
                                        var_pyr_cols.append("COD_cosmo_ref")
                        
                        # for x in var_pyr_cols: 
                        #     if x not in new_var_pyr_cols:
                        #         new_var_pyr_cols.append(x) 
                        
                        pyr_multiindex = pd.MultiIndex.from_product(
                            [[substat],var_pyr_cols,[key]], #,
                            names=['substat','variable','station']).swaplevel(0,1)
                        pyr_final_multiindex = pyr_final_multiindex.append(pyr_multiindex)
                else: pyr_multiindex = pd.MultiIndex(levels=[[],[],[]],
                             codes=[[],[],[]],
                             names=['substat','variable','station'])
            
                
                pv_station[f"df_delta_pyr_{str_window_avg}_{yrname}"].columns = \
                    pyr_final_multiindex
            else:  pv_station[f"df_delta_pyr_{str_window_avg}_{yrname}"] = pd.DataFrame()
            
            if f"df_delta_pv_{str_window_avg}_{yrname}" in pv_station:
                # pv_station[f"df_delta_pv_{str_window_avg}_{yrname}"] = pv_station[f"df_delta_pv_{str_window_avg}_{yrname}"].loc\
                # [:,~pv_station[f"df_delta_pv_{str_window_avg}_{yrname}"].columns.duplicated()].copy()
                
                if f'COD_550_poa_inv_{str_window_avg}_avg' in pv_station[f"df_delta_pv_{str_window_avg}_{yrname}"].columns.levels[0]:
                    new_substat_pv_cols = [col for col in pv_station[f"df_delta_pv_{str_window_avg}_{yrname}"].columns.levels[1]\
                                if (col != "apollo") and (col != "seviri")
                                and (col != "cosmo") and (("egrid" in col)
                                or ("auew" in col))]    #("Pyr" not in col) and ("CMP" not in col) 
                                # and ("RT" not in col) and (col != "suntracker") 
                                # and        
                    # if key != "MS_02":
                    #     new_substat_cols = [col.split('_')[0] for col in new_substat_cols if ("egrid" in col)]
                    # elif timeres == "15min":
                    #     new_cols = [col for col in new_cols if "auew" in col]
                    
                    new_var_pv_cols = ['delta_COD_pv_apollo_Wm2','COD_PV_inv',"COD_apollo_ref"]                    
                    
                    if "seviri" in pv_station[f"df_delta_pv_{str_window_avg}_{yrname}"].columns.levels[1]:
                        new_var_pv_cols.extend(["delta_COD_pv_seviri_Wm2","COD_seviri_ref"])                        
                        
                    if "cosmo" in pv_station[f"df_delta_pv_{str_window_avg}_{yrname}"].columns.levels[1]:
                        new_var_pv_cols.extend(["delta_COD_pv_cosmo_Wm2","COD_cosmo_ref"])                        
                        
                    pv_multiindex = pd.MultiIndex.from_product(
                        [new_substat_pv_cols,new_var_pv_cols,[key]], #,
                        names=['substat','variable','station']).swaplevel(0,1)
                else: pv_multiindex = pd.MultiIndex(levels=[[],[],[]],
                             codes=[[],[],[]],
                             names=['substat','variable','station'])
                                        
                pv_station[f"df_delta_pv_{str_window_avg}_{yrname}"].columns = \
                    pv_multiindex
            else:  pv_station[f"df_delta_pv_{str_window_avg}_{yrname}"] = pd.DataFrame()
                                         
    else: 
        df_stats_final = pd.DataFrame()
        pv_station[f"df_delta_pyr_{str_window_avg}_{yrname}"] = pd.DataFrame()
        pv_station[f"df_delta_pv_{str_window_avg}_{yrname}"] = pd.DataFrame()
        
    return df_stats_final

def combined_stats(dict_combo_stats,year,window_avgs):
    """
    Calculate combined stats for different averaging times    

    Parameters
    ----------
    dict_combo_stats : dictionary with combined stats for different averaging times
    year : string, year under consideration
    window_avgs : list of averaging window times

    Returns
    -------
    None.

    """    
            
    for window_avg in window_avgs:
        df_delta_all = dict_combo_stats[f"df_delta_all_pyr_{window_avg}"].stack(dropna=True)
        dict_combo_stats.update({window_avg:{}})
        
        for data_type in ["apollo","seviri","cosmo"]:
                    
            for radtype in ["poa","down"]:
                if f"delta_COD_pyr_{radtype}_{data_type}_Wm2" in df_delta_all:
                    rmse = ((((df_delta_all[f"delta_COD_pyr_{radtype}_{data_type}_Wm2"].stack())**2).mean())**0.5)
                    mad = abs(df_delta_all[f"delta_COD_pyr_{radtype}_{data_type}_Wm2"].stack()).mean()
                    mbe = df_delta_all[f"delta_COD_pyr_{radtype}_{data_type}_Wm2"].stack().mean()
                    
                    mean_obs = df_delta_all[f"COD_{data_type}_ref"].stack().mean()
                    rrmse = rmse/mean_obs*100
                    rmbe = mbe/mean_obs*100                    
                    
                    delta_max_plus = df_delta_all[f"delta_COD_pyr_{radtype}_{data_type}_Wm2"].stack().max()
                    delta_max_minus = df_delta_all[f"delta_COD_pyr_{radtype}_{data_type}_Wm2"].stack().min()
                    
                    n_delta_pyr = len(df_delta_all[f"delta_COD_pyr_{radtype}_{data_type}_Wm2"].stack().dropna())
                    
                    dict_combo_stats[window_avg].update({f"n_delta_pyr_{radtype}_{data_type}":n_delta_pyr})
                    dict_combo_stats[window_avg].update({f"RMSE_COD_pyr_{radtype}_{data_type}_Wm2":rmse})
                    dict_combo_stats[window_avg].update({f"rRMSE_COD_pyr_{radtype}_{data_type}_%":rrmse})
                    dict_combo_stats[window_avg].update({f"MAD_COD_pyr_{radtype}_{data_type}_Wm2":mad})
                    dict_combo_stats[window_avg].update({f"MBE_COD_pyr_{radtype}_{data_type}_Wm2":mbe})
                    dict_combo_stats[window_avg].update({f"rMBE_COD_pyr_{radtype}_{data_type}_%":rmbe})
                    dict_combo_stats[window_avg].update({f"max_Delta_COD_pyr_{radtype}_{data_type}_plus_Wm2":delta_max_plus})
                    dict_combo_stats[window_avg].update({f"max_Delta_COD_pyr_{radtype}_{data_type}_minus_Wm2":delta_max_minus})
                    
                    print(f"{year}: combined statistics at {window_avg} from "\
                          f"{dict_combo_stats[f'df_delta_all_pyr_{window_avg}'].columns.levels[2].to_list()}"\
                          f" calculated with {n_delta_pyr} measurements")
                    print(f"Combined RMSE for GHI from {radtype} Pyranometers in {year} is {rmse:.2f} ({rrmse:.1f}%)")
                    print(f"Combined MAE for GHI from {radtype} Pyranometers in {year} is {mad:.2f}")
                    print(f"Combined MBE for GHI from {radtype} Pyranometers in {year} is {mbe:.2f} ({rmbe:.1f}%)")     
        
        df_delta_all = dict_combo_stats[f"df_delta_all_pv_{window_avg}"].stack(dropna=True)        
        if window_avg not in dict_combo_stats:
            dict_combo_stats.update({window_avg:{}})
        for data_type in ["apollo","seviri","cosmo"]:
            if f"delta_COD_pv_{data_type}_Wm2" in df_delta_all:
                rmse = ((((df_delta_all[f"delta_COD_pv_{data_type}_Wm2"].stack())**2).mean())**0.5)
                mad = abs(df_delta_all[f"delta_COD_pv_{data_type}_Wm2"].stack()).mean()
                mbe = df_delta_all[f"delta_COD_pv_{data_type}_Wm2"].stack().mean()
                
                mean_obs = df_delta_all[f"COD_{data_type}_ref"].stack().mean()
                rrmse = rmse/mean_obs*100
                rmbe = mbe/mean_obs*100  
                
                delta_max_plus = df_delta_all[f"delta_COD_pv_{data_type}_Wm2"].stack().max()
                delta_max_minus = df_delta_all[f"delta_COD_pv_{data_type}_Wm2"].stack().min()
                
                n_delta_pv = len(df_delta_all[f"delta_COD_pv_{data_type}_Wm2"].stack().dropna())
                
                dict_combo_stats[window_avg].update({f"n_delta_pv_{data_type}":n_delta_pv})
                dict_combo_stats[window_avg].update({f"RMSE_COD_pv_{data_type}_Wm2":rmse})
                dict_combo_stats[window_avg].update({f"rRMSE_COD_pv_{data_type}_%":rrmse})
                dict_combo_stats[window_avg].update({f"MAD_COD_pv_{data_type}_Wm2":mad})
                dict_combo_stats[window_avg].update({f"MBE_COD_pv_{data_type}_Wm2":mbe})
                dict_combo_stats[window_avg].update({f"rMBE_COD_pv_{data_type}_%":rmbe})
                dict_combo_stats[window_avg].update({f"max_Delta_COD_pv_{data_type}_plus_Wm2":delta_max_plus})
                dict_combo_stats[window_avg].update({f"max_Delta_COD_pv_{data_type}_minus_Wm2":delta_max_minus})
                
                print(f"{year}: combined statistics at {window_avg} from "\
                      f"{dict_combo_stats[f'df_delta_all_pv_{window_avg}'].columns.levels[2].to_list()}"\
                      f" calculated with {n_delta_pv} measurements")
                print(f"Combined RMSE for GHI from PV in {year} is {rmse:.2f} ({rrmse:.1f}%)")
                print(f"Combined MAE for GHI from PV in {year} is {mad:.2f}")
                print(f"Combined MBE for GHI from PV in {year} is {mbe:.2f} ({rmbe:.1f}%)")                                      


def plot_all_cod_combined_scatter(dict_stats,list_stations,pvrad_config,T_model,folder,title_flag,window_avgs):
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
    savepath = os.path.join(folder,'COD_Plots')
    if 'COD_Plots' not in res_dirs:
        os.mkdir(savepath)    
        
    # res_dirs = list_dirs(savepath)
    # savepath = os.path.join(savepath,'Scatter')
    # if 'Scatter' not in res_dirs:
    #     os.mkdir(savepath)
        
    years = ["mk_" + campaign.split('_')[1] for campaign in pvrad_config["calibration_source"]]    
    stations_label = '_'.join(["".join(s.split('_')) for s in list_stations])
    
    tres_list = window_avgs #pvrad_config["timeres_comparison"] + 
    data_types = [("apollo","APOLLO_NG"),("cosmo","COSMO")]           #("seviri","SEVIRI-HRV"), 
    
    #Combined plot with all three time resolutions
    for data_type, data_label in data_types:
        
        #1. Plot comparing inverted irradiance with that from satellite and weather data
        fig, axs = plt.subplots(len(tres_list),len(years),sharex='all',sharey='all')
                
                                            
        print(f"Plotting combined frequency scatter plot for {data_label}...please wait....")
        plot_data = []
        norm = []
        max_cod = 0.; min_z = 10.; max_z = 0.
        for i, ax in enumerate(axs.flatten()):            
            year = years[np.fmod(i,2)]
            timeres = tres_list[int((i - np.fmod(i,2))/2)]
            
            cod_data = dict_stats[year][f"df_delta_all_pv_{timeres}"].stack()\
                .loc[:,["COD_PV_inv",f"COD_{data_type}_ref"]].stack().dropna(how='any')
            
            cod_ref = cod_data[f"COD_{data_type}_ref"].values.flatten()
            cod_inv = cod_data["COD_PV_inv"].values.flatten()
            xy = np.vstack([cod_ref,cod_inv])
            z = gaussian_kde(xy)(xy)
            idx = z.argsort()
            
            plot_data.append((cod_ref[idx], cod_inv[idx], z[idx]))
                            
            max_cod = np.max([max_cod,cod_ref.max(),cod_inv.max()])
            max_cod = np.ceil(max_cod/100)*100    
            if np.fmod(i,2) == 0:
                max_z = 0.
                min_z = 10.
            
            max_z = np.max([max_z,np.max(z)])
            min_z = np.min([min_z,np.min(z)])
            
            if np.fmod(i,2) != 0:
                norm.append(plt.Normalize(min_z,max_z))
        
        for i, ax in enumerate(axs.flatten()):
            year = years[np.fmod(i,2)]
            timeres = tres_list[int((i - np.fmod(i,2))/2)]
            
            sc = ax.scatter(plot_data[i][0],plot_data[i][1], s=8, c=plot_data[i][2], 
                            cmap="plasma",norm=norm[int((i - np.fmod(i,2))/2)])
            
            ax.plot([0, 1], [0, 1], ls = '--', transform=ax.transAxes,c='k')
            #ax.set_title(f"{timeres}",ifontsize=14)
                            
            print(f"Using {dict_stats[year][timeres][f'n_delta_pv_{data_type}']} data points for {timeres}, {year} plot")
            ax.annotate(rf"MBE = {dict_stats[year][timeres][f'MBE_COD_pv_{data_type}_Wm2']:.2f} ({dict_stats[year][timeres][f'rMBE_COD_pv_{data_type}_%']:.1f} %)" "\n" \
                        rf"RMSE = {dict_stats[year][timeres][f'RMSE_COD_pv_{data_type}_Wm2']:.2f} ({dict_stats[year][timeres][f'rRMSE_COD_pv_{data_type}_%']:.1f} %)" "\n"\
                        rf"n = ${dict_stats[year][timeres][f'n_delta_pv_{data_type}']:.0f}$",
                     xy=(0.05,0.82),xycoords='axes fraction',fontsize=9,color='k',
                     bbox = dict(facecolor='lightgrey',edgecolor='none', alpha=0.5),
                     horizontalalignment='left',multialignment='left')     
            # ax.annotate(rf"RMSE = {dict_stats[year][timeres]['RMSE_GTI_Wm2']:.2f} W/m$^2$",
            #          xy=(0.05,0.85),xycoords='axes fraction',fontsize=10,bbox = dict(facecolor='lightgrey', alpha=0.3))  
            # ax.annotate(rf"n = {dict_stats[year][timeres]['n_delta']:.0f}",
            #          xy=(0.05,0.78),xycoords='axes fraction',fontsize=10,bbox = dict(facecolor='lightgrey', alpha=0.3))                  
            #ax.set_xticks([0,400,800,1200])
                
        #fig.subplots_adjust(wspace=-0.4,hspace=0.15)    
        cb = fig.colorbar(sc,ticks=[min_z,max_z], ax=axs[:2], shrink=0.6, location = 'top', 
                           aspect=20)   
        cb.set_ticklabels(["Low", "High"])  # horizontal colorbar
        cb.ax.tick_params(labelsize=14) 
        cb.set_label("PDF", labelpad=-10, fontsize=16)
        
        #Set axis limits
        for i, ax in enumerate(axs.flatten()):
            ax.set_xlim([0.1,150])
            ax.set_ylim([0.1,150])
            ax.set_aspect('equal')
            ax.grid(False)
            ax.set_xscale('log')
            ax.set_yscale('log')
            # if max_gti < 1400:
            #ax.set_xticks([0,500,1000])
            # else:
            #     ax.set_xticks([0,400,800,1200,1400])
            if i == 0:
                ax.set_xlabel(rf"COD ({data_label})",position=(1.1,0))
            #if i == 2:
                ax.set_ylabel(r"COD 550nm (PV,inv)")
                        
        # fig.add_subplot(111, frameon=False)
        # plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        # plt.grid(False)
        
        plt.savefig(os.path.join(savepath,f"cod_scatter_hist_combo_all_pv_{data_label}_{T_model['model']}_"\
                 f"{T_model['type']}_{stations_label}.png"),bbox_inches = 'tight')  
            
        plt.close(fig)
        
        for radtype in ["poa","down"]:
            #1. Plot comparing inverted irradiance with that from satellite and weather data
            fig, axs = plt.subplots(len(tres_list),len(years),sharex='all',sharey='all')
                    
                                                
            print(f"Plotting combined frequency scatter plot for {data_label}...please wait....")
            plot_data = []
            norm = []
            max_cod = 0.; min_z = 10.; max_z = 0.
            for i, ax in enumerate(axs.flatten()):            
                year = years[np.fmod(i,2)]
                timeres = tres_list[int((i - np.fmod(i,2))/2)]
                
                cod_data = dict_stats[year][f"df_delta_all_pyr_{timeres}"].stack()\
                    .loc[:,[f'COD_PYR_{radtype}_inv',f"COD_{data_type}_ref"]].stack().dropna(how='any')
                
                cod_ref = cod_data[f"COD_{data_type}_ref"].values.flatten()
                cod_inv = cod_data[f'COD_PYR_{radtype}_inv'].values.flatten()
                xy = np.vstack([cod_ref,cod_inv])
                z = gaussian_kde(xy)(xy)
                idx = z.argsort()
                
                plot_data.append((cod_ref[idx], cod_inv[idx], z[idx]))
                                
                max_cod = np.max([max_cod,cod_ref.max(),cod_inv.max()])
                max_cod = np.ceil(max_cod/100)*100    
                if np.fmod(i,2) == 0:
                    max_z = 0.
                    min_z = 10.
                
                max_z = np.max([max_z,np.max(z)])
                min_z = np.min([min_z,np.min(z)])
                
                if np.fmod(i,2) != 0:
                    norm.append(plt.Normalize(min_z,max_z))
            
            for i, ax in enumerate(axs.flatten()):
                year = years[np.fmod(i,2)]
                timeres = tres_list[int((i - np.fmod(i,2))/2)]
                
                sc = ax.scatter(plot_data[i][0],plot_data[i][1], s=8, c=plot_data[i][2], 
                                cmap="plasma",norm=norm[int((i - np.fmod(i,2))/2)])
                
                ax.plot([0, 1], [0, 1], ls = '--', transform=ax.transAxes,c='k')
                #ax.set_title(f"{timeres}",fontsize=14)
                                
                print(f"Using {dict_stats[year][timeres][f'n_delta_pyr_{radtype}_{data_type}']} data points for {timeres}, {year} plot")
                ax.annotate(rf"MBE = {dict_stats[year][timeres][f'MBE_COD_pyr_{radtype}_{data_type}_Wm2']:.2f} ({dict_stats[year][timeres][f'rMBE_COD_pyr_{radtype}_{data_type}_%']:.1f} %)" "\n" \
                            rf"RMSE = {dict_stats[year][timeres][f'RMSE_COD_pyr_{radtype}_{data_type}_Wm2']:.2f} ({dict_stats[year][timeres][f'rRMSE_COD_pyr_{radtype}_{data_type}_%']:.1f} %)" "\n"\
                            rf"n = ${dict_stats[year][timeres][f'n_delta_pyr_{radtype}_{data_type}']:.0f}$",
                         xy=(0.05,0.82),xycoords='axes fraction',fontsize=9,color='k',
                         bbox = dict(facecolor='lightgrey',edgecolor='none', alpha=0.5),
                         horizontalalignment='left',multialignment='left')     
                # ax.annotate(rf"RMSE = {dict_stats[year][timeres]['RMSE_GTI_Wm2']:.2f} W/m$^2$",
                #          xy=(0.05,0.85),xycoords='axes fraction',fontsize=10,bbox = dict(facecolor='lightgrey', alpha=0.3))  
                # ax.annotate(rf"n = {dict_stats[year][timeres]['n_delta']:.0f}",
                #          xy=(0.05,0.78),xycoords='axes fraction',fontsize=10,bbox = dict(facecolor='lightgrey', alpha=0.3))                  
                #ax.set_xticks([0,400,800,1200])
                    
            #fig.subplots_adjust(wspace=-0.4,hspace=0.15)    
            cb = fig.colorbar(sc,ticks=[min_z,max_z], ax=axs[:2], shrink=0.6, location = 'top', 
                               aspect=20)    
            cb.set_ticklabels(["Low", "High"])  # horizontal colorbar
            cb.ax.tick_params(labelsize=14) 
            cb.set_label("PDF", labelpad=-10, fontsize=16)
            
            #Set axis limits
            for i, ax in enumerate(axs.flatten()):
                ax.set_xlim([0.1,150])
                ax.set_ylim([0.1,150])
                ax.set_aspect('equal')
                ax.grid(False)
                ax.set_xscale('log')
                ax.set_yscale('log')
                # if max_gti < 1400:
                #ax.set_xticks([0,500,1000])
                # else:
                #     ax.set_xticks([0,400,800,1200,1400])
                if i == 0:
                    ax.set_xlabel(rf"COD ({data_label})",position=(1.1,0))
                #if i == 2:
                    ax.set_ylabel(rf"COD 550nm (pyr,{radtype},inv)")
                            
            # fig.add_subplot(111, frameon=False)
            # plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            # plt.grid(False)
            
            plt.savefig(os.path.join(savepath,f"cod_scatter_hist_combo_all_pyr_{radtype}_{data_label}_{T_model['model']}_"\
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
                        pvrad_config["results_path"]["optical_depth"])
    
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
        
    #Wavelength range of simulation
    wvl_config = rt_config["common_base"]["wavelength"]["pyranometer"]
    
    if type(wvl_config) == list:
        wvl_folder_label = "Wvl_" + str(int(wvl_config[0])) + "_" + str(int(wvl_config[1]))
    elif wvl_config == "all":
        wvl_folder_label = "Wvl_Full"

    dirs_exist = list_dirs(fullpath)
    fullpath = os.path.join(fullpath,wvl_folder_label)        
    if wvl_folder_label not in dirs_exist:
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


def save_results(name,pv_station,info,pyr_substats,pv_substats,rt_config,
                 pyr_config,pvcal_config,pvrad_config,config,savepath):
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
    
    filename_stat = f"cod_analysis_results_{info}_{name}.data"            
    
    with open(os.path.join(savepath,filename_stat), 'wb') as filehandle:  
        # store the data as binary data stream
        pickle.dump((pv_station_save, rt_config, pyr_config, pvcal_config, pvrad_config), filehandle)

    print('Results written to file %s\n' % filename_stat)
    
    #Write COD data to CSV    
    year = info.split('_')[1]  
    cf_avg_res = pvrad_config["cloud_fraction"]["cf_avg_window"]
    
    substat_cols = list(pyr_substats.keys())
    for timeres in pv_station["timeres"]:        
        if f"df_codfit_pyr_pv_{year}_{timeres}" in pv_station:            
            
            for substat_type in pv_substats:                
                if timeres == pv_substats[substat_type]["t_res_inv"]:
                    substat_cols.extend(list(pv_substats[substat_type]["data"].keys()))
            
            dataframe = pv_station[f"df_codfit_pyr_pv_{year}_{timeres}"].loc[:,
                        pd.IndexSlice[["COD_550_down_inv","COD_550_poa_inv",
                        f"cf_down_{cf_avg_res}_avg_alt",f"cf_poa_{cf_avg_res}_avg_alt"],substat_cols]]        
    
            #Write all results to CSV file
            filename_csv = f'cod_results_{name}_{timeres}_{year}.dat'
            f = open(os.path.join(savepath,filename_csv), 'w')
            f.write('#Station: %s, Cloud optical depth and cloud fraction inverted from PV and pyranometer data\n' % name)    
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
            f.write('#second line ("substat") refers to measurement device\n')
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
    
    filename = f"cod_combo_results_stats_{T_model['model']}_{stations_label}.data"
    
    with open(os.path.join(folder,filename), 'wb') as filehandle:  
        # store the data as binary data stream
        pickle.dump((dict_stats, list_stations, pvrad_config, T_model, window_avgs), filehandle)

    
    #Write combined results to CSV
    for measurement in pvrad_config["inversion_source"]:
        year = f"mk_{measurement.split('_')[1]}"
        
        for window_avg in window_avgs:
            for sensor in ["pv","pyr"]:
                if f"df_delta_all_{sensor}_{window_avg}" in dict_stats[year]:
                    
                    dataframe = dict_stats[year][f"df_delta_all_{sensor}_{window_avg}"]
                    filename_csv = f'cod_combo_results_{sensor}_{window_avg}_{year}_{T_model["model"]}.dat'
                    f = open(os.path.join(folder,filename_csv), 'w')
                    f.write(f'#Cloud optical depth results from {sensor} combined and averaged to {window_avg}\n')  
                    f.write(f'#Stations considered: {list_stations}\n')
                    f.write('#Comparison results for COSMO and APOLLO also listed\n')
                    
                    f.write('\n#Multi-index: first line ("variable") refers to measured quantity\n')
                    f.write('#second line ("substat") refers to sensor used for inversion of COD\n')
                    f.write('#third line ("station") refers to PV station\n')
                    f.write('#First column is the time stamp, in the format %Y-%m-%d %HH:%MM:%SS\n')
                    f.write('\n')                       
                    
                    dataframe.to_csv(f,sep=';',float_format='%.6f',
                                      na_rep='nan')
                    
                    f.close()    
                    print('Combined results written to file %s\n' % filename_csv)    
                
        

#%%Main Program
#######################################################################
###                         MAIN PROGRAM                            ###
#######################################################################
#This program plots results from the COD retrievals calculated in pvpyr2cod_interopolate_fit
#and compares them to COD retrievals from SEVIRI, both hi-res and APNG, plus from COSMO
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

#Load PVCAL configuration
pvcal_config = load_yaml_configfile(config["pvcal_configfile"])

#Load PV2rad configuration
pvrad_config = load_yaml_configfile(config["pvrad_configfile"])

#Load radiative transfer configuration
rt_config = load_yaml_configfile(config["rt_configfile"])

homepath = os.path.expanduser('~') # #"/media/luke" #
plot_styles = config["plot_styles"]
plot_flags = config["plot_flags"]

if args.campaign:
    campaigns = args.campaign
    if type(campaigns) != list:
        campaigns = [campaigns]
else:
    campaigns = config["description"] 


sza_limit = rt_config["sza_max"]["cod_cutoff"]
window_avgs = config["window_size_moving_average"]
num_var_classes = 3
T_model = pvcal_config["T_model"]

#%%Load COD retrievals
combo_stats = {}
for campaign in campaigns:
    year = "mk_" + campaign.split('_')[1]    
    yrname = year.split('_')[-1]
    #Load pyranometer configuration
    
    pyr_config = load_yaml_configfile(config["pyrcalod_configfile"][year])
    
    cs_threshold = pyr_config["cloud_fraction"]["cs_threshold"]
    window_avg_cf = pyr_config["cloud_fraction"]["cf_avg_window"]
    
    if args.station:
        stations = args.station
        if stations[0] == 'all':
            stations = 'all'
    else:
        #Stations for which to perform inversion
        stations = ["PV_12"] #"MS_02" #pyr_config["stations"]

    print(f'COD comparison and analysis for {campaign} at {stations} stations')    
    #Load inversion results
    print('Loading PYR2COD and PV2COD results')
    pvsys, results_folder, station_list = load_pvpyr2cod_results(rt_config,pyr_config,pvcal_config,pvrad_config,
                                   campaign,stations,homepath)

    dfs_stats_all = []
    var_class_dict = {}
    
    combo_stats.update({year:{}})
    combo_stats[year].update({f"df_{yrname}_stats":pd.DataFrame(index=station_list)})
    dfs_pyr_deviations = {}    
    dfs_pv_deviations = {}    
    for tres in window_avgs:
        dfs_pyr_deviations.update({tres:[]})    
        dfs_pv_deviations.update({tres:[]})    
    
    for key in pvsys:        
        print(f'\nCOD analysis for {key}, {year}')
        #Prepare surface inputs
        pvsys[key] = prepare_cosmo_seviri_data(year,pyr_config,key,pvsys[key],homepath)                
        
        #combine_apollo_seviri_hires(yrname,pvsys[key])
        
        for str_window_avg in window_avgs:
            print(f'Using {str_window_avg} moving average for comparison')
            pvsys[key][f"df_compare_cod_{year.split('_')[1]}_{str_window_avg}"] = combine_cosmo_seviri(year,
                                                            pvsys[key], str_window_avg)
    
            #Go through Pyranometers to plot
            if key in pyr_config["pv_stations"]:
                #Get substation parameters
                pyr_substat_pars = pvsys[key]["substations_pyr"]                        
                #Perform analysis and plot
                pvsys[key][f"df_compare_cod_{year.split('_')[1]}_{str_window_avg}"] = cod_analysis_plots(key,
                       pvsys[key],pyr_substat_pars,year.split('_')[1],results_folder,
                       sza_limit,cs_threshold,str_window_avg,window_avg_cf,plot_flags)                                                
            else:
                pyr_substat_pars = {}
                    
            #Go through PVs to plot
            if key in pvrad_config["pv_stations"]:
                pv_substat_pars = pvsys[key]["substations_pv"]
                for substat_type in pv_substat_pars:
                    pv_substats = pv_substat_pars[substat_type]["data"]
                    if year in pv_substat_pars[substat_type]["source"]:                    
                        pvsys[key][f"df_compare_cod_{year.split('_')[1]}_{str_window_avg}"] = cod_analysis_plots(key,
                            pvsys[key],pv_substats,year.split('_')[1],results_folder,
                            sza_limit,cs_threshold,str_window_avg,window_avg_cf,plot_flags)
            else:
                pv_substat_pars = {}            
            
            #Scatter plots            
            if plot_flags["scatter"]:
                print(f"Creating scatter plots with data averaged over {str_window_avg} and all days")
                scatter_plot_cod_comparison_grid(key,pvsys[key][f"df_compare_cod_{year.split('_')[1]}_{str_window_avg}"], 
                              rt_config,pyr_substat_pars,pv_substat_pars,year,plot_styles,
                              results_folder, results_folder, str_window_avg,
                              plot_flags,day_type='all')
                
                for day_type in pyr_config["test_days"]:                    
                    df_compare = pd.concat([pvsys[key][f"df_compare_cod_{year.split('_')[1]}_{str_window_avg}"].loc[day.strftime('%Y-%m-%d')]
                                            for day in pyr_config["test_days"][day_type]],axis=0)
                    
                    print(f"Creating scatter plots with data averaged over {str_window_avg} and {day_type} days")
                    scatter_plot_cod_comparison_grid(key,df_compare, 
                             rt_config,pyr_substat_pars,pv_substat_pars,year,plot_styles,
                             results_folder, results_folder, str_window_avg,
                             plot_flags,day_type)
                    
            
            #Calculate variance classes
            if key in pvrad_config["pv_stations"]:
                pv_substats = pvsys[key]["substations_pv"]
            else:
                pv_substats = {}
            
            #Stats: calculation and plots per station
            pvsys[key][f"df_stats_cod_{year.split('_')[1]}_{str_window_avg}"], pvsys[key]["var_class"] = cod_stats_plots(key,
                    pvsys[key][f"df_compare_cod_{year.split('_')[1]}_{str_window_avg}"],
                    pyr_substat_pars,pv_substats,
                    year,str_window_avg,num_var_classes,plot_styles,
                    plot_flags,results_folder)
            
            #Join all dataframes into one for all stations
            df_stats = pvsys[key][f"df_stats_cod_{year.split('_')[1]}_{str_window_avg}"]
            idx = df_stats.columns.to_frame()
            idx.insert(2, 'station', key)
            df_stats.columns = pd.MultiIndex.from_frame(idx) 
            
            dfs_stats_all.append(df_stats)
            var_class_dict.update({key:pvsys[key]["var_class"]})
            
        #Calculate statistics
        if combo_stats[year][f"df_{yrname}_stats"].empty:
            combo_stats[year][f"df_{yrname}_stats"] = calc_statistics_cod(
                key,pvsys[key],year,pvrad_config,pyr_config,window_avgs,results_folder)                            
        else:
            combo_stats[year][f"df_{yrname}_stats"] = pd.concat([combo_stats[year][f"df_{yrname}_stats"],
                 calc_statistics_cod(key,pvsys[key],year,pvrad_config,
                                            pyr_config,window_avgs,results_folder)],axis=0)            
                
        for str_window_avg in window_avgs:
            dfs_pyr_deviations[str_window_avg].append(pvsys[key][f"df_delta_pyr_{str_window_avg}_{yrname}"])                            
            dfs_pv_deviations[str_window_avg].append(pvsys[key][f"df_delta_pv_{str_window_avg}_{yrname}"])                            
        
        results_path = generate_folders(rt_config,pvcal_config,pvrad_config,homepath)
        save_results(key, pvsys[key], campaign, pyr_substat_pars, pv_substats, rt_config, pyr_config, 
                     pvcal_config, pvrad_config, config, results_path)
        
    for str_window_avg in window_avgs:
        if not combo_stats[year][f"df_{yrname}_stats"].empty:
            combo_stats[year][f"df_delta_all_pyr_{str_window_avg}"] = pd.concat(dfs_pyr_deviations[str_window_avg],axis=1)
            combo_stats[year][f"df_delta_all_pv_{str_window_avg}"] = pd.concat(dfs_pv_deviations[str_window_avg],axis=1)
        else:
            combo_stats[year][f"df_delta_all_pyr_{str_window_avg}"] = pd.DataFrame()
            combo_stats[year][f"df_delta_all_pv_{str_window_avg}"] = pd.DataFrame()
        if combo_stats[year][f"df_delta_all_pyr_{str_window_avg}"].empty:
            del combo_stats[year][f"df_delta_all_pyr_{str_window_avg}"]      
        if combo_stats[year][f"df_delta_all_pv_{str_window_avg}"].empty:
            del combo_stats[year][f"df_delta_all_pv_{str_window_avg}"]      
        
    if stations == "all":
        
        df_stats_all = pd.concat(dfs_stats_all,axis=1)
        df_stats_all.sort_index(axis=1,level=[1,2],inplace=True)        
            
        #Stats for all stations
        df_mean_stats, var_classes_mean = combined_stats_plots(df_stats_all,
                   pyr_substat_pars,pv_substats,year,str_window_avg,
                 num_var_classes,plot_styles,plot_flags,results_folder)
                
    combined_stats(combo_stats[year],yrname,window_avgs)

save_combo_stats(combo_stats,station_list,pvrad_config,T_model,
                                  results_folder,window_avgs)

if plot_flags["combo_stats"]:
    plot_all_cod_combined_scatter(combo_stats,station_list,pvrad_config,T_model,
                                  results_folder,plot_flags["titles"],window_avgs)
        
