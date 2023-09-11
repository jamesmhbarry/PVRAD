#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 14:46:52 2021

@author: james
"""
#%% Preamble
import os
import numpy as np
import pandas as pd
from pvcal_forward_model import E_poa_calc
from file_handling_functions import *
from rt_functions import *
import pickle
import collections
from scipy.interpolate import interp1d
from copy import deepcopy
from astropy.convolution import convolve, Box1DKernel

#%%Functions
def generate_folder_names_pyr2cf(rt_config,pyrcal_config):
    """
    Generate folder structure to retrieve PYR2CF simulation results
    
    args:    
    :param rt_config: dictionary with RT configuration
    :param pyrcal_config: dictionary with PYRCAL configuration
    
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
    
    #Get DISORT resolution folder label
    disort_config = rt_config["disort_rad_res"]   
    theta_res = str(disort_config["theta"]).replace('.','-')
    phi_res = str(disort_config["phi"]).replace('.','-')
    
    disort_folder_label = 'Zen_' + theta_res + '_Azi_' + phi_res
    filename = 'cloud_fraction_results_'
    
    if rt_config["atmosphere"] == "default":
        atm_folder_label = "Atmos_Default"    
    elif rt_config["atmosphere"] == "cosmo":
        atm_folder_label = "Atmos_COSMO"
        filename = filename + 'atm_'
        
    if rt_config["aerosol"]["source"] == "default":
        aero_folder_label = "Aerosol_Default"
    elif rt_config["aerosol"]["source"] == "aeronet":
        aero_folder_label = "Aeronet_" + rt_config["aerosol"]["station"]
        filename = filename + 'asl_' + rt_config["aerosol"]["data_res"] + '_'
            
    sza_label = "SZA_" + str(int(pyrcal_config["sza_max"]["inversion"]))

    folder_label = os.path.join(atm_geom_folder,disort_folder_label,
                                atm_folder_label,aero_folder_label,sza_label)
        
    return folder_label, filename, (theta_res,phi_res)

def generate_folder_names_poarad(rt_config,pvcal_config,pvrad_config):
    """
    Generate folder structure to retrieve POA Rad results
    
    args:    
    :param rt_config: dictionary with RT configuration
    :param pvcal_config: dictionary with PVCAL configuration
    :param pvrad_config: dictionary with PV inversion configuration
    
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
    
    #Wavelength range of simulation
    wvl_config = rt_config["common_base"]["wavelength"]["pv"]
    
    if type(wvl_config) == list:
        wvl_folder_label = "Wvl_" + str(int(wvl_config[0])) + "_" + str(int(wvl_config[1]))
    elif wvl_config == "all":
        wvl_folder_label = "Wvl_Full"
    
    #Get DISORT resolution folder label
    disort_config = rt_config["disort_rad_res"]   
    theta_res = str(disort_config["theta"]).replace('.','-')
    phi_res = str(disort_config["phi"]).replace('.','-')
    
    disort_folder_label = 'Zen_' + theta_res + '_Azi_' + phi_res
    filename = 'tilted_irradiance_cloud_fraction_results_'
    
    if rt_config["atmosphere"] == "default":
        atm_folder_label = "Atmos_Default"    
    elif rt_config["atmosphere"] == "cosmo":
        atm_folder_label = "Atmos_COSMO"
        filename = filename + 'atm_'
        
    if rt_config["aerosol"]["source"] == "default":
        aero_folder_label = "Aerosol_Default"
    elif rt_config["aerosol"]["source"] == "aeronet":
        aero_folder_label = "Aeronet_" + rt_config["aerosol"]["station"]
        filename = filename + 'asl_' + rt_config["aerosol"]["data_res"] + '_'
        
    sza_label = "SZA_" + str(int(pvrad_config["sza_max"]["inversion"]))
    
    model = pvcal_config["inversion"]["power_model"]
    eff_model = pvcal_config["eff_model"]
    T_model = pvcal_config["T_model"]["model"]   

    folder_label = os.path.join(atm_geom_folder,wvl_folder_label,disort_folder_label,
                                atm_folder_label,aero_folder_label,sza_label,model,eff_model,
                                T_model)
    
    return folder_label, filename, (theta_res,phi_res)


def load_pyr2cf_pv2poarad_results(rt_config,pyr_config,pvcal_config,pvrad_config,
                                  info,station_list,home):
    """
    Load results from pyranometer calibration and cloud fraction calculation
    As well as PV to POA irradiance inversion and cloud fraction
    
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
    """
    
    mainpath = os.path.join(home,pyr_config['results_path']['main'],
                            pyr_config['results_path']['cloud_fraction'])
    
    folder_label, filename, (theta_res,phi_res) = \
    generate_folder_names_pyr2cf(rt_config,pyr_config)
        
    filename = filename + info + '_disortres_' + theta_res + '_' + phi_res + '_'
    
    year = info.split('_')[1]
    #Define new dictionary to contain all information, data etc
    pv_systems = {}    
        
    #Choose which stations to load    
    if type(station_list) != list:
        station_list = [station_list]
        
    #Pyranometer stations
    if station_list[0] == "all":
        stations = list(pyr_config["pv_stations"].keys())
    else:
        stations = station_list

    for station in stations:                
        #Read in binary file that was saved from pvcal_radsim_disort
        filename_stat = filename + station + '.data'
        try:
            with open(os.path.join(mainpath,folder_label,filename_stat), 'rb') as filehandle:  
                # read the data as binary data stream
                (pvstat, dummy, pyr_config) = pd.read_pickle(filehandle)            
            
            pvstat["substations_pyr"] = pvstat.pop("substations")
            delkeys = []
            for key in pvstat:
                if year in key:
                    key_data = deepcopy(key)
                    keystring = key.split('_')
                    
                if "path" not in key and "lat_lon" not in key and year not in key\
                    and "substations" not in key:
                    delkeys.append(key)
            
            pvstat['_'.join([keystring[0],'pyr',year,keystring[-1]])] = \
                    pvstat.pop(key_data)
            
            for key in delkeys:
                del pvstat[key]
            
            pv_systems.update({station:pvstat})
            
            print('Pyranometer & cloud fraction data for %s, %s loaded' % (station,info))
        except IOError:
            print('There are no cloud fraction results from pyranometers at %s for %s' % (station,info))
    
    mainpath = os.path.join(home,pvrad_config['results_path']['main'],
                            pvrad_config['results_path']['inversion'])
    
    #Generate folder structure for loading files
    folder_label, filename, (theta_res,phi_res) = \
    generate_folder_names_poarad(rt_config,pvcal_config,pvrad_config)    
    
    #Check calibration source for filename    
    if len(pvrad_config["calibration_source"]) > 1:
        infos = '_'.join(pvrad_config["calibration_source"])
    else:
        infos = pvrad_config["calibration_source"][0]
    
    filename = filename + infos + '_disortres_' + theta_res + '_' + phi_res + '_'        
    
    #PV stations
    if station_list[0] == "all":
        stations = list(pvrad_config["pv_stations"].keys())
    else:
        stations = station_list

    for station in stations:                
        #Read in binary file that was saved from pvcal_radsim_disort
        filename_stat = filename + station + '.data'
        try:
            with open(os.path.join(mainpath,folder_label,filename_stat), 'rb') as filehandle:  
                # read the data as binary data stream
                (pvstat, dummy, pvcal_config, pvrad_config) = pd.read_pickle(filehandle)            
            
            #Renaming things to add "pv" in the dataframe names
            pvstat["substations_pv"] = pvstat.pop("substations")
            delkeys = []
            key_data = []
            for key in pvstat:
                if f"df_{year}" in key:
                    key_data.append(key)
                    
                if ("path" not in key and "lat_lon" not in key and year not in key\
                    and "substations" not in key) or ("cosmo" in key or "sim" in key):
                    delkeys.append(key)
            
            for key in key_data:
                keystring = key.split('_')
                pvstat['_'.join([keystring[0],'pv',year,keystring[-1]])] = \
                    pvstat.pop(key)
            
            for key in delkeys:
                del pvstat[key]
            
            if station not in pv_systems:
                pv_systems.update({station:pvstat})
            else:    
                pv_systems[station] = merge_two_dicts(pv_systems[station],pvstat)
            
            print('PV and POA irradiance data for %s, %s loaded' % (station,infos))
        except IOError:
            print('There are no PV2POARAD results from PV systems at %s, %s' % (station,infos))
            
    return pv_systems

def generate_folder_names_sim_results(rt_config,sensortype,skytype):
    """
    Generate folder structure to retrieve PYR2COD simulation results
    
    args:    
    :param rt_config: dictionary with RT configuration
    :param sensortype: string, either PV or pyranometer
    :param skytype: string, either clear or cloudy
    
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
    wvl_config = rt_config["common_base"]["wavelength"][sensortype]
    if type(wvl_config) == list:
        wvl_folder_label = "Wvl_" + str(int(wvl_config[0])) + "_" + str(int(wvl_config[1]))
    elif wvl_config == "all":
        wvl_folder_label = "Wvl_Full"
    
    #Get DISORT resolution folder label
    if skytype == "clear":
        disort_config = rt_config["disort_rad_res"]   
    elif skytype == "cloudy":
        disort_config = rt_config["clouds"]["disort_rad_res"]   
    theta_res = str(disort_config["theta"]).replace('.','-')
    phi_res = str(disort_config["phi"]).replace('.','-')
    
    disort_folder_label = 'Zen_' + theta_res + '_Azi_' + phi_res
    
    filename_clear = "lrt_sim_results_"
    filename_cloudy = 'cod_lut_results_'
    
    if rt_config["atmosphere"] == "default":
        atm_folder_label = "Atmos_Default"    
    elif rt_config["atmosphere"] == "cosmo":
        atm_folder_label = "Atmos_COSMO"
        filename_clear = filename_clear + 'atm_'
        
    if rt_config["aerosol"]["source"] == "default":
        aero_folder_label = "Aerosol_Default"
    elif rt_config["aerosol"]["source"] == "aeronet":
        aero_folder_label = "Aeronet_" + rt_config["aerosol"]["station"]
        filename_clear = filename_clear + 'asl_' + rt_config["aerosol"]["data_res"] + '_'
            
    if skytype == "cloudy":
        sza_label = "SZA_" + str(int(rt_config["sza_max"]["lut"]))
    elif skytype == "clear":
        sza_label = ""

    folder_label = os.path.join(atm_geom_folder,wvl_folder_label,disort_folder_label,
                                atm_folder_label,aero_folder_label,sza_label)
    
    if skytype == "clear":
        filename = filename_clear
    elif skytype == "cloudy":
        filename = filename_cloudy
        
    return folder_label, filename, (theta_res,phi_res)

def load_pv_pyr2cod_lut_results(pv_systems,rt_config,pyr_config,pv_config,info,station_list,home):
    """
    Load results from cloud optical depth simulations with DISORT
    
    args:  
    :pv_systems: dictionary of PV systems with data
    :param rt_config: dictionary with current RT configuration
    :param pyr_config: dictionary with current pyranometer calibration configuration        
    :param pv_config: dictionary with current inversion configuration    
    :param info: string with description of current campaign
    :param station_list: list of stations
    :param home: string with homepath
    
    out:
    :return pv_systems: dictionary of PV systems with data    
    :return pyr_folder_label: string with label for storing pyranometer results
    :return pv_folder_label: string with label for storing PV results    
    """
    
    #Path and filename for COD LUT
    mainpath_cloudy = os.path.join(home,rt_config['save_path']['disort'],
                                   rt_config['save_path']['optical_depth_lut'])     
    
    folder_label_cloudy, filename, (theta_res,phi_res) = \
    generate_folder_names_sim_results(rt_config, "pyranometer", "cloudy")

    pyrlut_folder_label_cloudy = os.path.join(mainpath_cloudy,folder_label_cloudy)    
    
    filename_cloudy = filename + info + '_'#+ '_disortres_' + theta_res + '_' + phi_res + '_'
    
    #Path and filename for clear sky simulation
    mainpath_clear = os.path.join(home,rt_config['save_path']['disort'],
                                   rt_config['save_path']['clear_sky'])     
    
    folder_label_clear, filename, (theta_res,phi_res) = \
    generate_folder_names_sim_results(rt_config, "pyranometer", "clear")

    pyrlut_folder_label_clear = os.path.join(mainpath_clear,folder_label_clear)    
    
    filename_clear = filename + info + '_disortres_' + theta_res + '_' + phi_res + '_'
    
    #Choose which stations to load    
    if type(station_list) != list:
        station_list = [station_list]
    
    #Pyranometer stations
    if station_list[0] == "all":
        stations = list(pyr_config["pv_stations"].keys())
    else:
        stations = station_list

    year = info.split('_')[1]
    
    for station in stations:                
        #Read in cloudy sky LUT
        filename_stat = filename_cloudy + station + '.data'
        try:
            with open(os.path.join(pyrlut_folder_label_cloudy,filename_stat), 'rb') as filehandle:  
                # read the data as binary data stream
                pvstat, dummy, dummy = pd.read_pickle(filehandle)            
            
            pvstat[f"df_cod_pyr_{year}"] = pvstat.pop(f"df_cod_{year}")
            pv_systems[station] = merge_two_dicts(pv_systems[station], pvstat)
            
            print('Loading COD LUT for pyranometers at %s, %s' % (station,info))
        except IOError:
            print('There is no COD LUT simulation for pyranometers at %s in %s' % (station,info))                   
            
        #Read in clear sky LUT
        filename_stat = filename_clear + station + '.data'
        try:
            with open(os.path.join(pyrlut_folder_label_clear,filename_stat), 'rb') as filehandle:  
                # read the data as binary data stream
                pvstat, dummy = pd.read_pickle(filehandle)            
            
            pv_systems[station][f"df_clearsky_pyr_{year}"] = pvstat["df"]            
            
            print('Loading clear sky LUT for pyranometers at %s, %s' % (station,info))
        except IOError:
            print('There is no clear sky LUT simulation for pyranometers at %s in %s' % (station,info))                   
    
    results_path = os.path.join(home,pyr_config["results_path"]["main"],
                                pyr_config["results_path"]["optical_depth"])
    pyr_results_folder_label = os.path.join(results_path,folder_label_cloudy)
    
    #PV - changed the "senstype" to "pyranometer", since the spectral mismatch has been included
    #The parameter "senstype" is now obsolete
    folder_label_cloudy, filename, (theta_res,phi_res) = \
    generate_folder_names_sim_results(rt_config, "pyranometer", "cloudy")

    pv_folder_label_cloudy = os.path.join(mainpath_cloudy,folder_label_cloudy)       
    
    filename_cloudy = filename + info + '_'
    
    folder_label_clear, filename, (theta_res,phi_res) = \
    generate_folder_names_sim_results(rt_config, "pyranometer", "clear")

    pv_folder_label_clear = os.path.join(mainpath_clear,folder_label_clear)       
    
    filename_clear = filename + info + '_disortres_' + theta_res + '_' + phi_res + '_'
    
    #PV stations
    if station_list[0] == "all":
        stations = list(pv_config["pv_stations"].keys())
    else:
        stations = station_list
    
    for station in stations:                
        #Read in binary file that was saved from pvcal_radsim_disort
        filename_stat = filename_cloudy + station + '.data'
        
        try:
            with open(os.path.join(pv_folder_label_cloudy,filename_stat), 'rb') as filehandle:  
                # read the data as binary data stream
                pvstat, dummy, dummy = pd.read_pickle(filehandle)                                    
            
            if station not in pv_systems:
                pv_systems.update({station:pvstat})
            else:                
                if f"df_cod_{year}" in pvstat:
                    pvstat[f"df_cod_pv_{year}"] = pvstat.pop(f"df_cod_{year}")                                            
                    pv_systems[station] = merge_two_dicts(pv_systems[station],pvstat)                    
                                                
            print('Loading COD LUT for PV systems at %s, %s' % (station,info))
        except IOError:
            print('There is no COD LUT simulation for PV systems at %s in %s' % (station,info))
            
        #Read in clear sky LUT
        filename_stat = filename_clear + station + '.data'
        try:
            with open(os.path.join(pv_folder_label_clear,filename_stat), 'rb') as filehandle:  
                # read the data as binary data stream
                pvstat, dummy = pd.read_pickle(filehandle)            
            
            pv_systems[station][f"df_clearsky_pv_{year}"] = pvstat["df"]            
            
            print('Loading clear sky LUT for PV systems at %s, %s' % (station,info))
        except IOError:
            print('There is no clear sky LUT simulation for PV systems at %s in %s' % (station,info))     
            
    results_path = os.path.join(home,pv_config["results_path"]["main"],
                                pv_config["results_path"]["optical_depth"])
    pv_results_folder_label = os.path.join(results_path,folder_label_cloudy)
            
    return pv_systems, pyr_results_folder_label, pv_results_folder_label

def interpolate_COD_GTI(df_cod_old,df_clearsky_old,df_hires,timeres_new,substat,radnames):
    """
    Interpolate the COD lookuptable results onto a higher resolution
    
    args:
    :param df_cod_old: dataframe, low resolution data, COD LUT
    :param df_clearsky_old: dataframe, low resolution data, clear sky LUT
    :param df_hires: dataframe, high resolution data
    :param timeres_new: timedelta, new time resolution
    :param substat: string, name of substation
    :param radnames: list of strings of radiation names ("POA" or "down")
    
    out:
    :return dataframe with interpolated LUT results
    """
    
    days_lowres = pd.to_datetime(df_cod_old.index.date).unique().strftime('%Y-%m-%d')    
    days_hires = pd.to_datetime(df_hires.index.date).unique().strftime('%Y-%m-%d')    
    days = days_hires.intersection(days_lowres)
    
    cols = []
    radtypes = []
    for radname in radnames:
        if "poa" in radname:
            cols.extend(["k_index_poa","cloud_fraction_poa","error_Etotpoa_Wm2",
                         "Etotpoa_pv_inv","error_Etotpoa_pv_inv","Etotpoa_clear_Wm2",
                         "Etotpoa_pv_clear_Wm2","theta_IA"])
            radtypes.append("poa")
        elif "down" in radname:
            cols.extend(["k_index_down","cloud_fraction_down","error_Etotdown_Wm2",
                         "Etotdown_clear_Wm2"])
            radtypes.append("down")
    
    cols.extend(radnames)
    
    #Interpolate each day separately
    df_hires_new = []
    for day in days:
        #Low res data
        df_cod = df_cod_old.loc[day]        
        old_index_numeric = df_cod.index.values.astype(float)
                
        #High res index
        new_index = pd.date_range(start=df_cod.index[0],end=df_cod.index[-1],freq=timeres_new)
        new_index_numeric = new_index.values.astype(float)
        
        #Get required data from hi res dataframe
        df_hires_day = pd.concat([df_hires.loc[day,pd.IndexSlice[cols,substat]],
                                  df_hires.loc[day,pd.IndexSlice[:,'sun']],
                                  df_hires.loc[day,pd.IndexSlice[['Edirdown_Wm2',
                                  'Ediffdown_Wm2'],'cosmo']]],axis=1)
        
        df_hires_day = df_hires_day.loc[df_hires_day[('sza','sun')].notna()]        
        
        #Interpolate COD LUTs in time        
        for radtype in ["dir","diff"]:
            tablename = f'COD_{radtype}down_table'
            cod_old = np.stack(df_cod.loc[:,(tablename,"libradtran")].to_numpy())
            
            #interpolate
            f = interp1d(old_index_numeric,cod_old, kind='linear', axis=0)
            cod_new = f(new_index_numeric)
        
            #Add to series
            slice_data = [cod_new[j,:,:] for j in range(cod_new.shape[0])]
            df_hires_day[(tablename,"libradtran")] = pd.Series(index=new_index,data=slice_data)
            df_hires_day = df_hires_day.loc[df_hires_day[(tablename,"libradtran")].notna()]           
        
        for radtype in radtypes:
            tablename = f'COD_{radtype}_table'
            cod_old = np.stack(df_cod.loc[:,(tablename,substat)].to_numpy())
            
            #interpolate
            f = interp1d(old_index_numeric,cod_old, kind='linear', axis=0)
            cod_new = f(new_index_numeric)
        
            #Add to series
            slice_data = [cod_new[j,:,:] for j in range(cod_new.shape[0])]
            df_hires_day[(tablename,substat)] = pd.Series(index=new_index,data=slice_data)
            df_hires_day = df_hires_day.loc[df_hires_day[(tablename,substat)].notna()]        
                            
        #Clear sky part
        #Low res data
        df_clear = df_clearsky_old.loc[df_cod.index]        
        old_index_numeric = df_clear.index.values.astype(float)
        
        #Interpolate clearsky simulation in time
        for radtype in ["dir","diff"]:
            radname = f'E{radtype}down_clear'
            rad_clearsky_old = np.stack(df_clear.loc[:,(radname,"libradtran")].to_numpy())
            
            #interpolate
            f = interp1d(old_index_numeric,rad_clearsky_old, kind='linear', axis=0)
            rad_clearsky_new = f(new_index_numeric)
        
            #Add to series            
            df_hires_day[(f"{radname}","libradtran")] = pd.Series(index=new_index,data=rad_clearsky_new)
            df_hires_day = df_hires_day.loc[df_hires_day[(f"{radname}","libradtran")].notna()]        
        
        df_hires_new.append(df_hires_day)
        
    df_output = pd.concat(df_hires_new,axis=0)
    
    return df_output

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
    #               nan_treatment='interpolate') - avg**2),index=input_series.index,name='std_conv')
    edge = int(window_size/2.)
    # avg  = avg[edge:-edge]
    # std  = std[edge:-edge]
    
    #alternative method with pandas
    
    if center:
        shift = -edge
    else: shift = 0
    
    avg_alt = input_series.interpolate(method='linear',limit=int(edge/2)).\
        rolling(window=window_avg,min_periods=edge).\
            mean().shift(shift).rename('avg_pd')        
    std_alt = input_series.interpolate(method='linear',limit=int(edge/2)).\
        rolling(window=window_avg,min_periods=edge).\
        std().shift(shift).rename('std_pd')
    
    dataframe = pd.concat([avg_alt,std_alt],axis=1) #avg,std,
    
    return dataframe

def average_cloud_fraction(dataframe,timeres_old,timeres_window,substat,radtypes):
    """
    Average the cloud fraction over a certain period of time, for each day
    
    args:
    :param dataframe: dataframe with cloud fraction and other parameters
    :param timeres_old: string with old time resolution (high resolution)
    :param timeres_window: string with size of window for moving average
    :param substat: string with name of substation
    :param radtypes: list of strings, either tilted (poa) or downward (down)
    
    out:
    return: dataframe with average cloud fraction added
    """
    
    timeres_old_secs = pd.to_timedelta(timeres_old).seconds # measurement timeresolution in sec
    timeres_ave_secs = pd.to_timedelta(timeres_window).seconds
    kernelsize = timeres_ave_secs/timeres_old_secs # kernelsize 
    box_kernel = Box1DKernel(kernelsize)     
                
    days = pd.to_datetime(dataframe.index.date).unique().strftime('%Y-%m-%d')            
    for iday in days:                        
        for radtype in radtypes:
            df_day = deepcopy(dataframe.loc[iday,(f"cloud_fraction_{radtype}",substat)])
            df_day.loc[df_day < 0] = np.nan #0.5
            cf_avg = convolve(df_day.values.flatten(), box_kernel)
            # dE_avg = convolve(dataframe.loc[iday,(f"dE_overshoot_{radtype}_Wm2",substat)].values.flatten(), box_kernel,
            #                   preserve_nan=True)
                    
            # handle edges
            edge = int(kernelsize/2.)
            cf_avg  = cf_avg[edge:-edge]
            #dE_avg  = dE_avg[edge:-edge]
            index_cut = dataframe.loc[iday].index[edge:-edge]
            dataframe.loc[index_cut,(f"cf_{radtype}_{timeres_window}_avg",substat)] = cf_avg                
            #dataframe.loc[index_cut,(f"dE_overshoot_{radtype}_{timeres_window}_avg",substat)] = dE_avg                
            
            df_avg = moving_average_std(df_day, pd.Timedelta(timeres_old), pd.Timedelta(timeres_window), center = True)
            
            dataframe.loc[iday,(f"cf_{radtype}_{timeres_window}_avg_alt",substat)] = df_avg["avg_pd"]
            
    #Sort multi-index (makes it faster)
    dataframe.sort_index(axis=1,level=1,inplace=True)
    
    return dataframe
  
def extract_cod_irradiance_comps_from_lut(name,pv_station,namecoddata,nameclearskydata,
                                  namemeasdata,rt_config,substat_pars,
                                  angles,const_opt,year,window_avg_cf,
                                  path,flags):
    """
    
    Extract the COD from the look-up-table using interpolation, with two different options:
        
        1. real_cod --> Etotpoa(meas) = Etotpoa_sim_cloudy, only when cf = 1
        2. effective_cod --> Etotpoa(meas) = cf*(Etotpoa_sim_cloudy) + (1 - cf)*(Etotpoa_sim_clear)
    
    Once COD has been find, turn the LUT around to find the irradiance components

    Parameters
    ----------
    name : string, name of PV station
    pv_station : dictioanry with information and data from PV station
    namecoddata : string, name of COD dataframe
    nameclearskydata : string, name of clear sky dataframe
    namemeasdata : string, name of dataframe with measured data
    rt_config : dictionary with radiative transfer configuration
    substat_pars : dictionary with substation information
    angles : named tuple with angles for DISORT simulation
    const_opt : named tuple with optical constants
    year : string with year under consideration
    window_avg_cf : string with width of averaging window for cloud fraction
    path : string with path for saving results and plots
    flags : dictionary of booleans for plotting

    Returns
    -------
    df_merge : dataframe with COD results merged with other data

    """    
    
    df_cod = pv_station[namecoddata]   
    df_clearsky = pv_station[nameclearskydata]
    
    #This is a quick fix - should have the "clear" qualifier right from the start!!
    for col in df_clearsky.columns.levels[0]:
        if col in ["Edirdown","Ediffdown"]:
            df_clearsky.rename(columns={col:f"{col}_clear"},level='variable',inplace=True)    
            
    timeres_sim = rt_config["timeres"]    
    df_merge = pd.DataFrame()
    for substat in substat_pars:
        timeres = substat_pars[substat]["t_res_inv"]        
        
        #Get radnames
        radname = substat_pars[substat]["name"]                    
        if "pyr" in radname:
            radnames = [radname,radname.replace('poa','down')]            
        else:
            radnames = [radname]
            
        #If there is no data for a specific radname remove it
        for radname in radnames:
            if (radname,substat) not in pv_station[namemeasdata].columns:
                radnames.remove(radname)
                print(f"There is no data for {radname}")                
                
        radtypes = []
        newcols = []
        for radname in radnames:
            if "poa" in radname:
                radtypes.append("poa")
                newcols.extend(["COD_550_poa_inv","error_COD_550_poa_inv"])
            elif "down" in radname:
                radtypes.append("down")
                newcols.extend(["COD_550_down_inv","error_COD_550_down_inv"])                        
        
        if "opt_pars" in substat_pars[substat]:
            irrad_pars = [substat_pars[substat]["opt_pars"][i][1] for i in range(len(substat_pars[substat]["opt_pars"]))]        
        else:
            irrad_pars = [substat_pars[substat]["ap_pars"][i][1] for i in range(len(substat_pars[substat]["ap_pars"]))]        
        
        #The first step is to calculate the tilted components for the 15 minute lookup table
        for radtype in radtypes:
            print(f"Combining direct and diffuse components to get GTI LUT for {substat}, {year} using {radtype} irradiance")                                                
            
            #Calculate irradiance on the plane of array to get 15 minute GTI - COD LUT
            sun_pos = sun_position(np.deg2rad(df_cod[('sza','sun')].values),
                               np.deg2rad(df_cod[('phi0','sun')].values))   
            Edirdown_cloudy = np.stack(df_cod[("COD_dirdown_table","libradtran")].to_numpy())[:,:,1]
            Ediffdown_cloudy = np.stack(df_cod[("COD_diffdown_table","libradtran")].to_numpy())[:,:,1]
            #This is the COD array
            OD_array = np.stack(df_cod[("COD_dirdown_table","libradtran")].to_numpy())[:,:,0]
            
            if radtype == "poa":
                radname = substat_pars[substat]["name"]
                
                #Calculate POA irradiance                                            
                Idiff_field_cloudy = np.stack(df_cod[("COD_diff_field_table","libradtran")].to_numpy())
                if "pv" not in radname: # In this case there is no optical model
                    irrad_cloudy = E_poa_calc(irrad_pars,Edirdown_cloudy,Idiff_field_cloudy,sun_pos,angles,
                       const_opt,None,False)            
    
                    F_array = irrad_cloudy["Etotpoa"]   

                    #irrad_clear = df_cod_substat_day.loc[(f"Etot{radtype}_clear_Wm2",substat)]                                     
                else: #turn on optical model, i.e. glass surface
                    irrad_cloudy = E_poa_calc(irrad_pars,Edirdown_cloudy,Idiff_field_cloudy,sun_pos,angles,
                       const_opt,None,True)            
    
                    F_array = irrad_cloudy["Etotpoa_pv"]                                        
                    
                    #irrad_clear = df_cod_substat_day.loc[("Etotpoa_pv_clear_Wm2",substat)]                                     
                    
            elif radtype == "down":
                radname = substat_pars[substat]["name"].replace("poa","down")            
                #Calculate downward irradiance                    
                F_array = Edirdown_cloudy + Ediffdown_cloudy                                                
                            
            #Save tilted LUT to dataframe
            df_cod[(f"COD_{radtype}_table",substat)] = pd.Series([np.transpose(np.array([OD,F_array[i]])) 
                                  for i, OD in enumerate(OD_array)],index=df_cod.index)                  
            
        #Interpolate COD lookup table results onto higher resolution
        #This applies to both PV and pyranometer
        print(f"Interpolating COD LUT onto {timeres} resolution")
        if pd.to_timedelta(timeres) < pd.to_timedelta(timeres_sim):
            df_cod_substat = interpolate_COD_GTI(df_cod.loc[:,pd.IndexSlice[:,["libradtran",substat]]],
                             df_clearsky.loc[:,pd.IndexSlice[:,"libradtran"]],
                             pv_station[namemeasdata],timeres,substat,radnames)
        else: #This applies only to PV systems - use left merge in order to keep only cloudy days (left index)
            df_cod_substat = pd.merge(pd.merge(df_cod.loc[:,pd.IndexSlice[:,["sun","libradtran",substat]]],
                    df_clearsky.loc[:,pd.IndexSlice[:,"libradtran"]],how='left',left_index=True,right_index=True),
                    pv_station[namemeasdata].loc[:,pd.IndexSlice[["Etotpoa_pv_inv",
                    "error_Etotpoa_pv_inv","k_index_poa","cloud_fraction_poa","Etotpoa_pv_clear_Wm2","theta_IA"],
                     substat]],how='left',left_index=True,right_index=True)
            
        #Average cloud fraction
        print(f'Calculating average cloud fraction over {window_avg_cf} from {substat} measurements')
        df_cod_substat = average_cloud_fraction(df_cod_substat,timeres,window_avg_cf,substat,radtypes)
        df_cod_substat.sort_index(axis=1,level=1,inplace=True)   
        
        #df_cod_substat = df_cod_substat.iloc[0:5,:]
        index_df = df_cod_substat.index                
        retrieval_df = pd.DataFrame(np.full((len(index_df), len(newcols)), 0.0), index=index_df, 
            columns=pd.MultiIndex.from_product([newcols,[substat]],names=['variable','substat']))   
    
        retrieval_df.sort_index(axis=1,level=1,inplace=True)   
        
        for i, radtype in enumerate(radtypes):
            radname = radnames[i]
            print(f"Extracting COD from LUT for {name}, {substat}, {year} using {radtype} irradiance at {timeres} resolution, this takes a while")
            #Calculate overshoots
            if "pv" not in radname:                                
                df_cod_substat[(f"dE_overshoot_{radtype}_Wm2",substat)] = \
                    df_cod_substat[(radname,substat)] - \
                        df_cod_substat[(f"Etot{radtype}_clear_Wm2",substat)]  
            else:
                df_cod_substat[(f"dE_overshoot_{radtype}_Wm2",substat)] = \
                    df_cod_substat[(radname,substat)] - \
                        df_cod_substat[(f"Etot{radtype}_pv_clear_Wm2",substat)]  
                        
            df_cod_substat.loc[df_cod_substat[(f"dE_overshoot_{radtype}_Wm2",substat)] < 0,\
                   (f"dE_overshoot_{radtype}_Wm2",substat)] = np.nan
            
            #Average cloud fraction
            #df_cod_substat = average_cloud_fraction_overshoot(df_cod_substat,timeres,window_avg_cf,substat,radtype)        
            #notnan_index = df_cod_substat.loc[df_cod_substat[(radname,substat)].notna()].index
            
            #Perform fit for each time step
            for time in df_cod_substat.index: #enumerate(dates):                
                
                df_cod_substat_day = df_cod_substat.loc[time]                                
                
                if df_cod_substat_day.loc[('sza','sun')] < rt_config["sza_max"]["cod_cutoff"]: #Added this due to problems with DISORT at high SZA
                    
                    #This is the COD array from the LUT
                    OD_array = df_cod_substat_day.loc[(f"COD_{radtype}_table",substat)][:,0]
                    
                    #This is the total irradiance from the LUT, calculated above via integration and interpolation
                    irrad_cloudy = df_cod_substat_day.loc[(f"COD_{radtype}_table",substat)][:,1]                    
                    
                    #These are the LUTs for direct and diffuse components
                    Edirdown_cloudy = df_cod_substat_day.loc[("COD_dirdown_table","libradtran")][:,1]
                    Ediffdown_cloudy = df_cod_substat_day.loc[("COD_diffdown_table","libradtran")][:,1]   
                    
                    #Get the clear component
                    if "pv" not in radname: # In this case there is no optical model
                        irrad_clear = df_cod_substat_day.loc[(f"Etot{radtype}_clear_Wm2",substat)]                                     
                    else: 
                        irrad_clear = df_cod_substat_day.loc[("Etotpoa_pv_clear_Wm2",substat)]   
                    
                    #Check the cloud fraction 
                    cf = df_cod_substat_day.loc[(f"cloud_fraction_{radtype}",substat)] 
                    cf_avg = df_cod_substat_day.loc[(f"cf_{radtype}_{window_avg_cf}_avg_alt",substat)]
                    
                    dE = df_cod_substat_day.loc[(f"dE_overshoot_{radtype}_Wm2",substat)]
                    #dE_avg = df_cod_substat_day.loc[(f"dE_overshoot_{radtype}_{window_avg_cf}_avg",substat)]
                                        
                    #Set up linear interpolation from cloud simulation - Irradiance - Optical depth
                    f_interp = interp1d(irrad_cloudy, OD_array,kind='linear',bounds_error=False,
                                        fill_value=np.nan)
                    
                    #1. This is the "real COD"
                    
                    #If we are under a cloud then find COD
                    if cf == 1:
                        #Only consider cloudy conditions
                        y_meas = df_cod_substat_day.loc[(radname,substat)]
                                                    
                        # use function to retrieve the COD that corresponds to a certain irradiance.                        
                        if not np.isnan(y_meas):                            
                            # COD, d_CODup, d_CODdown, errorcode = retrieve_by_fit_or_interpolation(X, Y, 
                            #                                   dY, y_meas, dy_meas, threshold=1.)
                            # d_COD = np.mean(np.array([d_CODup, d_CODdown], dtype=float))
                                                                                    
                            COD = f_interp(y_meas)
                        else:                            
                            COD = np.nan
                            
                        #Now use retrieved COD to find irradiance components!
                        if not np.isnan(COD): # and d_COD and not np.isnan(COD) and not np.isnan(d_COD):
                            g_interp = interp1d(OD_array,Edirdown_cloudy,kind='linear',bounds_error=False,
                                                fill_value=np.nan)
                            Edirdown = g_interp(COD)
                            if Edirdown < 1e-6:
                                Edirdown = 0.
                            
                            #diffuse component                                                        
                            h_interp = interp1d(OD_array,Ediffdown_cloudy,kind='linear',bounds_error=False,
                                                fill_value=np.nan)
                            Ediffdown = h_interp(COD)                                                                                            
                            
                        else:
                            Edirdown = np.nan                        
                            Ediffdown = np.nan                                
                    
                    #In this case if we are not under a cloud then COD is undefined
                    else:                        
                        COD = np.nan
                        Edirdown = np.nan
                        Ediffdown = np.nan
    
                    #2. This is the effective COD part
                    
                    #If there are enough clouds then find COD (using average cloud fraction)
                    if cf_avg != 0:
                        #Linear combination of clear and cloudy - remember that here optical model is switched on
                        if not np.isnan(dE): #Overshoots
                            y_meas_lin_comb = ((df_cod_substat_day.loc[(radname,substat)]) - \
                                (1. - cf_avg)*irrad_clear)/cf_avg - dE
                        else: #No overshoots
                            y_meas_lin_comb = ((df_cod_substat_day.loc[(radname,substat)]) - \
                                (1. - cf_avg)*irrad_clear)/cf_avg
                                                                                  
                        # use function to retrieve the COD that corresponds to a certain irradiance.                        
                        if not np.isnan(y_meas_lin_comb):
                            # prepare the fit
                            # X, Y = OD_array, F_array #df["COD"].to_numpy(), df[rad_key].to_numpy()
                            # dY = 0.02*Y
                            # #dy_meas = 0.05 * y_meas
                            # dy_meas = df_cod_substat_day.loc[(errorname, substat)]
                            #print(f"the measured {radtype} irradiance values for timestep {time} were {y_meas} +/- {dy_meas}")
                            # COD, d_CODup, d_CODdown, errorcode = retrieve_by_fit_or_interpolation(X, Y, 
                            #                                   dY, y_meas, dy_meas, threshold=1.)
                            # d_COD = np.mean(np.array([d_CODup, d_CODdown], dtype=float))
                                                                                    
                            COD_eff = f_interp(y_meas_lin_comb)
                        else:
                            # COD = np.nan
                            # d_COD = np.nan   
                            COD_eff = np.nan
                            
                        #Now use retrieved COD to find irradiance components!
                        if not np.isnan(COD_eff): # and d_COD and not np.isnan(COD) and not np.isnan(d_COD):
                            #direct component                            
                            g_interp = interp1d(OD_array,Edirdown_cloudy,kind='linear',bounds_error=False,
                                                fill_value=np.nan)
                            Edirdown_eff = g_interp(COD_eff)
                            if Edirdown_eff < 1e-6:
                                Edirdown_eff = 0.
                            
                            #diffuse component                                                        
                            h_interp = interp1d(OD_array,Ediffdown_cloudy,kind='linear',bounds_error=False,
                                                fill_value=np.nan)
                            Ediffdown_eff = h_interp(COD_eff)
                                                        
                            #Calculate the "effective irradiance components"
                            Edirdown_eff = cf_avg*Edirdown_eff + (1.-cf_avg)*\
                                df_cod_substat_day.loc[("Edirdown_clear","libradtran")]  
                                    
                            if not np.isnan(dE):
                                Ediffdown_eff = cf_avg*(Ediffdown_eff + dE) + (1.-cf_avg)*\
                                df_cod_substat_day.loc[("Ediffdown_clear","libradtran")]
                            else:
                                Ediffdown_eff = cf_avg*Ediffdown_eff + (1.-cf_avg)*\
                                df_cod_substat_day.loc[("Ediffdown_clear","libradtran")]                            
                        else:
                            Edirdown_eff = np.nan
                            #d_Edirdown = np.nan
                            Ediffdown_eff = np.nan
                            #d_Ediffdown = np.nan
                                                    
                    #In this case if there are no clouds (cf = 0) then COD = 0 and use clear sky simulation
                    else:
                        # COD = 0.
                        # d_COD = 0.  
                        COD_eff = 0.

                        #Take the clear sky components from the clear sky simulation, since cf = 0
                        Edirdown_eff =  df_cod_substat_day.loc[("Edirdown_clear","libradtran")]                                      
                        Ediffdown_eff =  df_cod_substat_day.loc[("Ediffdown_clear","libradtran")]                                                              
                    
                                                    
                else: #In this case the DISORT LUT is unreliable (high SZA)
                    # COD = np.nan
                    # d_COD = np.nan
                    COD = np.nan                                        
                    COD_eff = np.nan
                    
                    Edirdown = np.nan
                    Ediffdown = np.nan
                    Edirdown_eff = np.nan
                    Ediffdown_eff = np.nan
                #print("found OD-value: ",COD,"+-", d_COD)                        
        
                # write into retrieval_df                
                retrieval_df.at[time, (f"COD_550_{radtype}_inv",substat)] = COD
                retrieval_df.at[time, (f"COD_eff_550_{radtype}_inv",substat)] = COD_eff
                # retrieval_df.at[time, (f"COD_alt_550_{radtype}_inv",substat)] = COD_alt
                # retrieval_df.at[time,(f"error_COD_550_{radtype}_inv",substat)] = d_COD
                
                if Edirdown < 1e-6:
                    Edirdown = 0.
                if Ediffdown < 1e-6:
                    Ediffdown = 0.
                if Edirdown_eff < 1e-6:
                    Edirdown_eff = 0.
                if Ediffdown_eff < 1e-6:
                    Ediffdown_eff = 0.
                
                retrieval_df.at[time, (f"Edirdown_{radtype}_inv",substat)] = Edirdown
                retrieval_df.at[time, (f"Ediffdown_{radtype}_inv",substat)] = Ediffdown
                retrieval_df.at[time, (f"Edirdown_eff_{radtype}_inv",substat)] = Edirdown_eff
                retrieval_df.at[time, (f"Ediffdown_eff_{radtype}_inv",substat)] = Ediffdown_eff
                
            #Remove interpolated LUT values for saving to file (to save space)
            df_cod_substat.loc[df_cod_substat.index.difference(df_cod.index),(f"COD_{radtype}_table",substat)] = np.nan
            
        #Merge dataframes from one radtype
        if df_merge.empty:
            df_merge = pd.concat([df_cod_substat.drop(
                labels=['libradtran'],axis=1,level=1),retrieval_df],axis=1)
        else:
            df_merge = pd.concat([retrieval_df,df_merge,df_cod_substat.drop(
                labels=['libradtran','sun'],axis=1,level=1)],axis=1)    
            if 'cosmo' in df_merge.columns.levels[1]:
                df_merge.drop(labels=['cosmo'],axis=1,level=1,inplace=True)
        
        df_merge.sort_index(axis=1,level=1,inplace=True)        
        
    return df_merge

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


def save_results(name,pv_station,name_df,info,timeres,path):
    """
    

    Parameters
    ----------
    name : string, name of PV station
    pv_station : dictionary with information and data from PV station
    name_df : string with name of dataframe containing results
    info : string with description of current campaign    
    timeres : string with timeres of retrieval
    path : string with path for saving data

    Returns
    -------
    None.

    """
    
    pv_station_save = deepcopy(pv_station)
    
    filename_stat = f"cod_fit_results_{info}_{name}_{timeres}.data"        
    
    #List of dataframes and information to save
    dfnames = [name_df]    
         
    dfnames.append('lat_lon')
    
    for key in list(pv_station_save):
        if key not in dfnames and "substations" not in key and "path" not in key:
            del pv_station_save[key]
            
    pv_station_save['station_name'] = name
    pv_station_save['timeres'] = [timeres]           
    
    with open(os.path.join(path,filename_stat), 'wb') as filehandle:  
        # store the data as binary data stream
        pickle.dump(pv_station_save, filehandle)

    print('Results written to file %s\n' % filename_stat)
    
#%%Main Program
#######################################################################
###                         MAIN PROGRAM                            ###
#######################################################################
#This program takes the DISORT simulation for COD and uses an interpolation routine
#developed by Grabenstein to find the COD, plus creates both time series and scatter plots
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

#%%Preparation
sun_position = collections.namedtuple('sun_position','sza azimuth')
disort_res = rt_config["clouds"]["disort_rad_res"]
grid_dict = define_disort_grid(disort_res)

angle_grid = collections.namedtuple('angle_grid', 'theta phi umu')
angle_arrays = angle_grid(grid_dict["theta"],np.deg2rad(grid_dict["phi"]),grid_dict["umu"])

#%%Load LUT + inversion results and interpolate to find COD
for campaign in campaigns:
    year = "mk_" + campaign.split('_')[1]    
    #Load pyranometer configuration
    
    pyr_config = load_yaml_configfile(config["pyrcalod_configfile"][year])
    
    if args.station:
        stations = args.station
        if stations[0] == 'all':
            stations = 'all'
    else:
        #Stations for which to perform inversion
        stations = "MS_02" #pyr_config["stations"]

    opt_dict = pyr_config["optics"]
    optics = collections.namedtuple('optics', 'kappa L')
    const_opt = optics(opt_dict["kappa"],opt_dict["L"])

    #Reset dictionary between campaigns    
    pvsys = {}
    #Load inversion results
    print('Loading PYRCAL and PV2POARAD results')
    pvsys = load_pyr2cf_pv2poarad_results(rt_config,pyr_config,pvcal_config,pvrad_config,campaign,stations,homepath)
        
    print('Loading PV2COD and PYR2COD look up table from DISORT simulation')
    pvsys, pyr_results_folder, pv_results_folder = \
        load_pv_pyr2cod_lut_results(pvsys, rt_config, pyr_config, 
                                    pvrad_config, campaign, stations, homepath)
    
    window_avg_cf = pyr_config["cloud_fraction"]["cf_avg_window"]
    cod_method = pyr_config["cod_fit_method"]
    
    #Perform interpolation for each system
    for key in pvsys:                       
        print(f'Calculating COD and irradiance components for {key}, {campaign}')
        tres_list = []
        #Pyranometers
        name_df_codlut = f"df_cod_pyr_{year.split('_')[-1]}"
        name_df_clearsky = f"df_clearsky_pyr_{year.split('_')[-1]}"
        
        if key in pyr_config["pv_stations"]:
            for substat in pyr_config["pv_stations"][key]['substat']:    
                pyr_config["pv_stations"][key]['substat'][substat] = \
                merge_two_dicts(pyr_config["pv_stations"][key]['substat'][substat],
                            pvsys[key]["substations_pyr"][substat])                        
            
            for val in pvsys[key]:
                if "df_pyr_" in val:
                    name_df_hires = val
            
            pyr_substat_dict = pyr_config["pv_stations"][key]["substat"]                                    
                        
            for substat in pyr_substat_dict:            
                timeres = pyr_substat_dict[substat]["t_res_inv"]
                if timeres not in tres_list:
                    tres_list.append(timeres)
                
            for timeres in tres_list:
                name_df_cod_result = f"df_cod_pyr_{year.split('_')[-1]}_{timeres}"
            
                #Use LUT to extract COD using Johannes method plus the new method with linear combination
                print(f"Extract COD from pyranometer measurements at {timeres} resolution")
                pvsys[key][name_df_cod_result] = extract_cod_irradiance_comps_from_lut(key,pvsys[key],
                                    name_df_codlut,name_df_clearsky,name_df_hires,rt_config,pyr_substat_dict,
                                    angle_arrays,const_opt,year.split('_')[-1],window_avg_cf,
                                    pyr_results_folder,plot_flags)
                #del pvsys[key][name_df_hires]
            
        #PV Systems
        name_df_codlut = "df_cod_pv_" + year.split('_')[-1]    
        name_df_clearsky = f"df_clearsky_pv_{year.split('_')[-1]}"
        
        if key in pvrad_config["pv_stations"]:
            for substat_type in pvrad_config["pv_stations"][key]:                
                timeres = pvrad_config["pv_stations"][key][substat_type]["t_res_inv"]
                if timeres not in tres_list:
                    tres_list.append(timeres)
                    
                for substat in pvrad_config['pv_stations'][key][substat_type]["data"]:                
                    pvsys[key]['substations_pv'][substat_type]["data"][substat].\
                    update({"name":"Etotpoa_pv_inv"})                
                    pvsys[key]['substations_pv'][substat_type]["data"][substat].\
                        update({"t_res_inv":timeres})
                    
                    pvrad_config["pv_stations"][key][substat_type]["data"]\
                    [substat] = merge_two_dicts(pvrad_config["pv_stations"][key][substat_type]["data"]\
                    [substat],pvsys[key]['substations_pv'][substat_type]["data"][substat])
                    
                
                pv_substat_dict = pvrad_config["pv_stations"][key][substat_type]["data"]
                #pv_substat_dict.update({"name":"Etotpoa_pv_inv"})                
                
                name_df_cod_result = "df_cod_pv_" + year.split('_')[-1] + '_' + timeres
                for val in pvsys[key]:
                    if "df_pv_" in val and timeres in val:
                        name_df_hires = val
                        
                window_avg_cf = pvrad_config["cloud_fraction"]["cf_avg_window"]
                
                if name_df_codlut in pvsys[key] and year in pvrad_config["pv_stations"][key][substat_type]["source"]:                                        
                    print(f"Extract COD from PV measurements at {timeres} resolution")
                    #Use LUT to extract cod
                    pvsys[key][name_df_cod_result] = extract_cod_irradiance_comps_from_lut(key,pvsys[key],name_df_codlut,
                                    name_df_clearsky,name_df_hires,rt_config,pv_substat_dict,angle_arrays,const_opt,
                                    year.split('_')[-1],window_avg_cf,pv_results_folder,plot_flags)                                                            
                    #del pvsys[key][name_df_hires]
                else:
                    print(f"No {timeres} data for {key}, {substat_type} in {year.split('_')[-1]}")
                    if not pyr_substat_dict:
                        tres_list.remove(timeres)
                    elif timeres == "15min":
                        tres_list.remove(timeres)
    
        #Save results depending on time resolution
        for timeres in tres_list:
            name_df_combine = "df_codfit_pyr_pv_" + year.split('_')[-1] + '_' + timeres                    
        
            #Merge data into one dataframe
            if f"df_cod_pyr_{year.split('_')[-1]}_{timeres}" in pvsys[key]\
            and f"df_cod_pv_{year.split('_')[-1]}_{timeres}" in pvsys[key]:
                df_cod_merge = pd.merge(pvsys[key][f"df_cod_pyr_{year.split('_')[-1]}_{timeres}"].loc[:,
                           pd.IndexSlice[:,[*pyr_substat_dict.keys()]]],
                           pvsys[key][f"df_cod_pv_{year.split('_')[-1]}_{timeres}"],
                           left_index=True,right_index=True) 
                del pvsys[key][f"df_cod_pyr_{year.split('_')[-1]}_{timeres}"]                                                              
                del pvsys[key][f"df_cod_pv_{year.split('_')[-1]}_{timeres}"]                                                              
                
            elif f"df_cod_pyr_{year.split('_')[-1]}_{timeres}" in pvsys[key]:
                df_cod_merge = pvsys[key][f"df_cod_pyr_{year.split('_')[-1]}_{timeres}"]   
                del pvsys[key][f"df_cod_pyr_{year.split('_')[-1]}_{timeres}"]                                                                                                                                                   
                
            elif f"df_cod_pv_{year.split('_')[-1]}_{timeres}" in pvsys[key]:
                df_cod_merge = pvsys[key][f"df_cod_pv_{year.split('_')[-1]}_{timeres}"]                                                                        
                del pvsys[key][f"df_cod_pv_{year.split('_')[-1]}_{timeres}"]                                                              

            pvsys[key][name_df_combine] = df_cod_merge 
            
            #Save results
            results_path = generate_folders(rt_config,pvcal_config,pvrad_config,homepath)
            save_results(key,pvsys[key],name_df_combine,campaign,
                      timeres,results_path)
    
    