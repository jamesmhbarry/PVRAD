#!/usr/bin/env python3
# -*- aoding: utf-8 -*-
"""
Created on Wed Oct 28 11:26:58 2020

@author: james
"""

#%% Preamble
import os
import pickle
import numpy as np
import collections
import pandas as pd
from copy import deepcopy

from pvcal_forward_model import azi_shift
from file_handling_functions import *
from data_process_functions import downsample
from gti2ghi_lookup import GTI2GHI

from astropy.convolution import convolve, Box1DKernel
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import host_subplot

import seaborn as sns

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

def generate_folder_names_poarad(rt_config,pvcal_config):
    """
    Generate folder structure to retrieve POA Rad results
    
    args:    
    :param rt_config: dictionary with RT configuration
    :param pvcal_config: dictionary with PVCAL configuration
    
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

def load_pyr2cf_pv2poarad_results(rt_config,pyr_config,pvcal_config,pvrad_config,info,station_list,home):
    """
    Load results from pyranometer calibration and cloud fraction calculation
    As well as PV to POA irradiance inversion and cloud fraction
    
    args:    
    :param rt_config: dictionary with current RT configuration
    :param pyr_config: dictionary with current RT configuration
    :param pvcal_config: dictionary with current calibration configuration    
    :param pvrad_config: dictionry with pv2rad configuration
    :param info: string with name of campaign
    :param station_list: list of PV stations
    :param home: string with homepath
    
    out:    
    :return pv_systems: dictionary of PV systems with data    
    :return station_list: list of stations
    :return pyr_folder_label: string with path for pyranometer results
    :return pv_folder_label: string with path for PV results
    """
    
    mainpath = os.path.join(home,pyr_config['results_path']["main"],
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
        if station_list[0] == "all":
            station_list = list(pyr_config["pv_stations"].keys())
            station_list.extend(list(pvrad_config["pv_stations"].keys()))
            station_list = list(set(station_list))
            station_list.sort()     
            #station_list = pvrad_config["pv_stations"]
    
    for station in station_list:                
        #Read in binary file that was saved from pvcal_radsim_disort
        filename_stat = filename + station + '.data'
        try:
            with open(os.path.join(mainpath,folder_label,filename_stat), 'rb') as filehandle:  
                # read the data as binary data stream
                (pvstat, dummy, dummy) = pd.read_pickle(filehandle)            
            
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
            
            print('Data for %s loaded from %s, %s' % (station,folder_label,filename))
        except IOError:
            print('There are no irradiance data for %s' % station)                   
    
    results_path = os.path.join(home,pyr_config['results_path']["main"],
                                pyr_config["results_path"]["irradiance"])
    pyr_folder_label = os.path.join(results_path,folder_label,'Pyranometer')    
    
    mainpath = os.path.join(home,pvrad_config['results_path']["main"],
                            pvrad_config['results_path']['inversion'])
    
    #Generate folder structure for loading files
    folder_label, filename, (theta_res,phi_res) = \
    generate_folder_names_poarad(rt_config,pvcal_config)    
    
    #Check calibration source for filename    
    if len(pvrad_config["calibration_source"]) > 1:
        infos = '_'.join(pvrad_config["calibration_source"])
    else:
        infos = pvrad_config["calibration_source"][0]
    
    filename = filename + infos + '_disortres_' + theta_res + '_' + phi_res + '_'        
    
    for station in station_list:                
        #Read in binary file that was saved from pvcal_radsim_disort
        filename_stat = filename + station + '.data'
        try:
            with open(os.path.join(mainpath,folder_label,filename_stat), 'rb') as filehandle:  
                # read the data as binary data stream
                (pvstat, rt_config, pvcal_config, dummy) = pd.read_pickle(filehandle)            
            
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
            
            print('Data for %s loaded from %s, %s' % (station,folder_label,filename))
        except IOError:
            print('There is no simulation for %s at %s' % (station,folder_label))   
            
    results_path = os.path.join(home,pvrad_config["results_path"]["main"],
                                pvrad_config["results_path"]["irradiance"])
    pv_folder_label = os.path.join(results_path,folder_label)    
    
    return pv_systems, station_list, pyr_folder_label, pv_folder_label

def average_cloud_fraction(dataframe,timeres_old,timeres_window,substat):
    """
    Average the cloud fraction over a certain period of time, for each day
    
    args:
    :param dataframe: dataframe with cloud fraction and other parameters
    :param timeres_old: string with old time resolution (high resolution)
    :param timeres_window: string with size of window for moving average
    :param substat: string with name of substation
    
    out:
    return: dataframe with average cloud fraction
    """
    
    timeres_old = pd.to_timedelta(timeres_old).seconds # measurement timeresolution in sec
    timeres_ave = pd.to_timedelta(timeres_window).seconds
    kernelsize = timeres_ave/timeres_old # kernelsize 
    box_kernel = Box1DKernel(kernelsize)     
    
    colnames = dataframe.xs(substat,level="substat",axis=1).columns
    radtypes = []
    if "cloud_fraction_down" in colnames:
        radtypes.append("down")
    if "cloud_fraction_poa" in colnames:
        radtypes.append("poa")        
        
    days = pd.to_datetime(dataframe.index.date).unique().strftime('%Y-%m-%d')    
    dfs = []
    for iday in days:
        #df_day = dataframe.loc[iday]        
        
        for radtype in radtypes:
            df_day = deepcopy(dataframe.loc[iday,(f"cloud_fraction_{radtype}",substat)])
            df_day.loc[df_day < 0] = np.nan #0.5
            cf_avg = convolve(df_day.values.flatten(), box_kernel)
                    
            # handle edges
            edge = int(kernelsize/2.)
            cf_avg  = cf_avg[edge:-edge]
            index_cut = dataframe.loc[iday].index[edge:-edge]
            dataframe.loc[index_cut,(f"cf_{radtype}_{timeres_window}_avg",substat)] = cf_avg                            
        
        dfs.append(dataframe.loc[iday])
    
    df_ave_cf = pd.concat(dfs,axis=0)
    
    #Sort multi-index (makes it faster)
    df_ave_cf.sort_index(axis=1,level=1,inplace=True)
    
    return df_ave_cf

# def apply_spectral_mismatch(ghi_pv,df_fit,n_h2o_cosmo_mm,cos_sza):
#     """
    

#     Parameters
#     ----------
#     ghi_pv : TYPE
#         DESCRIPTION.
#     df_fit : TYPE
#         DESCRIPTION.
#     n_h2o_cosmo_mm : TYPE
#         DESCRIPTION.

#     Returns
#     -------
#     None.

#     """
    
#     fitfunc = lambda x, w, c: x[0] - x[1] * w**3 / c - x[2] * w**2 / c - x[3] * w / c
    
#     f0_interp = interp1d(df_fit["cos_sza"],df_fit["x0"],kind='quadratic',
#                          bounds_error=False,fill_value="extrapolate")

#     f123_interp = interp1d(df_fit["cos_sza"],df_fit[["x1","x2","x3"]],kind='linear',
#                            axis=0,bounds_error=False)
    
#     x0 = f0_interp(cos_sza)
#     x123 = f123_interp(cos_sza)
    
#     x_vec = np.column_stack((x0,x123))
#     ghi_ratio = np.zeros(len(ghi_pv))
#     for i, ghi in enumerate(ghi_pv):
#         ghi_ratio[i] = np.exp(fitfunc(x_vec[i],n_h2o_cosmo_mm[i],cos_sza[i]))

#     return ghi_pv / ghi_ratio

# def load_spectral_mismatch_fit(config,home):
#     """
#     Load dataframe for spectral mismatch fit

#     Parameters
#     ----------
#     config : TYPE
#         DESCRIPTION.

#     Returns
#     -------
#     None.

#     """
    
#     folder = os.path.join(home,config["spectral_mismatch_lut"]["clear_sky"])
    
#     file = list_files(folder)[0]
    
#     df = pd.read_csv(os.path.join(folder,file),sep=',',comment='#')
    
#     return df

def ghifromgti(sens_type,dataframe,substat,stat_pars,radname,timeres_cf,
               wvl_range,lut_config,home):
    """
    
    Get GHI from GTI using MYSTIC lookup table
    
    Parameters
    ----------
    sens_type : string, PV or pyranometer
    dataframe : dataframe with irradiance data
    substat : string, name of substation
    stat_pars : dictionary with configuration of substation
    radname : string with name of irradiance variable
    timeres_cf: string with averaging window for cloud fraction calculation
    wvl_range: string with wavelength range of irradiance data
    lut_config : dictionary with configuration for LUT
    home : string with home path

    Returns
    -------
    df_return : dataframe with GHI
    nameghi : string with name of GHI variable

    """
    
    # if sens_type == "pv" and wvl_range != "all":
    #     df_spectral_fit = load_spectral_mismatch_fit(lut_config,home)        
    
    #Configure path to LUT, define LUT object
    fname_lut = os.path.join(home,lut_config["fname"])
    gti_converter = GTI2GHI(fname_lut)
    
    #Tilt
    tilt_sensor = np.rad2deg(stat_pars[0][1])
    
    splitstring = radname.split('_')
    nameghi = splitstring[0].replace("poa","down") + '_lut_Wm2' # + splitstring[-1]
    
    if tilt_sensor > lut_config_pyr["tilt_lims"][0] and\
        tilt_sensor < lut_config_pyr["tilt_lims"][1]:
        
        #Calculate relative azimuth angle
        rel_azi = np.abs(np.fmod(dataframe[("phi0","sun")]+180,360) - 
                         np.rad2deg(azi_shift(stat_pars[1][1])))
        
        #Define masks to make sure LUT works
        mask_azi = rel_azi <= lut_config["rel_azi_max"]
        mask_sza = (dataframe[("sza","sun")] >= lut_config["sza_lims"][0]) &\
                    (dataframe[("sza","sun")] <= lut_config["sza_lims"][1])    
        mask_cf = (dataframe[(f"cf_poa_{timeres_cf}_avg",substat)] >= lut_config["cf_lims"][0]) &\
                    (dataframe[(f"cf_poa_{timeres_cf}_avg",substat)] <= lut_config["cf_lims"][1])    
            
        #Combine masks                
        mask = mask_azi & mask_sza & mask_cf
        
        #Mask for relative azimuth
        rel_azi_filt = rel_azi[mask]        
        tilt_array = np.ones_like(rel_azi_filt) * tilt_sensor
        
        #Get data into list of tuples for LUT
        data = list(zip(dataframe.loc[mask,("sza","sun")], rel_azi_filt, tilt_array, 
                        dataframe.loc[mask,(f"cf_poa_{timeres_cf}_avg",substat)]*100))
        
        #Get GHI from LUT                 
        # if sens_type == "pv" and wvl_range != "all": #In this case apply optical and spectral correction
        #     print("Applying spectral correction")    
        #     ghi = gti_converter.get_ghi(data,dataframe.loc[mask,(radname,substat)]/
        #                                 dataframe.loc[mask,("tau_pv",substat)])    
        #     n_h2o = dataframe.loc[mask,("n_h2o_mm","cosmo")]
        #     cos_sza = np.cos(np.deg2rad(dataframe.loc[mask,("sza","sun")]))            
        #     ghi = apply_spectral_mismatch(ghi,df_spectral_fit,n_h2o,cos_sza)
        # else:
        # if sens_type == "pv":
        #     #Here we need to invert the optical model
        #     ghi = gti_converter.get_ghi(data,dataframe.loc[mask,(radname,substat)]/\
        #                                  dataframe.loc[mask,("tau_pv",substat)])    
        # else:
        ghi = gti_converter.get_ghi(data,dataframe.loc[mask,(radname,substat)])
        
        dataframe.loc[mask,(nameghi,substat)] = ghi

    else:
        print(f'Sensor tilt of {substat} is out of allowed range for LUT')
        dataframe[(nameghi,substat)] = np.nan        
        
    #Extract GHI column as dataframe
    df_return = dataframe #[[(nameghi,substat)]]
    
    return df_return, nameghi

def downsample_pyranometer_data(dataframe,substat,timeres_old,timeres_new,
                                poaradname,pyrdown):
    """
    Downsample pyranometer data to coarser resolution    

    Parameters
    ----------
    dataframe : dataframe with high resolution data
    substat : string, name of substation
    timeres_old : string, old time resolution
    timeres_new : string, desired time resolution
    poaradname : string, name of irradiance
    pyrname : string, name of pyranometer for downward irradiance validation

    Returns
    -------
    dataframe with downsample pyranometer data

    """
    
    #Convert time resolutions
    timeres_old = pd.to_timedelta(timeres_old)
    timeres_new = pd.to_timedelta(timeres_new)
    
    if "Pyr" in substat:
        downref = ["pyr",substat]
    elif "CMP11" in substat:
        downref = ["CMP11","CMP11_Horiz"]
    elif "SiRef" in substat:
        downref = ["CMP11","CMP11_Horiz"]
    elif "RT1" in substat:
        downref = ["pyr","Pyr055"]
    elif "auew" in substat or "egrid" in substat:        
        if "Pyr" in pyrdown:
            downref = ["pyr",pyrdown]
        elif "CMP11" in pyrdown:
            downref = ["CMP11",pyrdown]
    
    #Get correct names for irradiance columns
    if "Pyr" in substat:
        radnames = [poaradname,poaradname.replace('poa','down')]
    else:
        radnames = [poaradname]
        
    for radname in radnames:
        if (radname,substat) not in dataframe.columns:
            radnames.remove(radname)
            print(f"There is no data for {radname}")
        
    error_names = ['_'.join(["error",radname.split('_')[0],radname.split('_')[2]]) for radname in radnames]  
    if "pv" not in poaradname:
        radnames_clear = ['_'.join([radname.split('_')[0],'clear',radname.split('_')[2]]) for radname in radnames]
    else:
        radnames_clear = ['_'.join([radname.split('_')[0],radname.split('_')[1],
                                    'clear',"Wm2"]) for radname in radnames]
    
    #Flatten list
    radnames_all = [y for x in [radnames,error_names,radnames_clear] for y in x]
    
    if "Etotdown_lut_Wm2" in dataframe.columns.levels[0]:
        radnames_all.append("Etotdown_lut_Wm2")
        
    #Select correct columns
    df_old = dataframe.loc[:,pd.IndexSlice[radnames_all,substat]]
    
    if "egrid" in substat and downref[1] != substat:
        df_old = pd.concat([df_old,dataframe[f"Etotdown_{downref[0]}_Wm2",downref[1]],
                            dataframe["cloud_fraction_down",downref[1]]],axis=1)
    
    dfs_rs = []
    for day in pd.to_datetime(df_old.index.date).unique().strftime('%Y-%m-%d'):
        #Downsample data
        dfs_rs.append(downsample(df_old.loc[day], timeres_old, timeres_new))
    
    df_rs = pd.concat(dfs_rs,axis=0)
    
    #Calculate new cloud fraction for cloud mask
    for i, radname in enumerate(radnames):
        if "poa" in radname:
            df_rs[("k_index_poa",substat)] = df_rs[(radname,substat)]/\
                                            df_rs[radnames_clear[i],substat]
            
            df_rs[("cloud_fraction_poa",substat)] = \
                1. - ((0.8 < df_rs[("k_index_poa",substat)]) &
                      (df_rs[("k_index_poa",substat)] < 1.4)).astype(float)  
                
            df_rs[("cloud_fraction_poa",substat)].loc\
                [df_rs[("k_index_poa",substat)].isna()] = np.nan
                
        elif "down" in radname:
            df_rs[("k_index_down",substat)] = df_rs[(radname,substat)]/\
                                            df_rs[radnames_clear[i],substat]
                                            
            df_rs[("cloud_fraction_down",substat)] = \
                1. - ((0.8 < df_rs[("k_index_down",substat)]) &
                      (df_rs[("k_index_down",substat)] < 1.4)).astype(float)  
            
            df_rs[("cloud_fraction_down",substat)].loc\
                [df_rs[("k_index_down",substat)].isna()] = np.nan
    
    return df_rs

def plot_ghi_comparison_day(name,dataframe,timeres,df_lowres,timeres_low,substat,substat_pars,timeres_cf,path,
                            dict_paths):
    """
    

    Parameters
    ----------
    name : string, name of PV station
    dataframe : dataframe with relevant irradiance data
    timeres : string, time resolution of data
    df_lowres : dataframe with low resolution data
    timeres_low : string, time resolution of low res data
    substat : string, name of substaiton
    substat_pars : dictionary with substation parameters
    timeres_cf : string with width of averaging window for cloud fraction
    path : string with main path for saving plots
    dict_paths : dictionary of paths for plots

    Returns
    -------
    None.

    """
    
    plt.ioff()
    plt.close("all")
    plt.style.use("my_paper")
    
    res_dirs = list_dirs(path)
    savepath = os.path.join(path,dict_paths["ghi"])
    if dict_paths["ghi"] not in res_dirs:
        os.mkdir(savepath)    
        
    stat_dirs = list_dirs(savepath)
    savepath = os.path.join(savepath,name)
    if name not in stat_dirs:
        os.mkdir(savepath)
    
    res_dirs = list_dirs(savepath)
    savepath = os.path.join(savepath,'LUT')        
    if 'LUT' not in res_dirs:
        os.mkdir(savepath)
        
    substat_dirs = list_dirs(savepath)
    savepath = os.path.join(savepath,substat)
    if substat not in substat_dirs:
        os.mkdir(savepath)

    if "Pyr" in substat or "auew" in substat or "egrid" in substat:
        downref = ["pyr",substat]
    elif "CMP11" in substat:
        downref = ["CMP11","CMP11_Horiz"]
    elif "SiRef" in substat:
        downref = ["CMP11","CMP11_Horiz"]
    elif "RT1" in substat:
        downref = ["pyr","Pyr055"]
        

    for day in pd.to_datetime(dataframe.index.date).unique().strftime('%Y-%m-%d'):
        print(f"Plotting LUT results for {substat} on {day}")
        df_down_day = dataframe.xs(downref[1],level="substat",axis=1).loc[day]
        df_lut_day = dataframe.xs(substat,level="substat",axis=1).loc[day]
        df_lr_down_day = df_lowres.xs(downref[1],level="substat",axis=1).loc[day]
        df_lr_lut_day = df_lowres.xs(substat,level="substat",axis=1).loc[day]
        df_sun_day = dataframe.xs('sun',level="substat",axis=1).loc[day]

        fig = plt.figure(figsize=(14,8))
            
        # df_day.plot.scatter(("Etotdown_pyr_Wm2",substat),("Etotdown_lut_Wm2",substat),c=("sza","sun"),cmap="jet",ax=ax)
    
        # ax.set_xlim([0,1200])
        # ax.set_ylim([0,1200])
    
        # plt.plot([0, 1], [0, 1], ls = '--', transform=ax.transAxes,c='k')
        
        # Definieren eines gridspec-Objekts
        gs = GridSpec(3, 1, height_ratios=[2, 1, 1], hspace=0.08)
        
        ax = fig.add_subplot(gs[0])
        ax.set_title(f"GHI from LUT for {name}, {substat} on {day}")        
        ax.plot(df_down_day.index,df_down_day[f"Etotdown_{downref[0]}_Wm2"],color='g')
        ax.plot(df_lut_day.index,df_lut_day["Etotdown_lut_Wm2"],color='r')
        ax.plot(df_lr_down_day.index,df_lr_down_day[f"Etotdown_{downref[0]}_Wm2"],color='b')
        ax.plot(df_lr_lut_day.index,df_lr_lut_day["Etotdown_lut_Wm2"],color='orange')
        ax.plot(df_down_day.index,df_down_day["Etotdown_clear_Wm2"],color='k',linestyle='--')
        ax.set_ylim([0,1200])
        ax.set_ylabel(r'GHI (W/m$^2$)')#, color='b')        
        
        ax.legend([rf"$G_{{\rm tot}}^{{\downarrow}}$ ({downref[1]},{timeres})",
                  rf"$G_{{\rm tot}}^{{\downarrow}}$ ({substat},LUT,{timeres})",
                  rf"$G_{{\rm tot}}^{{\downarrow}}$ ({downref[1]},{timeres_low})",
                  rf"$G_{{\rm tot}}^{{\downarrow}}$ ({substat},LUT,{timeres_low})",
                  rf"$G_{{\rm tot,clear}}^{{\downarrow}}$ ({downref[1]})"])
                
        #axtop.set_xticklabels()
                
        #ax.yaxis.grid(False)           
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(df_down_day.index,df_lut_day["Etotdown_lut_Wm2"] - 
                 df_down_day[f"Etotdown_{downref[0]}_Wm2"],color='r')
        ax2.plot(df_lr_down_day.index,df_lr_lut_day["Etotdown_lut_Wm2"] - 
                 df_lr_down_day[f"Etotdown_{downref[0]}_Wm2"],color='b')
        ax2.set_ylabel(r'$\Delta$GHI (W/m$^2$)')
        ax2.legend([f"{timeres}",f"{timeres_low}"])
        
        ax3 = fig.add_subplot(gs[2])
        
        #ax2.yaxis.grid(False)   
        ax3.plot(df_lut_day.index,df_lut_day[f"cf_poa_{timeres_cf}_avg"],color='k',linestyle='--')
        ax3.set_ylim([-0.1,1.1])
        ax3.set_ylabel(rf"$<cf>_\mathrm{{{timeres_cf}}}$")
        ax3.set_xlabel('Time (UTC)')
        
        datemin = pd.Timestamp(df_down_day.index[0]) #- pd.Timedelta("30min"))
        datemax = pd.Timestamp(df_down_day.index[-1]) #+ pd.Timedelta("30min"))      
        for axis in [ax,ax2,ax3]:
            axis.set_xlim([datemin,datemax])        
            axis.xaxis.set_major_locator(mdates.HourLocator(interval=2))
            axis.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))      
        
        ax.set_xticklabels('')
        ax2.set_xticklabels('')
        
        # axtop = ax.twiny()
        # axtop.set_xticks(ax.get_xticks())        
        # axtop.set_xbound(ax.get_xbound())
        # szalabels = np.round(df_sun_day.loc[pd.to_datetime(ax.get_xticks(),unit='D').round('S'),'sza'],2)
        # axtop.set_xticklabels(szalabels)
        # axtop.set_xlabel(r"SZA ($\circ$)")
        
        # axtop2 = ax.twiny()
        # axtop2.set_xticks(ax.get_xticks())        
        # axtop2.set_xbound(ax.get_xbound())
        # phi0labels = np.round(np.fmod(df_sun_day.loc[pd.to_datetime(ax.get_xticks(),unit='D').round('S'),'phi0']+180,360),2)
        # axtop2.set_xticklabels(phi0labels)        
        # new_fixed_axis = axtop2.get_grid_helper().new_fixed_axis
        # axtop2.axis["top"] = new_fixed_axis(loc="top",
        #                             axes=axtop2,
        #                             offset=(0, 50))

        # axtop2.axis["top"].toggle(all=True)
                
        
        #fig.tight_layout()
        plt.savefig(os.path.join(savepath,f'ghi_lut_result_{key}_{substat}_{day}.png')
                    ,bbox_inches = 'tight')
        plt.close(fig)
    
# def scatter_plot_ghi_sza(name,dataframe,substat,substat_pars,angle_string,
#                          timeres_cf,year,path,dict_paths):
#     """
    

#     Parameters
#     ----------
#     name : TYPE
#         DESCRIPTION.
#     dataframe : TYPE
#         DESCRIPTION.
#     substat : TYPE
#         DESCRIPTION.
#     substat_pars : TYPE
#         DESCRIPTION.
#     timeres_cf : TYPE
#         DESCRIPTION.
#     year : TYPE
#         DESCRIPTION.
#     path : TYPE
#         DESCRIPTION.

#     Returns
#     -------
#     None.

#     """
    
    
#     plt.ioff()
#     plt.close("all")
#     plt.style.use("my_paper")
    
#     res_dirs = list_dirs(path)
#     savepath = os.path.join(path,dict_paths["ghi"])
#     if dict_paths["ghi"] not in res_dirs:
#         os.mkdir(savepath)    
        
#     stat_dirs = list_dirs(savepath)
#     savepath = os.path.join(savepath,name)
#     if name not in stat_dirs:
#         os.mkdir(savepath)
    
#     res_dirs = list_dirs(savepath)
#     savepath = os.path.join(savepath,'Scatter')        
#     if 'Scatter' not in res_dirs:
#         os.mkdir(savepath)

#     substat_dirs = list_dirs(savepath)
#     savepath = os.path.join(savepath,substat)
#     if substat not in substat_dirs:
#         os.mkdir(savepath)

#     if "Pyr" in substat:
#         downref = ["pyr",substat]
#     elif "CMP11" in substat:
#         downref = ["CMP11","CMP11_Horiz"]
#     elif "SiRef" in substat:
#         downref = ["CMP11","CMP11_Horiz"]
        
#     fig, ax = plt.subplots(figsize=(9,8))
#     year = year.split('_')[-1]
#     ax.set_title(f"GHI: {downref[1]} vs. LUT from {substat}, {name}, {year}")
#     sc = ax.scatter(dataframe[(f"Etotdown_{downref[0]}_Wm2",downref[1])],
#     dataframe[("Etotdown_lut_Wm2",substat)],c=dataframe[("sza","sun")],cmap="jet")

#     max_ghi = np.max([dataframe[(f"Etotdown_{downref[0]}_Wm2",downref[1])].max(),
#                       dataframe[("Etotdown_lut_Wm2",substat)].max()])
#     max_ghi = np.ceil(max_ghi/100)*100    
#     ax.set_xlim([0,max_ghi])
#     ax.set_ylim([0,max_ghi])
    
#     ax.set_xlabel(rf"$G_{{\rm tot,{downref[1].replace('_',' ')}}}^{{\downarrow}}$ (W/m$^2$)")
#     ax.set_ylabel(rf"$G_{{\rm tot,{substat.replace('_',' ')},LUT}}^{{\downarrow}}$ (W/m$^2$)")
    
#     cb = plt.colorbar(sc)
#     cb.set_label(r"SZA ($\circ$)")

#     plt.plot([0, 1], [0, 1], ls = '--', transform=ax.transAxes,c='k')
    
#     plt.annotate(rf"$<cf>_\mathrm{{{timeres_cf}}}$",
#                  xy=(0.13,0.9),xycoords='figure fraction',fontsize=14)     
    
#     plt.annotate(rf"$\theta = {np.round(np.rad2deg(substat_pars[0][1]),1)}$ ({angle_string[0]})",
#                  xy=(0.15,0.86),xycoords='figure fraction',fontsize=14)         
#     plt.annotate(rf"$\phi = {np.round(np.rad2deg(azi_shift(substat_pars[1][1])),1)}$ ({angle_string[1]})",
#                  xy=(0.15,0.82),xycoords='figure fraction',fontsize=14)
    
#     fig.tight_layout()
#     if "calibration" in angle_string:
#         angle_figstring = "calibration"
#     else:
#         angle_figstring = "apriori"
#     plt.savefig(os.path.join(savepath,f'ghi_lut_scatter_plot_{key}_{substat}_{year}_{angle_figstring}_cf_{timeres_cf}.png'))
#     plt.close(fig)
    
def scatter_plot_ghi_hist(name,dataframe,nameghilut,pyrdown,substat,substat_pars,angle_string,
                          timeres,timeres_cf,year,path,dict_paths,model_dict,title=True):
    """
    Scatter plot with histogram as colour

    Parameters
    ----------
    name : string, name of PV station
    dataframe : dataframe with relevant irradiance data
    nameghilut : string, name of GHI from LUT
    pyrdown : string, name of pyranometer for validation
    substat : string, name of substation
    substat_pars : dictionary of substation parameters
    angle_string : string describing whether angles are from calibration or a priori
    timeres : string, time resolution of data
    timeres_cf : string, width of averaging window for cloud fraction calculation
    year : string, year under consideration
    path : string, main path for saving plots
    dict_paths : dictionary with specific plot paths
    model_dict : dictionary with PV model specifics
    title : boolean, whether to put title on plot. The default is True.

    Returns
    -------
    dataframe : dataframe with deviation added

    """
    """
    Scatter plot with histogram as color

    Parameters
    ----------
    name : TYPE
        DESCRIPTION.
    dataframe : TYPE
        DESCRIPTION.
    substat : TYPE
        DESCRIPTION.
    substat_pars : TYPE
        DESCRIPTION.
    timeres_cf : TYPE
        DESCRIPTION.
    year : TYPE
        DESCRIPTION.
    path : TYPE
        DESCRIPTION.
    title : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    
    plt.ioff()
    plt.close("all")
    plt.style.use("my_paper")
    
    res_dirs = list_dirs(path)
    savepath = os.path.join(path,dict_paths["ghi"])
    if dict_paths["ghi"] not in res_dirs:
        os.mkdir(savepath)    
        
    stat_dirs = list_dirs(savepath)
    savepath = os.path.join(savepath,name)
    if name not in stat_dirs:
        os.mkdir(savepath)
            
    res_dirs = list_dirs(savepath)
    savepath = os.path.join(savepath,'Scatter')        
    if 'Scatter' not in res_dirs:
        os.mkdir(savepath)
        
    substat_dirs = list_dirs(savepath)
    savepath = os.path.join(savepath,substat)
    if substat not in substat_dirs:
        os.mkdir(savepath)

    # substat_dirs = list_dirs(savepath)    
    # if model_dict:
    #     if model_dict["cal_model"] == "current":
    #         savepath = os.path.join(savepath,"Diode_Model")  
    #         if "Diode_Model" not in substat_dirs:
    #             os.mkdir(savepath)
    #     elif model_dict["cal_model"] == "power":
    #         savepath = os.path.join(Ssavepath,model_dict["power_model"])  
    #         if model_dict["power_model"] not in substat_dirs:
    #             os.mkdir(savepath)
    
    #     substat_dirs = list_dirs(savepath)
    #     savepath = os.path.join(savepath,model_dict["eff_model"])  
    #     if model_dict["eff_model"] not in substat_dirs:
    #         os.mkdir(savepath)            
        
    #     substat_dirs = list_dirs(savepath)
    #     savepath = os.path.join(savepath,model_dict["T_model"]["model"])  
    #     if model_dict["T_model"]["model"] not in substat_dirs:
    #         os.mkdir(savepath)            

    if "Pyr" in substat:
        downref = ["pyr",substat]
    elif "CMP11" in substat:
        downref = ["CMP11","CMP11_Horiz"]
    elif "SiRef" in substat:
        downref = ["CMP11","CMP11_Horiz"]
    elif "RT1" in substat:
        downref = ["pyr","Pyr055"]
    elif "auew" in substat or "egrid" in substat:        
        if "Pyr" in pyrdown:
            downref = ["pyr",pyrdown]
        elif "CMP11" in pyrdown:
            downref = ["CMP11",pyrdown]
        
    
    dataframe[("Delta_Etotdown_lut_Wm2",substat)] = dataframe[(nameghilut,substat)] - \
        dataframe[(f"Etotdown_{downref[0]}_Wm2",downref[1])]
        
    dataframe.sort_index(axis=1,level=1,inplace=True)
    
    fig, ax = plt.subplots(figsize=(9,8))
    year = year.split('_')[-1]
    if title: ax.set_title(f"GHI: {downref[1]} vs. LUT from {substat}, {name}, {year}")
    
    #This mask checks whether the two sensors have differing cloud fractions
    ghi_data = dataframe.loc[dataframe[("cloud_fraction_down",downref[1])] 
                             == dataframe[("cloud_fraction_poa",substat)]]
    
    ghi_data = ghi_data.loc[:,[(f"Etotdown_{downref[0]}_Wm2",downref[1]),(nameghilut,substat)]]
    ghi_data.dropna(axis=0,how='any',inplace=True)
    
    #ghi_data_rs = downsample(ghi_data, pd.Timedelta("1S"), pd.Timedelta("1min"))
    #ghi_data_rs.dropna(axis=0,how='any',inplace=True)
    
    ghi_ref = ghi_data[(f"Etotdown_{downref[0]}_Wm2",downref[1])].values
    ghi_lut = ghi_data[(nameghilut,substat)].values
    # xy = np.vstack([ghi_ref,ghi_lut])
    # z = gaussian_kde(xy)(xy)
    # idx = z.argsort()
    # ghi_ref_sort, ghi_lut_sort, z = ghi_ref[idx], ghi_lut[idx], z[idx]
    
    # sc = ax.scatter(ghi_ref_sort,ghi_lut_sort, s=40, c=z, cmap="jet")
    
    sns.histplot(x=ghi_ref,y=ghi_lut,cbar=True,ax=ax,bins=100,cmap="icefire",
                 cbar_kws={'label': 'Frequency'})

    max_ghi = np.max([ghi_data[(f"Etotdown_{downref[0]}_Wm2",downref[1])].max(),
                      ghi_data[(nameghilut,substat)].max()])
    max_ghi = np.ceil(max_ghi/100)*100    
    ax.set_xlim([0,max_ghi])
    ax.set_ylim([0,max_ghi])
    
    ax.set_xlabel(rf"$G_{{\rm tot,{downref[1].replace('_',' ')}}}^{{\downarrow}}$ (W/m$^2$)")
    ax.set_ylabel(rf"$G_{{\rm tot,{substat.replace('_',' ')},LUT}}^{{\downarrow}}$ (W/m$^2$)")
    
    # cb = plt.colorbar(sc, ticks=[np.min(z), np.max(z)], pad=0.05)
    # cb.set_ticklabels(["Low", "High"])  # horizontal colorbar
    # cb.set_label("Frequency", labelpad=-20, fontsize=14)
    
    plt.plot([0, 1], [0, 1], ls = '--', transform=ax.transAxes,c='k')
    
    plt.annotate(rf"$<cf>_\mathrm{{{timeres_cf}}}$",
                 xy=(0.13,0.9),xycoords='figure fraction',fontsize=14)     
    
    plt.annotate(rf"$\theta = {np.round(np.rad2deg(substat_pars[0][1]),1)}$ ({angle_string[0]})",
                 xy=(0.15,0.86),xycoords='figure fraction',fontsize=14)         
    plt.annotate(rf"$\phi = {np.round(np.rad2deg(azi_shift(substat_pars[1][1])),1)}$ ({angle_string[1]})",
                 xy=(0.15,0.82),xycoords='figure fraction',fontsize=14)
    
    fig.tight_layout()
    if "calibration" in angle_string:
        angle_figstring = "calibration"
    else:
        angle_figstring = "apriori"
    plt.savefig(os.path.join(savepath,f'ghi_lut_scatter_hist_{timeres}_{key}_{substat}_{year}_{angle_figstring}_cf_{timeres_cf}.png'))
    plt.close(fig)
    
    return dataframe

# def combined_stats(dict_combo_stats,year,timeres_list):
#     """
    

#     Parameters
#     ----------
#     df_combo : TYPE
#         DESCRIPTION.
#     year : TYPE
#         DESCRIPTION.
#     folder : TYPE
#         DESCRIPTION.
#     title_flag : TYPE
#         DESCRIPTION.

#     Returns
#     -------
#     None.

#     """
#     for timeres in timeres_list:        
#         if f"df_delta_all_{timeres}" in dict_combo_stats:
#             #Stack all values on top of each other for combined stats
#             df_delta_all = dict_combo_stats[f"df_delta_all_{timeres}"].stack(dropna=True)
                        
#             rmse = ((((df_delta_all.delta_GHI_Wm2.stack())**2).mean())**0.5)
#             mad = abs(df_delta_all.delta_GHI_Wm2.stack()).mean()
#             mbe = df_delta_all.delta_GHI_Wm2.stack().mean()
            
#             delta_max_plus = df_delta_all.delta_GHI_Wm2.stack().max()
#             delta_max_minus = df_delta_all.delta_GHI_Wm2.stack().min()
            
#             n_delta = len(df_delta_all.delta_GHI_Wm2.stack().dropna())
            
#             dict_combo_stats.update({timeres:{}})
#             dict_combo_stats[timeres].update({"n_delta":n_delta})
#             dict_combo_stats[timeres].update({"RMSE_GHI_Wm2":rmse})
#             dict_combo_stats[timeres].update({"MAD_GHI_Wm2":mad})
#             dict_combo_stats[timeres].update({"MBE_GHI_Wm2":mbe})
#             dict_combo_stats[timeres].update({"max_Delta_GHI_plus_Wm2":delta_max_plus})
#             dict_combo_stats[timeres].update({"max_Delta_GHI_minus_Wm2":delta_max_minus})
        
#             print(f"{year}: combined statistics at {timeres} from "\
#                   f"{dict_combo_stats[f'df_delta_all_{timeres}'].columns.levels[2].to_list()}"\
#                   f" calculated with {n_delta} measurements")
#             print(f"Combined RMSE for GHI in {year} is {rmse}")
#             print(f"Combined MAE for GHI in {year} is {mad}")
#             print(f"Combined MBE for GHI in {year} is {mbe}")


# def plot_all_combined_scatter(dict_stats,list_stations,pvrad_config,T_model,folder,title_flag):
#     """
    

#     Parameters
#     ----------
#     dict_stats : TYPE
#         DESCRIPTION.
#     pvrad_config : TYPE
#         DESCRIPTION.
#     folder : TYPE
#         DESCRIPTION.
#     title_flag : TYPE
#         DESCRIPTION.

#     Returns
#     -------
#     None.

#     """
    
#     plt.ioff()
#     plt.style.use('my_presi_grid')        
    
#     res_dirs = list_dirs(folder)
#     savepath = os.path.join(folder,'GHI_GTI_Plots')
#     if 'GHI_GTI_Plots' not in res_dirs:
#         os.mkdir(savepath)    
        
#     # res_dirs = list_dirs(savepath)
#     # savepath = os.path.join(savepath,'Scatter')
#     # if 'Scatter' not in res_dirs:
#     #     os.mkdir(savepath)
        
#     years = ["mk_" + campaign.split('_')[1] for campaign in pvrad_config["calibration_source"]]    
#     stations_label = '_'.join(["".join(s.split('_')) for s in list_stations])
    
#     for timeres in pvrad_config["timeres_comparison"]:
    
#         fig, axs = plt.subplots(1,len(years),sharex='all',sharey='all')
        
#         print(f"Plotting combined frequency scatter plot for {timeres}...please wait....")
#         plot_data = []
#         max_ghi = 0.; min_z = 500.; max_z = 0.
#         for i, ax in enumerate(axs.flatten()):            
#             year = years[i]
            
#             ghi_data = dict_stats[year][f"df_delta_all_{timeres}"].stack()\
#                 .loc[:,["GHI_PV_inv","GHI_Pyr_ref"]].stack().dropna(how='any')
            
#             ghi_ref = ghi_data["GHI_Pyr_ref"].values.flatten()
#             ghi_inv = ghi_data["GHI_PV_inv"].values.flatten()
#             xy = np.vstack([ghi_ref,ghi_inv])
#             z = gaussian_kde(xy)(xy)
#             idx = z.argsort()
            
#             plot_data.append((ghi_ref[idx], ghi_inv[idx], z[idx]))
            
#             max_ghi = np.max([max_ghi,ghi_ref.max(),ghi_inv.max()])
#             max_ghi = np.ceil(max_ghi/100)*100    
#             max_z = np.max([max_z,np.max(z)])
#             min_z = np.min([min_z,np.min(z)])
            
#         norm = plt.Normalize(min_z,max_z)    
        
#         for i, ax in enumerate(axs.flatten()):
#             year = years[i]
            
#             sc = ax.scatter(plot_data[i][0],plot_data[i][1], s=20, c=plot_data[i][2], 
#                             cmap="jet",norm=norm)
            
#             ax.plot([0, 1], [0, 1], ls = '--', transform=ax.transAxes,c='k')
#             #ax.set_title(f"{station_label}, {year_label}",fontsize=14)
            
#             print(f"Using {dict_stats[year][timeres]['n_delta']} data points for {year} plot")
#             ax.annotate(rf"MBE = {dict_stats[year][timeres]['MBE_GHI_Wm2']:.2f} W/m$^2$" "\n" \
#                         rf"RMSE = {dict_stats[year][timeres]['RMSE_GHI_Wm2']:.2f} W/m$^2$" "\n"\
#                         rf"n = ${dict_stats[year][timeres]['n_delta']:.0f}$",
#                      xy=(0.05,0.8),xycoords='axes fraction',fontsize=10,color='r',bbox = dict(facecolor='lightgrey',edgecolor='none', alpha=0.5),
#                      horizontalalignment='left',multialignment='left')     
#             # ax.annotate(rf"RMSE = {dict_stats[year][timeres]['RMSE_GTI_Wm2']:.2f} W/m$^2$",
#             #          xy=(0.05,0.85),xycoords='axes fraction',fontsize=10,bbox = dict(facecolor='lightgrey', alpha=0.3))  
#             # ax.annotate(rf"n = {dict_stats[year][timeres]['n_delta']:.0f}",
#             #          xy=(0.05,0.78),xycoords='axes fraction',fontsize=10,bbox = dict(facecolor='lightgrey', alpha=0.3))                  
#             #ax.set_xticks([0,400,800,1200])
                
#         cb = fig.colorbar(sc,ticks=[min_z,max_z], ax=axs[:2], shrink=0.6, location = 'top', 
#                            aspect=20)    
#         cb.set_ticklabels(["Low", "High"])  # horizontal colorbar
#         cb.set_label("Frequency", labelpad=-10, fontsize=16)
        
#         #Set axis limits
#         for i, ax in enumerate(axs.flatten()):
#             ax.set_xlim([0.,max_ghi])
#             ax.set_ylim([0.,max_ghi])
#             ax.set_aspect('equal')
#             ax.grid(False)
#             # if max_gti < 1400:
#             #     ax.set_xticks([0,400,800,1200])
#             # else:
#             #     ax.set_xticks([0,400,800,1200,1400])
#             if i == 0:
#                 ax.set_xlabel(r"$G_\mathrm{tot,pyranometer}^{\angle}$ (W/m$^2$)",position=(1.1,0))
#                 ax.set_ylabel(r"$G_\mathrm{{tot,inv}}^{{\angle}}$ (W/m$^2$)")
        
#         #fig.subplots_adjust(wspace=0.1)    
#         # fig.add_subplot(111, frameon=False)
#         # plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
#         # plt.grid(False)
        
#         plt.savefig(os.path.join(savepath,f"ghi_scatter_hist_combo_all_{timeres}_{T_model['model']}_"\
#                  f"{T_model['type']}_{stations_label}.png"),bbox_inches = 'tight')  
    
def generate_results_folders(home,path,rt_config,pvcal_config):
    """
    Generate folders for results
    
    args:
    :param home: string, home directory
    :param path: main path for saving files or plots    
    :param rt_config: dictionary with DISORT config
    :param pyr_config: dictionary with pyranometer configuration
    
    out:
    :return fullpath: string with label for saving folders   
    """    
    
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
    
def save_results(name,pv_station,info,rt_config,pyr_config,
                 pvrad_config,timeres_list,path):
    """
    Save results to binary data stream


    Parameters
    ----------
    name : string, name of PV station
    pv_station : dictionary with information and data from PV station
    info : string, description of current campaign
    rt_config : dictionary with radiative transfer configuration 
    pyr_config : dictionary with pyranometer configuration
    pvrad_config : dictionary with PV inversion configuration
    timeres_list : list of time resolutions of results
    path : string with path for saving data    

    Returns
    -------
    None.

    """
    
    pv_station_save = deepcopy(pv_station)
    
    filename = f"ghi_lut_results_{info}_{name}.data"
    
    #List of dataframes and information to save
    dfnames = ['lat_lon']      
                         
    for key in list(pv_station_save):
        if key not in dfnames and "df_pyr_pv" not in key and "substations" not in key and "path" not in key:
            del pv_station_save[key]
            
    pv_station_save['station_name'] = name
    pv_station_save['timeres'] = timeres_list
    
    with open(os.path.join(path,filename), 'wb') as filehandle:  
        # store the data as binary data stream
        pickle.dump((pv_station_save, rt_config, pyr_config, pvrad_config), filehandle)

    print('Results written to file %s\n' % filename)

#%%Main Program
#######################################################################
###                         MAIN PROGRAM                            ###
#######################################################################
#This program takes POA irradiance from pyranometers and uses DISORT
#to find the cloud optical depth, for situations where the cloud fraction 
#is above a certain threshold
#def main():
import argparse
    
parser = argparse.ArgumentParser()
parser.add_argument("-f","--configfile", help="yaml file containing config")
parser.add_argument("-s","--station",nargs='+',help="station for which to perform inversion")
#parser.add_argument("-c","--campaign",nargs='+',help="measurement campaign for which to perform simulation")
args = parser.parse_args()
   
if args.configfile:
    config_filename = os.path.abspath(args.configfile) #"config_PYRCAL_2018_messkampagne.yaml" #
else:
    config_filename = "config_PVPYRODGHI_MetPVNet_messkampagne.yaml"

#config_filename = os.path.abspath(args.configfile) #"config_PYROD_2019_messkampagne.yaml" #
 
config = load_yaml_configfile(config_filename)

#Load radiative transfer configuration
rt_config = load_yaml_configfile(config["rt_configfile"])

#Load PVCAL configuration
pvcal_config = load_yaml_configfile(config["pvcal_configfile"])

#Load pv2rad configuration
pvrad_config = load_yaml_configfile(config["pvrad_configfile"])

homepath = os.path.expanduser('~') # #"/media/luke" #
plot_styles = config["plot_styles"]
plot_flags = config["plot_flags"]
    
campaigns = config["description"]
wvl_range_pv = rt_config["common_base"]["wavelength"]["pv"]
T_model = pvcal_config["T_model"]

if args.station:
        stations = args.station
        if stations[0] == 'all':
            stations = 'all'
else:
    #Stations for which to perform inversion
    stations = ["PV_12"] #,"MS_02"] #pyr_config["stations"]

#%%Run through PV systems and extract GHI
#combo_stats = {}
for campaign in campaigns:   
    year = "mk_" + campaign.split('_')[1]
    yrname = year.split('_')[-1]

    #Load PV configuration
    pyr_config = load_yaml_configfile(config["pyrcalod_configfile"][year])
        
    #Load inversion results
    print(f'Using GHILUT to get global horizontal irradiance from tilted irradiance for {year}')
    print('Loading PYRCAL and cloud fraction results')
    pvsys, station_list, pyr_results_folder, pv_results_folder = \
    load_pyr2cf_pv2poarad_results(rt_config, pyr_config, pvcal_config, pvrad_config, 
                                  campaign, stations, homepath)

    # combo_stats.update({year:{}})
    # combo_stats[year].update({f"df_{yrname}_stats":pd.DataFrame(index=station_list)})
    # dfs_deviations = {}
    # for tres in pvrad_config["timeres_comparison"]:
    #     dfs_deviations.update({tres:[]})        
    
    #LUT parameters
    lut_config_pyr = pyr_config["ghilut_config"]
    timeres_cf_pyr = pyr_config["cloud_fraction"]["cf_avg_window"]
    lut_config_pv = pvrad_config["ghilut_config"]
    timeres_cf_pv = pvrad_config["cloud_fraction"]["cf_avg_window"]
    
    #Paths for results and plots
    savepath_pyr = generate_results_folders(homepath,os.path.join(pyr_config["results_path"]["main"],
                                pyr_config["results_path"]["irradiance"]),rt_config,pvcal_config)        
    plotpaths_pyr = pyr_config["results_path"]["plots"]
    savepath_pv = generate_results_folders(homepath,os.path.join(pvrad_config["results_path"]["main"],
                                pvrad_config["results_path"]["irradiance"]),rt_config,pvcal_config)   
    plotpaths_pv = pvrad_config["results_path"]["plots"]    
    
    #Run through the PV systems and pyranometers
    for key in pvsys:
        tres_list = []
        print('Using GHI2GTI LUT to calculate GHI from tilted pyranometers')        
        
        year = "mk_" + campaign.split('_')[1]    
        name_df_lowres = 'df_pyr_ghi_' + year.split('_')[-1] + '_' + rt_config["timeres"]
        
        name_df_ghi = ''
        
        if key in pyr_config["pv_stations"]:
            #Iterate over pyranometer substats
            for substat in pvsys[key]["substations_pyr"]:    
                model_dict = {}
                
                #Get radnames
                radname = pvsys[key]["substations_pyr"][substat]["name"]       
                                
                if "poa" in radname:        
                    timeres = pvsys[key]["substations_pyr"][substat]["t_res_inv"]  
                    if timeres not in tres_list:
                        tres_list.append(timeres)
                    if rt_config["timeres"] not in tres_list:
                        tres_list.append(rt_config["timeres"])
                    
                    name_df_hires = 'df_pyr_' + year.split('_')[-1] + '_' + timeres
                    name_df_sim = 'df_sim' #_' + year.split('_')[-1]        
            
                    #This bit is temporary for the Falltage, to save time for simulation
                    # dfs = []
                    # for day in pyr_config["falltage"]:
                    #     dfs.append(pvsys[key][name_df_hires].loc[day.strftime('%Y-%m-%d')])
                    
                    # df_temp = pvsys[key][name_df_hires] #
                        
                    #Get optimisation parameters for angles from calibration
                    angles = []
                    if lut_config_pyr["angles"] == "apriori":
                        dict_stat_pars = pvsys[key]["substations"][substat]['ap_pars']        
                        angles = ["apriori","apriori"]
                    elif lut_config_pyr["angles"] == "calibration":
                        if 'opt_pars' in pvsys[key]["substations_pyr"][substat]:
                            dict_stat_pars = pvsys[key]["substations_pyr"][substat]["opt_pars"]
                            for par in dict_stat_pars:
                                if par[2] == 0:
                                    angles.append("apriori")
                                else:
                                    angles.append("calibration")
                        else:
                            dict_stat_pars = pvsys[key]["substations_pyr"][substat]['ap_pars']        
                            angles = ["apriori","apriori"]
                    
                    
                    name_df_ghi = 'df_pyr_ghi_' + year.split('_')[-1] + '_' + timeres                            
                    if name_df_ghi in pvsys[key]:
                        print(f"Calculating cloud fraction with {timeres_cf_pyr} moving average for {substat}")        
                        #Average cloud fraction over 15 minutes and add to simulation dataframe                
                        pvsys[key][name_df_ghi] = average_cloud_fraction(pvsys[key][name_df_ghi],
                                      timeres,timeres_cf_pyr,substat)      
                        
                        print(f"Using GTI2GHI LUT to get global horizontal irradiance from {substat}")
                        pvsys[key][name_df_ghi], nameghi = ghifromgti("pyr",pvsys[key][name_df_ghi],substat,dict_stat_pars,
                                                radname,timeres_cf_pyr,wvl_range_pv,lut_config_pyr,homepath)
                    else:
                        print(f"Calculating cloud fraction with {timeres_cf_pyr} moving average for {substat}")        
                        #Average cloud fraction over 15 minutes and add to simulation dataframe                
                        pvsys[key][name_df_hires] = average_cloud_fraction(pvsys[key][name_df_hires],
                                      timeres,timeres_cf_pyr,substat)        
                
                        print(f"Using GTI2GHI LUT to get global horizontal irradiance from {substat}")
                        pvsys[key][name_df_ghi],nameghi = ghifromgti("pyr",pvsys[key][name_df_hires],substat,dict_stat_pars,
                                        radname,timeres_cf_pyr,wvl_range_pv,lut_config_pyr,homepath)
                        
                    #Sort multi-index (makes it faster)
                    pvsys[key][name_df_ghi].sort_index(axis=1,level=1,inplace=True)                                                                                                                                                                    
                    
                    #Iterate over pyranometer substats
                    if plot_flags["scatter"] and not pvsys[key][name_df_ghi][(nameghi,substat)].isna().all()\
                    and (radname.replace('poa','down'),substat) in pvsys[key][name_df_ghi].columns:
                        print(f"Plotting scatter histogram for {key}, {substat} at {timeres} resolution")
                        pvsys[key][name_df_ghi] = scatter_plot_ghi_hist(key,pvsys[key][name_df_ghi],nameghi,None,substat,
                                                dict_stat_pars,angles,timeres,timeres_cf_pyr,year,
                                                savepath_pyr,plotpaths_pyr,model_dict,plot_flags["titles"])
                
                #Resample pyranometer data to lower resolution
                if name_df_lowres not in pvsys[key]:
                    pvsys[key][name_df_lowres] = downsample_pyranometer_data(pvsys[key]
                                                 [name_df_ghi],substat,timeres,
                                                 rt_config["timeres"],radname,None)
                else:
                    pvsys[key][name_df_lowres] = pd.concat([pvsys[key][name_df_lowres],
                                                downsample_pyranometer_data(pvsys[key]
                                                 [name_df_ghi],substat,timeres,
                                                 rt_config["timeres"],radname,None)],axis=1)
            
                #Sort multi-index (makes it faster)
                pvsys[key][name_df_lowres].sort_index(axis=1,level=1,inplace=True)
                    
                #Do the same for 15 minute data
                if "poa" in radname:
                    if plot_flags["scatter"] and not pvsys[key][name_df_lowres][(nameghi,substat)].isna().all()\
                    and (radname.replace('poa','down'),substat) in pvsys[key][name_df_ghi].columns:
                        print(f"Plotting scatter histogram for {key}, {substat} at {rt_config['timeres']} resolution")
                        pvsys[key][name_df_lowres] = scatter_plot_ghi_hist(key,pvsys[key][name_df_lowres],nameghi,None,substat,
                                                dict_stat_pars,angles,rt_config["timeres"],timeres_cf_pyr,year,
                                                savepath_pyr,plotpaths_pyr,model_dict,plot_flags["titles"])
                
                    #Plot look up tables
                    if plot_flags["lut"]:     
                        print(f"Plotting LUT results for {key}, {substat}")
                        plot_ghi_comparison_day(key,pvsys[key][name_df_ghi],timeres,pvsys[key][name_df_lowres],
                                                rt_config["timeres"],substat,
                                            dict_stat_pars,timeres_cf_pyr,savepath_pyr,plotpaths_pyr)       
                    
                # scatter_plot_ghi_sza(key,pvsys[key][name_df_ghi],substat,
                    #                         dict_stat_pars,angles,timeres_cf,year,
                    #                         savepath,plotpaths)            
                
        #PV Systems
        if key in pvrad_config["pv_stations"]:
            
            name_df_lowres = 'df_pv_ghi_' + year.split('_')[-1] + '_' + rt_config["timeres"]
            #Now do PV systems        
            #dfs_stats = []
            for substat_type in pvsys[key]["substations_pv"]:    
                timeres = pvsys[key]["substations_pv"][substat_type]["t_res_inv"]
                if timeres not in tres_list:
                    tres_list.append(timeres)
                if rt_config["timeres"] not in tres_list:
                    tres_list.append(rt_config["timeres"])
                    
                for substat in pvsys[key]["substations_pv"][substat_type]["data"]:    
                    pvsys[key]['substations_pv'][substat_type]["data"][substat].\
                    update({"name":"Etotpoa_pv_inv_tau"})
                    #Get radnames
                    radname = pvsys[key]["substations_pv"][substat_type]["data"][substat]["name"]                    
                    if "poa" in radname:                            
                        name_df_hires = 'df_pv_' + year.split('_')[-1] + '_' + timeres
                        name_df_sim = 'df_sim' #_' + year.split('_')[-1]        
                
                        if name_df_hires in pvsys[key] and \
                        year in pvsys[key]["substations_pv"][substat_type]["source"]:                
                        #This bit is temporary for the Falltage, to save time for simulation
                        # dfs = []
                        # for day in pyr_config["falltage"]:
                        #     dfs.append(pvsys[key][name_df_hires].loc[day.strftime('%Y-%m-%d')])
                        
                        # df_temp = pvsys[key][name_df_hires] #
                                                
                            #Get optimisation parameters for angles from calibration
                            angles = []
                            if lut_config_pv["angles"] == "apriori":
                                dict_stat_pars = pvsys[key]["substations_pv"][substat_type]["data"][substat]['ap_pars']        
                                angles = ["apriori","apriori"]
                            elif lut_config_pv["angles"] == "calibration":
                                if 'opt_pars' in pvsys[key]["substations_pv"][substat_type]["data"][substat]:
                                    dict_stat_pars = pvsys[key]["substations_pv"]\
                                        [substat_type]["data"][substat]["opt_pars"]
                                    for par in dict_stat_pars:
                                        if par[2] == 0:
                                            angles.append("apriori")
                                        else:
                                            angles.append("calibration")
                                else:
                                    dict_stat_pars = pvsys[key]["substations_pv"]\
                                        [substat_type]["data"][substat]['ap_pars']        
                                    angles = ["apriori","apriori"]
                    
                            pvsys[key][name_df_hires] = pvsys[key][name_df_hires].loc[pvsys[key][name_df_hires][(radname,substat)].notna()]
                                                        
                            name_df_ghi = 'df_pv_ghi_' + year.split('_')[-1] + '_' + timeres                            
                            if name_df_ghi in pvsys[key]:
                                print(f"Calculating cloud fraction with {timeres_cf_pv} moving average for {substat}")        
                                #Average cloud fraction over 15 minutes and add to simulation dataframe                
                                pvsys[key][name_df_ghi] = average_cloud_fraction(pvsys[key][name_df_ghi],
                                             timeres,timeres_cf_pv,substat)      
                                
                                print(f"Using GTI2GHI LUT to get global horizontal irradiance from {substat}")
                                pvsys[key][name_df_ghi], nameghi = ghifromgti("pv",pvsys[key][name_df_ghi],substat,dict_stat_pars,
                                                        radname,timeres_cf_pv,wvl_range_pv,lut_config_pv,homepath)
                            else:
                                print(f"Calculating cloud fraction with {timeres_cf_pv} moving average for {substat}")        
                                #Average cloud fraction over 15 minutes and add to simulation dataframe                
                                pvsys[key][name_df_hires] = average_cloud_fraction(pvsys[key][name_df_hires],
                                             timeres,timeres_cf_pv,substat)        
                        
                                print(f"Using GTI2GHI LUT to get global horizontal irradiance from {substat}")
                                pvsys[key][name_df_ghi], nameghi = ghifromgti("pv",pvsys[key][name_df_hires],substat,dict_stat_pars,
                                                radname,timeres_cf_pv,wvl_range_pv,lut_config_pv,homepath)
                                    
                                #Sort multi-index (makes it faster)
                                pvsys[key][name_df_ghi].sort_index(axis=1,level=1,inplace=True)                                                                                                                                            
                                        
            #This is the scatter plot part
            for substat_type in pvsys[key]["substations_pv"]:    
                timeres = pvsys[key]["substations_pv"][substat_type]["t_res_inv"]
                if year in pvsys[key]["substations_pv"][substat_type]["source"]:
                    name_df_ghi = 'df_pv_ghi_' + year.split('_')[-1] + '_' + timeres                            
                    
                    if f"df_pyr_ghi_{year.split('_')[-1]}_{timeres}" in pvsys[key]\
                        and name_df_ghi in pvsys[key]:
                            df_ghi_merge = pd.concat([pvsys[key][f"df_pyr_ghi_{year.split('_')[-1]}_{timeres}"].loc[:,
                               pd.IndexSlice[["cloud_fraction_down"],[*pvsys[key]["substations_pyr"].keys()]]],
                               pvsys[key][name_df_ghi]],axis=1)
                            df_ghi_merge = df_ghi_merge.loc[:,~df_ghi_merge.columns.duplicated()]  
                    elif name_df_ghi in pvsys[key]:
                        df_ghi_merge = pvsys[key][name_df_ghi]
                    else: df_ghi_merge = pd.DataFrame()
                        
                    pyr_name = pvrad_config["pv_stations"][key][substat_type]["pyr_down"][year]
                    if key == pyr_name[0]:
                        pyr_down_name = pyr_name[1]
                        for substat in pvsys[key]["substations_pv"][substat_type]["data"]:    
                            model_dict = {}
                            model_dict["cal_model"] = pvsys[key]["substations_pv"]\
                                [substat_type]["data"][substat]["model"]
                            model_dict["power_model"] = pvcal_config["inversion"]["power_model"]
                            model_dict["T_model"] = pvcal_config["T_model"]
                            model_dict["eff_model"] = pvcal_config["eff_model"]
                            
                            if name_df_ghi in pvsys[key]:
                                angles = []
                                if lut_config_pv["angles"] == "apriori":
                                    dict_stat_pars = pvsys[key]["substations_pv"][substat_type]["data"][substat]['ap_pars']        
                                    angles = ["apriori","apriori"]
                                elif lut_config_pv["angles"] == "calibration":
                                    if 'opt_pars' in pvsys[key]["substations_pv"][substat_type]["data"][substat]:
                                        dict_stat_pars = pvsys[key]["substations_pv"]\
                                            [substat_type]["data"][substat]["opt_pars"]
                                        for par in dict_stat_pars:
                                            if par[2] == 0:
                                                angles.append("apriori")
                                            else:
                                                angles.append("calibration")
                                    else:
                                        dict_stat_pars = pvsys[key]["substations_pv"]\
                                            [substat_type]["data"][substat]['ap_pars']        
                                        angles = ["apriori","apriori"]
                                        
                                if plot_flags["scatter"] and np.rad2deg(dict_stat_pars[0][1]) > lut_config_pyr["tilt_lims"][0] and\
                                    np.rad2deg(dict_stat_pars[0][1]) < lut_config_pyr["tilt_lims"][1]:
                                    print(f"Plotting scatter histogram for {key}, {substat} at {timeres} resolution")
                                    pvsys[key][name_df_ghi] = scatter_plot_ghi_hist(key,df_ghi_merge,nameghi,pyr_down_name,substat,
                                                                dict_stat_pars,angles,timeres,timeres_cf_pv,year,
                                                                savepath_pv,plotpaths_pv,model_dict,plot_flags["titles"])
                            
                            #Resample pyranometer data to lower resolution
                            if timeres != rt_config["timeres"]:
                                if name_df_lowres not in pvsys[key]:
                                    pvsys[key][name_df_lowres] = downsample_pyranometer_data(df_ghi_merge,substat,timeres,
                                                                 rt_config["timeres"],radname,pyr_down_name)                    
                                else:                                                                        
                                    pvsys[key][name_df_lowres] = pd.concat([pvsys[key][name_df_lowres],
                                                                downsample_pyranometer_data(df_ghi_merge,substat,timeres,
                                                                 rt_config["timeres"],radname,pyr_down_name)],axis=1)
                                    
                                pvsys[key][name_df_lowres] = pd.concat([pvsys[key][name_df_lowres],
                                                    pvsys[key][name_df_lowres.replace('pv','pyr')]],axis=1)
                                
                                pvsys[key][name_df_lowres] = pvsys[key][name_df_lowres].loc[:,
                                                ~pvsys[key][name_df_lowres].columns.duplicated()]  
                            
                                #Sort multi-index (makes it faster)
                                pvsys[key][name_df_lowres].sort_index(axis=1,level=1,inplace=True)
                                    
                                #Do the same for 15 minute data
                                if "poa" in radname:
                                    if plot_flags["scatter"] and not pvsys[key][name_df_lowres][(nameghi,substat)].isna().all():#\
                                    #and (radname.replace('poa','down'),substat) in pvsys[key][name_df_ghi].columns:
                                        print(f"Plotting scatter histogram for {key}, {substat} at {rt_config['timeres']} resolution")
                                        pvsys[key][name_df_lowres] = scatter_plot_ghi_hist(key,pvsys[key][name_df_lowres],nameghi,pyr_down_name,substat,
                                                                dict_stat_pars,angles,rt_config["timeres"],timeres_cf_pv,year,
                                                                savepath_pv,plotpaths_pv,model_dict,plot_flags["titles"])
                                        
                                    #Plot look up tables
                                    if plot_flags["lut"]:     
                                        print(f"Plotting LUT results for {key}, {substat}")
                                        plot_ghi_comparison_day(key,df_ghi_merge,timeres,pvsys[key][name_df_lowres],
                                                                rt_config["timeres"],substat,
                                                            dict_stat_pars,timeres_cf_pyr,savepath_pv,plotpaths_pv)       
                                        
        #Save results depending on time resolution
        for timeres in tres_list:
            name_df_combine = f"df_pyr_pv_ghi_{year.split('_')[-1]}_{timeres}"                    
        
            #Merge data into one dataframe
            if f"df_pyr_ghi_{year.split('_')[-1]}_{timeres}" in pvsys[key]\
            and f"df_pv_ghi_{year.split('_')[-1]}_{timeres}" in pvsys[key]:
                df_ghi_merge = pd.concat([pvsys[key][f"df_pyr_ghi_{year.split('_')[-1]}_{timeres}"].loc[:,
                           pd.IndexSlice[:,[*pvsys[key]["substations_pyr"].keys()]]],
                           pvsys[key][f"df_pv_ghi_{year.split('_')[-1]}_{timeres}"].loc[:,
                           pd.IndexSlice[:,[col for col in pvsys[key]
                            [f"df_pv_ghi_{year.split('_')[-1]}_{timeres}"].columns.levels[1]
                            if col not in pvsys[key]["substations_pyr"].keys()]]]],axis=1)
                           #left_index=True,right_index=True)                                                                    
                
            elif f"df_pyr_ghi_{year.split('_')[-1]}_{timeres}" in pvsys[key]:
                df_ghi_merge = pvsys[key][f"df_pyr_ghi_{year.split('_')[-1]}_{timeres}"]                                                                        
                
            elif f"df_pv_ghi_{year.split('_')[-1]}_{timeres}" in pvsys[key]:
                df_ghi_merge = pvsys[key][f"df_pv_ghi_{year.split('_')[-1]}_{timeres}"]                                                                        
                
            else: df_ghi_merge = pd.DataFrame()
                
            if not df_ghi_merge.empty:            
                df_ghi_merge.sort_index(axis=1,level=1,inplace=True)
            
            pvsys[key][name_df_combine] = df_ghi_merge 
        
        if not df_ghi_merge.empty:
            save_results(key,pvsys[key],campaign,rt_config,pyr_config,
                     pvrad_config,tres_list,savepath_pv)  
        else:
            print(f'There are no results for {key}')
    
    #For combination analysis
#     for timeres in pvrad_config["timeres_comparison"]:
#         combo_stats[year][f"df_delta_all_{timeres}"] = pd.concat(dfs_deviations[timeres],axis=1)
#         if combo_stats[year][f"df_delta_all_{timeres}"].empty:
#             del combo_stats[year][f"df_delta_all_{timeres}"]
            
#     combined_stats(combo_stats[year],yrname,pvrad_config["timeres_comparison"])

# if plot_flags["combo_stats"]:
#     plot_all_combined_scatter(combo_stats,station_list,pvrad_config,T_model,savepath_pv,plot_flags["titles"])


            


