#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 09:32:58 2020

@author: james
"""

#%% Preamble
import os
import numpy as np
from file_handling_functions import *
from rt_functions import *
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.dates as mdates
#from matplotlib.gridspec import GridSpec
from plotting_functions import confidence_band
from scipy.stats import gaussian_kde
import pandas as pd
#from pvcal_forward_model import azi_shift
from data_process_functions import downsample
from scipy import optimize
import datetime
import seaborn as sns
import pickle
from copy import deepcopy

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
            station_list = list(pvrad_config["pv_stations"].keys())
    
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
    
    # results_path = os.path.join(home,pyr_config['results_path']["main"],
    #                             pyr_config["results_path"]["irradiance"])
    # pyr_folder_label = os.path.join(results_path,folder_label,'Pyranometer')    
    
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
            
    # results_path = os.path.join(home,pvrad_config["results_path"]["main"],
    #                             pvrad_config["results_path"]["irradiance"])
    pv_folder_label = os.path.join(mainpath,folder_label)    
    
    return pv_systems, station_list, pv_folder_label

# def load_pv2poarad_inversion_results(rt_config,pvcal_config,pvrad_config,station_list,home):
#     """
#     Load results from inversion onto plane-of-array irradiance
    
#     args:        
#     :param rt_config: dictionary with current RT configuration
#     :param pvcal_config: dictionary with current calibration configuration    
#     :param pvrad_config: dictionary with current inversion configuration
#     :param station_list: list of stations
#     :param home: string with homepath
    
#     out:
#     :return pv_systems: dictionary of PV systems with data
#     :return station_list: list of all stations
#     :return folder_label: string with folder label for saving data
#     """
    
#     mainpath = os.path.join(home,pvrad_config['results_path']['main'],
#                             pvrad_config['results_path']['inversion'])
    
#     #atmosphere model
#     atm_geom_config = rt_config["disort_base"]["pseudospherical"]
    
#     if atm_geom_config == True:
#         atm_geom_folder = "Pseudospherical"
#     else:
#         atm_geom_folder = "Plane-parallel"
        
#     #Wavelength range of simulation
#     wvl_config = rt_config["common_base"]["wavelength"]["pv"]
    
#     if type(wvl_config) == list:
#         wvl_folder_label = "Wvl_" + str(int(wvl_config[0])) + "_" + str(int(wvl_config[1]))
#     elif wvl_config == "all":
#         wvl_folder_label = "Wvl_Full"
    
#     disort_config = rt_config["disort_rad_res"]   
#     theta_res = str(disort_config["theta"]).replace('.','-')
#     phi_res = str(disort_config["phi"]).replace('.','-')
    
#     disort_folder_label = 'Zen_' + theta_res + '_Azi_' + phi_res
#     filename = 'tilted_irradiance_cloud_fraction_results_'
    
#     if rt_config["atmosphere"] == "default":
#         atm_folder_label = "Atmos_Default"    
#     elif rt_config["atmosphere"] == "cosmo":
#         atm_folder_label = "Atmos_COSMO"
#         filename = filename + 'atm_'
        
#     if rt_config["aerosol"]["source"] == "default":
#         aero_folder_label = "Aerosol_Default"
#     elif rt_config["aerosol"]["source"] == "aeronet":
#         aero_folder_label = "Aeronet_" + rt_config["aerosol"]["station"]
#         filename = filename + 'asl_' + rt_config["aerosol"]["data_res"] + '_'

#     sza_label = "SZA_" + str(int(pvrad_config["sza_max"]["inversion"]))

#     model = pvcal_config["inversion"]["power_model"]
#     eff_model = pvcal_config["eff_model"]
#     T_model = pvcal_config["T_model"]["model"] 
    
#     folder_label = os.path.join(mainpath,atm_geom_folder,wvl_folder_label,
#                                 disort_folder_label,atm_folder_label,
#                                 aero_folder_label,sza_label,model,eff_model,
#                                 T_model)
    
#     if len(pvrad_config["calibration_source"]) > 1:
#         infos = '_'.join(pvrad_config["calibration_source"])
#     else:
#         infos = pvrad_config["calibration_source"][0]
    
#     filename = filename + infos + '_disortres_' + theta_res + '_' + phi_res + '_'
    
#     pv_systems = {}    
        
#     #Choose which stations to load    
#     if type(station_list) != list:
#         station_list = [station_list]    
#         if station_list[0] == "all":
#             station_list = list(pvrad_config["pv_stations"].keys())
    
#     for station in station_list:                
#         #Read in binary file that was saved from pvcal_radsim_disort
#         filename_stat = filename + station + '.data'
#         try:
#             with open(os.path.join(folder_label,filename_stat), 'rb') as filehandle:  
#                 # read the data as binary data stream
#                 (pvstat, dummy, dummy, dummy) = pd.read_pickle(filehandle)            
#             pv_systems.update({station:pvstat})
#             print('Data for %s loaded from %s, %s' % (station,folder_label,filename))
#         except IOError:
#             print('There are no inversion results for %s in %s' % (station,folder_label))
            
#     return pv_systems, station_list, folder_label



def write_results_table(key,substat,stats,pyrname,year,path):
    """
    
    Parameters
    ----------
    key : string, name of station
    substat: string, name of substation
    stats : dictionary with statistics of deviations
    pyrname : string with name of pyranometer used for validation 
    year : string with year (2018 or 2019)
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
    f.write(f"{key} {substat:>8} {model:>10} {stats[f'RMSE_GTI_{pyrname}_Wm2']:10.3f} "\
            f"{stats[f'MBE_GTI_{pyrname}_Wm2']:10.3f} {stats[f'MAD_GTI_{pyrname}_Wm2']:10.3f} "\
                f"{stats[f'max_Delta_GTI_plus_{pyrname}_Wm2']:13.3f} {stats[f'max_Delta_GTI_minus_{pyrname}_Wm2']:13.3f} "\
                    f"{stats['n_delta']:10.0f}\n")
    f.close()
    
def downsample_pyranometer(dataframe,timeres_old,timeres_new):
    """
    Downsample pyranometer data to coarser data

    Parameters
    ----------
    dataframe : dataframe with high resolution data    
    timeres_old : string, old time resolution
    timeres_new : string, desired time resolution    

    Returns
    -------
    dataframe with downsampled pyranometer data

    """
    
    #Convert time resolutions
    timeres_old = pd.to_timedelta(timeres_old)
    timeres_new = pd.to_timedelta(timeres_new)
    
    dfs_rs = []
    for day in pd.to_datetime(dataframe.index.date).unique().strftime('%Y-%m-%d'):
        #Downsample data
        dfs_rs.append(downsample(dataframe.loc[day], timeres_old, timeres_new))
    
    df_rs = pd.concat(dfs_rs,axis=0)
        
    return df_rs

def calc_statistics_gti(key,pv_station,year,pvrad_config,pyrcal_config,folder):
    """

    Calculate deviations between inverted and measured irradiance, and statistics    

    Parameters
    ----------
    key : string, name of station
    pv_station : dictionary with all information and data from PV station
    year : string with current year (2018 or 2019)        
    pvrad_config: dictionary with inversion configuration        
    pyrcal_config : dictionary with pyranometer configuration        
    folder : string with folder for saving results        

    Returns
    -------
    df_stats_final : dataframe with final statistics

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
                dfname = f"df_pv_{yrname}_{timeres}"
                
                dataframe = pv_station[dfname].loc[(pv_station[dfname][("sza","sun")]\
                          <= pvrad_config["sza_max"]["inversion"]) &\
                             (pv_station[dfname][("theta_IA",substat)] <= \
                              pvrad_config["sza_max"]["inversion"])]
                
                pyrname = pvrad_config["pv_stations"][key][substat_type]["pyr_down"][year][1]
                if "Horiz" in pyrname:
                    pyrname = pyrname.split('_')[0] + "_32S"
                    
                pyr_station = pvrad_config["pv_stations"][key][substat_type]["pyr_down"][year][0]
                radname = pyrcal_config["pv_stations"][pyr_station]["substat"][pyrname]["name"]                                
                
                delta_GTI = dataframe[("Etotpoa_pv_inv_tau",substat)]\
                    - dataframe[(radname,pyrname)]
                rmse = (((delta_GTI)**2).mean())**0.5
                mad = abs(delta_GTI).mean()
                mbe = delta_GTI.mean()
                
                mean_obs = dataframe[(radname,pyrname)].mean(axis=0)
                rmbe = mbe/mean_obs*100
                rrmse = rmse/mean_obs*100
                
                delta_max_plus = delta_GTI.max()
                delta_max_minus = delta_GTI.min()                
                
                n_delta = len(delta_GTI.dropna())
                
                stats.update({substat:{}})
                
                stats[substat].update({"n_delta":n_delta})
                stats[substat].update({f"RMSE_GTI_{pyrname}_Wm2":rmse})
                stats[substat].update({f"rRMSE_GTI_{pyrname}_%":rrmse})
                stats[substat].update({f"MAD_GTI_{pyrname}_Wm2":mad})
                stats[substat].update({f"MBE_GTI_{pyrname}_Wm2":mbe})
                stats[substat].update({f"rMBE_GTI_{pyrname}_%":rmbe})
                stats[substat].update({f"mean_GTI_{pyrname}_Wm2":mean_obs})
                stats[substat].update({f"max_Delta_GTI_plus_{pyrname}_Wm2":delta_max_plus})
                stats[substat].update({f"max_Delta_GTI_minus_{pyrname}_Wm2":delta_max_minus})
                
                print(f"{key}, {yrname}: statistics at {timeres} calculated with {n_delta} measurements")
                print(f"RMSE for GTI inverted from {substat} compared to {pyrname} is {rmse} ({rrmse:.1f} %)")
                print(f"MAE for GTI inverted from {substat} compared to {pyrname} is {mad}")
                print(f"MBE for GTI inverted from {substat} compared to {pyrname} is {mbe} ({rmbe:.1f} %)")
                
                #Assign delta to the dataframe
                pv_station[dfname].loc[dataframe.index,("delta_GTI_Wm2",substat)] = delta_GTI
                
                if f"df_stats_{timeres}_{yrname}" not in pv_station:
                    pv_station[f"df_stats_{timeres}_{yrname}"] = pv_station[dfname].loc[dataframe.index,[("delta_GTI_Wm2",substat),
                                                     ("Etotpoa_pv_inv",substat),(radname,pyrname)]]
                else:
                    pv_station[f"df_stats_{timeres}_{yrname}"] = pd.concat([pv_station[f"df_stats_{timeres}_{yrname}"],
                            pv_station[dfname].loc[dataframe.index,[("delta_GTI_Wm2",substat),
                                                    ("Etotpoa_pv_inv",substat),(radname,pyrname)]]],axis=1)            
                
                #Write stats results to text file
                write_results_table(key,substat,stats[substat],pyrname,year,folder)
                
                df_stats = pd.DataFrame(pv_station["stats"][year][substat],index=[key])
                new_columns = ["n_delta"]
                new_columns.extend(["_".join([val for i, val in enumerate(col.split('_')) 
                             if i != len(col.split('_')) - 2]) for col in df_stats.columns[1:]])
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
                            if ("Pyr" not in col) and ("CMP" not in col) and ("RT" not in col)
                            and ("suntracker" not in col)]            
                if timeres == "1min" and key != "MS_02":
                    new_cols = [col.split('_')[0] for col in new_cols if ("egrid" in col)]
                # elif timeres == "15min":
                #     new_cols = [col for col in new_cols if "auew" in col]
                pv_station[f"df_stats_{timeres}_{yrname}"].columns = pd.MultiIndex.from_product(
                    [new_cols,['delta_GTI_Wm2','GTI_PV_inv','GTI_Pyr_ref'],[key]],
                    names=['substat','variable','station']).swaplevel(0,1)
    else: df_stats_final = pd.DataFrame()
    
        
    return df_stats_final
                                
def plot_irradiance_comparison(key,pv_station,year,pvrad_config,folder):
    """
    

    Parameters
    ----------
    key : string, name of station
    pv_station : dictionary with all information and data from PV station
    year : string with current year (2018 or 2019)        
    pvrad_config: dictionary with inversion configuration            
    folder : string with folder for saving results            

    Returns
    -------
    None.

    """
    
    
    plt.ioff()
    plt.style.use('my_paper')
    
    res_dirs = list_dirs(folder)
    savepath = os.path.join(folder,'GTI_Plots')
    if 'GTI_Plots' not in res_dirs:
        os.mkdir(savepath)   
        
    res_dirs = list_dirs(savepath)
    savepath = os.path.join(savepath,'Comparison')
    if 'Comparison' not in res_dirs:
        os.mkdir(savepath)   
        
    stat_dirs = list_dirs(savepath)
    savepath = os.path.join(savepath,key)
    if key not in stat_dirs:
        os.mkdir(savepath)
        
    for substat_type in pv_station["substations_pv"]:    
        for substat in pv_station["substations_pv"][substat_type]["data"]:
            substat_dirs = list_dirs(savepath)
            plotpath = os.path.join(savepath,substat)
            if substat not in substat_dirs:
                os.mkdir(plotpath)
            
            if year in pv_station["substations_pv"][substat_type]["source"]:                                
                print('Generating irradiance plots for %s, %s' % (substat,year))
                timeres = pv_station["substations_pv"][substat_type]["t_res_inv"]
                dfname = 'df_pv_' + year.split('_')[-1] + '_' + timeres
                # dfcosmo = 'df_cosmo_' + year.split('_')[-1]
                dataframe = pv_station[dfname] #.xs(substat,level='substat',axis=1)             
                                
                pyrname = pvrad_config["pv_stations"][key][substat_type]["pyr_down"][year][1]
                if "Horiz" in pyrname:
                    pyrname = pyrname.split('_')[0] + "_32S"
                    
                pyr_station = pvrad_config["pv_stations"][key][substat_type]["pyr_down"][year][0]
                radname = pyrcal_config["pv_stations"][pyr_station]["substat"][pyrname]["name"]                
                if '_' in pyrname:
                    pyrlabel = f'{pyrname.split("_")[0]}_{{{pyrname.split("_")[1]}}}'                    
                else: pyrlabel = pyrname
                
                #Keep only data for which an inversion exists
                dataframe = dataframe.loc[dataframe['P_meas_W',substat].notna()]
                #test_days = pv_station["sim_days"][year]
                
                test_days = pd.to_datetime(dataframe.index.date).unique().strftime('%Y-%m-%d')    
                
                # Etotdown_cosmo = pv_station[dfcosmo][('Edirdown_Wm2','cosmo')] +\
                #                     pv_station[dfcosmo][('Ediffdown_Wm2','cosmo')]
                # #Shift COSMO irradiance data by 30 minutes
                # Etotdown_cosmo.index = Etotdown_cosmo.index - pd.Timedelta('30min')                                
                
                for iday in test_days:
                    df_test_day = dataframe.loc[iday]
                    
                    fig, ax = plt.subplots(figsize=(9,8))
                    
                    df_test_day["Etotpoa_pv_clear_Wm2",substat].plot(ax=ax,color='r',style='--',
                                                          label=r'$G_{\rm clear}^{\angle}$')
                    df_test_day["Etotpoa_pv_inv",substat].plot(ax=ax,color='g',
                                                    label=r'$G_{\rm PV inv}^{\angle}$')
                    confidence_band(ax,df_test_day.index,df_test_day["Etotpoa_pv_inv",substat],
                                   df_test_day["error_Etotpoa_pv_inv",substat],'g')                
                    df_test_day[radname,pyrname].plot(ax=ax,color='m',label=r'$G_{{{}}}^{{\angle}}$'.format(pyrlabel))
        
                    # if "diode_model" in pvcal_config["pv_stations"][key] and "2019" in dfname:
                    #     pv_station[dfname].loc[iday,('Etotpoa_pv_inv_diode','WR_3')].plot(ax=ax,legend=False,color='b')
                    #     confidence_band(pv_station[dfname].loc[iday,('Etotpoa_pv_inv_diode','WR_3')].index,
                    #                     pv_station[dfname].loc[iday,('Etotpoa_pv_inv_diode','WR_3')],
                    #                 pv_station[dfname].loc[iday,('error_Etotpoa_pv_inv_diode','WR_3')],'b')                
                    #     legstring.extend([r'$G_{\rm PV-I,[0.3,1.2]\mu m}^{\angle}$'])
                    
                    # pyrname = pvcal_config["pv_stations"][key]["input_data"]["irrad"][year]
                    # pv_station[dfname].loc[iday,('Etotpoa_pyr_Wm2',pyrname)].plot(ax=ax,legend=False,color='darkred')
                    # pv_station[dfname].loc[iday,('Etotdown_pyr_Wm2',pyrname)].plot(ax=ax,legend=False,color='k')
                    # legstring.extend([r'$G_{\rm ' + pyrname + r'}^{\angle}$',r'$G_{\rm ' + pyrname + r'}^{\downarrow}$'])
                    
                    #Etotdown_cosmo.loc[iday].plot(ax=ax,legend=False,color='c',label=r'$G_{\rm COSMO}^{\downarrow}$')
                    
                    ax.legend()
                    
                    datemin = np.datetime64(iday + ' 03:00:00')
                    datemax = np.datetime64(iday + ' 21:00:00')   
                    ax.set_xlim([datemin, datemax])
                    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  
                    
                    ax.set_xlabel('Time (UTC)')
                    ax.set_ylabel(r'Irradiance (W/m$^2$)')
                    ax.set_title('Irradiance ratio for ' + key + ' on ' + iday)
                    
                    plt.savefig(os.path.join(plotpath,f'irrad_inv_{key}_{substat}_vs_{pyrname}_{iday}.png'))
                    plt.close(fig)
            
def scatter_plot_gti_hist(key,pv_station,pvrad_config,folder):
    """
    

    Parameters
    ----------
    key : string, name of station
    pv_station : dictionary with all information and data from PV station    
    pvrad_config: dictionary with inversion configuration           
    folder : string with folder for saving results            

    Returns
    -------
    None.

    """
    
    plt.ioff()
    plt.style.use('my_paper')
    
    res_dirs = list_dirs(folder)
    savepath = os.path.join(folder,'GTI_Plots')
    if 'GTI_Plots' not in res_dirs:
        os.mkdir(savepath)    
        
    res_dirs = list_dirs(savepath)
    savepath = os.path.join(savepath,'Scatter')
    if 'Scatter' not in res_dirs:
        os.mkdir(savepath)
        
    stat_dirs = list_dirs(savepath)
    savepath = os.path.join(savepath,key)
    if key not in stat_dirs:
        os.mkdir(savepath)            
        
    for substat_type in pv_station["substations_pv"]:    
        for substat in pv_station["substations_pv"][substat_type]["data"]:                        
            if year in pv_station["substations_pv"][substat_type]["source"]:                                
                print('Generating scatter plots for %s, %s, %s' % (key,substat,year))
                timeres = pv_station["substations_pv"][substat_type]["t_res_inv"]
                dfname = 'df_pv_' + year.split('_')[-1] + '_' + timeres
                # dfcosmo = 'df_cosmo_' + year.split('_')[-1]
                dataframe = pv_station[dfname].loc[(pv_station[dfname][("sza","sun")]\
                          <= pvrad_config["sza_max"]["inversion"]) &\
                           (pv_station[dfname][("theta_IA",substat)] <= pvrad_config["sza_max"]["inversion"])]          
                
                if 'opt_pars' in pv_station["substations_pv"][substat_type]["data"][substat]:
                    opt_pars = pv_station["substations_pv"][substat_type]["data"][substat]['opt_pars'] 
                else:
                    opt_pars = pv_station["substations_pv"][substat_type]["data"][substat]['ap_pars'] 
                    
                tilt = np.rad2deg(opt_pars[0][1])
                azi = np.fmod(np.rad2deg(opt_pars[1][1])+180,360)
                
                stats = pv_station["stats"][year][substat]
                
                pyrname = pvrad_config["pv_stations"][key][substat_type]["pyr_down"][year][1]
                if "Horiz" in pyrname:
                    pyrname = pyrname.split('_')[0] + "_32S"
                    
                pyr_station = pvrad_config["pv_stations"][key][substat_type]["pyr_down"][year][0]
                radname = pyrcal_config["pv_stations"][pyr_station]["substat"][pyrname]["name"]                
                if '_' in pyrname:
                    pyrlabel = f'{pyrname.split("_")[0]}_{{{pyrname.split("_")[1]}}}'                    
                else: pyrlabel = pyrname
                
                #Keep only data for which an inversion exists
                dataframe = dataframe.loc[dataframe['P_meas_W',substat].notna()]
    
                fig, ax = plt.subplots(figsize=(9,8))
                            
                gti_data = dataframe.loc[:,[(radname,pyrname),("Etotpoa_pv_inv_tau",substat)]]
                gti_data.dropna(axis=0,how='any',inplace=True)
                
                gti_ref = gti_data[radname,pyrname].values
                gti_inv = gti_data["Etotpoa_pv_inv_tau",substat].values #
                xy = np.vstack([gti_ref,gti_inv])
                z = gaussian_kde(xy)(xy)
                idx = z.argsort()
                gti_ref_sort, gti_inv_sort, z = gti_ref[idx], gti_inv[idx], z[idx]
                
                sc = ax.scatter(gti_ref_sort,gti_inv_sort, s=40, c=z, cmap="jet")
            
                max_gti = np.max([gti_ref.max(),gti_inv.max()])
                max_gti = np.ceil(max_gti/100)*100    
                ax.set_xlim([0,max_gti])
                ax.set_ylim([0,max_gti])
                
                ax.set_xlabel(rf"$G_{{\rm tot,{pyrlabel}}}^{{\angle}}$ (W/m$^2$)")
                ax.set_ylabel(rf"$G_{{\rm tot,{substat.replace('_',' ')},inv}}^{{\angle}}$ (W/m$^2$)")
                
                cb = plt.colorbar(sc, ticks=[np.min(z), np.max(z)], pad=0.05)
                cb.set_ticklabels(["Low", "High"])  # horizontal colorbar
                cb.set_label("Frequency", labelpad=-20, fontsize=14)
                
                plt.plot([0, 1], [0, 1], ls = '--', transform=ax.transAxes,c='k')
                
                plt.annotate(rf"$\theta = {tilt:.2f}^\circ, \phi = {azi:.2f}^\circ$",
                 xy=(0.14,0.9),xycoords='figure fraction',fontsize=14)                     
                plt.annotate(rf"MBE = {stats[f'MBE_GTI_{pyrname}_Wm2']:.3f} W/m$^2$ ({stats[f'rMBE_GTI_{pyrname}_%']:.1f} %)",
                 xy=(0.14,0.86),xycoords='figure fraction',fontsize=14)     
                plt.annotate(rf"RMSE = {stats[f'RMSE_GTI_{pyrname}_Wm2']:.3f} W/m$^2$ ({stats[f'rRMSE_GTI_{pyrname}_%']:.1f} %)",
                 xy=(0.14,0.82),xycoords='figure fraction',fontsize=14)     
                plt.annotate(rf"n = {stats['n_delta']:.0f}",
                 xy=(0.14,0.78),xycoords='figure fraction',fontsize=14)  
            
                plt.savefig(os.path.join(savepath,f"gti_scatter_hist_{timeres}_{key}_{substat}_{pyrname}_{year}.png")
                            ,bbox_inches = 'tight')   
                plt.close(fig)                                         

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
    return: series with average cloud fraction
    """
    
    timeres_old = pd.to_timedelta(timeres_old).seconds # measurement timeresolution in sec
    timeres_ave = pd.to_timedelta(timeres_window).seconds
    kernelsize = timeres_ave/timeres_old # kernelsize 
    box_kernel = Box1DKernel(kernelsize)     
                
    days = pd.to_datetime(dataframe.index.date).unique().strftime('%Y-%m-%d')        
    for iday in days:                
        for radtype in radtypes:
            cf_avg = convolve(dataframe.loc[iday,(f"cloud_fraction_{radtype}",substat)].values.flatten(), box_kernel)
                    
            # handle edges
            edge = int(kernelsize/2.)
            cf_avg  = cf_avg[edge:-edge]
            index_cut = dataframe.loc[iday].index[edge:-edge]
            dataframe.loc[index_cut,(f"cf_{radtype}_{timeres_window}_avg",substat)] = cf_avg                
            
            df_avg = moving_average_std(dataframe.loc[iday,(f"cloud_fraction_{radtype}",substat)], 
                                        timeres_old, timeres_ave)
            
            dataframe.loc[iday,(f"cf_{radtype}_{timeres_window}_avg_alt",substat)] = df_avg["avg_pd"]
            
    #Sort multi-index (makes it faster)
    dataframe.sort_index(axis=1,level=1,inplace=True)
    
    return dataframe

def plot_cloud_fraction(key,pv_station,year,pvrad_config,folder):
    """
    

    Parameters
    ----------
    key : string, name of station
    pv_station : dictionary with all information and data from PV station
    year : string with current year (2018 or 2019)        
    pvrad_config: dictionary with inversion configuration            
    folder : string with folder for saving results        
    
    Returns
    -------
    None.

    """
    
    plt.ioff()
    plt.style.use('my_paper')
    
    res_dirs = list_dirs(folder)
    savepath = os.path.join(folder,'Cloud_Fraction_Plots')
    if 'Cloud_Fraction_Plots' not in res_dirs:
        os.mkdir(savepath)        
        
    stat_dirs = list_dirs(savepath)
    savepath = os.path.join(savepath,key)
    if key not in stat_dirs:
        os.mkdir(savepath)
        
    for substat_type in pv_station["substations_pv"]:    
        for substat in pv_station["substations_pv"][substat_type]["data"]:
            substat_dirs = list_dirs(savepath)
            plotpath = os.path.join(savepath,substat)
            if substat not in substat_dirs:
                os.mkdir(plotpath)
            
            if year in pv_station["substations_pv"][substat_type]["source"]:                
                print('Generating cloud fraction plots for %s, %s' % (substat,year))
                timeres = pv_station["substations_pv"][substat_type]["t_res_inv"]
                dfname = 'df_pv_' + year.split('_')[-1] + '_' + timeres
                                                
                dataframe = pv_station[dfname].xs(substat,level='substat',axis=1)
                if "cloudcam" in pv_station[dfname].columns.levels[1]:
                    df_cloudcam = pv_station[dfname].xs("cloudcam",level='substat',axis=1)
                
                #test_days = pv_station["sim_days"][year]
                
                test_days = pd.to_datetime(dataframe.index.date).unique().strftime('%Y-%m-%d')    
                #test_days = [day.strftime('%Y-%m-%d') for day in pvrad_config["falltage"][year]]
                for iday in test_days:                    
                    df_test_day = dataframe.loc[iday]
                    
                    if not df_test_day.cloud_fraction_poa.dropna(how='all',axis=0).empty:
                        #print(f"Plotting {iday}")
                        fig, ax = plt.subplots(figsize=(9,8))
                        
                        df_test_day.k_index_poa.plot(ax=ax,legend=False,color='r',style='--',label='Clearness index')
                        df_test_day.cloud_fraction_poa.plot(ax=ax,legend=False,color='g',label='Cloud fraction')                                
                        
                        cloud_fraction_ave = downsample(df_test_day.cloud_fraction_poa,
                                                        pd.to_timedelta(timeres),pd.to_timedelta('15min'))
                        
                        cloud_fraction_ave.plot(ax=ax,legend=False,color='b',label='15 min average cloud fraction')
                        
                        if "cloudcam" in pv_station[dfname].columns.levels[1]:
                            df_cloudcam.loc[iday].plot(ax=ax,color='c',label='Cloud camera cloud fraction')
                        
                        ax.legend()
                        
                        ax.set_xlabel('Time (UTC)')
                        ax.set_ylabel(r'')
                        ax.set_ylim([0,2])
                        ax.set_title('Cloud fraction and clearness index for '
                                     + key + ' on ' + iday)
                        
                        plt.savefig(os.path.join(plotpath,'cloud_fraction_' + key + '_' 
                                                 + substat + '_' + iday + '.png'))
                        plt.close(fig)

def plot_mean_spectral_fit(key,pv_station,year,pvrad_config,folder,title_flag):
    """
    

    Parameters
    ----------
    key : string, name of station
    pv_station : dictionary with all information and data from PV station
    year : string with current year (2018 or 2019)        
    pvrad_config: dictionary with inversion configuration            
    folder : string with folder for saving results            
    title_flag : boolean for plot titles

    Returns
    -------
    None.

    """
    
    plt.ioff()
    plt.style.use('my_presi_grid')
    
    sza_limit = pvrad_config["sza_max"]["inversion"]
    
    res_dirs = list_dirs(folder)
    savepath = os.path.join(folder,'Spectral_Fit_Plots')
    if 'Spectral_Fit_Plots' not in res_dirs:
        os.mkdir(savepath)        
        
    res_dirs = list_dirs(savepath)
    savepath = os.path.join(savepath,f"SZA_{int(sza_limit)}")
    if f"SZA_{int(sza_limit)}" not in res_dirs:
        os.mkdir(savepath)    
        
    stat_dirs = list_dirs(savepath)
    savepath = os.path.join(savepath,key)
    if key not in stat_dirs:
        os.mkdir(savepath)
        
    for substat_type in pv_station["substations_pv"]:    
        for substat in pv_station["substations_pv"][substat_type]["data"]:
            substat_dirs = list_dirs(savepath)
            plotpath = os.path.join(savepath,substat)
            if substat not in substat_dirs:
                os.mkdir(plotpath)
    
            dffit = pv_station["substations_pv"][substat_type]["data"][substat][f"df_spectral_fit_{year}"]
            dffit = dffit.loc[dffit.cos_IA != 0]
            dffit["cos_diff_phi"] = np.cos(np.deg2rad(dffit.diff_phi))
            dffit["phi0_real"] = np.fmod(dffit.phi0 + 180,360)
            dffit["cos_product"] = np.sign(np.sin(np.deg2rad(dffit.diff_phi)))\
                *np.cos(np.pi/2 - np.arccos(dffit.cos_IA))
            days = pd.to_datetime(dffit.index.date).unique().strftime('%Y-%m-%d')
            
            if 'opt_pars' in pv_station["substations_pv"][substat_type]["data"][substat]:
                opt_pars = pv_station["substations_pv"][substat_type]["data"][substat]['opt_pars'] 
            else:
                opt_pars = pv_station["substations_pv"][substat_type]["data"][substat]['ap_pars'] 
                    
            tilt = np.rad2deg(opt_pars[0][1])
            azi = np.fmod(np.rad2deg(opt_pars[1][1])+180,360)
            
            dfs = dffit.groupby(dffit.index.time)
            dfs_times = [group for group in dfs]
            # bins_cos = np.linspace(-1.,1.,51)
            # dfs_bins = dffit.groupby(pd.cut(dffit.cos_product,bins_cos))
            
            df_mean = dfs.mean()    
            df_mean["cos_product_mean"] = np.sign(np.sin(np.deg2rad(df_mean.diff_phi)))\
                *np.rad2deg(np.arccos(df_mean.cos_IA))
            sza_index = df_mean.loc[df_mean.sza <= sza_limit].index 
            x_dt_mean = [datetime.datetime.combine(datetime.date(int(year),1,1), t) for t in df_mean.index]
            sza_dt = [datetime.datetime.combine(datetime.date(int(year),1,1), t) for t in sza_index]
            
            #Fit function for mean values
            #split into morning and evening    
            t_zenith = df_mean.iloc[df_mean.cos_IA.argmax()].name
            df_mean_am = df_mean.loc[(df_mean.sza <= sza_limit) & (df_mean.index <= t_zenith)]
            df_mean_pm = df_mean.loc[(df_mean.sza <= sza_limit) & (df_mean.index > t_zenith)]            
            t_zenith_dt = datetime.datetime.combine(datetime.date(int(year),1,1), t_zenith)
            theta_max = df_mean.loc[t_zenith,"cos_product_mean"]
            
            mean_params = {}    
            print(f"Performing mean fit for {year}")
            for df,time in [(df_mean_am,"AM"),(df_mean_pm,"PM")]:
                
                #n_h2o_data_mean = df.n_h2o_mm
                ydata_mean = np.log(df.ratio_Etotpoa)
                cos_IA_mean = df.cos_IA
                #diff_phi_mean = df.diff_phi        
                
                #if time == "AM":
                fitfunc = lambda p, c: p[0] + p[1]/c + p[2]/c**2 # + p[3]/c**3
                # elif time == "PM":
                #     fitfunc = lambda p, c: p[0] + p[1]/c) + p[2]*np.log(1/c**2) + p[3]*np.log(1/c**3) # - p[2] * w**2 / c - p[1] * w**3 / c
                    
                errfunc = lambda p, c, y: y - fitfunc(p, c)
                pinit = [1.,1.,1.]
                
                params = np.zeros(len(pinit)) #(len(df_am),,        
                params = optimize.least_squares(errfunc,pinit,args=(cos_IA_mean,ydata_mean))["x"]  #df.n_h2o_mm
                mean_params.update({time:params})
                
                df_mean.loc[df.index,"fit_ratio_Etotpoa_mean"] = np.exp(fitfunc(params,cos_IA_mean))                
                #dfs_times = (*dfs_times 
                # x_dt_time = [datetime.datetime.combine(datetime.date(int(year),1,1), t) for t in df.index]
            
            fig, axs = plt.subplots(2,1,figsize=(12,12),sharey='all',sharex='all')
            fig.subplots_adjust(hspace=0.1)
            #fig.subplots_adjust(top=0.92)
            for i, (xvar, xlabel) in enumerate([("n_h2o_mm",'Precipitable water (mm)'),("AOD_500",r'AOD$_\mathrm{500nm}$')]):
                print(f"Plotting mean results over time and {xvar} in {year}")
                ax = axs.flatten()[i]
                ax.plot(x_dt_mean,df_mean["ratio_Etotpoa"],color='k',linewidth=2,label=rf'$\langle$GTI$\rangle^{{\theta = {tilt:.2f}}}_{{\phi = {azi:.2f}}}$')
                ax.plot(x_dt_mean,df_mean["ratio_Etotdown"],'k--',label=r'$\langle$GHI$\rangle$')
                
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))    
                                        
                ax.set_xlim([x_dt_mean[0], x_dt_mean[-1]])#, adjustable='box')
                ax.axvspan(x_dt_mean[0],sza_dt[0],alpha=0.2,color='gray')
                ax.axvspan(sza_dt[-1],x_dt_mean[-1],alpha=0.2,color='gray')                                      
                
                x_dt = [datetime.datetime.combine(datetime.date(int(year),1,1), t) for t in dffit.index.time]
                sc = ax.scatter(x_dt,dffit.ratio_Etotpoa,10,c=dffit[xvar],cmap='jet')
                # X, Y, Z = grid(x_dt, dffit.ratio_Etotpoa, dffit[xvar])                
                # sc = ax.contourf(x_dt,dffit.ratio_Etotpoa,dffit[xvar],cmap='jet')
                # #ax.scatter(x_dt,dffit.ratio_Etotdown,3,c=dffit.n_h2o_mm,marker='x',cmap='jet')
                cb = plt.colorbar(sc,ax=ax)
                cb.set_label(xlabel)        
                
                #Plot the fit to the mean ratio
                ax.plot(x_dt_mean,df_mean.fit_ratio_Etotpoa_mean,color = 'r',label=r'$\langle$GTI$\rangle_\mathrm{fit}$',
                        linestyle='--')
                        
                if i == 0:    
                    axtop = ax.twiny()                    
                    axtop.set_xticks(ax.get_xticks())        
                    axtop.set_xbound(ax.get_xbound())
                    szalabels = np.round(df_mean.loc[pd.to_datetime(ax.get_xticks(),unit='D').time,'cos_product_mean'],2)
                    axtop.set_xticklabels(szalabels)
                    #axtop.set_aspect('equal') 
                    axtop.set_xlabel(r"Mean incident angle $\Theta$ ($\circ$)",labelpad=10)
                    axtop.annotate(rf"$\Theta_\mathrm{{min}}$ = {theta_max:.2f}$^\circ$",
                                    xy=(t_zenith_dt + pd.Timedelta('15T'),axtop.get_ylim()[1] 
                                        - 0.1*(axtop.get_ylim()[1] - axtop.get_ylim()[0])),
                                    xycoords='data',fontsize=14)
                
                ax.axvline(x=t_zenith_dt,linestyle='--',color='k',linewidth=2)
                #ax.set_aspect('equal') 
            
            ax.legend(loc='upper center',bbox_to_anchor=(0.5, 1.25),framealpha=1.) #,facecolor='white')            
            ax.set_zorder(1)
            ax.set_xlabel('Time (UTC)')
            ax.set_ylabel(r'$\dfrac{G_\mathrm{si PV}}{G_\mathrm{broadband}}$',position=(-0.25,1),fontsize=18)
            if title_flag:
                fig.suptitle(f'Daily variation in spectral mismatch at {key} in {year}')
            
            plt.savefig(os.path.join(plotpath,f"mean_spectral_fit_{key}_{substat}_SZA_{int(sza_limit)}_{year}.png")
                        ,bbox_inches = 'tight')
            plt.close(fig)
            
def plot_spectral_fit_time_groups(key,pv_station,year,pvrad_config,folder,title_flag):
    """
    

    Parameters
    ----------
    key : string, name of station
    pv_station : dictionary with all information and data from PV station
    year : string with current year (2018 or 2019)        
    pvrad_config: dictionary with inversion configuration            
    folder : string with folder for saving results            
    title_flag : boolean for plot titles

    Returns
    -------
    None.

    """
    
    plt.ioff()
    plt.style.use('my_presi_grid')
    
    sza_limit = pvrad_config["sza_max"]["inversion"]
    
    res_dirs = list_dirs(folder)
    savepath = os.path.join(folder,'Spectral_Fit_Plots')
    if 'Spectral_Fit_Plots' not in res_dirs:
        os.mkdir(savepath)        
        
    res_dirs = list_dirs(savepath)
    savepath = os.path.join(savepath,f"SZA_{int(sza_limit)}")
    if f"SZA_{int(sza_limit)}" not in res_dirs:
        os.mkdir(savepath)    
        
    stat_dirs = list_dirs(savepath)
    savepath = os.path.join(savepath,key)
    if key not in stat_dirs:
        os.mkdir(savepath)
        
    for substat_type in pv_station["substations_pv"]:    
        for substat in pv_station["substations_pv"][substat_type]["data"]:
            substat_dirs = list_dirs(savepath)
            plotpath = os.path.join(savepath,substat)
            if substat not in substat_dirs:
                os.mkdir(plotpath)
    
            dffit = pv_station["substations_pv"][substat_type]["data"][substat][f"df_spectral_fit_{year}"]
            dffit = dffit.loc[dffit.cos_IA != 0]
            dffit["cos_diff_phi"] = np.cos(np.deg2rad(dffit.diff_phi))
            dffit["phi0_real"] = np.fmod(dffit.phi0 + 180,360)
            dffit["cos_product"] = np.sign(np.sin(np.deg2rad(dffit.diff_phi)))\
                *np.cos(np.pi/2 - np.arccos(dffit.cos_IA))
            days = pd.to_datetime(dffit.index.date).unique().strftime('%Y-%m-%d')
            
            if 'opt_pars' in pv_station["substations_pv"][substat_type]["data"][substat]:
                opt_pars = pv_station["substations_pv"][substat_type]["data"][substat]['opt_pars'] 
            else:
                opt_pars = pv_station["substations_pv"][substat_type]["data"][substat]['ap_pars'] 
                    
            tilt = np.rad2deg(opt_pars[0][1])
            azi = np.fmod(np.rad2deg(opt_pars[1][1])+180,360)
            
            dfs = dffit.groupby(dffit.index.time)
            dfs_times = [group for group in dfs]
            # bins_cos = np.linspace(-1.,1.,51)
            # dfs_bins = dffit.groupby(pd.cut(dffit.cos_product,bins_cos))
            
            df_mean = dfs.mean()    
            df_mean["cos_product_mean"] = np.sign(np.sin(np.deg2rad(df_mean.diff_phi)))\
                *np.rad2deg(np.arccos(df_mean.cos_IA))
            sza_index = df_mean.loc[df_mean.sza <= sza_limit].index 
            x_dt_mean = [datetime.datetime.combine(datetime.date(int(year),1,1), t) for t in df_mean.index]
            sza_dt = [datetime.datetime.combine(datetime.date(int(year),1,1), t) for t in sza_index]

            #Plots divided up into time categories
            num_plot_cols = int(np.ceil(np.sqrt(len(dfs_times))))
            if num_plot_cols**2 - len(dfs_times) > num_plot_cols:
                num_plot_rows = int(num_plot_cols - 1)
            else:
                num_plot_rows = int(num_plot_cols)
            
            # for i, (time, df) in enumerate(dfs_times):
            #     dfs_times[i] = (*dfs_times[i],df_mean.loc[time,"fit_ratio_Etotpoa_mean"])        
                
            for xvar, xlabel in [("n_h2o_mm",'Precipitable water (mm)'),("AOD_500",r'AOD$_\mathrm{500nm}$')]:
                fig,axs = plt.subplots(num_plot_rows,num_plot_cols,sharex='all',figsize=(15,15))
                fig.subplots_adjust(hspace=0.4,wspace=0.8)
                fig.subplots_adjust(top=0.92)
                        
                print(f"Performing mean fit for {xvar} in {year}")
                fit_params = []
                for i, (time, df) in enumerate(dfs_times):
                    #dfs_times[i] = (*dfs_times[i],df_mean.loc[time,"fit_ratio_Etotpoa_mean"])        
                    ax = axs.flatten()[i] 
                    df_hours = np.array([h for h in df.index.hour]) + np.array([m/60 for m in df.index.minute])
                    sc = ax.scatter(df[xvar],df.ratio_Etotpoa,5) #,c=df.cos_IA,cmap='jet')
                            
                    #ax.plot(df_mean.loc[time,xvar],df_mean.loc[time,"fit_ratio_Etotpoa_mean"],marker='x',color='k')
                    
                    ax.set_title(f"{time}",fontsize=12) #.strftime('%H:%M')
                    ax.tick_params(axis='both', labelsize=12 )
                    #ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                    
                    fitfunc = lambda x, w: (x[0] + x[1]*w + x[2]*w**2) # + x[3]*w**3 #dfs_times[i][2] + 
                    errfunc = lambda x, w, y: y - fitfunc(x, w)
                    xinit = [1.,1.,1.]
                    
                    params = np.zeros(len(xinit)) #(len(df_am),,        
                    fit = optimize.least_squares(errfunc,xinit,args=(df[xvar],
                                          np.log(df.ratio_Etotpoa)))
                    params = fit["x"]
                    fit_params.append(params)
                    
                    #print(params)
                    df[f"ratio_Etotpoa_fit_new_{xvar}"] = np.exp(fitfunc(params,df[xvar]))
                    
                    #dfs_times[i] = (*dfs_times[i],params)                
                    
                    if df_mean.loc[time,"sza"] > sza_limit:
                        linestring = ':'
                        colorstring='gray'
                    else:
                        linestring = '-'
                        colorstring = 'r'
                    
                    ax.plot(df.sort_values(xvar)[xvar],df.sort_values(xvar)[f"ratio_Etotpoa_fit_new_{xvar}"],
                            color=colorstring,linestyle=linestring)
                    # ax.plot(df.sort_values(xvar)[xvar],df.sort_values(xvar)["ratio_Etotpoa_fit"],
                    #         color='g',linestyle=linestring)
                
                df_mean[f"fit_params_{xvar}"] = pd.Series(fit_params,index=df_mean.index)  
                
                i += 1
                while i < len(axs.flatten()):
                     axs.flatten()[i].set_visible(False)
                     i += 1        
                    
                    #cb = plt.colorbar(sc,ax=ax)
                if title_flag:
                    fig.suptitle(f'Spectral mismatch at different times of day, at {key} in {year}')
                fig.add_subplot(111, frameon=False)
                plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
                plt.grid(False)
                #plt.xlabel(r'$n_\mathrm{H_2O}$ (mm)')
                plt.xlabel(f"{xlabel}")
                plt.ylabel(r'$\dfrac{GTI_\mathrm{si PV}}{GTI_\mathrm{broadband}}$',labelpad=20)
                plt.savefig(os.path.join(plotpath,f"spectral_fit_groups_{xvar}_{key}_{substat}_SZA_{int(sza_limit)}_{year}.png"),bbox_inches = 'tight')
                plt.close(fig)
                
            #Combining H2O and AOD
            fig,axs = plt.subplots(num_plot_rows,num_plot_cols,sharex='all',figsize=(15,15))
            fig.subplots_adjust(hspace=0.4,wspace=0.8)
            fig.subplots_adjust(top=0.92)
                    
            fit_params = []
            print(f"Performing combined fit in {year}")
            for i, (time, df) in enumerate(dfs_times):
                #dfs_times[i] = (*dfs_times[i],df_mean.loc[time,"fit_ratio_Etotpoa_mean"])        
                ax = axs.flatten()[i] 
                df["n_h2o_AOD_product"] = df["n_h2o_mm"]*df["AOD_500"]
                df_hours = np.array([h for h in df.index.hour]) + np.array([m/60 for m in df.index.minute])
                sc = ax.scatter(df["n_h2o_mm"],df.ratio_Etotpoa,5) #,c=df.cos_IA,cmap='jet')
                        
                #ax.plot(df_mean.loc[time,xvar],df_mean.loc[time,"fit_ratio_Etotpoa_mean"],marker='x',color='k')
                
                ax.set_title(f"{time}",fontsize=12) #.strftime('%H:%M')
                ax.tick_params(axis='both', labelsize=12 )
                #ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                
                fitfunc = lambda x, w, a: (x[0] + x[1]*w + x[2]*a) # + x[3]*a**2) # + x[3]*w**3 #dfs_times[i][2] + 
                errfunc = lambda x, w, a, y: y - fitfunc(x, w, a)
                xinit = [1.,1.,1.]
                
                params = np.zeros(len(xinit)) #(len(df_am),,        
                fit = optimize.least_squares(errfunc,xinit,args=(df["n_h2o_mm"],df["AOD_500"],
                                      np.log(df.ratio_Etotpoa)))
                params = fit["x"]
                fit_params.append(params)                
                
                #print(params)
                df["ratio_Etotpoa_fit_new_combo"] = np.exp(fitfunc(params,df["n_h2o_mm"],df["AOD_500"]))
                #df_mean.loc[df.index,"fit_ratio_Etotpoa_mean"] = df["ratio_Etotpoa_fit_new_combo"]
                
                #dfs_times[i] = (*dfs_times[i],params)                
                
                if df_mean.loc[time,"sza"] > sza_limit:
                    linestring = ':'
                    colorstring='gray'
                else:
                    linestring = '-'
                    colorstring = 'r'
                
                ax.plot(df.sort_values("n_h2o_mm")["n_h2o_mm"],
                        df.sort_values("n_h2o_mm")["ratio_Etotpoa_fit_new_combo"],
                        color=colorstring,linestyle=linestring)
                # ax.plot(df.sort_values("n_h2o_mm")["n_h2o_mm"],
                #         df.sort_values("n_h2o_mm")["ratio_Etotpoa_fit"],
                #         color='g',linestyle=linestring)
            
            df_mean["fit_params_combo"] = pd.Series(fit_params,index=df_mean.index)  
            
            i += 1
            while i < len(axs.flatten()):
                 axs.flatten()[i].set_visible(False)
                 i += 1        
                
                #cb = plt.colorbar(sc,ax=ax)
            if title_flag:
                fig.suptitle(f'Spectral mismatch at different times of day, at {key} in {year}')
            fig.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.grid(False)
            #plt.xlabel(r'$n_\mathrm{H_2O}$ (mm)')
            plt.xlabel("Precipitable water (mm)")
            plt.ylabel(r'$\dfrac{GTI_\mathrm{si PV}}{GTI_\mathrm{broadband}}$',labelpad=20)            
            plt.savefig(os.path.join(plotpath,f"spectral_fit_groups_combo_{key}_{substat}_SZA_{int(sza_limit)}_{year}.png")
                        ,bbox_inches = 'tight')
            plt.close(fig)
    
def plot_grid_combined_scatter(dict_pv_stations,list_stations,pvrad_config,T_model,folder):
    """
    

    Parameters
    ----------
    dict_pv_stations : dictionary with all PV stations
    list_stations : list of relevant stations
    pvrad_config : dictionary with inversion configuration        
    T_model : string with temperatur emodel used        
    folder : string with folder for saving results

    Returns
    -------
    None.

    """
    
    
    plt.ioff()
    plt.style.use('my_paper_grid')
    
    res_dirs = list_dirs(folder)
    savepath = os.path.join(folder,'GTI_Plots')
    if 'GTI_Plots' not in res_dirs:
        os.mkdir(savepath)    
        
    res_dirs = list_dirs(savepath)
    savepath = os.path.join(savepath,'Scatter')
    if 'Scatter' not in res_dirs:
        os.mkdir(savepath)
    
    numplots = len(list_stations)
    years = ["mk_" + campaign.split('_')[1] for campaign in pvrad_config["calibration_source"]]    
    stations_label = '_'.join([s[0] for s in list_stations])
        
    max_gti = 0.
    min_z = 500.; max_z = 0.
    fig, axs = plt.subplots(numplots,2,sharex='all',sharey='all')#,figsize=(16,16))    
    
    plot_data = []
    print("Printing grid of scatter plots...")
    for i, ax in enumerate(axs.flatten()):
        year = years[np.fmod(i,2)]
        station = list_stations[int((i - np.fmod(i,2))/2)][0]
        if station != "PV_11":
            substat_type = list_stations[int((i - np.fmod(i,2))/2)][1]
        else:
            substat_type = list_stations[int((i - np.fmod(i,2))/2)][1] + f'_{year.split("_")[-1]}'
        if station != "PV_11":
            substat = list_stations[int((i - np.fmod(i,2))/2)][2]
        else:
            k = np.fmod(i,2) + 1
            substat = list_stations[int((i - np.fmod(i,2))/2)][2] + f"_{k}"
            
        pv_station = dict_pv_stations[station]        
        #print(f"Adding plot for {station}, {year}")
        
        timeres = pv_station["substations_pv"][substat_type]["t_res_inv"]
        dfname = 'df_pv_' + year.split('_')[-1] + '_' + timeres
        # dfcosmo = 'df_cosmo_' + year.split('_')[-1]
        dataframe = pv_station[dfname] #.xs(substat,level='substat',axis=1)                             
        
        pyrname = pvrad_config["pv_stations"][station][substat_type]["pyr_down"][year][1]
        if "Horiz" in pyrname:
            pyrname = pyrname.split('_')[0] + "_32S"
            
        pyrcal_config = load_yaml_configfile(config["pyrcal_configfile"][year])
        
        pyr_station = pvrad_config["pv_stations"][key][substat_type]["pyr_down"][year][0]
        radname = pyrcal_config["pv_stations"][pyr_station]["substat"][pyrname]["name"]                
        if '_' in pyrname:
            pyrlabel = f'{pyrname.split("_")[0]}_{{{pyrname.split("_")[1]}}}'                    
        else: pyrlabel = pyrname                
        
        #Keep only data for which an inversion exists
        dataframe = dataframe.loc[dataframe['P_meas_W',substat].notna()]

        #fig, ax = plt.subplots(figsize=(9,8))
                    
        gti_data = dataframe.loc[:,[(radname,pyrname),("Etotpoa_pv_inv",substat)]]
        gti_data.dropna(axis=0,how='any',inplace=True)
        
        gti_ref = gti_data[radname,pyrname].values
        gti_inv = gti_data["Etotpoa_pv_inv",substat].values
        xy = np.vstack([gti_ref,gti_inv])
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        
        plot_data.append((gti_ref[idx], gti_inv[idx], z[idx]))
        
        max_gti = np.max([max_gti,gti_ref.max(),gti_inv.max()])
        max_gti = np.ceil(max_gti/100)*100    
        max_z = np.max([max_z,np.max(z)])
        min_z = np.min([min_z,np.min(z)])
        
    norm = plt.Normalize(min_z,max_z)    
    
    for i, ax in enumerate(axs.flatten()):
        year = years[np.fmod(i,2)]
        station = list_stations[int((i - np.fmod(i,2))/2)][0]
        if station != "PV_11" and station != "MS_02":
            substat_type = list_stations[int((i - np.fmod(i,2))/2)][1]
        else:
            substat_type = list_stations[int((i - np.fmod(i,2))/2)][1] + f'_{year.split("_")[-1]}'
        if station != "PV_11" and station != "MS_02":
            substat = list_stations[int((i - np.fmod(i,2))/2)][2]
        else:
            k = np.fmod(i,2) + 1
            substat = list_stations[int((i - np.fmod(i,2))/2)][2] + f"_{k}"
            
        pv_station = dict_pv_stations[station]        
        
        timeres = pv_station["substations_pv"][substat_type]["t_res_inv"]
        dfname = 'df_pv_' + year.split('_')[-1] + '_' + timeres
        # dfcosmo = 'df_cosmo_' + year.split('_')[-1]
        dataframe = pv_station[dfname]
        
        pyrname = pvrad_config["pv_stations"][station][substat_type]["pyr_down"][year][1]
        
        stats = pv_station["stats"][year][substat]
        station_label = "".join([s for s in station.split('_')])
        year_label = year.split('_')[-1]
     
        sc = ax.scatter(plot_data[i][0],plot_data[i][1], s=20, c=plot_data[i][2], 
                        cmap="jet",norm=norm)
        
        ax.plot([0, 1], [0, 1], ls = '--', transform=ax.transAxes,c='k')
        ax.set_title(f"{station_label}, {year_label}",fontsize=14)
        
        ax.annotate(rf"MBE = {stats[f'MBE_GTI_{pyrname}_Wm2']:.2f} W/m$^2$ ({stats[f'rMBE_GTI_{pyrname}_%']:.2f} %)",
                 xy=(0.05,0.92),xycoords='axes fraction',fontsize=10)     
        ax.annotate(rf"RMSE = {stats[f'RMSE_GTI_{pyrname}_Wm2']:.2f} W/m$^2$ ({stats['rRMSE_GTI_{pyrname}_%']:.2f} %)",
                 xy=(0.05,0.85),xycoords='axes fraction',fontsize=10)  
        ax.annotate(rf"n = {stats['n_delta']:.0f}",
                 xy=(0.05,0.78),xycoords='axes fraction',fontsize=10)  
        
        ax.grid(False)
        if max_gti < 1400:
            ax.set_xticks([0,400,800,1200])
        else:
            ax.set_xticks([0,400,800,1200,1400])
        #ax.set_xticks([0,400,800,1200])
            
    cb = plt.colorbar(sc,ticks=[min_z,max_z],cax = fig.add_axes([0.89, 0.11, 0.02, 0.77]),pad=0.5)    
    cb.set_ticklabels(["Low", "High"])  # horizontal colorbar
    cb.set_label("Frequency", labelpad=-20, fontsize=14)
    
    #Set axis limits
    for i, ax in enumerate(axs.flatten()):
        ax.set_xlim([0.,max_gti])
        ax.set_ylim([0.,max_gti])
        ax.set_aspect('equal')
    
    fig.subplots_adjust(hspace=0.15,wspace=-0.1)    
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel(r"$G_\mathrm{tot,pyranometer}^{\angle}$ (W/m$^2$)")
    plt.ylabel(rf"$G_\mathrm{{tot,{substat.replace('_',' ')},inv}}^{{\angle}}$ (W/m$^2$)")
    
    #fig.tight_layout()
    plt.savefig(os.path.join(savepath,f"gti_scatter_hist_combo_grid_{T_model['model']}"\
             f"_{T_model['type']}_{timeres}_{stations_label}.png"),bbox_inches = 'tight')  
    
def plot_histogram_deviations(key,pv_station,pvrad_config,folder):
    """
    

    Parameters
    ----------
    key : string, name of station
    pv_station : dictionary with all information and data from PV station
    year : string with current year (2018 or 2019)        
    pvrad_config: dictionary with inversion configuration            
    folder : string with folder for saving results            
    

    Returns
    -------
    None.

    """
    plt.ioff()
    plt.style.use('my_paper')        
    
    res_dirs = list_dirs(folder)
    savepath = os.path.join(folder,'GTI_Plots')
    if 'GTI_Plots' not in res_dirs:
        os.mkdir(savepath)    
        
    res_dirs = list_dirs(savepath)
    savepath = os.path.join(savepath,'Stats')
    if 'Stats' not in res_dirs:
        os.mkdir(savepath)
        
    campaigns = pvrad_config["calibration_source"]    
    for substat_type in pv_station["substations_pv"]:    
        for substat in pv_station["substations_pv"][substat_type]["data"]:                        
            
            fig, ax = plt.subplots(figsize=(9,8))
            
            if 'opt_pars' in pv_station["substations_pv"][substat_type]["data"][substat]:
                opt_pars = pv_station["substations_pv"][substat_type]["data"][substat]['opt_pars'] 
            else:
                opt_pars = pv_station["substations_pv"][substat_type]["data"][substat]['ap_pars'] 
                    
            tilt = np.rad2deg(opt_pars[0][1])
            azi = np.fmod(np.rad2deg(opt_pars[1][1])+180,360)
            
            ax.annotate(rf"$\theta = {tilt:.2f}^\circ, \phi = {azi:.2f}^\circ$",
                 xy=(0.14,0.9),xycoords='figure fraction',fontsize=14)                     
            
            for i, campaign in enumerate(campaigns):
                year = f"mk_{campaign.split('_')[1]}"
                if year in pv_station["substations_pv"][substat_type]["source"]:                                
                    print('Generating histograms for %s, %s, %s' % (key,substat,year))
                    timeres = pv_station["substations_pv"][substat_type]["t_res_inv"]
                    dfname = 'df_pv_' + year.split('_')[-1] + '_' + timeres
                    # dfcosmo = 'df_cosmo_' + year.split('_')[-1]
                    dataframe = pv_station[dfname] #.xs(substat,level='substat',axis=1)             
                    
                    stats = pv_station["stats"][year][substat]
                    
                    pyrname = pvrad_config["pv_stations"][key][substat_type]["pyr_down"][year][1]
                    if "Horiz" in pyrname:
                        pyrname = pyrname.split('_')[0] + "_32S"
                        
                    pyrcal_config = load_yaml_configfile(config["pyrcal_configfile"][year])
                    
                    pyr_station = pvrad_config["pv_stations"][key][substat_type]["pyr_down"][year][0]
                    radname = pyrcal_config["pv_stations"][pyr_station]["substat"][pyrname]["name"]                
                    if '_' in pyrname:
                        pyrlabel = f'{pyrname.split("_")[0]}_{{{pyrname.split("_")[1]}}}'                    
                    else: pyrlabel = pyrname
                    
                    #Keep only data for which an inversion exists
                    dataframe = dataframe.loc[dataframe['P_meas_W',substat].notna()]
                    
                    delta_max = np.ceil(stats[f"max_Delta_GTI_plus_{pyrname}_Wm2"]/10)*10
                    delta_min = np.floor(stats[f"max_Delta_GTI_minus_{pyrname}_Wm2"]/10)*10
                    
                    bins = np.arange(delta_min,delta_max+10,10)                            
                    
                    ax.hist(dataframe["delta_GTI_Wm2",substat],bins=bins,
                            label=year.split('_')[-1],alpha=0.3,log=True)
                    
                    if key == "PV_11":
                        offset = 0.0 #4/(i+1)
                    else:
                        offset = 0.08
                    ax.annotate(rf"MBE$_{{{year.split('_')[-1]}}}$ = {stats[f'MBE_GTI_{pyrname}_Wm2']:.3f} W/m$^2$",
                                 xy=(0.14,0.86 - i*offset),xycoords='figure fraction',fontsize=14)     
                    ax.annotate(rf"RMSE$_{{{year.split('_')[-1]}}}$ = {stats[f'RMSE_GTI_{pyrname}_Wm2']:.3f} W/m$^2$",
                                 xy=(0.14,0.82 - i*offset),xycoords='figure fraction',fontsize=14)     
                                
            ax.set_xlabel(r"$\Delta G_\mathrm{{tot}}^{{\angle}}$")
            ax.set_ylabel("Frequency")
            ax.legend()
                        
            plt.savefig(os.path.join(savepath,f"delta_gti_hist_{timeres}_{key}_{substat}_{pyrname}.png")
                            ,bbox_inches = 'tight')   
            plt.close(fig)  
                
def combined_stats(dict_combo_stats,year,timeres_list):
    """
    

    Parameters
    ----------
    dict_combo_stats : dictionary with combined stats for all stations
    year : string with year in question (2018 or 2019)        
    timeres_list : list of time resolutions for stats calculations

    Returns
    -------
    None.

    """
    for timeres in timeres_list:        
        if f"df_delta_all_{timeres}" in dict_combo_stats:
            #Stack all values on top of each other for combined stats
            df_delta_all = dict_combo_stats[f"df_delta_all_{timeres}"].stack(dropna=True)
                        
            rmse = ((((df_delta_all.delta_GTI_Wm2.stack())**2).mean())**0.5)
            mad = abs(df_delta_all.delta_GTI_Wm2.stack()).mean()
            mbe = df_delta_all.delta_GTI_Wm2.stack().mean()
            
            mean_obs = df_delta_all.GTI_Pyr_ref.stack().mean()
            rmbe = mbe/mean_obs*100
            rrmse = rmse/mean_obs*100
            
            delta_max_plus = df_delta_all.delta_GTI_Wm2.stack().max()
            delta_max_minus = df_delta_all.delta_GTI_Wm2.stack().min()
            
            n_delta = len(df_delta_all.delta_GTI_Wm2.stack().dropna())
            
            dict_combo_stats.update({timeres:{}})
            dict_combo_stats[timeres].update({"n_delta":n_delta})
            dict_combo_stats[timeres].update({"RMSE_GTI_Wm2":rmse})
            dict_combo_stats[timeres].update({"rRMSE_GTI_%":rrmse})
            dict_combo_stats[timeres].update({"MAD_GTI_Wm2":mad})
            dict_combo_stats[timeres].update({"MBE_GTI_Wm2":mbe})
            dict_combo_stats[timeres].update({"rMBE_GTI_%":rmbe})
            dict_combo_stats[timeres].update({"mean_GTI_Wm2":mean_obs})
            dict_combo_stats[timeres].update({"max_Delta_GTI_plus_Wm2":delta_max_plus})
            dict_combo_stats[timeres].update({"max_Delta_GTI_minus_Wm2":delta_max_minus})
        
            print(f"{year}: combined statistics at {timeres} from "\
                  f"{dict_combo_stats[f'df_delta_all_{timeres}'].columns.levels[2].to_list()}"\
                  f" calculated with {n_delta} measurements")
            print(f"Combined RMSE for GTI in {year} is {rmse} ({rrmse:.1f} %)")
            print(f"Combined MAE for GTI in {year} is {mad}")
            print(f"Combined MBE for GTI in {year} is {mbe} ({rmbe:.1f} %)")
        
def plot_all_combined_scatter(dict_stats,list_stations,pvrad_config,T_model,folder,title_flag):
    """
    

    Parameters
    ----------
    dict_stats : dictionary with statistics of deviations
    list_stations : list of PV stations
    pvrad_config : dictionary with inversion configuration        
    T_model : string with temperature model
    folder : string with folder for saving results and plots
    title_flag : boolean for plot titles

    Returns
    -------
    None.

    """
    
    
    plt.ioff()
    plt.style.use('my_presi_grid')        
    
    res_dirs = list_dirs(folder)
    savepath = os.path.join(folder,'GTI_Plots')
    if 'GTI_Plots' not in res_dirs:
        os.mkdir(savepath)    
        
    res_dirs = list_dirs(savepath)
    savepath = os.path.join(savepath,'Scatter')
    if 'Scatter' not in res_dirs:
        os.mkdir(savepath)
        
    years = ["mk_" + campaign.split('_')[1] for campaign in pvrad_config["calibration_source"]]    
    stations_label = '_'.join(["".join(s.split('_')) for s in list_stations])
    
    for timeres in pvrad_config["timeres_comparison"]:
    
        fig, axs = plt.subplots(1,len(years),sharex='all',sharey='all')
        #cbar_ax = fig.add_axes([.27, .75, .43,.015]) 
        
        print(f"Plotting combined frequency scatter plot for {timeres}...please wait....")
        plot_data = []
        max_gti = 0.; min_z = 500.; max_z = 0.
        for i, ax in enumerate(axs.flatten()):            
            year = years[i]
            
            gti_data = dict_stats[year][f"df_delta_all_{timeres}"].stack()\
                .loc[:,["GTI_PV_inv","GTI_Pyr_ref"]].stack().dropna(how='any')
            
            gti_ref = gti_data["GTI_Pyr_ref"].values.flatten()
            gti_inv = gti_data["GTI_PV_inv"].values.flatten()
            xy = np.vstack([gti_ref,gti_inv])
            z = gaussian_kde(xy)(xy)
            idx = z.argsort()
            
            #plot_data.append((gti_ref[idx], gti_inv[idx], z[idx]))
            
            # sc = sns.kdeplot(x=gti_ref,y=gti_inv,cbar=i==0,ax=ax,cmap="icefire",fill=True,#bins=100,
            #       cbar_kws={'orientation': 'horizontal'},
            #       cbar_ax=None if i else cbar_ax)
            
            # if i == 0:
            #     sc.figure.axes[-1].xaxis.set_ticks(sc.figure.axes[-1].get_xlim())                    
            #     sc.figure.axes[-1].tick_params(axis="x",direction="in", pad=-30)
            #     sc.figure.axes[-1].xaxis.set_ticklabels(["Low", "High"])
            #     sc.figure.axes[-1].set_xlabel("PDF",size=16,labelpad=-30)
            
            max_gti = np.max([max_gti,gti_ref.max(),gti_inv.max()])
            max_gti = np.ceil(max_gti/100)*100    
            max_z = np.max([max_z,np.max(z)])
        #     min_z = np.min([min_z,np.min(z)])
            
        # norm = plt.Normalize(min_z,max_z)    
        
        # for i, ax in enumerate(axs.flatten()):
            # year = years[i]
            
            sc = ax.scatter(gti_ref[idx],gti_inv[idx], s=8, c=z[idx], 
                            cmap="plasma") #,
                            #norm=norm)
            
            ax.plot([0, 1], [0, 1], ls = '--', transform=ax.transAxes,c='k')
            #ax.set_title(f"{station_label}, {year_label}",fontsize=14)
            
            print(f"Using {dict_stats[year][timeres]['n_delta']} data points for {year} plot")
            ax.annotate(rf"MBE = {dict_stats[year][timeres]['MBE_GTI_Wm2']:.2f} W m$^{{-2}}$ ({dict_stats[year][timeres]['rMBE_GTI_%']:.1f} %)" "\n" \
                        rf"RMSE = {dict_stats[year][timeres]['RMSE_GTI_Wm2']:.2f} W m$^{{-2}}$ ({dict_stats[year][timeres]['rRMSE_GTI_%']:.1f} %)" "\n"\
                            r"$\langle G_\mathrm{ref} \rangle$ =" \
                            rf" {dict_stats[year][timeres][f'mean_GTI_Wm2']:.2f} W m$^{{-2}}$" "\n"\
                        rf"n = ${dict_stats[year][timeres]['n_delta']:.0f}$",
                     xy=(0.05,0.73),xycoords='axes fraction',fontsize=10,bbox = dict(facecolor='lightgrey',edgecolor='k', alpha=0.5),
                     horizontalalignment='left',multialignment='left')     
            # ax.annotate(rf"RMSE = {dict_stats[year][timeres]['RMSE_GTI_Wm2']:.2f} W/m$^2$",
            #          xy=(0.05,0.85),xycoords='axes fraction',fontsize=10,bbox = dict(facecolor='lightgrey', alpha=0.3))  
            # ax.annotate(rf"n = {dict_stats[year][timeres]['n_delta']:.0f}",
            #          xy=(0.05,0.78),xycoords='axes fraction',fontsize=10,bbox = dict(facecolor='lightgrey', alpha=0.3))                  
            #ax.set_xticks([0,400,800,1200])
                
        cb = fig.colorbar(sc,ticks=[np.min(z),np.max(z)], 
                          ax=axs[:2], shrink=0.6, location = 'top', 
                            aspect=20)    
        cb.set_ticklabels(["Low", "High"])  # horizontal colorbar
        cb.set_label("PDF",labelpad=-10, fontsize=16) 
        
        #Set axis limits
        for i, ax in enumerate(axs.flatten()):
            ax.set_xlim([0.,max_gti])
            ax.set_ylim([0.,max_gti])
            ax.set_aspect('equal')
            ax.grid(color='gray',linestyle=':')
            # if max_gti < 1400:
            #     ax.set_xticks([0,400,800,1200])
            # else:
            #     ax.set_xticks([0,400,800,1200,1400])
            if i == 0:
                ax.set_xlabel(r"$G_\mathrm{tot,pyranometer}^{\angle}$ (W m$^{-2}$)",position=(1.1,0))
                ax.set_ylabel(r"$G_\mathrm{{tot,inv}}^{{\angle}}$ (W m$^{-2}$)")
        
        #fig.subplots_adjust(wspace=0.1)    
        # fig.add_subplot(111, frameon=False)
        # plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        # plt.grid(False)
        
        plt.savefig(os.path.join(savepath,f"gti_scatter_hist_combo_all_{timeres}_{T_model['model']}_"\
                 f"{T_model['type']}_{stations_label}.png"),bbox_inches = 'tight')  
        
def save_combo_stats(dict_stats,list_stations,pvrad_config,T_model,folder):
    """
    

    Parameters
    ----------
    dict_stats : dictioanry with combined statistics for different averaging times
    list_stations : list of PV stations
    pvrad_config : dictionary with PV inversion configuration
    T_model : string, temperature model used
    folder : string, folder for saving resul    

    Returns
    -------
    None.

    """
        
    stations_label = '_'.join(["".join(s.split('_')) for s in list_stations])
    
    filename = f"gti_combo_results_stats_{T_model['model']}_{stations_label}.data"
    
    with open(os.path.join(folder,filename), 'wb') as filehandle:  
        # store the data as binary data stream
        pickle.dump((dict_stats, list_stations, pvrad_config, T_model), filehandle)
    
    #Write combined results to CSV
    for measurement in pvrad_config["inversion_source"]:
        year = f"mk_{measurement.split('_')[1]}"
        
        for timeres in pvrad_config["timeres_comparison"]:
            if f"df_delta_all_{timeres}" in dict_stats[year]:
                
                dataframe = dict_stats[year][f"df_delta_all_{timeres}"]
                filename_csv = f'gti_combo_results_{timeres}_{year}_{T_model["model"]}.dat'
                f = open(os.path.join(folder,"CSV_Results",filename_csv), 'w')
                f.write(f'#Global tilted irradiance inverted from PV data at {timeres} resolution\n')                    
                f.write('#Comparison data from pyranometer (GTI_Pyr_ref)\n')                    
                f.write(f'#Stations considered: {list_stations}\n')                    
                
                f.write('\n#Multi-index: first line ("variable") refers to measured quantity\n')
                f.write('#second line ("substat") refers to sensor used for inversion onto GTI\n')
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
#This program makes plots of irradiance ratios, simple optical depths from the
#two-stream model as well as cloud fractions
#def main():
import argparse
    
parser = argparse.ArgumentParser()
#parser.add_argument("configfile", help="yaml file containing config")
parser.add_argument("-s","--station",nargs='+',help="station for which to perform inversion")
parser.add_argument("-c","--campaign",nargs='+',help="measurement campaign for which to perform simulation")
args = parser.parse_args()
   
config_filename = "config_PV2RAD_MetPVNet_messkampagne.yaml" #os.path.abspath(args.configfile)
 
config = load_yaml_configfile(config_filename)

#Load PV configuration
pvcal_config = load_yaml_configfile(config["pvcal_configfile"])

#Load PV configuration
pvrad_config = load_yaml_configfile(config["pvrad_configfile"])

#Load radiative transfer configuration
rt_config = load_yaml_configfile(config["rt_configfile"])

homepath = os.path.expanduser('~')

if args.station:
    stations = args.station
    if stations[0] == 'all':
        stations = 'all'
else:
    #Stations for which to perform inversion
    stations = ["PV_11","PV_12"] #,"PV_01","PV_12"] #"all" #["PV_12"] #,"PV_15","PV_11","PV_19","PV_06","PV_01","PV_04"] #pvrad_config["stations"]

#Choose measurement campaign
if args.campaign:
    pvrad_config["inversion_source"] = args.campaign
#%%Load inversion results
#Load calibration results, including DISORT RT simulation for clear sky days and COSMO data.
#print('Loading PV2POARAD inversion results')

# pvsys, station_list, results_folder = \
# load_pv2poarad_inversion_results(rt_config, pvcal_config, pvrad_config, stations, homepath)

#%%Plotting
plt.close('all')

plot_flags = config["plot_flags"]

T_model = pvcal_config["T_model"]

combo_stats = {}
print('Performing statistical analysis and plotting results of PV2POARAD for %s' % stations)
for campaign in pvrad_config["calibration_source"]:    
    year = "mk_" + campaign.split('_')[1]
    yrname = year.split('_')[-1]
    
    pyrcal_config = load_yaml_configfile(config["pyrcal_configfile"][year])
    
    pvsys, station_list, results_folder = \
    load_pyr2cf_pv2poarad_results(rt_config, pyrcal_config, pvcal_config, pvrad_config, 
                                  campaign, stations, homepath)    
    
    combo_stats.update({year:{}})
    combo_stats[year].update({f"df_{yrname}_stats":pd.DataFrame(index=station_list)})
    dfs_deviations = {}
    for tres in pvrad_config["timeres_comparison"]:
        dfs_deviations.update({tres:[]})
    #Go through stations, calculate statistics and make plots for GTI and CF
    for key in pvsys:
        #Replace pyranometer data with cosine bias corrected data!     06.08.2023            
        for substat_type in pvsys[key]["substations_pv"]:    
            for substat in pvsys[key]["substations_pv"][substat_type]["data"]:
                if year in pvsys[key]["substations_pv"][substat_type]["source"]:                                
                    
                    yrname = year.split('_')[-1]
                    timeres = pvsys[key]["substations_pv"][substat_type]["t_res_inv"]                    
                    dfname = f"df_pv_{year.split('_')[-1]}_{timeres}"
                    dfname_pyr = f"df_pyr_{year.split('_')[-1]}_{timeres}"
        
                    pyrname = pvrad_config["pv_stations"][key][substat_type]["pyr_down"][year][1]
                    if "Pyr" in pyrname:
                        print(f"Replacing pyranometer data for {pyrname} at {timeres} with cosine bias corrected values")
                        
                        pyr_station = pvrad_config["pv_stations"][key][substat_type]["pyr_down"][year][0]
                        radname = pyrcal_config["pv_stations"][pyr_station]["substat"][pyrname]["name"]                
                    
                        if dfname_pyr in pvsys[key]:
                            pvsys[key][dfname][(radname,pyrname)] = pvsys[key][dfname_pyr][(radname,pyrname)]
                        else:
                            pvsys[key][dfname][(radname,pyrname)] = downsample_pyranometer(pvsys[key][f"df_pyr_{year.split('_')[-1]}_1min"]\
                                [(radname,pyrname)],"1min",timeres)
                
        #Calculate statistics
        if combo_stats[year][f"df_{yrname}_stats"].empty:
            combo_stats[year][f"df_{yrname}_stats"] = calc_statistics_gti(key,pvsys[key],year,
                                          pvrad_config,pyrcal_config,results_folder)                            
        else:
            combo_stats[year][f"df_{yrname}_stats"] = pd.concat([combo_stats[year][f"df_{yrname}_stats"],
                         calc_statistics_gti(key,pvsys[key],year,pvrad_config,pyrcal_config,results_folder)],axis=0)
        
        for timeres in pvrad_config["timeres_comparison"]:
            if f"df_stats_{timeres}_{yrname}" in pvsys[key]:
                dfs_deviations[timeres].append(pvsys[key][f"df_stats_{timeres}_{yrname}"])
            else:
                dfs_deviations[timeres].append(pd.DataFrame())
        
        if plot_flags["scatter"]:
            scatter_plot_gti_hist(key, pvsys[key],pvrad_config,results_folder)
        
        if plot_flags["compare"]:
            #Plot irradiance ratio        
            plot_irradiance_comparison(key,pvsys[key],year,pvrad_config,results_folder)
    
            #Plot cloud fraction        
            plot_cloud_fraction(key,pvsys[key],year,pvrad_config,results_folder)
        
        if plot_flags["spectral"]:
            #Plot mean spectral fit
            plot_mean_spectral_fit(key,pvsys[key],yrname,pvrad_config,results_folder,plot_flags["titles"])
            
            #PLot time grouped fits
            plot_spectral_fit_time_groups(key,pvsys[key],yrname,pvrad_config,results_folder,plot_flags["titles"])            
    
    for timeres in pvrad_config["timeres_comparison"]:
        combo_stats[year][f"df_delta_all_{timeres}"] = pd.concat(dfs_deviations[timeres],axis=1)
        if combo_stats[year][f"df_delta_all_{timeres}"].empty:
            del combo_stats[year][f"df_delta_all_{timeres}"]
    
    combined_stats(combo_stats[year],yrname,pvrad_config["timeres_comparison"])
    
for key in pvsys:
    if plot_flags["stats"]:
        #Plot histograms of deviations
        plot_histogram_deviations(key,pvsys[key],pvrad_config,results_folder)
            
# test_stations = [("PV_12","p_ac_1sec","egrid"),("PV_15","p_ac_1sec","egrid"), #,
#                  ("PV_11","p_ac_1sec","egrid"),("PV_19","p_ac_1sec","egrid")] #

save_combo_stats(combo_stats,station_list,pvrad_config,T_model,results_folder)

if plot_flags["combo_stats"]:
    plot_all_combined_scatter(combo_stats,station_list,pvrad_config,T_model,results_folder,plot_flags["titles"])
#     plot_grid_combined_scatter(pvsys,test_stations,pvrad_config,T_model,results_folder)
        
    
    
            
    
