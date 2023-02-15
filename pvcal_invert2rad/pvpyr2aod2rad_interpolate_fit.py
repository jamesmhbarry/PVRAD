#!/usr/bin/env python3
# -*- aoding: utf-8 -*-
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
from l_retrieval_lib import *
import pickle
import collections
from scipy.interpolate import interp1d
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
                                atm_folder_label,aero_folder_label,sza_label,
                                model,eff_model,T_model)
    
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
        
    #Choose which stations to load    
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
    generate_folder_names_poarad(rt_config,pvcal_config)    
    
    #Check calibration source for filename    
    if len(pvrad_config["calibration_source"]) > 1:
        infos = '_'.join(pvrad_config["calibration_source"])
    else:
        infos = pvrad_config["calibration_source"][0]
    
    filename = filename + infos + '_disortres_' + theta_res + '_' + phi_res + '_'        
    
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
                (pvstat, rt_config, pvcal_config, pvrad_config) = pd.read_pickle(filehandle)            
            
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

def generate_folder_names_pyr2aod(rt_config):
    """
    Generate folder structure to retrieve PYR2aod simulation results
    
    args:    
    :param rt_config: dictionary with RT configuration    
    
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
    disort_config = rt_config["aerosol"]["disort_rad_res"]   
    theta_res = str(disort_config["theta"]).replace('.','-')
    phi_res = str(disort_config["phi"]).replace('.','-')
    
    disort_folder_label = 'Zen_' + theta_res + '_Azi_' + phi_res
    filename = 'aod_lut_results_'
    
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

    folder_label = os.path.join(atm_geom_folder,wvl_folder_label,disort_folder_label,
                                atm_folder_label,aero_folder_label,sza_label)
        
    return folder_label, filename, (theta_res,phi_res)

def generate_folder_names_pv2aod(rt_config):
    """
    Generate folder structure to retrieve PV2aod simulation results
    
    args:    
    :param rt_config: dictionary with RT configuration
    
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
    disort_config = rt_config["aerosol"]["disort_rad_res"]   
    theta_res = str(disort_config["theta"]).replace('.','-')
    phi_res = str(disort_config["phi"]).replace('.','-')
    
    disort_folder_label = 'Zen_' + theta_res + '_Azi_' + phi_res
    filename = 'aod_lut_results_'
    
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

    folder_label = os.path.join(atm_geom_folder,wvl_folder_label,disort_folder_label,
                                atm_folder_label,aero_folder_label,sza_label)
        
    return folder_label, filename, (theta_res,phi_res)

def load_pv_pyr2aod_lut_results(pv_systems,rt_config,pyr_config,pvrad_config,
                                info,station_list,home):
    """
    Load results from aerosol optical depth simulations with DISORT
    
    args:  
    :pv_systems: dictionary of PV systems with data
    :param rt_config: dictionary with current RT configuration
    :param pyr_config: dictionary with current pyranometer calibration configuration        
    :param pvrad_config: dictionary with current inversion configuration    
    :param info: string with description of current campaign
    :param station_list: list of stations
    :param home: string with homepath
    
    out:
    :return pv_systems: dictionary of PV systems with data    
    :return pyr_folder_label: string with label for storing pyranometer results
    :return pv_folder_label: string with label for storing PV results    
    """
    
    mainpath = os.path.join(home,rt_config['save_path']['disort'],
                                   rt_config['save_path']['optical_depth_lut'])     
    
    folder_label, filename, (theta_res,phi_res) = \
    generate_folder_names_pyr2aod(rt_config)

    pyr_folder_label = os.path.join(mainpath,folder_label)    
    
    filename = filename + info + '_'
        
    #Choose which stations to load    
    if type(station_list) != list:
        station_list = [station_list]
        
    #Choose which stations to load    
    if station_list[0] == "all":
        stations = list(pyr_config["pv_stations"].keys())
    else:
        stations = station_list
            
    year = info.split('_')[1]
    
    for station in stations:                
        #Read in binary file that was saved from pvcal_radsim_disort
        filename_stat = filename + station + '.data'        
        try:
            with open(os.path.join(pyr_folder_label,filename_stat), 'rb') as filehandle:  
                # read the data as binary data stream
                pvstat, dummy, dummy = pd.read_pickle(filehandle)                                    

            if "substations" in pvstat:
                del pvstat["substations"]                        
            pvstat[f"df_aod_pyr_{year}"] = pvstat.pop(f"df_aod_{year}")
            pv_systems[station] = merge_two_dicts(pv_systems[station], pvstat)
            
            print('Loading AOD LUT for pyranometers at %s, %s' % (station,info))
        except IOError:
            print('There is no AOD LUT simulation for pyranometers at %s in %s' % (station,info))     
            
    results_path = os.path.join(home,pyr_config["results_path"]["main"],
                                pyr_config["results_path"]["optical_depth"])
    pyr_folder_label = os.path.join(results_path,folder_label)    
        
    folder_label, filename, (theta_res,phi_res) = \
    generate_folder_names_pv2aod(rt_config)
    
    pv_folder_label = os.path.join(mainpath,folder_label)       
    
    filename = filename + info + '_'
    
    if station_list[0] == "all":
        stations = list(pvrad_config["pv_stations"].keys())
    else:
        stations = station_list
        
    for station in stations:                
        #Read in binary file that was saved from pvcal_radsim_disort
        filename_stat = filename + station + '.data'
        
        try:
            with open(os.path.join(pv_folder_label,filename_stat), 'rb') as filehandle:  
                # read the data as binary data stream
                pvstat, dummy, dummy = pd.read_pickle(filehandle)                        
            
            if station not in pv_systems:
                pv_systems.update({station:pvstat})
            else:
                if f"df_aod_{year}" in pvstat:
                    pvstat[f"df_aod_pv_{year}"] = pvstat.pop(f"df_aod_{year}")                                            
                    pv_systems[station] = merge_two_dicts(pv_systems[station],pvstat)                                    
                                                
            print('Loading AOD LUT for PV systems at %s, %s' % (station,info))
        except IOError:
            print('There is no AOD LUT simulation for PV systems at %s in %s' % (station,info))
    
    results_path = os.path.join(home,pv_config["results_path"]["main"],
                                pv_config["results_path"]["optical_depth"])
    pv_folder_label = os.path.join(results_path,folder_label)
            
    return pv_systems, pyr_folder_label, pv_folder_label
    
def interpolate_AOD(df_old,df_hires,timeres_new,substat,radnames):
    """
    Interpolate the AOD lookuptable results onto a higher resolution  

    Parameters
    ----------
    df_old : dataframe with AOD LUT from simulation
    df_hires : dataframe with high resolution measured irradiance data
    timeres_new : time resolution of high resolution dataframe
    substat : string, name of substration
    radnames : string, radiation name "poa" or "down"

    Returns
    -------
    df_output : dataframe with interpolated AOD LUT

    """
    
    days = pd.to_datetime(df_old.index.date).unique().strftime('%Y-%m-%d')    
    
    cols = []
    radtypes = []
    for radname in radnames:
        if "poa" in radname:
            cols.extend(["k_index_poa","cloud_fraction_poa","error_Etotpoa_Wm2",
                         "Etotpoa_pv_inv","error_Etotpoa_pv_inv","theta_IA"])
            radtypes.append("poa")
        elif "down" in radname:
            cols.extend(["k_index_down","cloud_fraction_down","error_Etotdown_Wm2"])
            radtypes.append("down")
    
    cols.extend(radnames)
    
    #Interpolate each day separately
    df_hires_new = []
    for day in days:
        #Low res data
        df = df_old.loc[day]        
        old_index_numeric = df.index.values.astype(float)        
        
        #High res index
        new_index = pd.date_range(start=df.index[0],end=df.index[-1],freq=timeres_new)
        new_index_numeric = new_index.values.astype(float)        
        
        #Get required data from 
        df_hires_day = pd.concat([df_hires.loc[day,pd.IndexSlice[cols,substat]],
                                  df_hires.loc[day,pd.IndexSlice[:,'sun']],
                                  df_hires.loc[day,pd.IndexSlice[['Edirdown_Wm2',
                                  'Ediffdown_Wm2'],'cosmo']]],axis=1)
        
        df_hires_day = df_hires_day.loc[df_hires_day[('sza','sun')].notna()]        
                
        #Interpolate inverted AODs
        for radtype in ["dir","diff"]:
            tablename = f'AOD_{radtype}down_table'
            aod_old = np.stack(df.loc[:,(tablename,"libradtran")].to_numpy())
            
            #interpolate
            f = interp1d(old_index_numeric,aod_old, kind='linear', axis=0)
            aod_new = f(new_index_numeric)
        
            #Add to series
            slice_data = [aod_new[j,:,:] for j in range(aod_new.shape[0])]
            df_hires_day[(tablename,"libradtran")] = pd.Series(index=new_index,data=slice_data)
            df_hires_day = df_hires_day.loc[df_hires_day[(tablename,"libradtran")].notna()]        
            
        #Interpolate diff field
        aod_old = np.stack([table for table in 
                            df.loc[:,("AOD_diff_field_table","libradtran")]])
                
        f = interp1d(old_index_numeric,aod_old, kind='linear', axis=0)
        aod_new = f(new_index_numeric)
        slice_data = [aod_new[j,:,:,:] for j in range(aod_new.shape[0])]
        df_hires_day[("AOD_diff_field_table","libradtran")] = pd.Series(index=new_index,data=slice_data)
        df_hires_day = df_hires_day.loc[df_hires_day[("AOD_diff_field_table","libradtran")].notna()]    
                
        #Interpolate AERONET
        if ("AOD_500","Aeronet") in df.columns:
            #Interpolate AERONET
            aod_old = np.stack(df.loc[:,pd.IndexSlice[:,"Aeronet"]].to_numpy())
            f = interp1d(old_index_numeric,aod_old, kind='linear',axis=0)
            aod_new = f(new_index_numeric)
        
            cols_aeronet = pd.MultiIndex.from_product([df.xs('Aeronet',level='substat',axis=1)
                    .columns.values.tolist(),['Aeronet']],names=['variable','substat'])
            df_hires_day[cols_aeronet] = pd.DataFrame(index=new_index,data=aod_new)        
        
        df_hires_new.append(df_hires_day)
        
    df_output = pd.concat(df_hires_new,axis=0)
    
    return df_output

  
def extract_aod_from_lut(name,pv_station,nameaoddata,namemeasdata,rt_config,substat_pars,
                              angles,const_opt,year):
    """
    Extract the aod from the look-up-table using interpolation

    Parameters
    ----------
    name : string, name of PV station
    pv_station : dictionary with information and data from PV station
    nameaoddata : string, name of AOD dataframe
    namemeasdata : string, name of dataframe with measured data
    rt_config : dictionary with radiative transfer configuration
    substat_pars : dictionary with information on substations
    angles: collections.namedtuple with angles for Idiff integration
    const_opt : collections.namedtuple with optical constants        
    year : string with year under consideration                

    Returns
    -------
    df_merge : dataframe with data merged, and results from AOD LUT method

    """

    df_aod = pv_station[nameaoddata]   
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
        
        #Interpolate AOD lookup table results onto higher resolution
        if pd.to_timedelta(timeres) < pd.to_timedelta(timeres_sim):
            df_aod_substat = interpolate_AOD(df_aod,pv_station[namemeasdata],
                                                  timeres,substat,radnames)
        else:
            df_aod_substat = pd.merge(df_aod.loc[:,pd.IndexSlice[:,["Aeronet","sun","libradtran"]]],
                          pv_station[namemeasdata].loc[:,pd.IndexSlice[["Etotpoa_pv_inv","error_Etotpoa_pv_inv",
                                     "k_index_poa","cloud_fraction_poa"],substat]],
                                   how='left',left_index=True,right_index=True)
                
        radtypes = []
        newcols = []
        for radname in radnames:
            if "poa" in radname:
                radtypes.append("poa")
                newcols.extend(["AOD_500_poa_inv","error_AOD_500_poa_inv"])
            elif "down" in radname:
                radtypes.append("down")
                newcols.extend(["AOD_500_down_inv","error_AOD_500_down_inv"])  
        
        #df_aod_substat = df_aod_substat.iloc[0:5,:]
        index_df = df_aod_substat.index                
        retrieval_df = pd.DataFrame(np.full((len(index_df), len(newcols)), 0.0), index=index_df, 
            columns=pd.MultiIndex.from_product([newcols,[substat]],names=['variable','substat']))   
    
        retrieval_df.sort_index(axis=1,level=1,inplace=True)        
        
        #Here we need to calculate the total irradiance, either POA or downward
        for radtype in radtypes:
            print(f"Extracting AOD from LUT for {substat}, {year} using {radtype} irradiance at {timeres} resolution, this takes a while")
            if radtype == "poa":                
                radname = substat_pars[substat]["name"]
                if "pv" not in radname:
                    errorname = "error_" + '_'.join([radname.split('_')[0],radname.split('_')[-1]])
                else:
                    errorname = "error_" + radname
                
            elif radtype == "down":
                radname = substat_pars[substat]["name"].replace("poa","down")            
                errorname = "error_" + '_'.join([radname.split('_')[0],radname.split('_')[-1]])                        
            
            if "opt_pars" in substat_pars[substat]:
                irrad_pars = [substat_pars[substat]["opt_pars"][i][1] for i in range(len(substat_pars[substat]["opt_pars"]))]        
            else:
                irrad_pars = [substat_pars[substat]["ap_pars"][i][1] for i in range(len(substat_pars[substat]["ap_pars"]))]        
            
            AOD_table = [] #pd.Series(dtype=float)]*len(df_aod_substat)
            
            #notnan_index = df_aod_substat.loc[df_aod_substat[(radname,substat)].notna()].index
            
            for itime in df_aod_substat.index: # notnan_index: #enumerate(dates):                
                sun_pos = sun_position(np.deg2rad(df_aod_substat.loc[itime,('sza','sun')]),
                               np.deg2rad(df_aod_substat.loc[itime,('phi0','sun')]))   
                
                #Calculate POA irradiance for LUT, integrate diffuse rad, rotate direct irrad
                OD_array = df_aod_substat.loc[itime,("AOD_dirdown_table","libradtran")][:,0]
                Edirdown = df_aod_substat.loc[itime,("AOD_dirdown_table","libradtran")][:,1]                
                Ediffdown = df_aod_substat.loc[itime,("AOD_diffdown_table","libradtran")][:,1]                    
                    
                if radtype == "poa":
                    radname = substat_pars[substat]["name"]
                   
                    Idiff_field = df_aod_substat.loc[itime,("AOD_diff_field_table","libradtran")]                     
                    if "pv" not in radname:
                        irrad = E_poa_calc(irrad_pars,Edirdown,Idiff_field,sun_pos,angles,
                           const_opt,None,False)            
        
                        F_array = irrad["Etotpoa"]                                        
                    else: #turn on optical model
                        irrad = E_poa_calc(irrad_pars,Edirdown,Idiff_field,sun_pos,angles,
                           const_opt,None,True)            
        
                        F_array = irrad["Etotpoa_pv"]                                        
                        
                elif radtype == "down":
                    #Calculate downward irradiance                                        
                    F_array = Edirdown + Ediffdown                                    
                
                #assign to dataframe for plot
                AOD_table.append(np.transpose(np.array([OD_array,F_array])))
                    
                #if settings['print_flag'] == True: print(f"OD: {OD_array}, F: {F_array}")
                aod_range = OD_array
                # read out the measured OD
                y_meas = df_aod_substat.loc[itime.strftime("%Y-%m-%d %H:%M:%S"),(radname,substat)]
                # use function to retrieve the aod that corresponds to a certain irradiance.
                
                if not np.isnan(y_meas):
                    # prepare the fit
                    X, Y = OD_array, F_array #df["aod"].to_numpy(), df[rad_key].to_numpy()
                    dY = 0.02*Y
                    #dy_meas = 0.05 * y_meas
                    dy_meas = df_aod_substat.loc[itime.strftime("%Y-%m-%d %H:%M:%S"),(errorname, substat)]
                    #print(f"the measured {radtype} irradiance values for timestep {time} were {y_meas} +/- {dy_meas}")
                    #starttime = time.time()
                    aod, d_aodup, d_aoddown, erroraode = retrieve_by_fit_or_interpolation(X, Y, dY, y_meas,
                                                        dy_meas, threshold=1.)
                    d_aod = np.mean(np.array([d_aodup, d_aoddown], dtype=float))
                    #endtime = time.time()  
                else:
                    aod = np.nan
                    d_aod = np.nan
                
                #Now use retrieved AOD to find irradiance components!
                if aod and d_aod and not np.isnan(aod) and not np.isnan(d_aod):
                    #direct component                            
                    g_interp = interp1d(OD_array,Edirdown,kind='linear',bounds_error=False,
                                        fill_value=np.nan)
                    Edirdown = g_interp(aod)
                    if Edirdown < 1e-6:
                        Edirdown = 0.
                    
                    #diffuse component                                                        
                    h_interp = interp1d(OD_array,Ediffdown,kind='linear',bounds_error=False,
                                        fill_value=np.nan)
                    Ediffdown = h_interp(aod)
                                                                    
                else:
                    Edirdown = np.nan                    
                    Ediffdown = np.nan                    
                
                # write into retrieval_df
                retrieval_df.at[itime, (f"AOD_500_{radtype}_inv",substat)] = aod
                retrieval_df.at[itime,(f"error_AOD_500_{radtype}_inv",substat)] = d_aod     
                
                retrieval_df.at[itime, (f"Edirdown_{radtype}_inv",substat)] = Edirdown
                retrieval_df.at[itime, (f"Ediffdown_{radtype}_inv",substat)] = Ediffdown
            
            # retrieval_df.loc[~notnan_index,(f"AOD_500_{radtype}_inv",substat)] = np.nan
            # retrieval_df.loc[~notnan_index,(f"error_AOD_500_{radtype}_inv",substat)] = np.nan
            #Assigne to dataframe for plot
            retrieval_df[(f"AOD_{radtype}_table",substat)] = pd.Series(AOD_table,
                                                            index=retrieval_df.index)
            
            # if flags["lut"]:
            #     print("Generating LUT plots for %s, %s for %s irradiance" % (name,substat,radtype))
            #     plot_aod_lut(name,substat,radtype,radname,errorname,df_aod_substat,retrieval_df,
            #                   aod_range,path,flags["titles"])
            
        #Merge dataframes from one radtype
        if df_merge.empty:
            df_merge = pd.concat([df_aod_substat.drop(
                labels=['libradtran'],axis=1,level=1),retrieval_df],axis=1)
        else:
            df_merge = pd.concat([retrieval_df,df_merge,df_aod_substat.drop(
                labels=['libradtran','sun','Aeronet'],axis=1,level=1)],axis=1) 
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
    name_df : name of dataframe with relevant data to save to file
    info : string describing current campaign    
    path : string, path to save file to

    Returns
    -------
    None.

    """
    
    pv_station_save = deepcopy(pv_station)
    
    filename_stat = f"aod_fit_results_{info}_{name}_{timeres}.data"        
    
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
#This program takes the DISORT simulation for aod and uses an interpolation routine
#developed by Grabenstein to find the aod, plus creates both time series and scatter plots
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

#Load pv2rad configuration
pvrad_config = load_yaml_configfile(config["pvrad_configfile"])

#Load radiative transfer configuration
rt_config = load_yaml_configfile(config["rt_configfile"])

homepath = os.path.expanduser('~') # #"/media/luke" #

if args.campaign:
    campaigns = args.campaign
    if type(campaigns) != list:
        campaigns = [campaigns]
else:
    campaigns = config["description"]     
#%%Preparation
sun_position = collections.namedtuple('sun_position','sza azimuth')
disort_res = rt_config["aerosol"]["disort_rad_res"]
grid_dict = define_disort_grid(disort_res)

angle_grid = collections.namedtuple('angle_grid', 'theta phi umu')
angle_arrays = angle_grid(grid_dict["theta"],np.deg2rad(grid_dict["phi"]),grid_dict["umu"])
#%%Load inversion results
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
    
    #Load inversion results
    print(f'Loading PYRCAL and PV2POARAD results for {campaign}')
    pvsys = load_pyr2cf_pv2poarad_results(rt_config,pyr_config,pvcal_config,pvrad_config,campaign,stations,homepath)
    
    #Load inversion results
    print(f'Loading PV2AOD and PYR2AOD LUT from DISORT simulation for {campaign}')
    pvsys, pyr_results_folder, pv_results_folder = \
    load_pv_pyr2aod_lut_results(pvsys,rt_config, pyr_config, pvrad_config, 
                                campaign, stations, homepath)

# if pyr_config["pmax_doas_station"]:
#     pvsys = load_pmaxdoas_results(pyr_config["pmax_doas_station"], timeres_sim, homepath)

    #Perform analysis for each system
    for key in pvsys:  
        print(f'\nExtracting AOD for {key}, {year}')
        tres_list = []
        
        #Pyranometers
        name_df_aodlut = "df_aod_pyr_" + year.split('_')[-1] 
        
        if key in pyr_config["pv_stations"]:
            for substat in pyr_config["pv_stations"][key]['substat']:    
                pyr_config["pv_stations"][key]['substat'][substat] = \
                merge_two_dicts(pyr_config["pv_stations"][key]['substat'][substat],
                            pvsys[key]["substations_pyr"][substat])
                                    
            for val in pvsys[key]:
                if "df_pyr_" in val:
                    name_df_hires = val
            
            pyr_substat_dict = pyr_config["pv_stations"][key]["substat"]
            
            name_df_aod_result = f"df_aod_pyr_{year.split('_')[-1]}_1min"
            tres_list.append('1min')
                    
            #Use LUT to extract aod using Johannes method and plot time series
            pvsys[key][name_df_aod_result] = extract_aod_from_lut(key,pvsys[key],name_df_aodlut,
                                    name_df_hires,rt_config,pyr_substat_dict,angle_arrays,const_opt,
                                    year.split('_')[-1])  

        else:
            pyr_substat_dict = {}                       

        name_df_aodlut = "df_aod_pv_" + year.split('_')[-1]
        
        #PV systems with different time resolutions
        if key in pvrad_config["pv_stations"]:
            #There are different types associated with different time resolutions       
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
                            
                name_df_aod_result = f"df_aod_pv_{year.split('_')[-1]}_{timeres}"
                for val in pvsys[key]:
                    if "df_pv_" in val and timeres in val:
                        name_df_hires = val
                
                if name_df_aodlut in pvsys[key] and year in pvrad_config["pv_stations"][key][substat_type]["source"]:                
                    #Use LUT to extract aod and plot time series
                    pvsys[key][name_df_aod_result] = extract_aod_from_lut(key,pvsys[key],name_df_aodlut,
                                    name_df_hires,rt_config,pv_substat_dict,angle_arrays,const_opt,
                                    year.split('_')[-1],pv_results_folder)    
                                
                else:
                    print(f"No {timeres} data for {key}, {substat_type} in {year.split('_')[-1]}")
                    if not pyr_substat_dict:
                        tres_list.remove(timeres)
                    elif timeres == "15min":
                        tres_list.remove(timeres)
        else:
            pv_substat_dict = {}                        
                    
        #Save results depending on time resolution
        for timeres in tres_list:            
            #Plot time series of aod for each day, compare with weather & satellite data                    
            name_df_combine = "df_aodfit_pyr_pv_" + year.split('_')[-1] + '_' + timeres
        
            #Merge data into one dataframe
            if f"df_aod_pyr_{year.split('_')[-1]}_{timeres}" in pvsys[key]\
            and f"df_aod_pv_{year.split('_')[-1]}_{timeres}" in pvsys[key]:
                df_aod_merge = pd.merge(pvsys[key][f"df_aod_pyr_{year.split('_')[-1]}_{timeres}"].loc[:,
                           pd.IndexSlice[:,[*pyr_substat_dict.keys()]]],
                           pvsys[key][f"df_aod_pv_{year.split('_')[-1]}_{timeres}"],
                           left_index=True,right_index=True)                                                                    
                
            elif f"df_aod_pyr_{year.split('_')[-1]}_{timeres}" in pvsys[key]:
                df_aod_merge = pvsys[key][f"df_aod_pyr_{year.split('_')[-1]}_{timeres}"]                                                                        
                
            elif f"df_aod_pv_{year.split('_')[-1]}_{timeres}" in pvsys[key]:
                df_aod_merge = pvsys[key][f"df_aod_pv_{year.split('_')[-1]}_{timeres}"]                                                                            
            
            pvsys[key][name_df_combine] = df_aod_merge             
            pvsys[key][name_df_combine].sort_index(axis=1,level=1,inplace=True)                                        
                
            
            results_folder = generate_folders(rt_config,pvcal_config,pvrad_config,homepath)
            #Save results
            save_results(key,pvsys[key],name_df_combine,campaign,
                      timeres,results_folder)
      
                

    
    
