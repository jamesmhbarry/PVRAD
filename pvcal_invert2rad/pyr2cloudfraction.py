#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 10:44:33 2020

@author: james
"""
import os

import numpy as np
import pandas as pd
import pickle
import collections
from file_handling_functions import *
from rt_functions import *
from pvcal_forward_model import E_poa_calc, azi_shift, cos_incident_angle

import subprocess
import ephem

###############################################################
###   general functions to load and process data    ###
###############################################################

def generate_folder_names_disort(mainpath,config):
    """
    Generate folder structure to retrieve DISORT simulation results
    
    args:
    :param mainpath: string with parent folder
    :param config: dictionary with RT configuration
    
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
    wvl_config = config["common_base"]["wavelength"]["pyranometer"]
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

def generate_folder_names_pyrcal(rt_config,pyrcal_config):
    """
    Generate folder structure to retrieve PYRCAL simulation results
    
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
    filename = 'calibration_results_'
    
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
            
    sza_label = "SZA_" + str(int(pyrcal_config["sza_max"]["calibration"]))

    folder_label = os.path.join(atm_geom_folder,disort_folder_label,
                                atm_folder_label,aero_folder_label,sza_label)
        
    return folder_label, filename, (theta_res,phi_res)


def load_resampled_data(station,timeres,measurement,config,home):
    """
    Load data that has already been resampled to a specified time resolution
    
    args:    
    :param station: string, name of PV station to load
    :param timeres: string, timeresolution of the data
    :param measurement: string, description of current measurement campaign
    :param config: dictionary with paths for loading data
    :param home: string, homepath    
    
    out:
    :return pv_stat: dictionary of dataframes and other information on PV station    
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

def load_radsim_calibration_results(info,rt_config,pyrcal_config,
                                    station_list,home):
    """
    Load results from DISORT radiation simulation as well as calibration results
    
    args:
    :param info: string with description of results
    :param rt_config: dictionary with current RT configuration
    :param pyrcal_config: dictionary with PVCAL configuration    
    :param station_list: list of stations for which simulation was run
    :param home: string with homepath
    
    out:
    :return pv_systems: dictionary of PV systems with data
    """
            
    mainpath_pyrcal = os.path.join(home,pyrcal_config['results_path']['main'],
                                   pyrcal_config['results_path']['calibration'])
    
    pv_systems = {}
    if type(station_list) != list:
        station_list = [station_list]
        if station_list[0] == "all":
            station_list = pyrcal_config["pv_stations"]
        
    #calibration_source = pvrad_config["calibration_source"]    
    
    #get description/s
    # if len(calibration_source) > 1:
    #     infos = '_'.join(calibration_source)
    # else:
    #    infos = calibration_source

    folder_label_pvcal, filename_pvcal, (theta_res,phi_res) = \
    generate_folder_names_pyrcal(rt_config,pyrcal_config)

    for station in station_list:                
        #Load calibration data for inversion onto irradiance
        filename_pvcal_stat = filename_pvcal + info + '_disortres_' +\
                      theta_res + '_' + phi_res + '_' + station + '.data'                      
        
        with open(os.path.join(mainpath_pyrcal,folder_label_pvcal,
                               filename_pvcal_stat), 'rb') as filehandle:  
            # read the data as binary data stream
            (pvstat, rtcon, pvcalcon) = pd.read_pickle(filehandle)            

#        for measurement in calibration_source:
#            year = "mk_" + measurement.split('_')[1]            
#            new_df_sim = 'df_sim_' + year.split('_')[-1]  
#            #Get only columns from libradtran, sun position, aerosol, albedo
#            pvstat[new_df_sim] = pd.concat([pvstat[new_df_sim].loc[:,pd.IndexSlice[:,['sun','Aeronet']]],
#            pvstat[new_df_sim].loc[:,pd.IndexSlice['albedo',:]],
#            pvstat[new_df_sim].loc[:,pd.IndexSlice[:,'libradtran']]],axis=1)
        
        print('Calibration and simulation data for %s loaded from %s' % (station,filename_pvcal_stat))
        pv_systems.update({station:pvstat})
    
    return pv_systems

###############################################################
###   Preparation for inversion                             ###
###############################################################
def find_nearest_cosmo_grid_folder(configfile,pv_systems,datatype,home):
    """
    Search through the output of cosmomystic or cosmopvcal to find the 
    gridpoint (folder) corresponding to the location of each PV station
    
    args:
    :param configfile: string, configfile for cosmomystic
    :param pv_systems: dictionary of pv_systems
    :param datatype: string, either surf or atmo
    :param home: string, home path
    
    out:
    :return pv_systems: dictionary updated with path to cosmo atmfiles
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
                    elif datatype == "irrad":
                        pv_systems[key]['path_cosmo_irrad'] = os.path.join(path,folder)
    
    return pv_systems
 
def import_cosmo_surf_data(pv_systems,days):
    """
    Import surface data from cosmo2pvcal
    
    args:
    :param pv_systems: dictionary of pv_systems
    :param days: list of days to consider
    
    out:
    :return pv_systems: dictionary of pv_systems including surface data
    """
    for key in pv_systems:                
        #Extract data from COSMO files
        dataframe = pd.DataFrame()
        dfs = [pd.read_csv(os.path.join(pv_systems[key]['path_cosmo_surface'],\
                iday.replace('-','') + '_surface_props.dat'),sep='\s+',index_col=0,skiprows=2,
                header=None,names=['v_wind_10M','dir_wind_10M','T_ambient_2M_C']) for iday in days]
         
        dataframe = pd.concat(dfs,axis=0)
        dataframe.index = pd.to_datetime(dataframe.index,format='%d.%m.%Y;%H:%M:%S')
            
        dataframe.T_ambient_2M_C = dataframe.T_ambient_2M_C - 273.15     
        
        #Create Multi-Index for cosmo data
        dataframe.columns = pd.MultiIndex.from_product([dataframe.columns.values.tolist(),['cosmo']],
                                                                       names=['substat','variable'])       
                       
        #Assign to special cosmo dataframe, and join with main dataframe
        pv_systems[key]['df_cosmo'] = dataframe
        pv_systems[key]['df_sim'] = pd.concat([pv_systems[key]['df_sim'],dataframe],axis=1,join='inner')
       
    return pv_systems

def prepare_surface_data(info,inv_config,pv_systems,days,home):
    """
    args:
    :param info: string, description of simulation
    :param inv_config: dictionary with configuration
    :param pv_systems: dictionary of pv systems
    :param days: list of days to consider
    :param home: string, home path
    
    out:
    :return pv_systems: dictionary of pv systems updated with surface data
    """
    
    configfile = os.path.join(home,inv_config["cosmopvcal_configfile"])
    cosmo_config = load_yaml_configfile(configfile)
    finp = open(configfile,'a')
    if "stations_lat_lon" not in cosmo_config:
        finp.write('# latitude, longitude of PV stations within the COSMO grid)\n')
        finp.write('stations_lat_lon:\n')
    
        #Write lat lon into config file    
        for key in pv_systems:        
            finp.write('    %s: [%.5f,%.5f]\n' % (key,pv_systems[key]['lat_lon'][0],pv_systems[key]['lat_lon'][1]))
    else:
        for key in pv_systems:
            if key not in cosmo_config["stations_lat_lon"]:
                finp.write('    %s: [%.5f,%.5f]\n' % (key,pv_systems[key]['lat_lon'][0],
                           pv_systems[key]['lat_lon'][1]))
    
    finp.close()
    
    year = 'mk_' + info.split('_')[1]
    #Which days to consider
    
    test_days = create_daterange(days[year]["all"])
    
    #Prepare surface data from COSMO
    if inv_config["cosmo_sim"]:
        # call cosmo2pvcal 
        print('Running cosmo2pvcal to extract surface properties')
        child = subprocess.Popen('cosmo2pvcal ' + configfile, shell=True)
        child.wait()
    else:
        print('cosmo2pvcal already run, read in surface files')        

    pv_systems = find_nearest_cosmo_grid_folder(configfile,pv_systems,'surf',home)   
    pv_systems = find_nearest_cosmo_grid_folder(configfile,pv_systems,'irrad',home)   
    pv_systems = import_cosmo_surf_data(pv_systems,test_days)
    
    return pv_systems

def sun_position(pvstation,df,sza_limit):
    """
    Using PyEphem to calculate the sun position
    
    args:    
    :param pvstation: dictionary of one PV system with data
    :param df: string, name of dataframe
    :param sza_limit: float defining maximum solar zenith angle for simulation
    
    out:
    :return: dataframe including sun position
    
    """        
    dataframe = pvstation[df]
    len_time = len(dataframe)
    index_time = dataframe.index

    # initalize observer object
    observer = ephem.Observer()
    observer.lat = np.deg2rad(pvstation['lat_lon'][0])
    observer.lon = np.deg2rad(pvstation['lat_lon'][1])

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
    dataframe = dataframe.loc[dataframe[('sza','sun')] <= sza_limit]        
        
    return dataframe

def cosine_bias_correction(dataframe,stat_config,coeff_poly):
    """
    Correct the small pyranometers from TROPOS for cosine bias using
    polynomial correction factor calculated by Jonas Witthuhn, but only
    apply it when conditions are clear (cloud fraction = 0)

    Parameters
    ----------
    dataframe : pandas dataframe with all data including sun position in degrees
    stat_config : dictionary with stations configuration including angles in radians
    coeff_poly : dictionary with coefficients for correction

    Returns
    -------
    dataframe : new dataframe with corrected irradiance values

    """
        
    #GTI bias correction
    radname = stat_config["name"]
    
    if ("cloud_fraction_poa",substat) in dataframe.columns:
        #Find clear sky periods
        cf_poa_mask = dataframe[("cloud_fraction_poa",substat)] == 0
        
        if "opt_pars" not in stat_config:
            mu_IA = cos_incident_angle(np.deg2rad(dataframe.loc[cf_poa_mask,("sza","sun")].values), 
                azi_shift(np.deg2rad(dataframe.loc[cf_poa_mask,("phi0","sun")].values)),
                  stat_config["ap_pars"][0][1],azi_shift(stat_config["ap_pars"][1][1]))
        else:
            mu_IA = cos_incident_angle(np.deg2rad(dataframe.loc[cf_poa_mask,("sza","sun")].values), 
                azi_shift(np.deg2rad(dataframe.loc[cf_poa_mask,("phi0","sun")].values)),
                  stat_config["opt_pars"][0][1],azi_shift(stat_config["opt_pars"][1][1]))
        
        C_GTI = coeff_poly["c_0"]*mu_IA**3 + coeff_poly["c_1"]*mu_IA**2\
            + coeff_poly["c_2"]*mu_IA + coeff_poly["c_3"]
        
        dataframe.loc[cf_poa_mask,(radname,substat)] = \
        dataframe.loc[cf_poa_mask,(radname,substat)]*C_GTI
    
    #GHI bias correction
    radname = stat_config["name"].replace("poa","down")
    
    if ("cloud_fraction_down",substat) in dataframe.columns:
        cf_down_mask = dataframe[("cloud_fraction_down",substat)] == 0
        
        mu0 = np.cos(np.deg2rad(dataframe.loc[cf_down_mask,("sza","sun")].values))
        
        C_GHI = coeff_poly["c_0"]*mu0**3 + coeff_poly["c_1"]*mu0**2\
            + coeff_poly["c_2"]*mu0 + coeff_poly["c_3"]
        
        dataframe.loc[cf_down_mask,(radname,substat)] = \
        dataframe.loc[cf_down_mask,(radname,substat)]*C_GHI                               
                               
    return dataframe   

def cloud_fraction(station,pv_station,dfname,substat_inv,pyr_config,const_opt,angles,deltas):
    """
    Calculate cloud fraction by first calculating clear sky reference and then
    taking the ratio
    
    args:    
    :param station: string, name of station
    :param pv_station: dictionary with dataframe of PV data etc
    :param dfname: string, name of dataframe with relevant data
    :param substat_inv: string, substation to invert
    :param pyr_config: dictionary with configuration for calibration    
    :param const_opt: named tuple with optical constants
    :param angles: named tuple with angle grid for DISORT
    :param deltas: named tuple with deltas for numerical differentiation (not used here)
    
    out:
    :return pv_station: dictionary for PV station with cloud fraction calculation in dataframe
    
    """
    
    print('Calculating POA irradiance, clear sky reference for %s' % substat_inv)
                
    dataframe = pv_station['df_sim']

    #Extract irradiance parameters
    substat_pars = pv_station["substations"][substat_inv]
    
    radname = substat_pars["name"]                    
    if "pyr" in radname:
        radnames = [radname,radname.replace('poa','down')]
    else:
        radnames = [radname]
    
    #Extract irradiance parameters (here without optical model)    
    if "opt_pars" in substat_pars:    
        print("Using angles from calibration")
        irrad_pars = [substat_pars["opt_pars"][0][1],
                  substat_pars["opt_pars"][1][1]]                  
    else:
        print("Using angles from a-priori information")
        irrad_pars = [substat_pars["ap_pars"][0][1],
                  substat_pars["ap_pars"][1][1]]
    
    if irrad_pars[0] == 0 and irrad_pars[1] == 0:
        #This is the horizontal case
        dataframe[('Etotdown_clear_Wm2',substat_inv)] = \
        dataframe['Edirdown','libradtran'] + dataframe['Ediffdown','libradtran']
        
    else:
        #Calculate clear sky reference
        Edirdown = dataframe.Edirdown.values.flatten()
        sun_position = collections.namedtuple('sun_position','sza azimuth')
        sun_pos = sun_position(np.deg2rad(dataframe.sza.values.flatten()),
                           np.deg2rad(dataframe.phi0.values.flatten()))                
        
        #Get the diffuse radiance field
        diff_field = np.zeros((len(dataframe),len(angles.theta),len(angles.phi)))                
        for i, itime in enumerate(dataframe.index):
            diff_field[i,:,:] = dataframe.loc[itime,('Idiff','libradtran')].values #.flatten()
            
        dataframe[('Etotpoa_clear_Wm2',substat_inv)] = E_poa_calc(irrad_pars,Edirdown,diff_field,
                  sun_pos,angles,const_opt,deltas,False)['Etotpoa']
                
    #dataframe.sort_index(axis=1,level=1,inplace=True)
    
    cs_threshold = pyr_config["cloud_fraction"]["cs_threshold"]
    
    #Calculate clearness index
    print('Calculating clearness index and cloud fraction for %s' % substat_inv)
    print(f'Clear sky threshold is set at {cs_threshold}')
    
    # radname = substat_pars["name"]
    # statname = "ISE"        
    
    #interpolate clear sky irradiance (POA)
    pv_station[dfname][('Etotpoa_clear_Wm2',substat_inv)] = pv_station['df_sim'][('Etotpoa_clear_Wm2',\
                                   substat_inv)].reindex(pv_station[dfname].index).interpolate('cubic')
        
    #interpolate clear sky irradiance (downward)
    pv_station[dfname][('Etotdown_clear_Wm2',substat_inv)] = pv_station['df_sim'][('Etotdown',\
                                   'libradtran')].reindex(pv_station[dfname].index).interpolate('cubic')
        
    for radname in radnames: #pv_station[dfname].xs(substat_inv,level="substat",axis=1).columns:
        if "poa" in radname:
            radtype = "poa"
        elif "down" in radname:
            radtype = "down"
            
        if (pv_station[dfname][(radname,substat_inv)] == 0).all():
            pv_station[dfname].drop(columns=[(radname,substat_inv)],inplace=True)
        else:
            #Calculate error vector for irradiance
            pv_station[dfname][(f'error_Etot{radtype}_Wm2',substat_inv)] = pv_station[dfname]\
            [(radname,substat_inv)]*substat_pars["e_err_rel"]
            pv_station[dfname][(f'error_Etot{radtype}_Wm2',substat_inv)].where(pv_station[dfname]\
            [(radname,substat_inv)] > substat_pars["e_err_min"],substat_pars["e_err_min"]\
            ,inplace=True)
            
            #Calculate clearness index
            pv_station[dfname][(f'k_index_{radtype}',substat_inv)] = pv_station[dfname][(radname,\
                                    substat_inv)]/pv_station[dfname][(f'Etot{radtype}_clear_Wm2',\
                                           substat_inv)]
            #Calculate cloud fraction
            pv_station[dfname][(f'cloud_fraction_{radtype}',substat)] = 1. -\
            (cs_threshold[0] < pv_station[dfname][(f'k_index_{radtype}',substat)]).astype(float) # & 
            
            pv_station[dfname].loc[pv_station[dfname][(f'k_index_{radtype}',substat)].isna(),\
                                   (f'cloud_fraction_{radtype}',substat)] = np.nan
                
            pv_station[dfname].loc[pv_station[dfname][(f'k_index_{radtype}',substat)] > cs_threshold[1],\
                                   (f'cloud_fraction_{radtype}',substat)] = -999                        
    
    pv_station[dfname].sort_index(axis=1,level=1,inplace=True)
    
    return pv_station     

def generate_results_folders(rt_config,pyr_config,path,home):
    """
    Generate folders for results
    
    args:
    :param rt_config: dictionary with DISORT config
    :param pyr_config: dictionary with pyranometer configuration
    :param path: main path for saving files or plots    
    :param home: string, home directory
    
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
    
    disort_config = rt_config["disort_rad_res"]
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
        
    return fullpath

def save_results(info,key,pv_station,dfname,pyr_config,rt_config,path,home):
    """
    Save inversion results to a binary file
    
    args:
    :param info: string, description of simulation
    :param key: string, name of PV station
    :param pv_station: dictionary with data and info of PV system    
    :param dfname: string, name of dataframe with relevant data
    :param pyr_config: dictionary with configuration for calibration
    :param rt_config: dictionary with configuration of RT simulation
    :param path: path for saving results
    :param home: string, home path
    
    """                
    
    #get description/s
#    if len(pyr_config["calibration_source"]) > 1:
#        infos = '_'.join(pyr_config["calibration_source"])
#    else:
#        infos = pyr_config["calibration_source"][0]
        
    atm_source = rt_config["atmosphere"]
    asl_source = rt_config["aerosol"]["source"]
    asl_res = rt_config["aerosol"]["data_res"]
    res = rt_config["disort_rad_res"]    
    
    filename = 'cloud_fraction_results_'
    if atm_source == 'cosmo':
        filename = filename + 'atm_'
    if asl_source != 'default':
        filename = filename + 'asl_' + asl_res + '_'
    
    filename = filename + info + '_disortres_' + str(res["theta"]).replace('.','-')\
                    + '_' + str(res["phi"]).replace('.','-') + '_'
    
    filename_stat = filename + key + '.data'
    
    #Remove raw data
#    del pv_station["raw_data"]
    
    with open(os.path.join(path,filename_stat), 'wb') as filehandle:  
        # store the data as binary data stream
        pickle.dump((pv_station, rt_config, pyr_config), filehandle)

    print('Results written to file %s\n' % filename_stat)

    if pyr_config["results_path"]["csv_files"] not in list_dirs(path):
        os.mkdir(os.path.join(path,pyr_config["results_path"]["csv_files"]))        

    path = os.path.join(path,pyr_config["results_path"]["csv_files"])
    
    #Write to CSV file                
    dataframe = pv_station[dfname]
        
    for substat in pv_station["substations"]:
        #Select values we want    
        for radname in dataframe.xs(substat,level="substat",axis=1).columns:
            if "Etot" in radname and "clear" not in radname and "error" not in radname:
                if "poa" in radname:
                    suffix = "poa"
                    colname = "POA_Irradiance_Wm2"
                elif "down" in radname:
                    suffix = "down"
                    colname = "Downward_Irradiance_Wm2"
                #Save all values
                filename_csv = 'cloud_fraction_' + suffix + '_' + substat + '_' + info + '.dat'
                        
                f = open(os.path.join(path,filename_csv), 'w')
                f.write('#Irradiance data and cloud fraction based on DISORT simulation from %s\n' % key)    
                f.write('#Irradiance (%s) from %s\n' % (suffix,substat))
                if "poa" in radname:
                    ap_pars = pv_station["substations"][substat]["ap_pars"]
                    f.write('#A-priori angles (value,error):\n')
                    for par in ap_pars:
                        f.write('#%s: %g (%g)\n' % par)                        
                    if "opt_pars" in pv_station["substations"][substat]:
                        opt_pars = pv_station["substations"][substat]["opt_pars"]
                        f.write('#Optimisation angles (value,error):\n')
                        for par in opt_pars:
                            f.write('#%s: %g (%g)\n' % par)     
                    else:
                        f.write('No solution found by the optimisation routine, using a-priori values\n')
                                
                dfall =  pd.concat([dataframe.loc[:,(radname,substat)],
                                           dataframe.loc[:,('cloud_fraction_' + suffix,substat)]],
                                          axis='columns')               
                dfall.to_csv(f,sep=';',float_format='%.6f',header=[colname,"Cloud_Fraction"],
                                  index_label="Timestamp_UTC",na_rep='nan')
                f.close()    
                print('Results written to file %s\n' % filename_csv) 
                
                #Save values for each Falltag
                for day in pyr_config["falltage"]:
                    dfday = pd.concat([dataframe.loc[day.strftime('%Y-%m-%d'),(radname,substat)],
                                           dataframe.loc[day.strftime('%Y-%m-%d'),('cloud_fraction_'+suffix,substat)]],
                                          axis='columns')
                                
                    filename_csv = 'cloud_fraction_' + suffix + '_' + substat + '_' + day.strftime('%Y-%m-%d') + '.dat'
                        
                    f = open(os.path.join(path,filename_csv), 'w')
                    f.write('#Irradiance data and cloud fraction based on DISORT simulation from %s\n' % key)    
                    f.write('#Irradiance from %s' % substat)
                    f.write('\n')                   
                    dfday.to_csv(f,sep=';',float_format='%.6f',header=[colname,"Cloud_Fraction"],
                                      index_label="Timestamp_UTC",na_rep='nan')
                    f.close()    
                    print('Results written to file %s' % filename_csv) 

#%%Main Program
#######################################################################
###                         MAIN PROGRAM                            ###
#######################################################################
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
    config_filename = "config_PYRCAL_2019_messkampagne.yaml"
 
config = load_yaml_configfile(config_filename)

#Load data configuration
data_config = load_yaml_configfile(config["data_configfile"])

#Load pyranometer
pyr_config = load_yaml_configfile(config["pyr_configfile"])

#Load radiative transfer configuration
rt_config = load_yaml_configfile(config["rt_configfile"])

homepath = os.path.expanduser('~') #"/media/luke/" #

if args.station:
    stations = args.station
    if stations[0] == 'all':
        stations = 'all'
else:
    #Stations for which to perform inversion
    stations = "PV_12" #pyr_config["stations"]

print('Cloud fraction calculation from pyranometer for %s' % stations)
print('Loading results from radiative transfer simulation and calibration')
pvsys = load_radsim_calibration_results(config["description"],rt_config,pyr_config,
                                 stations,homepath)

inv_params = pyr_config["calibration"]

albedo = rt_config["albedo"]

disort_res = rt_config["disort_rad_res"]
grid_dict = define_disort_grid(disort_res)

opt_dict = pyr_config["optics"]
optics_flag = opt_dict["flag"]
optics = collections.namedtuple('optics', 'kappa L')
const_opt = optics(opt_dict["kappa"],opt_dict["L"])

angle_grid = collections.namedtuple('angle_grid', 'theta phi umu')
angle_arrays = angle_grid(grid_dict["theta"],np.deg2rad(grid_dict["phi"]),grid_dict["umu"])

#These are the increments to use for numerical differentiation
diff_theta = np.deg2rad(disort_res["theta"]/2)
diff_phi = np.deg2rad(disort_res["phi"]/2)
diff_n = inv_params["n_diff"]

diffs = collections.namedtuple('diffs', 'theta phi n')

diffs = diffs(diff_theta,diff_phi,diff_n)

info = config["description"]
sza_limit = pyr_config["sza_max"]["inversion"]

for key in pvsys:        
    for substat in pvsys[key]["substations"]:                
        timeres = pvsys[key]["substations"][substat]["t_res_inv"]
        print(f"\nCloud fraction calculation for {key}, {substat} at {timeres} resolution")
        #Load data, rename dataframe for the year
        dfname = "df_" + info.split('_')[1]  + '_' + timeres
        if dfname not in pvsys[key]:
            pvstat_rs = load_resampled_data(key,timeres,info,data_config,homepath)
            pvstat_rs[dfname] = pvstat_rs["df"]
            del pvstat_rs["df"]
            pvsys[key] = merge_two_dicts(pvsys[key],pvstat_rs)
                
            #Throw away night time values
            print("Calculate sun position and throw away night time values")
            pvsys[key][dfname] = sun_position(pvsys[key],dfname,sza_limit)                    
                
            #Throw away times before and after which we have simulation
            pvsys[key][dfname] = pvsys[key][dfname].loc[(pvsys[key][dfname].index 
                       >= pvsys[key]["df_sim"].index[0]) & (pvsys[key][dfname].index <= 
                       pvsys[key]["df_sim"].index[-1])]  
    
        # #The data has values at 05,10,15, need to shift by 10 seconds
        # newindex = pvsys[key][dfname].index - pd.Timedelta('5s')
        
        # #Resample to 5s and then reindex back to 10s on new index
        # pvsys[key][name_raw_df] = pvsys[key][name_raw_df].resample('5s').\
        #     interpolate('linear').reindex(newindex)
        
        #Calculate cloud fraction
        pvsys[key] = cloud_fraction(key,pvsys[key],dfname,substat,pyr_config,
         const_opt,angle_arrays,diffs)
        
        if "Pyr" in substat and pyr_config["bias_correction"]["flag"]:
            print("Correcting cosine bias")
            pvsys[key][dfname] = cosine_bias_correction(pvsys[key][dfname], 
                                pvsys[key]["substations"][substat], 
                                pyr_config["bias_correction"]["poly"])

    pyr_results_path = os.path.join(pyr_config['results_path']['main'],
                                    pyr_config['results_path']['cloud_fraction'])
    
    savepath = generate_results_folders(rt_config,pyr_config,pyr_results_path,homepath)
        
    # Save solution for key to file
    save_results(config["description"],key,pvsys[key],dfname,pyr_config,rt_config,savepath,homepath)
