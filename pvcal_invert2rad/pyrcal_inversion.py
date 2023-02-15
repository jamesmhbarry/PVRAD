#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 15:02:41 2018

@author: james

Inversion using results of DISORT simulation for irradiance input and power data as 
measurement vector
"""
#%% Preamble
import os
import numpy as np
from numpy import deg2rad, rad2deg, nan
import pandas as pd
import pickle
import collections
from file_handling_functions import *
from rt_functions import *
from pvcal_forward_model import E_poa_calc, cos_incident_angle, azi_shift
import inversion_functions as inv
import data_process_functions as dpf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import subprocess

#%%Functions
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

def load_data_radsim_results(info,inv_config,rt_config,station_list,home):
    """
    Load results from measurement and from DISORT radiation simulation
    
    args:
    :param info: string with description    
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
    folder_label, filename, (theta_res,phi_res) = \
    generate_folder_names_disort(mainpath_disort,rt_config)
        
    #Choose which stations to load
    if type(station_list) != list:
        station_list = [station_list]
        if station_list[0] == "all":
            station_list = inv_config["pv_stations"]
            #select_system_info = system_info                
    
    #Define empty dictionary of PV systems
    pv_systems = {}    
    
    #Import data for each station, from each campaign
    for station in station_list:                        
        #Load data configuration (different for each campaign)        
        mainpath_data = os.path.join(home,data_config["paths"]["savedata"]
        ["main"],data_config["paths"]["savedata"]["binary"])
        #Define filename
        filename_data = info + '_' + station + '_'\
        + rt_config["timeres"] + '.data'
        
        #Load data from measurement campaign "measurement"
        data_types = data_config["data_types"]
        pvstat, dummy = dpf.load_station_data(mainpath_data,filename_data,data_types,False)
        
        print('Data for %s loaded from %s' %(station,filename_data))
        
        #Load results from DISORT simulation
        filename_sim = filename + info + '_disortres_' + theta_res\
        + '_' + phi_res + '_' + station + '.data'

        try:
            with open(os.path.join(folder_label,filename_sim), 'rb') as filehandle:  
                # read the data as binary data stream
                (pvstat_sim, temp) = pd.read_pickle(filehandle)    
            
            #If the data file is available, extract only RT sim from pvstat_sim
            if pvstat:
                #Get only columns from libradtran, sun position, aerosol, albedo
                pvstat_sim["df_sim"] = pd.concat([pvstat_sim['df'].loc[:,pd.IndexSlice[:,['sun','Aeronet']]],
                pvstat_sim['df'].loc[:,pd.IndexSlice['albedo',:]],pvstat_sim['df'].loc[:,pd.IndexSlice[:,'libradtran']]],axis=1)
                del pvstat_sim['df']
                    
                #Merge dictionaries
                pvstat = merge_two_dicts(pvstat,pvstat_sim)
                del pvstat_sim
                
                #Merge dataframes
                pvstat["df_sim"] = pd.merge(pvstat["df"],pvstat["df_sim"],
                      left_index=True,right_index=True)
                del pvstat["df"]
            else:
                #If datafile was not available, use all info from RT sim (data included)
                #This should normally be the case
                pvstat = pvstat_sim
                pvstat["df_sim"] = pvstat['df']
                del pvstat['df']
            
            #Update dictionary with current station
            if station not in pv_systems:
                pv_systems.update({station:pvstat})                
            else:                    
                pv_systems[station].update({"df_sim":pvstat["df_sim"]})                                

            print('Data for %s loaded from %s' % (station,filename_sim))
        except IOError:
            print('There is no simulation for %s, %s' % (station,info))                
    
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
                        
                    print(f"Nearest COSMO grid point to {key} is at {folder[4:9]}, {folder[14:19]}")
    
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
        
        dfs = [pd.read_csv(os.path.join(pv_systems[key]['path_cosmo_irrad'],\
            iday.replace('-','') + '_irradiance.dat'),sep='\s+',index_col=0,skiprows=2,
            header=None,names=['Edirdown_Wm2','Edirdown_mean_Wm2','Edirdown_iqr_Wm2','Ediffdown_Wm2',
                               'Ediffdown_mean_Wm2','Ediffdown_iqr_Wm2']) for iday in days]
     
        dataframe2 = pd.concat(dfs,axis=0)
        dataframe2.index = pd.to_datetime(dataframe2.index,format='%d.%m.%Y;%H:%M:%S')
    
        dataframe = pd.concat([dataframe,dataframe2],axis=1)
        
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
    if inv_config["cosmo_input_flag"]["surface"]:     
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

def cosine_bias_correction(dataframe,stat_config,coeff_poly):
    """
    Correct the small pyranometers from TROPOS for cosine bias using
    polynomial correction factor calculated by Jonas Witthuhn

    Parameters
    ----------
    dataframe : pandas dataframe with all data including sun position
    stat_config : dictionary with stations configuration including angles
    coeff_poly : dictionary with coefficients for correction

    Returns
    -------
    dataframe : new dataframe with corrected irradiance values

    """

    mu0 = np.cos(np.deg2rad(dataframe[("sza","sun")]))
    
    for substat in stat_config["substat"]:
        if "Pyr" in substat and "SiRef" not in substat:
            #GTI bias correction
            radname = stat_config["substat"][substat]["name"]
            
            mu_IA = cos_incident_angle(np.deg2rad(dataframe[("sza","sun")].values), 
                    azi_shift(np.deg2rad(dataframe[("phi0","sun")].values)),
                      np.deg2rad(stat_config["substat"][substat]["tilt_ap"][0]),
                      np.deg2rad(stat_config["substat"][substat]["azimuth_ap"][0]))
            
            C_GTI = coeff_poly["c_0"]*mu_IA**3 + coeff_poly["c_1"]*mu_IA**2\
                + coeff_poly["c_2"]*mu_IA + coeff_poly["c_3"]
            
            dataframe[(radname,substat)] = dataframe[(radname,substat)]*C_GTI
            
            #GHI bias correction
            radname = stat_config["substat"][substat]["name"].replace("poa","down")
            
            C_GHI = coeff_poly["c_0"]*mu0**3 + coeff_poly["c_1"]*mu0**2\
                + coeff_poly["c_2"]*mu0 + coeff_poly["c_3"]
                                           
            dataframe[(radname,substat)] = dataframe[(radname,substat)]*C_GHI
                                       
    return dataframe                                
       

def inversion_setup(key,pv_station,substat_inv,pyr_config,resolution,optical_flag):
    """
    Setup the various quantities to run the non-linear inversion
    
    args:
    :param key: Code of PV system
    :param pv_station: dictionary with information and data on "key" PV systems
    :param substat_inv: substation to use for inversion
    :param pyr_config: dictionary with pyranometer configuration    
    :param resolution: dictionary with resolution of DISORT simulation    
    :param optical_flag: boolean with flag for optical model
    
    out:
    :return pv_systems: dictionary of PV systems with quantities for inverison added
    :return invdict: dictionary with parameters for inversion
    """      

    invpars = pyr_config["calibration"]
    e_err_rel = pyr_config["pv_stations"][key]["substat"][substat_inv]["e_err_rel"]    
    e_err_min = pyr_config["pv_stations"][key]["substat"][substat_inv]["e_err_min"]    
    var_name = pyr_config["pv_stations"][key]["substat"][substat_inv]["name"]    

    #These are the increments to use for numerical differentiation
    diff_theta = deg2rad(resolution["theta"]/2)
    diff_phi = deg2rad(resolution["phi"]/2)
    diff_n = 0.01

    diffs = collections.namedtuple('diffs', 'theta phi n')

    diffs = diffs(diff_theta,diff_phi,diff_n)

    #Parameters for optimisation
    invdict = {}
    invdict['diffs'] = diffs
    invdict['max_iterations'] = invpars["max_iterations"]
    invdict['converge'] = invpars["convergence_limit"]
    invdict['gamma'] = invpars["gamma"] #Parameter for Levenberg-Marquardt
    
        
    #Set up quantities for inversion
    #Apriori values
    apriori = pv_station["substations"][substat_inv]
    
    #Uncertainties
    #Angles explicitly defined in config file
    theta_ap = deg2rad(apriori['tilt_ap'][0])
    phi_ap = azi_shift(deg2rad(apriori['azimuth_ap'][0]))
    theta_err = deg2rad(apriori['tilt_ap'][1])
    phi_err = deg2rad(apriori['azimuth_ap'][1])
    
    n_ap = invpars["n_ap"][0]
    n_err = invpars["n_ap"][0]*invpars["n_ap"][1]
    
    #This is a list of tuples with
    #(Name,apriori value,apriori error)
    #If apriori error is zero then this parameter will be fixed!
    invdict['pars'] = [('theta',theta_ap,theta_err),
                   ('phi',phi_ap,phi_err)]
    
    if optical_flag:
        invdict['pars'].append(('n',n_ap,n_err))
    
    opt_dict = {}
        
    #Define a-priori state vector
    opt_dict['x_a'] = np.array([invdict['pars'][i][1] for i 
                in range(len(invdict['pars'])) if invdict['pars'][i][2] != 0])
    
    #Define covariance matrix
    opt_dict['S_a'] = np.diag(np.array([invdict['pars'][i][2] for i 
                in range(len(invdict['pars'])) if invdict['pars'][i][2] != 0]))**2            
    
    #Define parameter space x
    opt_dict['x'] = np.zeros((invdict['max_iterations'] + 1,len(opt_dict['x_a'])))
    
    #Save dict to larger dictionary
    pv_station['substations'][substat_inv]['opt'] = opt_dict
    
    #Get the measurement vector depending on type of Pyranometer
    
    pv_station['df_cal'][('E_poa_meas_Wm2',substat_inv)] = pv_station['df_cal'][(var_name,substat_inv)]    
    
    #Measurement uncertainty of 1% but not less than 100W
    pv_station['df_cal'][('E_error_meas',substat_inv)] = pv_station['df_cal'][('E_poa_meas_Wm2',substat_inv)]*e_err_rel
    pv_station['df_cal'][('E_error_meas',substat_inv)].where(pv_station['df_cal']
        [('E_poa_meas_Wm2',substat_inv)] > e_err_min,e_err_min,inplace=True)
    
    #Get the solar angles in radians
    pv_station['df_cal'][('theta0rad','sun')] = deg2rad(pv_station['df_cal'].sza)
    pv_station['df_cal'][('phi0rad','sun')] = deg2rad(pv_station['df_cal'].phi0)        
    
    return pv_station, invdict

def generate_folders(rt_config,pyr_config,path):
    """
    Generate folders for results
    
    args:
    :param rt_config: dictionary with radiative transfer config
    :param pyr_config: dictionary with pyranometer configuration
    :param path: main path for saving files or plots    
    
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
        
    sza_label = "SZA_" + str(int(pyr_config["sza_max"]["calibration"]))
    
    dirs_exist = list_dirs(fullpath)
    fullpath = os.path.join(fullpath,sza_label)
    if sza_label not in dirs_exist:
        os.mkdir(fullpath)
        
#    dirs_exist = list_dirs(os.path.join(path,disort_folder_label,atm_folder_label,aero_folder_label))
#    if model not in dirs_exist:
#        os.mkdir(os.path.join(path,disort_folder_label,atm_folder_label,aero_folder_label,model))
#        
#    dirs_exist = list_dirs(os.path.join(path,disort_folder_label,atm_folder_label,aero_folder_label,model))
#    if T_model not in dirs_exist:
#        os.mkdir(os.path.join(path,disort_folder_label,atm_folder_label,aero_folder_label,model,T_model))    
    
    return fullpath, res_string_label
    
###############################################################
###   Plotting results                                      ###
###############################################################

def plot_fit_results(key,pv_station,substat_inv,rt_config,pyr_config,styles,home,
                     flag,pars):
    """
    Plot results from the non-linear inversion
    
    args:
    :param key: string, name of current pv station
    :param pv_station: dictionary of one PV system given by key
    :param substat_inv: string, name of substation used for inversion
    :param rt_config: dictionary with disort configuration for saving
    :param pyr_config: dictionary with inversion configuration for saving
    :param mainpath: dictionary with paths to save plots
    :param styles: dictionary with plot styles
    :param home: string with homepath
    :param flag: boolean, whether solution has been found or not
    :param pars: list of tuples with parameters (name,ap_value,ap_error)
    """
    
    plt.ioff()
    plt.style.use(styles["single_small"])
    
    mainpath = pyr_config['results_path']
    sza_limit = pyr_config["sza_max"]["calibration"]
    str_sza = str(int(np.round(sza_limit,0)))
    
    savepath = os.path.join(home,mainpath["main"],mainpath["calibration"])
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    
    folder_label, res_string_label = generate_folders(rt_config,pyr_config,savepath)    
    
    if mainpath["plots"]["fits"] not in list_dirs(folder_label):
        os.mkdir(os.path.join(folder_label,mainpath["plots"]["fits"]))
    
    plotpath = os.path.join(folder_label,mainpath["plots"]["fits"])
    
    station_folders = list_dirs(plotpath)
    if key not in station_folders:        
        os.mkdir(os.path.join(plotpath,key))
        
    substat_folders = list_dirs(os.path.join(plotpath,key))
    if substat_inv not in substat_folders:
        os.mkdir(os.path.join(plotpath,key,substat_inv))
    
    save_folder = os.path.join(plotpath,key,substat_inv)
        
    opt_pars = pv_station['substations'][substat_inv]['opt']
    
    if flag:
        if 'Epoa_MAP' in pv_station['df_cal']:
            for iday in pv_station['substations'][substat_inv]['cal_days']:
                fig, ax = plt.subplots(figsize=(9,8))
                
                #Slice all values coming from inversion
                df_inversion_day = pv_station['df_cal'].xs(substat_inv,level='substat',axis=1).loc[iday]
                
                df_sun_day = pv_station['df_cal'].xs('sun',level='substat',axis=1).loc[iday]
                sza_index_day = df_sun_day.loc[df_sun_day.sza <= sza_limit].index    
                
                #Slice all values coming from libradtran
                df_simulation_day = pv_station['df_cal'].xs('libradtran',level='substat',axis=1).loc[iday]
            
                ax.plot(df_inversion_day.index,df_inversion_day.E_poa_meas_Wm2, color='r',linestyle = '-')
                ax.fill_between(df_inversion_day.index, df_inversion_day.E_poa_meas_Wm2 - df_inversion_day.E_error_meas,
                                 df_inversion_day.E_poa_meas_Wm2 + df_inversion_day.E_error_meas,color='r', alpha=0.3)
                
                ax.plot(df_inversion_day.index,df_inversion_day.Epoa_MAP, color='b',linestyle = '-')
                
                ax.fill_between(df_inversion_day.index, df_inversion_day.Epoa_MAP - df_inversion_day.error_fit,
                                 df_inversion_day.Epoa_MAP + df_inversion_day.error_fit,color='b', alpha=0.3)
                
                ax.plot(df_inversion_day.index,df_inversion_day.Epoa_MAP_extra,color='b',linestyle = '--')        
                
                plt.legend((r'$G_{\rm tot,meas}^{\angle}$',r'$G^{\angle}_{\rm tot,mod,SZA \leq ' + str_sza + '^{\circ}}$',
                            r'$G^{\angle}_{\rm tot,mod,SZA > ' + str_sza + '^{\circ}}$'),loc='upper right')
                
                # Make the y-axis label, ticks and tick labels match the line color.
                plt.ylabel('Irradiance ($W/m^2$)')#, color='b')
                plt.xlabel('Time (UTC)')
                plt.title('MAP solution for ' + key + ', ' + substat_inv + ' on ' + iday)
                #plt.ylim([0,50000])
                
                datemin = np.datetime64(iday + ' 04:00:00')
                datemax = np.datetime64(iday + ' 18:00:00')      
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))      
                ax.axvspan(datemin,sza_index_day[0] , alpha=0.2, color='gray')
                ax.axvspan(sza_index_day[-1],datemax , alpha=0.2, color='gray')
                ax.set_xlim([datemin, datemax])
                #plt.axis('square')
                start_y = 0.6
                start_x = 0.42
                
                c_par = 0
                units = [r'$^\circ$',r'$^\circ$','']
                for i, par in enumerate(pars[0:5]):
                    if par[2] != 0:
                        if par[0] == 'theta':
                            parstring = par[0] + ' = ' + str(np.round(np.rad2deg(opt_pars['x_min'][c_par]),2)) + '$\pm$' +\
                                 str(np.round(np.rad2deg(opt_pars['s_vec'][c_par]),2)) + ' ' + units[i]
                        elif par[0] == 'phi':
                            parstring = par[0] + ' = ' + str(np.round(np.rad2deg(azi_shift(opt_pars['x_min'][c_par])),2)) + '$\pm$' +\
                                 str(np.round(np.rad2deg(opt_pars['s_vec'][c_par]),2)) + ' ' + units[i]                        
                        else:                                 
                             parstring = par[0] + ' = ' + str(np.round(opt_pars['x_min'][c_par],2)) + '$\pm$' +\
                                 str(np.round(opt_pars['s_vec'][c_par],2)) + ' ' + units[i]
                        c_par = c_par + 1
                    else:
                        if par[0] == 'theta':
                            parstring = par[0] + ' = ' + str(np.round(np.rad2deg(par[1]),2)) + ' (fixed)'
                        elif par[0] == 'phi':
                            parstring = par[0] + ' = ' + str(np.round(np.rad2deg(azi_shift(par[1])),2)) + ' (fixed)'
                        else:
                            parstring = par[0] + ' = ' + str(par[1]) + ' (fixed)'
                        
                    plt.annotate(parstring,xy=(start_x,start_y - 0.05*i),xycoords='figure fraction',fontsize=14)
                
                plt.annotate('$\chi^2$ = ' + str(np.round(opt_pars['min_chisq'],2)) 
                             ,xy=(start_x,0.15),xycoords='figure fraction',fontsize=14)
                
                fig.tight_layout()
                plt.savefig(os.path.join(save_folder,'chi_sq_fit_' + key + '_' + iday + '_' + 
                                         res_string_label + 'irrad_poa.png'))
                #plt.savefig('chi_sq_fit_' + pv_station['code'] + '_' + test_days[iday] + '_power.eps')
                plt.close(fig)
                
                fig2 = plt.figure(figsize=(9,8))            
                
                df_simulation_day.Etotdown.plot(color='g',style = '-',legend=True)            
                if "Pyr" in substat_inv and "SiRef" not in substat_inv:
                    df_inversion_day.Etotdown_pyr_Wm2.plot(color='g',linestyle = '--',legend=True)                
                    
                df_inversion_day.Etotpoa.plot(color='r',style = '-',legend=True)
                
                df_inversion_day.Edirpoa.plot(color='k',style = '-',legend=True)
                df_inversion_day.Ediffpoa.plot(color='b',style = '-',legend=True)
                #df_inversion_day.Ereflpoa.plot(color='m',style = '-',legend=True)
                df_simulation_day.Ediffdown.plot(color='b',style = '--',legend=True)
                                    
                if "Pyr" in substat_inv:
                    plt.legend((r'$G_{\rm tot,sim}^{\downarrow}$',r'$G_{\rm tot,meas}^{\downarrow}$',
                            r'$G_{\rm tot,sim}^{\angle}$', r'$G_{\rm dir,sim}^{\angle}$',
                            r'$G_{\rm diff,sim}^{\angle}$',r'$G_{\rm diff,sim}^{\downarrow}$'))
                else:
                    plt.legend((r'$G_{\rm tot,sim}^{\downarrow}$',
                            r'$G_{\rm tot,sim}^{\angle}$', r'$G_{\rm dir,sim}^{\angle}$',
                            r'$G_{\rm diff,sim}^{\angle}$',r'$G_{\rm diff,sim}^{\downarrow}$'))
                
                # Make the y-axis label, ticks and tick labels match the line color.
                plt.ylabel('Irradiance ($W/m^2$)')#, color='b')
                plt.xlabel('Time (UTC)')
                plt.title('Irradiance components after calibration for ' + key  + ' on ' + iday)
                #plt.axis('square')
                #plt.ylim([0,1000])
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
                
                fig2.tight_layout()
                plt.savefig(os.path.join(save_folder,'chi_sq_fit_' + key + '_' + iday + '_' + res_string_label
                                         + 'irradiance_comps.png'))
                #plt.savefig('chi_sq_fit_' + pv_station['code'] + '_' + test_days[iday] + '_irradiance.eps')
                plt.close(fig2)
                
            #plot_grid_power_fit(key,pv_station,substat_inv,styles,save_folder,res_string_label,sza_limit)
            
    return folder_label

# def plot_grid_power_fit(key,pv_station,substat_inv,styles,savepath,res_string_label,sza_limit):
#     """
    

#     Parameters
#     ----------
#     key : TYPE
#         DESCRIPTION.
#     pv_station : TYPE
#         DESCRIPTION.
#     substat_inv : TYPE
#         DESCRIPTION.
#     styles : TYPE
#         DESCRIPTION.
#     savepath : TYPE
#         DESCRIPTION.
#     res_string_label : TYPE
#         DESCRIPTION.
#     sza_limit : TYPE
#         DESCRIPTION.

#     Returns
#     -------
#     None.

#     """
    
    
#     plt.ioff()     
#     plt.close()
#     plt.style.use(styles["combo_small"])   
        
#     opt_pars = pv_station['substations'][substat_inv]['opt']
#     num_plots = len(pv_station['cal_days'])
#     if num_plots >= 10:
#         fig, axs = plt.subplots(4, 3, sharey='row')
#     elif num_plots <= 9:
#         fig, axs = plt.subplots(3, 3, sharey='row', figsize=(10,9))    
    
#     axvec = axs.flatten()
#     opt_string = r'$\theta$ (tilt) = ' + "{:.2f}".format(np.round(np.rad2deg(opt_pars['x_min'][0]),2)) + '$\pm $' +\
#                  "{:.2f}".format(np.round(np.rad2deg(opt_pars['s_vec'][0]),2)) + '$^\circ$, $\phi$ (azimuth) = '\
#                  + "{:.2f}".format(np.round(np.rad2deg(azi_shift(opt_pars['x_min'][1])),2)) + '$\pm $'\
#                  + "{:.2f}".format(np.round(np.rad2deg(opt_pars['s_vec'][1]),2)) + '$^\circ$'
    
#     fig.suptitle(key + ', ' + substat_inv + ' : ' + opt_string)    
    
    
#     #Slice all values coming from inversion
#     df_inversion = pv_station['df_cal'].xs(substat_inv,level='substat',axis=1)
    
#     max_df = np.max(np.max(df_inversion[['P_meas_W','P_MAP']]))/1000
#     if max_df > 50:
#         max_pv = np.ceil(max_df/10)*10
#     elif max_df > 25:
#         max_pv = np.ceil(max_df/5)*5
#     else:
#         max_pv = np.ceil(max_df/2)*2

#     for ix, iday in enumerate(pv_station['cal_days']):
        
#         #Slice all values coming from inversion
#         df_inversion_day = df_inversion.loc[iday]
        
#         #Slice all values coming from libradtran
#         #df_simulation_day = pv_station['df_cal'].xs('libradtran',level='substat',axis=1).loc[iday]

#         df_sun_day = pv_station['df_cal'].xs('sun',level='substat',axis=1).loc[iday]
#         sza_index_day = df_sun_day.loc[df_sun_day.sza <= sza_limit].index   
    
#         axvec[ix].plot(df_inversion_day.index,df_inversion_day.P_meas_W/1000,color='r')
#         axvec[ix].fill_between(df_inversion_day.index, df_inversion_day.P_meas_W/1000 - df_inversion_day.P_error_meas/1000,
#                          df_inversion_day.P_meas_W/1000 + df_inversion_day.P_error_meas/1000,color='r', alpha=0.3)
        
#         axvec[ix].plot(df_inversion_day.index,df_inversion_day.P_MAP/1000,color='b')        
#         axvec[ix].fill_between(df_inversion_day.index, df_inversion_day.P_MAP/1000 - df_inversion_day.error_fit/1000,
#                          df_inversion_day.P_MAP/1000 + df_inversion_day.error_fit/1000,color='b', alpha=0.3)
       
#         axvec[ix].plot(df_inversion_day.index,df_inversion_day.P_MAP_extra/1000,color='b',linestyle = '--')        
#         #axvec[ix].set_title(iday + " ",fontsize=14,loc="right",pad=-16)
#         axvec[ix].text(0.5, 0.01, iday + " ",verticalalignment='bottom', horizontalalignment='center',
#         transform=axvec[ix].transAxes, fontsize=14)
#         datemin = np.datetime64(iday + ' 04:00:00')
#         datemax = np.datetime64(iday + ' 18:00:00')        
        
#         axvec[ix].xaxis.set_major_locator(mdates.HourLocator(interval=6))
#         axvec[ix].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
#         axvec[ix].set_xlim([datemin, datemax])
#         axvec[ix].set_xlabel("")
#         axvec[ix].set_ylim([0,max_pv])

#         axvec[ix].axvspan(datemin,sza_index_day[0] , alpha=0.2, color='gray')
#         axvec[ix].axvspan(sza_index_day[-1],datemax , alpha=0.2, color='gray')
    
#     if len(axvec) - len(pv_station['cal_days']) != 0:
#         for ix_extra in range(ix + 1,len(axvec)):
#             axvec[ix_extra].xaxis.set_major_locator(mdates.HourLocator(interval=6))
#             axvec[ix_extra].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
#             axvec[ix_extra].set_xlim([datemin, datemax])        
#             axvec[ix_extra].set_ylim([0,max_pv])
            
#     for ax in axvec:
#         ax.label_outer()
    
# #    plt.annotate(r'$\theta$ (tilt) = ' + "{:.2f}".format(np.round(np.rad2deg(opt_pars['x_min'][0]),2)) + '$\pm $' + 
# #                 "{:.2f}".format(np.round(np.rad2deg(opt_pars['s_vec'][0]),2)) + '$^\circ$, $\phi$ (azimuth) = ' 
# #                 + "{:.2f}".format(np.round(np.rad2deg(azi_shift(opt_pars['x_min'][1])),2)) + '$\pm $'
# #                 + "{:.2f}".format(np.round(np.rad2deg(opt_pars['s_vec'][1]),2)) + '$^\circ$',
# #                 xy=(0.2,0.95),xycoords='figure fraction',fontsize=16)     
# #    
# #    fig.text(0.5, 0.04, 'Time (UTC)', ha='center',fontsize=20)
# #    fig.text(0.05, 0.5, 'Power (kW)', va='center', rotation='vertical',fontsize=20)
    
#     #fig.autofmt_xdate(rotation=45,ha='center') 
    
#     fig.legend((r'$P_{\rm AC,meas}$',r'$P_{\rm AC,mod}$'),bbox_to_anchor=(1., 0.58), loc='upper right')
#     #fig.legend((r'$P_{\rm AC,meas}$',r'$P_{\rm AC,mod}$'),loc='upper center',bbox_to_anchor=(0.51, 0.45))#loc=[0.43,0.38])
#     # hide tick and tick label of the big axes
    
#     fig.add_subplot(111, frameon=False)
#     plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
#     plt.grid(False)
#     plt.xlabel("Time (UTC)")
#     plt.ylabel("Power (kW)")
        
#     fig.subplots_adjust(hspace=0.07,wspace=0.05)
#     #fig.tight_layout()
#     #fig.autolayout = False
#     fig.subplots_adjust(top=0.94)
#     plt.savefig(os.path.join(savepath,'chi_sq_fit_grid_' + key + '_' + substat_inv + '_' + res_string_label + 'power.png'))

def save_calibration_results(key,pv_station,config,rt_config,pyr_config,path,home):
    """
    Save calibration results to a binary file
    
    args:
    :param key: string, name of PV station
    :param pv_station: dictionary of PV station with information and data
    :param config: dictionary with data configuration
    :param rt_config: dictionary with configuration of RT simulation
    :param pyr_config: dictionary with configuration of pyranometers
    :param path: path for saving results
    :param home: string, home path
    
    """
    info = config["description"]
    atm_source = rt_config["atmosphere"]
    asl_source = rt_config["aerosol"]["source"]
    asl_res = rt_config["aerosol"]["data_res"]
    res = rt_config["disort_rad_res"]    
    
    filename = 'calibration_results_'
    if atm_source == 'cosmo':
        filename = filename + 'atm_'
    if asl_source != 'default':
        filename = filename + 'asl_' + asl_res + '_'
    
    filename = filename + info + '_disortres_' + str(res["theta"]).replace('.','-')\
                    + '_' + str(res["phi"]).replace('.','-') + '_'
    
    filename_stat = filename + key + '.data'
    with open(os.path.join(path,filename_stat), 'wb') as filehandle:  
        # store the data as binary data stream
        pickle.dump((pv_station, rt_config, pyr_config), filehandle)
        
    print(f'Results written to file {filename_stat} in folder {path}')

def pyranometer_calibration(key,pv_station,substat_inv,invdict,sza_limit,const_opt,
                         angles,plot_styles,optical_flag):
    """
    Perform non-linear inversion to calibrate angles of pyranometer
    
    args:
    :param key: string, name of PV station
    :param pv_station: dictionary with PV station info and dataframe
    :param substat_inv: name of substation (pyranometer) to use for inversion
    :param invdict: dictionary with parameters for inversion procedure
    :param sza_limit: float, maximum SZA to use for inversion
    :param const_opt: named tuple with optical constants
    :param angles: named tuple with angles for DISORT grid
    :param plot_styles: dictionary with plot styles
    :param optical_flag: boolean with flag for optical model
    
    out:
    :return pv_station: dictionary with PV station info including calibration
    :return folder_label: string with folder label for saving results
    """
    sun_position = collections.namedtuple('sun_position','sza azimuth')    
        
    found = False    
    switch = False
    #temporary declaration to get dataframe from list
    dataframe = pv_station['df_cal']
    opt_pars = pv_station['substations'][substat_inv]['opt']
    x = opt_pars['x']
    x_a = opt_pars['x_a']
    S_a = opt_pars['S_a']
    #x_constr = opt_pars['x_constr']
    chi_sq = np.zeros(invdict['max_iterations'] + 1)
    chi_sq_retr = np.zeros(invdict['max_iterations'] + 1)
    delta_chisq = np.zeros(invdict['max_iterations'])
    grad_chisq = np.zeros((invdict['max_iterations'],len(x_a)))
    di_sq = np.zeros(invdict['max_iterations'] + 1)
    gamma = np.zeros(invdict['max_iterations'] + 1)
    
    #Assign a-priori parameters to station dictionary
    pv_station['substations'][substat_inv]['ap_pars'] = invdict['pars']
    
    #Use only values up to defined SZA limit
    print(('Extracting values with SZA < %g' % sza_limit))
    sza_index = dataframe.loc[dataframe[('sza','sun')] <= sza_limit].index
    
    #Drop missing values
    print('Dropping missing data')
    notnan_index = dataframe.loc[dataframe[('E_poa_meas_Wm2',substat_inv)].notna()].index    
    
    #Check whether all inversion days have data
    pv_station['substations'][substat_inv]['cal_days'] = \
    pd.to_datetime(notnan_index.date).unique().strftime('%Y-%m-%d')    
    
    new_index = sza_index.intersection(notnan_index)
    
    #Convert all pandas series into numpy arrays for calculation speed
    Edirdown = dataframe.Edirdown.loc[new_index].astype(float).values.flatten()
    #TambC = dataframe.T_ambient_2M_C.loc[new_index].values.flatten()
    sun_pos = sun_position(dataframe.theta0rad.loc[new_index].values.flatten(),
                           dataframe.phi0rad.loc[new_index].values.flatten())
    Epoameas = dataframe[('E_poa_meas_Wm2',substat_inv)].loc[new_index].values.flatten()
    E_error_meas = dataframe[('E_error_meas',substat_inv)].loc[new_index].values.flatten()
    #alb = dataframe.albedo.loc[new_index].values.flatten()
    #vwind = dataframe.v_wind_10M.loc[new_index].values.flatten()
    
    #Get the diffuse radiance field
    diff_field = np.zeros((len(new_index),len(angles.theta),len(angles.phi)))                
    for i, itime in enumerate(new_index):
        diff_field[i,:,:] = dataframe.loc[itime,('Idiff','libradtran')].values #.flatten()
    
    #Define Jacobian matrix
    K_matrix = np.zeros((len(new_index),len(x_a)))
    
    #Measurement covariance matrix
    S_eps = np.identity(len(new_index))*E_error_meas**2 # Measurement error in W^2
            
    print(('Clear sky days for %s calibration are: %s' % (substat_inv,
        [day for day in pv_station['substations'][substat_inv]["cal_days"]])))
    print(('Running PYRCAL inversion for %s, substation %s with a-priori values:' 
          % (key,substat_inv)))
    substat_ap = pv_station['substations'][substat_inv]    
    
    if optical_flag:
        print(('theta: %s, phi: %s, n: %s' 
          % (substat_ap["tilt_ap"],substat_ap["azimuth_ap"],inv_dict["n_ap"])))
    else:
        print(('theta: %s, phi: %s'
          % (substat_ap["tilt_ap"],substat_ap["azimuth_ap"])))
            
    print(('Error on irradiance measurement is %s (relative) and at least %s W/m^2'
          % (substat_ap["e_err_rel"],substat_ap["e_err_min"])))
        
    #Start by using simple Gauss-Newton
    method = 'lm' 
    
    while not switch:
        k = 0           
        if method == 'lm':
            #Set the intial value of gamma    
            gamma[k] = invdict["gamma"]["initial_value"]    
            print(('Using the Levenberg-Marquardt method with gamma_0 = %g' % gamma[k]))
        elif method == 'gn':
            print('Using the Gauss-Newton method (gamma = 0)')
            gamma[k] = 0

        while True:
            #Start at a-priori values
            if k == 0:
                x[k,:] = x_a
            
            irrad_params = []
    
            c_par = 0
            #Here we take parameters either from state update or keep them fixed
            #Depending on the uncertainty in config file
            for i, par in enumerate(invdict['pars']):
                if par[0] in ['theta','phi','n']:
                    if par[2] == 0:
                        irrad_params.append(par[1])
                    else:
                        irrad_params.append(x[k,c_par])
                        c_par = c_par + 1
            
            #Calculate modelled irradiance and derivatives
            irrad_model = E_poa_calc(irrad_params,Edirdown,diff_field,
                         sun_pos,angles,const_opt,invdict['diffs'],optical_flag)
            
            c_par = 0
            for i, par in enumerate(invdict['pars']):
                if par[2] != 0:
                    if par[0] == 'theta':
                        #Tilt angle theta
                        K_matrix[:,c_par] = irrad_model['dEs_theta']['dEtot_dtheta']
                    if par[0] == 'phi':
                        #Azimuth angle phi            
                        K_matrix[:,c_par] = irrad_model['dEs_phi']['dEtot_dphi']
                    if par[0] == 'n':    
                        #Refractive index n
                        K_matrix[:,2] = irrad_model['dEs_n']['dEtot_dn']
                    
                    c_par = c_par + 1
            
            #Calculate chi squared function for current iteration
            chi_sq[k] = inv.chi_square_fun(x[k,:],x_a,Epoameas,irrad_model['Etotpoa'],S_a,S_eps)
            chi_sq_retr[k] = inv.chi_sq_retrieval(Epoameas,irrad_model['Etotpoa'],K_matrix,S_a,S_eps)
            
            if k < invdict['max_iterations']:
                if k > 0:
                    #Change in chi squared function
                    delta_chisq[k] = chi_sq[k] - chi_sq[k - 1]
                                
                    #Levenberg-Marquardt
                    if method == 'lm':
                        if delta_chisq[k] > 0:
                            gamma[k + 1] = gamma[k]*invdict["gamma"]["factor"]
                            x[k + 1,:] = inv.x_non_lin(x[k,:],x_a,Epoameas,irrad_model['Etotpoa'],K_matrix,S_a,S_eps,gamma[k + 1])                    
                        else:
                             #Iterate to find better solution
                            x[k + 1,:] = inv.x_non_lin(x[k,:],x_a,Epoameas,irrad_model['Etotpoa'],K_matrix,S_a,S_eps,gamma[k])
                            gamma[k + 1] = gamma[k]/invdict["gamma"]["factor"]
                    else:
                        x[k + 1,:] = inv.x_non_lin(x[k,:],x_a,Epoameas,irrad_model['Etotpoa'],K_matrix,S_a,S_eps,gamma[k])
                        gamma[k + 1] = gamma[k] # leave gamma unchanged
                        
                    grad_chisq[k,:] = inv.grad_chi_square(x[k,:],x_a,Epoameas,irrad_model['Etotpoa'],K_matrix,S_a,S_eps)
                    di_sq[k] = inv.d_i_sq(x[k,:],x[k + 1,:],x_a,Epoameas,irrad_model['Etotpoa'],K_matrix,S_a,S_eps)
    
                else:
                    #First iteration
                    x[k + 1,:] = inv.x_non_lin(x[k,:],x_a,Epoameas,irrad_model['Etotpoa'],K_matrix,S_a,S_eps,gamma[k])
                    delta_chisq[k] = nan
                    di_sq[k] = nan
                    gamma[k + 1] = gamma[k]
                
                if np.fmod(k,10) == 0 and k > 0:
                    print('%d: x: %s chi_sq: %.6f, delta chi_sq: %.6f, d_i_sq: %.6f'\
                          % (k,x[k,:],chi_sq[k],delta_chisq[k],di_sq[k]))                     
                
                #Check whether minimum has been found
                if k > 0:
                    #Check whether iteration converges using di_sq from Rodgers
                    if di_sq[k] > invdict["converge"] or di_sq[k] < 0.0:  # or abs(delta_chisq[k]) > 1e-2:
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
                        
                        print('There were %d measurements used, retrieval chi_sq is %.3f' % (len(S_eps),chi_sq_retr[k]))
                        print('MAP Solution for %s, %s after %d iterations: Chi_sq: %.6f, delta Chi_sq: %.6f, di_sq: %.6f, chi_sq_ret: %.6f, d_s: %.3f'
                                   % (key,substat_inv,min_ix, min_chisq, delta_chisq[k], min_dsq, chi_sq_retr[k], d_s))
                              
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
                        
                        # if optical_flag:
                        #     print(('MAP Solution for %s, %s after %d iterations: Chi_sq: %.6f, delta Chi_sq: %.6f, di_sq: %.6f, chi_sq_ret: %.6f, d_s: %.3f,\
                        #            Tilt angle: %g, Rotation angle: %g, n: %g'
                        #            % (key, substat_inv, min_ix, min_chisq, delta_chisq[k], min_dsq, chi_sq_retr, d_s, rad2deg(x_min[0]),
                        #               rad2deg(azi_shift(x_min[1])), x_min[2])))
                        # else:
                        #     print(('MAP Solution for %s, %s after %d iterations: Chi_sq: %.6f, delta Chi_sq: %.6f, di_sq: %.6f, chi_sq_ret: %.6f, d_s: %.3f,\
                        #            Tilt angle: %g, Rotation angle: %g'
                        #            % (key, substat_inv, min_ix, min_chisq, delta_chisq[k], min_dsq, chi_sq_retr, d_s, rad2deg(x_min[0]),
                        #               rad2deg(azi_shift(x_min[1])))))
                        
                        break
                else:
                    k = k + 1
            else:                
                if method == "gn":
                    print(('Could not converge to a solution for %s, %s after %d iterations' % (key,substat_inv,k)))
                    print('Trying with Levenberg-Marquardt method')      
                    method = "lm"
                    
                elif method == "lm":
                    print(('Could not converge to a solution for %s, %s after %d iterations' % (key,substat_inv,k)))
                    x_min = nan
                    min_chisq = nan
                    S_hat = nan
                    s_vec = nan
                    chi_sq_retr = nan
                    A = nan
                    info = nan
                    S_dely = nan
                    switch = True
                    
                break            
    
    #Save optimisation parameters to dictionary
    opt_pars['x_min'] = x_min
    opt_pars['chi_sq'] = chi_sq
    opt_pars['min_chisq'] = min_chisq
    opt_pars['delta_chisq'] = delta_chisq
    opt_pars['K_mat'] = K_matrix
    opt_pars['S_eps'] = S_eps
    opt_pars['S_hat'] = S_hat
    opt_pars['s_vec'] = s_vec      
    opt_pars['chi_sq_retr'] = chi_sq_retr
    opt_pars['A'] = A
    opt_pars['info'] = info
    opt_pars['S_dely'] = S_dely
    opt_pars['d_isq'] = di_sq
       
    #Calculate power, irradiance etc at the solution x_min
    if found:
        #Go back to the entire dataframe (all SZA from simulation)
        Edirdown = dataframe.Edirdown.astype(float).values.flatten()
        #TambC = dataframe.T_ambient_2M_C.values.flatten()
        sun_pos = sun_position(dataframe.theta0rad.values.flatten(),
                           dataframe.phi0rad.values.flatten())
        Epoameas = dataframe[('E_poa_meas_Wm2',substat_inv)].values.flatten()
        E_error_meas = dataframe[('E_error_meas',substat_inv)].values.flatten()
        #alb = dataframe.albedo.values.flatten()
        #vwind = dataframe.v_wind_10M.values.flatten()
        
        diff_field = np.zeros((len(dataframe),len(angles.theta),len(angles.phi)))                
        for i, itime in enumerate(dataframe.index):
            diff_field[i,:,:] = dataframe.loc[itime,('Idiff','libradtran')].values
        
        K_matrix = np.zeros((len(dataframe.index),len(x_a)))

        c_par = 0
        irrad_params = []
        #Here we take parameters either from state update or keep them fixed
        #Depending on the uncertainty in config file
        for i, par in enumerate(invdict['pars']):
            if par[0] in ['theta','phi','n']:
                if par[2] == 0:
                    irrad_params.append(par[1])
                else:
                    irrad_params.append(x_min[c_par])
                    c_par = c_par + 1        
        
        model_solution = \
        E_poa_calc(irrad_params,Edirdown,diff_field,
                sun_pos,angles,const_opt,invdict['diffs'],optical_flag) #,K_matrix,T_model)
        
        #Optimal parameters - both those varied and fixed!
        optpars = []    
        c_par = 0
        for i, par in enumerate(invdict['pars']):
            if par[2] != 0:
                optpars.append((par[0],irrad_params[i],s_vec[c_par]))
                c_par = c_par + 1
            else:
                optpars.append(par)
                        
        pv_station['substations'][substat_inv]['opt_pars'] = optpars
        
        #Solution for whole SZA range
        dataframe[('Epoa_MAP',substat_inv)] = model_solution['Etotpoa']
        
        #Extra part of solution (include endpoints for pretty plots)
        dataframe[('Epoa_MAP_extra',substat_inv)] = dataframe[('Epoa_MAP',substat_inv)]
        sza_index_extra = dataframe.loc[dataframe[('sza','sun')] < sza_limit - 5.0].index    
        dataframe.loc[sza_index_extra,('Epoa_MAP_extra',substat_inv)] = np.nan
        
        #Real part of solution (up to SZA limit)
        dataframe[('Epoa_MAP',substat_inv)] = dataframe.loc[sza_index,('Epoa_MAP',substat_inv)]\
                                            .reindex(dataframe.index)
        
        dataframe[('Etotpoa',substat_inv)] = model_solution['Etotpoa']
        dataframe[('Edirpoa',substat_inv)] = model_solution['Edirpoa']
        dataframe[('Ediffpoa',substat_inv)] = model_solution['Ediffpoa']
        #dataframe[('Ereflpoa',substat_inv)] = model_solution['Ereflpoa']
#        dataframe[('T_module_inv_C',substat_inv)] = model_solution['T_module']
#        dataframe[('eff_temp_inv',substat_inv)] = model_solution['eff_temp']

        dataframe[('error_fit',substat_inv)] = np.nan
        dataframe.loc[new_index,('error_fit',substat_inv)] = np.sqrt(np.diagonal(opt_pars['S_dely']))
    
        pv_station['df_cal'] = dataframe
    
    #Plot results for key
    print(('Plotting results for %s, %s' % (key,substat_inv)))
    folder_label = plot_fit_results(key,pv_station,substat_inv,rt_config,pyr_config,
                                    plot_styles,homepath,found,invdict["pars"])    
    
    return pv_station, folder_label

def recalibrate_bias_correction(substat_name,dataframe,stat_pars,coeff_poly):
    """
    Recalibrate the irradiance data to take into account bias correction

    Parameters
    ----------
    substat_name : stirng, name of substation
    dataframe : dataframe with relevant irradiance and sun position data
    stat_pars : dictionary with parameters of specific substation
    coeff_poly : dictionary with coefficients for correction

    Returns
    -------
    dataframe with recalibrated irradiance

    """
    
    radname = stat_pars["name"]
            
    mu_IA_old = cos_incident_angle(np.deg2rad(dataframe[("sza","sun")].values), 
                    azi_shift(np.deg2rad(dataframe[("phi0","sun")].values)),
                      np.deg2rad(stat_pars["tilt_ap"][0]),
                      np.deg2rad(stat_pars["azimuth_ap"][0]))
    
    C_GTI_old = coeff_poly["c_0"]*mu_IA_old**3 + coeff_poly["c_1"]*mu_IA_old**2\
                + coeff_poly["c_2"]*mu_IA_old + coeff_poly["c_3"]
    
    etotpoa_old = dataframe[(radname,substat_name)]/C_GTI_old
    
    mu_IA_new = cos_incident_angle(np.deg2rad(dataframe[("sza","sun")].values), 
                    azi_shift(np.deg2rad(dataframe[("phi0","sun")].values)),
                      stat_pars["opt_pars"][0][1],azi_shift(stat_pars["opt_pars"][1][1]))
    
    C_GTI_new = coeff_poly["c_0"]*mu_IA_new**3 + coeff_poly["c_1"]*mu_IA_new**2\
                + coeff_poly["c_2"]*mu_IA_new + coeff_poly["c_3"]
                
    dataframe[(radname,substat_name)] = etotpoa_old*C_GTI_new
    
    return dataframe
    

#%%Main program      
#######################################################################
###                         MAIN PROGRAM                            ###
#######################################################################
#def main():
import argparse
    
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--configfile", help="yaml file containing config")
parser.add_argument("-s","--station",nargs='+',help="station for which to perform inversion")
#parser.add_argument("-c","--campaign",nargs='+',help="measurement campaign for which to perform simulation")
args = parser.parse_args()
    
if args.configfile:
    config_filename = os.path.abspath(args.configfile) #"config_PYRCAL_2018_messkampagne.yaml" #
else:
    config_filename = "config_PYRCAL_2018_messkampagne.yaml"
 
config = load_yaml_configfile(config_filename)

#Load data configuration
data_config = load_yaml_configfile(config["data_configfile"])

#Load pyranometer configuration
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

print(('Pyranometer angle calibration with Bayesian inversion for %s' % stations))
print('Using data from %s' % config["description"])
print('Loading results from radiative transfer simulation')
if rt_config["disort_base"]["pseudospherical"]:
    print('Results are with pseudospherical atmosphere')
else:
    print('Results are with plane-parallel atmosphere')
print('Wavelength range is set to %s nm' % rt_config["common_base"]["wavelength"]["pyranometer"])
print('Molecular absorption is calculated using %s' % rt_config["common_base"]["mol_abs_param"])

pvsys = load_data_radsim_results(config["description"],pyr_config,rt_config,stations,homepath)

sza_max = pyr_config["sza_max"]["calibration"]

albedo = rt_config["albedo"]

disort_res = rt_config["disort_rad_res"]
grid_dict = define_disort_grid(disort_res)

opt_dict = pyr_config["optics"]
optics_flag = opt_dict["flag"]
optics = collections.namedtuple('optics', 'kappa L')
const_opt = optics(opt_dict["kappa"],opt_dict["L"])

angle_grid = collections.namedtuple('angle_grid', 'theta phi umu')
angle_arrays = angle_grid(grid_dict["theta"],deg2rad(grid_dict["phi"]),grid_dict["umu"])

#Prepare surface inputs
if pyr_config["surface_data"] == "cosmo":
    pvsys = prepare_surface_data(config["description"],pyr_config,pvsys,
                                 rt_config["test_days"],homepath)

for key in pvsys:            
    print("\nPerforming calibration for %s" % key)
    if pyr_config["bias_correction"]["flag"]:
        print('Correcting cosine bias')
        pvsys[key]['df_sim'] = cosine_bias_correction(pvsys[key]['df_sim'],pyr_config["pv_stations"][key],
                                              pyr_config["bias_correction"]["poly"])
        
    #Define days to use for inversion (the same for all substations at a particular station)
    pvsys[key]['cal_days'] = [day.strftime('%Y-%m-%d') for day in 
         pyr_config["pv_stations"][key]["calibration_days"]]
    
    #Extract only the days we want for the inversion (some clear sky days are not clear!)
    dfs = [pvsys[key]['df_sim'].loc[iday] for iday in pvsys[key]['cal_days']]
    pvsys[key]['df_cal'] = pd.concat(dfs,axis=0)    
    
    pvsys[key]['substations'] = pyr_config["pv_stations"][key]["substat"]
        
    for substat in pvsys[key]['substations']:        
        pvsys[key], inv_dict = inversion_setup(key,pvsys[key],substat,pyr_config,
             disort_res,optics_flag)

        pvsys[key], savepath = pyranometer_calibration(key,pvsys[key],substat,
             inv_dict,sza_max,const_opt,angle_arrays,
             data_config["plot_styles"],optics_flag)
        
        if "Pyr" in substat and "SiRef" not in substat and "opt_pars" in pvsys[key]["substations"][substat]:
            print(f"Recalibrating cosine bias for {substat}")
            for df in ["df_cal","df_sim"]:
                pvsys[key][df] = recalibrate_bias_correction(substat,
                                 pvsys[key][df],pvsys[key]["substations"][substat],
                                pyr_config["bias_correction"]["poly"]) 
            
        
    # Save solution for key to file
    save_calibration_results(key,pvsys[key],config,rt_config,pyr_config,savepath,homepath)


