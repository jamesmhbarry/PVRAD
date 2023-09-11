#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 08:21:05 2020

@author: james
"""

import os
import numpy as np
import pandas as pd
import pickle
import collections
from file_handling_functions import *
from rt_functions import *
from pvcal_forward_model import E_poa_calc, ang_response, cos_incident_angle, azi_shift
from pvcal_forward_model import temp_model_tamizhmani, temp_model_faiman, temp_model_king 
from vorwaertsmodell_reg_new_ohne_Matrizen_schalter_JB import F_temp_model_dynamic, F_temp_model_static
from vorwaertsmodell_reg_new_ohne_Matrizen_schalter_JB import gewichtsfaktoren_unnormiert, matrix_smoothing_function
import data_process_functions as dpf
import diode_models as dm
import ephem
import scipy.constants as const
from scipy import optimize
from scipy.interpolate import interp1d



###############################################################
###   general functions to load and process data    ###
###############################################################

def generate_folder_names_pvcal(rt_config,pvcal_config,inv_model):
    """
    Generate folder structure to retrieve PVCAL simulation results
    
    args:    
    :param rt_config: dictionary with RT configuration
    :param pvcal_config: dictionary with PVCAL configuration
    :param inv_model: string, either "current" or "power"
    
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
        
    if inv_model == "power":
        model = pvcal_config["inversion"]["power_model"]
        eff_model = pvcal_config["eff_model"]
        
    elif inv_model == "current":
        model = "Diode_Model"
        eff_model = ""
    T_model = pvcal_config["T_model"]["model"]
    sza_label = "SZA_" + str(int(pvcal_config["sza_max"]["disort"]))

    folder_label = os.path.join(atm_geom_folder,wvl_folder_label,disort_folder_label,
                                atm_folder_label,aero_folder_label,model,eff_model,
                                T_model,sza_label)    
    
    return folder_label, filename, (theta_res,phi_res)
    

def load_radsim_calibration_results(info,rt_config,pvcal_config,
                                    pvrad_config,station_list,home):
    """
    Load results from DISORT radiation simulation as well as calibration results
    
    args:
    :param info: string with description of results
    :param rt_config: dictionary with current RT configuration
    :param pvcal_config: dictionary with PVCAL configuration
    :param pvrad_config: dictionary with PV2rad configuration
    :param station_list: list of stations for which simulation was run
    :param home: string with homepath
    
    out:
    :return pv_systems: dictionary of PV systems with data
    """
            
    mainpath_pvcal = os.path.join(home,pvrad_config['results_path']['main'],
                                  pvrad_config['results_path']['calibration'])
    
    pv_systems = {}
    if type(station_list) != list:
        station_list = [station_list]
        if station_list[0] == "all":
            station_list = pvrad_config["pv_stations"]
        
    calibration_source = pvrad_config["calibration_source"]    
    
    #get description/s
    if len(calibration_source) > 1:
        infos = '_'.join(calibration_source)
    else:
        if type(calibration_source) == list:
            infos = calibration_source[0]
        else:
            infos = calibration_source    

    for station in station_list:      
        for substat_type in pvrad_config["pv_stations"][station]:
            model_type = pvrad_config["pv_stations"][station][substat_type]["type"]
            folder_label_pvcal, filename_pvcal, (theta_res,phi_res) = \
                generate_folder_names_pvcal(rt_config,pvcal_config,model_type)
          
            #Load calibration data for inversion onto irradiance
            filename_pvcal_stat = filename_pvcal + infos + '_disortres_' +\
                          theta_res + '_' + phi_res + '_' + station + '.data'                      
            
            with open(os.path.join(mainpath_pvcal,folder_label_pvcal,
                                   filename_pvcal_stat), 'rb') as filehandle:  
                # read the data as binary data stream
                (pvstat, rtcon, pvcalcon) = pd.read_pickle(filehandle)            
    
            pvstat[f"df_cal_{model_type}"] = pvstat.pop("df_cal")
            print('Calibration and simulation data for %s, %s loaded from %s' % (station,substat_type,filename_pvcal_stat))
    
            if station not in pv_systems:
                pv_systems.update({station:pvstat})
            else:
                pv_systems[station] = merge_two_dicts(pv_systems[station], pvstat)
    
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
    :return pv_systems: dictionary of PV systems with dataframes and other information    
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


def get_sun_position(pvstation,df,sza_limit):
    """
    Using PyEphem to calculate the sun position
    
    args:    
    :param pvstation: dictionary of one PV system with data
    :param df: string, name of dataframe
    :param sza_limit: float defining maximum solar zenith angle for simulation
    
    out:
    :return: dataframe with sun position
    
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

def linear_fit_water_vapour(key, pv_station, pvrad_config, const_opt, angles):
    """
    Perform linear fit of spectral mismatch with water vapour

    Parameters
    ----------
    key : string, name of station
    pv_station : dictionary with information and data for PV station
    pvrad_config : dictionary with inversion configuration
    const_opt : named tuple, constants for optical model
    angles : named tuple with angles for DISORT radiance grid

    Returns
    -------
    pv_station : dictionary with all information and data for PV station

    """
    
    
    years = [campaign.split('_')[1] for campaign in pvrad_config["inversion_source"]]            
        
    for substat_type in pv_station["substations"]:
        
        df_h2o = pd.concat([pv_station[f"df_sim_{year}"].loc[:,pd.IndexSlice[:,["libradtran_pv",\
                   "libradtran_pyr","sun","cosmo","Aeronet"]]] for year in years if f"mk_{year}"
                            in pv_station["substations"][substat_type]["source"]],axis=0)
        
        for substat in pv_station["substations"][substat_type]["data"]:
            
            substat_pars = pv_station["substations"][substat_type]["data"][substat] 
            substat_pars["pars_spec_fit"] = {}
            
            if "opt_pars" in substat_pars:    
                 irrad_pars = [substat_pars["opt_pars"][0][1],
                  substat_pars["opt_pars"][1][1],
                  substat_pars["opt_pars"][2][1]]
            else:
                 irrad_pars = [substat_pars["ap_pars"][0][1],
                  substat_pars["ap_pars"][1][1],
                  substat_pars["ap_pars"][2][1]]
                 
            for year in years:
                print(f"Performing H2O spectral fit for {substat} in {year}")
                year_flag = df_h2o.index.year == int(year)
            
                substat_pars["pars_spec_fit"].update({year:{}})
                     
                #This is the horizontal case, with absorption due to optical model
                if irrad_pars[0] == 0 and irrad_pars[1] == 0:
                    
                    df_h2o.loc[year_flag,('Etotpoa_pv_Wm2',substat)] = \
                    (df_h2o.loc[year_flag,('Edirdown','libradtran_pv')] +\
                    df_h2o.loc[year_flag,('Ediffdown','libradtran_pv')])*\
                    ang_response(0., irrad_pars[2], const_opt.kappa, const_opt.L)
                    
                    df_h2o.loc[year_flag,('Etotpoa_pyr_Wm2',substat)] = \
                    (df_h2o.loc[year_flag,('Edirdown','libradtran_pyr')] +\
                    df_h2o.loc[year_flag,('Ediffdown','libradtran_pyr')])*\
                    ang_response(0., irrad_pars[2], const_opt.kappa, const_opt.L)
                    
                    df_h2o.loc[year_flag,("ratio_Etotpoa",substat)] = \
                    df_h2o.loc[year_flag,('Etotpoa_pv_Wm2',substat)]/\
                    df_h2o.loc[year_flag,('Etotpoa_pyr_Wm2',substat)]
                        
                    df_h2o.loc[year_flag,("ratio_Etotdown",substat)] = \
                    df_h2o.loc[year_flag,('Etotpoa_pv_Wm2',substat)]/\
                    df_h2o.loc[year_flag,('Etotpoa_pyr_Wm2',substat)]
                        
                    df_h2o.loc[year_flag,("diffuse_ratio_Etotdown_pyr",substat)] = \
                    df_h2o.loc[year_flag,'Ediffdown','libradtran_pyr']/\
                    df_h2o.loc[year_flag,('Etotpoa_pyr',substat)]
                        
                    df_h2o.loc[year_flag,('cos_poa_dir',substat)] = \
                    np.cos(np.deg2rad(df_h2o.loc[year_flag,("sza","sun")]))
                    df_h2o.loc[year_flag,('cos_sza',"sun")] = \
                    np.cos(np.deg2rad(df_h2o.loc[year_flag,("sza","sun")]))
                            
                else:
                    #Calculate plane of array irradiance
                    Edirdown_pv = df_h2o.loc[year_flag,("Edirdown","libradtran_pv")].values.flatten()
                    Edirdown_pyr = df_h2o.loc[year_flag,("Edirdown","libradtran_pyr")].values.flatten()
                    Ediffdown_pv = df_h2o.loc[year_flag,("Ediffdown","libradtran_pv")].values.flatten()
                    Ediffdown_pyr = df_h2o.loc[year_flag,("Ediffdown","libradtran_pyr")].values.flatten()
                    sun_position = collections.namedtuple('sun_position','sza azimuth')
                    sun_pos = sun_position(np.deg2rad(df_h2o.loc[year_flag,("sza","sun")].values.flatten()),
                                       np.deg2rad(df_h2o.loc[year_flag,("phi0","sun")].values.flatten()))                
                    
                    #Get the diffuse radiance field
                    diff_field_pv = np.zeros((len(df_h2o.loc[year_flag]),len(angles.theta),len(angles.phi)))                
                    diff_field_pyr = np.zeros((len(df_h2o.loc[year_flag]),len(angles.theta),len(angles.phi)))                
                    for i, itime in enumerate(df_h2o.loc[year_flag].index):
                        diff_field_pv[i,:,:] = df_h2o.loc[itime,('Idiff','libradtran_pv')].values #.flatten()
                        diff_field_pyr[i,:,:] = df_h2o.loc[itime,('Idiff','libradtran_pyr')].values #.flatten()
                        
                    #Calculate POA irradiance for PV, with optical model
                    dict_E_poa_calc_pv = E_poa_calc(irrad_pars,Edirdown_pv,diff_field_pv,
                              sun_pos,angles,const_opt,None,True)
                    
                    #Calculate POA irradiance for broadband irradiance, with optical model (need this since it is for temperature)
                    dict_E_poa_calc_pyr = E_poa_calc(irrad_pars,Edirdown_pyr,diff_field_pyr,
                              sun_pos,angles,const_opt,None,True)
                    
                    #Note that this is the ratio under glass, ie including optical model for both silicon PV (dict_E_poa_calc_pv) and
                    #for broadband irradiance (dict_E_poa_calc_pyr)
                    df_h2o.loc[year_flag,('ratio_Etotpoa',substat)] = dict_E_poa_calc_pv['Etotpoa_pv']/\
                                                    dict_E_poa_calc_pyr['Etotpoa_pv']
                                                    
                    df_h2o.loc[year_flag,('ratio_Etotdown',substat)] = (Edirdown_pv + Ediffdown_pv)/\
                            (Edirdown_pyr + Ediffdown_pyr)
                                                    
                    df_h2o.loc[year_flag,("diffuse_ratio_Etotdown_pyr",substat)] = \
                        Ediffdown_pyr/(Edirdown_pyr + Ediffdown_pyr)
                                                    
                    df_h2o.loc[year_flag,('Etotpoa_pyr_Wm2',substat)] = dict_E_poa_calc_pyr['Etotpoa']
                        
                    df_h2o.loc[year_flag,('cos_IA',substat)] = dict_E_poa_calc_pv["cos_poa_dir"]
                    df_h2o.loc[year_flag,('cos_sza','sun')] = np.cos(np.deg2rad(df_h2o.loc[year_flag,("sza","sun")]))
                    # cos_incident_angle(np.deg2rad(df_h2o_year.sza),
                    #                                  azi_shift(np.deg2rad(df_h2o_year.phi0)), 
                    #                                  irrad_pars[0],azi_shift(irrad_pars[1]))
                    # #
                                        
                df_fit = df_h2o.loc[year_flag,[('Etotpoa_pyr_Wm2',substat),('ratio_Etotpoa',substat),('ratio_Etotdown',substat),
                                       ("diffuse_ratio_Etotdown_pyr",substat),("n_h2o_mm","cosmo"),("cos_sza","sun"),
                                       ("cos_IA",substat),("sza","sun"),("phi0","sun"),("AOD_500","Aeronet")]]
                #Calculate azimuth difference for later fitting
                df_fit["diff_phi"] = np.fmod(df_fit.phi0 + 180,360) - np.rad2deg(azi_shift(irrad_pars[1]))                
                df_fit.columns = df_fit.columns.droplevel(level='substat') 
                df_fit = df_fit[df_fit['cos_IA'] != 0]
                                
                if pvrad_config["spectral_mismatch_all_sky"]["model"] == "cos_IA":
                    
                    df_fit.sort_values(by="n_h2o_mm",axis=0,inplace=True)                    
                
                    days = pd.to_datetime(df_fit.index.date).unique().strftime('%Y-%m-%d')
        
                    #This is version 1
                    dfs_am = []
                    dfs_pm = []
                    for day in days:    
                        df_day = df_fit.loc[day]
                        t_zenith = df_day.iloc[df_day.sza.argmin()].name
                        dfs_am.append(df_day.loc[df_day.index <= t_zenith])
                        dfs_pm.append(df_day.loc[df_day.index > t_zenith])
                        
                    df_am = pd.concat(dfs_am,axis=0)
                    df_pm = pd.concat(dfs_pm,axis=0)
                                    
                    for df,time in [(df_am,"AM"),(df_pm,"PM")]:
        
                        #df = df.loc[(df.n_h2o_mm < 15) & (df.n_h2o_mm > 10)]
                        xdata = df.n_h2o_mm.values
                        #ydata = df_deltas.values
                        cos_IA_data = df.cos_IA.values #np.cos(np.deg2rad(df_fit.sza)) #
                        ydata = np.log(df.ratio_Etotpoa.values) #/(cos_sza_data[:,np.newaxis])
                        
                        if time == "PM":
                            fitfunc = lambda p, w, c: p[0] + (p[1]*np.log(w/c) + p[2]*np.log(w/c**2) + p[3]*np.log(w/c**3)) # - p[2] * w**2 / c - p[1] * w**3 / c
                        elif time == "AM":
                            fitfunc = lambda p, w, c: p[0] + p[1]*w/c + p[2]*w/c**2 + p[3]*w/c**3
                    
                        errfunc = lambda p, w, c, y: y - fitfunc(p, w, c)
                        pinit = [1.,1.,1.,1.]
                
                        params = np.zeros(len(pinit)) #(len(dffit),
                    
                        # #perform fit for each time step                    
                        params = optimize.leastsq(errfunc,pinit,args=(xdata,cos_IA_data,
                                                                            ydata))[0]                                                            
                        
                        df_fit.loc[df.index,"ratio_Etotpoa_fit"] = np.exp(fitfunc(params,xdata,cos_IA_data))
                        
                        substat_pars["pars_spec_fit"][f"{year}"].update({f"{time}":params})
                    
                elif pvrad_config["spectral_mismatch_all_sky"]["model"] == "interpolated":
                    #This is version 2.
                    dfs = df_fit.groupby(df_fit.index.time)
                    dfs_times = [group for group in dfs]
                    df_mean = dfs.mean()
                    
                    fit_params = []
                    for i, (time, df) in enumerate(dfs_times):                                                            
                        fitfunc = lambda x, w, a: (x[0] + x[1]*w + x[2]*a + x[3]*a**2) # + x[3]*a**2) # + x[3]*w**3 #dfs_times[i][2] + 
                        errfunc = lambda x, w, a, y: y - fitfunc(x, w, a)
                        xinit = [1.,1.,1.,1.]
                        
                        params = np.zeros(len(xinit)) #(len(df_am),,        
                        fit = optimize.least_squares(errfunc,xinit,args=(df["n_h2o_mm"],df["AOD_500"],
                                              np.log(df.ratio_Etotpoa)))
                        params = fit["x"]                    
                        fit_params.append(params)                    
            
                        #print(params)
                        df_fit.loc[df.index,"ratio_Etotpoa_fit_new_combo"] = np.exp(fitfunc(params,df["n_h2o_mm"],df["AOD_500"]))
                    
                    df_mean["fit_params_combo"] = pd.Series(fit_params,index=df_mean.index)                                  
                    
                    # df_final = pd.concat(dfs,axis=0)
                    df_fit.sort_index(axis=0,inplace=True)
                                            
                    substat_pars[f"df_spectral_fit_{year}"] = df_fit #final                
                    substat_pars[f"df_mean_spectral_fit_{year}"] = df_mean #final                
                 
    #pv_station["spectral_fit_function"] = fitfunc                    
        
    return pv_station

def check_uncertainty(error):
    """
    Check if uncertainty is zero, if so return zero. If not then return 1
    
    args:
    :param error: uncertainty of parameters
    
    out:
    :return: 0 or 1
    """
    
    if error == 0:
        return 0.
    else:
        return 1.
    
def solve_quadratic_eqn(a,b,c):
    """
    Solve the quadratic formula with the usual method, take the positive solution
    
    args:
    :param a: float, coefficient of quadratic term
    :param b: float, coefficient of linear term
    :param c: float, coefficient of constant term
    
    out:
    :return x: float, solution of equation
    :return dx_da: array, derivative wrt a
    :return dx_db: array, derivative wrt b
    :return dx_dc: array, derivative wrt c
    """
    z = b**2 - 4.*a*c
    
    #z[z<0] = 0.001
    
    x = (-b + np.sqrt(z))/(2.*a)
    
    dx_da = (-4.*a*c/np.sqrt(z) - 2.*np.sqrt(z))/(4.*a**2)
    
    dx_db = 1./(2*a)*(-1. + b/np.sqrt(z))
    
    dx_dc = -1./np.sqrt(z)
    
    return x, dx_da, dx_db, dx_dc

def linear_combination_fn(b1,b2,b3,b4,b5,P,chi_spec,Tamb,v,Tsky,T_mod_dict):
    """
    Combine parameters together in the simple PVUSA model
    P = G*(b1 + b2*G + b3*Tamb + b4*vwind + b5*Tsky i.e.
    G**2(b2) + G(b1 + b3T + b4v + b5*Tsky) - P = 0
    
    args:
    :param b1: float, coefficient b1
    :param b2: float, coefficient b2
    :param b3: float, coefficient b3
    :param b4: float, coefficient b4
    :param b5: float, coefficient b5
    :param P: array of float, power values in Watt    
    :param chi_spec: array, spectral mismatch factor
    :param Tamb: array of float, temperature values in C
    :param v: array of float, windspeed values in m/s
    :param Tsky: array of float, sky temperature values in C
    :param T_mod_dict: dictionary with temperature model configuration
    
    
    out:
    :return a: float, coefficient of quadratic term
    :return b: float, coefficient of linear term
    :return c: float, coefficient of constant term
    :return diffs_a: list of derivatives wrt a
    :return diffs_b: list of derivatives wrt b
    :return diffs_c: list of derivatives wrt c
    """
    
    if T_mod_dict:
        a = b2*dynamic_smoothing(chi_spec,T_mod_dict)
        da_db2 = dynamic_smoothing(chi_spec,T_mod_dict)
    else:
        a = b2*chi_spec
        da_db2 = chi_spec
        
    b = (b1 + b3*Tamb + b4*v + b5*Tsky)*chi_spec
    c = - P    
    
    #Derivative of 
    diffs_a = [0.,da_db2,0.,0.,0.,0.,0.,0.,0.]
    
    db_db1 = 1.*chi_spec
    db_db3 = Tamb*chi_spec
    db_db4 = v*chi_spec
    db_db5 = Tsky*chi_spec
    db_dTamb = b3*chi_spec
    db_dv = b4*chi_spec
    db_dTsky = b5*chi_spec
    
    diffs_b = [db_db1,0.,db_db3,db_db4,db_db5,0.,db_dTamb,db_dv,db_dTsky]
        
    dc_dP = -1.
    
    diffs_c = [0.,0.,0.,0.,0.,dc_dP,0.,0.,0.]
    
    return (a,b,c), diffs_a, diffs_b, diffs_c

def faiman_combination_fn(b1,b2,b3,b4,b5,P,chi_spec,Tamb,v,Tsky):
    """
    Combine parameters together in the Evans + Faiman
    P = G*(b1 + b3*T + G/(b2 + b4*vwind)) + b5
    
    args:
    :param b1: float, coefficient b1
    :param b2: float, coefficient b2
    :param b3: float, coefficient b3
    :param b4: float, coefficient b4
    :param b5: float, coefficient b5
    :param P: array of float, power values in Watt    
    :param chi_spec: array, spectral mismatch factor
    :param Tamb: array of float, temperature values in C
    :param v: array of float, windspeed values in m/s
    :param Tsky: array of float, sky temperature values in C
    
    out:
    :return a: float, coefficient of quadratic term
    :return b: float, coefficient of linear term
    :return c: float, coefficient of constant term
    :return diffs_a: list of derivatives wrt a
    :return diffs_b: list of derivatives wrt b
    :return diffs_c: list of derivatives wrt c
    """
    
    a = chi_spec/(b2 + b4*v)
    b = (b1 + b3*Tamb + b5*Tsky)*chi_spec
    c = - P
    
    da_db2 = -chi_spec/((b2 + b4*v))**2    
    da_db4 = -v*chi_spec/((b2 + b4*v))**2    
    da_dv = -b4*chi_spec/((b2 + b4*v))**2
    
    diffs_a = [0.,da_db2,0.,da_db4,0.,0.,0.,da_dv,0.]
    
    db_db1 = 1.*chi_spec    
    db_db3 = Tamb*chi_spec   
    db_db5 = Tsky*chi_spec
    db_dTamb = b3*chi_spec  
    db_dTsky = b5*chi_spec
    
    diffs_b = [db_db1,0.,db_db3,0.,db_db5,0.,db_dTamb,0.,db_dTsky]
        
    dc_dP = -1.
    
    diffs_c = [0.,0.,0.,0.,0.,dc_dP,0.,0.,0.]
    
    return (a,b,c), diffs_a, diffs_b, diffs_c

def king_combination_fn(b1,b2,b3,b4,b5,P,T,v):
    """
    Combine parameters together in the Evans + King model
    P = G*(b1 + b3*T + G*b2 + b5)
    In this case b2 is a function of vwind
    
    args:
    :param b1: float, coefficient b1
    :param b2: float, coefficient b2
    :param b3: float, coefficient b3
    :param b4: float, coefficient b4
    :param b5: float, coefficient b5
    :param P: array of float, power values in Watt    
    :param T: array of float, temperature values in C    
    :param v: array of float, windspeed values in m/s
    
    out:
    :return a: float, coefficient of quadratic term
    :return b: float, coefficient of linear term
    :return c: float, coefficient of constant term
    :return diffs_a: list of derivatives wrt a
    :return diffs_b: list of derivatives wrt b
    :return diffs_c: list of derivatives wrt c
    """
    
    a = b2
    b = b1 + b3*T
    c = b5 - P
    
    da_db2 = 1.
    
    diffs_a = [0.,da_db2,0.,0.,0.,0.,0.,0.]
    
    db_db1 = 1.
    db_db3 = T
    db_db4 = 0.
    db_dT = b3
    db_dv = 0.
    
    diffs_b = [db_db1,0.,db_db3,db_db4,0.,0.,db_dT,db_dv]
    
    dc_db5 = 1.
    dc_dP = -1.
    
    diffs_c = [0.,0.,0.,0.,dc_db5,dc_dP,0.,0.]
    
    return (a,b,c), diffs_a, diffs_b, diffs_c
    
def mul_lists(l1,l2):    
    """
    Multiply two lists

    Parameters
    ----------
    l1 : list 1
    l2 : list 2

    Returns
    -------
    output the product of two lists

    """
    return sum([l1[i]*l2[i] for i in range(len(l1))])
    
def prepare_inversion_data(mkname,dataframe,data_source,error_days,substat_inv,T_model):
    """
    Prepare dataframe for inversion depending on source config
    Check for NaNs
    Create inversion dataframe with
    PV power, Irradiance, Temperature (Ambient and Module), Wind
    
    args:    
    :param mkname: name of measurement campaign
    :param dataframe: dataframe for inversion "df_sim_"
    :param data_source: dictionary with config for data sources
    :param error_days: list of days to leave out due to errors
    :param substat_inv: substation to use for inversion
    :param T_model: string, temperature model to use for inversion
    
    out:
    :return: index with all values that are valid (not NAN)
    """
            
    #Find all valid power values
    notnan_index_P = dataframe.loc[dataframe[('P_meas_W',substat_inv)].notna()].index    
    
    error_days_index = pd.DatetimeIndex(error_days)
    notnan_index_P = notnan_index_P[~np.isin(pd.to_datetime(notnan_index_P.date),
                         error_days_index)]
    
    if error_days:
        for day in error_days:
            print(f'Removing data from {day}')
    
    notnan_index_chi = dataframe.loc[dataframe[("chi_spec_fit",substat_inv)].notna()].index
    
    notnan_index = notnan_index_P.intersection(notnan_index_chi)
    
    
    if T_model == "Dynamic_or_Measured":  
        if "model" in data_source['temp_module'][mkname]:
            T_substat_name = f"{substat_inv}_{data_source['temp_module'][mkname].split('_')[0]}"
        else:
            T_substat_name = data_source['temp_module'][mkname]
        notnan_index_T = dataframe.loc[dataframe[('T_module_C',T_substat_name)].notna()].index
                
        notnan_index = notnan_index.intersection(notnan_index_T)

    else:    
        #Check for missing temperature values
        if data_source["temp_amb"][year] == "cosmo":
            notnan_index_T = dataframe.loc[dataframe[("T_ambient_2M_C",
                                        "cosmo")].notna()].index  
        elif "Pyr" in data_source["temp_amb"][year]:
            notnan_index_T = dataframe.loc[dataframe[("T_ambient_pyr_C",
                               data_source["temp_amb"][year])].notna()].index
        elif "Windmast" in data_source["temp_amb"]:
            notnan_index_T = dataframe.loc[dataframe[("T_ambient_C",
                               data_source["temp_amb"][year])].notna()].index
        
        notnan_index = notnan_index.intersection(notnan_index_T)
        
        #Check for missing wind values
        if data_source["wind"][year] == "cosmo":
            notnan_index_v = dataframe.loc[dataframe[("v_wind_10M",
                                        "cosmo")].notna()].index
        elif data_source["wind"][year] == "Windmast":
            notnan_index_v = dataframe.loc[dataframe[("v_wind_mast_ms",
                                        data_source["wind"][year])].notna()].index
    
        notnan_index = notnan_index.intersection(notnan_index_v)
        
        notnan_index_T = dataframe.loc[dataframe[("T_sky_C",
                                        data_source['longwave'][mkname][1])].notna()].index
        #Final index of valid values
        notnan_index = notnan_index.intersection(notnan_index_T)
    
    return notnan_index

def load_lw_data(station,timeres,description,path):
    """
    Load the LW downward welling irradiance measured at MS01

    Parameters
    ----------
    station : list with station information
    timeres : sting with time resolution        
    description : string with description of current campaign        
    path : string with path where data is saved

    Returns
    -------
    lw_pv_stat: dataframe with longwave data

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
    
def prepare_longwave_data(key,pv_station,dfname,measurement,lw_station,timeres_data,emissivity):
    """
    
    Prepare longwave data and calculate sky temperature

    Parameters
    ----------
    key : string, name of station
    pv_station : dictionary with information and data from PV station
    dfname : string with name of relevant dataframe
    measurement : string giving the year of measurement campaign        
    lw_station : list with information about station where LW measurement was taken
    timeres_data : string with time resolution
    emissivity : float, emissivity of the atmosphere

    Returns
    -------
    dataframe with calculated sky temperature

    """
    
       
    year = "mk_" + measurement.split('_')[1]        
                    
    data_config = load_yaml_configfile(config["data_configfile"][year])

    #Get configuration        
    loadpath_lw = os.path.join(homepath,data_config["paths"]["savedata"]["main"],
                            data_config["paths"]["savedata"]["binary"])                                                    

    #Load data for temperature model
    df_lw = load_lw_data(lw_station,pv_station["t_res_lw"],measurement,loadpath_lw)
        
    df_lw[("error_G_lw",lw_station[1])] = df_lw[(lw_station[2],lw_station[1])]*\
           pvrad_config["meas_errors"]["g_lw"][0]
    df_lw[("T_sky_C",lw_station[1])], df_lw[("error_T_sky_C",lw_station[1])] \
        = calculate_sky_temperature(df_lw[(lw_station[2],lw_station[1])],
          df_lw[("error_G_lw",lw_station[1])],emissivity)
    
    tres_lw = pd.Timedelta(pv_station["t_res_lw"])
    tres_data = pd.Timedelta(timeres_data)
    
    if tres_lw < tres_data:
        lw_days = pd.to_datetime(df_lw.index.date).unique().strftime('%Y-%m-%d')    
        dfs_rs = []
        for day in lw_days:
            df_lw_day = df_lw.loc[day]
            try:
                df_rs_day = dpf.downsample(df_lw_day,tres_lw, tres_data)
                dfs_rs.append(df_rs_day)
            except:
                print(f'error in resampling on {day}')                    
        df_rs = pd.concat(dfs_rs,axis=0)
    else:
        df_rs = df_lw
        
    #combined_index = pv_station[f"df_sim_{year.split('_')[1]}"].index.intersection(df_rs.index)
    pv_station[dfname] = pd.concat([pv_station[dfname],df_rs],axis=1,join='inner') 
    
    return pv_station[dfname]   

def kontrolle_zeitaufloesung_hilfsfunktion(ein_merge, ein_merge_day_index_diff, \
                                           time_resolution, iday):
    """
    Hilfsfunktion zur Funktion kontrolle_zeitaufloesung
    """
    ein_merge_day_index_diff_min = ein_merge_day_index_diff.min()
    if ein_merge_day_index_diff_min != time_resolution:
        print("Die zeitliche Aufloesung an Tag %s bei der Groesse %s stimmt \
              nicht mit der allgemeinen Zeitaufloesung ueberein" % (iday, ein_merge.name))
    elif ein_merge_day_index_diff.max() != ein_merge_day_index_diff_min:
        print("Die minimale und die maximale zeitliche Aufloesung an Tag %s bei \
              der Groesse %s stimmt nicht ueberein" % (iday, ein_merge.name))
            

def kontrolle_zeitaufloesung(dataframe, time_res_new, test_days):
    """
    kontrolliert, ob die Daten alle dieselbe Zeitauflösung haben
    """
    
    for iday in test_days:
        for col in dataframe.columns:
            day_index_diff = dataframe[col].loc[iday].index.to_series().diff()
            time_resolution = day_index_diff.min()
            kontrolle_zeitaufloesung_hilfsfunktion(dataframe[col],
                                                   day_index_diff, \
                                                   time_resolution, iday)
    return

def calculate_sky_temperature(G_lw,error_G_lw,emissivity):
    """
    
    Calculate sky temperature using measured longwave irradiance and emissivity

    Parameters
    ----------
    G_lw : array, longwave irradiance
    error_G_lw : array, error in longwave irradiance        
    emissivity : float, emissivity of the atmosphere        

    Returns
    -------
    T_sky_C : array of floats, sky temperature in C
    error_Tsky_C : array of floats, error in sky temperature in C

    """    
    
    T_sky_C = np.power(G_lw/const.sigma/emissivity,1./4) - 273.15
    
    error_Tsky_C = 0.25*np.power(G_lw/const.sigma/emissivity,-3./4)/const.sigma/emissivity*error_G_lw
    
    return T_sky_C, error_Tsky_C

def calculate_temp_module(station,pv_station,substat,pvcal_config,pvrad_config,data_config,
                          timeres,measurement,home):
    """
    Calculate module temperature with dynamic model, using atmospheric conditions
    
    args:
    :param station: string, name of PV station    
    :param pv_station: dictionary with information and data from PV station
    :param substat: string with name of substat
    :param pvcal_config: dictionary with configuration for calibration
    :param pvrad_config: dictionary with configuration for inversion
    :param data_config: dictionary with configuration for loading data        
    :param timeres: string with resolution of substation data
    :param measurement: string defining which measurement campaign to use
    :param home: string, homepath
    
    out:
    :return df_temp_final: dataframe with final temperature
    
    """
    
    #Get configuration
    mount_type = pvcal_config["pv_stations"][station]["substat"][substat]["mount"]                                
    
    pvtemp_config= pv_station[f"pvtemp_config_{mount_type}"]
    year = "mk_" + measurement.split('_')[1]                
        
    source = pvcal_config["pv_stations"][station]["input_data"]
    T_source = source["temp_module"][year]
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
    
    temp_model = pv_station[f"temp_model_{mount_type}"][T_model]
    
    df_temp = pv_station[f"df_{year.split('_')[1]}_{timeres}"]
    #Put together data for temperature model
    dataframe = pd.DataFrame()      
            
    if station == "PV_11" or (station == "PV_12" and year == "mk_2019"):
        dataframe["Gtotpoa"] = df_temp[('Etotpoa_RT1_Wm2',
             source["irrad"][year])]
    # elif station == "MS_02":
    #     dataframe["Gtotpoa"] = df_temp[('Etotpoa_CMP11_Wm2',
    #          source["irrad"][year])]
    else:
        dataframe["Gtotpoa"] = df_temp[('Etotpoa_pyr_Wm2',
             source["irrad"][year])]            
    
    dataframe["T_ambient"] = df_temp[(T_amb_name,
             source["temp_amb"][year])]
    
    dataframe["v_wind"] = df_temp[(v_wind_name,
             source["wind"][year])]
    
    dataframe["G_lw_down"] = df_temp\
    .loc[:,(source["longwave"][year][2],source["longwave"][year][1])]
    
    dataframe.dropna(axis=0,inplace=True)
    
    #Get unique days
    days = pd.to_datetime(dataframe.index.date).unique().strftime('%Y-%m-%d')    
        
    time_res_temp = dataframe.index.to_series().diff().min()
    
    kontrolle_zeitaufloesung(dataframe,time_res_temp,days)
    
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
    
    #Extract just the modelled temperature and create Multiindex
    dataframe.columns = pd.MultiIndex.from_product([dataframe.columns.values.tolist(),
                             [f"{substat}_{T_model}"]],names=['variable','substat'])                    
    
    df_temp_final = dataframe[("T_module_C",f"{substat}_{T_model}")]            
    
    return df_temp_final
   
def load_temp_model_results(station,pv_station,pvcal_config,home):
    """
    Load results from dynamic temperature temperature model
    
    args:
    :param station: string, name of PV station
    :param pv_station: dictionary with info and dataframes    
    :param pvcal_config: dictionary with configuration for calibration
    :param home: string, homepath
    
    out:
    :return pv_station: dictionary with information and data for PV system
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
    
def resample_cosmo(df_cosmo,timeres_substat):
    """
    Resample COSMO data to match the resolution of the dataframe for inversion
    
    args:
    :param df_cosmo: dataframe with COSMO data in 15 minute resolution
    :param timeres_substat: string with required time resolution
    
    out:
    :return: dataframe resampled to required resolution
    """
    
    time_res_substat = pd.to_timedelta(timeres_substat)
    
    time_res_cosmo = df_cosmo.index.to_series().diff().min()
    
    if time_res_cosmo != time_res_substat:
        try:
            df_rs = dpf.interpolate(df_cosmo,time_res_substat)        
        except:
            print('error in resampling')                
    else:
        df_rs = df_cosmo
    
    return df_rs

def interpolate_fit_params(dffit,t_res,substat):
    """
    

    Parameters
    ----------
    dffit : dataframe with spectral mismatch fit        
    t_res : string with time resolution for interpolation        
    substat : string with name of substation

    Returns
    -------
    df_hires : dataframe with interpolated parameters

    """
    
    days = pd.to_datetime(dffit.index.date).unique().strftime('%Y-%m-%d')
    
    df_hires_new = []
    for day in days:
        #Low res data
        df_fit_day = dffit.loc[day]        
        old_index_numeric = df_fit_day.index.values.astype(float)
                
        #High res index
        new_index = pd.date_range(start=df_fit_day.index[0],end=df_fit_day.index[-1],freq=t_res)
        new_index_numeric = new_index.values.astype(float)
        
        fit_pars = np.stack(df_fit_day["fit_params_combo"].to_numpy())
        f = interp1d(old_index_numeric,fit_pars, kind='linear', axis=0)
        fit_pars_new = f(new_index_numeric)
        
        slice_data = [fit_pars_new[j,:] for j in range(fit_pars_new.shape[0])]
        df_hires_fit = pd.Series(data=slice_data,index=new_index,name=("fit_params_combo",substat))
        
        #Interpolate AERONET
        if "AOD_500" in df_fit_day.columns:
            #Interpolate AERONET
            aod_old = np.stack(df_fit_day["AOD_500"].to_numpy())
            f = interp1d(old_index_numeric,aod_old, kind='linear',axis=0)
            aod_new = f(new_index_numeric)
                    
            df_hires_AOD = pd.Series(index=new_index,data=aod_new,name=("AOD_500","Aeronet"))
            
            df_hires_fit = pd.concat([df_hires_fit,df_hires_AOD],axis=1)
        
        df_hires_new.append(df_hires_fit)
                
    df_hires = pd.concat(df_hires_new,axis=0)
    
    return df_hires

def apply_spectral_mismatch_cos_IA(params,dataframe,opt_pars):
    """
    Apply spectral mismatch based on incident angle

    Parameters
    ----------
    params : list of fit parameters
    dataframe : dataframe with input parameters
    opt_pars : parameters of optical model

    Returns
    -------
    df_ratio.values : array of spectral mismatch values

    """
    
    sza = np.deg2rad(dataframe[("sza","sun")])
    phi0 = np.deg2rad(dataframe[("phi0","sun")])
    cos_IA = cos_incident_angle(sza,phi0,opt_pars[0][1],opt_pars[1][1])
    
    days = pd.to_datetime(dataframe.index.date).unique().strftime('%Y-%m-%d')

    #This is version 1.
    dfs_am = []
    dfs_pm = []
    for day in days:    
        df_day = dataframe.loc[day]
        t_zenith = df_day.iloc[df_day["sza","sun"].argmin()].name
        dfs_am.append(df_day.loc[df_day.index <= t_zenith])
        dfs_pm.append(df_day.loc[df_day.index > t_zenith])
        
    df_am = pd.concat(dfs_am,axis=0)
    df_pm = pd.concat(dfs_pm,axis=0)
    
    dfs = []
    for df,time in [(df_am,"AM"),(df_pm,"PM")]:
        if time == "PM":
            fitfunc = lambda p, w, c: p[0] + (p[1]*np.log(w/c) + p[2]*np.log(w/c**2) + p[3]*np.log(w/c**3)) # - p[2] * w**2 / c - p[1] * w**3 / c
        elif time == "AM":
            fitfunc = lambda p, w, c: p[0] + p[1]*w/c + p[2]*w/c**2 + p[3]*w/c**3
        
        dfs.append(np.exp(fitfunc(params[time],df[("n_h2o_mm","cosmo")],cos_IA.loc[df.index])))    
    
    df_ratio = pd.concat(dfs,axis=0)
    df_ratio.sort_index(axis=0,inplace=True)       
    
    return df_ratio.values

def apply_spectral_mismatch_interpolated(df_fit,df_mean,dataframe,t_res,substat): 
    """
    
    Apply spectral mismatch using interpolated fit function
    
    Parameters
    ----------
    df_fit : dataframe with interpolated fit function
    df_mean : dataframe with mean over time of fit
    dataframe : dataframe with original input data    
    t_res : string, time resolution of data
    substat : string, name of substation

    Returns
    -------
    df_ratio : series with spectral mismatch factor
    """
    
    
    #This is version 2.
    #First translate mean fit parameters into dataframe, then interpolate
    fit_params_final = [np.ones(3)*np.nan]*len(df_fit)
    df_fit["fit_params_combo"] = pd.Series(data=fit_params_final,index=df_fit.index)
    for time in df_mean.index:
        time_index = df_fit.loc[df_fit.index.time == time].index
        for t in time_index:
            df_fit.at[t,"fit_params_combo"] = \
            df_mean.loc[time,"fit_params_combo"]
    
    dataframe = pd.merge(dataframe,interpolate_fit_params(df_fit,t_res,substat),
                         left_index=True,right_index=True)
    
    #Use the combined fit function
    fitfunc = lambda x, w, a: (x[0] + x[1]*w + x[2]*a + x[3]*a**2) # + x[3]*a**2)
    
    print(f'Applying fit function to all data from {year}')
    #for time in dataframe.dropna(subset=[("fit_params_combo",substat)],axis=0).index:
    fit_pars_array = np.transpose(np.stack(dataframe[("fit_params_combo",substat)].values))
    dataframe[("chi_spec_fit",substat)] = np.exp(fitfunc(
        fit_pars_array,dataframe[("n_h2o_mm","cosmo")],
           dataframe[("AOD_500","Aeronet")]))
    
    df_ratio = dataframe[("chi_spec_fit",substat)] #.rename(("chi_spec_fit",substat))

    return df_ratio

def dynamic_smoothing(input_vector,T_mod_dict):
    """
    

    Parameters
    ----------
    input_vector : input data series
    T_mod_dict : dictionary with temperature model parameters

    Returns
    -------
    output : smoothed data series

    """
    
    if type(T_mod_dict["n_past"]) == int:
        n_past = T_mod_dict["n_past"]
    elif type(T_mod_dict["n_past"]) == np.ndarray:
        n_past = T_mod_dict["n_past"][0]
    else:
        print("Fehler. n_regul ist weder integer noch numpy.ndarray")
    
    # Vorbereitungen für die Option alle_messpunkte
    if T_mod_dict["flag"]:
        n_past_max = T_mod_dict["day_length"].max() - 1
        n_past = n_past_max
        
    #Calculate the weighting factors for smoothing
    f, n_past = gewichtsfaktoren_unnormiert(n_past, T_mod_dict["time_res_temp_sec"], 
                                             T_mod_dict["tau"])
    g_norm = np.cumsum(np.flip(f))
    
    output = matrix_smoothing_function(input_vector, n_past, T_mod_dict["day_length"],
                                       f, g_norm, T_mod_dict["flag"])
    
    return output
    
    
    
def temp_model_input(dataframe,year,data_source,pvrad_config,T_model_type,T_mod_dict):
    """
    

    Parameters
    ----------
    dataframe : dataframe with input data
    year : string with year under consideration
    data_source : dictionary with configuration for data source
    pvrad_config : dictionary with inversion configuration
    T_model_type : string, temperature model type (static or dynamic)
    T_mod_dict : dictionary with temperature model parameters

    Returns
    -------
    TambC : array of ambient temperature in C
    err_Tamb : array of error in ambient temperature in C
    vwind : array of windspeed in m/s
    err_vwind : array of error in windspeed in m/s
    TskyC : array of sky temperature in C
    err_Tsky : array of error in sky temperature in C

    """
    
    
    if data_source["temp_amb"][year] == "cosmo":
        TambC = dataframe.T_ambient_2M_C.values.flatten()        
    elif "Pyr" in data_source["temp_amb"][year]:
        TambC = dataframe[("T_ambient_pyr_C",data_source["temp_amb"][year])].values.flatten()        

    if data_source["wind"][year] == "cosmo":
        vwind = dataframe.v_wind_10M.values.flatten()       
    elif data_source["wind"][year] == "Windmast":
        vwind = dataframe[("v_wind_mast_ms",data_source["wind"][year])].values.flatten()                 
        
    TskyC = dataframe[("T_sky_C",data_source["longwave"][year][1])].values.flatten()         
    
    if T_model_type == "dynamic":     
        #Get unique days
        days = pd.to_datetime(dataframe.index.date).unique().strftime('%Y-%m-%d')    
            
        time_res_temp = dataframe.index.to_series().diff().min()        
        
        number_days = len(days)
        
        time_res_temp_sec = time_res_temp/np.timedelta64(1, 's')
        day_length = np.zeros(number_days, dtype=int)
        for i, iday in enumerate(days):
            day_length[i] = len(dataframe.loc[iday]) 
            
        T_mod_dict.update({"day_length":day_length})
        T_mod_dict.update({"number_days":number_days})
        T_mod_dict.update({"time_res_temp_sec":time_res_temp_sec})
        print('Applying dynamic smoothing')
        TambC  = dynamic_smoothing(TambC,T_mod_dict)
        vwind  = dynamic_smoothing(vwind,T_mod_dict)
        TskyC  = dynamic_smoothing(TskyC,T_mod_dict)
        
    if data_source["temp_amb"][year] == "cosmo":      
        err_Tamb = np.ones(len(dataframe))*pvrad_config["cosmo_errors"]["temp"][0]
    elif "Pyr" in data_source["temp_amb"][year]:        
        err_Tamb = np.ones(len(dataframe))*pvrad_config["meas_errors"]["t_amb_pyr"][0]    

    if data_source["wind"][year] == "cosmo":
        err_vwind = vwind*pvrad_config["cosmo_errors"]["wind"][0]
    elif data_source["wind"][year] == "Windmast":        
        err_vwind = np.ones(len(dataframe))*pvrad_config["meas_errors"]["v_wind"][0]
            
    err_Tsky = dataframe[("error_T_sky_C",data_source["longwave"][year][1])].values.flatten()    
    
    return TambC, err_Tamb, vwind, err_vwind, TskyC, err_Tsky
    
    
def etotpoa_power_model(station,pv_station,dfname,substat_inv,year,pvcal_config,
                                  pvrad_config,substat_type,timeres):
    """
    Invert the calibrated PV power model in order to calculate the irradiance in 
    the plane of the array for all-sky conditions
    
    args:    
    :param station: string, name of station
    :param pv_station: dictionary with dataframe of PV data etc
    :param dfname: string, name of dataframe
    :param substat_inv: string, substation to invert
    :param year: string, name of campaign (mk_2018 or mk_2019)
    :param pvcal_config: dictionary with configuration for calibration    
    :param pvrad_config: dictionary with configuration for inversion
    :param substat_type: string describing type of PV substation (data)
    :param timeres: string, time resolution of data
    
    out:
    :return: return dataframe with Etotpoa calculation included
    """
    
    T_model =  pvcal_config["T_model"]["model"]   
    T_model_type = pvcal_config["T_model"]["type"]
    eff_model = pvcal_config["eff_model"]
    p_err_rel = substat_pars["p_err_rel"]    
    p_err_min = substat_pars["p_err_min"]    
    
    #Get data source from config file
    data_source = pvcal_config["pv_stations"][station]["input_data"]
    #cal_state = pv_station['substations'][substat_inv]['opt']['x_min']
    
    if 'opt_pars' in pvrad_config["pv_stations"][station][substat_type]\
                ["data"][substat_inv]:
        opt_pars = pvrad_config["pv_stations"][station][substat_type]\
                ["data"][substat_inv]['opt_pars']        
        err_state = pvrad_config["pv_stations"][station][substat_type]\
                ["data"][substat_inv]['opt']['S_hat']
    
    else:
        opt_pars = pvrad_config["pv_stations"][station][substat_type]\
                ["data"][substat_inv]['ap_pars']        
        err_state = pvrad_config["pv_stations"][station][substat_type]\
                ["data"][substat_inv]['opt']['S_a']
    
        print('No solution found, using a-priori values for inversion')
    
    #First two parameters are always angles, third is n
    #x_0 = theta, x_1 = phi, x_2 = n
    #Next two are scaling factor and temperature coefficient
    #x_3 = s, x_4 = zeta
    #Then come efficiency parameters followed by temperature parameters
    
    #check if theta, phi or n are varied or not
    #if some are varied then they are in the error state
    start_index = 0 #In this case all three would be fixed
    for par in opt_pars[0:3]:
        if par[0] == 'theta' or par[0] == 'phi' or par[0] == 'n':
            if par[2] != 0:
                start_index = start_index + 1
            
    errors = err_state[start_index:,start_index:]
    
    #Extract s and zeta
    s = opt_pars[3][1]        
    Ds = check_uncertainty(opt_pars[3][2])
    
    zeta = opt_pars[4][1]            
    Dzeta = check_uncertainty(opt_pars[4][2])
                       
    #Check whether parameters are fixed or not
    if eff_model == 'Beyer':                
        a1 = opt_pars[5][1]
        Da1 = check_uncertainty(opt_pars[5][2])
        a2 = opt_pars[6][1]
        Da2 = check_uncertainty(opt_pars[6][2])
        a3 = opt_pars[7][1]
        Da3 = check_uncertainty(opt_pars[7][2])
        index = 8
    elif eff_model == "Ransome":
        c3 = opt_pars[5][1]
        Dc3 = check_uncertainty(opt_pars[5][2])
        c6 = opt_pars[6][1]
        Dc6 = check_uncertainty(opt_pars[6][2])
        index = 7
    else:
        index = 5
        
    if T_model == "Tamizhmani":
        #, x_5 = u_0, x_6 = u_1, x_7 = u_2, x_8 = u_3                            
        u0 = opt_pars[index][1]
        Du0 = check_uncertainty(opt_pars[index][2])
        u1 = opt_pars[index + 1][1] 
        Du1 = check_uncertainty(opt_pars[index + 1][2])
        u2 = opt_pars[index + 2][1]
        Du2 = check_uncertainty(opt_pars[index + 2][2])
        u3 = opt_pars[index + 3][1]
        Du3 = check_uncertainty(opt_pars[index + 3][2])                                        
    elif T_model == "Faiman" or T_model == "Barry":
        u1 = opt_pars[index][1]
        Du1 = check_uncertainty(opt_pars[index][2])
        u2 = opt_pars[index + 1][1]
        Du2 = check_uncertainty(opt_pars[index + 1][2])
        u3 = opt_pars[index + 2][1]
        Du3 = check_uncertainty(opt_pars[index + 2][2])
    elif T_model == "King":
        a0 = opt_pars[index][1]
        Da0 = check_uncertainty(opt_pars[index][2])
        b0 = opt_pars[index + 1][1]
        Db0 = check_uncertainty(opt_pars[index + 1][2])
        dT = opt_pars[index + 2][1]
        DdT = check_uncertainty(opt_pars[index + 2][2])
        
    #Split parameters into bi and corresponding uncertainties
    if eff_model == 'Beyer':        
#        eff_mpp = (a1 + a2*irrad['Etotpoa_pv'] + a3*np.log(irrad['Etotpoa_pv']))
#        eff_temp_corr = eff_mpp*(1.0 - zeta*(temp_model['T_mod'] - 25.0))            
        if T_model == "Tamizhmani":
            b1 = s*(1 - zeta*(u3 - 25.))
            b2 = -s*zeta*u1
            b3 = -s*zeta*u0
            b4 = -s*zeta*u2    
            b5 = 0.
    elif eff_model == "Ransome":        
#        eff_temp_corr = 1.0 - zeta*(temp_model['T_mod'] - 25.0)\
#            + c3*np.log(irrad['Etotpoa_pv']) + c6/irrad['Etotpoa_pv']        
        if T_model == "Tamizhmani":
            b1 = s*(1 - zeta*(u3 - 25.))
            b2 = -s*zeta*u1
            b3 = -s*zeta*u0
            b4 = -s*zeta*u2
            b5 = s*c6        
    #This is the Evans case
    else:
        #combine parameters into simplified equation 
        #P = Etotpoa(b1 + b2*Etotpoa + b3*Tamb + b4*vwind) + b5
        if T_model == "Tamizhmani":            
            #Calculate parameters            
            b1 = s*(1. + zeta*25.)
            b2 = -s*zeta*u1
            b3 = -s*zeta*u0
            b4 = -s*zeta*u2
            b5 = -s*zeta*u3
            
            #Calculate derivatives for uncertainties in quadrature
            #derivatives wrt s
            db1_ds = 1. + zeta*25.
            db2_ds = -u1*zeta
            db3_ds = -u0*zeta
            db4_ds = -u2*zeta
            db5_ds = -u3*zeta
            
            diffs_s = [db1_ds,db2_ds,db3_ds,db4_ds,db5_ds]
            
            #derivatives wrt zeta
            db1_dzeta = s*25.
            db2_dzeta = -u1*s
            db3_dzeta = -u0*s
            db4_dzeta = -u2*s
            db5_dzeta = -u3*s
            
            diffs_zeta = [db1_dzeta,db2_dzeta,db3_dzeta,db4_dzeta,db5_dzeta]
            
            #derivatives wrt u0
            db3_du0 = -s*zeta
            
            diffs_u0 = [0.,0.,db3_du0,0.,0.]
            
            #derivatives wrt u1
            db2_du1 = -s*zeta
            
            diffs_u1 = [0.,db2_du1,0.,0.,0.]
            
            #derivatives wrt u2
            db4_du2 = -s*zeta
            
            diffs_u2 = [0.,0.,0.,db4_du2,0.]
            
            #derivatives wrt u3
            db5_du3 = -s*zeta
            
            diffs_u3 = [0.,0.,0.,0.,db5_du3]
            
        elif T_model == "Faiman" or T_model == "Barry":
            b1 = s*(1. + zeta*25.)
            b2 = -u1/(s*zeta)
            b3 = s*zeta*(u3 - 1.)
            b4 = -u2/(s*zeta)
            b5 = -s*zeta*u3
            
            #Calculate derivatives for uncertainties in quadrature
            #derivatives wrt s
            db1_ds = 1. + zeta*25
            db2_ds = u1/zeta/s**2
            db3_ds = zeta*(u3 - 1.)
            db4_ds = u2/zeta/s**2
            db5_ds = -zeta*u3
            
            diffs_s = np.array([db1_ds,db2_ds,db3_ds,db4_ds,db5_ds])
            
            #derivatives wrt zeta
            db1_dzeta = s*25.
            db2_dzeta = u1/s/zeta**2
            db3_dzeta = s*(u3 - 1.)            
            db4_dzeta = u2/s/zeta**2
            db5_dzeta = -s*u3
            
            diffs_zeta = np.array([db1_dzeta,db2_dzeta,db3_dzeta,db4_dzeta,db5_dzeta])
            
            #derivatives wrt u1
            db2_du1 = -1./(s*zeta)
            
            diffs_u1 = np.array([0.,db2_du1,0.,0.,0.])
            
            #derivatives wrt u2
            db4_du2 = -1./(s*zeta)
            
            diffs_u2 = np.array([0.,0.,0.,db4_du2,0.])
            
            #derivatives wrt u3
            db3_du3 = s*zeta
            db5_du3 = -s*zeta
            
            diffs_u3 = np.array([0.,0.,db3_du3,0.,db5_du3])        
                        
    print('Calculating POA irradiance from power model for %s' % substat_inv)
        
    pv_station[dfname][('P_meas_W',substat_inv)] = pv_station[dfname][('P_kW',substat_inv)]*1000.                 

    #Calculate spectral mismatch factor
    if pvrad_config["spectral_mismatch_all_sky"]["flag"]:  
        if pvrad_config["spectral_mismatch_all_sky"]["model"] == "cos_IA":
            pv_station[dfname][("chi_spec_fit",substat_inv)] = \
                apply_spectral_mismatch_cos_IA(pv_station["substations"]
               [substat_type]["data"][substat]["pars_spec_fit"][f"{year.split('_')[1]}"],
               pv_station[dfname], opt_pars)            
        elif pvrad_config["spectral_mismatch_all_sky"]["model"] == "interpolated":
            pv_station[dfname] = pd.concat([pv_station[dfname],apply_spectral_mismatch_interpolated(\
            pv_station["substations"][substat_type]["data"][substat][f"df_spectral_fit_{year.split('_')[1]}"],
            pv_station["substations"][substat_type]["data"][substat][f"df_mean_spectral_fit_{year.split('_')[1]}"],
            pv_station[dfname],timeres,substat_inv)],axis=1)
    else:
        pv_station[dfname][("chi_spec_fit",substat_inv)] = 1.                    

    #Apply dynamic matrix smoothing Method
    if T_model != "Dynamic_or_Measured":
        T_mod_dict = {}
        if T_model_type == "dynamic":            
            mount_type = pvcal_config["pv_stations"][station]["substat"][substat]["mount"]                                    
            pvtemp_config= pv_station[f"pvtemp_config_{mount_type}"]
            T_mod_dict.update({"tau":pv_station[f"temp_model_{mount_type}"]["dynamic"]["x_dach"][3]})
            T_mod_dict.update({"t_res":pvtemp_config["timeres"]})
            T_mod_dict.update({"n_past":pvtemp_config["inversion"]["inv_param"]["n_past"]})        
            T_mod_dict.update({"flag":pvtemp_config["inversion"]["inv_param"]["all_points"]})
    else:
        T_mod_dict = {}
    
    #Remove days with errors
    if 'error_days' in pvrad_config["pv_stations"][station][substat_type]["data"][substat_inv]:
        error_days = pvrad_config["pv_stations"][station][substat_type]["data"][substat_inv]["error_days"][year]
    else: 
        error_days = []
    
    #Get rid of NaNs in power, temperature, wind            
    notnan_index = prepare_inversion_data(year,pv_station[dfname],data_source,error_days,
                                      substat_inv,T_model) 
    
    dataframe = pv_station[dfname].loc[notnan_index]
    
    #Extract relevant quantities
    dataframe[('P_error_meas',substat_inv)] = dataframe[('P_meas_W',substat_inv)]*p_err_rel
    dataframe[('P_error_meas',substat_inv)].where(dataframe[('P_error_meas',substat_inv)] 
    > p_err_min,p_err_min,inplace=True)
            
    if T_model == "Dynamic_or_Measured": 
        if data_source["temp_module"][year] != "None":
            if "model" in data_source['temp_module'][year]:
                T_substat_name = f"{substat_inv}_{data_source['temp_module'][year].split('_')[0]}"
            else:
                T_substat_name = data_source['temp_module'][year]
            TmodC = dataframe[("T_module_C",T_substat_name)].values.flatten()
            err_Tmod = np.ones(len(dataframe))*pvrad_config["meas_errors"]["t_mod"][0]            
        else:
            print('Please specify data source for measured module temperature')
        err_vwind = 0.
        err_Tamb = 0.
        err_Tsky = 0.
    else:
        TambC, err_Tamb, vwind, err_vwind, TskyC, err_Tsky = temp_model_input(dataframe,year,data_source,
                   pvrad_config,T_model_type,T_mod_dict)                
        
        err_Tmod = 0.                
              
    Pmeas = dataframe[('P_meas_W',substat_inv)].values.flatten()
    P_error_meas = dataframe[('P_error_meas',substat_inv)].values.flatten()
    chi_spec = dataframe[("chi_spec_fit",substat_inv)].values.flatten()
                            
    #Calculate Etotpoa as solution to the quadratic equation: 
    if eff_model == "Beyer":
        test = 0.
    elif eff_model == "Ransome":
        test = 0.
    else:
        if T_model == "Dynamic_or_Measured":
            #INcluded spectral factor - extract E_totpoa_broadband
            E_solution = Pmeas/(s*(1. - zeta*(TmodC - 25.)))/chi_spec
            
            dE_ds = -Ds*Pmeas/((1 - zeta*(TmodC - 25.))*chi_spec)/s**2
            dE_dzeta = Dzeta*Pmeas/s/chi_spec/(1 - zeta*(TmodC - 25.))**2*TmodC
            dE_du0 = np.zeros(len(Pmeas))
            dE_du1 = np.zeros(len(Pmeas))
            dE_du2 = np.zeros(len(Pmeas))
            dE_du3 = np.zeros(len(Pmeas))      
            
            dE_dP = 1./(s*(1 - zeta*(TmodC - 25.)))/chi_spec
            dE_dTmod = Pmeas/s/chi_spec/(1 - zeta*(TmodC - 25.))**2*zeta
            dE_dv = np.zeros(len(Pmeas))
            dE_dTamb = np.zeros(len(Pmeas))
            dE_dTsky = np.zeros(len(Pmeas))
            
            #Combine derivatives into a matrix
            dE_vec = np.array([dE_ds,dE_dzeta,dE_du0,dE_du1,dE_du2,dE_du3])
        
        elif T_model == "Tamizhmani":
            (a,b,c), d_a, d_b, d_c = linear_combination_fn(b1,b2,b3,b4,b5,Pmeas,chi_spec,TambC,vwind,TskyC,T_mod_dict)
            
            #i.e.  b2*Etotpoa^2 + (b1 + b3*Tamb + b4*vwind)*Etotpoa - P = 0    
            
            E_solution, dE_da, dE_db, dE_dc = solve_quadratic_eqn(a,b,c)
            
            # if T_model_type == "dynamic":
            #     E_solution = dynamic_smoothing(E_solution, T_mod_dict)
            
            
            #Chain rule of all derivatives            
            dE_ds = Ds*(dE_da*mul_lists(d_a[0:5],diffs_s) + dE_db*mul_lists(d_b[0:5],diffs_s)
                        + dE_dc*mul_lists(d_c[0:5],diffs_s))
            dE_dzeta = Dzeta*(dE_da*mul_lists(d_a[0:5],diffs_zeta) +
                       dE_db*mul_lists(d_b[0:5],diffs_zeta) + dE_dc*mul_lists(d_c[0:5],diffs_zeta))
            
            dE_du0 = Du0*(dE_da*mul_lists(d_a[0:5],diffs_u0) + dE_db*mul_lists(d_b[0:5],diffs_u0)
                        + dE_dc*mul_lists(d_c[0:5],diffs_u0))
            dE_du1 = Du1*(dE_da*mul_lists(d_a[0:5],diffs_u1) + dE_db*mul_lists(d_b[0:5],diffs_u1)
                        + dE_dc*mul_lists(d_c[0:5],diffs_u1))
            dE_du2 = Du2*(dE_da*mul_lists(d_a[0:5],diffs_u2) + dE_db*mul_lists(d_b[0:5],diffs_u2)
                        + dE_dc*mul_lists(d_c[0:5],diffs_u2))
            dE_du3 = Du3*(dE_da*mul_lists(d_a[0:5],diffs_u3) + dE_db*mul_lists(d_b[0:5],diffs_u3)
                        + dE_dc*mul_lists(d_c[0:5],diffs_u3))
            
            dE_dP    = dE_da*d_a[5] + dE_db*d_b[5] + dE_dc*d_c[5]
            dE_dTamb = dE_da*d_a[6] + dE_db*d_b[6] + dE_dc*d_c[6]
            dE_dv    = dE_da*d_a[7] + dE_db*d_b[7] + dE_dc*d_c[7]
            dE_dTsky = dE_da*d_a[8] + dE_db*d_b[8] + dE_dc*d_c[8]
            dE_dTmod = np.zeros(len(Pmeas))
            
            #Combine derivatives into a matrix
            dE_vec = np.array([dE_ds,dE_dzeta,dE_du0,dE_du1,dE_du2,dE_du3])
        
        elif T_model == "Faiman" or T_model == "Barry":
            (a,b,c), d_a, d_b, d_c = faiman_combination_fn(b1,b2,b3,b4,b5,Pmeas,chi_spec,TambC,vwind,TskyC)
                            
            E_solution, dE_da, dE_db, dE_dc = solve_quadratic_eqn(a,b,c)
            
            #Chain rule of all derivatives                                
            dE_ds = Ds*(dE_da*mul_lists(d_a[0:5],diffs_s) + dE_db*mul_lists(d_b[0:5],diffs_s) \
                        + dE_dc*mul_lists(d_c[0:5],diffs_s))
            dE_dzeta = Dzeta*(dE_da*mul_lists(d_a[0:5],diffs_zeta) + dE_db*mul_lists(d_b[0:5],diffs_zeta) \
                        + dE_dc*mul_lists(d_c[0:5],diffs_zeta))
            
            dE_du0 = np.zeros(len(Pmeas))
            dE_du1 = Du1*(dE_da*mul_lists(d_a[0:5],diffs_u1) + dE_db*mul_lists(d_b[0:5],diffs_u1) \
                        + dE_dc*mul_lists(d_c[0:5],diffs_u1))
            dE_du2 = Du2*(dE_da*mul_lists(d_a[0:5],diffs_u2) + dE_db*mul_lists(d_b[0:5],diffs_u2) \
                        + dE_dc*mul_lists(d_c[0:5],diffs_u2))
            dE_du3 = Du3*(dE_da*mul_lists(d_a[0:5],diffs_u3) + dE_db*mul_lists(d_b[0:5],diffs_u3) \
                        + dE_dc*mul_lists(d_c[0:5],diffs_u3))
            
            dE_dP = dE_da*d_a[5] + dE_db*d_b[5] + dE_dc*d_c[5]
            dE_dTamb = dE_da*d_a[6] + dE_db*d_b[6] + dE_dc*d_c[6]
            dE_dv = dE_da*d_a[7] + dE_db*d_b[7] + dE_dc*d_c[7]
            dE_dTsky = dE_da*d_a[8] + dE_db*d_b[8] + dE_dc*d_c[8]
            dE_dTmod = np.zeros(len(Pmeas))
            
            #Combine derivatives into a matrix
            dE_vec = np.array([dE_ds,dE_dzeta,dE_du0,dE_du1,dE_du2,dE_du3])
            
        elif T_model == "King":            
            b1 = s*(1. + zeta*25.)    
            b2 = -s*zeta*(np.exp(a0+b0*vwind) + dT/1000.)
            b3 = -s*zeta
            b4 = 0.
            b5 = 0.
            
            #Calculate derivatives for uncertainties in quadrature
            #derivatives wrt s
            db1_ds = 1. + zeta*25.
            db2_ds = -zeta*(np.exp(a0+b0*vwind) + dT/1000.)
            db3_ds = -zeta            
            
            diffs_s = [db1_ds,db2_ds,db3_ds,0.,0.]
            
            #derivatives wrt zeta
            db1_dzeta = s*25.
            db2_dzeta = -s*(np.exp(a0+b0*vwind) + dT/1000.)
            db3_dzeta = -s            
            
            diffs_zeta = [db1_dzeta,db2_dzeta,db3_dzeta,0.,0.]
            
            #derivatives wrt a
            db2_da0 = -s*zeta*np.exp(a0+b0*vwind)
            
            diffs_a0 = [0.,db2_da0,0.,0.,0.]
            
            #derivatives wrt b
            db2_db0 = -s*zeta*vwind*np.exp(a0+b0*vwind)
            
            diffs_b0 = [0.,db2_db0,0.,0.,0.]
            
            #derivatives wrt dT
            db2_ddT = -s*zeta/1000.
            
            diffs_dT = [0.,db2_ddT,0.,0.,0.]  

            (a,b,c), d_a, d_b, d_c = king_combination_fn(b1,b2,b3,b4,b5,Pmeas,TambC,vwind)
            #P = Etotpoa(b1 + b2*Etotpoa + b3*Tamb + b4*vwind)
            #i.e.  b2*Etotpoa^2 + (b1 + b3*Tamb + b4*vwind)*Etotpoa - P = 0    
            
            E_solution, dE_da, dE_db, dE_dc = solve_quadratic_eqn(a,b,c)
            
            dE_ds = Ds*(dE_da*mul_lists(d_a[0:5],diffs_s) + dE_db*mul_lists(d_b[0:5],diffs_s)
                        + dE_dc*mul_lists(d_c[0:5],diffs_s))
            dE_dzeta = Dzeta*(dE_da*mul_lists(d_a[0:5],diffs_zeta) +
                       dE_db*mul_lists(d_b[0:5],diffs_zeta) + dE_dc*mul_lists(d_c[0:5],diffs_zeta))
            
            dE_da0 = Da0*(dE_da*mul_lists(d_a[0:5],diffs_a0) + dE_db*mul_lists(d_b[0:5],diffs_a0)
                        + dE_dc*mul_lists(d_c[0:5],diffs_a0))
            dE_db0 = Db0*(dE_da*mul_lists(d_a[0:5],diffs_b0) + dE_db*mul_lists(d_b[0:5],diffs_b0)
                        + dE_dc*mul_lists(d_c[0:5],diffs_b0))
            dE_ddT = DdT*(dE_da*mul_lists(d_a[0:5],diffs_dT) + dE_db*mul_lists(d_b[0:5],diffs_dT)
                        + dE_dc*mul_lists(d_c[0:5],diffs_dT))
            
            dE_dP = dE_da*d_a[5] + dE_db*d_b[5] + dE_dc*d_c[5]
            dE_dTamb = dE_da*d_a[6] + dE_db*d_b[6] + dE_dc*d_c[6]
            dE_dv = dE_da*d_a[7] + dE_db*d_b[7] + dE_dc*d_c[7]  
            dE_dTsky = np.zeros(len(Pmeas))
            dE_dTmod = np.zeros(len(Pmeas))
            
        
            #Combine derivatives into a matrix
            dE_vec = np.array([dE_ds,dE_dzeta,dE_da0,dE_db0,dE_ddT])
            
        #Get rid of the zero entries
        dE_vec_nonzero = dE_vec[~np.all(dE_vec == 0, axis=1),:]                            
    
    #This is now the shortwave broadband irradiance
    dataframe[('Etotpoa_pv_inv',substat_inv)] = E_solution
    
    #Calculate the errors in Gaussian quadrature, also taking into account correlation terms                
    dataframe[('error_Etotpoa_pv_inv',substat_inv)] = np.sqrt(np.diag(np.dot(np.dot(
            np.transpose(dE_vec_nonzero),errors),dE_vec_nonzero)) +
            dE_dP**2*P_error_meas**2 + dE_dTmod**2*err_Tmod**2 + dE_dTamb**2*err_Tamb**2 + 
                dE_dv**2*err_vwind**2 + dE_dTsky**2*err_Tsky**2)
    
    #Calculate modelled PV temperature
    if T_model == "Tamizhmani":
        dataframe[('T_module_inv_C',substat_inv)] = temp_model_tamizhmani(\
            [u0,u1,u2,u3],TambC,dataframe[('Etotpoa_pv_inv',substat_inv)].values,\
            vwind,TskyC)["T_mod"]
    elif T_model == "Faiman" or T_model == "Barry":
        dataframe[('T_module_inv_C',substat_inv)] = temp_model_faiman(\
            [u1,u2,u3],TambC,dataframe[('Etotpoa_pv_inv',substat_inv)].values,\
            vwind,TskyC)["T_mod"]
    elif T_model == "King":
        dataframe[('T_module_inv_C',substat_inv)] = temp_model_king(\
            [a0,b0,dT],TambC,dataframe[('Etotpoa_pv_inv',substat_inv)].values,\
            vwind)["T_mod"]
    
    #Calculate efficiency correction
#        dataframe[('eff_temp_inv',substat_inv)] = 1.0 - zeta*(
#                dataframe[('T_module_inv_C',substat_inv)] - 25.0)    
    
    #Assign dataframe back to dictionary
    df_result = dataframe.loc[:,pd.IndexSlice[['Etotpoa_pv_inv','error_Etotpoa_pv_inv',
                                               'T_module_inv_C'],substat_inv]]
    
    return df_result

def etotpoa_diode_model(dataframe,pvrad_config,station,substat_inv,substat_type):
    """
    

    Parameters
    ----------
    dataframe : dataframe with relevant PV data
    pvrad_config : dictionary with inversion configuration
    station : string, name of PV station
    substat_inv : string, name of substation for inversion
    substat_type : string, type of PV substation (data)

    Returns
    -------
    df_result : dataframe with inverted result

    """
      
    
    if 'opt_pars' in pvrad_config["pv_stations"][station][substat_type]\
                ["data"][substat_inv] and pvrad_config["pv_stations"][station][substat_type] == "opt":
        opt_pars = pvrad_config["pv_stations"][station][substat_type]\
                ["data"][substat_inv]['opt_pars']        
        err_state = pvrad_config["pv_stations"][station][substat_type]\
                ["data"][substat_inv]['opt']['S_hat']
    
    else:
        opt_pars = pvrad_config["pv_stations"][station][substat_type]\
                ["data"][substat_inv]['ap_pars']        
        err_state = pvrad_config["pv_stations"][station][substat_type]\
                ["data"][substat_inv]['opt']['S_a']
    
        print('No solution found, using a-priori values for inversion')
    
    impn = [par for name, par, err in opt_pars if name == "impn"][0]
    npp = [par for name, par, err in opt_pars if name == "npp"][0]
    #dataframe[("T_module_C",substat)] = dataframe[("T_module_C","model")]

    df_substat = dataframe.xs(substat,level="substat",axis=1)
    dfs_result = []
    
    for col in df_substat.columns:
        if "Idc" in col:            
            num = '_' + col.split('_')[1]  
            if num == "_A":
                num = ""
            
            # #get G from two diode model
            # Etotpoa = pd.Series(dm.G_IV_twodiode(df_substat[col].values,
            #       df_substat["Udc_"+num].values.flatten(),
            #       df_substat["T_module_C"].values,dpars["Impn"],dpars["Iscn"],
            #       dpars["Ki"],dpars["Vocn"],dpars["Kv"],dpars["Ns"],dpars["a1"],dpars["a2"],
            #       dpars["Rs"],dpars["Rp"],dpars["Nss"],dpars["Npp"]),
            #      index=df_substat.index,name='Etotpoa_inv_current_' + num)
    
            # #Get G from simplified model (without diode terms)
            # Etotpoa_simp = pd.Series(dm.G_IV_nodiode(df_substat[col].values,
            #       df_substat["Udc_"+num].values.flatten(),df_substat["T_module_C"].values,
            #       dpars["Impn"],dpars["Ki"],dpars["Npp"]),
            #     index=df_substat.index,name='Etotpoa_inv_simp_current_' + num)
            
            #Get G from simplified model (without diode terms) and no temperature dependence
            Etotpoa_simp_no_temp = pd.Series(dm.G_IV_nodiode(df_substat[col].values,
                  0.,impn,0.,npp),
                index=df_substat.index,name='Etotpoa_inv_simp_current_no_temp' + num)
            
            df_new = pd.concat([Etotpoa_simp_no_temp],axis=1)
            
            df_new.columns = pd.MultiIndex.from_product([df_new.columns.values.tolist(),[substat]],
                                                               names=['variable','substat'])  
            dfs_result.append(df_new)

    df_result = pd.concat(dfs_result,axis=1)                                

    return df_result

def iam_diffuse_inversion(tilt):
    """
    This is the incidence angle modifier for diffuse irradiance, as a function of
    tilt angle. Taken from Duffie and Beckman
    
    args:
    :param tilt: float, angle in degrees
    
    out:
    :return effective diffuse angle of incidence
    
    """    
    return 59.7 - 0.1388*tilt + 0.001497*tilt**2

def cloud_fraction(dataframe_sim,dataframe,substat_inv,year,substat_pars,
                   const_opt,angles,cs_threshold):
    """
    Calculate cloud fraction by first calculating clear sky reference (under the glass
    cover) and then taking the ratio
    
    args:    
    :param dataframe_sim: dataframe with clear sky simulation
    :param dataframe: dataframe with PV data and inverted irradiance
    :param substat_inv: string, substation to invert
    :param year: sring with year of measurement data
    :param substat_pars: dictionary with configuration for substation
    :param const_opt: named tuple with optical constants
    :param angles: named tuple with angle grid for DISORT   
    :param cs_threshold: float, threshold for defining clear sky
    
    out:
    :return dataframe_sim: dataframe with clear sky simulation
    :return dataframe: dataframe with PV data and cloud fraction 
    
    """
    
    print('Calculating POA clear sky irradiance for %s' % substat_inv)    

    #Extract irradiance parameters    
    if "opt_pars" in substat_pars:    
        irrad_pars = [substat_pars["opt_pars"][0][1],
                  substat_pars["opt_pars"][1][1],
                  substat_pars["opt_pars"][2][1]]
    else:
        irrad_pars = [substat_pars["ap_pars"][0][1],
                  substat_pars["ap_pars"][1][1],
                  substat_pars["ap_pars"][2][1]]
    
    #Get clear sky curve, changed to full spectrum "pyr" due to modified inversion, 20.5.2022
    if irrad_pars[0] == 0 and irrad_pars[1] == 0:
        #This is the horizontal case, with absoroption due to optical model
        dataframe_sim[('Etotpoa_pv_clear',substat_inv)] = \
        (dataframe_sim['Edirdown','libradtran_pyr'] + dataframe_sim['Ediffdown','libradtran_pyr'])\
        *ang_response(0., irrad_pars[2], const_opt.kappa, const_opt.L)
                        
    else:
        #Calculate clear sky reference
        Edirdown = dataframe_sim["Edirdown","libradtran_pyr"].values.flatten()
        sun_position = collections.namedtuple('sun_position','sza azimuth')
        sun_pos = sun_position(np.deg2rad(dataframe_sim.sza.values.flatten()),
                           np.deg2rad(dataframe_sim.phi0.values.flatten()))                
        
        #Get the diffuse radiance field
        diff_field = np.zeros((len(dataframe_sim),len(angles.theta),len(angles.phi)))                
        for i, itime in enumerate(dataframe_sim.index):
            diff_field[i,:,:] = dataframe_sim.loc[itime,('Idiff','libradtran_pyr')].values #.flatten()
            
        dict_E_poa_calc = E_poa_calc(irrad_pars,Edirdown,diff_field,
                  sun_pos,angles,const_opt,None,True)
        
        dataframe_sim[('Etotpoa_pv_clear',substat_inv)] = dict_E_poa_calc['Etotpoa_pv']
        dataframe_sim[('tau_pv',substat_inv)] = dict_E_poa_calc["trans_dir"]
        dataframe_sim[('theta_IA',substat_inv)] = np.rad2deg(np.arccos(dict_E_poa_calc["cos_poa_dir"]))
        
    effective_angle_diffuse = np.deg2rad(iam_diffuse_inversion(np.rad2deg(irrad_pars[0])))
    
    tau_diff = ang_response(effective_angle_diffuse, irrad_pars[2], const_opt.kappa, const_opt.L)
    
    #Calculate clearness index
    print('Calculating clearness index and cloud fraction for %s' % substat_inv)
    df_substat = dataframe.xs(substat,level="substat",axis=1)
    dfs_result = []
    for col in df_substat:
        if "Etotpoa" in col and "error" not in col:
            #interpolate clear sky irradiance
            Etotpoa_pv_clear_Wm2 = dataframe_sim[('Etotpoa_pv_clear',\
                  substat_inv)].reindex(df_substat.index).interpolate().\
                  rename("Etotpoa_pv_clear_Wm2",inplace=True)
                  
            #interpolate transmission function
            tau_pv = dataframe_sim[('tau_pv',\
                  substat_inv)].reindex(df_substat.index).interpolate().\
                  rename("tau_pv",inplace=True)
                  
            theta_IA = dataframe_sim[('theta_IA',\
                  substat_inv)].reindex(df_substat.index).interpolate().\
                  rename("theta_IA",inplace=True)
            
            #Calculate clearness index
            k_index = df_substat[col]/Etotpoa_pv_clear_Wm2
            k_index.rename('k_index_poa',inplace=True)
            
            #If clearness index is below a threshold, use diffuse IAM
            tau_pv[(k_index < 0.3)] = tau_diff # & (tau_pv > tau_diff)
            
            Etotpoa_pv_inv_tau = df_substat[col]/tau_pv
            Etotpoa_pv_inv_tau.rename('Etotpoa_pv_inv_tau',inplace=True)
            
            #Calculate cloud fraction
            cloud_fraction = 1. - (cs_threshold[0] < k_index).astype(float)  
            cloud_fraction.rename('cloud_fraction_poa',inplace=True)
                
            #No value if k_index is undefined
            cloud_fraction.loc[k_index.isna()] = np.nan
            
            #Mark the overshoots
            cloud_fraction.loc[k_index > cs_threshold[1]] = -999
            
            df_new = pd.concat([Etotpoa_pv_clear_Wm2,Etotpoa_pv_inv_tau,theta_IA,
                                tau_pv,k_index,cloud_fraction],axis=1)
            df_new.columns = pd.MultiIndex.from_product([df_new.columns.values.tolist(),[substat]],
                                                               names=['variable','substat'])  
            dfs_result.append(df_new)
            
    df_result = pd.concat(dfs_result,axis=1)                                            
    
    dataframe = pd.concat([dataframe,df_result],axis=1)
    
    dataframe_sim.sort_index(axis=1,level=1,inplace=True)
    dataframe.sort_index(axis=1,level=1,inplace=True)
    
    return dataframe_sim, dataframe

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

def generate_folders(inv_model,rt_config,pvcal_config,pvrad_config,home):
    """
    Generate folders for results
    
    args:
    :param inv_model: string, either "current" or "power"
    :param rt_config: dictionary with configuration of RT simulation
    :param pvcal_config: dictionary with configuration for calibration
    :param pvrad_config: dictionary with configuration for inversion
    :param home: string, home directory
    
    out:
    :return fullpath: string with label for saving folders    
    
    """    

    path = os.path.join(pvrad_config["results_path"]["main"],
                        pvrad_config["results_path"]["inversion"])
    
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
    wvl_config = rt_config["common_base"]["wavelength"]["pv"]
    
    if type(wvl_config) == list:
        wvl_folder_label = "Wvl_" + str(int(wvl_config[0])) + "_" + str(int(wvl_config[1]))
    elif wvl_config == "all":
        wvl_folder_label = "Wvl_Full"

    dirs_exist = list_dirs(fullpath)
    fullpath = os.path.join(fullpath,wvl_folder_label)        
    if wvl_folder_label not in dirs_exist:
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
        
    sza_label = "SZA_" + str(int(pvrad_config["sza_max"]["inversion"]))
    
    dirs_exist = list_dirs(fullpath)
    fullpath = os.path.join(fullpath,sza_label)    
    if sza_label not in dirs_exist:
        os.mkdir(fullpath)
        
    if inv_model == "power":
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
    
    elif inv_model == "current":
        model = "Diode_Model"
        dirs_exist = list_dirs(fullpath)
        fullpath = os.path.join(fullpath,model)  
        if model not in dirs_exist:
            os.mkdir(fullpath)        
        
    T_model = pvcal_config["T_model"]["model"]
    dirs_exist = list_dirs(fullpath)
    fullpath = os.path.join(fullpath,T_model)  
    if T_model not in dirs_exist:
        os.mkdir(fullpath)        
        
    return fullpath

def save_results(key,pv_station,substat_type,pvrad_config,pvcal_config,rt_config,
                 path,home,inv_model):
    """
    Save inversion results to a binary file and to text files
    
    args:    
    :param key: string, name of PV station
    :param pv_station: dictionary with data and info of PV system        
    :param substat_type: string, type of data, either "p_ac.." or "i_dc..."
    :param pvrad_config: dictionary with configuration for inversion
    :param pvcal_config: dictionary with configuration for calibration
    :param rt_config: dictionary with configuration of RT simulation
    :param path: path for saving results
    :param home: string, home path
    :param inv_model: string, either "current" or "power"
    
    """
                    
    atm_source = rt_config["atmosphere"]
    asl_source = rt_config["aerosol"]["source"]
    asl_res = rt_config["aerosol"]["data_res"]
    res = rt_config["disort_rad_res"]    
        
    #get description/s
    if len(pvrad_config["inversion_source"]) > 1:
        infos = '_'.join(pvrad_config["inversion_source"])
    else:
        infos = pvrad_config["inversion_source"][0]
    
    filename = 'tilted_irradiance_cloud_fraction_results_'
    if atm_source == 'cosmo':
        filename = filename + 'atm_'
    if asl_source != 'default':
        filename = filename + 'asl_' + asl_res + '_'
    
    filename = filename + infos + '_disortres_' + str(res["theta"]).replace('.','-')\
                    + '_' + str(res["phi"]).replace('.','-') + '_'
    
    filename_stat = filename + key + '.data'
    
    with open(os.path.join(path,filename_stat), 'wb') as filehandle:  
        # store the data as binary data stream
        pickle.dump((pv_station, rt_config, pvcal_config, pvrad_config), filehandle)

    print('Results written to file %s\n' % filename_stat)

    #Write to CSV file            
    if inv_model == "power":
        model = pvcal_config["inversion"]["power_model"]
        eff_model = pvcal_config["eff_model"]
    elif inv_model == "current":
        model = "diode model"
        eff_model = ""
    
    T_model = pvcal_config["T_model"]
    sza_limit = pvrad_config["sza_max"]["poa"]

    #Get timeres
    timeres = pvrad_config["pv_stations"][key][substat_type]["t_res_inv"]

    #Put both dataframes together
    dfs = []
    falltage = []
    for measurement in pvrad_config["pv_stations"][key][substat_type]["source"]:
        year = "mk_" + measurement.split('_')[1]
        dfname = 'df_' + year.split('_')[-1] + '_' + timeres
        dfs.append(pv_station[dfname])
        falltage.extend(pvrad_config["falltage"][year])
    
    dataframe = pd.concat(dfs,axis='index')
                        
    for day in falltage:
        dfday = dataframe.loc[day.strftime('%Y-%m-%d')]
                    
        filename_csv = 'tilted_irradiance_cloud_fraction_' + key + '_' +\
             substat_type + '_' + day.strftime('%Y-%m-%d') + '.dat'
            
        f = open(os.path.join(path,"CSV_Results",filename_csv), 'w')
        f.write('#Station: %s, Irradiance data and cloud fraction from DISORT calibration/simulation\n' % key)    
        f.write('#Tilted irradiance and cloud fraction inferred from %s\n' % substat_type)
        f.write('#Results up to maximum SZA of %d degrees\n' % sza_limit)
        if inv_model == "power":
            f.write('#PV model: %s, efficiency model: %s, temperature model: %s\n' % (model,eff_model,T_model))        
        elif inv_model == "current":
            f.write('#PV model: %s, temperature model: %s\n' % (model,T_model))        
        for substat in pv_station["substations"][substat_type]["data"]:                
            ap_pars = pv_station["substations"][substat_type]["data"][substat]["ap_pars"]
            f.write('#A-priori parameters (value,error):\n')
            for par in ap_pars:
                f.write('#%s: %g (%g)\n' % par)                        
            if "opt_pars" in pv_station["substations"][substat_type]["data"][substat]:
                opt_pars = pv_station["substations"][substat_type]["data"][substat]["opt_pars"]
                f.write('#Optimisation parameters (value,error):\n')
                for par in opt_pars:
                    f.write('#%s: %g (%g)\n' % par)     
            else:
                f.write('No solution found by the optimisation routine, using a-priori values\n')

        f.write('\n#Multi-index: first line ("variable") refers to measured quantity\n')
        f.write('#second line ("substat") refers to measurement device\n')
        f.write('\n')                       
        
        dfday.to_csv(f,sep=';',float_format='%.6f',
                          na_rep='nan')
        f.close()    
        print('Results written to file %s\n' % filename_csv) 
        
    for measurement in pvrad_config["inversion_source"]:
        year = measurement.split('_')[1] 
        if f"mk_{year}" in pv_station['substations'][substat_type]["source"]:
            dfname = f'df_{year}_{timeres}'        
        
            #Write all results to CSV file
            filename_csv = f'tilted_irradiance_cloud_fraction_{key}_{timeres}_{year}.dat'
            f = open(os.path.join(path,"CSV_Results",filename_csv), 'w')
            f.write('#Station: %s, Irradiance data and cloud fraction from DISORT calibration/simulation, %s\n' % (key,year))    
            f.write('#Tilted irradiance and cloud fraction inferred from %s\n' % substat_type)
            f.write('#Results up to maximum SZA of %d degrees\n' % sza_limit)
            if inv_model == "power":
                f.write('#PV model: %s, efficiency model: %s, temperature model: %s\n' % (model,eff_model,T_model))        
            elif inv_model == "current":
                f.write('#PV model: %s, temperature model: %s\n' % (model,T_model))        
            for substat in pv_station["substations"][substat_type]["data"]:                
                ap_pars = pv_station["substations"][substat_type]["data"][substat]["ap_pars"]
                f.write('#A-priori parameters (value,error):\n')
                for par in ap_pars:
                    f.write('#%s: %g (%g)\n' % par)                        
                if "opt_pars" in pv_station["substations"][substat_type]["data"][substat]:
                    opt_pars = pv_station["substations"][substat_type]["data"][substat]["opt_pars"]
                    f.write('#Optimisation parameters (value,error):\n')
                    for par in opt_pars:
                        f.write('#%s: %g (%g)\n' % par)     
                else:
                    f.write('No solution found by the optimisation routine, using a-priori values\n')
    
            f.write('\n#Multi-index: first line ("variable") refers to measured quantity\n')
            f.write('#second line ("substat") refers to measurement device\n')
            f.write('#First column is the time stamp, in the format %Y-%m-%d %HH:%MM:%SS\n')
            f.write('\n')                       
                    
            pv_station[dfname].loc[:,pd.IndexSlice[:,[substat,"sun"]]].to_csv(f,sep=';',float_format='%.6f',
                              na_rep='nan')
            f.close()    
            print('Results written to file %s\n' % filename_csv)             

#%%Main Program
#######################################################################
###                         MAIN PROGRAM                            ###
#######################################################################
#The goal of this program is to take a calibrated PV plant and infer the 
#irradiance on the plane of the array,
#Depending on which model is used, i.e. power model, current from diode model
#The POA irradiance, clearness index and cloud fraction is output 
#and can then be used to calculate atmospheric optical properties
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
    config_filename = "config_PV2RAD_MetPVNet_messkampagne.yaml" #os.path.abspath(args.configfile)
 
config = load_yaml_configfile(config_filename)                                                                                                                                                                                                                                                                                                          

#Load PV configuration
pvcal_config = load_yaml_configfile(config["pvcal_configfile"])

#Load PV configuration
pvrad_config = load_yaml_configfile(config["pvrad_configfile"])

#Load radiative transfer configuration
rt_config = load_yaml_configfile(config["rt_configfile"])

homepath = os.path.expanduser('~') #"/media/luke" #

if args.station:                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
    stations = args.station
    if stations[0] == 'all':
        stations = 'all'
else:
    #Stations for which to perform inversion
    stations = "PV_11" #"MS_02" #pvrad_config["stations"]

#Choose measurement campaign
if args.campaign:
    pvrad_config["inversion_source"] = args.campaign

#Load calibration results, including DISORT RT simulation for clear sky days and COSMO data.
print('Inversion of PV power onto plane-of-array irradiance for %s' % stations)
print('Loading DISORT simulation data and calibration results')
pvsys = load_radsim_calibration_results(config["description"],rt_config,
                                        pvcal_config,pvrad_config,stations,
                                        homepath)

inv_params = pvcal_config["inversion"]
sza_limit = pvrad_config["sza_max"]["poa"]
cs_threshold = pvrad_config["cloud_fraction"]["cs_threshold"]

albedo = rt_config["albedo"]

disort_res = rt_config["disort_rad_res"]
grid_dict = define_disort_grid(disort_res)

opt_dict = pvcal_config["optics"]
optics_flag = opt_dict["flag"]
optics = collections.namedtuple('optics', 'kappa L')
const_opt = optics(opt_dict["kappa"],opt_dict["L"])

angle_grid = collections.namedtuple('angle_grid', 'theta phi umu')
angle_arrays = angle_grid(grid_dict["theta"],np.deg2rad(grid_dict["phi"]),grid_dict["umu"])

#Perform inversion for each system
for key in pvsys: 
    print(f'Performing inversion for {key}')
    #Put the results from calibration into the config dictionary for PV2RAD
    for substat_type in pvrad_config["pv_stations"][key]:
        #if "p_ac" in substat_type:
        for substat in pvsys[key]['substations']:
            if substat in pvrad_config["pv_stations"][key][substat_type]["data"]:     
                if "error_days" in pvrad_config["pv_stations"][key][substat_type]["data"]\
                [substat]:
                    pvsys[key]['substations'][substat]["error_days"] = \
                    pvrad_config["pv_stations"][key][substat_type]["data"][substat]["error_days"]
                    
                pvrad_config["pv_stations"][key][substat_type]["data"]\
                [substat] = merge_two_dicts(pvrad_config["pv_stations"][key][substat_type]["data"]\
                [substat],pvsys[key]['substations'][substat])        
                
                    
    #Load substation config
    pvsys[key]['substations'] = pvrad_config["pv_stations"][key]
    
    #Calculate linear fit using ratio of Pv to pyranometer irradiance
    if pvrad_config["spectral_mismatch_all_sky"]:
        pvsys[key] = linear_fit_water_vapour(key, pvsys[key], pvrad_config, const_opt, angle_arrays)
        
    #Load temperature model results if they exist
    #if pvcal_config["T_model"] == "Dynamic_or_Measured":  
    pvsys[key] = load_temp_model_results(key,pvsys[key],pvcal_config,homepath)
    #pvsys[key], pvtemp_config = load_temp_model_results(key,pvsys[key],pvrad_config,homepath)        
    
    #Perform inversion onto POA irradiance for each substation
    for substat_type in pvsys[key]['substations']:        #["inverter"]: #
    
        #Get model type
        model_type = pvsys[key]["substations"][substat_type]["type"]
        
        #Generate folders for results
        savepath = generate_folders(model_type,rt_config,pvcal_config,pvrad_config,homepath)
        
        print(f"Inverting power onto tilted irradiance using {model_type} data")
        print(f"Using {pvcal_config['inversion']['power_model']}, {pvcal_config['T_model']['type']}, "\
              f"{pvcal_config['T_model']['model']}, {pvcal_config['eff_model']} model")       
                    
        for measurement in pvrad_config["inversion_source"]:
            year = "mk_" + measurement.split('_')[1]  
            #If there is data for this year, perform inversion
            if year in pvsys[key]['substations'][substat_type]["source"]:
                data_config = load_yaml_configfile(config["data_configfile"][year])
                timeres = pvsys[key]["substations"][substat_type]["t_res_inv"]
                
                #Load data, rename dataframe for the year
                dfname = "df_" + measurement.split('_')[1]  + '_' + timeres
                pvstat_rs = load_resampled_data(key,timeres,measurement,data_config,homepath)
                pvstat_rs[dfname] = pvstat_rs["df"]
                del pvstat_rs["df"]
                pvsys[key] = merge_two_dicts(pvsys[key],pvstat_rs)
                
                #Drop nans if all columns are nan
                pvsys[key][dfname].dropna(axis=0,how='all',inplace=True)                
                                                            
                #If no wind measurement then we need to resample COSMO data to correct resolution
                #This is not necessarily accurate but better than nothing? 23.10.2020 James
                # if (pvcal_config["pv_stations"][key]["input_data"]["wind"][year] == "cosmo")\
                # or (pvcal_config["pv_stations"][key]["input_data"]["temp_amb"][year] == "cosmo"):
                pvsys[key][dfname] = pd.concat([pvsys[key][dfname],resample_cosmo(
                        pvsys[key]["df_cosmo_" + year.split('_')[1]],timeres)],axis=1)
                    
                #This is a temperorary bit to resolve memory issues, 27.10.2020
                # dfs = []
                # for day in pvrad_config["falltage"][year]:
                #     dfs.append(pvsys[key][dfname].loc[day.strftime('%Y-%m-%d')])
                # pvsys[key][dfname] = pd.concat(dfs,axis=0)     
                            
                #Throw away night time values
                print('Calculating sun position, keep only values up to SZA = %d degrees' % sza_limit)
                pvsys[key][dfname] = get_sun_position(pvsys[key],dfname,sza_limit)                        
                
                dfsim = "df_sim_" + measurement.split('_')[1] 
                #Throw away times before and after which we have simulation
                pvsys[key][dfname] = pvsys[key][dfname].loc[(pvsys[key][dfname].index 
                           >= pvsys[key][dfsim].index[0]) & (pvsys[key][dfname].index <= 
                                       pvsys[key][dfsim].index[-1])]                                                        
                
                if pvcal_config["longwave_data"]:
                    lw_station = pvcal_config["pv_stations"][key]["input_data"]["longwave"][year]                
                    pvsys[key][dfname] = prepare_longwave_data(key, pvsys[key], dfname,
                                           measurement, lw_station, timeres, pvcal_config["atm_emissivity"])
                                                             
                for substat in pvsys[key]["substations"][substat_type]["data"]:
                    print('Inverting onto irradiance for %s' % substat)
                    substat_pars = pvrad_config["pv_stations"][key]\
                                    [substat_type]["data"][substat]                    

                    #If necessary, calculate modelled temperature
                    if pvcal_config["T_model"]["model"] == "Dynamic_or_Measured" and \
                    "model" in pvcal_config["pv_stations"][key]["input_data"]["temp_module"][year]:
                        pvsys[key][dfname] = pd.concat([pvsys[key][dfname],calculate_temp_module(key,
                         pvsys[key],substat,pvcal_config,pvrad_config,data_config,timeres,
                         measurement,homepath)],axis=1)                    

                    if model_type == "current":                        
                        pvsys[key][dfname] = pd.concat([pvsys[key][dfname],
                              etotpoa_diode_model(pvsys[key][dfname],pvrad_config,
                                                  key,substat,substat_type)],axis=1)
                    
                    elif model_type == "power":
                        pvsys[key][dfname] = pd.concat([pvsys[key][dfname],
                              etotpoa_power_model(key,pvsys[key],dfname,substat,year,
                                                  pvcal_config,pvrad_config,
                                                  substat_type,timeres)],axis=1)
                                            
                    #Calculate cloud fraction
                    pvsys[key][dfsim], pvsys[key][dfname] = \
                        cloud_fraction(pvsys[key][dfsim],pvsys[key][dfname],\
                        substat,year,substat_pars,const_opt,angle_arrays,cs_threshold)
                            
                #Make sure pyranometer data is bias corrected for validation
                pyrcal_config = load_yaml_configfile(config["pyrcal_configfile"][year])
                pyrname = pvrad_config["pv_stations"][key][substat_type]["pyr_down"][year][1]
                if "Horiz" in pyrname:
                    pyrname = pyrname.split('_')[0] + "_32S"
                    
                pyr_station = pvrad_config["pv_stations"][key][substat_type]["pyr_down"][year][0]
                radname = pyrcal_config["pv_stations"][pyr_station]["substat"][pyrname]["name"]                
    
    
        # Save solution for key to file
        save_results(key,pvsys[key],substat_type,pvrad_config,pvcal_config,
                     rt_config,savepath,homepath,model_type)
