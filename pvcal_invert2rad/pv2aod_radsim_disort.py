#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 11:26:58 2020

@author: james
"""

#%% Preamble
import os
import tempfile
import numpy as np
import collections
import pandas as pd
from copy import deepcopy

from file_handling_functions import *
from rt_functions import *

import multiprocessing as mp
import subprocess
import time
import pickle

#%%Functions
def generate_folder_names(rt_config,pvcal_config,sens_type,inv_model):
    """
    Generate folder structure to retrieve POA Rad results
    
    args:    
    :param rt_config: dictionary with RT configuration
    :param pvcal_config: dictionary with PVCAL configuration
    :param sens_type: string specifying type of sensor 
    :param inv_model: string, either power or current
    
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
    wvl_config = rt_config["common_base"]["wavelength"][sens_type]
    
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
        
    if inv_model == "power":
        model = pvcal_config["inversion"]["power_model"]
        eff_model = pvcal_config["eff_model"]
    elif inv_model == "current":
        model = "Diode_Model"
        eff_model = ""
        
    T_model = pvcal_config["T_model"]    

    folder_label = os.path.join(atm_geom_folder,wvl_folder_label,disort_folder_label,
                                atm_folder_label,aero_folder_label,model,eff_model,
                                T_model)
    
    return folder_label, filename, (theta_res,phi_res)
    

def load_pv2poarad_inversion_results(rt_config,pvcal_config,pvrad_config,station_list,
                                     sens_type,home):
    """
    Load results from inversion onto plane-of-array irradiance
    
    args:    
    :param rt_config: dictionary with current RT configuration
    :param pvcal_config: dictionary with current calibration configuration    
    :param pvrad_config: dictionary with current inversion configuration
    :param station_list: list of stations
    :param sens_type: string specifying type of sensor 
    :param home: string with homepath
    
    out:
    :return pv_systems: dictionary of PV systems with data
    :return rt_config: dictionary with current RT configuration
    :return pvcal_config: dictionary with current calibration configuration    
    :return pvrad_config: dictionary with current inversion configuration
    :return folder_label: string with folder for results
    """
    
    mainpath = os.path.join(pvrad_config['results_path']['main'],
                            pvrad_config['results_path']['inversion'])    
    
    #Check calibration source for filename    
    if len(pvrad_config["calibration_source"]) > 1:
        infos = '_'.join(pvrad_config["calibration_source"])
    else:
        infos = pvrad_config["calibration_source"][0]        
    
    pv_systems = {}    
        
    #Choose which stations to load    
    if type(station_list) != list:
        station_list = [station_list]
        if station_list[0] == "all":
            station_list = pvrad_config["pv_stations"]
    
    for station in station_list:
        for substat_type in pvrad_config["pv_stations"][station]:
            model_type = pvrad_config["pv_stations"][station][substat_type]["type"]                
            #Read in binary file that was saved from pvcal_radsim_disort
            
            #Generate folder structure for loading files
            folder_label, filename, (theta_res,phi_res) = \
                generate_folder_names(rt_config,pvcal_config,sens_type,model_type)   
                
            filename = filename + infos + '_disortres_' + theta_res + '_' + phi_res + '_'
            
            filename_stat = filename + station + '.data'
            try:
                with open(os.path.join(mainpath,folder_label,filename_stat), 'rb') as filehandle:  
                    # read the data as binary data stream
                    (pvstat, rt_config, pvcal_config, pvrad_config) = pd.read_pickle(filehandle)            
                pv_systems.update({station:pvstat})
                print('Data for %s loaded from %s, %s' % (station,folder_label,filename))
            except IOError:
                print('There is no simulation for %s' % station)   
            
    return pv_systems, rt_config, pvcal_config, pvrad_config, folder_label
  
def generate_disort_files(rt_config,home,cosmo_path,modis_path,itime, #cloudfile,
                          grid,data,albedo,i_aod,aod_sample,sens_type):
    """
    Generate input files for libradtran based on time series data, one for each time
    step defined by itime
    
    args:
    :param rt_config: dictionary of config details for simulation
    :param home: string, home path
    :param cosmo_path: string with path for cosmo files
    :param modis_path: string with path for spectral albedo files from MODIS    
    :param itime: timestamp to be simulated
    :param grid: dictionary with grid for diffuse radiance field
    :param data: data point (row of dataframe) for which file should be generated
    :param albedo: tuple of lists with albedo files from MODIS
    :param i_aod: integer, index of AOD array
    :param aod_sample: float, sample AOD
    :param sens_type: string specifying type of sensor 
    
    out:
    :return inputfile: string, name of input file
    :return outputfile: string, name of output file
       
    """
    lrt_dir = os.path.join(home,rt_config["working_directory"])
    
    base_config = rt_config["common_base"]
    for key, value in rt_config["disort_base"].items():
        base_config.update({key:value})
    base_config["data_files_path"] = os.path.join(home,base_config["data_files_path"])
            
    #prepare input file for libradtran
    if rt_config["atmosphere"] == 'cosmo': #include COSMO-modified atmosphere
        atm_path_temp = os.path.join(home,cosmo_path,
        str(itime.date()).replace('-',''),str(itime.time()).replace(':','') +
        '_atmofile.dat')
    else:
        atm_path_temp = os.path.join(home,rt_config["atmosphere_file"])
                    
    inp = open(os.path.join(lrt_dir,'disort_run_' + str(i_aod) + '.inp'),'w+')
    inp.write('# Input file for DISORT simulation for PV calibration\n\n')
    inp.write('# Basis configuration (same for all points)\n\n')
    #Write all configuration steps to file (independent of time stamp)
    for key in base_config:
        if base_config[key]:
            if key == 'running_mode' or key == "aerosol": 
                inp.write(base_config[key] + '\n')
            elif key == 'wavelength':    
                if base_config[key][sens_type] != "all":
                    inp.write(key + ' ' + str(base_config[key][sens_type][0]) + ' ' + str(base_config[key][sens_type][1]) + '\n')
            elif type(base_config[key]) == bool:
                if base_config[key]:
                    inp.write(key + '\n')            
            else:
                if key != "aerosol_species_library" and key != "aerosol_species_file":
                    inp.write(key + ' ' + base_config[key] + '\n')      
                
    inp.write('umu %s \n' % grid["umustring"])
    inp.write('phi %s \n\n' % grid["phistring"])
    inp.write('zout 0.0015\n')    
              
    #Write time-varying part (atmosphere and albedo, day of year, sun position)
    inp.write('# Time-varying part\n\n')
    inp.write('atmosphere_file ' + atm_path_temp + ' \n')                
        
    # if rt_config["aerosol"]["source"] != 'default' and not np.isnan(data[('AOD_500', 'Aeronet')].values):
    #     inp.write('aerosol_angstrom %g %g \n' % (data[('alpha','Aeronet')],data[('beta','Aeronet')]))
    
    #SSA scale removed for the moment
    #                if asl_config["source"] != 'default' and not np.isnan(dataframe.loc[itime].ssa_vis):
    #                    inp.write('aerosol_modify ssa scale %g \n' % dataframe.loc[itime].ssa_vis)
    
    #Constant albedo from dataframe    
    if rt_config['albedo']['choice'] == 'constant':
        inp.write('albedo %g \n' % data[('albedo','constant')])
    #Spectral Albedo from MODIS data
    elif rt_config['albedo']['choice'] == 'MODIS':
        ix = albedo[0].index(str(itime.date()))
        modis_file = albedo[1][ix]
        if not modis_file:
            inp.write('albedo %g \n' % data[('albedo','constant')])
        else:
            brdf_path = os.path.join(modis_path,modis_file)
            inp.write('albedo_file %s \n' % brdf_path)
        
    #Don't need time and lat lon, day of year and sun position is enough
    inp.write('day_of_year %s \n' % itime.strftime('%j'))
    inp.write('phi0 %g \n' % data[('phi0','sun')])
    inp.write('sza %g \n' % data[('sza','sun')])    
    
    #Set aerosol optical depth    
    wvl_aod = str(rt_config["aerosol"]["lut_config"]["wavelength"])
    inp.write(f'aerosol_set_tau_at_wvl {wvl_aod} {aod_sample}\n')    
    inp.close()
    
    inputfile = inp.name
    outputfile = os.path.join(lrt_dir,'disort_' + str(i_aod) + '.out')
    
    return inputfile, outputfile

def read_disort_output(grid,output_file):
    """
    Read disort output file for one time stamp
    
    args:    
    :param grid: dictionary of grid for libradtran    
    :param output_file: string, name of DISORT output file
    
    out:
    :return Edirdown: array with direct downward irradiance
    :return Ediffdown:  array with diffuse downward irradiance
    :return Idiff: dataframe with radiance field distribution for one time step
    """    
    
    # read in irradiance values calculated in libradtran
    [Edirdown,Ediffdown] = pd.read_csv(output_file,sep='\s+',
                        header=None,nrows=1,usecols=[1,2]).values.flatten()

    # read in individual radiance values as per the defined umu, phi grid                
    Idiff = pd.read_csv(output_file,sep='\s+',
                    header=None,skiprows=2,usecols=np.arange(2,len(grid["phi"]) + 2,1))                
        
    return Edirdown, Ediffdown, Idiff.values

def disort_simulation_aod(dataframe,rt_config,grid,angles,
                          albedo_series,modis_path,cosmo_path,lrt_dir,
                          sens_type,home):
                          #cloudfile,substat,home):
    """    
    Run disort simulation for all times with high enough cloud fraction, to 
    generate a look-up table for the cloud optical depth
    
    args:    
    :param dataframe: dataframe for which to run DISORT
    :param rt_config: dictionary of config details for simulation                
    :param grid: dictionary with DISORT grid for radiance distribution
    :param angles: named tuple with angle grid for DISORT    
    :param albedo_series: tuple with information for albedo
    :param modis_path: string, path for MODIS albedo files
    :param cosmo_path: string, path for COSMO atmosphere files
    :param lrt_dir: string with path for saving libradtran files
    :param sens_type: string specifying type of sensor 
    :param home: string, home path    
    
    out:
    :return: dataframe with simulation results
    
    """            
    
    #Keep the whole dataframe and apply cloud fraction mask afterwards for comparison, 30.10.2020
    df_lrt = dataframe #.iloc[0:2,:] #[dataframe[("cloud_fraction_Etotpoa_pv_inv",substat)] > 0.8] #.iloc[0:4,:]
    
    df_lrt = pd.concat([df_lrt.loc[:,pd.IndexSlice[:,['sun','Aeronet']]],
                        df_lrt.loc[:,pd.IndexSlice['albedo',:]],
                        df_lrt.loc[:,pd.IndexSlice[['Edirdown_Wm2','Ediffdown_Wm2']
                        ,'cosmo']]],axis=1)                
    
    #List of simulation calculation times
    total_time = []
    
    #Get number of processors
    num_cores = mp.cpu_count()
            
    #Generate AOD for LUT
    lut_config = rt_config["aerosol"]["lut_config"]
    array_aod = np.logspace(np.log10(lut_config["range"][0]),
                            np.log10(lut_config["range"][1]),
                            lut_config["samples"])#.tolist()
    print('AOD will be sampled with %g points: %s' % (len(array_aod),array_aod))
    
    #List of CPUs
    if num_cores <= len(array_aod):
        n_time_vec = np.arange(1)
        n_aod_vec = np.arange(num_cores)
    else:
        n_time_vec = np.arange(0,int(num_cores/len(array_aod)))
        n_aod_vec = np.arange(len(array_aod))
    
    #List of disort files, tuple (input, output)
    disort_files = []
    
    #List of processes
    processes = []
    
    #Collect results    
    AOD_dirdown_tables = [np.zeros((len(n_time_vec)*len(array_aod),2),dtype=float)]*len(df_lrt)
    AOD_diffdown_tables = [np.zeros((len(n_time_vec)*len(array_aod),2),dtype=float)]*len(df_lrt)
    AOD_diff_field_tables = [np.zeros((len(array_aod),len(grid["theta"]),
                               len(grid["phi"])),dtype=float)]*len(df_lrt)            
    
    while True:                                            
        #Declare result arrays
        irrad_dict = {"dir_down":np.zeros(len(n_time_vec)*len(array_aod)),
                  "diff_down":np.zeros(len(n_time_vec)*len(array_aod)),
                  "diff_field":np.zeros((len(n_time_vec)*len(array_aod),
                  len(grid["theta"]),len(grid["phi"])))}
        
        #Check whether we have gone too far
        if np.max(n_time_vec) >= len(df_lrt):
            n_time_vec = n_time_vec[n_time_vec < len(df_lrt)]
            if len(n_time_vec) == 0:
                break
            else:
                continue                  
                
        #Get the timestamps corresponding to the points to be simulated
        itime_vec = df_lrt.index[n_time_vec]        
        
        #List of CPUs
        if num_cores <= len(array_aod):            
            n_aod_vec = np.arange(num_cores)
        else:
            n_aod_vec = np.arange(len(array_aod)) 
        
        #This only works if aod_vec is a multiple of n_cores
        while True:
            if np.max(n_aod_vec) >= len(array_aod):
                n_aod_vec = n_aod_vec[n_aod_vec < len(array_aod)]
                if len(n_aod_vec) == 0:
                    break
                else:
                    continue                
            
            #Prepare the input and output files, depending on number of cores
            for n in range(len(n_time_vec)):
                itime = itime_vec[n]            
            
                for i in n_aod_vec:
                    aod = array_aod[i]
                    #Generate input files to match number of cores                
                    disort_input, disort_output = generate_disort_files(rt_config,
                                                  home,cosmo_path,modis_path,
                                                  itime,grid,df_lrt.loc[[itime]],
                                                  albedo_series,n*len(n_aod_vec)+i,
                                                  aod,sens_type)
                        
                    disort_files.append((aod,disort_input,disort_output))
        
            #Start timer
            start = time.time()
            
            #Start subprocesses with libradtran simulation
            for (aod,input_file,output_file) in disort_files:
                logfile = tempfile.TemporaryFile()
                p = subprocess.Popen(['uvspec < ' + input_file + ' > ' + 
                                      output_file,input_file,output_file],
                                    stdout=logfile,shell=True,cwd=lrt_dir)
                processes.append((p, output_file, logfile))
                        
            for n, (p, output, log) in enumerate(processes):
                p.wait()
                log.close()
                #Read output files for the specified time step and save the results
                irrad_dict["dir_down"][n_aod_vec[n]], irrad_dict["diff_down"][n_aod_vec[n]],\
                irrad_dict["diff_field"][n_aod_vec[n]][:][:]\
                = read_disort_output(grid,output)
                
            #Clear lists of files
            del processes[:]
            del disort_files[:]
            
            n_aod_vec = n_aod_vec + len(n_aod_vec)                
                                
        #Save LUT to dataframe    
        for n in range(len(n_time_vec)):                
            AOD_dirdown_tables[n_time_vec[n]] = np.transpose(np.array([array_aod,irrad_dict["dir_down"]]))
            AOD_diffdown_tables[n_time_vec[n]] = np.transpose(np.array([array_aod,irrad_dict["diff_down"]]))
            AOD_diff_field_tables[n_time_vec[n]] = irrad_dict["diff_field"]
        
        
        print('%d: %s, %s: SZA %g, phi0 %g, Aeronet %g'  
              % (n_time_vec[n],itime_vec[n],key,
                 df_lrt.loc[itime_vec[n],'sza'],
                 df_lrt.loc[itime_vec[n],'phi0'],                 
                 df_lrt.loc[itime_vec[n],('AOD_500','Aeronet')]))#,        
        
        #stop timer and calculate runtime and total time
        end = time.time()
        runtime = end - start   
        total_time.append(runtime)
        
        print('Simulation of %d time steps took %g seconds' % (len(n_time_vec),runtime))                
        
        #Move to the next group of times
        if num_cores <= len(array_aod):
            n_time_vec = n_time_vec + 1
        else:
            n_time_vec = n_time_vec + int(num_cores/len(array_aod))
                
    #Put results into dataframe
    df_lrt[('AOD_dirdown_table',"libradtran")] = pd.Series(AOD_dirdown_tables,index=df_lrt.index)
    df_lrt[('AOD_diffdown_table',"libradtran")] = pd.Series(AOD_diffdown_tables,index=df_lrt.index)
    df_lrt[('AOD_diff_field_table',"libradtran")] = pd.Series(AOD_diff_field_tables,index=df_lrt.index)
    
    print('Simulation for %s took %g secs ' % (key,sum(total_time)))        
    
    return df_lrt #ataframe       

def generate_results_folders(rt_config,sens_type,path,home):
    """
    Generate folders for results
    
    args:
    :param rt_config: dictionary with DISORT config
    :param sens_type: string specifying type of sensor 
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
        
    #Wavelength range of simulation
    wvl_config = rt_config["common_base"]["wavelength"][sens_type]
    
    if type(wvl_config) == list:
        wvl_folder_label = "Wvl_" + str(int(wvl_config[0])) + "_" + str(int(wvl_config[1]))
    elif wvl_config == "all":
        wvl_folder_label = "Wvl_Full"

    dirs_exist = list_dirs(fullpath)
    fullpath = os.path.join(fullpath,wvl_folder_label)        
    if wvl_folder_label not in dirs_exist:
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
        
    sza_label = "SZA_" + str(int(rt_config["sza_max"]["lut"]))
    
    dirs_exist = list_dirs(fullpath)
    fullpath = os.path.join(fullpath,sza_label)
    if sza_label not in dirs_exist:
        os.mkdir(fullpath)
        
    return fullpath
    
def save_results(name,pv_station,info,name_df,rt_config,pvcal_config,
                 pvrad_config,path):
    """
    

    Parameters
    ----------
    name : string, name of station
    pv_station : dictionary with all information and data for PV station
    info : string with description of current campaign
    name_df : string with name of dataframe to be saved
    rt_config : dictionary with radiative transfer configuration
    pvcal_config : dictionary with calibraion configuration
    pvrad_config : dictionary with inversion configuration
    path : string with path to save results

    Returns
    -------
    None.

    """
    
    pv_station_save = deepcopy(pv_station)
             
    filename_stat = f"aod_lut_results_{info}_{name}.data"    
    
    dfnames = [name_df]
    dfnames.append('lat_lon')
    # dfnames.append('substations')
    # pv_station_save['substations'] = {}
    # for substat in pv_station['substations'][substat_type]['data']:
    #     pv_station_save['substations'][substat] = pv_station['substations'][substat_type]\
    #                                     ['data'][substat]
    #     for key in list(pv_station_save['substations'][substat]):
    #         if key not in ['opt_pars','ap_pars','name']:
    #             del pv_station_save['substations'][substat][key]
        
    # dataframe = pd.concat(dfs,axis='index')
        
    for key in list(pv_station_save):
        if key not in dfnames and "path" not in key:
            del pv_station_save[key]
            
    pv_station_save['station_name'] = name
    
    with open(os.path.join(path,filename_stat), 'wb') as filehandle:  
        # store the data as binary data stream
        pickle.dump((pv_station_save, rt_config, pvcal_config, pvrad_config), filehandle)

    print('Results written to file %s\n' % filename_stat)

#%%Main Program
#######################################################################
###                         MAIN PROGRAM                            ###
#######################################################################
#This program takes POA irradiance and cloud fraction data and uses DISORT
#to find the aerosol optical depth, for situations where the cloud fraction 
#is below a certain threshold
#def main():
import argparse
    
parser = argparse.ArgumentParser()
#parser.add_argument("configfile", help="yaml file containing config")
parser.add_argument("-s","--station",nargs='+',help="station for which to perform inversion")
parser.add_argument("-c","--campaign",nargs='+',help="measurement campaign for which to perform simulation")
args = parser.parse_args()
   
config_filename = "config_PV2RAD_solarwatt.yaml" #"MetPVNet_messkampagne.yaml" #os.path.abspath(args.configfile)
 
config = load_yaml_configfile(config_filename)

sensor_type = "pv"

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
    stations = pvrad_config["stations"]

#Choose measurement campaign
if args.campaign:
    pvrad_config["inversion_source"] = args.campaign
#%%Load inversion results
#Load calibration results, including DISORT RT simulation for clear sky days and COSMO data.
print('Simulating DISORT lookup table for AOD')
print('Loading PV2POARAD inversion results')

pvsys, dummy, pvcal_config, dummy, results_folder = \
load_pv2poarad_inversion_results(rt_config, pvcal_config, pvrad_config, stations, sensor_type, homepath)

#%%Calculate AOD with DISORT

disort_res = rt_config["aerosol"]["disort_rad_res"]
grid_dict = define_disort_grid(disort_res)

angle_grid = collections.namedtuple('angle_grid', 'theta phi umu')
angle_arrays = angle_grid(grid_dict["theta"],np.deg2rad(grid_dict["phi"]),grid_dict["umu"])

sun_position = collections.namedtuple('sun_position','sza azimuth')
sza_limit = rt_config["sza_max"]["lut"]

timeres_sim = rt_config["timeres"]

results_path = os.path.join(rt_config["save_path"]["disort"],
                                rt_config["save_path"]["optical_depth_lut"])
savepath = generate_results_folders(rt_config,sensor_type,results_path,homepath)


for key in pvsys:    
    print('Simulating clear sky radiative transfer for %s using DISORT and wavelength range %s' 
                      % (key,rt_config['common_base']['wavelength']["pv"]))
    print(f'SZA limit is {sza_limit}')
    print(f'Diffuse radiance grid resolution is {disort_res}')
    if rt_config["disort_base"]["pseudospherical"]:
        print('Results are with pseudospherical atmosphere')
    else:
        print('Results are with plane-parallel atmosphere')
    print('Molecular absorption is calculated using %s' % rt_config["common_base"]["mol_abs_param"])
                
    for measurement in pvrad_config["inversion_source"]:    
        year = "mk_" + measurement.split('_')[1]        
        name_df_sim = 'df_sim_' + year.split('_')[-1]        
        
        if name_df_sim in pvsys[key]:  
            lrt_dir = os.path.join(homepath,rt_config["working_directory"])                    
                                
            if rt_config['atmosphere'] == 'cosmo':
                cosmo_path = pvsys[key]['path_cosmo_lrt']
                #temporary fix
                # cosmo_path = '/'.join(cosmo_path.split('/')[0:-1]) + '/reduced_levels/cosmo_d2_2km/'\
                #             + cosmo_path.split('/')[-1]                                
            else:
                cosmo_path = ''
            
            modis_path = os.path.join(homepath,rt_config['albedo']['brdf_folder']) #,key)  #Turned off for now - same MODIS file for all!      
            
            if rt_config['albedo']['choice'] == 'MODIS':
                albedo_series = pvsys[key]['albedo_series']
            else:
                albedo_series = ()                    
                    
            #Select clear sky days
            dfs = []
            for day in rt_config["test_days"][year]["clear"]:
                if day.strftime('%Y-%m-%d') in pd.to_datetime(pvsys[key]
                [name_df_sim].index.date).unique().strftime('%Y-%m-%d'):            
                    dfs.append(pvsys[key][name_df_sim].loc[day.strftime('%Y-%m-%d')])
            
            df_temp = pd.concat(dfs,axis=0)
            df_temp = df_temp.loc[df_temp[('sza','sun')] <= sza_limit]
              
            print(f'Clear days for {year} are {[day.strftime("%Y-%m-%d") for day in rt_config["test_days"][year]["clear"]]}')
            
            df_aod = "df_aod_"+ year.split('_')[-1]
            pvsys[key][df_aod] = disort_simulation_aod(df_temp,
                                 rt_config,grid_dict,
                                 angle_arrays,albedo_series,
                                 modis_path,cosmo_path,lrt_dir,sensor_type,homepath) #cloudfile,
                    
            save_results(key,pvsys[key],measurement,df_aod,rt_config,pvcal_config,
                      pvrad_config,savepath)
        
    