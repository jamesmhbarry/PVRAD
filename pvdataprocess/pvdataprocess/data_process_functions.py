#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 12:00:24 2019

@author: james
"""

import os
import numpy as np
import pandas as pd
import time
import datetime as dt
from copy import deepcopy
import re


def list_dirs(path):
    """
    list all directories in a given directory
    
    args:
    :param path: string with the path to the search directory
    
    out:
    :return: all directories within the search directory
    """
    dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path,d))]
    return sorted(dirs)

def list_files(path):
    """
    lists all filenames in a given directory
    args:
    :param path: string with the path to the search directory
    
    out:
    :return: all files within the search directory
    """
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path,f))]
    return sorted(files)        

def extract_station_info(info,paths):
    """
    

    Parameters
    ----------
    info : dataframe
        dataframe with information about each station read in from Excel
    paths : dictionary
        dictionary from config file with paths for each type of data 

    Returns
    -------
    paths : dictionary
        dictionary from config file with paths for each type of data, updated with stations
    pv_systems : dictionary
        dictionary with each PV system, that will in the end contain all information and data

    """    
    
    pv_systems = {sys : {} for sys in info.index}
            
    #Import latitude and longitude and check for commas!!
    for key in pv_systems:
        pv_systems[key].update({'lat_lon':[info.loc[key].Breitengrad,info.loc[key].Laengengrad]})
        for i in range(2):
            if type(pv_systems[key]['lat_lon'][i]) == str:
                pv_systems[key]['lat_lon'][i] = float(pv_systems[key]
                ['lat_lon'][i].replace(',','.'))
        pv_systems[key]['lat_lon'] = [np.round(coord,6) for coord in pv_systems[key]['lat_lon']]    
    
    if 'auew' in paths:
        paths['auew'].update({'stations':[]})
        for station in info.index:
            if info.loc[station].Lastmessung_AUEW == 'ja':
                paths['auew']['stations'].append(station)
            
    if 'egrid' in paths:
        paths['egrid'].update({'stations':[]})
        for station in info.index:
            if info.loc[station].egrid_Messbox == 'ja':
                paths['egrid']['stations'].append(station)                
                
    if 'solarwatt' in paths:
        paths['solarwatt'].update({'stations':[]})
        for station in info.index:
            if info.loc[station].PV_Messung_DC == 'ja':
                paths['solarwatt']['stations'].append(station)                
                
    if 'inverter' in paths:
        paths['inverter'].update({'stations':[]})
        for station in info.index:
            if info.loc[station].PV_Messung_DC == 'ja':
                paths['inverter']['stations'].append(station)                
    
    return paths, pv_systems
        
def convert_wrong_format (input_value):
    """
    Convert values with two decimal points in the CSV file
    """
    if type(input_value) == str:
        if input_value:
            split_string = input_value.split(".")
            
            if len(split_string) == 2:
                value = float(input_value)
            elif len(split_string) == 3:
                value = float(".".join(("".join(split_string[0:2]),split_string[-1])))
        else:
            value = np.nan
    else:
        value = input_value
        
    return value

def filter_suntracker_time (value):
    """
    Special filter for the suntracker data

    Parameters
    ----------
    value : string
        string to be filtered

    Returns
    -------
    new_time : string
        filtered string

    """
    
    new_time = value.split('-')[2] + ':' + value.split('-')[1] +\
        ':' + value.split('-')[0][0:2]
                
    return new_time

def filter_suntracker_irrad (value):
    """
    Filter for the irradiance data from suntracker

    Parameters
    ----------
    value : string
        irradiance value as string

    Returns
    -------
    new_irrad : float
        irradiance filtered

    """
    
    new_irrad = float(value.split('-')[0][2:])
                
    return new_irrad

def downsample(dataframe,old_period,new_period):
    """
    Downsample the data to required frequency
    
    args:
    :param dataframe: Dataframe to be resampled
    :param old_period: timedelta, old period given in seconds
    :param new_period: timedelta, new period given in seconds
        
    out:
    :return: resampled dataframe
    """
    #Check whether there are gaps in the data
    t_delta_max = dataframe.index.to_series().diff().max() #.round('1s')    

    #If some parts of series are different, upsample them!    
    if t_delta_max > old_period:
        start_ts = dataframe.first_valid_index().round(old_period)
        end_ts = dataframe.last_valid_index().round(old_period)
        #Resample at higher frequency        
        newindex = pd.date_range(freq=old_period/2,start=start_ts,end=end_ts)
        df_upsample = dataframe.reindex(newindex.union(dataframe.index)).interpolate('linear')
        
        #Go back to desired frequency
        newindex = pd.date_range(freq=old_period,start=newindex[0],end=newindex[-1])
        #Make sure we don't have extra entries from tomorrow!
        newindex = newindex[newindex.date==newindex[0].date()]
        df_upsample = df_upsample.reindex(newindex)
    else:
        df_upsample = dataframe    
    
    #Calculate number of periods to shift by
    shift_periods = int(new_period/old_period/2)
    
    #Shift to the left, resample to new period (with the mean), shift back to the right
    df_rs = df_upsample.shift(-shift_periods).resample(new_period).mean().shift(1)   
    
    return df_rs

def interpolate(dataframe,new_period):
    """
    Interpolate data by upsampling
    
    args:
    :param dataframe: pandas dataframe to be resampled
    :param new_period: timedelta, new period for resampling
    
    out:
    :return: dataframe with resampled data
    """        
    
    df_upsample = dataframe.resample(new_period).interpolate('linear')
    
    return df_upsample

def shift_auew_data(dataframe,config):    
    """
    Shift AÜW data to take into account for the fact that the measured values are
    actually the moving average power values of the last 15 minutes, or in other words
    the energy is counted every 15 minutes and the power is simply assigned to the end
    time stamp
    
    args:
    :param dataframe: dataframe to be shifted
    :param config: dictionary with information about time shift
    
    out:
    :return: modified dictionary
    """
    
    timeres = config["time_integral"]
    units = config["units"]
    
    #Convert new time resolution to a timedelta object
    if units == "seconds" or units == "s" or units == "secs" or units == "sec":
        t_res_new = pd.Timedelta(int(timeres/60),'m')
    elif units == "minutes" or units == "m" or units == "mins" or units == "min":
        t_res_new = pd.Timedelta(int(timeres),'m')
        
    if t_res_new.components.minutes in dataframe.index.minute:                    
        t_half = str(t_res_new.components.minutes/2.)
        
        #Create new index to take averaging into account
        shifted_index = dataframe.index - pd.Timedelta(t_half + 'm')
        df_shifted = dataframe.reindex(shifted_index,method='bfill')
        
        #Resample data at double frequency and linearly interpolate
        df_rs = df_shifted.resample(t_half + 'T').interpolate('linear')
                            
        #Put new values into dataframe                    
        df_new = df_rs.reindex(dataframe.index)
                        
    return df_new

def shift_module_temperature_data(dataframe, config):
    """
    This function corrects the time shift in the module temperature data
    
    args:
    :param dataframe: Dataframe with module temperature data     
    :param config: dictionary with configuration for timeshift
                   slope: float, slope of the time correction in [s/s]    
                   start_time: string, start time at which it is assumed 
                   that the time is synchronised
    
    out:
    :return: dataframe with corrected temperature data
    """
    
    t_delta = dataframe.index.to_series().diff()
    resolution = int(t_delta.min().round('1s').total_seconds())
    
    slope = config["slope"]
    start_time = config["start_time"]
    
    data_start_datetime = dataframe.index[0]
    time_synch_datetime = pd.to_datetime(start_time)
    time_delta = (data_start_datetime - time_synch_datetime).total_seconds()
    offset = slope*time_delta
        
    # Time correction
#    if round(resolution) != resolution:
#        print("Fehler. Die zeitliche Auflösung lautet nicht auf volle Sekunden.")
    j_vec = np.arange(len(dataframe))
    f = dt.timedelta(seconds = slope)*resolution*j_vec + \
    dt.timedelta(seconds = offset)    
  
    td_idx = pd.TimedeltaIndex(data = f)
    td_idx_round = td_idx.round('1s')
    
    index_shifted = dataframe.index + td_idx_round    
    df_corrected = pd.DataFrame(data=dataframe.values, index=index_shifted,
                                columns=dataframe.columns)
                                
    # Durchführen der Interpolation und Anpassen an das ursprüngliche Zeitgit-
    # ter ohne Verwenden eines Zwischengitters
    df_corrected_rs = \
    df_corrected.reindex(df_corrected.index.union(dataframe.index)).interpolate('index').\
    reindex(dataframe.index)
        
    return df_corrected_rs

def resample_interpolate_merge(raw_data,station,timeres,process_config,datatypes):
    """
    Resample dataframe to the required resolution, as set in the config file,
    then merge all dataframes into one
    
    args:
    :param raw_data: dictionary with raw data separated into types
    :param station: string, name of station
    :param timeres: dictionary with required time resolution for further simulation    
    :param process_config: dictionary with information for data processing
    :param datatypes: list with different data types 
    
    out:
    :return: dictionary of PV systems with resampled data
    """
    
#    #Create a copy of the dictionary        
    raw_data_rs = deepcopy(raw_data) #.copy()
    del raw_data
#    
    if timeres != "raw":
        t_res_new = pd.to_timedelta(timeres)
    
    #Go through all the data and resample, interpolate, merge    
    print(("Processing data from %s" % station))
    #This will be the merged dataframe with all values averaged to the same time stamp
    df_merge = pd.DataFrame()
    #Loop through the datatypes
    for idata in datatypes:
        if idata in raw_data_rs and raw_data_rs[idata]:
            #This is a list of dataframes, one per sensor of one data type
            #resampled to the desired resolution
            dfs_total_time = []
            #Dictionary for full dataframes, to be added to main dictionary later (for completeness, 31.07.2019)
            dict_full = {}
            #Loop through the substations of one datatype
            for i, substat in enumerate(raw_data_rs[idata]):
                if raw_data_rs[idata][substat]:                        
                    dfs_total_time.append(pd.DataFrame())  
                    
                    #Full dataframe for one sensor, except duplicates or wrong time stamps
                    df_full = pd.concat(raw_data_rs[idata][substat][1],axis=0)
                    
                    #Remove duplicates
                    if df_full.index.duplicated().any():
                        df_full = df_full[~df_full.index.duplicated()]
                        
                    #Check for nonsensical timestamps that occur in the wrong place
                    t_delta = df_full.index.to_series().diff()
                    idx_negative = df_full[t_delta < pd.Timedelta(0)].index
                    for idx in idx_negative:
                        int_idx = df_full.index.get_loc(idx)
                        if df_full.index[int_idx] - df_full.index[int_idx - 2] > pd.Timedelta(0):
                            df_full.drop(df_full.index[int_idx - 1],inplace=True)                            
                        else:
                            df_full.drop(df_full.index[int_idx],inplace=True)                            
                    
                    #Round timestamps to nearest second
                    df_full = df_full.reindex(df_full.index.round('S'),
                                              method='nearest')
                    
                    #Full dataframe of all values
                    dict_full.update({'df_' + substat:df_full})
                    
                    #Redefine lists of dataframes after duplicates have been removed
                    dfs = [group[1] for group in df_full.groupby(df_full.index.date)]
                    days = pd.to_datetime([group[0] for group in df_full.groupby(df_full.index.date)])
                    
                    raw_data_rs[idata][substat] = (days,dfs)
                    
                    if timeres != "raw":
                        #Iterate through days to interpolate / resample
                        for ix, iday in enumerate(raw_data_rs[idata][substat][0]):                            
                            dataframe = raw_data_rs[idata][substat][1][ix]
                            if len(dataframe) > int(pd.Timedelta('1D')/t_res_new/100):
                            
                                #Check for duplicates and throw away if necessary
                                if dataframe.index.duplicated().any():
                                    dataframe = dataframe[~dataframe.index.duplicated()]
                                    #This would shift the data, but is not general enough!
                                    #t_shift = pd.Timedelta(t_delta_mean)*dataframe.index.duplicated()
                                    #new_index = dataframe.index + t_shift
                                    #dataframe = pd.DataFrame(index=new_index,data=dataframe.values,
                                    #                         columns=dataframe.columns)                                                    
                                
                                dataframe.sort_index(axis=0,inplace=True)
                                
                                #Check for nonsensical timestamps that occur in the wrong place
                                t_delta = dataframe.index.to_series().diff()
                                idx_negative = dataframe[t_delta < pd.Timedelta(0)].index
                                for idx in idx_negative:
                                    int_idx = dataframe.index.get_loc(idx)
                                    if dataframe.index[int_idx] - dataframe.index[int_idx - 2] > pd.Timedelta(0):
                                        dataframe.drop(dataframe.index[int_idx - 1],inplace=True)    
                                    else:
                                        dataframe.drop(dataframe.index[int_idx],inplace=True)    
                                t_delta = dataframe.index.to_series().diff()
                                
                                #Check if there are big gaps in the data
                                #t_delta_max = dataframe.index.to_series().diff().max().round('1s')
                                #time_max = dataframe.index.to_series().diff().idxmax()
                                
                                #If more than one hour in the day time is missing, throw away the day of data
                                #if t_delta_max < pd.Timedelta(1,'h') or time_max.hour > 19 or time_max.hour < 3:                                                        
                                
                                #Find the frequency of the dataframe                                
                                t_delta_min = t_delta.min() #.round('1s')
                                
                                #SHIFT AUEW data by 15 minutes!!
                                if 'auew' in substat:
                                    dataframe = shift_auew_data(dataframe,process_config["auew"])
                                    #print('AUEW data for %s, %s shifted' % (station,substat))
                                                            
                                #Resampling and interpolation                            
                                if t_delta_min != t_res_new:
                                    try:
                                        if t_delta_min < t_res_new:
                                            try: 
                                                dataframe = downsample(dataframe,t_delta_min,t_res_new)
                                            except:
                                                print(('error in data from %s, %s on %s' % (station,substat,iday)))
                                        elif t_delta_min > t_res_new:
                                            dataframe = interpolate(dataframe,t_res_new)
                                    except:
                                        print(('error %s, %s, %s' % (station,substat,iday)))
                                else:
                                    #Check if timestamps are correct
                                    new_index = pd.date_range(start=dataframe.index[0].round(timeres),
                                                  end=dataframe.index[-1].round('T'),freq=timeres)
                                    dataframe = dataframe.reindex(new_index,method='nearest').loc[iday.strftime('%Y-%m-%d')]
                                dfs_total_time[i] = pd.concat([dfs_total_time[i],dataframe],axis=0)                                                                                                
                                
                            else: print(('Data has less than 1/100 of a day, throwing away %s' % iday.date()))
                                
                        #Create Multi-Index
                        if type(dfs_total_time[i].columns) != pd.MultiIndex:
                            col_index = pd.MultiIndex.from_product([dfs_total_time[i].columns.values.tolist(),[substat]],
                                                                    names=['variable','substat'])
                            dfs_total_time[i].columns = col_index
                        
                        if process_config:
                            #Shift module temperature data by a specific time shift
                            if "PV-Modul_Temperatursensor" in substat and\
                            station in process_config["module_temp"] and\
                            process_config["module_temp"][station]["flag"]:
                                dfs_total_time[i] = shift_module_temperature_data(dfs_total_time[i],
                                              process_config["module_temp"][station])
                                print(('Module temperature data for %s, %s shifted' % (station,substat)))
                    
            #Concatenate different substations
            if len(dfs_total_time) > 1:
                #Create multiindex, with substation and variable
                df_total = pd.concat(dfs_total_time,axis=1)                
                #,keys=raw_data_rs[idata].keys(),
                                     #names=['substat','variable'])
                #df_total.columns = df_total.columns.swaplevel(0,1)
            else:
                df_total = dfs_total_time[0]
                
            if type(df_total.columns) != pd.MultiIndex: 
                df_total.columns = pd.MultiIndex.from_product([df_total.columns.values.tolist(),[substat]],
                                                                   names=['variable','substat'])                
            
            #This is to rename a column since substation name changed between campaigns!
            if process_config:
                for substat in raw_data_rs[idata]:
                    if "substat_switch" in process_config and \
                    station in process_config["substat_switch"] and substat ==\
                    process_config["substat_switch"][station]["old_name"]:
                        oldname = process_config["substat_switch"][station]["old_name"]
                        newname = process_config["substat_switch"][station]["new_name"]
                        df_total.rename(columns={oldname:newname},
                                        level='substat',inplace=True)
                        print(('Substation name changed from %s to %s for %s' % (oldname,newname,station)))
            
            #Merge dataframes into one
            if df_merge.empty:
                df_merge = df_total
                #Added this for Spyder bug but not sure if it is a good idea...
                #If the station has only one datatype then this will drop the Nans
                df_merge.dropna(axis=0,how='all',inplace=True)
            else:
                #Here there should be no rows with only Nans, except those from interpolation
                df_merge = pd.merge(df_merge,df_total,how="outer",left_index=True,right_index=True)
                
            #This overwrites the dictionaries of tuples, now a dictionary of long dataframes for each sensor
            raw_data_rs[idata] = dict_full
        else: 
            del raw_data_rs[idata]              
       
    print(("Data from %s has been resampled to %s" % (station,timeres)))
    
    return df_merge, raw_data_rs
    
def load_pv_data(pv_systems,info,paths,description,process):
    """
    Load PV power data into dataframe
    
    args:
    :param pv_systems: dictionary of PV systems
    :param info: dictionary with information about PV systems
    :param paths: dictionary of paths
    :param description: string, description of measurement campaign
    :param process: dictionary with process configuration
    
    out:
    :return: dictionary of PV systems
    
    """    
    
    for station in info.index:
        station_dirs = list_dirs(os.path.join(paths['mainpath'],station))    
        
        pv_systems[station]['pv'] = {}
        pv_systems[station]['irrad'] = {}
        pv_systems[station]['temp'] = {}
        for substat in station_dirs:
            #read in 15 minute PV data                          
            if 'auew' in paths and substat == paths['auew']['path']:                
                path = os.path.join(paths['mainpath'],station,paths['auew']['path'])
                files = list_files(path)
                
                if not files:
                    print(('Station %s has no AUEW power data' % station))
                    paths['auew']['stations'].remove(station)
                else:
                    if "2018" in description:
                        #Only use first column with MEZ            
                        dfs_all = [pd.read_csv(os.path.join(path,ifile),header=None,sep=';',usecols=[0,2],index_col=0,
                                  skiprows=6,names=['Timestamp','P_kW'],converters={'P_kW':convert_wrong_format}) 
                                 for ifile in files]                  
                    elif "2019" in description:
                        dfs_all = [pd.read_csv(os.path.join(path,ifile),header=None,sep=';',usecols=[0,2],index_col=0,
                                  skiprows=6,names=['Timestamp','P_kW']) 
                                 for ifile in files]                  
                    
                    #If there are several files create new substation dictionaries
                    for i, dataframe in enumerate(dfs_all):
                        substat_name = 'auew_' + str(i + 1)
                        pv_systems[station]['pv'][substat_name] = ()
                        
                        dataframe.index = pd.to_datetime(dataframe.index,format='%d.%m.%Y %H:%M:%S')
                        for cols in dataframe.columns:
                            dataframe[cols] = convert_wrong_format(dataframe[cols].values)
                                        
                            #Shift to UTC, data files are given in CET (only first column)
                            dataframe.index = dataframe.index - pd.Timedelta(hours=1)
                    
                        #get list of unique days
                        dfs = [group[1] for group in dataframe.groupby(dataframe.index.date)]
                        days = pd.to_datetime([group[0] for group in dataframe.groupby(dataframe.index.date)])
                    
                        pv_systems[station]['pv'][substat_name] = (days,dfs)

                    print(('15 minute PV power data from station %s successfully imported' % station))
        
            #read in 1s PV data
            if 'egrid' in paths and substat == paths['egrid']['path']:
                dfs = []
                print(("Importing 1s PV data from %s, please wait....." % station))
                
                path = os.path.join(paths['mainpath'],station,paths['egrid']['path'])
                files = list_files(path)
                dirs = list_dirs(path)
                
                if not files:
                    if not dirs:
                        print(('Station %s has no egrid power data' % station))
                        paths['egrid']['stations'].remove(station)
                    else:
                        for wr in dirs:
                            substat_name = 'egrid_' + wr[-1]
                            pv_systems[station]['pv'][substat_name] = ()
                            files = list_files(os.path.join(path,wr))
                            if not files:
                                print(('Station %s, %s has no egrid power data' % (station,wr)))    
                            else:
                                dfs = [pd.read_csv(os.path.join(path,wr,ifile),header=0,sep=',',
                                        index_col=0,comment='#',
                                        names=['Timestamp','P_W']) for ifile in files]
                                dataframe = pd.concat(dfs,axis='index')
                                dataframe.index = pd.to_datetime(dataframe.index,format='%Y.%m.%d %H:%M:%S')

                                #Throw away nonsense data with wrong year
                                if "2018" in description:
                                    dataframe = dataframe[dataframe.index.year == 2018]
                                elif "2019" in description:
                                    dataframe = dataframe[dataframe.index.year == 2019]
                                        
                                dataframe['P_kW'] = dataframe.P_W/1000
                                dataframe.drop(['P_W'],axis=1,inplace=True)
                    
                                #get list of unique days                    
                                dfs = [group[1] for group in dataframe.groupby(dataframe.index.date)]
                                days = pd.to_datetime([group[0] for group in dataframe.groupby(dataframe.index.date)])
                                
                                pv_systems[station]['pv'][substat_name] = (days,dfs)                                        
                                print(('1 second PV power data from station %s, %s successfully imported' % (station,wr)))   
                            
                else:
                    pv_systems[station]['pv']['egrid'] = ()
                    dfs = [pd.read_csv(os.path.join(path,ifile),header=0,sep=',',index_col=0,
                              comment='#',names=['Timestamp','P_W']) for ifile in files]
                    dataframe = pd.concat(dfs,axis='index')
                    dataframe.index = pd.to_datetime(dataframe.index,format='%Y.%m.%d %H:%M:%S')
                                        
                    dataframe['P_kW'] = dataframe.P_W/1000
                    dataframe.drop(['P_W'],axis=1,inplace=True)
                    
                    #get list of unique days                    
                    dfs = [group[1] for group in dataframe.groupby(dataframe.index.date)]
                    days = pd.to_datetime([group[0] for group in dataframe.groupby(dataframe.index.date)])
                    
                    pv_systems[station]['pv']['egrid'] = (days,dfs)
                    print(('1 second PV power data from station %s successfully imported' % station)) 
            
            #read in 1s PV data
            if 'solarwatt' in paths and substat == paths['solarwatt']['path']:
                print(("Importing Solarwatt PV data from %s, please wait....." % station))                
                
                path = os.path.join(paths['mainpath'],station,paths['solarwatt']['path'])
                
                files = list_files(path)
                
                if "2018" in description:
                    dfs = [pd.read_csv(os.path.join(path,ifile),header=0,sep='|',index_col=0)
                            for ifile in files]
                    dataframe = pd.concat(dfs,axis=1)
                    try:
                        dataframe.index = pd.to_datetime(dataframe.index,format='%Y.%m.%d %H:%M:%S')                
                    except TypeError:
                        print("Wrong datetime format")                        
                        
                    dataframe.index.rename('Timestamp',inplace=True)
                    
                    #Shift data to UTC
                    dataframe.index = dataframe.index - pd.Timedelta(hours=2)                    
                    
                    #Change label to P_kW
                    dataframe['P_kW'] = dataframe.P_PV/1000
                    dataframe.drop(['P_PV'],axis=1,inplace=True)
                elif "2019" in description:
                    dfs = [pd.read_csv(os.path.join(path,ifile),header=0,sep=',',index_col=0,na_values=(''))
                            for ifile in files]
                    dataframe = pd.concat(dfs,axis=0)
                                            
                    dataframe.index = pd.to_datetime(dataframe.index,format='%Y-%m-%d %H:%M:%S',
                                             errors='coerce')            
                    
                    #Set timezone, times early on last Sunday in October will be ambiguous - marked as nat
                    dataframe.index = dataframe.index.tz_localize(tz='Europe/Berlin',
                                                          ambiguous='NaT')
                    #Convert data to UTC
                    dataframe.index = dataframe.index.tz_convert('UTC')
                    
                    
                    #Change label to P_kW
                    dataframe['P_kW'] = dataframe["V_PV"]*dataframe["I_PV_filtered"]/1000
                    dataframe.drop(['P_PV'],axis=1,inplace=True)
                    
                    dataframe['Idc_A'] = dataframe["I_PV_filtered"]
                    dataframe.drop(['I_PV_filtered'],axis=1,inplace=True)
                elif "2021" in description:
                    dfs = [pd.read_csv(os.path.join(path,ifile),header=0,sep=',',index_col=0,na_values=(''))
                            for ifile in files]
                    dataframe = pd.concat(dfs,axis=0)
                                            
                    dataframe.index = pd.to_datetime(dataframe.index,format='%Y-%m-%d %H:%M:%S',
                                             errors='coerce')            
                    
                    #Set timezone, times early on last Sunday in October will be ambiguous - marked as nat
                    dataframe.index = dataframe.index.tz_localize(tz='UTC',
                                                          ambiguous='NaT')                    
                    
                    #Change label to P_kW
                    dataframe['P_kW'] = dataframe["VPV"]*dataframe["IPV"]/1000
                    #dataframe.drop(['P_PV'],axis=1,inplace=True)
                    
                    dataframe['Idc_A'] = dataframe["IPV"]
                    dataframe.drop(['IPV'],axis=1,inplace=True)                                                                                                                         
                       
                if "2021" not in description:                
                    days = pd.to_datetime([group[0] for group in dataframe.groupby(dataframe.index.date)])
                    dfs = [group[1] for group in dataframe.groupby(dataframe.index.date)]
        
                    pv_systems[station]['pv']['myreserve'] = (days,dfs)
                else:
                    dataframe.rename(columns={"GHI":"Etotdown_RT1_Wm2","GTI":"Etotpoa_RT1_Wm2"},inplace=True)
                    dataframe.rename(columns={"T_module":"T_module_C","T_ambient":"T_ambient_C"},inplace=True)
                    
                    df_pv = dataframe[["P_kW","Idc_A","VPV","IBat","VBat","SoC"]]
                    df_rad = dataframe[["Etotdown_RT1_Wm2","Etotpoa_RT1_Wm2"]]                    
                    df_temp = dataframe[["T_module_C","T_ambient_C"]]                    
                    
                    days = pd.to_datetime([group[0] for group in df_pv.groupby(df_pv.index.date)])
                    dfs = [group[1] for group in df_pv.groupby(df_pv.index.date)]
        
                    pv_systems[station]['pv']['myreserve'] = (days,dfs)
                    
                    days = pd.to_datetime([group[0] for group in df_rad.groupby(df_rad.index.date)])
                    dfs = [group[1] for group in df_rad.groupby(df_rad.index.date)]
        
                    pv_systems[station]['irrad']['RT1'] = (days,dfs)
                    
                    days = pd.to_datetime([group[0] for group in df_temp.groupby(df_temp.index.date)])
                    dfs = [group[1] for group in df_temp.groupby(df_temp.index.date)]
        
                    pv_systems[station]['temp']['RT1'] = (days,dfs)
                
            if 'inverter' in paths and substat == paths["inverter"]["path"]:
                print(('Importing inverter data from %s, please wait.....' % station))
                
                path = os.path.join(paths['mainpath'],station,paths['inverter']['path'])
                
                files = list_files(path)
                
                dfs = [pd.read_csv(os.path.join(path,filename),sep=';',header=0)
                for filename in files if "min" in filename]
        
                dataframe = pd.concat(dfs,axis='index')
                dataframe.index = pd.to_datetime(dataframe.iloc[:,0] + ' ' + 
                                    dataframe.iloc[:,1],format='%d.%m.%y %H:%M:%S')
                dataframe.drop(columns=["#Datum","Uhrzeit"],inplace=True)
                              
                #Set timezone, times early on last Sunday in October will be ambiguous - marked as nat
                #dataframe.index = dataframe.index.tz_localize(tz='Europe/Berlin',ambiguous='NaT')
                              
                #Convert data to UTC
                dataframe.index = dataframe.index - pd.Timedelta(hours=2)
                #dataframe.index = dataframe.index.tz_convert('UTC')
        
                #Sort index since data is back to front
                dataframe.sort_index(inplace=True)
                
                dataframe = dataframe.filter(regex='^Pac|^Pdc|^Udc', axis=1)
                
                #Make multiindex and combine inverters in the correct way
                dfs = []
                inverters = process["inverters"][station]["names"]
                n_phase = process["inverters"][station]["phases"]
                n_wr = len(inverters)
                n_cols = len(dataframe.columns)/n_wr
                
                for ix, inv in enumerate(inverters):                
                    dfs.append(pd.DataFrame)
                    old_columns = dataframe.columns[int(n_cols*ix):int(n_cols*(ix+1))].values.tolist()
                    new_columns = []

                    #This is the case of KACO inverters where there are actually only 3 inverters
                    if n_phase == 3:
                        for name in old_columns:    
                            if name.split('.')[0] == "Pdc1":
                                new_columns.append(name.split('.')[0][0:-1] + '_' + str(ix + 1))                             
                            else:
                                new_columns.append(name.split('.')[0] + '_' + str(ix + 1))                             
                    
                        dfs_inv = []
                        for k, inv in enumerate(inverters):
                            dfs_inv.append(pd.DataFrame)
                            col_index =pd.MultiIndex.from_product([new_columns[int(n_cols/n_wr*k)
                                       :int(n_cols/n_wr*(k+1))],[inv]],names=['variable','substat'])
                            dfs_inv[k] = dataframe.iloc[:,int(n_cols*ix+n_cols/n_wr*k)
                                      :int(n_cols*ix+n_cols/n_wr*(k+1))]
                            dfs_inv[k].columns = col_index
                            
                        dfs[ix] = pd.concat(dfs_inv,axis='columns')
                    
                    #This is the case where there are simply 9 inverters along the columns
                    elif n_phase == 1:
                        for name in old_columns:
                            new_columns.append(name.split('.')[0])
                            
                        col_index =pd.MultiIndex.from_product([new_columns,[inv]],names=['variable','substat'])
                        dfs[ix] = dataframe.iloc[:,int(n_cols*ix):int(n_cols*(ix+1))]
                        dfs[ix].columns = col_index                        
                    
                dataframe = pd.concat(dfs,axis='columns')
                
                #Sort multi-index (makes it faster)
                dataframe.sort_index(axis=1,level=1,inplace=True)
                
                #Add each inverter to a separate tuple in dictionary
                for inv in inverters:
                    if n_phase == 3:
                        #Calculate sum of three phase power
                        dataframe[('P_kW',inv)] = dataframe.loc[:,
                        pd.IndexSlice[['Pac_1','Pac_2','Pac_3'],inv]].sum(axis='columns')/1000.
                        
                        #Calculate DC current
                        for nstring in range(int(n_cols/n_wr)):
                            dataframe[('Idc_' + str(nstring+1),inv)] =\
                            dataframe[('Pdc_' + str(nstring+1),inv)]/\
                            dataframe[('Udc_' + str(nstring+1),inv)]                
                        
                    elif n_phase == 1:
                        dataframe[('P_kW',inv)] = dataframe[('Pac',inv)]/1000.

                        for nstring in range(int((n_cols - 1)/2)):
                            dataframe[('Idc' + str(nstring+1),inv)] =\
                            dataframe[('Pdc' + str(nstring+1),inv)]/\
                            dataframe[('Udc' + str(nstring+1),inv)]                
         
                    dataframe.sort_index(axis=1,level=1,inplace=True)
                    df_inv = dataframe.loc[:,pd.IndexSlice[:,inv]]
                
                    days = pd.to_datetime([group[0] for group in df_inv.groupby(df_inv.index.date)])
                    dfs = [group[1] for group in df_inv.groupby(df_inv.index.date)]
    
                    pv_systems[station]['pv'][inv] = (days,dfs)
                    print(('5 minute inverter data from station %s, inverter %s successfully imported' % (station,inv))) 
                    
    print('All PV power data imported\n')
    return pv_systems

def load_rad_data (pv_systems,info,paths,description):
    """
    Load irradiance data into dataframe
    
    args:
    :param pv_systems: dictionary of PV systems
    :param info: dictionary with information about PV systems
    :param paths: dictionary of paths
    :param description: string, description of measurement campaign
    
    out:
    :return: dictionary of PV systems
    
    """    
    
    for station in info.index:
        mainpath = os.path.join(paths['mainpath'],station)
        station_dirs = list_dirs(mainpath)  
        
        if "irrad" not in pv_systems[station]:
            pv_systems[station]['irrad'] = {}
        for substat in station_dirs:            
            if "Pyr" in substat and "old" not in substat:                
                pv_systems[station]['irrad'][substat] = ()
                print(("Importing pyranometer data from %s, %s, please wait....." % (station,substat)))
                
                rad_files =  list_files(os.path.join(mainpath,substat))
                if not rad_files:
                    print(('Substation %s at station %s has no radiation data' % (substat,station)))
                    del pv_systems[station]['irrad'][substat]

                else:
                    #Go through the data files (one for each day) and import to a list
                    dfs = [pd.read_csv(os.path.join(mainpath,substat,filename)
                                       ,header=None,sep='\s+',comment='#',usecols=[0,1,2,3,4]) for filename in rad_files]
                    
                    #Concatenate list 
                    dataframe = pd.concat(dfs,axis=0)                    
                    
                    #Set index to be in datetime object
                    dataframe.index = pd.to_datetime(dataframe[0] + ' ' + dataframe[1],format='%Y.%m.%d %H:%M:%S')
                    dataframe.drop(columns=[0,1],inplace=True)
                    
                    #Name columns
                    dataframe.rename(columns={2:'Etotdown_pyr_Wm2',3:'Etotpoa_pyr_Wm2',4:'T_amb_pyr_K'},inplace=True)
                    dataframe['T_amb_pyr_K'] = dataframe['T_amb_pyr_K'] - 273.15
                    dataframe.rename(columns={'T_amb_pyr_K':'T_ambient_pyr_C'},inplace=True)                    
                    
                    #get list of unique days                    
                    dfs = [group[1] for group in dataframe.groupby(dataframe.index.date)]
                    days = pd.to_datetime([group[0] for group in dataframe.groupby(dataframe.index.date)])
 
                    pv_systems[station]['irrad'][substat] = (days,dfs)    

                    print(('Pyranometer data from station %s, substation %s successfully imported' % (station,substat)))
            
            if "Sun-Tracker" in substat:
                pv_systems[station]['irrad']['suntracker'] = ()
                print(("Importing data from %s, %s, please wait....." %(station,substat)))
                
                rad_files =  list_files(os.path.join(mainpath,substat))
                if not rad_files:
                    print(('Substation %s at station %s has no radiation data' % (substat,station)))
                    del pv_systems[station]['irrad']['suntracker']

                else:
                    #Go through the data files (one for each day) and import to a list
                    if '2018' in description:
                        #Go through the data files (one for each day) and import to a list
                        dfs = [pd.read_csv(os.path.join(mainpath,substat,filename),
                                       header=None,sep=';',comment='#',usecols=[0,1,2,3,4,5,6,7],
                                       names=['Date','Time','Etotdown_CMP11_Wm2','Ediffdown_CMP11_Wm2',
                                              'Etotdown_SP2Lite_Wm2','Edirnorm_CHP1_Wm2','T_pyrhel_C',
                                              'T_ambient_suntrack_C']) for filename in rad_files]
                    elif '2019' in description:
                        #Adding index_col = False fixes problems with delimiters at the end of the line
                        dfs = [pd.read_csv(os.path.join(mainpath,substat,filename),
                                       header=None,sep=';',comment='#',index_col=False,
                                       names=['Date','Time','Etotdown_CMP11_Wm2','Ediffdown_CMP11_Wm2',
                                              'T_module_upper_C','Etotdown_SP2Lite_Wm2','T_module_lower_C',
                                              'Edirnorm_CHP1_Wm2','T_pyrhel_C','T_ambient_suntrack_C']) for filename in rad_files]
                    
                    #Concatenate list 
                    dataframe = pd.concat(dfs,axis=0)                    
                    
                    #for dataframe in dfs:
                    
                    #Set index to be in datetime object
                    dataframe.index = pd.to_datetime(dataframe.Date + ' ' + dataframe.Time,errors='coerce',format='%Y.%m.%d %H:%M:%S')
                                        
                    #Shift values to the right
                    dataframe.iloc[pd.isnull(dataframe.index),2:8] = dataframe.iloc[pd.isnull(dataframe.index),2:8].shift(1,axis=1)
                    
                    #Extract irradiance from string
                    dataframe.loc[pd.isnull(dataframe.index),'Etotdown_CMP11_Wm2'] = dataframe.loc[pd.isnull(dataframe.index),'Time'].apply(filter_suntracker_irrad)
                    
                    #Extract time from string
                    dataframe.loc[pd.isnull(dataframe.index),'Time'] = dataframe.loc[pd.isnull(dataframe.index),'Time'].apply(filter_suntracker_time)
                    
                    #Retry index
                    dataframe.index = pd.to_datetime(dataframe.Date + ' ' + dataframe.Time,errors='coerce',format='%Y.%m.%d %H:%M:%S')
                    
                    #Drop columns
                    dataframe.drop(columns=['Date','Time'],inplace=True)      
                    
                    dataframe = dataframe.astype(float)
    
                    #get list of unique days                    
                    dfs = [group[1] for group in dataframe.groupby(dataframe.index.date)]
                    days = pd.to_datetime([group[0] for group in dataframe.groupby(dataframe.index.date)])
 
                    pv_systems[station]['irrad']['suntracker'] = (days,dfs)    

                    print(('Irradiance data from station %s, substation %s successfully imported' % (station,substat)))  
                    
            if "MORDOR" in substat:
                pv_systems[station]['irrad']['mordor'] = ()
                print(("Importing data from %s, %s, please wait....." %(station,substat)))
                
                rad_files =  list_files(os.path.join(mainpath,substat))
                if not rad_files:
                    print(('Substation %s at station %s has no radiation data' % (substat,station)))
                    del pv_systems[station]['irrad']['mordor']

                else:
                    #Go through the data files (one for each day) and import to a list
                    dfs = [pd.read_csv(os.path.join(mainpath,substat,filename),
                                       header=None,sep='\s+',comment='#',usecols=[0,1,2,3,4,5,6,7,8],
                                       names=['Date','Time','Edirnorm_MS56_Wm2','Etotdown_CMP21_Wm2','Ediffdown_CMP21_Wm2',
                                              'Etotdownlw_CGR4_Wm2','Ediffdownlw_CGR4_Wm2','Etotdown_ML020VM_Wm2',
                                              'Ediffdown_ML020VM_Wm2']) 
                                              for filename in rad_files]
                                              
                    #Concatenate list 
                    dataframe = pd.concat(dfs,axis=0)                    
                    
                    #Set index to be in datetime object
                    dataframe.index = pd.to_datetime(dataframe.Date + ' ' + dataframe.Time,errors='coerce',format='%Y.%m.%d %H:%M:%S')
                    
                    #Drop columns
                    dataframe.drop(columns=['Date','Time'],inplace=True)      
                    
                    dataframe = dataframe.astype(float)
    
                    #get list of unique days                    
                    dfs = [group[1] for group in dataframe.groupby(dataframe.index.date)]
                    days = pd.to_datetime([group[0] for group in dataframe.groupby(dataframe.index.date)])
 
                    pv_systems[station]['irrad']['mordor'] = (days,dfs)    

                    print(('Irradiance data from station %s, substation %s successfully imported' % (station,substat)))  
                    
            if "RT1" in substat:
                pv_systems[station]['irrad']['RT1'] = ()
                print(("Importing data from %s, %s, please wait....." %(station,substat)))
                
                rad_files =  list_files(os.path.join(mainpath,substat))
                if not rad_files:
                    print(('Substation %s at station %s has no radiation data' % (substat,station)))
                    del pv_systems[station]['irrad']['RT1']

                else:
                    #Go through the data files (one for each day) and import to a list
                    dfs = [pd.read_csv(os.path.join(mainpath,substat,filename),
                                       header=None,sep='\s+',comment='#',usecols=[0,1,2,3,4],skiprows=1,
                                       names=['Date','Time','Etotpoa_RT1_Wm2','T_module_C','p_air_Pa'],
                                       na_values = "---") for filename in rad_files]                                              

                    #Concatenate list 
                    dataframe = pd.concat(dfs,axis=0)                    
                    
                    #Set index to be in datetime object
                    dataframe.index = pd.to_datetime(dataframe.Date + ' ' + 
                                         dataframe.Time,errors='coerce',
                                         format='%d.%m.%Y %H:%M:%S')
                    
                    #Drop columns
                    dataframe.drop(columns=['Date','Time'],inplace=True)      
                    
                    #Make sure data is of the right type
                    dataframe = dataframe.astype(float)
    
                    #get list of unique days                    
                    dfs = [group[1] for group in dataframe.groupby(dataframe.index.date)]
                    days = pd.to_datetime([group[0] for group in dataframe.groupby(dataframe.index.date)])
 
                    pv_systems[station]['irrad']['RT1'] = (days,dfs)    

                    print(('Irradiance data from station %s, substation %s successfully imported' % (station,substat)))  
            
            if "Jahresstrahlungsmessung" in substat:
                #pv_systems[station]['irrad']['Pyr_SiRef'] = ()
                print(("Importing data from %s, %s, please wait....." %(station,substat)))
                
                rad_files =  list_files(os.path.join(mainpath,substat))
                if not rad_files:
                    print(('Substation %s at station %s has no radiation data' % (substat,station)))
                    #del pv_systems[station]['irrad']['Pyr_SiRef']

                else:
                    #Go through the data files (one for each day) and import to a list
                    cols = ['Date','Time','Etotdown_CMP11_Wm2','Etotpoa_32_S_CMP11_Wm2',
                            'Etotpoa_32_E_Si02_Wm2','Etotpoa_32_S_Si02_Wm2',
                            'Etotpoa_32_W_Si02_Wm2']                    
                    dfs = [pd.read_csv(os.path.join(mainpath,substat,filename),
                                       header=None,sep=';',comment='#',index_col=False,
                                       names=cols,converters=dict(list(zip(cols[2:],
                                       [convert_wrong_format]*len(cols[2:])))))
                                    for filename in rad_files]                                              

                    #Concatenate list 
                    dataframe = pd.concat(dfs,axis=0)                    
                    
                    #Set index to be in datetime object
                    dataframe.index = pd.to_datetime(dataframe.Date + ' ' + 
                                         dataframe.Time,errors='coerce',
                                         format='%Y-%m-%d %H:%M:%S')
                    
                    #Drop columns
                    dataframe.drop(columns=['Date','Time'],inplace=True)      
                    
#                    for col in dataframe.columns:
#                        dataframe[col] = convert_wrong_format(dataframe[col].values)
                        
#                        for cols in dataframe.columns:
#                            dataframe[cols] = convert_wrong_format(dataframe[cols].values)
#                    
                    #Make sure data is of the right type
                    dataframe = dataframe.astype(float)
                    
                    oldcols = cols[2:]
                    newcols = [re.sub('_32_.', '', col) for col in oldcols]
                    dataframe.rename(columns=dict(zip(oldcols,newcols)),inplace=True)                               
                    
                    for i, substat_rad in enumerate(['CMP11_Horiz','CMP11_32S','SiRef_32E','SiRef_32S','SiRef_32W']):
                        pv_systems[station]['irrad'][substat_rad] = ()
                        
                        df_rad = dataframe.iloc[:,[i]]                                               
    
                        #get list of unique days                    
                        dfs = [group[1] for group in df_rad.groupby(df_rad.index.date)]
                        days = pd.to_datetime([group[0] for group in df_rad.groupby(df_rad.index.date)])
 
                        pv_systems[station]['irrad'][substat_rad] = (days,dfs)    

                    print(('Irradiance data from station %s, substation %s successfully imported' % (station,substat)))  
            
            if "Bedeckungsgrad" in substat:
                pv_systems[station]['irrad']['cloudcam'] = ()
                print(("Importing data from %s, %s, please wait....." %(station,substat)))
                
                rad_files =  list_files(os.path.join(mainpath,substat))
                if not rad_files:
                    print(('Substation %s at station %s has no cloudcam data' % (substat,station)))
                    del pv_systems[station]['irrad']['cloudcam']

                else:
                    #Go through the data files (one for each day) and import to a list
                    dfs = [pd.read_csv(os.path.join(mainpath,substat,filename),
                                       header=None,sep='\s+',comment='%',usecols=[0,1,3],
                                       names=['Date','Time','cf_cloudcam'],
                                       dtype={"Date":str,"Time":str,"cf_cloudcam":np.float64})
                                       for filename in rad_files]                                              

                    #Concatenate list 
                    dataframe = pd.concat(dfs,axis=0)                    
                    
                    #Set index to be in datetime object
                    dataframe.index = pd.to_datetime(dataframe.Date + ' ' + 
                                         dataframe.Time,errors='coerce',
                                         format='%Y%m%d %H%M%S')
                    
                    #Drop columns
                    dataframe.drop(columns=['Date','Time'],inplace=True)      
                    
                    #Make sure data is of the right type
                    dataframe = dataframe.astype(float)
    
                    #get list of unique days                    
                    dfs = [group[1] for group in dataframe.groupby(dataframe.index.date)]
                    days = pd.to_datetime([group[0] for group in dataframe.groupby(dataframe.index.date)])
 
                    pv_systems[station]['irrad']['cloudcam'] = (days,dfs)    

                    print(('Cloud cam data from station %s, substation %s successfully imported' % (station,substat)))  
                                      
    print('All pyranometer data imported\n')                    
    return pv_systems

def load_temp_data (pv_systems,info,paths):
    """
    Load temperature data into dataframe
    
    args:
    :param pv_systems: dictionary of PV systems
    :param info: dictionary with information about PV systems
    :param paths: dictionary of paths
    
    out:
    :return: dictionary of PV systems
    
    """    
    
    for station in info.index:
        mainpath = os.path.join(paths['mainpath'],station)
        station_dirs = list_dirs(mainpath)  
        
        if "temp" not in pv_systems[station]:
            pv_systems[station]['temp'] = {}
        for substat in station_dirs:
            if substat == paths['temp']['path']:
                pv_systems[station]['temp'][substat] = ()
                print(("Importing temperature data from %s, %s, please wait....." % (station,substat)))
                
                temp_files =  list_files(os.path.join(mainpath,substat))
                if not temp_files:
                    print(('Substation %s at station %s has no temperature data' % (substat,station)))
                    del pv_systems[station]['temp'][substat]

                else:
                    dfs = [pd.read_csv(os.path.join(mainpath,substat,filename)
                                   ,header=None,sep=';',comment='#') for filename in temp_files]
                
                    for dataframe in dfs:
                        dataframe.index = pd.to_datetime(dataframe[0] + ' ' + dataframe[1],format='%Y.%m.%d %H:%M:%S')
                        dataframe.drop(columns=[0,1],inplace=True)
                            
                        #In this case we create Multi-Index now since the data has both sensors in one file
                        dataframe.columns = pd.MultiIndex.from_product([['T_module_C'],
                                             ['PVTemp_' + str(i + 1) for i in range(len(dataframe.columns))]],
                                             names=['variable','substat'])
                        dataframe.dropna(axis=1,how='all',inplace=True)
                    
                    #Put all data into one frame
                    dataframe = pd.concat(dfs,axis=0)

                    dfs = [group[1] for group in dataframe.groupby(dataframe.index.date)]
                    days = pd.to_datetime([group[0] for group in dataframe.groupby(dataframe.index.date)])
                    
                    pv_systems[station]['temp'][substat] = (days,dfs)

                    print(('Temperature data from station %s, substation %s successfully imported' % (station,substat)))
                                      
    print('All temperature data imported\n')                    
    return pv_systems

def load_wind_data (pv_systems,info,paths):
    """
    Load wind data into dataframe
    
    args:
    :param pv_systems: dictionary of PV systems
    :param info: dictionary with information about PV systems
    :param paths: dictionary of paths
    
    out:
    :return: dictionary of PV systems
    
    """    
    
    for station in info.index:
        mainpath = os.path.join(paths['mainpath'],station)
        station_dirs = list_dirs(mainpath)  
        
        if "wind" not in pv_systems[station]:
            pv_systems[station]['wind'] = {}
        for substat in station_dirs:
            if substat == paths['wind']['path']:
                pv_systems[station]['wind'][substat] = ()
                print(("Importing wind data from %s, %s, please wait....." % (station,substat)))
                
                wind_files =  list_files(os.path.join(mainpath,substat))
                if not wind_files:
                    print(('Substation %s at station %s has no wind data' % (substat,station)))
                    del pv_systems[station]['wind'][substat]

                else:
                    if "Solarwatt" in mainpath:
                        dfs = [pd.read_csv(os.path.join(mainpath,substat,filename)
                                       ,header=None,skiprows=6,sep='\s+',decimal=',',
                                       usecols=[0,1,2,3,5])
                                       for filename in wind_files]
                        
                        for dataframe in dfs:
                            dataframe.index = pd.to_datetime(dataframe[0] + ' ' + dataframe[1],format='%d.%m.%Y %H:%M')
                            dataframe.drop(columns=[0,1],inplace=True)
                            dataframe.columns = pd.MultiIndex.from_product([['T_ambient_C','dir_wind','v_wind_mast_ms'],
                                                 ['Windmast']],names=['variable','substat'])
                            
                    else:
                        dfs = [pd.read_csv(os.path.join(mainpath,substat,filename)
                                       ,header=None,sep=',',comment='#',usecols=[0,1,3,4]) for filename in wind_files]
                    
                        #In this case we create Multi-Index now since the data has both sensors in one file
                        for dataframe in dfs:
                            dataframe.index = pd.to_datetime(dataframe[0] + ' ' + dataframe[1],format='%Y-%m-%d %H:%M:%S')
                            dataframe.drop(columns=[0,1],inplace=True)
                            dataframe.columns = pd.MultiIndex.from_product([['T_ambient_C','v_wind_mast_ms'],
                                                 ['Windmast']],names=['variable','substat'])
                        
                        #dataframe.fillna(0,inplace=True)
                    
                    dataframe = pd.concat(dfs,axis=0)

                    dfs = [group[1] for group in dataframe.groupby(dataframe.index.date)]
                    days = pd.to_datetime([group[0] for group in dataframe.groupby(dataframe.index.date)])
                    
                    pv_systems[station]['wind'][substat] = (days,dfs)

                    print(('Wind data from station %s, substation %s successfully imported' % (station,substat)))
                    
                                      
    print('All wind data imported\n')                    
    return pv_systems

def load_pmax_data (pv_systems,info,paths):
    """
    Load PMAX-DOAS data into dataframe
    
    args:
    :param pv_systems: dictionary of PV systems
    :param info: dictionary with information about PV systems
    :param paths: dictionary of paths
    
    out:
    :return: dictionary of PV systems
    
    """    
    
    for station in info.index:
        mainpath = os.path.join(paths['mainpath'],station)
        station_dirs = list_dirs(mainpath)  
        
        pv_systems[station]['pmax'] = {}
        for substat in station_dirs:
            if substat == paths['pmaxdoas']['path'][0]:
                pv_systems[station]['pmax'][substat] = ()
                print(("Importing PMAX-DOAS data from %s, %s, please wait....." % (station,substat)))
                
                pmax_dirs =  list_dirs(os.path.join(mainpath,substat))
                if not pmax_dirs:
                    print(('Substation %s at station %s has no PMAX-DOAS data' % (substat,station)))
                    del pv_systems[station]['pmax'][substat]

                else:
                    for day_dir in pmax_dirs:
                        filepath = os.path.join(mainpath,substat,day_dir,
                                                paths['pmaxdoas']['path'][1])
                        pmax_files = list_files(filepath)
                        dfs = [pd.read_csv(os.path.join(filepath,filename)
                               ,header=0,usecols=(0,1,10,11),sep='\s+',decimal='.')
                               for filename in pmax_files if "retrieval" in filename
                               and "aerosol" not in filename]
                        
                        for dataframe in dfs:
                            dataframe.index = pd.to_datetime(dataframe["Date"] + ' ' + dataframe["Time"],
                                                     format='%d/%m/%Y %H:%M:%S') 
                            dataframe.drop(columns=["Date","Time"],inplace=True)                                                       
                            dataframe.columns = pd.MultiIndex.from_product([['AOD_361','error_AOD_361'],
                                                 ['PMAX-DOAS']],names=['variable','substat'])
                        
                        #dataframe.fillna(0,inplace=True)
                    
                    dataframe = pd.concat(dfs,axis=0)

                    dfs = [group[1] for group in dataframe.groupby(dataframe.index.date)]
                    days = pd.to_datetime([group[0] for group in dataframe.groupby(dataframe.index.date)])
                    
                    pv_systems[station]['pmax'][substat] = (days,dfs)

                    print(('PMAX-DOAS data from station %s, substation %s successfully imported' % (station,substat)))
                    
                                      
    print('All wind data imported\n')                    
    return pv_systems

def extract_config_load_data(config,stations,home,info):
    """
    Extract station configuration from Excel table and load data
    
    args:
    :param config: dictionary loaded from data configuration file
    :param stations: string, which station to extract, can also be "all"    
    :param home: string, homepath
    :param info: string, description of simulation
    
    out:
    :return py_systems: dictonary of stations with data
    :return select_system_info: list of stations that are loaded
    :return runtime: time it took to load data
    :return loadpath: dictionary with paths for loading data
    """
    
    #Location of PV files
    loadpath = config["paths"]
    
    #Configuration for data processing, if necessary
    process_config = config["data_processing"]
    
    #get system info from Excel table
    system_info  = pd.read_excel(os.path.join(home,loadpath["savedata"]["main"],config["configtable"]),index_col=0)        
    print("System info loaded\n")
    print(system_info)
    
    #Choose which stations to load
    if stations != "all":
        if type(stations) != list:
            stations = [stations]
            
        select_system_info = system_info.loc[stations]
    else:
        select_system_info = system_info
    
    #Extract data from table
    paths, pv_systems = extract_station_info(select_system_info,loadpath)
    
    start = time.time()
        
    #Load PV data
    pv_systems = load_pv_data(pv_systems,select_system_info,loadpath,info,process_config)
        
    #Load radiation data    
    pv_systems = load_rad_data(pv_systems,select_system_info,loadpath,info)
    
    if "temp" in loadpath:
        #Load temperature data
        pv_systems = load_temp_data(pv_systems,select_system_info,loadpath)
        
    if "wind" in loadpath :
        #Load wind data
        pv_systems = load_wind_data(pv_systems,select_system_info,loadpath)
        
    if "pmaxdoas" in loadpath:
        #Load PMAX data
        pv_systems = load_pmax_data(pv_systems,select_system_info,loadpath)
    
    end = time.time()
    runtime = end - start
    print(("Loading data took %g seconds" % runtime))
    
    return pv_systems, select_system_info, runtime, loadpath

def load_binary_data(config,home):
    """
    Load data that has been stored as a python binary stream
    
    args:
    :param config: config file for data    
    :param home: string, home path
    
    out:
    :return pv_systems: dictionary of PV stations with data
    :return sys_info: dataframe with station information from table
    """
    
    savedir = os.path.join(home,config["paths"]["savedata"]["main"])
    files = list_files(savedir)
        
    #Choose which stations to load
    if config["stations"] == "all":
        #get system info from Excel table
        sys_info  = pd.read_excel(os.path.join(savedir,config["configtable"]),index_col=0)                
        stations = sys_info.index
    else:
        sys_info = pd.DataFrame()
        stations = config["stations"]
        if type(stations) != list:
            stations = [stations]
        
    pv_systems = {}    
    
    binarypath = os.path.join(savedir,config["paths"]["savedata"]["binary"])
    
    for station in stations:
        filename = config["description"] + '_' + station + ".data"
        if filename in files:        
            with open(os.path.join(binarypath,filename), 'rb') as filehandle:  
                (pvstat, info) = pd.read_pickle(filehandle)            
            pv_systems.update({station:pvstat})
            print(('Data for %s loaded from %s' % (station,filename)))
        #Extract config and load data
        else:
            print(('No binary data file for %s found, loading from CSV...' % station))                   
            pvstat, info, loadtime = extract_config_load_data(config,station,home)    
            pv_systems.update({list(pvstat.keys())[0]:list(pvstat.values())[0]})
            
        sys_info = pd.concat([sys_info,info],axis=0)

    return pv_systems, sys_info    

def load_station_data(savedir,filename,data_types,data_flag=False):
    """
    Load data that has already been resampled to a specified time resolution
    
    args:    
    :param savedir: string, path where data is saved
    :param filename: string, name of file
    :param data_types: dictionary with datatypes
    :param data_flag: boolean, whether to keep original data
    
    out:
    :return pvstat: dictionary of PV system dataframes and other information
    :return info: table with information about each station
    """

    try:
        with open(os.path.join(savedir,filename), 'rb') as filehandle:  
            (pvstat, info) = pd.read_pickle(filehandle)  
                    
#        pvstat.update({"raw_data":{}})
#        for idata in data_types:
#            if idata in pvstat:
#                pvstat["raw_data"].update({idata:pvstat[idata]})
#                del pvstat[idata]
                            
        #reduce file size by removing original data                
#        if not data_flag:
#            print('Removing original high frequency data')
#            del pvstat["raw_data"]
    except IOError:
        print(('File %s not found' % os.path.join(savedir,filename)))
        return None, None
        
    return pvstat, info

def load_resampled_data(timeres,config,home):
    """
    Load data that has already been resampled to a specified time resolution
    
    args:    
    :param timeres: string, timeresolution of the data
    :param config: dictionary with paths for loading data
    :param home: string, homepath    
    
    out:
    :return pv_systems: dictionary of PV systems with dataframes and other information
    :return sys_info: table with information about each station
    """

    savedir = os.path.join(home,config["paths"]["savedata"]["main"])    
    
    #Choose which stations to load
    if config["stations"] == "all":
        #get system info from Excel table
        sys_info  = pd.read_excel(os.path.join(savedir,config["configtable"]),index_col=0)                
        stations = sys_info.index
    else:
        sys_info = pd.DataFrame()
        stations = config["stations"]
        if type(stations) != list:
            stations = [stations]
        
    pv_systems = {}       
    
    binarypath = os.path.join(savedir,config["paths"]["savedata"]["binary"])
    files = list_files(binarypath)    
    
    for station in stations:
        filename = config["description"] + '_' + station + "_" + timeres + ".data"
        if filename in files:        
            with open(os.path.join(binarypath,filename), 'rb') as filehandle:  
                (pvstat, info) = pd.read_pickle(filehandle)  
                
            pv_systems.update({station:pvstat})
            sys_info = pd.concat([sys_info,info],axis=0)
            
            print(('Data for %s loaded from %s' % (station,filename)))
        else:
            print('Required file not found')
                    
    return pv_systems, sys_info   
