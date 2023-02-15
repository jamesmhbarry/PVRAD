#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 10:15:05 2018

@author: james
"""

#%% Preamble
import os
import yaml
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from scipy import optimize

#%% Functions

def load_yaml_configfile(fname):
    """
    load yaml config file
    
    args:
    :param fname: string, complete name of config file
    
    out:
    :return: config dictionary
    """
    with open(fname, 'r') as ds:
        try:
            config = yaml.load(ds,Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)
    return config  

def import_aeronet_day_average (path,file_name,version,padding,datatype):
    """
    Import Aeronet data (daily averages) to dataframe
    
    args:
    :param path: string, path where aeronet files are located
    :param file_name: string, name of downloaded file, with extension
    :param version: integer giving file version
    :param padding: boolean to decide whether to pad values by interpolation
    :param datatype: string to define whether it is AOD or SSA data
    
    out:
    :return dataframe: timeseries data imported from file
    :return time_range: string defining time range of data
    :return empty_days: list with days that contain no data
    """
    
    if version == 3:
        rows_preamble = 6
    elif version == 2:
        if datatype == "aod":
            rows_preamble = 4
        elif datatype == "ssa":
            rows_preamble = 3
    
    #1. read in data from file and create timestamp index            
    if version == 3 and datatype == "ssa": 
        dataframe = pd.read_csv(os.path.join(path,file_name),index_col=(1,2),header=0,skiprows=rows_preamble)
        dataframe.index = pd.to_datetime(dataframe.index,format='(\'%d:%m:%Y\', \'%H:%M:%S\')')
        dataframe = dataframe.drop('AERONET_Site',axis=1)
    else:
        dataframe = pd.read_csv(os.path.join(path,file_name),index_col=(0,1),header=0,skiprows=rows_preamble)
        dataframe.index = pd.to_datetime(dataframe.index,format='(\'%d:%m:%Y\', \'%H:%M:%S\')')
    dataframe.index.name = 'Timestamp (UTC)'

    #2. Drop columns we don't need
    if datatype == "aod":
        if version == 2:
            dataframe = dataframe.filter(regex='^AOT_|Angstrom$', axis=1)
        elif version == 3:
            dataframe = dataframe.filter(regex='^AOD_(?!Empty.*)|Angstrom_Exponent$', axis=1)
    elif datatype == "ssa":
        if version == 2:
            dataframe = dataframe.filter(regex='^SSA', axis=1)
        elif version == 3:
            dataframe = dataframe.filter(regex='^Single_Scattering_Albedo', axis=1)
        
    #Get rid of negative AOD and nans
    dataframe = dataframe[dataframe>0]    
    dataframe.dropna(axis='columns',how='all',inplace=True)    
    
    #3. Extend file to include all values in the range (interpolate)
    empty_days = []
    if padding:
        if version == 2:
            timestring = ' 00:00:00'
        elif version == 3:
            timestring = ' 12:00:00'
        #Get time range from file name
        if len(file_name.split('_')[0]) == 6:
            time_range = ['20' + date[0:2] + '-' + date[2:4] + '-' + date[4:6] 
            for date in file_name[0:13].split('_')]
            pad_index = pd.date_range(start=time_range[0] + timestring,
                                      end=time_range[1] + timestring,freq='D',
                                      name=dataframe.index.name)
        elif len(file_name.split('_')[0]) == 8:
            time_range = [date[0:4] + '-' + date[4:6] + '-' + date[6:8] 
            for date in file_name[0:17].split('_')]                
            pad_index = pd.date_range(start=time_range[0] + timestring,
                                      end=time_range[1] + timestring,freq='D',
                                      name=dataframe.index.name)
        
        #Fill with Nans
        dataframe = dataframe.reindex(pad_index)
        #Get list of empty days
        [empty_days.append(day.date().strftime('%Y-%m-%d')) 
        for day in dataframe.index if dataframe.loc[day].isna().all()]
        #For AOD, interpolate
        if datatype == 'aod':
            dataframe = dataframe.interpolate('index')
                
        time_range = time_range[0] + '_' + time_range[-1]
    else:
        time_range = dataframe.index[0].strftime(format="%Y-%m-%d") + '_' +\
                            dataframe.index[-1].strftime(format="%Y-%m-%d")
    
    return dataframe, time_range, empty_days

def import_aeronet_all(path,stat_dict,version,timeres,padding,datatype):
    """
    Import Aeronet data to dataframe
    
    args:
    :param path: string, path where aeronet files are located
    :param stat_dict: dictionary with all information about aeronet station
    :param version: integer giving file version
    :param timeres: integer defining the time resolution for interpolation
    :param padding: boolean to decide whether to pad values with average
    :param datatype: string to define data type (AOD or SSA)
    
    out:
    :return dataframe: timeseries data imported from file
    :return time_range: string defining time range of data
    :return empty_days: list with days that contain no data
    """
    
    if datatype == 'aod':
        file_name = stat_dict['aod_files']['all']
    elif datatype == 'ssa':
        file_name = stat_dict['ssa_files']['all']    
    
    if version == 3:
        rows_preamble = 6
    elif version == 2:
        if datatype == "aod":
            rows_preamble = 4
        elif datatype == "ssa":
            rows_preamble = 3
    
    #1. read in data from file and create timestamp index
    if version == 3 and datatype == "ssa": 
        dataframe = pd.read_csv(os.path.join(path,file_name),index_col=(1,2),header=0,skiprows=rows_preamble)
        dataframe.index = pd.to_datetime(dataframe.index,format='(\'%d:%m:%Y\', \'%H:%M:%S\')')
        dataframe = dataframe.drop('Site',axis=1)
    else:
        dataframe = pd.read_csv(os.path.join(path,file_name),index_col=(0,1),header=0,skiprows=rows_preamble)
        dataframe.index = pd.to_datetime(dataframe.index,format='(\'%d:%m:%Y\', \'%H:%M:%S\')')
    dataframe.index.name = 'Timestamp (UTC)'

    #2. Drop columns we don't need    
    if datatype == "aod":
        if version == 2:
            dataframe = dataframe.filter(regex='^AOT_|Angstrom$', axis=1)
            df_day_ave = stat_dict['df_day'].filter(regex='^AOT_|Angstrom$', axis=1)
        elif version == 3:
            dataframe = dataframe.filter(regex='^AOD_(?!Empty.*)|Angstrom_Exponent$', axis=1)            
            df_day_ave = stat_dict['df_day'].filter(regex='^AOD_(?!Empty.*)|Angstrom_Exponent$', axis=1)            
    elif datatype == "ssa":
        if version == 2:
            dataframe = dataframe.filter(regex='^SSA', axis=1)
        elif version == 3:
            dataframe = dataframe.filter(regex='^Single_Scattering_Albedo', axis=1)
    
    dataframe = dataframe[dataframe>0]
    dataframe.dropna(axis='columns',how='all',inplace=True)

    #3. Interpolate values (do it by day)            
    #Get time range from file name
    if len(file_name.split('_')[0]) == 6:
        time_range = ['20' + date[0:2] + '-' + date[2:4] + '-' + date[4:6] 
        for date in file_name[0:13].split('_')]
    elif len(file_name.split('_')[0]) == 8:
        time_range = [date[0:4] + '-' + date[4:6] + '-' + date[6:8] 
        for date in file_name[0:17].split('_')]        
    
    #Interpolate
    time_res_string = str(timeres) + 'min'
    
    #Choose a smaller grid for first step of interpolation
    fine_res_string = str(np.ceil(timeres/5)) + 'min'    
    
    #Split data into days
    dfs = [group[1] for group in dataframe.groupby(dataframe.index.date)]
    days = pd.to_datetime([group[0] for group in dataframe.groupby(dataframe.index.date)])

    df_rs = []    
    for ix, iday in enumerate(days):        
        #First interpolate onto finer grid, allow filling
        newindex_fine = pd.date_range(start=iday, end=dfs[ix].index[-1],
                                      freq=fine_res_string)
        
        #Define index for whole day, this will simply then fill with NaNs
        newindex_timeres = pd.date_range(start=iday, end=iday + pd.Timedelta('1D')
             - pd.Timedelta(time_res_string), freq=time_res_string)
    
        #Define interpolation limit, only up to an hour (60 minutes) of filling
        #interp_limit = int(60*24/(float(fine_res_string[0:-1])))
        df_new = dfs[ix].reindex(dfs[ix].index.union(newindex_fine)).\
                     interpolate('index').reindex(newindex_timeres)

        #Fill NaNs with average of the daily values         
        if padding:            
            df_rs.append(df_new.fillna(df_new.mean()))
        else:
            df_rs.append(df_new)
            
#            if datatype == 'aod':
#                fig, ax = plt.subplots(figsize=(9,8))
#                
#                dfs[ix].loc[:,'AOD_500nm'].plot(ax=ax,style='*',legend=False)
#                df_rs[ix].loc[:,'AOD_500nm'].plot(ax=ax,style='--',legend=False)
#                ax.set_ylabel('AOD at 500nm')
#                plt.savefig('aod_500nm_interp_' + iday.strftime('%Y-%m-%d') +  '.png')
#                plt.close(fig)
    
    #Put all data together again
    dataframe_rs = pd.concat(df_rs,axis=0)
    
    #Define end_time (make sure still in the same day)
    start_time = pd.to_datetime(time_range[0])
    end_time = pd.to_datetime(time_range[1]) + pd.Timedelta('1D') -\
        pd.Timedelta(minutes=timeres)
    total_index = pd.date_range(start=start_time, end=end_time, freq=time_res_string)
                               
    if padding:
        if datatype == 'aod':
            dataframe_rs = dataframe_rs.reindex(total_index)
            dfs_day = [group[1] for group in df_day_ave.groupby(df_day_ave.index.date)]
            day_index = [group[1].index for group in dataframe_rs.groupby(dataframe_rs.index.date)]
            dfs_day_full = pd.concat([df_day.reindex(day_index[i]).fillna(df_day.mean())
            for i, df_day in enumerate(dfs_day)],axis=0)
            dataframe_rs = dataframe_rs.fillna(dfs_day_full)
            print('Filled NaNs with daily averages')
#            #These are test plots to check the interpolation
#            #Split data into days
#            dfs_rs = [group[1] for group in dataframe_rs.groupby(dataframe_rs.index.date)]
#            days = pd.to_datetime([group[0] for group in dataframe_rs.groupby(dataframe_rs.index.date)])
#        
#            for ix, iday in enumerate(days):     
#                if datatype == 'aod':
#                    fig, ax = plt.subplots(figsize=(9,8))
#                                
#                    dfs_rs[ix].loc[:,'AOD_500nm'].plot(ax=ax,style='--',legend=False)
#                    ax.set_ylabel('AOD at 500nm')
#                    plt.savefig('aod_500nm_full_interp_' + iday.strftime('%Y-%m-%d') +  '.png')
#                    plt.close(fig)
                    
        elif datatype == 'ssa':
            dataframe_rs = dataframe_rs.reindex(total_index)
     
    #Define time range string for file names
    time_range_string = dataframe_rs.index[0].strftime(format="%Y-%m-%dT%H%M%S") + '_' +\
                            dataframe_rs.index[-1].strftime(format="%Y-%m-%dT%H%M%S")
                            
    empty_days = []
    
    return dataframe_rs, time_range_string, empty_days


def load_plot_aerosol_data(load_path,station_dict,dict_paths,timeres,plotting=False,padding=False):
    """
    Load Aeronet data for aerosol optical depth and single scattering albedo from file
    and plot raw data if required
    
    First import daily average and then all values.
    If a day has no data, interpolation is used with surrounding days, to create new
    daily average AODs. These values are then used to fill up the data in the 
    import_aeronet_all function
    
    args:
    :param load_path: string, path where aeronet files are located    
    :param station_dict: dictionary of aeronet stations (without data)
    :param dict_path: dictionary with paths to save plots to
    :param timeres: integer defining the time resolution for interpolation
    :param plotting: flag for plotting
    :param padding: boolean to decide whether to pad values with nan to fill a year

    
    out:
    :return: dictionary of aeronet stations including data as well as time range strings
    """        
    
    save_path = os.path.join(dict_paths["main"],dict_paths["raw"])
    
    df_aod = pd.DataFrame()
    df_ssa = pd.DataFrame()
    range_aod_day = ""
    range_ssa_day = ""
    range_aod = ""
    range_ssa = ""
    
    for key in station_dict:            
        print("Importing daily average data from %s" % station_dict[key]["name"])
        station_dict[key]['df_day'] = pd.DataFrame()
        if station_dict[key]['aod_files']['day_ave']:
            #Import data for daily averages
            aod_version = station_dict[key]['aod_files']['version']
            df_aod, range_aod_day, empty_aod = import_aeronet_day_average(load_path,station_dict[key]['aod_files']['day_ave'],
                                                               aod_version,padding,datatype="aod")
            if not df_aod.empty:
                if plotting:   
                    data_aod = df_aod.filter(regex='^AO.',axis=1)
                    ax1 = data_aod.plot(legend=True,figsize=(10,6*10/8),
                               title='Daily average aerosol optical depth at ' + 
                               station_dict[key]['name'],grid=True,colormap='jet')
                    ax1.set_ylabel('AOD')
                    plt.savefig(os.path.join(save_path,'aod_raw_data_' + range_aod_day + '_' + key + '.png'))
            else:
                print("All AOD values are NAN, no plots made")
            
            station_dict[key]['df_day'] = df_aod
            station_dict[key]['empty_aod'] = empty_aod
        else:
            print("No AOD data to import")
            station_dict[key]['empty_aod'] = "All days"
        
        if station_dict[key]['ssa_files']['day_ave']:
            ssa_version = station_dict[key]['ssa_files']['version']
            df_ssa, range_ssa_day, empty_ssa = import_aeronet_day_average(load_path,station_dict[key]['ssa_files']['day_ave'],
                                                               ssa_version,padding,datatype="ssa")
            if not df_ssa.empty:
                if plotting:  
                    ax2 = df_ssa.plot(legend=True,figsize=(10,6*10/8),
                               title='Daily average single scattering albedo at ' + 
                               station_dict[key]['name'],grid=True)
                    ax2.set_ylabel('SSA')
                    plt.savefig(os.path.join(save_path,'ssa_raw_data_' + range_ssa_day + '_' + key + '.png'))
            else:
                print("All SSA values are NAN, no plots made")
                
            if station_dict[key]['df_day'].empty:
                station_dict[key]['df_day'] = df_ssa
            else:
                station_dict[key]['df_day'] = pd.concat([df_aod,df_ssa],axis=1)
                
            station_dict[key]['empty_ssa'] = empty_ssa
        else:
            print("No SSA data to import")            
            station_dict[key]['empty_ssa'] = "All days"
            
        print("Importing all data from %s" % station_dict[key]["name"])
        station_dict[key]['df_all'] = pd.DataFrame()
        if station_dict[key]['aod_files']['all']:
            aod_version = station_dict[key]['aod_files']['version']
            df_aod, range_aod, empty_aod = import_aeronet_all(load_path,
            station_dict[key],aod_version,timeres,padding,datatype="aod")
            
            if not df_aod.empty:
                if plotting:
                    data_aod = df_aod.filter(regex='^AO.',axis=1)                
                    ax1 = data_aod.plot(legend=True,figsize=(10,6*10/8),
                               title='Aerosol optical depth inerpolated to ' + str(timeres) + ' minutes at ' + 
                               station_dict[key]['name'],grid=True,colormap='jet')
                    ax1.set_ylabel('AOD')
                    plt.savefig(os.path.join(save_path,'aod_raw_data_' + range_aod + '_' + key + '.png'))
            else:
                print("All AOD values are NAN, no plots made")
            
            station_dict[key]['df_all'] = df_aod
            station_dict[key]['empty_aod'] = empty_aod
        else:
            print("No AOD data to import")
            station_dict[key]['empty_aod'] = "All days"
        
        if station_dict[key]['ssa_files']['all']:
            ssa_version = station_dict[key]['ssa_files']['version']
            df_ssa, range_ssa, empty_ssa = import_aeronet_all(load_path,
            station_dict[key],ssa_version,timeres,padding,datatype="ssa")
            
            if not df_ssa.empty:
                if plotting:
                    ax2 = df_ssa.plot(legend=True,figsize=(10,6*10/8),
                               title='Daily average single scattering albedo inerpolated to ' + str(timeres) + ' minutes at ' + 
                               station_dict[key]['name'],grid=True)
                    ax2.set_ylabel('SSA')
                    plt.savefig(os.path.join(save_path,'ssa_raw_data_' + range_ssa + '_' + key + '.png'))
            else:
                print("All SSA values are NAN")
                
            if station_dict[key]['df_all'].empty:
                station_dict[key]['df_all'] = df_ssa
            else:
                station_dict[key]['df_all'] = pd.concat([df_aod,df_ssa],axis=1)
                station_dict[key]['empty_ssa'] = empty_ssa

        else:
            print("No SSA data to import")
            station_dict[key]['empty_ssa'] = "All days"
    
    return {"aero_stats": station_dict, "time_range_aod_day": range_aod_day, 
            "time_range_ssa_day": range_ssa_day, "time_range_aod_all": range_aod, 
            "time_range_ssa_all": range_ssa}

def extract_wavelengths(dataframe,fit_range,version):
    """
    Extract wavelengths (integers) from column names
    
    args:
    :param dataframe: dataframe with AOD and other values
    :param fit_range: wavelength range to fit
    
    out:
    :return data_aod: dataframe with only AOD values
    :return xdata: Wavelength values for fit procedure    
    
    """
    
    if version == 2:
        #Extract AOD values
        data_aod = dataframe.filter(regex="^AOT_",axis=1)        
        #Get wavelengths from the column names
        aod_wvl = [x.replace('AOT_','') for x in data_aod.columns.values.tolist()]
    elif version == 3:
        #Extract AOD values
        data_aod = dataframe.filter(regex="^AOD_",axis=1)        
        #Get wavelengths from the column names
        aod_wvl = [x.replace('AOD_','') for x in data_aod.columns.values.tolist()]
        aod_wvl = [x.replace('nm','') for x in aod_wvl]
        aod_wvl = [re.sub('Empty.*','',x) for x in aod_wvl]
    
    #wavelength in micrometers    
    xdata = np.array(aod_wvl,dtype=float)/1000. 
    #Define Boolean mask from fit range
    mask = (xdata >= fit_range[0]/1000.) & (xdata <= fit_range[1]/1000.)
    xdata = xdata[mask]
    
    data_aod = data_aod.loc[:,mask]
    
    return data_aod, xdata

def extract_visible_ssa(dataframe,version):
    """
    Extract SSA values in the visible wavelength band
    
    args:
    :param dataframe: dataframe with AOD and other values    
    :param version: integer, Aeronet version
    
    out:
    :return: dataframe with visible SSA values
    
    """
    #Extract SSA values
    if version == 2:        
        data_ssa = dataframe.filter(regex="^SSA",axis=1)
        #Get wavelengths from column names
        ssa_wvl = [x.replace('SSA','') for x in data_ssa.columns.values.tolist()]
        ssa_wvl = np.array([x.replace('-T','') for x in ssa_wvl],dtype=float)
    elif version == 3:
        data_ssa = dataframe.filter(regex='^Single_Scattering_Albedo', axis=1)
        #Get wavelengths from column names
        ssa_wvl = [x.replace('Single_Scattering_Albedo[','') for x in data_ssa.columns.values.tolist()]
        ssa_wvl = np.array([x.replace('nm]','') for x in ssa_wvl],dtype=float)
        
    #Define Boolean mask from visible wavelength band
    mask_ssa_vis = (ssa_wvl >= 380.0) & (ssa_wvl <= 740.0)
    series_ssa_vis = pd.Series(data_ssa.loc[:,mask_ssa_vis].mean(axis=1),
            index=data_ssa.index,name='ssa_vis')
    
    data_ssa = pd.concat([data_ssa,series_ssa_vis],axis=1)
    
    return data_ssa
    

def angstrom_fit (fit_config, station_dict, fit_range, dict_paths, plotting,
                  curvature=False):
    """
    Fit Aeronet data using Angstrom's formula for each station in the config file
    and plot if required
    
    args:
    :param fit_config: string defining whether to use daily data or all data
    :param station_dict: dictionaries with station info and data    
    :param fit_range: array giving wavelength range for AOD fit    
    :param dict_paths: dictionary with plot paths
    :param plotting: dict, output plots or not 
    :param curvature: bool, whether to add quadratic term for wavelength dependence of alpha
    
    out:
    :return station_dict: dictionaries with station data and also including fit parameters    
    :return xdata: array of wavelengths used for the fit
    """    
    version_aod = station_dict["aod_files"]["version"]
    version_ssa = station_dict["ssa_files"]["version"]
    save_path = os.path.join(dict_paths["main"],dict_paths["fits"])
    
    #Get data from dictionaries
    if fit_config == "day_ave":
        data = station_dict['df_day']
        print("Extracting alpha and beta from log-linear fit for daily average AOD data from %s" 
              % station_dict['name'])
    elif fit_config == "all":
        data = station_dict['df_all']
        print("Extracting alpha and beta from log-linear fit for all AOD data from %s ... this takes a while"
              % station_dict['name'])
        
    if plotting:
        print("Plotting is turned on, the fit for each day will be plotted, please be patient...")
        
    #Get original index before processing
    ix = data.index
            
    #Get wavelengths from column names    
    data_aod, xdata = extract_wavelengths(data,fit_range,version_aod)
    
    #Get index of valid values (remove NANs)
    data_aod = data_aod.dropna(axis='rows')
    #notnan_ix = data_aod[data_aod.notna()].index    
    
    #Apply mask to AOD values to extract data from fit
    ydata = data_aod.values    
    #Get angstrom data from dataframe
    if version_aod == 2:
        data_alpha = data.filter(regex="Angstrom",axis=1).reindex(data_aod.index)
    elif version_aod == 3:
        data_alpha = data.filter(regex="Angstrom.",axis=1).reindex(data_aod.index)
    alphadata = data_alpha.values
    alphadata_ave = np.average(alphadata,axis=1) 
    
    # The fit is performed according to the following formula
    # AOD = beta*lambda**(-alpha) 
    # log(AOD) = log(beta) - alpha*log(lambda)
    
    #Take logarithm of the data
    logx = np.log(xdata)
    
    #Some values are negative in the data, assign small non-zero positive value
    ydata[ydata<0.] = 0.0001
    logy = np.log(ydata)
        
    if not curvature:        
        fitfunc = lambda p, l: p[1] - p[0] * l
        errfunc = lambda p, l, y: y - fitfunc(p, l)
        pinit = [1, 1]
        
        params = np.zeros((len(data_aod),2))
    else:
        fitfunc = lambda p, l: p[1] - p[0] * l + p[2] * l**2
        errfunc = lambda p, l, y: y - fitfunc(p, l)
        pinit = [1, 1, 1]
        
        params = np.zeros((len(data_aod),3))
    #parcov = np.zeros((len(data)))
    
#    data_aod['alpha_fit'] = pd.Series(index=data_aod.index)    
#    data_aod['beta_fit'] = pd.Series(index=data_aod.index)
    
    #Perform fit for each day
    for i, datetime  in enumerate(data_aod.index):
        params[i,:] = optimize.leastsq(errfunc,pinit,args=(logx,logy[i,:]))[0]
        
        if plotting[fit_config]: # and fit_config == "day_ave":
            if fit_config == "day_ave":
                titlestring = 'Daily average AOD at ' + station_dict['name'] + ' on ' +\
                      datetime.strftime(format="%Y-%m-%d")
                filestring = 'aod_fit_day_ave_' + station_dict["name"] + '_'\
                + datetime.strftime(format="%Y%m%d") + '.png'  
            elif fit_config == "all":
                titlestring = 'AOD at ' + station_dict['name'] + ' on ' +\
                      datetime.strftime(format="%Y-%m-%d %H:%M:%S")
                filestring = 'aod_fit_' + station_dict["name"] + '_'\
                    + datetime.strftime(format="%Y%m%d_%H%M%S") + '.png'
            
            fig = plt.figure()
            plt.plot(xdata*1000, np.exp(fitfunc(params[i,:],logx)))
            plt.plot(xdata*1000,ydata[i,:],linestyle='None',marker = 'o')
            plt.ylabel(r'$\tau(\lambda)$',fontsize=16)
            plt.xlabel(r"$\lambda$ (nm)",fontsize=16)
            plt.title(titlestring)
            plt.annotate(r'$\alpha_{\rm fit}$ = '+ str(np.round(params[i,0],2))
                    ,xy=(0.6,0.8),xycoords='figure fraction',fontsize=14)       
            plt.annotate(r'$\beta_{\rm fit}$ = '+ str(np.round(np.exp(params[i,1]),3))
                    ,xy=(0.6,0.7),xycoords='figure fraction',fontsize=14)     
            plt.annotate(r'$\langle\alpha_{\rm meas}\rangle$ = '+ 
                         str(np.round(alphadata_ave[i],2)),xy=(0.6,0.6),
                         xycoords='figure fraction',fontsize=14)
            if curvature:
                plt.annotate(r'$\gamma_{\rm fit}$ = '+ str(np.round(np.exp(params[i,2]),3)),xy=(0.6,0.5),
                         xycoords='figure fraction',fontsize=14)     
        
            plt.annotate(r'$\tau = \beta \lambda^{-\alpha}$',xy=(0.4,0.5),
                         xycoords='figure fraction',fontsize=14)
            fig.tight_layout()
            plt.savefig(os.path.join(save_path,filestring))
            plt.close(fig)
         
    #add fit parameters to dataframe    
    data_aod['alpha_fit'] = pd.Series(params[:,0],index=data_aod.index)
    data_aod['beta_fit'] = pd.Series(np.exp(params[:,1]),index=data_aod.index)
    if curvature:
        data_aod['gamma_fit'] = pd.Series(params[:,2],index=data_aod.index)
    
    #Put back NANs
    data_aod = data_aod.reindex(ix)
    data_alpha = data_alpha.reindex(ix)
    
    #Extract the SSA data and create mask for visible values    
    data_ssa = extract_visible_ssa(data,version_ssa)
        
    data = pd.concat([data_aod,data_alpha,data_ssa],axis=1)
    
    #Assign back to dictionary
    if fit_config == "day_ave":
        station_dict['df_day'] = data
    elif fit_config == "all":
        station_dict['df_all'] = data
        
    return station_dict, xdata

def angstrom_fit_mean (fit_config, station_dict, mean_stats, xdata, fit_range, dict_paths, plotting,
                       curvature=False):
    """
    Fit Aeronet data using Angstrom's formula to the mean AOD for several stations
    
    args:
    :param fit_config: string defining whether to use daily data or all data
    :param station_dict: dictionaries with station info and data    
    :param mean_stats: list of stations to use for taking average
    :param xdata: array of wavelengths to use for fitting
    :param fit_range: array giving wavelength range for AOD fit    
    :param dict_paths: dictionary with plot paths
    :param plotting: dict, output plots or not 
    :param curvature: bool, whether to add quadratic term for wavelength dependence of alpha
    
    out:
    :return data_mean: dataframe with averaged data
    """        
    
    save_path = os.path.join(dict_paths["main"],dict_paths["fits"])
    
    #Get the number of stations to use for the mean
    num_stats = len(mean_stats)
    mean_label = '_'.join([s for s in mean_stats])        
    print("Extracting alpha and beta from log-linear fit for mean AOD data from %s ... this takes a while" 
              % mean_label)    
        
    #Fit data for the mean optical depth between several stations        
    for key in station_dict:
        version_aod = station_dict[key]["aod_files"]["version"]    
        #Get data
        if key in mean_stats:
            if fit_config == "day_ave":
                data = station_dict[key]['df_day']
            elif fit_config == "all":
                data = station_dict[key]['df_all']

            ix = data.index            
            data_aod, xdata = extract_wavelengths(data,fit_range,version_aod)            
            aod_array = np.zeros((num_stats,len(data),len(xdata)))
            aod_array[mean_stats.index(key),:,:] = data_aod.values
    
    data_mean = pd.DataFrame(np.mean(aod_array,axis=0),index=data.index,
                             columns=data_aod.columns)
    data_mean.dropna(axis='rows',inplace=True) #remove NANs for fitting
    
    ydata = data_mean.values
    
    logx = np.log(xdata)
    ydata[ydata<0] = 0.0001
    logy = np.log(ydata)

    if not curvature:        
        fitfunc = lambda p, l: p[1] - p[0] * l
        errfunc = lambda p, l, y: y - fitfunc(p, l)
        pinit = [1, 1]
        
        params = np.zeros((len(data_mean),2))
    else:
        fitfunc = lambda p, l: p[1] - p[0] * l + p[2] * l**2
        errfunc = lambda p, l, y: y - fitfunc(p, l)
        pinit = [1, 1, 1]
        
        params = np.zeros((len(data_mean),3))
    #parcov = np.zeros((len(df_aod_ssa_day)))
    
    for iday in range(len(data_mean)):
        params[iday,:] = optimize.leastsq(errfunc,pinit,args=(logx,logy[iday,:]))[0]
        
        if plotting[fit_config]:
            fig = plt.figure()
            plt.plot(xdata*1000, np.exp(fitfunc(params[iday,:],logx)))
            plt.plot(xdata*1000,ydata[iday,:],linestyle='None',marker = 'o')
            plt.ylabel(r'$\tau(\lambda)$',fontsize=16)
            plt.xlabel(r"$\lambda$ (nm)",fontsize=16)
            plt.title('Mean (' + mean_label + ') AOD on ' + str(data_mean.index[iday]))            
            plt.annotate(r'$\alpha_{\rm fit}$ = '+ str(np.round(params[iday,0],2)),xy=(0.6,0.8),
                         xycoords='figure fraction',fontsize=14)       
            plt.annotate(r'$\beta_{\rm fit}$ = '+ str(np.round(np.exp(params[iday,1]),3)),xy=(0.6,0.7),
                         xycoords='figure fraction',fontsize=14)     
            if curvature:
                plt.annotate(r'$\gamma_{\rm fit}$ = '+ str(np.round(np.exp(params[iday,2]),3)),xy=(0.6,0.6),
                         xycoords='figure fraction',fontsize=14)     
            
            plt.annotate(r'$\tau = \beta \lambda^{-\alpha}$',xy=(0.4,0.5),
                         xycoords='figure fraction',fontsize=14)
            fig.tight_layout()
            plt.savefig(os.path.join(save_path,'aod_fit_mean_' + mean_label + '_' 
                                     + str(data_mean.index[iday]).split(' ')[0] + '.png'))
            plt.close()
        
        data_mean['alpha_fit'] = pd.Series(params[:,0],index=data_mean.index)    
        data_mean['beta_fit'] = pd.Series(np.exp(params[:,1]),index=data_mean.index) 
        if curvature:
            data_mean['gamma_fit'] = pd.Series(params[:,2],index=data_mean.index)    
        
    #Put the NANs back to keep the dataframes aligned
    data_mean = data_mean.reindex(ix)
    
    ssa_array = np.zeros((num_stats,len(data)))
    for key in station_dict:  
        #Get data
        if key in mean_stats:
            if fit_config == "day_ave":
                data = station_dict[key]['df_day']
            elif fit_config == "all":
                data = station_dict[key]['df_all']
        
            ssa_array[mean_stats.index(key),:] = data["ssa_vis"].values
    
    data_mean['ssa_vis'] = pd.DataFrame(np.mean(ssa_array,axis=0),index=data_mean.index)
    
    return data_mean

def save_aeronet_dataframe(data,path,filename,fit_type,version,empty_dict,
                           padding,station_name):
    """
    Save Aeronet data to file
    
    args:
    :param data: dataframe to save to file
    :param path: path for saving files
    :param filename: name of file to be saved
    :param fit_type: define whether we are saving all values or daily averages
    :param version: integer defining aeronet version
    :param empty_dict: dictionary with empty days
    :param padding: boolean to decide whether to pad values with average
    :param station_name: name of station
    
    """    
    #Set the date format
    if fit_type == 'day_ave':
        datestring = "%Y-%m-%d"
    elif fit_type == 'all':
        datestring = "%Y-%m-%dT%H:%M:%S"
        
    if version == 2:
        cols = ['AOT_500','alpha_fit','beta_fit','ssa_vis']
    elif version == 3:
        cols = ['AOD_500nm','alpha_fit','beta_fit','ssa_vis']
    
    #Save data to file       
    f = open(os.path.join(path,filename), 'w')
    f.write('#Aerosol data extracted from Aeronet for %s\n' % station_name)
    if padding:
        f.write('#Empty AOD data points have been filled with daily averages\n')
        f.write('#Days of year with no AOD data: %s\n' % empty_dict['aod'])
        f.write('#Days of year with no SSA data: %s\n' % empty_dict['ssa'])
    
    f.write('#Data\n')
    data.to_csv(f,columns=cols,float_format='%.6f', index_label='Date_Time', sep=' ',
                          header=['AOD_500','alpha','beta','ssa_vis'],na_rep='nan',
                          date_format=datestring)        
    f.close()
    

def save_aerosol_files(save_path,main_dict,mean_stats,fit_config,
                       padding,description):
    """
    Save Aeronet data to file
    
    args:
    :param save_path: path for saving files
    :param main_dict: dictionary of Aeronet stations and range strings
    :param mean_stats: list of stations used for averaging
    :param fit_config: define whether we are saving all values or daily averages    
    :param padding: boolean to decide whether to pad values with average
    :param description: string with description of simulation
    
    """    
    station_dict = main_dict['aero_stats']    
    mean_label = '_'.join([s for s in mean_stats])
    
    df_filelist = pd.DataFrame(index=station_dict.keys(),columns=['day_ave','all'])
    df_filelist.index.name="Station"
        
    for key in station_dict:
        version = station_dict[key]["aod_files"]["version"]
        empty_days = {'aod':station_dict[key]['empty_aod'],'ssa':station_dict[key]['empty_ssa']}
        for fit_type in fit_config:
            if fit_type == 'day_ave' and fit_config[fit_type]:
                time_range = main_dict['time_range_aod_day']
                filename = 'aerosol_angstrom_params_day_ave_' + time_range + '_' + key + '.dat'
                longname = station_dict[key]["name"]
                save_aeronet_dataframe(station_dict[key]['df_day'],save_path,filename,fit_type,version,
                                       empty_days,padding,longname)
                print("Saved daily average aeronet parameters from %s to file %s" % (longname,filename))
                df_filelist.loc[key,"day_ave"] = filename
                    
            if fit_type == 'all' and fit_config[fit_type]:
                time_range = main_dict['time_range_aod_all']
                filename = 'aerosol_angstrom_params_all_' + time_range + '_' + key + '.dat'
                longname = station_dict[key]["name"]
                save_aeronet_dataframe(station_dict[key]['df_all'],save_path,filename,fit_type,version,
                                       empty_days,padding,longname)
                print("Saved all aeronet parameters from %s to file %s" % (station_dict[key]["name"],filename))
                df_filelist.loc[key,"all"] = filename
                        
    if type(mean_stats) == list:
        version = 3
        empty_days = {'aod': "See list of empty days in individual fit files",
                    'ssa': "See list of empty days in individual fit files"}
        longname = [station_dict[key]["name"] for key in mean_stats]
        filename = 'aerosol_angstrom_params_mean_day_ave_' + time_range + '_'\
            + mean_label + '.dat'
        save_aeronet_dataframe(main_dict['aeronet_mean_day'],save_path,filename,
                               fit_type,version,empty_days,padding,longname)
        print("Saved daily average mean parameters from %s to file %s" % (mean_stats,filename))
        df_filelist.loc["mean","day_ave"] = filename
            
        filename = 'aerosol_angstrom_params_mean_all_' + time_range + '_'\
            + mean_label + '.dat'        
        save_aeronet_dataframe(main_dict['aeronet_mean_all'],save_path,
                               filename,fit_type,version,empty_days,padding,longname)
        print("Saved all mean parameters from %s to file %s" % (mean_stats,filename))
        df_filelist.loc["mean","all"] = filename                        

    filename_list = "aerosol_filelist_" + description + ".dat"
    df_filelist.to_csv(os.path.join(save_path,filename_list),sep=' ',
                       header=df_filelist.columns.values,na_rep='nan')
    print("List of files written to %s" % filename_list)
    
      
#%%Main Program
#######################################################################
###                         MAIN PROGRAM                            ###
#######################################################################
def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("configfile", help="yaml file containing config")
    args = parser.parse_args()

    plt.ioff()
    plt.style.use('my_paper')
    
    #Get configuration file name
    config_filename = os.path.abspath(args.configfile)#""#
    
    #Read in values from configuration file
    config = load_yaml_configfile(config_filename)
    
    info = config["description"]
    print("Extracting Aeronet data for %s" % info)
    
    homepath = os.path.expanduser("~")
    
    #get path for loading data
    load_path = os.path.join(homepath,config["path_aerosol_data"])
    
    #Load stations and other info from config file
    aeronet_stats = config["aeronet_stations"]
    plot_paths = config["path_plots"]
    plot_paths["main"] = os.path.join(homepath,plot_paths["main"])
    save_path = os.path.join(homepath,config["path_fit_data"])
    plot_flag = config["plot_flag"] 
    curvature = config["curvature"]
    pad_flag = config["pad_values"]    
    mean_stats = config["mean_stations"]    
    resolution = config["timeres"]    
    fit_range = config["fit_range"]
    
    #Load Aeronet data and plot all values if required
    aeronet_dict = load_plot_aerosol_data(load_path,aeronet_stats,
                                          plot_paths,resolution,plot_flag,pad_flag)
    del aeronet_stats   
    
    for key in aeronet_dict["aero_stats"]:
        for fit_type in config["fit_config"]:
            if config["fit_config"][fit_type]:
                aeronet_dict["aero_stats"][key], xfit_data = angstrom_fit(fit_type,aeronet_dict["aero_stats"][key],
                            fit_range,plot_paths,plot_flag,curvature)
    
    aeronet_mean_all = pd.DataFrame()
    aeronet_mean_day = pd.DataFrame()
    if type(mean_stats) == list:
        for fit_type in config["fit_config"]:
            if config["fit_config"][fit_type]:
                aeronet_mean = angstrom_fit_mean(fit_type,aeronet_dict["aero_stats"],mean_stats,xfit_data,
                                                 fit_range,plot_paths,plot_flag,curvature)
                if fit_type == 'all':
                    aeronet_mean_all = aeronet_mean
                elif fit_type == 'day_ave':
                    aeronet_mean_day = aeronet_mean
            
    aeronet_dict.update({"aeronet_mean_all":aeronet_mean_all})
    aeronet_dict.update({"aeronet_mean_day":aeronet_mean_day})
        
    save_aerosol_files(save_path,aeronet_dict,mean_stats,config["fit_config"],pad_flag,info)

if __name__ == "__main__":
    main()
