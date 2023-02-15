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
import matplotlib.dates as mdates

from plotting_functions import confidence_band
import pandas as pd

from astropy.convolution import convolve, Box1DKernel
from scipy.stats import gaussian_kde
from matplotlib.gridspec import GridSpec
from copy import deepcopy

#import datetime

#%%Functions
def load_pyr2cf_inversion_results(info,rt_config,pyr_config,station_list,home):
    """
    Load results from inversion onto plane-of-array irradiance
    
    args:    
    :param info: string, descripton of current campaign
    :param rt_config: dictionary with current RT configuration
    :param pyr_config: dictionary with current pyranometer calibration configuration    
    :param station_list: list of PV station
    :param home: string with homepath
    
    out:
    :return pv_systems: dictionary of PV systems with data
    :return folder_label: string with name of folder where data is saved
    """
    
    mainpath = os.path.join(home,pyr_config['results_path']['main'],
                            pyr_config['results_path']['cloud_fraction'])
    
    #atmosphere model
    atm_geom_config = rt_config["disort_base"]["pseudospherical"]    
    if atm_geom_config == True:
        atm_geom_folder = "Pseudospherical"
    else:
        atm_geom_folder = "Plane-parallel"            
    
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

    sza_label = "SZA_" + str(int(pyr_config["sza_max"]["inversion"]))
    
    folder_label = os.path.join(mainpath,atm_geom_folder,disort_folder_label,
                                atm_folder_label,aero_folder_label,sza_label)
        
    filename = filename + info + '_disortres_' + theta_res + '_' + phi_res + '_'
    
    pv_systems = {}    
        
    #Choose which stations to load    
    if type(station_list) != list:
        station_list = [station_list]    
        if station_list[0] == "all":
            station_list = pyr_config["pv_stations"]
    
    for station in station_list:                
        #Read in binary file that was saved from pvcal_radsim_disort
        filename_stat = filename + station + '.data'
        try:
            with open(os.path.join(folder_label,filename_stat), 'rb') as filehandle:  
                # read the data as binary data stream
                (pvstat, dummy, dummy) = pd.read_pickle(filehandle)            
            pv_systems.update({station:pvstat})
            print('Data for %s loaded from %s, %s' % (station,folder_label,filename))
        except IOError:
            print('There is no simulation for %s' % station)   
            
    return pv_systems, folder_label

def plot_irradiance_ratio(key,pv_station,rt_config,pvcal_config,folder):
    """
    

    Parameters
    ----------
    key : string, name of PV station
    pv_station : dictionary with information and data on PV station
    rt_config : dictionary with radiative transfer configuration
    pvcal_config : dictionary with PV calibration configuration
    folder : string with name of folder to save plots

    Returns
    -------
    None.

    """
    
    plt.ioff()
    plt.style.use('my_paper')
    
    res_dirs = list_dirs(folder)
    savepath = os.path.join(folder,'Transmission_Plots')
    if 'Transmission_Plots' not in res_dirs:
        os.mkdir(savepath)        
        
    stat_dirs = list_dirs(savepath)
    savepath = os.path.join(savepath,key)
    if key not in stat_dirs:
        os.mkdir(savepath)
        
    for substat_type in pv_station["substations"]:    
        for substat in pv_station["substations"][substat_type]["data"]:
            substat_dirs = list_dirs(savepath)
            plotpath = os.path.join(savepath,substat)
            if substat not in substat_dirs:
                os.mkdir(plotpath)
            
            for year in pv_station["substations"][substat_type]["source"]:                                
                print('Generating plots for %s, %s' % (substat,year))
                timeres = pv_station["substations"][substat_type]["t_res_inv"]
                dfname = 'df_' + year.split('_')[-1] + '_' + timeres
                dfcosmo = 'df_cosmo_' + year.split('_')[-1]
                dataframe = pv_station[dfname].xs(substat,level='substat',axis=1)             
                
                #Keep only data for which an inversion exists
                dataframe = dataframe.loc[dataframe['P_meas_W'].notna()]
                #test_days = pv_station["sim_days"][year]
                
                test_days = pd.to_datetime(dataframe.index.date).unique().strftime('%Y-%m-%d')    
                
                Etotdown_cosmo = pv_station[dfcosmo][('Edirdown_Wm2','cosmo')] +\
                                    pv_station[dfcosmo][('Ediffdown_Wm2','cosmo')]
                #Shift COSMO irradiance data by 30 minutes
                Etotdown_cosmo.index = Etotdown_cosmo.index - pd.Timedelta('30min')
                
                for iday in test_days:
                    df_test_day = dataframe.loc[iday]
                    
                    fig, ax = plt.subplots(figsize=(9,8))
                    legstring = []
                    df_test_day.Etotpoa_pv_clear_Wm2.plot(ax=ax,legend=False,color='r',style='--')
                    df_test_day.Etotpoa_pv_inv.plot(ax=ax,legend=False,color='g')
                    confidence_band(ax,df_test_day.index,df_test_day.Etotpoa_pv_inv,
                                   df_test_day.error_Etotpoa_pv_inv,'g')                
                    
                    legstring.extend([r'$G_{\rm clear,[0.3,1.2]\mu m}^{\angle}$',
                               r'$G_{\rm PV-P,[0.3,1.2]\mu m}^{\angle}$'])
        
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
                    
                    Etotdown_cosmo.loc[iday].plot(ax=ax,legend=False,color='c')
                    legstring.extend([r'$G_{\rm COSMO}^{\downarrow}$'])
                    
                    ax.legend(legstring)
                    
                    datemin = np.datetime64(iday + ' 03:00:00')
                    datemax = np.datetime64(iday + ' 21:00:00')   
                    ax.set_xlim([datemin, datemax])
                    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  
                    
                    ax.set_xlabel('Time (UTC)')
                    ax.set_ylabel(r'Irradiance (W/m$^2$)')
                    ax.set_title('Irradiance ratio for ' + key + ' on ' + iday)
                    
                    plt.savefig(os.path.join(plotpath,'irrad_ratio_' + key + '_' 
                                            + substat + '_' + iday + '.png'))
                    plt.close(fig)
            
def import_apollo_cloud_data(key,pv_station,year,configfile,home):
    """
    Import cloud data from APOLLO retrieval, already split into stations
    by DLR
    
    args:
    :param key: string, name of station
    :param pv_station: dictionary with all info and data of pv station    
    :param year: string describing the data source
    :param configfile: string with name of configfile
    :param home: string with home path
    
    out:
    :return pv_station: dictionary of information and data on PV station
    """

    config = load_yaml_configfile(os.path.join(home,configfile))
    path_apollo = os.path.join(home,config["path_apollo"])

    #Read in satellite COD values    
    dataframe = pd.DataFrame()        
    
    dfs = [pd.read_csv(os.path.join(path_apollo,filename),sep='\s+',
           comment='#',header=None,na_values=-999, parse_dates={'Timestamp':[0,1,2,3]},
           date_parser=lambda a,b,c,d: pd.to_datetime(a+b+c+d,format="%Y%m%d%H%M")) for filename in 
           list_files(path_apollo) if key in filename]
    
    dataframe = pd.concat(dfs,axis=0)
    dataframe.index = dataframe["Timestamp"]
    dataframe.drop(columns="Timestamp",inplace=True)
    
    #Sort index - bug in data found (JB, 20.07.2021)
    dataframe.sort_index(axis=0,inplace=True)
    
    #Add offset to get true observation time
    dataframe.index = dataframe.index + pd.Timedelta("668S")
    
    dataframe.columns = ['type','cov','phase','cot_AP','ctt','scat_mix','snow_prob']
    
    
    dataframe["cot_AP"] = dataframe["cot_AP"]/100
    dataframe["ctt"] = dataframe["ctt"]/10
    
    #Create Multi-Index for cosmo data
    dataframe.columns = pd.MultiIndex.from_product([dataframe.columns.values.tolist(),['apollo']],
                                                                   names=['variable','substat'])       
                   
    #Assign to special cosmo dataframe, and join with main dataframe
    pv_station['df_apollo_' + year.split('_')[-1]] = dataframe
       
    return pv_station

def moving_average_std(input_series,data_freq,window_avg,nan_limit="10min"):
    """
    

    Parameters
    ----------
    input_series : series, input data
    data_freq : timedelta with time resolution
    window_avg : timedelta with width of averaging window
    nan_limit : string with limit for NANs        

    Returns
    -------
    dataframe with average and standard deviation of input series

    """        
    
    window_size = int(window_avg/data_freq) 

    #Calculate moving average with astropy convolution       
    # avg = pd.Series(data=convolve(input_series,Box1DKernel(window_size),nan_treatment='interpolate'),
    #                 index=input_series.index,name='avg_conv')
    
    # #Calculate standard deviation
    # std = pd.Series(np.sqrt(convolve(input_series.values**2,Box1DKernel(window_size),
    #              nan_treatment='interpolate') - avg**2),index=input_series.index,name='std_conv')
    edge = int(window_size/2.)
    # avg  = avg[edge:-edge]
    # std  = std[edge:-edge]
    
    int_nan_limit = int(pd.Timedelta(nan_limit)/data_freq)
    
    #alternative method with pandas
    #.\
    avg_alt = input_series.interpolate(method='linear',limit=int_nan_limit).\
        rolling(window=window_avg,min_periods=edge).\
            mean().rename('avg_pd')        #.shift(-edge)
    std_alt = input_series.interpolate(method='linear',limit=int_nan_limit).\
        rolling(window=window_avg,min_periods=edge).\
        std().rename('std_pd') #shift(-edge).
    
    dataframe = pd.concat([avg_alt,std_alt],axis=1) #avg,std,
    
    return dataframe

def average_cloud_fraction(dataframe,df_cloudcam,df_apollo,timeres_old,timeres_window,
                           timeres_cloudcam,timeres_apollo,substat):
    """
    Average the cloud fraction over a certain period of time, for each day
    
    args:
    :param dataframe: dataframe with cloud fraction and other parameters
    :param df_cloudcam: dataframe with cloud camera cloud fraction
    :param df_apollo: dataframe with APOLLO cloud data
    :param timeres_old: string with old time resolution (high resolution)
    :param timeres_window: string with size of window for moving average
    :param timeres_cloudcam: string with time res of cloud cam data
    :param timeres_apollo: string with time resolution of APOLLO data
    :param substat: string with name of substation    
    
    out:
    :return dataframe: dataframe with averaged cloud fraction
    :return df_cloudcam: cloudcam dataframe with averaged data
    :return df_combine_final: dataframe with combined data from inversion, cloud cam and APOLLO
    """
    
    timeres_data = pd.to_timedelta(timeres_old).seconds # measurement timeresolution in sec
    timeres_ave = pd.to_timedelta(timeres_window).seconds
    timeres_cc = pd.to_timedelta(timeres_cloudcam).seconds
    timeres_apng = pd.to_timedelta(timeres_apollo).seconds
    
    kernelsize = timeres_ave/timeres_data # kernelsize 
    box_kernel = Box1DKernel(kernelsize)     
    
    kernelsize_cloudcam = timeres_ave/timeres_cc
    box_kernel_cloudcam = Box1DKernel(kernelsize_cloudcam)
    
    kernelsize_apollo = timeres_ave/timeres_apng
    box_kernel_apng = Box1DKernel(kernelsize_apollo)
    
    colnames = dataframe.xs(substat,level="substat",axis=1).columns
    radtypes = []
    if "cloud_fraction_down" in colnames:
        radtypes.append("down")
    if "cloud_fraction_poa" in colnames:
        radtypes.append("poa")    
                
    days = pd.to_datetime(dataframe.index.date).unique().strftime('%Y-%m-%d')        
    
    dfs = []
    for iday in days:                        
        if "cloudcam" in df_cloudcam.columns.levels[1]:
            #Moving average for cloudcam
            cf_cc_avg = convolve(df_cloudcam.loc[iday,("cf_cloudcam","cloudcam")].values.flatten(),
                                 box_kernel_cloudcam,preserve_nan=True)
            
            #handle edges
            edge_cc = int(kernelsize_cloudcam/2.)
            cf_cc_avg  = cf_cc_avg[edge_cc:-edge_cc]
            index_cut_cc = df_cloudcam.loc[iday].index[edge_cc:-edge_cc]
            df_cloudcam.loc[index_cut_cc,(f"cf_cloudcam_{timeres_window}_avg","cloudcam")] = cf_cc_avg  
            
            df_combine = df_cloudcam.loc[index_cut_cc,(f"cf_cloudcam_{timeres_window}_avg","cloudcam")]                        
        else:
            df_combine = pd.DataFrame()
        
        #Moving average for apollo        
        cf_AP_avg = convolve(df_apollo.loc[iday,("cov","apollo")].values.flatten()/100,
                             box_kernel_apng,preserve_nan=True)
        
        #handle edges
        edge_AP = int(kernelsize_apollo/2.)
        cf_AP_avg  = cf_AP_avg[edge_AP:-edge_AP]
        index_cut_AP = df_apollo.loc[iday].index[edge_AP:-edge_AP]
        df_apollo.loc[index_cut_AP,(f"cf_apollo_{timeres_window}_avg","apollo")] = cf_AP_avg  
        
        #df_combine = df_cloudcam.loc[index_cut_cc,(f"cf_cloudcam_{timeres_window}_avg","cloudcam")]                        
        
        #Moving average for retrieved cloud fraction
        for radtype in radtypes:
            df_day = deepcopy(dataframe.loc[iday,(f"cloud_fraction_{radtype}",substat)])
            df_day.loc[df_day < 0] = 0.5 #np.nan #0.5 #np.nan #
            cf_avg = convolve(df_day.values.flatten(), box_kernel,preserve_nan=True)
                    
            # handle edges
            edge = int(kernelsize/2.)
            cf_avg  = cf_avg[edge:-edge]
            index_cut = dataframe.loc[iday].index[edge:-edge]
            dataframe.loc[index_cut,(f"cf_{radtype}_{timeres_window}_avg",substat)] = cf_avg                
            
            df_avg_std = moving_average_std(df_day,pd.Timedelta(timeres_old),pd.Timedelta(timeres_window))
            dataframe.loc[df_day.index,(f"cf_{radtype}_{timeres_window}_avg_alt",substat)] = df_avg_std["avg_pd"]
            
            if timeres_data <= timeres_cc:                                
                df_cf_reindex = dataframe.loc[index_cut,(f"cf_{radtype}_{timeres_window}_avg",substat)].\
                                reindex(pd.date_range(start=df_combine.index[0].round(timeres_cloudcam),
                                end=df_combine.index[-1].round(timeres_cloudcam),freq=timeres_cloudcam),
                                method='nearest',tolerance=pd.Timedelta(timeres_cloudcam)/12) #.loc[day.strftime("%Y-%m-%d")]        
                
                df_combine = pd.concat([df_combine,df_cf_reindex],axis=1)
                
            elif timeres_old == timeres_cc:
                df_combine = pd.concat([df_combine,dataframe.loc[index_cut,
                         (f"cf_{radtype}_{timeres_window}_avg",substat)]],axis=1)
                
        dfs.append(df_combine)
            
    #Sort multi-index (makes it faster)
    dataframe.sort_index(axis=1,level=1,inplace=True)
    
    df_combine_final = pd.concat(dfs,axis=0)
    
    return dataframe, df_cloudcam, df_combine_final

def average_plot_cloud_fraction(key,pv_station,year,pyr_config,folder,plot_flags):
    """
    

    Parameters
    ----------
    key : string, name of PV station
    pv_station : dictionary with information and data from PV station
    year : string, year under consideration
    pyr_config : dictionary with pyranometer configuration
    folder : string with folder for saving plots
    plot_flags : dictionary with booleans for plotting

    Returns
    -------
    None.

    """
    
    plt.ion()
    plt.style.use('my_paper')
    
    res_dirs = list_dirs(folder)
    savepath = os.path.join(folder,'Cloud_Fraction_Plots')
    if 'Cloud_Fraction_Plots' not in res_dirs:
        os.mkdir(savepath)        
        
    stat_dirs = list_dirs(savepath)
    savepath = os.path.join(savepath,key)
    if key not in stat_dirs:
        os.mkdir(savepath)
        
    plot_dirs = list_dirs(savepath)
    savepath = os.path.join(savepath,'Comparison')
    if 'Comparison' not in plot_dirs:
        os.mkdir(savepath)
        
    timeres_cf = ["15min"] #,"30min","60min"]
    timeres_cloudcam = "1min"
    timeres_apollo = "15min"
    radtypes = ["down","poa"]
    cs_threshold = pyr_config["cloud_fraction"]["cs_threshold"]
        
    df_cloudcam_name = f'df_{year.split("_")[-1]}_{timeres_cloudcam}'
    df_combine_name = f'df_cf_combine_{year.split("_")[-1]}_{timeres_cloudcam}'
    
    dfs = []
    for substat in pv_station["substations"]: #["CMP11_Horiz","suntracker"]: #
        print('Calculating average cloud fraction for %s, %s for each day' % (key,substat))
        substat_dirs = list_dirs(savepath)
        plotpath = os.path.join(savepath,substat)
        if substat not in substat_dirs:
            os.mkdir(plotpath)
                
        timeres = pv_station["substations"][substat]["t_res_inv"]
        dfname = f'df_{year.split("_")[-1]}_{timeres}'        
        df_apollo = pv_station[f"df_apollo_{year.split('_')[-1]}"]
        
        radname = pv_station["substations"][substat]["name"]
        
        #Calculate moving average
        for timeres_cf_pyr in timeres_cf:
            pv_station[dfname], pv_station[df_cloudcam_name], df_combine \
                = average_cloud_fraction(pv_station[dfname],
                pv_station[df_cloudcam_name],df_apollo,timeres,timeres_cf_pyr,
                timeres_cloudcam,timeres_apollo,substat)                         

            #df_combine.loc[:,~df_combine.columns.duplicated()]
            dfs.append(df_combine)
        
        if plot_flags["compare"]:
            print(f"Plotting cloud fraction comparison for {key}, {substat} for each day")
            #Get data for plotting
            dataframe = pv_station[dfname].xs(substat,level='substat',axis=1)
            df_cloudcam = pv_station[df_cloudcam_name]\
                .xs("cloudcam",level='substat',axis=1)                 
            
            test_days = pd.to_datetime(dataframe.index.date).unique().strftime('%Y-%m-%d')    
            
            #test_days = [day.strftime('%Y-%m-%d') for day in pvrad_config["falltage"][year]]
            for iday in test_days:
                df_test_day = dataframe.loc[iday]
                
                fig = plt.figure(figsize=(14,8))
                        
                # Definieren eines gridspec-Objekts
                gs = GridSpec(3, 1, height_ratios=[1.5, 1, 1.5], hspace=0.08)
                
                ax = fig.add_subplot(gs[0])  
                if plot_flags["titles"]:
                    ax.set_title(f'Irradiance and cloud fraction for {key}, {substat} on {iday}')                      
                
                for radtype in radtypes:
                    if radtype == "poa":
                        radlabel = "\\angle"
                    
                    elif radtype == "down":
                        radlabel = "\\downarrow"                            
                                    
                    if radtype in radname:                                                        
                        if radtype == "down" and "Pyr" in substat:
                            radname_plot = radname.replace("poa","down")
                        else:
                            radname_plot = radname
                            
                        ax.plot(df_test_day.index,df_test_day[radname_plot],
                                label=rf'$G_{{\rm meas}}^{{{radlabel}}}$')
                        
                        ax.plot(df_test_day.index,df_test_day[f"Etot{radtype}_clear_Wm2"],
                                label=rf'$G_{{\rm clear}}^{{{radlabel}}}$',linestyle='--')
                
                ax.legend()
                ax.set_ylabel(r'Irradiance (W/m$^2$)')
                            
                ax2 = fig.add_subplot(gs[1])
                ax3 = fig.add_subplot(gs[2])
                
                ax2.axhline(y = cs_threshold[0], color ="green", linestyle ="--")
                ax2.axhline(y = cs_threshold[1], color ="green", linestyle ="--")
                #ax2.axhline(y = 1.1, color ="green", linestyle =":")
                
                for radtype in radtypes:
                    if f"k_index_{radtype}" in df_test_day.columns:
                        ax2.plot(df_test_day.index,df_test_day[f"k_index_{radtype}"],
                                    label=rf"$k_i$ ({radtype})",linestyle='--')
                        ax2.plot(df_test_day.index,df_test_day[f"cloud_fraction_{radtype}"],
                                    label=f"cf ({radtype})")                                                        
                
                        for tres_cf in timeres_cf:
                            ax3.plot(df_test_day.index,df_test_day[f"cf_{radtype}_{tres_cf}_avg"],
                                     label=rf"$\langle$cf$\rangle_{{\rm {tres_cf}}}$ ({radtype})")
                            ax3.plot(df_test_day.index,df_test_day[f"cf_{radtype}_{tres_cf}_avg_alt"],
                                       label=rf"$\langle$cf$\rangle_{{\rm {tres_cf}}}$ ({radtype}) (pd)")
                
                ax3.plot(df_cloudcam.loc[iday].index,df_cloudcam.loc[iday,"cf_cloudcam"],color='gray',
                          label="cf (cloudcam)",linestyle='--')
                for tres_cf in timeres_cf:
                    ax3.plot(df_cloudcam.loc[iday].index,df_cloudcam.loc[iday,f"cf_cloudcam_{tres_cf}_avg"],
                              label=rf"$\langle$cf$\rangle_{{\rm {tres_cf}}}$ (cloudcam)")
                    
                ax3.plot(df_apollo.loc[iday].index,df_apollo.loc[iday,("cov","apollo")]/100,
                          label="cf (APNG)",linestyle=":")
                
                
                ax2.legend()
                ax3.legend(fontsize=12)
                
                ax3.set_xlabel('Time (UTC)')
                ax2.set_ylabel(r'$k_i$ / cf')
                ax2.set_ylim([0,2])    
                ax3.set_ylabel('cf')
                ax3.set_ylim([0,1])
                
                datemin = pd.Timestamp(df_test_day.index[0]) #- pd.Timedelta("30min"))
                datemax = pd.Timestamp(df_test_day.index[-1]) #+ pd.Timedelta("30min"))      
                for axis in [ax,ax2,ax3]:
                    axis.set_xlim([datemin,datemax])        
                    axis.xaxis.set_major_locator(mdates.HourLocator(interval=2))
                    axis.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))      
                    
                ax.set_xticklabels('')     
                ax2.set_xticklabels('')     
                
                plt.savefig(os.path.join(plotpath,'cloud_fraction_' + key + '_' 
                                         + substat + '_' + iday + '.png'),bbox_inches = 'tight')
                #plt.close(fig)
            
    pv_station[df_combine_name] = pd.concat(dfs,axis=1)
    
    #Sort multi-index (makes it faster)
    pv_station[df_combine_name].sort_index(axis=1,level=1,inplace=True)    
    pv_station[df_combine_name] = pv_station[df_combine_name].loc[:,\
                              ~pv_station[df_combine_name].columns.duplicated()]

def scatter_plots_cloud_fraction(key,pv_station,year,pyr_config,folder,plot_flags):
    """
    

    Parameters
    ----------
    key : string with name of PV station
    pv_station : dictionary with information and data from PV station
    year : string with year under consideration
    pyr_config : dictionary with pyranometer configuration
    folder : string with name of folder for saving plots
    plot_flags : dictionary with booleans for plotting

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
        
    plot_dirs = list_dirs(savepath)
    savepath = os.path.join(savepath,'Scatter')
    if 'Scatter' not in plot_dirs:
        os.mkdir(savepath)
        
    timeres_cf = ["30min","60min"]
    timeres_cloudcam = "1min"
    radtypes = ["down","poa"]
    cs_threshold = pyr_config["cloud_fraction"]["cs_threshold"]
        
    df_combine_name = f'df_cf_combine_{year.split("_")[-1]}_{timeres_cloudcam}'
          
    days = pd.to_datetime(pv_station[df_combine_name].index.date).unique().strftime('%Y-%m-%d')        
    
    #Reindex dataframes for comparison    
    for timeres in timeres_cf:    
        df_avg_name = f"df_cf_combine_{timeres}_avg"
        
        dfs = []        
        for iday in days:
            df_combine_day = pv_station[df_combine_name].filter(regex=f'{timeres}', axis=1).loc[iday]
            dfs.append(df_combine_day.reindex(pd.date_range(start=df_combine_day.index[0].round(timeres),
                            end=df_combine_day.index[-1].round(timeres),freq=timeres),
                            method='nearest',tolerance=pd.Timedelta(timeres)/12))
            
        pv_station[df_avg_name] = pd.concat(dfs,axis=0)
        
    if plot_flags["scatter"]:        
        #Go through substations and make scatter plots
        for substat in ["CMP11_Horiz","suntracker"]:        
            print(f"Plotting scatter plots for {key}, {substat}")
            fig, axs = plt.subplots(1,len(timeres_cf),sharey='row',sharex='col',figsize=(16,9))
            
            for n, timeres in enumerate(timeres_cf):                        
                df_avg = pv_station[f"df_cf_combine_{timeres}_avg"].loc[:,
                            [(f"cf_cloudcam_{timeres}_avg","cloudcam"),
                             (f"cf_down_{timeres}_avg",substat)]].dropna(how='any',axis=0)
                if len(df_avg > 0):
                    cf_cc = df_avg[(f"cf_cloudcam_{timeres}_avg","cloudcam")]
                    cf_pyr = df_avg[(f"cf_down_{timeres}_avg",substat)]
                    xy = np.vstack([cf_cc,cf_pyr])
                    z = gaussian_kde(xy)(xy)
                    idx = z.argsort()
                    cf_cc_sort, cf_pyr_sort, z = cf_cc[idx], cf_pyr[idx], z[idx]
                    
                    sc = axs.flat[n].scatter(cf_cc_sort,cf_pyr_sort, s=40, c=z, cmap="jet")            
                    axs.flat[n].set_title(f"{timeres} average"),
                                         #xy=(0.02,0.96),xycoords='axes fraction',
                                         #fontsize=16)        
                
            for ax in axs.flat:
                ax.set_xlim([0,1])
                ax.set_ylim([0,1])
                ax.set(adjustable='box', aspect='equal')        
                ax.plot([0, 1], [0, 1], ls = '--', transform=ax.transAxes,c='r')                
             
            cb = plt.colorbar(sc, cax = fig.add_axes([0.92, 0.22, 0.02, 0.67]),
                              ticks=[np.min(z), np.max(z)], pad=0.05)
            cb.set_ticklabels(["Low", "High"])  # horizontal colorbar
            cb.set_label("Frequency", labelpad=-20, fontsize=14)
            
            fig.subplots_adjust(wspace=0.07)
            fig.subplots_adjust(top=1.01)   
                        
            axs.flat[0].set_xlabel("cloud fraction (cloudcam)",position=[1.05,0])
            axs.flat[0].set_ylabel(f"cloud fraction ({substat})")
            
            if plot_flags["titles"]: 
                fig.suptitle(f"Cloud fraction comparison, {key}, {substat}, {year.split('_')[1]}",fontsize=20)#,
                                      #position=[1.05,1])   
            
            #fig.tight_layout()
            plt.savefig(os.path.join(savepath,f'scatter_cloud_fraction_{key}_{substat}_{year.split("_")[1]}.png'),
                        bbox_inches = 'tight')
            plt.close(fig)

def histogram_cloud_fraction(key,pv_station,year,pyr_config,folder,plot_flags):
    """
    

    Parameters
    ----------
    key : string with name of PV station
    pv_station : dictionary with information and data from PV station
    year : string with year under consideration
    pyr_config : dictionary with pyranometer configuration
    folder : string with name of folder for saving plots
    plot_flags : dictionary with booleans for plotting

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
        
    plot_dirs = list_dirs(savepath)
    savepath = os.path.join(savepath,'Stats')
    if 'Stats' not in plot_dirs:
        os.mkdir(savepath)
        
    timeres_cf = ["30min","60min"]
    timeres_cloudcam = "1min"
    radtypes = ["down","poa"]
    cs_threshold = pyr_config["cloud_fraction"]["cs_threshold"]                     
               
    
    #Go through substations and make scatter plots
    for substat in ["CMP11_Horiz","suntracker"]:  
        print(f"Plotting histograms for {key}, {substat}")
        fig, axs = plt.subplots(1,len(timeres_cf),sharex='col',figsize=(16,9))
        
        for n, timeres in enumerate(timeres_cf):                        
            df_avg = pv_station[f"df_cf_combine_{timeres}_avg"].loc[:,
                        [(f"cf_cloudcam_{timeres}_avg","cloudcam"),
                         (f"cf_down_{timeres}_avg",substat)]].dropna(how='any',axis=0)
            if len(df_avg > 0):
                bins = np.arange(0.,1.+0.05,0.05)
                print(bins)

                axs.flat[n].hist(df_avg[(f"cf_cloudcam_{timeres}_avg","cloudcam")],bins=bins,
                                 alpha = 0.5,label="cloudcam")
                axs.flat[n].hist(df_avg[(f"cf_down_{timeres}_avg",substat)],bins=bins,
                                 alpha = 0.5,label=substat)
                axs.flat[n].set_title(f"{timeres} average"),
                                     #xy=(0.02,0.96),xycoords='axes fraction',
                                     #fontsize=16)                        
        for ax in axs.flat:
            ax.set_xlim([0,1])
            #ax.set_ylim([0,1])
            #ax.set(adjustable='box', aspect='equal')                    
            ax.legend(loc="upper right")
         
        #fig.subplots_adjust(wspace=0.07)
        #fig.subplots_adjust(top=1.01)   
                    
        axs.flat[0].set_xlabel("cloud fraction",position=[1.05,0])        
        axs.flat[0].set_ylabel("Frequency")
        axs.flat[1].set_ylabel("Frequency")
         #, bbox_to_anchor=(0.82,0.45))    
        #axs.flat[0].legend()
        
        if plot_flags["titles"]: 
            fig.suptitle(f"Cloud fraction histogram, {key}, {substat}, {year.split('_')[1]}",
                         fontsize=20)#,
                                  #position=[1.05,1])   
        
        #fig.tight_layout()
        plt.savefig(os.path.join(savepath,f'histogram_cloud_fraction_{key}_{substat}_{year.split("_")[1]}.png'),
                    bbox_inches = 'tight')
        plt.close(fig)

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
   
config_filename = "config_PYRCAL_2018_messkampagne.yaml" #os.path.abspath(args.configfile)
 
config = load_yaml_configfile(config_filename)

#Load pyranometer
pyr_config = load_yaml_configfile(config["pyr_configfile"])

#Load radiative transfer configuration
rt_config = load_yaml_configfile(config["rt_configfile"])

homepath = os.path.expanduser('~')

if args.station:
    stations = args.station
    if stations[0] == 'all':
        stations = 'all'
else:
    #Stations for which to perform inversion
    stations = "MS_02" #pvrad_config["stations"]

#%%Load inversion results
#Load calibration results, including DISORT RT simulation for clear sky days and COSMO data.
print('Loading PYR2CF results')

pvsys, results_folder = \
load_pyr2cf_inversion_results(config["description"], rt_config, pyr_config, stations, homepath)

print('Plotting cloud fraction results of PYR2CF for %s' % stations)

plot_flags = config["plot_flags"]

#%%Plotting
plt.close('all')
year = f"mk_{config['description'].split('_')[1]}"

for key in pvsys:
    #Plot irradiance ratio
    # print('Plotting irradiance ratio for %s each day' % key)
    # plot_irradiance_ratio(key,pvsys[key],rt_config,pvcal_config,results_folder)
    print(f'Importing data from APOLLO retrieval for {key}, {year}')
    pvsys[key] = import_apollo_cloud_data(key,pvsys[key],year,pyr_config["seviripvcod_configfile"],
                                          homepath)

    #Plot cloud fraction    
    average_plot_cloud_fraction(key,pvsys[key],year,pyr_config,results_folder,plot_flags)
    
    #Scatter plots
    scatter_plots_cloud_fraction(key,pvsys[key],year,pyr_config,results_folder,plot_flags)
    
    if plot_flags["stats"]:
        #Histograms
        histogram_cloud_fraction(key,pvsys[key],year,pyr_config,results_folder,plot_flags)

