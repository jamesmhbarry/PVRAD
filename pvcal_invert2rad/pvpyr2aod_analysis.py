#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 10:33:16 2021

@author: james
"""

#%% Preamble
import os
import numpy as np
import ephem
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# from matplotlib.ticker import ScalarFormatter
#from matplotlib.gridspec import GridSpec
from pvcal_forward_model import azi_shift
import pandas as pd
from file_handling_functions import *
from plotting_functions import confidence_band
# from astropy.convolution import convolve, Box1DKernel
# from mpl_toolkits.axes_grid1 import make_axes_locatable

def generate_folder_names_pvpyr2aod(rt_config,pvcal_config):
    """
    Generate folder structure to retrieve PYR2AOD simulation results
    
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
    filename = 'aod_fit_results_'
    
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

    model = pvcal_config["inversion"]["power_model"]
    eff_model = pvcal_config["eff_model"]
    
    T_model = pvcal_config["T_model"]["model"]

    folder_label = os.path.join(atm_geom_folder,wvl_folder_label,disort_folder_label,
                                atm_folder_label,aero_folder_label,sza_label,
                                model,eff_model,T_model)
        
    return folder_label, filename, (theta_res,phi_res)

def load_pvpyr2aod_results(rt_config,pyr_config,pvcal_config,pvrad_config,
                           info,station_list,home):
    """
    Load results from AOD simulations with DISORT
    
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
    :return pyr_folder_label: string, pyranometer folder label
    :return pv_folder_label: string, pv folder label
    """
    
    mainpath = os.path.join(home,pyr_config['results_path']['main'],
                                   pyr_config['results_path']['optical_depth'])     
    
    folder_label, filename, (theta_res,phi_res) = \
    generate_folder_names_pvpyr2aod(rt_config,pvcal_config)

    pyr_folder_label = os.path.join(mainpath,folder_label)    
    
    filename = filename + info + '_'#+ '_disortres_' + theta_res + '_' + phi_res + '_'
    
    #Choose which stations to load    
    if type(station_list) != list:
        station_list = [station_list]
        if station_list[0] == "all":
            station_list = list(pyr_config["pv_stations"].keys())
            station_list.extend(list(pvrad_config["pv_stations"].keys()))
            station_list = list(set(station_list))
            station_list.sort()
    
    year = info.split('_')[1]
    
    pv_systems = {}
    
    for station in station_list:     
        if station in pyr_config["pv_stations"]:
            timeres = "1min"
            #Read in binary file that was saved from pvcal_radsim_disort
            filename_stat = filename + station + f'_{timeres}.data'
            try:
                with open(os.path.join(pyr_folder_label,filename_stat), 'rb') as filehandle:  
                    # read the data as binary data stream
                    pvstat = pd.read_pickle(filehandle)            
                
                if station not in pv_systems:
                    pv_systems.update({station:pvstat})
                else:
                    if timeres not in pv_systems[station]["timeres"]:
                            pvstat["timeres"].extend(pv_systems[station]["timeres"])
                        
                    pv_systems[station] = merge_two_dicts(pv_system[station], pvstat)
                
                print('Loaded AOD data for %s at %s in %s' % (station,timeres,year))
            except IOError:
                print('There is no AOD retrieval for pyranometers at %s, %s in %s' % (station,timeres,year))                   
        if station in pvrad_config["pv_stations"]:
            for substat_type in pvrad_config["pv_stations"][station]:
                timeres = pvrad_config["pv_stations"][station][substat_type]["t_res_inv"]
                
                #Read in binary file that was saved from pvcal_radsim_disort
                filename_stat = filename + station + f'_{timeres}.data'
                try:
                    with open(os.path.join(pyr_folder_label,filename_stat), 'rb') as filehandle:  
                        # read the data as binary data stream
                        pvstat = pd.read_pickle(filehandle)  
                        
                    if station not in pv_systems:
                        pv_systems.update({station:pvstat})
                    else:
                        if timeres not in pv_systems[station]["timeres"]:
                            pvstat["timeres"].extend(pv_systems[station]["timeres"])
                        
                        pv_systems[station] = merge_two_dicts(pv_systems[station], pvstat)
                
                    print('Loaded data for %s at %s in %s' % (station,timeres,year))
                except IOError:
                    print('There is no AOD retrieval for PV systems at %s, %s in %s' % (station,timeres,year)) 
                                            
    results_path = os.path.join(home,pyr_config["results_path"]["main"],
                                pyr_config["results_path"]["optical_depth"])
    pyr_folder_label = os.path.join(results_path,folder_label)            
            
    results_path = os.path.join(home,pvrad_config["results_path"]["main"],
                                pvrad_config["results_path"]["optical_depth"])
    pv_folder_label = os.path.join(results_path,folder_label)
            
    return pv_systems, pyr_folder_label, pv_folder_label

def load_pmaxdoas_results(station,timeres,description,path):
    """
    
    Load PMAX DOAS Aerosol results

    Parameters
    ----------
    station : string, name of station
    timeres : string, time resolution of data
    description : string, description of current campaign
    path : string, path where data is saved

    Returns
    -------
    lw_pvstat : dictionary  with information and data from LW pv station

    """
   
    
    filename = description + '_' + station[0] + '_' + timeres + ".data"
    
    files = list_files(path)    
        
    if filename in files:        
        with open(os.path.join(path,filename), 'rb') as filehandle:  
            (lw_pvstat, info) = pd.read_pickle(filehandle)  
        
        print('LW data from ' + station[0] + ' loaded from %s' % filename)
        return lw_pvstat
    else:
        print('Required file not found')
        return None  

def moving_average_std(input_series,data_freq,window_avg):
    """
    

    Parameters
    ----------
    input_series : series with input data
    data_freq : timedelta, data frequency
    window_avg : timedelta, width of averaging window        

    Returns
    -------
    dataframe with average and standard deviation

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
    
    #alternative method with pandas
    avg_alt = input_series.interpolate(method='linear',limit=int(edge/2)).\
        rolling(window=window_avg,min_periods=edge).\
            mean().shift(-edge).rename('avg_pd')        
    std_alt = input_series.interpolate(method='linear',limit=int(edge/2)).\
        rolling(window=window_avg,min_periods=edge).\
        std().shift(-edge).rename('std_pd')
    
    dataframe = pd.concat([avg_alt,std_alt],axis=1) #avg,std,
    
    return dataframe

def get_sun_position(dataframe,lat_lon,sza_limit):
    """
    Using PyEphem to calculate the sun position
    
    args:    
    :param dataframe: pandas dataframe for which to calculate sun position
    :param lat_lon: list of floats, coordinates
    :param sza_limit: float defining maximum solar zenith angle for simulation
    
    out:
    :return: dataframe with sun position
    
    """        
    
    len_time = len(dataframe)
    index_time = dataframe.index

    # initalize observer object
    observer = ephem.Observer()
    observer.lat = np.deg2rad(lat_lon[0])
    observer.lon = np.deg2rad(lat_lon[1])

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
    if sza_limit:    
        dataframe = dataframe.loc[dataframe['sza'] <= sza_limit]        
        
    return dataframe

def combine_aeronet(year,pv_station,str_window_avg):
    """

    Parameters
    ----------
    year : string with year under consideration
    pv_station : dictionary with information and data from PV station
    str_window_avg : string with width of averaging window

    Returns
    -------
    pv_station : dictionary with information and data from PV station

    """
        
    #List for comparison of different timeres
    dfs_compare_list = []         
    
    for timeres in pv_station["timeres"]:
        df_result = pv_station[f"df_aodfit_pyr_pv_{year.split('_')[1]}_{timeres}"] 
        #Size of moving average window
        window_avg = pd.Timedelta(str_window_avg)  
                        
        #List for concatenating days
        dfs = []
        
        #List for moving averages
        dfs_avg_std = []    
        #2. Calculate moving average of Aeronet data        
        for day in pd.to_datetime(df_result.index.date).unique():                                            
            df_avg_std = moving_average_std(df_result.loc[day.strftime("%Y-%m-%d"),("AOD_500","Aeronet")],
                                      timeres, window_avg) 
            
            dfs_avg_std.append(df_avg_std)            
            
            df_reindex_60 = df_avg_std.reindex(pd.date_range(start=df_avg_std.index[0].round(window_avg),
                                                     end=df_avg_std.index[-1].round(window_avg),freq=window_avg),
                                                     method='nearest',tolerance='5T').loc[day.strftime("%Y-%m-%d")]        
            dfs.append(df_reindex_60)       
        
        df_total = pd.concat(dfs_avg_std,axis=0)
        df_result[(f"AOD_500_{str_window_avg}_avg","Aeronet")] = df_total["avg_pd"]
        df_result[(f"AOD_500_{str_window_avg}_std","Aeronet")] = df_total["std_pd"]
        df_result.sort_index(axis=1,level=1,inplace=True)   
        
        #Assign reindexed values to comparison list
        df_compare = pd.concat(dfs,axis=0)
        df_compare.columns = pd.MultiIndex.from_product([[f"AOD_500_{str_window_avg}_avg",f"AOD_500_{str_window_avg}_std"]#,
                                                        ,['Aeronet']],names=['variable','substat'])                       
        dfs_compare_list.append(df_compare)                        

    pv_station[f"df_compare_{year.split('_')[1]}"] = pd.concat(dfs_compare_list,axis=1)    
    
    pv_station[f"df_compare_{year.split('_')[1]}"] = get_sun_position(
        pv_station[f"df_compare_{year.split('_')[1]}"],pv_station["lat_lon"],None)
        
    return pv_station

def plot_aod_lut(name,substat,df_day_result,day,radtype,
                 radname,errorname,radlabel,titlelabel,df_day_aeronet,
                 plotpath,titleflag=True):
    """
    

    Parameters
    ----------
    name : string, name of system
    substat : string, name of substation
    df_day_result : dataframe with result from specific day
    day : string, day under consideration
    radtype : string, radiation type (poa or down)
    radname : string, name of irradiance variable
    errorname : string, name of error vector
    radlabel : string, label for latex plot labels
    titlelabel : string, plot title label
    df_day_aeronet : dataframe, aeronet data from specific day
    plotpath : string, path to save plots
    titleflag : boolean, whether to add title. The default is True.

    Returns
    -------
    None.

    """
    
    plt.ioff()
    plt.close('all')
    plt.style.use("my_paper")
            
    fig = plt.figure(figsize=(15, 7))
    if titleflag:
        fig.suptitle(f"AOD retrieved from {titlelabel} irradiance at {name.replace('_','')}, {substat.replace('_',' ')} on {day:%Y-%m-%d}", fontsize=16)
    
    ax = fig.add_subplot(1, 2, 1)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))      

    ## first the simulated
     #aod_range = np.linspace(0.0, 1.0, 5)
    lut_range = df_day_result[f'AOD_{radtype}_table'].mean()[:,0]
    for aod_index, aod_value in enumerate(lut_range):
        irrad_simul, d_irrad_simul = [], []
        for time in df_day_result.index:
            if type(df_day_result.loc[time.strftime("%Y-%m-%d %H:%M:%S"),
                      f'AOD_{radtype}_table']) != np.ndarray:
                irrad_simul.append(np.nan)
                d_irrad_simul.append(np.nan)
            else:                
                # reads out the dataframe containing the aod / irrad lookuptable for each timestep
                
                OD_array, F_array = df_day_result.loc[time.strftime("%Y-%m-%d %H:%M:%S"),
                                  f'AOD_{radtype}_table'].transpose()  
                        
                F_value = F_array[aod_index]
                #if abs(F_value) > 1000: F_value = np.nan
                irrad_simul.append(F_value)
                d_irrad_simul.append(F_value*0.02)
    
        if aod_index%2 == 0 or aod_index == len(lut_range) - 1:
            ax.plot(df_day_result.index, irrad_simul,
                label= r"$\tau_\mathrm{{a}}={:.2f}$".format(aod_value), ls="--")
            confidence_band(ax,df_day_result.index, irrad_simul, d_irrad_simul)
            #if settings['print_flag'] == True: print(f"plotting the line of simulated values: t, F: \n {timeindex},{irrad_simul}")

    # then the measured

    #dF_meas = 0.02*irradiance_meas
    dF_meas = df_day_result.loc[:,errorname]
    df_plot = df_day_result.loc[:,radname]
    
    ax.plot(df_plot.index,df_plot,label=r"$G^{}_\mathrm{{pyr,{},{}}}$"
             .format(radlabel,name.replace("_",""),substat.replace('_',' ')))
    confidence_band(ax,df_plot.index,df_plot, dF_meas)
    ax.legend(loc = "best")
    ax.set_ylim(0,1200)
    ax.set_ylabel(r"Irradiance $G^{}_\mathrm{{pyr}}$ {}".format(radlabel,plotlabels["wmsq"]))
    ax.set_xlabel(r"Time (UTC)")
    ax.legend(loc = "best")
    
    # plot the retrieved aod
    ax2 = fig.add_subplot(1,2,2)
    datemin = pd.Timestamp(df_day_result.index[0] - pd.Timedelta('30T'))
    datemax = pd.Timestamp(df_day_result.index[-1] + pd.Timedelta('30T'))                        
    ax2.set_xlim([datemin, datemax])
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))      
    # Todo: Do a subplot routine to combine this with the other plot!!
    
    ax2.plot(df_day_result.index , df_day_result[f"AOD_500_{radtype}_inv"], 
             label = name.replace("_",""))
    confidence_band(ax2,df_day_result.index , df_day_result[f"AOD_500_{radtype}_inv"],\
                    df_day_result[f"error_AOD_500_{radtype}_inv"])            

    ax2.set_ylabel(plotlabels["AOD500"])
    #ax2.set_ylim([0,150])
    
    ax2.plot(df_day_aeronet.index,df_day_aeronet["AOD_500"])                        
    max_aod = np.max(df_day_result[f"AOD_500_{radtype}_inv"].max())            
        
    ax2.set_ylim([0,max_aod])
    ax2.set_xlabel(r"Time (UTC)")
    ax2.legend([r'$\tau_\mathrm{a,500nm,' + name.replace("_","") + 
               ',' + substat.replace('_',' ') + ' }$ ',r'$\tau_\mathrm{a,500nm,AERONET}$']) #,r'$\tau_\mathrm{wc,600nm,COSMO}$'])
        
    fig.tight_layout()
    fig.subplots_adjust(top=0.94)
    
    plt.savefig(os.path.join(plotpath,"AOD_lut_from_Etot{}_{}_{}_{}.png".format(
        radtype,name,substat,day.strftime("%Y-%m-%d")))) #, dpi = 300)
    plt.close(fig)
    
def scatter_plot_aod(xvals,yvals,cvals,labels,titlestring,figname,
                     aod_range,plot_style,title=True,logscale=False):
    """
    
    Parameters
    ----------
    xvals : vector of floats for scatter plot (x)
    yvals : vector of floats for scatter plot (y)
    cvals : vector of floats for scatter plot (z)
    labels : list of labels for plot axes
    titlestring : string with title for plot
    figname : string with name of figure for saving
    aod_range : list with min and max for axes
    plot_style : string, name of plot style
    title : boolean, whether to add title to plot. The default is True
    logscale : boolean, whether to use log scale in plots. The default is True.

    Returns
    -------
    figure handle

    """
        
    plt.ioff()
    #plt.close('all')
    plt.style.use(plot_style)

    fig, ax = plt.subplots(figsize=(9,8))           
            
    sc = ax.scatter(xvals,yvals,c=cvals,cmap='jet')        

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])

    ax.set_xlim(aod_range)
    ax.set_ylim(aod_range)
    if logscale:
        ax.set_yscale('log')
        ax.set_xscale('log')        
        for axis in [ax.xaxis, ax.yaxis]:
            formatter = FuncFormatter(lambda y, _: '{:.16g}'.format(y))
            axis.set_major_formatter(formatter) 
            # axis.set_major_formatter(ScalarFormatter())
            #  axis.set_major_formatter(FormatStrFormatter("%.2f"))               
        
    ax.plot([0, 1], [0, 1], ls = '--', transform=ax.transAxes,c='r')
    
    cb = plt.colorbar(sc)
    cb.set_label(labels[2])
    
    if title: ax.set_title(titlestring)
    fig.tight_layout()
    plt.savefig(figname)   
    plt.close(fig)
    
    return fig

def scatter_plot_grid_3(plot_vals_dict,sources,aod_range,plot_style,
                        title=True,logscale=False):
    """
    
    Plot three scatter plots in a grid

    Parameters
    ----------
    plot_vals_dict : dictionary with plot values
    sources : list of sources    
    aod_range : list for min and max of plot
    plot_style : string, plot style
    title : boolean, whether to put title above plot    
    logscale : boolean, whether the plot with logscale. The default is True.

    Returns
    -------
    None.

    """
    
    plt.ioff()
    plt.close('all')
    plt.style.use(plot_style)
    
    fig, axs = plt.subplots(2, 2,sharey='row',sharex='col',figsize=(10,9))            
        
    #Upper left plot        
    n_ax = 0  
    
    figstring = plot_vals_dict["figstringprefix"]
    titlestring = plot_vals_dict["titlestring"]
    for i, source in enumerate(sources):
        figstring += plot_vals_dict["sources"][source]["figstring"]        
        if i < len(sources) - 1:
            titlestring += f"{source} vs. "
        else:
            titlestring += f"{source}"
    
    sc1 = axs.flat[n_ax].scatter(plot_vals_dict["sources"][sources[0]]["data"],
                                      plot_vals_dict["sources"][sources[1]]["data"],
                                      c=plot_vals_dict["sources"][sources[0]]["colourdata"],
                                      cmap="jet")
    #axs.flat[n_ax].set_xlabel(plot_vals_dict["sources"][sources[0]]["label"])
    axs.flat[n_ax].set_ylabel(plot_vals_dict["sources"][sources[1]]["label"])    
    axs.flat[n_ax].set_xticklabels('')    
    
    # l, b, w, h = axs.flat[n_ax].get_position().bounds
    # axs.flat[n_ax].set_position([-3.,b,w,h])
        
    #Delete upper right plot
    n_ax += 1
    axs.flat[n_ax].set_visible(False)
    
    #Lower left plot
    n_ax += 1
    
    sc2 = axs.flat[n_ax].scatter(plot_vals_dict["sources"][sources[0]]["data"],
                                      plot_vals_dict["sources"][sources[2]]["data"],
                                      c=plot_vals_dict["sources"][sources[0]]["colourdata"],
                                      cmap="jet")
    axs.flat[n_ax].set_xlabel(plot_vals_dict["sources"][sources[0]]["label"])
    axs.flat[n_ax].set_ylabel(plot_vals_dict["sources"][sources[2]]["label"])
    
    # l, b, w, h = axs.flat[n_ax].get_position().bounds
    # axs.flat[n_ax].set_position([-3,b,w,h])
    
    #Lower right plot
    n_ax += 1
    
    sc3 = axs.flat[n_ax].scatter(plot_vals_dict["sources"][sources[1]]["data"],
                                      plot_vals_dict["sources"][sources[2]]["data"],
                                      c=plot_vals_dict["sources"][sources[0]]["colourdata"],
                                      cmap="jet")
    axs.flat[n_ax].set_xlabel(plot_vals_dict["sources"][sources[1]]["label"])
    #axs.flat[n_ax].set_ylabel(plot_vals_dict["sources"][sources[2]]["label"])    
                
    axs.flat[n_ax].set_yticklabels('')
    
    # cb3 = plt.colorbar(sc3)
    # cb3.set_label(plot_vals_dict["sources"][sources[2]]["colourlabel"])
        
    for ax in axs.flat:
        ax.set_xlim(aod_range)
        ax.set_ylim(aod_range)    
        ax.set(adjustable='box', aspect='equal')
        if logscale:
            ax.set_yscale('log')
            ax.set_xscale('log')
            for axis in [ax.xaxis, ax.yaxis]:
                formatter = FuncFormatter(lambda y, _: '{:.16g}'.format(y))
                axis.set_major_formatter(formatter) 
                #axis.set_major_formatter(ScalarFormatter())      
                
                
        ax.plot([0, 1], [0, 1], ls = '--', transform=ax.transAxes,c='r')
            
    # fig.add_subplot(111, frameon=False)
    # plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    # plt.grid(False)

    cb = plt.colorbar(sc3,cax = fig.add_axes([0.89, 0.11, 0.02, 0.81]),pad=0.5)
    cb.set_label(plot_vals_dict["sources"][sources[0]]["colourlabel"])
    
    fig.subplots_adjust(hspace=0.06,wspace=0.0)
    fig.subplots_adjust(top=0.93)   
    
    if title: fig.suptitle(titlestring)   
    figstring += plot_vals_dict["figstringsuffix"]
    
    plt.savefig(figstring,bbox_inches = 'tight')   
    plt.close(fig)
        
    
def scatter_plot_aod_comparison_grid(name,df_compare,rt_config,pyr_substats,
                             pv_substats,info,styles,pv_path,pyr_path,
                             avg_window,title_flag):
    """
    

    Parameters
    ----------
    name : string, name of station
    df_compare : dataframe with all results 
    rt_config : dictionary with radiative transfer configuration
    pyr_substats : dictionary with pyranometer substations
    pv_substats : dictionary with pv substations
    info : string, description of current campaign    
    styles : dictionary with plot styles
    pv_path : string with path for PV plots
    pyr_path : string with path for pyranometer plots
    avg_window : string with width of window for moving averages
    title_flag : boolean, whether to add title to plot

    Returns
    -------
    None.

    """    
    
    year = info.split('_')[1]
        
    #Generate scatter plots
    aod_range = rt_config["aerosol"]["lut_config"]["range"]    
    
    #Paths for PV plots
    res_dirs = list_dirs(pv_path)
    pv_path = os.path.join(pv_path,'AOD_Plots')
    if 'AOD_Plots' not in res_dirs:
        os.mkdir(pv_path)   
    
    stat_dirs = list_dirs(pv_path)
    pv_path = os.path.join(pv_path,name)
    if name not in stat_dirs:
        os.mkdir(pv_path)
        
    res_dirs = list_dirs(pv_path)
    pv_path = os.path.join(pv_path,'Scatter')        
    if 'Scatter' not in res_dirs:
        os.mkdir(pv_path)
        
    res_dirs = list_dirs(pyr_path)
    pyr_path = os.path.join(pyr_path,'AOD_Plots')
    if 'AOD_Plots' not in res_dirs:
        os.mkdir(pyr_path)   
        
    stat_dirs = list_dirs(pyr_path)
    pyr_path = os.path.join(pyr_path,name)
    if name not in stat_dirs:
        os.mkdir(pyr_path)
        
    res_dirs = list_dirs(pyr_path)
    pyr_path = os.path.join(pyr_path,'Scatter')        
    if 'Scatter' not in res_dirs:
        os.mkdir(pyr_path)            

    #Generate plots by looping through substations        
    for substat in pyr_substats:        
        radtypes = []
        colnames = df_compare.xs(substat,level="substat",axis=1).columns
        if f"AOD_500_poa_inv_{avg_window}_avg" in colnames:
            radtypes.append("poa")
            
            #Get angle parameters 
            if "opt_pars" in pyr_substats[substat]:
                pyrtilt = np.rad2deg(pyr_substats[substat]["opt_pars"][0][1])
                pyrazimuth = np.rad2deg(azi_shift(pyr_substats[substat]["opt_pars"][1][1]))
            else:
                pyrtilt = np.rad2deg(pyr_substats[substat]["ap_pars"][0][1])
                pyrazimuth = np.rad2deg(azi_shift(pyr_substats[substat]["ap_pars"][1][1]))
                
        if f"AOD_500_down_inv_{avg_window}_avg" in colnames:
            radtypes.append("down")
        
        #Plot poa vs. down for pyranometers
        if "poa" in radtypes and "down" in radtypes:
            
            labels = [r"AOD 500nm ($G^\downarrow_\mathrm{{{}}}$)".format(substat),
                      r"AOD 500nm ($G^\angle_\mathrm{{{}}}, \theta={:.1f}^\circ,\phi={:.1f}^\circ$)"
                      .format(substat,pyrtilt,pyrazimuth),
                      #r"clearness index $k_i$ ($G^\downarrow_\mathrm{{{}}}$)".format(substat)]
                      r"SZA ($\circ$)"]
            
            titlestring = f'AOD, {substat} tilted vs. downward irradiance: {name}, {year}'
            figstring = os.path.join(pyr_path,f"AOD_poa_vs_downward_{substat}_{name}_{year}.png")
            
            scatter_plot_aod(df_compare[(f"AOD_500_down_inv_{avg_window}_avg",substat)],
                            df_compare[(f"AOD_500_poa_inv_{avg_window}_avg",substat)],
                            df_compare[("sza","sun")],
                            labels, titlestring, figstring, aod_range, 
                            styles["single_small"],title_flag)

        plot_dict = {}    
        
        #Compare pyranometer to AERONET, PV
        for radtype in radtypes:
            if radtype == "poa":
                radlabel = "\\angle"
                anglelabelsmall = "^{{\\theta={:.1f}^\circ}}_{{\phi={:.1f}^\circ}}".format(pyrtilt,pyrazimuth)
                anglelabellarge = ", \\theta={:.1f}^\circ,\phi={:.1f}^\circ".format(pyrtilt,pyrazimuth)
                titlelabel = "tilted"
            elif radtype == "down":
                radlabel = "\\downarrow"
                titlelabel = "downward"
                anglelabelsmall = ""
                anglelabellarge = ""
        
            #Add relevant info to plot dictionary for grid plots
            plot_dict.update({"sources":{}})
            plot_dict["sources"].update({substat:{}})
            plot_dict["sources"][substat].update({"data":
                          df_compare[(f"AOD_500_{radtype}_inv_{avg_window}_avg",substat)]})           
            plot_dict["sources"][substat].update({"label":
                          r"AOD 500nm (${{G^{}_\mathrm{{{}}}}}\ {}$)".format(radlabel,substat,anglelabelsmall)})
            plot_dict["sources"][substat].update({"colourdata":
                          df_compare[("sza","sun")]})           
            plot_dict["sources"][substat].update({"colourlabel":
                          #r"clearness index $k_i$ ($G^{}_\mathrm{{{}}}$)".format(radlabel,substat)})
                          r"SZA ($\circ$)"})
            plot_dict["sources"][substat].update({"figstring":f"{radtype}_{substat}_"})
                
            plot_dict.update({"titlestring":f"AOD retrieval at {name}, {year}: "})            
            plot_dict.update({"figstringprefix":os.path.join(pyr_path,"AOD_scatter_grid_")})
            plot_dict.update({"figstringsuffix":f"{name}_{year}.png"})
        
            #AERONET
            labels = [r"AOD 500nm ($G^{}_\mathrm{{{}}}{}$)".format(radlabel,substat,anglelabellarge),
                      r"AOD 500nm (AERONET)",
                      #r"clearness index $k_i$ ($G^{}_\mathrm{{{}}}$)".format(radlabel,substat)]
                      r"SZA ($\circ$)"]
            
            titlestring = f'AOD, {substat} {titlelabel} irradiance vs. AERONET: {name}, {year}'
            figstring = os.path.join(pyr_path,f"AOD_{radtype}_{substat}_vs_aeronet_{name}_{year}.png")
            
            scatter_plot_aod(df_compare[(f"AOD_500_{radtype}_inv_{avg_window}_avg",substat)],
                            df_compare[(f"AOD_500_{avg_window}_avg","Aeronet")],
                            df_compare[("sza","sun")],
                            labels, titlestring, figstring, aod_range, 
                            styles["single_small"],title_flag)
            
            plot_dict["sources"].update({"aeronet":{}})
            plot_dict["sources"]["aeronet"].update({"data":df_compare[(f"AOD_500_{avg_window}_avg","Aeronet")]})
            plot_dict["sources"]["aeronet"].update({"label":r"AOD 500nm (AERONET)"})                        
            plot_dict["sources"]["aeronet"].update({"figstring":"aeronet_"})
            
            # #COSMO
            # labels = [r"AOD 500nm ($G^{}_\mathrm{{{}}}{}$)".format(radlabel,substat,anglelabellarge),
            #           r"AOD 600nm (COSMO)",
            #           r"clearness index $k_i$ ($G^{}_\mathrm{{{}}}$)".format(radlabel,substat)]
            
            # titlestring = f'AOD, {substat} {titlelabel} irradiance vs. COSMO: {name}, {year}'
            # figstring = os.path.join(pyr_path,f"AOD_{radtype}_{substat}_vs_cosmo_{name}_{year}.png")
            
            # scatter_plot_aod(df_compare[(f"AOD_500_{radtype}_inv",substat)],
            #                 df_compare[("AOD_w_600","cosmo")],
            #                 df_compare[(f"k_index_{radtype}",substat)],
            #                 labels, titlestring, figstring, aod_range, 
            #                 styles["single_small"],title_flag)
            
            # plot_dict["sources"].update({"cosmo":{}})
            # plot_dict["sources"]["cosmo"].update({"data":df_compare[("AOD_w_600","cosmo")]})
            # plot_dict["sources"]["cosmo"].update({"label":r"AOD 600nm (COSMO)"}) 
            # plot_dict["sources"]["cosmo"].update({"figstring":"cosmo_"})
            
            #Compare to PV
            if pv_substats:                
                for substat_type in pv_substats:
                    if info in pv_substat_pars[substat_type]["source"]:
                        for substat_pv in pv_substats[substat_type]["data"]:   
                            #Get angle parameters 
                            if "opt_pars" in pv_substats[substat_type]["data"][substat_pv]:
                                pvtilt = np.rad2deg(pv_substats[substat_type]["data"][substat_pv]["opt_pars"][0][1])
                                pvazimuth = np.rad2deg(azi_shift(pv_substats[substat_type]["data"]
                                                         [substat_pv]["opt_pars"][1][1]))
                            else:
                                pvtilt = np.rad2deg(pv_substats[substat_type]["data"]
                                            [substat_pv]["ap_pars"][0][1])
                                pvazimuth = np.rad2deg(azi_shift(pv_substats[substat_type]["data"]
                                                         [substat_pv]["ap_pars"][1][1]))
                                
                            pvanglelabelsmall = "^{{\\theta={:.1f}^\circ}}_{{\phi={:.1f}^\circ}}".format(pvtilt,pvazimuth)
                            pvanglelabellarge = ", \\theta={:.1f}^\circ,\phi={:.1f}^\circ".format(pvtilt,pvazimuth)
                            
                            labels = [r"AOD 500nm ($G^{}_\mathrm{{{}}}{}$)".format(radlabel,substat,anglelabellarge),                      
                                  r"AOD 500nm ($G^\angle_\mathrm{{{}}}{}$)"
                                  .format(substat_pv,pvanglelabellarge),
                                  #r"clearness index $k_i$ ($G^{}_\mathrm{{{}}}$)".format(radlabel,substat)]
                                  r"SZA ($\circ$)"]
                        
                            titlestring = f'AOD, {titlelabel} irradiance, {substat} vs. {substat_pv}: {name}, {year}'
                            figstring = os.path.join(pyr_path,f"AOD_{radtype}_{substat}_vs_{substat_pv}_{year}.png")
                        
                            scatter_plot_aod(df_compare[(f"AOD_500_{radtype}_inv_{avg_window}_avg",substat)],
                                        df_compare[(f"AOD_500_poa_inv_{avg_window}_avg",substat_pv)],
                                        df_compare[("sza","sun")],
                                        labels, titlestring, figstring, aod_range, 
                                        styles["single_small"],title_flag)
                            
                            plot_dict["sources"].update({substat_pv:{}})
                            plot_dict["sources"][substat_pv].update({"data":df_compare[(f"AOD_500_poa_inv_{avg_window}_avg",substat_pv)]})
                            plot_dict["sources"][substat_pv].update({"label":
                                      r"AOD 500nm (${{G^\angle_\mathrm{{{}}}}}\ {}$)".format(substat_pv,pvanglelabelsmall)})
                            plot_dict["sources"][substat_pv].update({"colourdata":df_compare[("sza","sun")]})         
                            plot_dict["sources"][substat_pv].update({"colourlabel":
                                      #r"clearness index $k_i$ ($G^\angle_\mathrm{{{}}}$)".format(substat_pv)})
                                      r"SZA ($\circ$)"})
                            plot_dict["sources"][substat_pv].update({"figstring":f"{substat_pv}_"})
                                        
                            #Create grid of scatter plots
                            scatter_plot_grid_3(plot_dict,[substat,'aeronet',substat_pv],
                                                  aod_range,styles["combo_small"],title_flag)
                            
                        # scatter_plot_grid_3(plot_dict,[substat,'cosmo',substat_pv],
                        #                     aod_range,styles["combo_small"],title_flag)
                            
    #Same for PV 
    if pv_substats:   
        #Same for PV            
        print("Plotting PV scatter plots")
        plot_dict = {}    
        plot_dict.update({"sources":{}})
        for substat_type in pv_substats:
            if info in pv_substat_pars[substat_type]["source"]:
                for substat_pv in pv_substats[substat_type]["data"]:                 
                    #Get angle parameters 
                    if "opt_pars" in pv_substats[substat_type]["data"][substat_pv]:
                        pvtilt = np.rad2deg(pv_substats[substat_type]["data"][substat_pv]["opt_pars"][0][1])
                        pvazimuth = np.rad2deg(azi_shift(pv_substats[substat_type]["data"]
                                                 [substat_pv]["opt_pars"][1][1]))
                    else:
                        pvtilt = np.rad2deg(pv_substats[substat_type]["data"]
                                    [substat_pv]["ap_pars"][0][1])
                        pvazimuth = np.rad2deg(azi_shift(pv_substats[substat_type]["data"]
                                                 [substat_pv]["ap_pars"][1][1]))
                        
                    pvanglelabelsmall = "^{{\\theta={:.1f}^\circ}}_{{\phi={:.1f}^\circ}}".format(pvtilt,pvazimuth)
                    pvanglelabellarge = ", \\theta={:.1f}^\circ,\phi={:.1f}^\circ".format(pvtilt,pvazimuth)
                    
                    titlelabel = "tilted"
                    #AERONET
                    labels = [r"AOD 500nm ($G^\angle_\mathrm{{{}}}{}$)".format(substat_pv,pvanglelabellarge),
                              r"AOD 500nm (Aeronet)",
                              r"SZA ($\circ$)"]
                    
                    titlestring = f'AOD, {substat_pv} {titlelabel} irradiance vs. AERONET: {name}, {year}'
                    figstring = os.path.join(pv_path,f"AOD_poa_{substat_pv}_vs_aeronet_{name}_{year}.png")
                    
                    scatter_plot_aod(df_compare[(f"AOD_500_poa_inv_{avg_window}_avg",substat_pv)],
                                    df_compare[(f"AOD_500_{avg_window}_avg","Aeronet")],
                                    df_compare[("sza","sun")],
                                    labels, titlestring, figstring, aod_range, 
                                    styles["single_small"],title_flag)
                    
                # #COSMO
                # labels = [r"AOD 500nm ($G^\angle_\mathrm{{{}}}{}$)".format(substat_pv,pvanglelabellarge),
                #           r"AOD 600nm (COSMO)",
                #           r"clearness index $k_i$ ($G^\angle_\mathrm{{{}}}$)".format(substat_pv)]
                
                # titlestring = f'AOD, {substat_pv} {titlelabel} irradiance vs. COSMO: {name}, {year}'
                # figstring = os.path.join(pv_path,f"AOD_poa_{substat_pv}_vs_cosmo_{name}_{year}.png")
                
                # scatter_plot_aod(df_compare[("AOD_500_inv",substat_pv)],
                #                 df_compare[("AOD_w_600","cosmo")],
                #                 df_compare[("k_index_Etotpoa_pv_inv",substat_pv)],
                #                 labels, titlestring, figstring, aod_range, 
                #                 styles["single_small"],title_flag)
                    
                # scatter_plot_grid_3(plot_dict,[substat_pv,'cosmo','aeronet'],
                #                     aod_range,styles["combo_small"],title_flag)    
                
    
            
    return  

    
def avg_std_cod_retrieval(day,df_day_result,substat,radtype,timeres,
                       str_window_avg):
    """
    

    Parameters
    ----------
    day : string, day under consideration
    df_day_result : dataframe with results for specific day
    substat : string, name of substation
    radtype : string, type of irradiance (poa or down)
    timeres : string, time resolution of data 
    str_window_avg : string, width of window for moving average

    Returns
    -------
    dataframe with combined results

    """
    
    
    #Original time resolution
    data_freq = pd.Timedelta(timeres)
    #Size of moving average window
    window_avg = pd.Timedelta(str_window_avg)    
    
    #1. Calculate moving average of COD retrieval        
    df_aod_avg_std = moving_average_std(df_day_result[f"AOD_500_{radtype}_inv"],data_freq,window_avg)        
    
    #Calculate moving average of cloud fraction
    df_cf_avg_std = moving_average_std(df_day_result[f"cloud_fraction_{radtype}"],data_freq,window_avg)
    
    #Assign to dataframe    
    df_day_result[f"AOD_500_{radtype}_inv_{str_window_avg}_avg"] = df_aod_avg_std["avg_pd"]
    df_day_result[f"AOD_500_{radtype}_inv_{str_window_avg}_std"] = df_aod_avg_std["std_pd"]
    #df_day_result[f"k_index_{radtype}_{str_window_avg}_avg"] = df_ki_avg_std["avg_conv"]
    df_day_result[f"cf_{radtype}_{str_window_avg}_avg"] = df_cf_avg_std["avg_pd"]        
        
    #Reindex the moving average to the nearest hour
    df_aod_reindex_60 = df_aod_avg_std.reindex(pd.date_range(start=df_aod_avg_std.index[0].round(window_avg),
                                                end=df_aod_avg_std.index[-1].round(window_avg),freq=window_avg)
                                                 ,method='nearest',tolerance='5T')  
    df_aod_reindex_60.columns = pd.MultiIndex.from_product([[f"AOD_500_{radtype}_inv_{str_window_avg}_avg",
                             f"AOD_500_{radtype}_inv_{str_window_avg}_std"],[substat]],
                               names=['variable','substat'])                       
        
    #Reindex the moving average cloud fraction to the nearest hour
    cf_avg_reindex_60 = df_cf_avg_std["avg_pd"].reindex(pd.date_range(start=df_cf_avg_std.index[0].round(window_avg),
                                                end=df_cf_avg_std.index[-1].round(window_avg),freq=window_avg)
                                                 ,method='nearest',tolerance='5T')  
    cf_avg_reindex_60.rename((f"cf_{radtype}_{str_window_avg}_avg",substat),inplace=True)            
    
    return pd.concat([df_aod_reindex_60,cf_avg_reindex_60],axis=1) #ki_avg_reindex_60,

 
def aod_analysis_plots(name,pv_station,substat_pars,year,
                 savepath,sza_limit,str_window_avg,flags):
    """
    

    Parameters
    ----------
    name : string, name of PV station
    pv_station : dictionary with information and data of PV station
    substat_pars : dictionary with parameters for substations
    year : string, year under consideration
    savepath : string, path to save plots
    sza_limit : float, SZA limit
    str_window_avg : string, width of averaging window
    flags : dictionary, flags for plotting

    Returns
    -------
    df_combine : dataframe with combined results after moving averages

    """
    
    res_dirs = list_dirs(savepath)
    savepath = os.path.join(savepath,'AOD_Plots')
    if 'AOD_Plots' not in res_dirs:
        os.mkdir(savepath)    
        
    stat_dirs = list_dirs(savepath)
    savepath = os.path.join(savepath,name)
    if name not in stat_dirs:
        os.mkdir(savepath)
    
    res_dirs = list_dirs(savepath)
    savepath_LUT = os.path.join(savepath,'LUT')        
    if 'LUT' not in res_dirs:
        os.mkdir(savepath_LUT)
    savepath_comp = os.path.join(savepath,'Comparison')        
    if 'Comparison' not in res_dirs:
        os.mkdir(savepath_comp)

    dfs_combine = []        
    
    for substat in substat_pars:
        substat_dirs_LUT = list_dirs(savepath_LUT)
        plotpath_LUT = os.path.join(savepath_LUT,substat)
        if substat not in substat_dirs_LUT:
            os.mkdir(plotpath_LUT)
            
        substat_dirs_comp = list_dirs(savepath_comp)
        plotpath_comp = os.path.join(savepath_comp,substat)
        if substat not in substat_dirs_comp:
            os.mkdir(plotpath_comp)
    
        timeres = substat_pars[substat]["t_res_inv"]
        
        #Get radnames
        radname = substat_pars[substat]["name"]                    
        if "pyr" in radname:
            radnames = [radname,radname.replace('poa','down')]
        else:
            radnames = [radname]
        
        df_result = pv_station[f"df_aodfit_pyr_pv_{year}_{timeres}"]    
        
        #If there is no data for a specific radname remove it
        for radname in radnames:
            if (radname,substat) not in df_result.columns:
                radnames.remove(radname)
                print(f"There is no data for {radname}")
        
        radtypes = []        
        for radname in radnames:
            if "poa" in radname:
                radtypes.append("poa")                
            elif "down" in radname:
                radtypes.append("down")                    
        
        dfs_results = []
        
        #df_cosmo = pv_station[f"df_cosmo_{year}"]        
        
        day_list = [group[0] for group in df_result
                               .groupby(df_result.index.date)]
        day_index_list = [pd.to_datetime(group[1].index) for group in df_result
                               .groupby(df_result.index.date)]                        
        
        for radtype in radtypes:
            print("AOD analysis and plots for %s, %s for %s irradiance at %s" % (name,substat,radtype,timeres))            
            if radtype == "poa":                
                radname = substat_pars[substat]["name"]
                if "pv" not in radname:
                    errorname = "error_" + '_'.join([radname.split('_')[0],radname.split('_')[-1]])
                else:
                    errorname = "error_" + radname
                
            elif radtype == "down":
                radname = substat_pars[substat]["name"].replace("poa","down")            
                errorname = "error_" + '_'.join([radname.split('_')[0],radname.split('_')[-1]])                                               
            
            #Plotlabels
            if radtype == "poa":
                radlabel = "\\angle"
                titlelabel = "tilted"
            elif radtype == "down":
                radlabel = "\\downarrow"
                titlelabel = "downward"
            # elif radtype == "":
            #     radlabel = "\\angle"
            #     titlelabel = "tilted"
            
            dfs_radtype_combine = []   
            dfs_radtype_results = []
            # loop over the dates and plot for each day
            for day_no, day_index in enumerate(day_index_list):
                day = day_list[day_no] 
            
                df_day_result = df_result.xs(substat,level='substat',axis=1).loc[day_index]
                df_day_aeronet = df_result.xs("Aeronet",level='substat',axis=1).loc[day_index]
                sza_index_day = df_result.loc[day_index].loc[df_result[("sza","sun")] <= sza_limit].index 
                
                if df_day_result[f"AOD_500_{radtype}_inv"].dropna(how="all").empty:
                    print(f"No AOD data from {substat}, {radtype} on {day}")
                else:                                                    
                    if flags["lut"]:
                        print(f'Plotting AOD LUT for {substat}, {radtype} on {day}')
                        plot_aod_lut(name,substat,df_day_result,day,radtype,
                                  radname,errorname,radlabel,titlelabel,
                                  df_day_aeronet,plotpath_LUT)                    
                    
                    #Calculate moving average and standard deviation for COD retrieval
                    dfs_radtype_combine.append(avg_std_cod_retrieval(day,df_day_result,
                                         substat,radtype,timeres,str_window_avg))                    
                    
                    #Plot comparison plot for each day if required
                    # if flags["compare"]:                        
                    #     plot_cod_comparison(name,substat,df_day_result,day,sza_index_day,
                    #         timeres,str_window_avg,radtype,radname,errorname,radlabel,
                    #         titlelabel,df_seviri,df_apollo,df_cosmo,plotpath_comp)
                        
                    dfs_radtype_results.append(df_day_result)
                        
            if len(dfs_radtype_combine) > 0:
                df_radtype_combine = pd.concat(dfs_radtype_combine,axis=0)        
                dfs_combine.append(df_radtype_combine)
            
                df_radtype_results = pd.concat(dfs_radtype_results,axis=0)
                dfs_results.append(df_radtype_results)
            
        #Combine all results after calculating moving averages
        df_results_combine = pd.concat(dfs_results,axis=1).filter(regex='avg|std', axis=1)        
        df_results_combine.columns = pd.MultiIndex.from_product([df_results_combine.columns.to_list(),[substat]],
                               names=['variable','substat'])   
            
        pv_station[f"df_aodfit_pyr_pv_{year}_{timeres}"] = pd.concat([df_result,
                                  df_results_combine],axis=1)
        pv_station[f"df_aodfit_pyr_pv_{year}_{timeres}"].sort_index(axis=1,level=1,inplace=True)        
        
        #Combine results for statistics
        df_combine = pd.concat(dfs_combine,axis=1,join='inner')
        df_combine = pd.concat([df_combine,pv_station[f"df_compare_{year}"]],axis=1)
        df_combine.columns.names = ['variable','substat']                
            
    return df_combine            
            
# def variance_class(input_series,num_classes):
#     """
#     Split up an input series into variance classes

#     Parameters
#     ----------
#     input_series : series, input data
#     num_classes : integer, number of classes

#     Returns
#     -------
#     var_class_series : series with variance classes
#     class_limits : list with class limits

#     """
    
#     max_std = input_series.max()
#     min_std = input_series.min()
        
#     bin_size = (max_std - min_std)/num_classes
    
#     class_limits = [min_std + i*bin_size for i in range(num_classes)]
        
#     var_class_series = pd.Series(dtype=int,index=input_series.index)    
#     for i in range(num_classes):
#         if i < num_classes - 1: 
#             mask = (input_series >= min_std + i*bin_size) &\
#             (input_series < min_std + (i+1)*bin_size)            
#         else:
#             mask = (input_series >= min_std + i*bin_size) &\
#             (input_series <= min_std + (i+1)*bin_size)            
        
#         var_class_series.loc[mask] = i        
                            
#     return var_class_series, class_limits

# def box_plots_variance(name,dataframe,year,names,str_window_avg,num_classes,
#                        var_class_dict,title,styles,plotpath):
#     """
    

#     Parameters
#     ----------
#     name : string, name of PV station
#     dataframe : dataframe with relevant data
#     year : string, year under consideration
#     names : list of tuples with plot names 
#     str_window_avg : string, width of averaging window
#     num_classes : integer, number of classes
#     var_class_dict : dictionary with variance classes
#     title : string, plot title
#     styles : dictionary with plot styles
#     plotpath : string with plot path

#     Returns
#     -------
#     None.

#     """

#     plt.ioff()
#     plt.style.use(styles["combo_small"])
    
#     res_dirs = list_dirs(plotpath)
#     plotpath = os.path.join(plotpath,'COD_Plots')
#     if 'COD_Plots' not in res_dirs:
#         os.mkdir(plotpath)    
        
#     stat_dirs = list_dirs(plotpath)
#     plotpath = os.path.join(plotpath,name.replace(' ','_'))
#     if name.replace(' ','_') not in stat_dirs:
#         os.mkdir(plotpath)
    
#     res_dirs = list_dirs(plotpath)
#     plotpath = os.path.join(plotpath,'Stats')        
#     if 'Stats' not in res_dirs:
#         os.mkdir(plotpath)
    
#     figstring = f'COD_box_grid_{str_window_avg}_avg_{name.replace(" ","_")}_{year}.png'
    
#     if num_classes == 2:
#         class_labels = ["Low","High"]
#     elif num_classes == 3:
#         class_labels = ["Low","Medium","High"]
#     elif num_classes == 4:
#         class_labels = ["Low","Low-Medium","High-Medium","High"]
    
#     if num_classes <= 3:
#         fig, axs = plt.subplots(num_classes,1,sharey='all',figsize=(10,9))            
#     elif num_classes == 4:
#         fig, axs = plt.subplots(2, 2,sharey='all',figsize=(10,9))            
                                
#     for i in range(num_classes):
#         plot_list = []
#         label_list = []
#         class_list = []
#         for avg, var, substat, label in names: 
#             if np.isnan(dataframe[(var,substat)]).all():
#                 plot_list.append(dataframe.loc[:,
#                        (avg,substat)])
#             else:
#                 plot_list.append(dataframe.loc[dataframe[(var,substat)]==i,
#                        (avg,substat)])
#             label_list.append(label)
#             class_list.append(var_class_dict[label][i:i+2])
        
#         axs.flat[i].boxplot(plot_list)
        
#         axs.flat[i].set_ylabel('COD')
#         axs.flat[i].set_title(f"{class_labels[i]} variance",fontsize=14)
#         axs.flat[i].set_xticklabels([])
#         #axs.flat[i].set_yscale('log')                
#         #axs.flat[i].set_ylim((-1, 10))
#         # axs.flat[i].yaxis.set_ticks([0,5,10])        
#         # axs.flat[i].set_yticklabels([0,5,''])        
        
#         # divider = make_axes_locatable(axs.flat[i])
#         # logAxis = divider.append_axes("top", size=1, pad=0.) #, sharex=axs.flat[i])
#         # logAxis.boxplot(plot_list)
#         # logAxis.set_yscale('log')
#         # logAxis.set_ylim((10, 150));
#         # logAxis.set_xticklabels([])
#         # logAxis.set_title(f"{class_labels[i]} variance",fontsize=14)
#         # if i==1: logAxis.set_ylabel('COD',position=[0,0])
    
#     if len(label_list) > 5:                    
#         axs.flat[i].set_xticklabels(label_list,rotation=45,fontsize=14)
#     else:
#         axs.flat[i].set_xticklabels(label_list)
    
#     if title:
#         fig.suptitle(f'{str_window_avg} average COD comparison at {name}, {year}')
#         fig.subplots_adjust(top=0.93)   
        
#     fig.tight_layout()
#     plt.savefig(os.path.join(plotpath,figstring))   
#     plt.close(fig)    

# def aod_histograms(name,dataframe,year,names,str_window_avg,num_classes,
#                        var_class_dict,title,styles,plotpath):
#     """
    

#     Parameters
#     ----------
#     name : string, name of PV station
#     dataframe : dataframe with relevant data
#     year : string with year under consideration    
#     names : list of tuples with plot names 
#     str_window_avg : string, width of averaging window
#     num_classes : integer, number of classes
#     var_class_dict : dictionary with variance classes
#     title : string, plot title
#     styles : dictionary with plot styles
#     plotpath : string with plot path

#     Returns
#     -------
#     None.

#     """
    
#     plt.ioff()
#     plt.style.use(styles["combo_small"])
    
#     res_dirs = list_dirs(plotpath)
#     plotpath = os.path.join(plotpath,'COD_Plots')
#     if 'COD_Plots' not in res_dirs:
#         os.mkdir(plotpath)    
        
#     stat_dirs = list_dirs(plotpath)
#     plotpath = os.path.join(plotpath,name.replace(' ','_'))
#     if name.replace(' ','_') not in stat_dirs:
#         os.mkdir(plotpath)
    
#     res_dirs = list_dirs(plotpath)
#     plotpath = os.path.join(plotpath,'Stats')        
#     if 'Stats' not in res_dirs:
#         os.mkdir(plotpath)
    
#     figstring = f'COD_histogram_grid_{str_window_avg}_avg_{name.replace(" ","_")}_{year}.png'
    
#     if num_classes == 2:
#         class_labels = ["Low","High"]
#     elif num_classes == 3:
#         class_labels = ["Low","Medium","High"]
#     elif num_classes == 4:
#         class_labels = ["Low","Low-Medium","High-Medium","High"]
        
#     if len(names) <= 3:
#         fig, axs = plt.subplots(len(names),1,sharey='all',figsize=(10,9))            
#     elif len(names) == 4:
#         fig, axs = plt.subplots(2, 2,sharey='all',sharex='all',figsize=(10,9))     
#     elif len(names) <= 6:
#         fig, axs = plt.subplots(3, 2,sharey='all',sharex='all',figsize=(10,9))     
#     elif len(names) <= 8:
#         fig, axs = plt.subplots(4, 2,sharey='all',sharex='all',figsize=(10,9))     
#     elif len(names) == 9:
#         fig, axs = plt.subplots(3, 3,sharey='all',sharex='all',figsize=(10,9))     
#     elif len(names) <= 12:
#         fig, axs = plt.subplots(4, 3,sharey='all',sharex='all',figsize=(10,9))
#     elif len(names) <= 16:
#         fig, axs = plt.subplots(4, 4,sharey='all',sharex='all',figsize=(10,9))    
#     elif len(names) <= 20:
#         fig, axs = plt.subplots(5, 4,sharey='all',sharex='all',figsize=(10,9))
#     else:
#         print('Too many plots to fit in')
        
#     bins = np.logspace(np.log10(0.5),np.log10(150),20)
    
#     for i, (avg, var, substat, label) in enumerate(names):                     
#         if substat == "cosmo":
#             axs.flat[i].hist(dataframe[(avg,substat)],bins=bins,
#                              weights=dataframe[("cf_tot_avg",substat)])
#         else:
#             axs.flat[i].hist(dataframe[(avg,substat)],bins=bins)
#         axs.flat[i].set_xscale('log')
#         axs.flat[i].set_title(label)
                
#     fig.add_subplot(111, frameon=False)
#     plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
#     plt.grid(False)
#     plt.xlabel("COD")
#     plt.ylabel("Frequency")    
        
#     if title:
#         fig.suptitle(f'{str_window_avg} average COD comparison at {name}, {year}')
#         fig.subplots_adjust(top=0.93)   
        
#     fig.tight_layout()
#     plt.savefig(os.path.join(plotpath,figstring),bbox_inches = 'tight')   
#     plt.close(fig)    
        

# def aod_stats_plots(name,dataframe,pyr_substats,pv_substats,year,
#                         str_window_avg,num_classes,styles,flags,plotpath):
#     """
    

#     Parameters
#     ----------
        
#     name : string, name of PV station
#     dataframe : dataframe with relevant data
#     pyr_substats : dictionary with pyranometer substations
#     pv_substats : dictionary with pv substations
#     year : string with year under consideration    
#     str_window_avg : string, width of averaging window
#     num_classes : integer, number of classes
#     styles : dictionary with plot styles
#     flags : dictionary with booleans for plots
#     plotpath : string with plot path

#     Returns
#     -------
#     dataframe with statistics
#     dictionary with variance classes

#     """  
#     variance_class_dict = {}
    
#     df_stats_index = dataframe.drop("seviri",level=1,axis=1).dropna(how='any',axis=0).index

#     #Calculate variance classes            
#     var_class, limits = variance_class(dataframe.loc[df_stats_index,(f"cot_AP_{str_window_avg}_std","apollo")],
#                        num_classes)
#     dataframe.loc[df_stats_index,(f"cot_AP_{str_window_avg}_varclass","apollo")] = var_class
#     variance_class_dict.update({"SEVIRI-APNG_1.1":limits})
        
        
#     # dataframe.loc[df_stats_index,(f"COD_500_{str_window_avg}_varclass","seviri")] =\
#     #     variance_class(dataframe.loc[df_stats_index,(f"COD_500_{str_window_avg}_std","seviri")],
#     #                    num_classes)     
    
#     var_class, limits = variance_class(dataframe.loc[df_stats_index,("COD_tot_600_iqr","cosmo")],num_classes)
#     dataframe.loc[df_stats_index,("COD_tot_600_varclass","cosmo")] = var_class     
#     variance_class_dict.update({"COSMO":limits})
        
#     plot_names = [(f"cot_AP_{str_window_avg}_avg",f"cot_AP_{str_window_avg}_varclass",
#                    "apollo","SEVIRI-APNG_1.1"),
#                   # (f"COD_500_{str_window_avg}_avg",f"COD_500_{str_window_avg}_varclass",
#                   #  "seviri","SEVIRI-HRV")]
#                   ("COD_tot_600_avg","COD_tot_600_varclass",
#                     "cosmo","COSMO")]
    
#     for substat in pyr_substats:
#         #Get radnames
#         radname = pyr_substat_pars[substat]["name"]                    
#         if "pyr" in radname:
#             radnames = [radname,radname.replace('poa','down')]
#         else:
#             radnames = [radname]                    
        
#         radtypes = []        
#         for radname in radnames:
#             if "poa" in radname:
#                 radtypes.append("poa")                
#             elif "down" in radname:
#                 radtypes.append("down")       

#         #Calculate variance classes for pyranometers                
#         for radtype in radtypes:
#             if (f"COD_550_{radtype}_inv_{str_window_avg}_avg",substat) in dataframe.columns:
#                 var_class, limits = variance_class(dataframe.loc[df_stats_index,(f"COD_550_{radtype}_inv_{str_window_avg}_std",
#                                  substat)],num_classes)
#                 dataframe.loc[df_stats_index,(f"COD_550_{radtype}_inv_{str_window_avg}_varclass",substat)] =\
#                     var_class
#                 variance_class_dict.update({f"{substat}, {radtype}":limits})
                    
#                 plot_names.append((f"COD_550_{radtype}_inv_{str_window_avg}_avg",
#                                    f"COD_550_{radtype}_inv_{str_window_avg}_varclass",
#                                    substat,f"{substat}, {radtype}"))
                
#     for substat_type in pv_substats:
#         for substat in pv_substats[substat_type]["data"]:
#             if year in pv_substats[substat_type]["source"]:  
#                 #Get radnames
#                 radtype = "poa"
                
#                 #Calculate variance classes for PV stations
#                 var_class, limits = variance_class(dataframe.loc[df_stats_index,(f"COD_550_{radtype}_inv_{str_window_avg}_std",
#                                  substat)],num_classes)
#                 dataframe.loc[df_stats_index,(f"COD_550_{radtype}_inv_{str_window_avg}_varclass",substat)] =\
#                     var_class
#                 variance_class_dict.update({f"{substat}":limits})                
                    
#                 plot_names.append((f"COD_550_{radtype}_inv_{str_window_avg}_avg",
#                                    f"COD_550_{radtype}_inv_{str_window_avg}_varclass",
#                                    substat,substat))            
                              
#     dataframe.sort_index(axis=1,level=1,inplace=True)        
    
#     if flags["stats"]:
#         print(f'Plotting box-whisker plots for {name}, {year}')
#         df_plot = dataframe.loc[df_stats_index]
#         box_plots_variance(name,df_plot,year.split('_')[1],plot_names,str_window_avg,num_classes,
#                            variance_class_dict,flags["titles"],styles,plotpath)
        
#         print(f'Plotting histograms for {name}, {year}')
#         aod_histograms(name,df_plot,year.split('_')[1],plot_names,str_window_avg,num_classes,
#                            variance_class_dict,flags["titles"],styles,plotpath)
        

#     return dataframe.drop("seviri",level=1,axis=1).loc[df_stats_index], variance_class_dict    

# def combined_stats_plots(dataframe,pyr_substats,pv_substats,year,
#                         str_window_avg,num_classes,
#                         styles,flags,plotpath):
#     """
    

#     Parameters
#     ----------   
            
#     dataframe : dataframe with relevant data
#     pyr_substats : dictionary with pyranometer substations
#     pv_substats : dictionary with pv substations
#     year : string with year under consideration    
#     str_window_avg : string, width of averaging window
#     num_classes : integer, number of classes
#     styles : dictionary with plot styles
#     flags : dictionary with booleans for plots
#     plotpath : string with plot path

#     Returns
#     -------
#     dataframe with mean values
#     dictionary with variance classes

#     """
    
#     variance_class_dict = {}
    
#     cod_names = [('COD_tot_600_avg',"cosmo"),('COD_tot_600_iqr',"cosmo"),('cf_tot_avg',"cosmo"),
#                  (f'cot_AP_{str_window_avg}_avg',"apollo"),
#                  (f'cot_AP_{str_window_avg}_std',"apollo")]
    
#     #Combine Pyranometers and PV stations
#     for col in dataframe.columns:
#         if "Pyr" in col[1]:
#             dataframe.rename(columns={col[1]:"Pyr"},level='substat',inplace=True)
#             cod_names.append((col[0],"Pyr"))
#         if "egrid" in col[1]:
#             dataframe.rename(columns={col[1]:"PV_1min"},level='substat',inplace=True)
#             cod_names.append((col[0],"PV_1min"))
#         if "auew" in col[1]:
#             dataframe.rename(columns={col[1]:"PV_15min"},level='substat',inplace=True)
#             cod_names.append((col[0],"PV_15min"))
       
#     new_names = []
#     [new_names.append(x) for x in cod_names if x not in new_names]
#     cod_names = [x for x in new_names if "varclass" not in x[0]]# and "cf" not in x[0]]

#     #Calculate means        
#     df_mean = pd.concat([dataframe.loc[:,pd.IndexSlice[cod_name[0],cod_name[1],:]].mean(axis=1)
#                 for cod_name in cod_names],axis=1,keys=cod_names,names=['variable','substat'])        

#     df_mean.dropna(how='any',axis=0,inplace=True)
    
#     #Calculate variance classes            
#     var_class, limits = variance_class(df_mean[(f"cot_AP_{str_window_avg}_std","apollo")],
#                        num_classes)
#     df_mean[(f"cot_AP_{str_window_avg}_varclass","apollo")] = var_class
#     variance_class_dict.update({"SEVIRI-APNG_1.1":limits})
        
        
#     # dataframe.loc[df_stats_index,(f"COD_500_{str_window_avg}_varclass","seviri")] =\
#     #     variance_class(dataframe.loc[df_stats_index,(f"COD_500_{str_window_avg}_std","seviri")],
#     #                    num_classes)     
    
#     var_class, limits = variance_class(df_mean[("COD_tot_600_iqr","cosmo")],num_classes)
#     df_mean[("COD_tot_600_varclass","cosmo")] = var_class     
#     variance_class_dict.update({"COSMO":limits})
        
#     plot_names = [(f"cot_AP_{str_window_avg}_avg",f"cot_AP_{str_window_avg}_varclass",
#                    "apollo","SEVIRI-APNG_1.1"),
#                   # (f"COD_500_{str_window_avg}_avg",f"COD_500_{str_window_avg}_varclass",
#                   #  "seviri","SEVIRI-HRV")]
#                   ("COD_tot_600_avg","COD_tot_600_varclass",
#                     "cosmo","COSMO")]
        
#     radtypes = ["poa","down"]        
    
#     #Calculate variance classes for pyranometers                
#     for radtype in radtypes:
#         if f"COD_550_{radtype}_inv_{str_window_avg}_avg" in df_mean.columns.levels[0]:
#             var_class, limits = variance_class(df_mean[(f"COD_550_{radtype}_inv_{str_window_avg}_std","Pyr")],
#                                                num_classes)
#             df_mean[(f"COD_550_{radtype}_inv_{str_window_avg}_varclass","Pyr")] =\
#                 var_class
#             variance_class_dict.update({f"Pyr, {radtype}":limits})
                
#             plot_names.append((f"COD_550_{radtype}_inv_{str_window_avg}_avg",
#                                f"COD_550_{radtype}_inv_{str_window_avg}_varclass",
#                                "Pyr",f"Pyr, {radtype}"))
    
#     #Calculate variance classes for PV
#     for pv_type in ["PV_1min","PV_15min"]:           
#         if pv_type in df_mean.columns.levels[1]:
#             var_class, limits = variance_class(df_mean[(f"COD_550_poa_inv_{str_window_avg}_std",pv_type)],
#                                                num_classes)
#             df_mean[(f"COD_550_poa_inv_{str_window_avg}_varclass",pv_type)] =\
#                 var_class
#             variance_class_dict.update({pv_type:limits})
                
#             plot_names.append((f"COD_550_poa_inv_{str_window_avg}_avg",
#                                f"COD_550_poa_inv_{str_window_avg}_varclass",
#                                pv_type,pv_type))
    
#     df_mean.sort_index(axis=1,level=1,inplace=True)        
    
    
#     if flags["combo_stats"]:
#         print(f'Plotting box-whisker plots for all stations, {year}')        
#         box_plots_variance("all stations",df_mean,year.split('_')[1],plot_names,str_window_avg,num_classes,
#                            variance_class_dict,flags["titles"],styles,plotpath)
        
#         print(f'Plotting histograms for all stations, {year}')
#         aod_histograms("all stations",df_mean,year.split('_')[1],plot_names,str_window_avg,num_classes,
#                            variance_class_dict,flags["titles"],styles,plotpath)
    
#     return df_mean, variance_class_dict

#%%Main Program
#######################################################################
###                         MAIN PROGRAM                            ###
#######################################################################
#This program plots results from the COD retrievals calculated in pvpyr2cod_interopolate_fit
#and compares them to COD retrievals from SEVIRI, both hi-res and APNG, plus from COSMO
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

#Load PV2rad configuration
pvrad_config = load_yaml_configfile(config["pvrad_configfile"])

#Load radiative transfer configuration
rt_config = load_yaml_configfile(config["rt_configfile"])

homepath = os.path.expanduser('~') # #"/media/luke" #
plot_styles = config["plot_styles"]
plot_flags = config["plot_flags"]

if args.campaign:
    campaigns = args.campaign
    if type(campaigns) != list:
        campaigns = [campaigns]
else:
    campaigns = config["description"] 


sza_limit = rt_config["sza_max"]["cod_cutoff"]
window_avgs = config["window_size_moving_average"]
num_var_classes = 3
#%%Load AOD retrievals
for campaign in [campaigns[1]]: # ["messkampagne_2018_clear_sky"]    
    year = "mk_" + campaign.split('_')[1]    
    #Load pyranometer configuration
    
    pyr_config = load_yaml_configfile(config["pyrcalod_configfile"][year])
    
    if args.station:
        stations = args.station
        if stations[0] == 'all':
            stations = 'all'
    else:
        #Stations for which to perform inversion
        stations = ["PV_12"] #pyr_config["stations"]

    print(f'AOD comparison and analysis for {campaign} at {stations} stations')    
    #Load inversion results
    print('Loading PYR2AOD and PV2AOD results')
    pvsys, pyr_results_folder, pv_results_folder = load_pvpyr2aod_results(rt_config,pyr_config,pvcal_config,pvrad_config,
                                   campaign,stations,homepath)

    # if pyr_config["pmax_doas_station"]:
#     pvsys = load_pmaxdoas_results(pyr_config["pmax_doas_station"], timeres_sim, homepath)
    dfs_stats_all = []
    var_class_dict = {}
    
    for key in pvsys:        
        for str_window_avg in window_avgs:
            print(f'\nAOD analysis for {key}, {year}')
                            
            pvsys[key] = combine_aeronet(year,pvsys[key], str_window_avg)
    
            #Go through Pyranometers to plot
            if key in pyr_config["pv_stations"]:
                #Get substation parameters
                pyr_substat_pars = pvsys[key]["substations_pyr"]                        
                #Perform analysis and plot
                pvsys[key][f"df_compare_{year.split('_')[1]}"] = aod_analysis_plots(key,
                        pvsys[key],pyr_substat_pars,year.split('_')[1],pyr_results_folder,
                        sza_limit,str_window_avg,plot_flags)                                                
            else:
                pyr_substat_pars = {}
                    
            #Go through PVs to plot
            if key in pvrad_config["pv_stations"]:
                pv_substat_pars = pvsys[key]["substations_pv"]
                for substat_type in pv_substat_pars:
                    pv_substats = pv_substat_pars[substat_type]["data"]
                    if year in pv_substat_pars[substat_type]["source"]:                    
                        pvsys[key][f"df_compare_{year.split('_')[1]}"] = aod_analysis_plots(key,
                            pvsys[key],pv_substats,year.split('_')[1],pv_results_folder,
                            sza_limit,str_window_avg,plot_flags)
            else:
                pv_substat_pars = {}            
            
            #Scatter plots            
            if plot_flags["scatter"]:
                print(f"Creating scatter plots with data averaged over {str_window_avg}")
                scatter_plot_aod_comparison_grid(key,pvsys[key][f"df_compare_{year.split('_')[1]}"], 
                             rt_config,pyr_substat_pars,pv_substat_pars,year,plot_styles,
                             pv_results_folder, pyr_results_folder, str_window_avg,
                             plot_flags)
                       
        
    #     #Calculate variance classes
    #     if key in pvrad_config["pv_stations"]:
    #         pv_substats = pvsys[key]["substations_pv"]
    #     else:
    #         pv_substats = {}
        
    #     #Stats: calculation and plots per station
    #     pvsys[key][f"df_stats_{year.split('_')[1]}"], pvsys[key]["var_class"] = cod_stats_plots(key,
    #             pvsys[key][f"df_compare_{year.split('_')[1]}"],
    #             pyr_substat_pars,pv_substats,
    #             year,str_window_avg,num_var_classes,plot_styles,
    #             plot_flags,pyr_results_folder)
        
    #     #Join all dataframes into one for all stations
    #     df_stats = pvsys[key][f"df_stats_{year.split('_')[1]}"]
    #     idx = df_stats.columns.to_frame()
    #     idx.insert(2, 'station', key)
    #     df_stats.columns = pd.MultiIndex.from_frame(idx) 
        
    #     dfs_stats_all.append(df_stats)
    #     var_class_dict.update({key:pvsys[key]["var_class"]})
        
    # df_stats_all = pd.concat(dfs_stats_all,axis=1)
    # df_stats_all.sort_index(axis=1,level=[1,2],inplace=True)        
        
    # #Stats for all stations
    # df_mean_stats, var_classes_mean = combined_stats_plots(df_stats_all,
    #            pyr_substat_pars,pv_substats,year,str_window_avg,
    #          num_var_classes,plot_styles,plot_flags,pyr_results_folder)
                
            