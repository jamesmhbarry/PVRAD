#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 12:59:05 2019

@author: james
"""

import os
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

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


def plot_pv_data (pv_systems,home,paths,info,styles,description):
    """
    Plot PV data for each station

    Parameters
    ----------
    pv_systems : dictionary
        dictionary with information and data from each system
    home : string
        home path
    paths : dictionary
        dictionary with paths for loading data and saving plots
    info : dataframe
        dataframe with station info
    styles : dictionary
        dictionary with plotting styles
    description : string
        description of current data

    Returns
    -------
    None.

    """
    
    plt.ioff()
    plt.style.use(styles['single_small'])
    plt.close('all')

    main_dirs_exist = list_dirs(os.path.join(home,paths['plots']['main']))
    dir_pv = os.path.join(home,paths['plots']['main'],paths['plots']['pv'])
    
    if paths['plots']['pv'] not in main_dirs_exist:
        os.mkdir(dir_pv)
    
    dirs_exist = list_dirs(dir_pv)
    if 'Total' not in dirs_exist:
        os.mkdir(os.path.join(dir_pv,'Total'))
            
    if 'Daily' not in dirs_exist:
        os.mkdir(os.path.join(dir_pv,'Daily'))

    stat_dirs_exist = list_dirs(os.path.join(dir_pv,'Daily'))
    
    print("Plotting PV power")
    for station in info.index:
            
        if station not in stat_dirs_exist:
            os.mkdir(os.path.join(dir_pv,'Daily',station))
        
        #AUEW 15 minute measurement
        if "auew" in paths and station in paths['auew']['stations']:
            print(('Plotting 15 min power for %s' % station))
            fig, ax = plt.subplots(figsize=(9,8))
            dfs = [(substat,pd.concat(pv_systems[station]["raw_data"]['pv'][substat][1],axis=0)) 
                for substat in pv_systems[station]["raw_data"]['pv'] if 'auew' in substat]
            [df[1].plot(ax=ax) for df in dfs]
            plt.legend([df[0] for df in dfs])
            
            plt.ylabel('Power (kW)')#, color='b')
            plt.xlabel('Time (UTC)')
            plt.title('Measured PV power (AUEW 15min) at %s' % station)
            plt.savefig(os.path.join(dir_pv,'Total','p_ac_' + station + '_15min_' + description + '.png'))        
            plt.close(fig)   
            
            combined_days = pd.DatetimeIndex
            for substat in pv_systems[station]["raw_data"]['pv']:
                if 'auew' in substat:
                    if not combined_days.empty:
                        combined_days = combined_days.union(pv_systems[station]["raw_data"]['pv'][substat][0])                    
                    else:
                        combined_days = pv_systems[station]["raw_data"]['pv'][substat][0]
            
            for ix, iday in enumerate(combined_days.strftime("%Y-%m-%d")):
                print(('Generating PV power plot for %s on %s' % (station,iday)))
                fig, ax = plt.subplots(figsize=(9,8))
                ax.set_prop_cycle(color=['blue', 'k','darkred'])
                leg_auew = []
                for substat in pv_systems[station]["raw_data"]['pv']:
                    if iday in pv_systems[station]["raw_data"]['pv'][substat][0] and 'auew' in substat:
                        new_ix = list(pv_systems[station]["raw_data"]['pv'][substat][0].strftime("%Y-%m-%d")).index(iday)                
                        df = pv_systems[station]["raw_data"]['pv'][substat][1][new_ix]
                        ax.plot(df.index,df)
                        leg_auew.append('AUEW ' + substat[-1])
                
                ax.legend(leg_auew)
                ax.set_ylabel('Power (kW)')#, color='b')
                ax.set_xlabel('Time (UTC)')
                plt.title('Measured PV power (AUEW 15min) at %s on %s' % (station,iday))
                
                datemin = np.datetime64(iday + ' 03:30:00')
                datemax = np.datetime64(iday + ' 19:00:00')
                ax.set_xlim([datemin, datemax])
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                fig.autofmt_xdate(rotation=0,ha='center')   
                
                plt.savefig(os.path.join(dir_pv,'Daily',station,'p_ac_' + station + '_' + 
                                         str(iday) + '_15min.png'))         
                plt.close(fig)
        
        #eGrid 1s measurement 
        if "egrid" in paths and station in paths['egrid']['stations']:
            print(('Plotting 1 sec power for %s' % station))
            fig, ax = plt.subplots(figsize=(9,8))
            dfs = [(substat,pd.concat(pv_systems[station]["raw_data"]['pv'][substat][1],axis=0)) 
                for substat in pv_systems[station]["raw_data"]['pv'] if 'egrid' in substat]
            [df[1].plot(ax=ax) for df in dfs]
            plt.legend([df[0] for df in dfs])
                        
            plt.ylabel('Power (kW)')#, color='b')
            plt.xlabel('Time (UTC)')
            plt.title('Measured PV power (eGrid 1Hz) at %s' % station)
            plt.savefig(os.path.join(dir_pv,'Total','p_ac_' + station + '_1s_' + description + '.png'))        
            plt.close(fig)
            
            combined_days = pd.DatetimeIndex
            for substat in pv_systems[station]["raw_data"]['pv']:
                if 'egrid' in substat:
                    if not combined_days.empty:
                        combined_days = combined_days.union(pv_systems[station]["raw_data"]['pv'][substat][0])                    
                    else:
                        combined_days = pv_systems[station]["raw_data"]['pv'][substat][0]
                        
            for ix, iday in enumerate(combined_days.strftime("%Y-%m-%d")):
                print(('Generating PV power plot for %s on %s' % (station,iday)))
                fig, ax = plt.subplots(figsize=(9,8))
                ax.set_prop_cycle(color=['blue', 'k','darkred'])
                leg_egrid = []
                for substat in pv_systems[station]["raw_data"]['pv']:
                    if iday in pv_systems[station]["raw_data"]['pv'][substat][0] and 'egrid' in substat:
                        new_ix = list(pv_systems[station]["raw_data"]['pv'][substat][0].strftime("%Y-%m-%d")).index(iday)                
                        df = pv_systems[station]["raw_data"]['pv'][substat][1][new_ix]
                        ax.plot(df.index,df)
                        if substat[-1] != 'd':
                            leg_egrid.append('Inverter ' + substat[-1])
            
                ax.legend(leg_egrid)
                ax.set_ylabel('Power (kW)')#, color='b')
                ax.set_xlabel('Time (UTC)')
                plt.title('Measured PV power (eGrid 1Hz) at %s on %s' % (station,iday))
                
                datemin = np.datetime64(iday + ' 03:30:00')
                datemax = np.datetime64(iday + ' 19:00:00')
                ax.set_xlim([datemin, datemax])
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                fig.autofmt_xdate(rotation=0,ha='center')   
                
                plt.savefig(os.path.join(dir_pv,'Daily',station,'p_ac_' + station + '_' + 
                                         str(iday) + '_1s.png'))        
                plt.close(fig)
                
        #Solarwatt
        if "solarwatt" in paths and station in paths['solarwatt']['stations']:
            print(('Plotting 1 sec power for %s' % station))
            fig, ax = plt.subplots(figsize=(9,8))
            dfs = [(substat,pd.concat(pv_systems[station]["raw_data"]['pv'][substat][1],axis=0)) 
                for substat in pv_systems[station]["raw_data"]['pv'] if 'myreserve' in substat]
            [ax.plot(df[1].index,df[1]["P_kW"]) for df in dfs]
            plt.legend([df[0] for df in dfs])
                        
            plt.ylabel('Power (kW)')#, color='b')
            plt.xlabel('Time (UTC)')
            plt.title('Measured PV power (Solarwatt MyReserve) at %s' % station)
            plt.savefig(os.path.join(dir_pv,'Total','p_ac_' + station + '_1s_' + description + '.png'))        
            plt.close(fig)
            
            combined_days = pd.DatetimeIndex
            for substat in pv_systems[station]["raw_data"]['pv']:
                if 'myreserve' in substat:
                    if not combined_days.empty:
                        combined_days = combined_days.union(pv_systems[station]["raw_data"]['pv'][substat][0])                    
                    else:
                        combined_days = pv_systems[station]["raw_data"]['pv'][substat][0]
                        
            for ix, iday in enumerate(combined_days.strftime("%Y-%m-%d")):
                print(('Generating PV power plot for %s on %s' % (station,iday)))
                fig, ax = plt.subplots(figsize=(9,8))
                ax.set_prop_cycle(color=['blue', 'k','darkred'])
                leg_solarwatt = []
                for substat in pv_systems[station]["raw_data"]['pv']:
                    if iday in pv_systems[station]["raw_data"]['pv'][substat][0] and 'myreserve' in substat:
                        new_ix = list(pv_systems[station]["raw_data"]['pv'][substat][0].strftime("%Y-%m-%d")).index(iday)                
                        df = pv_systems[station]["raw_data"]['pv'][substat][1][new_ix]
                        ax.plot(df.index,df["P_kW"])   
                        leg_solarwatt.append(substat)
            
                ax.legend(leg_solarwatt)
                ax.set_ylabel('Power (kW)')#, color='b')
                ax.set_xlabel('Time (UTC)')
                plt.title('Measured PV power (Solarwatt MyReserve) at %s on %s' % (station,iday))
                
                datemin = np.datetime64(iday + ' 03:30:00')
                datemax = np.datetime64(iday + ' 19:00:00')
                ax.set_xlim([datemin, datemax])
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                fig.autofmt_xdate(rotation=0,ha='center')   
                
                plt.savefig(os.path.join(dir_pv,'Daily',station,'p_ac_' + station + '_' + 
                                         str(iday) + '_5s.png'))        
                plt.close(fig)
        
        #Inverter 5min measurement 
        if "inverter" in paths and station in paths['inverter']['stations']:
            print(('Plotting 5 min power for %s' % station))
            for substat in pv_systems[station]["raw_data"]['pv']: 
                if 'WR' in substat:
                    fig, ax = plt.subplots(figsize=(9,8))
                    df = pd.concat(pv_systems[station]["raw_data"]['pv'][substat][1],axis=0)
                    df.loc[:,pd.IndexSlice[['Pac_1','Pac_2','Pac_3'],:]].plot(ax=ax)
                    #plt.legend([df[0] for df in dfs])
                            
                    plt.ylabel('Power (W)')#, color='b')
                    plt.xlabel('Time (UTC)')
                    plt.title('Measured PV power (inverter) at %s' % station)
                    plt.savefig(os.path.join(dir_pv,'Total','p_ac_' + station + '_' + substat 
                                             + '_5min_' + description + '.png'))        
                    plt.close(fig)
            
            combined_days = pd.DatetimeIndex
            for substat in pv_systems[station]["raw_data"]['pv']:
                if 'WR' in substat:
                    if not combined_days.empty:
                        combined_days = combined_days.union(pv_systems[station]["raw_data"]['pv'][substat][0])                    
                    else:
                        combined_days = pv_systems[station]["raw_data"]['pv'][substat][0]
                        
            for ix, iday in enumerate(combined_days.strftime("%Y-%m-%d")):
                print(('Generating PV power plots for %s on %s' % (station,iday)))
                for substat in pv_systems[station]["raw_data"]['pv']:
                    if 'WR' in substat:
                        fig, ax = plt.subplots(figsize=(9,8))
                        ax.set_prop_cycle(color=['blue', 'k','darkred'])
                        leg_inv = []
                        if iday in pv_systems[station]["raw_data"]['pv'][substat][0]:
                            new_ix = list(pv_systems[station]["raw_data"]['pv'][substat][0].strftime("%Y-%m-%d")).index(iday)                
                            df = pv_systems[station]["raw_data"]['pv'][substat][1][new_ix].loc[:,pd.IndexSlice[['Pac_1','Pac_2','Pac_3'],:]]
                            ax.plot(df.index,df)
                            leg_inv = ['Pac_1','Pac_2','Pac_3'] #.append('Inverter ' + substat[-1])
                
                        ax.legend(leg_inv)
                        ax.set_ylabel('Power (W)')#, color='b')
                        ax.set_xlabel('Time (UTC)')
                        plt.title('Measured PV power (inverter %s) at %s on %s' % (substat,station,iday))
                        
                        datemin = np.datetime64(iday + ' 03:30:00')
                        datemax = np.datetime64(iday + ' 19:00:00')
                        ax.set_xlim([datemin, datemax])
                        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
                        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                        fig.autofmt_xdate(rotation=0,ha='center')   
                        
                        plt.savefig(os.path.join(dir_pv,'Daily',station,'p_ac_' + station 
                                   + '_' + substat + '_' + str(iday) + '_5min.png'))        
                        plt.close(fig)
                    
            
def plot_rad_data (pv_systems,home,paths,info,styles,description):
    """
    Plot irradiance data for each station

    Parameters
    ----------
    pv_systems : dictionary
        dictionary with information and data from each system
    home : string
        home path
    paths : dictionary
        dictionary with paths for loading data and saving plots
    info : dataframe
        dataframe with station info
    styles : dictionary
        dictionary with plotting styles
    description : string
        description of current data

    Returns
    -------
    None.

    """
    
    plt.ioff()
    plt.style.use(styles['single_small'])
    plt.close('all')
    
    main_dirs_exist = list_dirs(os.path.join(home,paths['plots']['main']))
    dir_rad = os.path.join(home,paths['plots']['main'],paths['plots']['rad'])
    
    if paths['plots']['rad'] not in main_dirs_exist:
        os.mkdir(os.path.join(dir_rad))
    
    dirs_exist = list_dirs(os.path.join(dir_rad))    
    if 'Total' not in dirs_exist:
        os.mkdir(os.path.join(dir_rad,'Total'))
            
    if 'Daily' not in dirs_exist:
        os.mkdir(os.path.join(dir_rad,'Daily'))

    stat_dirs_exist = list_dirs(os.path.join(dir_rad,'Daily'))    

    for station in info.index:
    
        if station not in stat_dirs_exist:
            os.mkdir(os.path.join(dir_rad,'Daily',station))

    #get maximum values
    if pv_systems[station]["raw_data"]['irrad']:
        dfs = [pd.concat(pv_systems[station]["raw_data"]['irrad'][substat][1],axis=0) for substat in pv_systems[station]["raw_data"]['irrad'] 
        if 'Pyr' in substat]
        if len(dfs) > 0:
            max_df = np.max(np.max(pd.concat(dfs,axis=0)))
            max_pyr = int(np.ceil(max_df/100)*100)
        
        dfs = [pd.concat(pv_systems[station]["raw_data"]['irrad'][substat][1],axis=0) for substat in pv_systems[station]["raw_data"]['irrad'] 
        if 'suntracker' in substat]
        if len(dfs) > 0:
            max_df = np.max(np.max(pd.concat(dfs,axis=0)))
            max_suntrack = int(np.ceil(max_df/100)*100)
            
        dfs = [pd.concat(pv_systems[station]["raw_data"]['irrad'][substat][1],axis=0) for substat in pv_systems[station]["raw_data"]['irrad'] 
        if 'mordor' in substat]
        if len(dfs) > 0:
            max_df = np.max(np.max(pd.concat(dfs,axis=0)))
            max_mordor = int(np.ceil(max_df/100)*100)
            
        dfs = [pd.concat(pv_systems[station]["raw_data"]['irrad'][substat][1],axis=0) for substat in pv_systems[station]["raw_data"]['irrad'] 
        if 'RT1' in substat]
        if len(dfs) > 0:
            max_df = np.max(np.max(pd.concat(dfs,axis=0).Etotpoa_RT1_Wm2))
            max_rt1 = int(np.ceil(max_df/100)*100)
    
        dfs = [pd.concat(pv_systems[station]["raw_data"]['irrad'][substat][1],axis=0) for substat in pv_systems[station]["raw_data"]['irrad'] 
        if 'ISE' in substat]
        if len(dfs) > 0:
            max_df = np.max(np.max(pd.concat(dfs,axis=0)))
            max_ise = int(np.ceil(max_df/100)*100)

        for substat in pv_systems[station]["raw_data"]['irrad']:
            dataframe = pd.concat(pv_systems[station]["raw_data"]['irrad'][substat][1],axis=0)
            fig, ax = plt.subplots(figsize=(9,8))       
            print(('Generating irradiance plot for %s' % substat))
            if 'Pyr' in substat:
                dataframe.plot(y=[0,1],ax=ax, color = ['r','darkgreen'])
                plt.legend((r'$G_{\rm tot}^{\downarrow}$',r'$G_{\rm tot}^{\angle}$'))
            elif 'suntracker' in substat:
                df = dataframe[['Etotdown_CMP11_Wm2','Ediffdown_CMP11_Wm2','Etotdown_SP2Lite_Wm2','Edirnorm_CHP1_Wm2']]
                plt.legend((r'$G_{\rm tot,CMP11}^{\downarrow}$',r'$G_{\rm diff,CMP11}^{\downarrow}$',
                            r'$G_{\rm tot,SP2Lite}^{\downarrow}$',r'$G_{\rm dir,CMP11}^{\odot}$'))
                ax.set_prop_cycle(color=['r','b','g','m'])
                #ax.set_prop_cycle(linestyle=['-','-','--','-'])
                ax.plot(df.index,df)
            elif 'mordor' in substat:
                df = dataframe[['Edirnorm_MS56_Wm2','Etotdown_CMP21_Wm2','Ediffdown_CMP21_Wm2',
                                'Etotdownlw_CGR4_Wm2','Ediffdownlw_CGR4_Wm2','Etotdown_ML020VM_Wm2',
                                'Ediffdown_ML020VM_Wm2']]
                plt.legend((r'$G_{\rm dir,MS56}^{\odot}$',r'$G_{\rm tot,CMP21}^{\downarrow}$',
                            r'$G_{\rm diff,CMP21}^{\downarrow}$',r'$G_{\rm tot,LW,CGR4}^{\downarrow}$',
                            r'$G_{\rm diff,LW,CGR4}^{\downarrow}$',r'$G_{\rm tot,ML020VM}^{\downarrow}$',
                            r'$G_{\rm diff,ML020VM}^{\downarrow}$'))
                ax.set_prop_cycle(color=['r','b','g','m'])
                #ax.set_prop_cycle(linestyle=['-','-','--','-'])
                ax.plot(df.index,df)
            elif 'RT1' in substat:
                if "Etotdown_RT1_Wm2" in dataframe.columns:
                    #df = dataframe[['Etotpoa_RT1_Wm2',"Etotdown_RT1_Wm2"]]                    
                    dataframe.plot(y=[0,1],ax=ax, color = ['r','darkgreen'])
                    ax.legend((r'$G_{\rm tot}^{\downarrow}$',r'$G_{\rm tot}^{\angle}$'))            
                else:
                    df = dataframe['Etotpoa_RT1_Wm2']
                    plt.legend((r'$G_{\rm tot}^{\angle}$'))
                    ax.plot(df.index,df,color = 'darkgreen')
            elif 'ISE' in substat:
                df = dataframe[['Etotdown_CMP11_Wm2','Etotpoa_32_S_CMP11_Wm2',
                            'Etotpoa_32_E_Si02_Wm2','Etotpoa_32_S_Si02_Wm2',
                            'Etotpoa_32_W_Si02_Wm2']]
                plt.legend((r'$G_{\rm tot,CMP11}^{\downarrow}$',r'$G_{\rm tot,CMP11}^{\angle 32,S}$',
                            r'$G_{\rm tot,Si02}^{\angle 32,E}$',r'$G_{\rm tot,Si02}^{\angle 32,S}$',
                            r'$G_{\rm tot,Si02}^{\angle 32,W}$'))                
                ax.set_prop_cycle(color=['r','b','g','m','c','k','y'])
                #ax.set_prop_cycle(linestyle=['-','-','--','-'])
                ax.plot(df.index,df)
                
            ax.set_ylabel('Irradiance (W/m$^2$)')#, color='b')
            ax.set_xlabel('Time (UTC)')
            plt.title('Measured irradiance at ' + station + ' from ' + substat)
            plt.savefig(os.path.join(dir_rad,'Total','rad_' + station + '_' + substat + '_' + description + '.png'))        
            plt.close(fig)            
            
            for ix, iday in enumerate(pv_systems[station]["raw_data"]['irrad'][substat][0].strftime("%Y-%m-%d")):#['days'].strftime("%Y-%m-%d"):                
                print(('Generating irradiance plot for %s, %s on %s' % (station,substat,iday)))
                fig, ax = plt.subplots(figsize=(9,8)) 
                ax.set_ylabel('Irradiance (W/m$^2$)')
                ax.set_xlabel('Time (UTC)')                
                
                if 'Pyr' in substat:
                    df = pv_systems[station]["raw_data"]['irrad'][substat][1][ix][['Etotdown_pyr_Wm2','Etotpoa_pyr_Wm2']]
                    ax.set_prop_cycle(color=['r', 'darkgreen'])
                    ax.plot(df.index,df)
                    ax.set_ylim([0,max_pyr])
                    
                    plt.legend((r'$G_{\rm tot}^{\downarrow}$',r'$G_{\rm tot}^{\angle}$'))                                
                elif 'suntracker' in substat:
                    df = pv_systems[station]["raw_data"]['irrad'][substat][1][ix][['Etotdown_CMP11_Wm2','Ediffdown_CMP11_Wm2',
                                   'Etotdown_SP2Lite_Wm2','Edirnorm_CHP1_Wm2']]
                    ax.set_prop_cycle(color=['r','b','g','m'])
                    #ax.set_prop_cycle(linestyle=['-','-','--','-'])
                    ax.plot(df.index,df)
                    ax.set_ylim([0,max_suntrack])
                    
                    plt.legend((r'$G_{\rm tot,CMP11}^{\downarrow}$',r'$G_{\rm diff,CMP11}^{\downarrow}$',
                            r'$G_{\rm tot,SP2Lite}^{\downarrow}$',r'$G_{\rm dir,CMP11}^{\odot}$'))
                elif 'mordor' in substat:
                    df = pv_systems[station]["raw_data"]['irrad'][substat][1][ix][['Edirnorm_MS56_Wm2','Etotdown_CMP21_Wm2',
                                   'Ediffdown_CMP21_Wm2','Etotdownlw_CGR4_Wm2','Ediffdownlw_CGR4_Wm2',
                                   'Etotdown_ML020VM_Wm2','Ediffdown_ML020VM_Wm2']]
                    ax.set_prop_cycle(color=['r','b','g','m','c','k','y'])
                    #ax.set_prop_cycle(linestyle=['-','-','--','-'])
                    ax.plot(df.index,df)
                    ax.set_ylim([0,max_mordor])
                    
                    plt.legend((r'$G_{\rm dir,MS56}^{\odot}$',r'$G_{\rm tot,CMP21}^{\downarrow}$',
                            r'$G_{\rm diff,CMP21}^{\downarrow}$',r'$G_{\rm tot,LW,CGR4}^{\downarrow}$',
                            r'$G_{\rm diff,LW,CGR4}^{\downarrow}$',r'$G_{\rm tot,ML020VM}^{\downarrow}$',
                            r'$G_{\rm diff,ML020VM}^{\downarrow}$'))
                elif 'RT1' in substat:
                    if "Etotdown_RT1_Wm2" in pv_systems[station]["raw_data"]['irrad'][substat][1][ix].columns:
                        df = pv_systems[station]["raw_data"]['irrad'][substat][1][ix][['Etotpoa_RT1_Wm2',"Etotdown_RT1_Wm2"]]
                        ax.set_prop_cycle(color=['darkgreen',"red"])
                        ax.plot(df.index,df)
                        ax.set_ylim([0,max_rt1])
                        plt.legend([r'$G_{\rm tot}^{\angle}$',r'$G_{\rm tot}^{\downarrow}$'])
                    else:
                        df = pv_systems[station]["raw_data"]['irrad'][substat][1][ix][['Etotpoa_RT1_Wm2']]
                        ax.set_prop_cycle(color=['darkgreen'])
                        ax.plot(df.index,df)
                        ax.set_ylim([0,max_rt1])
                        plt.legend([r'$G_{\rm tot}^{\angle}$'])
                elif 'ISE' in substat:
                    df = pv_systems[station]["raw_data"]['irrad'][substat][1][ix][['Etotdown_CMP11_Wm2',
                                   'Etotpoa_32_S_CMP11_Wm2','Etotpoa_32_E_Si02_Wm2',
                                   'Etotpoa_32_S_Si02_Wm2','Etotpoa_32_W_Si02_Wm2']]
                    ax.set_prop_cycle(color=['r','b','g','m','c','k','y'])
                    #ax.set_prop_cycle(linestyle=['-','-','--','-'])
                    ax.plot(df.index,df)
                    ax.set_ylim([0,max_ise])
                    
                    plt.legend((r'$G_{\rm tot,CMP11}^{\downarrow}$',r'$G_{\rm tot,CMP11}^{\angle 32,S}$',
                            r'$G_{\rm tot,Si02}^{\angle 32,E}$',r'$G_{\rm tot,Si02}^{\angle 32,S}$',
                            r'$G_{\rm tot,Si02}^{\angle 32,W}$'))                    
                    
                plt.title('Measured irradiance at ' + station + ' from ' + substat + ' on ' + iday)
                
                datemin = np.datetime64(iday + ' 03:30')
                datemax = np.datetime64(iday + ' 19:00')
                ax.set_xlim([datemin, datemax])
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                fig.autofmt_xdate(rotation=0,ha='center')               
                 
                plt.savefig(os.path.join(dir_rad,'Daily',station,'rad_' + 
                                         station + '_' + substat + '_' + iday + '.png'))         
                
                plt.close(fig)                  

def plot_temp_data (pv_systems,home,paths,info,styles,description):
    """
    Plot temperature data for each station

    Parameters
    ----------
    pv_systems : dictionary
        dictionary with information and data from each system
    home : string
        home path
    paths : dictionary
        dictionary with paths for loading data and saving plots
    info : dataframe
        dataframe with station info
    styles : dictionary
        dictionary with plotting styles
    description : string
        description of current data
    
    Returns
    -------
    None.

    """
    
    plt.ioff()
    plt.style.use(styles['single_small'])
    plt.close('all')
    
    main_dirs_exist = list_dirs(os.path.join(home,paths['plots']['main']))
    dir_temp = os.path.join(home,paths['plots']['main'],paths['plots']['temp'])
    
    if paths['plots']['temp'] not in main_dirs_exist:
        os.mkdir(os.path.join(dir_temp))
    
    dirs_exist = list_dirs(os.path.join(dir_temp))    
    if 'Total' not in dirs_exist:
        os.mkdir(os.path.join(dir_temp,'Total'))
            
    if 'Daily' not in dirs_exist:
        os.mkdir(os.path.join(dir_temp,'Daily'))

    stat_dirs_exist = list_dirs(os.path.join(dir_temp,'Daily'))    

    for station in info.index:
        if pv_systems[station]["raw_data"]['temp']:
                leg_temp = [r'$T_{\rm module}$']
        elif 'RT1' in pv_systems[station]["raw_data"]['irrad']:
                leg_temp = [r'$T_{\rm module}$']
        elif 'suntracker' in pv_systems[station]["raw_data"]['irrad']:
                leg_temp = [r'$T_{\rm module,upper}$',r'$T_{\rm module,lower}$']
        else:
            leg_temp = []
        if pv_systems[station]["raw_data"]['irrad']:
            for substat in pv_systems[station]["raw_data"]['irrad']:
                leg_temp.append(r'$T_{\rm amb,' + substat + '}$')
    
        if pv_systems[station]["raw_data"]['temp']:
            if station not in stat_dirs_exist:
                os.mkdir(os.path.join(dir_temp,'Daily',station))
    
            for substat in pv_systems[station]["raw_data"]['temp']:
                dataframe = pd.concat(pv_systems[station]["raw_data"]['temp'][substat][1],axis=0)
                fig, ax = plt.subplots(figsize=(9,8))       
                print(('Generating temperature plot for %s' % station))
                dataframe.plot(ax=ax, color = 'r',legend=False)
                #plt.legend((r'$G_{\rm tot}^{\downarrow}$',r'$G_{\rm tot}^{\angle}$'))
                ax.set_ylabel(r'Temperature ($^{\circ}$C)')#, color='b')
                ax.set_xlabel('Time (UTC)')
                plt.title('Measured module temperature at ' + station)
                plt.savefig(os.path.join(dir_temp,'Total','temp_' + station + '_' + description + '.png'))        
                plt.close(fig)            
                
                for ix, iday in enumerate(pv_systems[station]["raw_data"]['temp'][substat][0].strftime("%Y-%m-%d")):#['days'].strftime("%Y-%m-%d"):                
                    print(('Generating temperature plot for %s, %s on %s' % (station,substat,iday)))
                    fig, ax = plt.subplots(figsize=(9,8)) 
                    ax.set_ylabel(r'Temperature ($^{\circ}$C)')
                    ax.set_xlabel('Time (UTC)')
                    ax.set_ylim([-10,60])
                    #pv_systems[station]["raw_data"]['irrad'][substat][1][ix].plot(y=[0,1], ax=ax, legend=True, color = ['r','darkgreen'])
                    df = pv_systems[station]["raw_data"]['temp'][substat][1][ix]
                    ax.set_prop_cycle(color=['r', 'darkgreen'])
                    ax.plot(df.index,df)
                    
                    for substat_pyr in pv_systems[station]["raw_data"]['irrad']:
                        #Set labels and limits                        
                        #Plot data
                        if iday in pv_systems[station]["raw_data"]['irrad'][substat_pyr][0]:
                            new_ix = list(pv_systems[station]["raw_data"]['irrad'][substat_pyr][0].strftime("%Y-%m-%d")).index(iday)
                            if "Pyr" in substat_pyr:
                                df = pv_systems[station]["raw_data"]['irrad'][substat_pyr][1][new_ix]['T_ambient_pyr_C']                            
                            ax.plot(df.index,df)
                            
                    ax.legend(leg_temp)                                
                    plt.title('Measured temperature at ' + station + ' on ' + iday)
                    
                    datemin = np.datetime64(iday + ' 00:00')
                    datemax = np.datetime64(iday + ' 23:59')
                    ax.set_xlim([datemin, datemax])
                    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                    fig.autofmt_xdate(rotation=0,ha='center')               
                    
                    plt.savefig(os.path.join(dir_temp,'Daily',station,'temp_' + 
                                             station + '_' + iday + '.png'))         
                    
                    plt.close(fig)   
        
        if 'RT1' in pv_systems[station]["raw_data"]['irrad'] and 'RT1' not in pv_systems[station]["raw_data"]['temp']:
            substat = 'RT1'
            if station not in stat_dirs_exist:
                os.mkdir(os.path.join(dir_temp,'Daily',station))
    
            
            dataframe = pd.concat(pv_systems[station]["raw_data"]['irrad'][substat][1],axis=0)
            fig, ax = plt.subplots(figsize=(9,8))       
            print(('Generating temperature plot for %s' % station))
            dataframe.T_module_C.plot(ax=ax, color = 'r',legend=False)
            #plt.legend((r'$G_{\rm tot}^{\downarrow}$',r'$G_{\rm tot}^{\angle}$'))
            ax.set_ylabel(r'Temperature ($^{\circ}$C)')#, color='b')
            ax.set_xlabel('Time (UTC)')
            plt.title('Measured module temperature at ' + station)
            plt.savefig(os.path.join(dir_temp,'Total','temp_' + station + '_' + description + '.png'))        
            plt.close(fig)            
                
            for ix, iday in enumerate(pv_systems[station]["raw_data"]['irrad'][substat][0].strftime("%Y-%m-%d")):#['days'].strftime("%Y-%m-%d"):                
                print(('Generating temperature plot for %s, %s on %s' % (station,substat,iday)))
                fig, ax = plt.subplots(figsize=(9,8)) 
                ax.set_ylabel(r'Temperature ($^{\circ}$C)')
                ax.set_xlabel('Time (UTC)')
                ax.set_ylim([-10,60])
                #pv_systems[station]["raw_data"]['irrad'][substat][1][ix].plot(y=[0,1], ax=ax, legend=True, color = ['r','darkgreen'])
                df = pv_systems[station]["raw_data"]['irrad'][substat][1][ix].T_module_C
                ax.set_prop_cycle(color=['r', 'darkgreen','blue','k'])
                ax.plot(df.index,df)
                
                for substat_pyr in pv_systems[station]["raw_data"]['irrad']:
                    if 'Pyr' in substat_pyr:
                        #Set labels and limits                        
                        #Plot data
                        if iday in pv_systems[station]["raw_data"]['irrad'][substat_pyr][0]:
                            new_ix = list(pv_systems[station]["raw_data"]['irrad'][substat_pyr][0].strftime("%Y-%m-%d")).index(iday)
                            df = pv_systems[station]["raw_data"]['irrad'][substat_pyr][1][new_ix]['T_ambient_pyr_C']
                            ax.plot(df.index,df)
                        
                ax.legend(leg_temp)                                
                plt.title('Measured temperature at ' + station + ' on ' + iday)
                
                datemin = np.datetime64(iday + ' 00:00')
                datemax = np.datetime64(iday + ' 23:59')
                ax.set_xlim([datemin, datemax])
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                fig.autofmt_xdate(rotation=0,ha='center')               
                
                plt.savefig(os.path.join(dir_temp,'Daily',station,'temp_' + 
                                         station + '_' + iday + '.png'))         
                
                plt.close(fig)              
        
        if 'suntracker' in pv_systems[station]["raw_data"]['irrad']:
            substat = 'suntracker'
            if station not in stat_dirs_exist:
                os.mkdir(os.path.join(dir_temp,'Daily',station))
    
            
            dataframe = pd.concat(pv_systems[station]["raw_data"]['irrad'][substat][1],axis=0)\
                        .filter(regex='^T_module', axis=1)
            fig, ax = plt.subplots(figsize=(9,8))       
            print(('Generating temperature plot for %s' % station))
            ax.set_prop_cycle(linestyle=['-', '--',':','-.'])
            dataframe.plot(ax=ax, color = 'r')
            
            #plt.legend((r'$G_{\rm tot}^{\downarrow}$',r'$G_{\rm tot}^{\angle}$'))
            ax.set_ylabel(r'Temperature ($^{\circ}$C)')#, color='b')
            ax.set_xlabel('Time (UTC)')
            plt.title('Measured module temperature at ' + station)
            plt.savefig(os.path.join(dir_temp,'Total','temp_' + station + '_' + description + '.png'))        
            plt.close(fig)            
                
            for ix, iday in enumerate(pv_systems[station]["raw_data"]['irrad'][substat][0].strftime("%Y-%m-%d")):#['days'].strftime("%Y-%m-%d"):                
                print(('Generating temperature plot for %s, %s on %s' % (station,substat,iday)))
                fig, ax = plt.subplots(figsize=(9,8)) 
                ax.set_ylabel(r'Temperature ($^{\circ}$C)')
                ax.set_xlabel('Time (UTC)')
                ax.set_ylim([-10,60])
                #pv_systems[station]["raw_data"]['irrad'][substat][1][ix].plot(y=[0,1], ax=ax, legend=True, color = ['r','darkgreen'])
                df = pv_systems[station]["raw_data"]['irrad'][substat][1][ix].filter(regex='^T_module', axis=1)
                ax.set_prop_cycle(color=['r', 'darkgreen','blue','k'])
                ax.plot(df.index,df)
                
                for substat_pyr in pv_systems[station]["raw_data"]['irrad']:
                    if 'Pyr' in substat_pyr:
                        #Set labels and limits                        
                        #Plot data
                        if iday in pv_systems[station]["raw_data"]['irrad'][substat_pyr][0]:
                            new_ix = list(pv_systems[station]["raw_data"]['irrad'][substat_pyr][0].strftime("%Y-%m-%d")).index(iday)
                            df = pv_systems[station]["raw_data"]['irrad'][substat_pyr][1][new_ix]['T_ambient_pyr_C']
                            ax.plot(df.index,df)
                        
                ax.legend(leg_temp)                                
                plt.title('Measured temperature at ' + station + ' on ' + iday)
                
                datemin = np.datetime64(iday + ' 00:00')
                datemax = np.datetime64(iday + ' 23:59')
                ax.set_xlim([datemin, datemax])
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                fig.autofmt_xdate(rotation=0,ha='center')               
                
                plt.savefig(os.path.join(dir_temp,'Daily',station,'temp_' + 
                                         station + '_' + iday + '.png'))         
                
                plt.close(fig)              
            
def plot_combined_data_power_rad (pv_systems,home,paths,info,styles,description):
    """    
    Plot combined PV and irradiance data for each station

    Parameters
    ----------
    pv_systems : dictionary
        dictionary with information and data from each system
    home : string
        home path
    paths : dictionary
        dictionary with paths for loading data and saving plots
    info : dataframe
        dataframe with station info
    styles : dictionary
        dictionary with plotting styles
    description : string
        description of current data
        
    Returns
    -------
    None.

    """
    
    plt.ioff()    
    plt.close('all')
    
    main_dirs_exist = list_dirs(os.path.join(home,paths['plots']['main']))
    dir_combo = os.path.join(home,paths['plots']['main'],paths['plots']['combo'])
    
    if paths['plots']['combo'] not in main_dirs_exist:
        os.mkdir(dir_combo)
        
    dirs_exist = list_dirs(os.path.join(dir_combo))        
            
    if 'Power_Rad_Grid' not in dirs_exist:
        os.mkdir(os.path.join(dir_combo,'Power_Rad_Grid'))

    stat_dirs_exist = list_dirs(os.path.join(dir_combo,'Power_Rad_Grid'))  
    
    #Plotting loop for combining all values into a grid    
    for station in info.index:        
        
        # Find the union of all the days in each datasets
        combined_days = pd.DatetimeIndex
        if pv_systems[station]["raw_data"]['pv']:
            for substat in pv_systems[station]["raw_data"]['pv']:
                if not combined_days.empty:
                    combined_days = combined_days.union(pv_systems[station]["raw_data"]['pv'][substat][0])                    
                else:
                    combined_days = pv_systems[station]["raw_data"]['pv'][substat][0]
                
        if info['Irrad-Sensor'][station] != 0:
            for substat in pv_systems[station]["raw_data"]['irrad']:
                if 'Pyr' in substat or "RT1" in substat:
                    if not combined_days.empty:
                        combined_days = combined_days.union(pv_systems[station]["raw_data"]['irrad'][substat][0])
                    else:
                        combined_days = pv_systems[station]["raw_data"]['irrad'][substat][0]
        
        pv_systems[station]['days'] = combined_days
        
        if combined_days.empty:
            print(('No data to plot for %s' % station))
        else:
            if station not in stat_dirs_exist:
                os.mkdir(os.path.join(dir_combo,'Power_Rad_Grid',station))
            #Get maximum for plotting axis (round to nearest 10 or 5) and get legend
            num_plots = 0
            if pv_systems[station]["raw_data"]['pv']:
                dfs = [(substat,pd.concat(pv_systems[station]["raw_data"]['pv'][substat][1],axis=0)) 
                for substat in pv_systems[station]["raw_data"]['pv']]
                                
                dfs_auew = [df[1] for df in dfs if 'auew' in df[0]]                
                if dfs_auew:
                    num_plots = num_plots + 1
                    dataframe_auew = pd.concat(dfs_auew,axis=0)
                    max_df = np.max(np.max(dataframe_auew))
                    if max_df > 50:
                        max_auew = np.ceil(max_df/10)*10
                    else:
                        max_auew = np.ceil(max_df/5)*5
                    
                    leg_auew = ['$' + df[1].columns.values[0].strip('_kW') + '_' + df[0].split('_')[1] + '$' 
                              for df in dfs if 'auew' in df[0]]
                
                dfs_egrid = [df[1] for df in dfs if 'egrid' in df[0]]
                if dfs_egrid:
                    num_plots = num_plots + 1
                    dataframe_egrid = pd.concat(dfs_egrid,axis=0)
                    max_df = np.max(np.max(dataframe_egrid))
                    if max_df > 50:
                        max_egrid = np.ceil(max_df/10)*10
                    else:
                        max_egrid = np.ceil(max_df/5)*5
                    
                    leg_egrid = ['$' + df[1].columns.values[0].strip('_kW') + '_' + df[0][-1] + '$' 
                              for df in dfs if 'egrid' in df[0] and 'd' != df[0][-1]]
                    
                dfs_solarwatt = [df[1] for df in dfs if 'myreserve' in df[0]]
                if dfs_solarwatt:
                    num_plots = num_plots + 1
                    dataframe_solarwatt = pd.concat(dfs_solarwatt,axis=0)
                    max_df = np.max(np.max(dataframe_solarwatt["P_kW"]))
                    if max_df > 50:
                        max_solarwatt = np.ceil(max_df/10)*10
                    elif max_df > 10:
                        max_solarwatt = np.ceil(max_df/5)*5
                    else:
                        max_solarwatt = np.ceil(max_df)
                    
                    leg_solarwatt = ['myreserve']

            if pv_systems[station]["raw_data"]['irrad']:
                dfs_rad = [pd.concat(pv_systems[station]["raw_data"]['irrad'][substat][1],axis=0) 
                for substat in pv_systems[station]["raw_data"]['irrad'] if 'Pyr' in substat]
                if dfs_rad:
                    num_plots = num_plots + len(dfs_rad)
                    max_df = np.max(np.max(pd.concat(dfs_rad,axis=0)))
                    max_pyr = int(np.ceil(max_df/100)*100)  
                else:
                    max_pyr = 0
                dfs_rad = [pd.concat(pv_systems[station]["raw_data"]['irrad'][substat][1],axis=0) 
                for substat in pv_systems[station]["raw_data"]['irrad'] if 'RT1' in substat]
                if dfs_rad:
                    num_plots = num_plots + len(dfs_rad)                                     
                    max_df = np.max([np.max(pd.concat(dfs_rad,axis=0).Etotpoa_RT1_Wm2),max_pyr])
                    max_pyr = int(np.ceil(max_df/100)*100)                 
                dfs_rad = [pd.concat(pv_systems[station]["raw_data"]['irrad'][substat][1],axis=0) 
                for substat in pv_systems[station]["raw_data"]['irrad'] if 'ISE' in substat]
                if dfs_rad:
                    num_plots = num_plots + len(dfs_rad)
                    max_df = np.max(np.max(pd.concat(dfs_rad,axis=0)))
                    max_pyr = int(np.ceil(max_df/100)*100)  
            
            for ix, iday in enumerate(combined_days.strftime("%Y-%m-%d")):
                print(("Generating power-irradiance combo plots for %s on %s" % (station,iday)))                
                plt.style.use(styles['combo_small'])                 
                if num_plots <= 4:
                    fig, axs = plt.subplots(2, 2)#, sharex='all', sharey='none')                    
                    fig.subplots_adjust(wspace=0.35)  
                    legwidth = 2.0
                else:
                    fig, axs = plt.subplots(2, 3)#, sharex='all', sharey='none')   
                    fig.subplots_adjust(wspace=0.1) 
                    legwidth = 1.0
                
                n_ax = 0                                    
                #PV power plots
                if pv_systems[station]["raw_data"]['pv']:
                    if dfs_auew:
                        axs.flat[n_ax].set_ylabel('Power (kW)')
                        axs.flat[n_ax].set_title('AUEW PV',loc='left',pad=-18)
                        axs.flat[n_ax].set_ylim([0,max_auew])
                        axs.flat[n_ax].set_prop_cycle(color=['blue', 'k','darkred'])
                        
                        for substat in pv_systems[station]["raw_data"]['pv']:
                            if iday in pv_systems[station]["raw_data"]['pv'][substat][0] and 'auew' in substat:
                                new_ix = list(pv_systems[station]["raw_data"]['pv'][substat][0].strftime("%Y-%m-%d")).index(iday)
                                df = pv_systems[station]["raw_data"]['pv'][substat][1][new_ix]
                                axs.flat[n_ax].plot(df.index,df)                                
                                axs.flat[n_ax].legend(leg_auew,handlelength=legwidth,loc='upper right')
                                
                        n_ax = n_ax + 1
                        
                    if dfs_egrid:
                        if num_plots <= 4:
                            axs.flat[n_ax].set_ylabel('Power (kW)')
                            
                        axs.flat[n_ax].set_title('egrid PV',loc='left',pad=-18)
                        axs.flat[n_ax].set_ylim([0,max_egrid])
                        axs.flat[n_ax].set_prop_cycle(color=['blue', 'k','darkred'])
                        
                        for substat in pv_systems[station]["raw_data"]['pv']:
                            if iday in pv_systems[station]["raw_data"]['pv'][substat][0] and 'egrid' in substat:
                                new_ix = list(pv_systems[station]["raw_data"]['pv'][substat][0].strftime("%Y-%m-%d")).index(iday)
                                df = pv_systems[station]["raw_data"]['pv'][substat][1][new_ix]
                                axs.flat[n_ax].plot(df.index,df)
                                axs.flat[n_ax].legend(leg_egrid,handlelength=legwidth,loc='upper right')
                            
                        n_ax = n_ax + 1
                        
                    if dfs_solarwatt:
                        if num_plots <= 4:
                            axs.flat[n_ax].set_ylabel('Power (kW)')
                            
                        axs.flat[n_ax].set_title('Solarwatt PV',loc='left',pad=-18)
                        axs.flat[n_ax].set_ylim([0,max_solarwatt])
                        axs.flat[n_ax].set_prop_cycle(color=['blue', 'k','darkred'])
                        
                        for substat in pv_systems[station]["raw_data"]['pv']:
                            if iday in pv_systems[station]["raw_data"]['pv'][substat][0] and 'myreserve' in substat:
                                new_ix = list(pv_systems[station]["raw_data"]['pv'][substat][0].strftime("%Y-%m-%d")).index(iday)
                                df = pv_systems[station]["raw_data"]['pv'][substat][1][new_ix]
                                axs.flat[n_ax].plot(df.index,df["P_kW"])
                                axs.flat[n_ax].legend(leg_solarwatt,handlelength=legwidth,loc='upper right')
                            
                        n_ax = n_ax + 1
                
                if num_plots > 4:
                    n_ax = n_ax + 1
                #Irradiance plots
                if pv_systems[station]["raw_data"]['irrad']:
                    for substat in pv_systems[station]["raw_data"]['irrad']:
#                        if 'suntracker' in substat:
#                            #Set labels and limits                            
#                            axs.flat[n_ax].set_ylabel('Irradiance (W/m$^2$)')
#                            axs.flat[n_ax].set_yticklabels('')
#                            axs.flat[n_ax].set_title(substat,loc='left',pad=-18)                    
#                            axs.flat[n_ax].set_ylim([0,max_pyr])
#                            
#                            #Plot data
#                            if iday in pv_systems[station]["raw_data"]['irrad'][substat][0]:
#                                new_ix = list(pv_systems[station]["raw_data"]['irrad'][substat][0].strftime("%Y-%m-%d")).index(iday)
#                                df = pv_systems[station]["raw_data"]['irrad'][substat][1][new_ix][['Etotdown_CMP11_Wm2','Ediffdown_CMP11_Wm2',
#                                   'Etotdown_SP2Lite_Wm2','Edirnorm_CHP1_Wm2']]
#                                axs.flat[n_ax].set_prop_cycle(color=['r','b','g','m'])
#                                axs.flat[n_ax].plot(df.index,df)
#                    
#                                axs.flat[n_ax].legend((r'$G_{\rm tot,CMP11}^{\downarrow}$',r'$G_{\rm diff,CMP11}^{\downarrow}$',
#                                        r'$G_{\rm tot,SP2Lite}^{\downarrow}$',r'$G_{\rm dir,CMP11}^{\odot}$'))
#                                                  
#                            n_ax = n_ax + 1
                            
                        if 'Pyr' in substat:
                            #Set labels and limits
                            if num_plots <= 4 or n_ax == 3:
                                axs.flat[n_ax].set_ylabel('Irradiance (W/m$^2$)')
                            else:
                                axs.flat[n_ax].set_yticklabels('')
                            axs.flat[n_ax].set_title(substat,loc='left',pad=-18)                    
                            axs.flat[n_ax].set_ylim([0,max_pyr])
                            
                            #Plot data
                            if iday in pv_systems[station]["raw_data"]['irrad'][substat][0]:
                                new_ix = list(pv_systems[station]["raw_data"]['irrad'][substat][0].strftime("%Y-%m-%d")).index(iday)
                                df = pv_systems[station]["raw_data"]['irrad'][substat][1][new_ix][['Etotdown_pyr_Wm2','Etotpoa_pyr_Wm2']]
                                axs.flat[n_ax].set_prop_cycle(color=['r', 'darkgreen'])
                                axs.flat[n_ax].plot(df.index,df)
                                #pv_systems[station]["raw_data"]['irrad'][substat][1][new_ix].plot(y=[0,1],ax=axs.flat[n_ax],
                                          #legend=False, color = ['r','darkgreen'])
                                if num_plots <= 4:
                                    axs.flat[n_ax].legend((r'$G_{\rm tot}^{\downarrow}$',r'$G_{\rm tot}^{\angle}$'))
                                else:
                                    if n_ax == 4:
                                        axs.flat[n_ax].legend((r'$G_{\rm tot}^{\downarrow}$',r'$G_{\rm tot}^{\angle}$')
                                        ,handlelength=legwidth,loc='upper right')
                                                  
                            n_ax = n_ax + 1
                            
                        if 'RT1' in substat:
                            #Set labels and limits
                            if num_plots <= 4 or n_ax == 3:
                                axs.flat[n_ax].set_ylabel('Irradiance (W/m$^2$)')
                            else:
                                axs.flat[n_ax].set_yticklabels('')
                            axs.flat[n_ax].set_title(substat,loc='left',pad=-18)                    
                            axs.flat[n_ax].set_ylim([0,max_pyr])
                            
                            #Plot data
                            if iday in pv_systems[station]["raw_data"]['irrad'][substat][0]:
                                new_ix = list(pv_systems[station]["raw_data"]['irrad'][substat][0].strftime("%Y-%m-%d")).index(iday)
                                if 'Etotdown_RT1_Wm2' in pv_systems[station]["raw_data"]['irrad'][substat][1][new_ix].columns:
                                    df = pv_systems[station]["raw_data"]['irrad'][substat][1][new_ix]\
                                    [['Etotpoa_RT1_Wm2','Etotdown_RT1_Wm2']]
                                    leg = [r'$G_{\rm tot}^{\angle}$',r'$G_{\rm tot}^{\downarrow}$']
                                    
                                else:
                                    df = pv_systems[station]["raw_data"]['irrad'][substat][1][new_ix]['Etotpoa_RT1_Wm2']
                                    leg = [r'$G_{\rm tot}^{\angle}$']
                                axs.flat[n_ax].set_prop_cycle(color=['darkgreen','red'])
                                axs.flat[n_ax].plot(df.index,df)
                                axs.flat[n_ax].legend(leg,handlelength=legwidth,loc='upper right')
                                #pv_systems[station]["raw_data"]['irrad'][substat][1][new_ix].plot(y=[0,1],ax=axs.flat[n_ax],
                                          #legend=False, color = ['r','darkgreen'])
                                #if num_plots <= 4:                                
                                                  
                            n_ax = n_ax + 1
                            
                        if 'ISE' in substat:
                            #Set labels and limits
                            if num_plots <= 4 or n_ax == 3:
                                axs.flat[n_ax].set_ylabel('Irradiance (W/m$^2$)')
                            else:
                                axs.flat[n_ax].set_yticklabels('')
                            axs.flat[n_ax].set_title(substat,loc='left',pad=-18)                    
                            axs.flat[n_ax].set_ylim([0,max_pyr])
                            
                            #Plot data
                            if iday in pv_systems[station]["raw_data"]['irrad'][substat][0]:
                                new_ix = list(pv_systems[station]["raw_data"]['irrad'][substat][0].strftime("%Y-%m-%d")).index(iday)
                                df = pv_systems[station]["raw_data"]['irrad'][substat][1][new_ix][['Etotdown_CMP11_Wm2','Etotpoa_32_S_CMP11_Wm2']]
                                axs.flat[n_ax].set_prop_cycle(color=['r','darkgreen'])
                                axs.flat[n_ax].plot(df.index,df)
                                #pv_systems[station]["raw_data"]['irrad'][substat][1][new_ix].plot(y=[0,1],ax=axs.flat[n_ax],
                                          #legend=False, color = ['r','darkgreen'])
                                #if num_plots <= 4:
                                axs.flat[n_ax].legend([r'$G_{\rm tot}^{\downarrow}$',r'$G_{\rm tot}^{\angle}$'],
                                        handlelength=legwidth,loc='upper right')
                                                  
                            n_ax = n_ax + 1
                    
                # Set axis limits                                            
                for ax in axs.flat:
                    datemin = np.datetime64(iday + ' 03:30')
                    datemax = np.datetime64(iday + ' 19:00')     
                    ax.set_xlim([datemin, datemax])
                    ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
                fig.autofmt_xdate(rotation=0,ha='center') 
                    
                fig.add_subplot(111, frameon=False)
                # hide tick and tick label of) the big axes
                plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
                plt.grid(False)
                plt.xlabel("Time (UTC)")
                fig.suptitle("Data from %s on %s" % (station,iday),fontsize=18)   
                fig.tight_layout()
                fig.autolayout = False
                fig.subplots_adjust(top=0.94)            
                fig.savefig(os.path.join(dir_combo,'Power_Rad_Grid',station,'combo_plot_' + 
                                         station + '_' + iday))
                plt.close(fig)            
        
    return pv_systems

def plot_combined_data_power_temp(pv_systems,home,paths,info,styles,description):
    """
    Plot combined power and temperature data for each station

    Parameters
    ----------
    pv_systems : dictionary
        dictionary with information and data from each system
    home : string
        home path
    paths : dictionary
        dictionary with paths for loading data and saving plots
    info : dataframe
        dataframe with station info
    styles : dictionary
        dictionary with plotting styles
    description : string
        description of current data

    Returns
    -------
    None.

    """
    
    plt.ioff()    
    plt.close('all')
    
    main_dirs_exist = list_dirs(os.path.join(home,paths['plots']['main']))
    dir_combo = os.path.join(home,paths['plots']['main'],paths['plots']['combo'])
    
    if paths['plots']['combo'] not in main_dirs_exist:
        os.mkdir(dir_combo)
        
    dirs_exist = list_dirs(os.path.join(dir_combo))        
            
    if 'Power_Temp' not in dirs_exist:
        os.mkdir(os.path.join(dir_combo,'Power_Temp'))

    stat_dirs_exist = list_dirs(os.path.join(dir_combo,'Power_Temp'))  
    
    #Plotting loop for combining temperature and power
    for station in info.index:  
        combined_days = pd.DatetimeIndex        
        if info['Temp-Sensor'][station] != 0:
            for substat in pv_systems[station]["raw_data"]['temp']:
                if not combined_days.empty:
                    combined_days = combined_days.union(pv_systems[station]["raw_data"]['temp'][substat][0])
                else:
                    combined_days = pv_systems[station]["raw_data"]['temp'][substat][0]
	
        if pv_systems[station]["raw_data"]['pv']:
            for substat in pv_systems[station]["raw_data"]['pv']:
                if not combined_days.empty:
                    combined_days = combined_days.union(pv_systems[station]["raw_data"]['pv'][substat][0])                    
                else:
                    combined_days = pv_systems[station]["raw_data"]['pv'][substat][0]
    
        if info['PN-Einheit'][station] != 0:
            for substat in pv_systems[station]["raw_data"]['irrad']:
                if 'Pyr' in substat:
                    if not combined_days.empty:
                        combined_days = combined_days.union(pv_systems[station]["raw_data"]['irrad'][substat][0])
                    else:
                        combined_days = pv_systems[station]["raw_data"]['irrad'][substat][0]
                    
        pv_systems[station]['days'] = combined_days
        
        if combined_days.empty:
            print(('No data to plot for %s' % station))
        else:
            if station not in stat_dirs_exist:
                os.mkdir(os.path.join(dir_combo,'Power_Temp',station))

    	    #Get maximum for plotting axis (round to nearest 10 or 5) and get legend
            if pv_systems[station]["raw_data"]['pv']:
                dfs = [pd.concat(pv_systems[station]["raw_data"]['pv'][substat][1],axis=0) 
                    for substat in pv_systems[station]["raw_data"]['pv']]
                max_df = np.max(np.max(pd.concat(dfs,axis=0)))
                if max_df > 50:
                    max_pv = np.ceil(max_df/10)*10
                else:
                    max_pv = np.ceil(max_df/5)*5
                    
#                leg_auew = ['$' + df[1].columns.values[0].strip('_kW') + r'_{\rm auew ' + df[0].split('_')[1] + '}$' 
#                              for df in dfs if 'auew' in df[0]]
#                
#                leg_egrid = ['$' + df[1].columns.values[0].strip('_kW') + r'_{\rm egrid}$' 
#                              for df in dfs if 'egrid' in df[0]]
#                
#                leg_pv = leg_auew + leg_egrid
            
            plt.style.use(styles['single_small']) 
            #Make combined temperature and power plots
            for ix, iday in enumerate(combined_days.strftime("%Y-%m-%d")):
                if pv_systems[station]["raw_data"]['pv'] and (pv_systems[station]["raw_data"]['temp'] or
                             'RT1' in pv_systems[station]["raw_data"]['irrad'] or 
                             'suntracker' in pv_systems[station]["raw_data"]['irrad']):
                    print(("Generating temperature combo plots for %s on %s" % (station,iday)))
                    fig, ax1 = plt.subplots(figsize=(9,8))
                    ax1.set_prop_cycle(linestyle=['-', '--',':','-.'])
                    leg_pv = []
                    #Rather plot the high resolution data with temperature
                    for substat in pv_systems[station]["raw_data"]['pv']:
                        ax1.set_ylabel('Power (kW)',color='blue')
                        ax1.set_ylim([0,max_pv])
                        
                        if iday in pv_systems[station]["raw_data"]['pv'][substat][0]:
                            new_ix = list(pv_systems[station]["raw_data"]['pv'][substat][0].strftime("%Y-%m-%d")).index(iday)
                            df = pv_systems[station]["raw_data"]['pv'][substat][1][new_ix]
                            ax1.plot(df.index,df,color='blue')
                            if substat[-1] != 'd':
                                leg_pv.append(r'$P_{\rm ' + substat + '}$')                                                                   
                            
                    ax1.legend(leg_pv,loc=(0.02,0.78)) 
                    ax1.tick_params('y', colors='blue')   
                    plt.grid(False,axis='y')
                
                    datemin = np.datetime64(iday + ' 00:00')
                    datemax = np.datetime64(iday + ' 23:59')     
                    ax1.set_xlim([datemin, datemax])
                    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=3))
                    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                    ax1.set_xlabel('Time (UTC)')
                    
                    ax2 = ax1.twinx()
                    plt.grid(False,axis='y')
                    #Plot temperature data
                    leg_temp = []
                    for substat in pv_systems[station]["raw_data"]['temp']:
                        #Set labels and limits
                        ax2.set_ylabel(r'Temperature ($^{\circ}$C)',color='r')
                        ax2.set_ylim([-10,60])
                        
                        #Plot data
                        if iday in pv_systems[station]["raw_data"]['temp'][substat][0]:
                            new_ix = list(pv_systems[station]["raw_data"]['temp'][substat][0].strftime("%Y-%m-%d")).index(iday)
                            df = pv_systems[station]["raw_data"]['temp'][substat][1][new_ix]
                            ax2.set_prop_cycle(linestyle=['-', '--',':','-.'])
                            ax2.plot(df.index,df,color='r')                            
                            ax2.tick_params('y', colors='r')
                            leg_temp.append(r'$T_{\rm module}$')
                            #ax2.legend(leg_temp,loc='upper right')
                            
                    for substat in pv_systems[station]["raw_data"]['irrad']:
                        if 'RT1' in substat:
                            #Set labels and limits
                            ax2.set_ylabel(r'Temperature ($^{\circ}$C)',color='r')
                            ax2.set_ylim([-10,60])
                            
                            #Plot data
                            if iday in pv_systems[station]["raw_data"]['irrad'][substat][0]:
                                new_ix = list(pv_systems[station]["raw_data"]['irrad'][substat][0].strftime("%Y-%m-%d")).index(iday)
                                df = pv_systems[station]["raw_data"]['irrad'][substat][1][new_ix]
                                ax2.set_prop_cycle(linestyle=['-', '--',':','-.'])
                                ax2.plot(df.index,df.T_module_C,color='r')                            
                                ax2.tick_params('y', colors='r')
                                leg_temp.append(r'$T_{\rm module}$')
                                #ax2.legend(leg_temp,loc='upper right')
                        
                        elif 'suntracker' in substat:
                            #Set labels and limits
                            ax2.set_ylabel(r'Temperature ($^{\circ}$C)',color='r')
                            ax2.set_ylim([-10,60])
                            
                            #Plot data
                            if iday in pv_systems[station]["raw_data"]['irrad'][substat][0]:
                                new_ix = list(pv_systems[station]["raw_data"]['irrad'][substat][0].strftime("%Y-%m-%d")).index(iday)
                                df = pv_systems[station]["raw_data"]['irrad'][substat][1][new_ix].filter(regex='^T_module',axis=1)
                                ax2.set_prop_cycle(linestyle=['-', '--',':','-.'])
                                ax2.plot(df.index,df,color='r')                            
                                ax2.tick_params('y', colors='r')
                                leg_temp.append([r'$T_{\rm module,upper}$',r'$T_{\rm module,lower}$'])
                                #ax2.legend(leg_temp,loc='upper right')
                        #else:                            
                    
                    for substat in pv_systems[station]["raw_data"]['irrad']:
                        #Set labels and limits                        
                        #Plot data
                        if iday in pv_systems[station]["raw_data"]['irrad'][substat][0] and 'Pyr' in substat:
                            new_ix = list(pv_systems[station]["raw_data"]['irrad'][substat][0].strftime("%Y-%m-%d")).index(iday)
                            df = pv_systems[station]["raw_data"]['irrad'][substat][1][new_ix]['T_ambient_pyr_C']
                            #ax2.set_prop_cycle(linestyle=['-', '--',':','-.'])
                            ax2.plot(df.index,df,color='r')
                            ax2.tick_params('y', colors='r')
                            leg_temp.append(r'$T_{\rm amb,' + substat + r'}$')
                    
                    ax2.legend(leg_temp,loc='upper right')                    
                    ax2.set_xlim([datemin, datemax])
                    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=3))
                    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                    fig.autofmt_xdate(rotation=0,ha='center') 
                    ax2.set_title("PV power and temperature from %s on %s" % (station,iday)) 

                    fig.tight_layout()
                    fig.savefig(os.path.join(dir_combo,'Power_Temp',station,'combo_plot_temp_power_' + 
                                             station + '_' + iday))
                    plt.close(fig)                
