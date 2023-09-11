#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 13:20:01 2023

@author: james
"""
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import ephem
from copy import deepcopy

from file_handling_functions import *
from plotting_functions import confidence_band
from data_process_functions import downsample
import pickle
import subprocess
from matplotlib.gridspec import GridSpec
from pvcal_forward_model import azi_shift
from scipy.stats import gaussian_kde
import seaborn as sns

def generate_folder_names_pvpyr2ghi(rt_config,pvcal_config):
    """
    Generate folder structure to retrieve GHI results from LUT
    
    args:    
    :param rt_config: dictionary with RT configuration    
    :param pvcal_config: dictionary with PV calibration configuration

    out:
    :return folder_label: string with complete folder path
    """        
    
    #geometry model
    atm_geom_config = rt_config["disort_base"]["pseudospherical"]
    if atm_geom_config == True:
        atm_geom_folder = "Pseudospherical"
    else:
        atm_geom_folder = "Plane-parallel"    
        
    disort_config = rt_config["clouds"]["disort_rad_res"]   
    theta_res = str(disort_config["theta"]).replace('.','-')
    phi_res = str(disort_config["phi"]).replace('.','-')
    
    disort_folder_label = 'Zen_' + theta_res + '_Azi_' + phi_res
        
    if rt_config["atmosphere"] == "default":
        atm_folder_label = "Atmos_Default"    
    elif rt_config["atmosphere"] == "cosmo":
        atm_folder_label = "Atmos_COSMO"        
        
    if rt_config["aerosol"]["source"] == "default":
        aero_folder_label = "Aerosol_Default"
    elif rt_config["aerosol"]["source"] == "aeronet":
        aero_folder_label = "Aeronet_" + rt_config["aerosol"]["station"]
            
    sza_label = "SZA_" + str(int(rt_config["sza_max"]["lut"]))

    model = pvcal_config["inversion"]["power_model"]
    eff_model = pvcal_config["eff_model"]        

    folder_label = os.path.join(atm_geom_folder,disort_folder_label,
                                atm_folder_label,aero_folder_label,sza_label,
                                model,eff_model)
        
    return folder_label

def plot_all_ghi_combined_scatter_new(dict_stats,list_stations,pvrad_config,T_models,folder,title_flag,window_avgs):
    """
    Plot combined scatter plots for all stations

    Parameters
    ----------
    dict_stats : dictionary with combined statistics for different averaging times
    list_stations : list of PV stations
    pvrad_config : dictionary with PV inversion configuration
    T_model : string, temperature model used
    folder : string, folder to save plots
    title_flag : boolean, whether to add title to plots
    window_avgs : list of averaging windows

    Returns
    -------
    None.
    """
    
    plt.ioff()
    plt.style.use('my_presi_grid')        
    
    res_dirs = list_dirs(folder)
    savepath = os.path.join(folder,'GHI_GTI_Plots')
    if 'GHI_GTI_Plots' not in res_dirs:
        os.mkdir(savepath)    
        
    # res_dirs = list_dirs(savepath)
    # savepath = os.path.join(savepath,'Scatter')
    # if 'Scatter' not in res_dirs:
    #     os.mkdir(savepath)
        
    years = ["mk_" + campaign.split('_')[1] for campaign in pvrad_config["calibration_source"]]    
    stations_label = '_'.join(["".join(s.split('_')) for s in list_stations])
    
    tres_list = pvrad_config["timeres_comparison"]  + window_avgs #
    data_types = [("Pyr","pyr","pyranometer"),("sat","sat","CAMS")]        
    T_labels = ["linear","non-linear"]
                        
    #Combined plot with all three time resolutions
    for data_type, data_type_short, data_label in data_types:
        for timeres in tres_list:
            #For pyranometer - use all three types of inversion
            #For CAMS, use only the OD-based type 
            if timeres == "1min" and data_type == "Pyr":
                inv_types = ["aod","cod","lut"]   
                width = -0.07
                height = 0.05
                leg_posy =  0.67
                font = 6
            else:
                inv_types = ["aod","cod"]
                if data_type == "Pyr":
                    width = 0.15
                else:
                    width = 0.21
                height = -0.38
                leg_posy = 0.65
                font = 7
            
            #1. Plot comparing inverted irradiance with that from pyranometers
            fig, axs = plt.subplots(len(inv_types),len(years)*len(T_models),sharex='all',sharey='all')
            #cbar_ax = fig.add_axes([.76, .25, .015, .45])                    
                                                
            print(f"Plotting combined scatter plot for {timeres}, {inv_types}, {data_label}...please wait....")           
            max_ghi = 0. #; min_z = 500.; max_z = 0.
                
            for i, ax in enumerate(axs.flatten()):            
                year = years[int((i - np.fmod(i,2))/2)%2]
                inv_type = inv_types[int((i - np.fmod(i,4))/4)]
                T_model = T_models[np.fmod(i,2)]
                T_label = T_labels[np.fmod(i,2)]
                                
                if timeres in window_avgs and "od" in inv_type:
                    if inv_type == "aod":
                        day_type = 0
                    elif inv_type == "cod":
                        day_type = 1
                    stacked_data = dict_stats[T_model][year][f"df_delta_all_{timeres}"].stack()\
                        .loc[:,["GHI_PV_od_inv",f"GHI_{data_type}_ref","day_type"]].stack()
                    ghi_data = stacked_data.loc[stacked_data["day_type"] == day_type].dropna()
                    
                    ghi_ref = ghi_data[f"GHI_{data_type}_ref"].values.flatten()
                    ghi_inv = ghi_data["GHI_PV_od_inv"].values.flatten()
                else:                    
                    ghi_data = dict_stats[T_model][year][f"df_delta_all_{timeres}"].stack()\
                        .loc[:,[f"GHI_PV_{inv_type}_inv",f"GHI_{data_type}_ref"]].stack().dropna(how='any')
                    
                    ghi_ref = ghi_data[f"GHI_{data_type}_ref"].values.flatten()
                    ghi_inv = ghi_data[f"GHI_PV_{inv_type}_inv"].values.flatten()
                    
                xy = np.vstack([ghi_ref,ghi_inv])
                z = gaussian_kde(xy)(xy)
                idx = z.argsort()
                
                max_ghi = np.max([max_ghi,ghi_ref.max(),ghi_inv.max()])
                max_ghi = np.ceil(max_ghi/100)*100    
                
                
                sc = ax.scatter(ghi_ref[idx], ghi_inv[idx], s=5, c=z[idx], 
                                cmap="plasma") #,norm=norm[int((i - np.fmod(i,2))/2)])
                
                ax.plot([0, 1], [0, 1], ls = '--', transform=ax.transAxes,c='k')
                ax.annotate(f"{T_label}",fontsize=10,
                            xy=(1.,0.01),xycoords='axes fraction',
                            horizontalalignment='right')
                                
                print(f"Using {dict_stats[T_model][year][timeres][f'n_delta_{inv_type}_{data_type_short}']} data points for {timeres}, {year}, {inv_type}, {T_model} plot")
                ax.annotate(rf"rMBE = {dict_stats[T_model][year][timeres][f'rMBE_GHI_{inv_type}_{data_type_short}_%']:.1f} %" "\n" \
                            rf"rRMSE = {dict_stats[T_model][year][timeres][f'rRMSE_GHI_{inv_type}_{data_type_short}_%']:.1f} %" "\n"\
                            r"$\langle G_\mathrm{ref} \rangle$ =" \
                                    rf" {dict_stats[T_model][year][timeres][f'mean_GHI_{inv_type}_{data_type_short}_Wm2']:.2f} W m$^{{-2}}$" "\n"\
                            rf"n = ${dict_stats[T_model][year][timeres][f'n_delta_{inv_type}_{data_type_short}']:.0f}$",
                          xy=(0.05,leg_posy),xycoords='axes fraction',fontsize=font,color='k',
                          bbox = dict(facecolor='lightgrey',alpha=0.5),
                          horizontalalignment='left',multialignment='left')
                                
                ax.set_xlim([0.,max_ghi])
                ax.set_ylim([0.,max_ghi])
                ax.set_aspect('equal')
                ax.grid(False)
                # if max_ghi > 1400:                
                
                # else:
                #     ax.set_xticks([0,400,800,1200,1400])
                if len(inv_types) == 3:
                    if i == 9:
                        ax.set_xlabel(rf"$G_\mathrm{{tot,{data_label}}}^{{\downarrow}}$ (W m$^{{-2}}$)",position=(1.1,0))
                    if i == 4:
                        ax.set_ylabel(r"$G_\mathrm{{tot,PV,inv}}^{{\downarrow}}$ (W m$^{-2}$)")
                    # if i < 8:
                    #     ax.get_xaxis().set_visible(False)
                    # if np.fmod(i,4) != 0:
                    #     ax.get_yaxis().set_visible(False)
                    if i == 3:
                        ax.annotate("clear", xy=(1.05, 0.4),xycoords='axes fraction',
                                    fontsize=14, annotation_clip=False,rotation=90)
                    if i == 7:
                        ax.annotate("cloudy", xy=(1.05, 0.34),xycoords='axes fraction',
                                    fontsize=14, annotation_clip=False,rotation=90)
                    if i == 11:
                        ax.annotate("broken clouds", xy=(1.05, 0.1),xycoords='axes fraction',
                                    fontsize=14, annotation_clip=False,rotation=90)
                else:
                    if i == 5:
                        ax.set_xlabel(rf"$G_\mathrm{{tot,{data_label}}}^{{\downarrow}}$ (W m$^{{-2}}$)",position=(1.1,0))
                    if i == 4:
                        ax.set_ylabel(r"$G_\mathrm{{tot,PV,inv}}^{{\downarrow}}$ (W m$^{-2}$)",position=(0,1.1))
                        
                    if i == 3:
                        ax.annotate("clear", xy=(1.05, 0.38),xycoords='axes fraction',
                                    fontsize=14, annotation_clip=False,rotation=90)
                    if i == 7:
                        ax.annotate("cloudy", xy=(1.05, 0.34),xycoords='axes fraction',
                                    fontsize=14, annotation_clip=False,rotation=90)
                    
                ax.set_xticks([0,500,1000])            
                ax.set_yticks([0,500,1000])
                
                if i == 1 or i == 3:
                    ax.set_title(f"{year.split('_')[1]}",x=-0.08,y=0.98,fontsize=14)
                
            fig.subplots_adjust(wspace=width,hspace=height)
            
            cb = fig.colorbar(sc,ticks=[np.min(z),np.max(z)], ax=axs, shrink=0.6, location = 'top', 
                                aspect=20) 
            #cb.set_ticks()
            #cb.set_ticklabels([f"{val:.2f}" for val in cb.get_ticks()*1e5])
            cb.set_ticklabels(["Low", "High"])  # horizontal colorbar
            cb.ax.tick_params(labelsize=14) 
            # cb.set_label("PDF", labelpad=-18, 
            #               fontsize=14)
            cb.set_label("PDF", labelpad=-10, fontsize=16)
            
            plt.savefig(os.path.join(savepath,f"ghi_scatter_hist_combo_all_{timeres}_{data_label}_modelcombo"\
                      f"_{stations_label}.png"),bbox_inches = 'tight')  
                
            plt.close(fig)            
            
            
    for timeres in window_avgs:
        inv_types = ["aod","cod"]
                    
        print(f"Plotting combined frequency scatter plot for {inv_types}, {timeres}, COSMO...please wait....")
        #3. Plot comparing inverted irradiance with that from cosmo
        fig, axs = plt.subplots(len(inv_types),len(years)*len(T_models),sharex='all',sharey='all')
        #cbar_ax = fig.add_axes([.27, .75, .43,.015]) 
                
        # plot_data = []
        max_ghi = 0. #; min_z = 500.; max_z = 0.
        for i, ax in enumerate(axs.flatten()):            
            year = years[int((i - np.fmod(i,2))/2)%2]
            inv_type = inv_types[int((i - np.fmod(i,4))/4)]
            T_model = T_models[np.fmod(i,2)]
            T_label = T_labels[np.fmod(i,2)]
            
            if "od" in inv_type:
                if inv_type == "aod":
                    day_type = 0
                elif inv_type == "cod":
                    day_type = 1
                stacked_data = dict_stats[T_model][year][f"df_delta_all_{timeres}"].stack()\
                    .loc[:,["GHI_PV_od_inv","GHI_cosmo_ref","day_type"]].stack()
                ghi_data = stacked_data.loc[stacked_data["day_type"] == day_type].dropna()
                
                ghi_ref = ghi_data["GHI_cosmo_ref"].values.flatten()
                ghi_inv = ghi_data["GHI_PV_od_inv"].values.flatten()
            else:                    
                ghi_data = dict_stats[year][f"df_delta_all_{timeres}"].stack()\
                    .loc[:,[f"GHI_PV_{inv_type}_inv","GHI_cosmo_ref"]].stack().dropna(how='any')
                
                ghi_ref = ghi_data["GHI_cosmo_ref"].values.flatten()
                ghi_inv = ghi_data[f"GHI_PV_{inv_type}_inv"].values.flatten()
            xy = np.vstack([ghi_ref,ghi_inv])
            z = gaussian_kde(xy)(xy)
            idx = z.argsort()
            
            max_ghi = np.max([max_ghi,ghi_ref.max(),ghi_inv.max()])
            max_ghi = np.ceil(max_ghi/100)*100    

            
            sc = ax.scatter(ghi_ref[idx], ghi_inv[idx],  s=8, c=z[idx],
                            cmap="plasma")#,norm=norm)
            
            ax.plot([0, 1], [0, 1], ls = '--', transform=ax.transAxes,c='k')
            ax.annotate(f"{T_label}",fontsize=10,
                            xy=(1.,0.01),xycoords='axes fraction',
                            horizontalalignment='right')
            
            print(f"Using {dict_stats[T_model][year][timeres][f'n_delta_{inv_type}_cosmo']} data points for {timeres}, {year}, {inv_type}, {T_model} plot")
            ax.annotate(rf"rMBE = {dict_stats[T_model][year][timeres][f'rMBE_GHI_{inv_type}_cosmo_%']:.1f} %" "\n" \
                        rf"rRMSE = {dict_stats[T_model][year][timeres][f'rRMSE_GHI_{inv_type}_cosmo_%']:.1f} %" "\n"\
                                r"$\langle G_\mathrm{ref} \rangle$ ="\
                            rf" {dict_stats[T_model][year][timeres][f'mean_GHI_{inv_type}_cosmo_Wm2']:.2f} W m$^{{-2}}$" "\n"\
                        rf"n = ${dict_stats[T_model][year][timeres][f'n_delta_{inv_type}_cosmo']:.0f}$",
                      xy=(0.05,leg_posy),xycoords='axes fraction',fontsize=7,color='k',
                      bbox = dict(facecolor='lightgrey', alpha=0.5),
                      horizontalalignment='left',multialignment='left')     

            ax.set_xlim([0.,max_ghi])
            ax.set_ylim([0.,max_ghi])
            ax.set_aspect('equal')
            ax.grid(False)
            # if max_gti < 1400:
            #     ax.set_xticks([0,400,800,1200])
            # else:
            #     ax.set_xticks([0,400,800,1200,1400])
            if i == 5:
                ax.set_xlabel(r"$G_\mathrm{tot,COSMO}^{\downarrow}$ (W m$^{-2}$)",position=(1.1,0))                    
            if i == 4:
                ax.set_ylabel(r"$G_\mathrm{{tot,PV,inv}}^{{\downarrow}}$ (W m$^{-2}$)",position=(0,1.1))
        
            if i == 1 or i == 3:
                    ax.set_title(f"{year.split('_')[1]}",x=-0.08,y=0.98,fontsize=14)
                    
            if i == 3:
                 ax.annotate("clear", xy=(1.05, 0.38),xycoords='axes fraction',
                                fontsize=14, annotation_clip=False,rotation=90)
            if i == 7:
                ax.annotate("cloudy", xy=(1.05, 0.34),xycoords='axes fraction',
                            fontsize=14, annotation_clip=False,rotation=90)
        
        fig.subplots_adjust(wspace=width,hspace=height) 
            
        cb = fig.colorbar(sc,ticks=[np.min(z),np.max(z)], ax=axs, shrink=0.6, location = 'top', 
                                aspect=20) 
           
        cb.set_ticklabels(["Low", "High"])  # horizontal colorbar
        cb.ax.tick_params(labelsize=14) 
        cb.set_label("PDF", labelpad=-10, fontsize=16)
        #fig.subplots_adjust(wspace=0.1)    
        # fig.add_subplot(111, frameon=False)
        # plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        # plt.grid(False)
        
        plt.savefig(os.path.join(savepath,f"ghi_scatter_hist_combo_all_COSMO_{timeres}_modelcombo"\
                  f"_{stations_label}.png"),bbox_inches = 'tight')  
        plt.close(fig)


#%%Main Program
#######################################################################
###                         MAIN PROGRAM                            ###
#######################################################################
#This program takes makes plots of GHI, DNI and GHI for comparison with measurements
#All data are averaged to 60 minutes, in order for comparison with COSMO
#In addition, different moving averages (1min, 15min) are calculated, dependent on the measurements available
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

#Load radiative transfer configuration
rt_config = load_yaml_configfile(config["rt_configfile"])

#Load PVCAL configuration
pvcal_config = load_yaml_configfile(config["pvcal_configfile"])

#Load pv2rad configuration
pvrad_config = load_yaml_configfile(config["pvrad_configfile"])

homepath = os.path.expanduser('~') # #"/media/luke" #
    
if args.campaign:
    campaigns = args.campaign
    if type(campaigns) != list:
        campaigns = [campaigns]
else:
    campaigns = config["description"]  
    
plot_styles = config["plot_styles"]
plot_flags = config["plot_flags"]
sza_limit_cod = rt_config["sza_max"]["cod_cutoff"]
sza_limit_aod = rt_config["sza_max"]["lut"]
window_avgs = config["window_size_moving_average"]
#T_model = pvcal_config["T_model"]
                              
#%% Run through campaigns, load OD results and do a DISORT simuation
od_types = ["aod","cod"]
str_window_avg = config["window_size_moving_average"]
num_var_classes = 3

if args.station:
    station_list = args.station
    if station_list[0] == 'all':
        station_list = 'all'
else:
    #Stations for which to perform inversion
    station_list = "all" #["PV_12","PV_15"] #pyr_config["stations"]

if type(station_list) != list:
    station_list = [station_list]
if station_list[0] == "all":
    station_list = list(pvrad_config["pv_stations"].keys())

stations_label = '_'.join(["".join(s.split('_')) for s in station_list])

homepath = os.path.expanduser('~')

pvsys_stats = {}
folder_label = generate_folder_names_pvpyr2ghi(rt_config,pvcal_config)                

results_path = os.path.join(homepath,pvrad_config["results_path"]["main"],
                                pvrad_config["results_path"]["irradiance"]) 


T_models = ["Tamizhmani","Faiman"]

for T_model in T_models:        
    filename = f"ghi_combo_results_stats_{T_model}_{stations_label}.data"
    with open(os.path.join(results_path,folder_label,T_model,filename), 'rb') as filehandle:  
                        # read the data as binary data stream
        dict_stats, dummy, dummy, dummy, dummy = pd.read_pickle(filehandle)   

    pvsys_stats.update({T_model:dict_stats})

if plot_flags["combo_stats"]:    
    plot_all_ghi_combined_scatter_new(pvsys_stats,station_list,pvrad_config,T_models,
                                  results_path,plot_flags["titles"],window_avgs)

