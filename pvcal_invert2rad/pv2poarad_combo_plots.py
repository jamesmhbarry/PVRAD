#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 16:21:50 2023

@author: james
"""

#%% Preamble
import os
import numpy as np
from file_handling_functions import *
from rt_functions import *
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.dates as mdates
#from matplotlib.gridspec import GridSpec
from plotting_functions import confidence_band
from scipy.stats import gaussian_kde
import pandas as pd
#from pvcal_forward_model import azi_shift
from data_process_functions import downsample
from scipy import optimize
import datetime
import seaborn as sns
import pickle

#%%Functions
def generate_folder_names_pv2poarad(rt_config,pvcal_config):
    """
    Generate folder structure to retrieve POA results
    
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
        
    #Get wavelength folder label
    wvl_config = rt_config["common_base"]["wavelength"]["pv"]
    if type(wvl_config) == list:
        wvl_folder_label = "Wvl_" + str(int(wvl_config[0])) + "_" + str(int(wvl_config[1]))
    elif wvl_config == "all":
        wvl_folder_label = "Wvl_Full"
    
    disort_config = rt_config["disort_rad_res"]   
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

    folder_label = os.path.join(atm_geom_folder,wvl_folder_label,disort_folder_label,
                                atm_folder_label,aero_folder_label,sza_label,
                                model,eff_model)
        
    return folder_label

def plot_all_combo_scatter(dict_stats,list_stations,pvrad_config,T_models,folder):
    """
    

    Parameters
    ----------
    dict_stats : dictionary with statistics of deviations
    list_stations : list of PV stations
    pvrad_config : dictionary with inversion configuration        
    T_model : string with temperature model
    folder : string with folder for saving results and plots    

    Returns
    -------
    None.

    """
    
    
    plt.ioff()
    plt.style.use('my_presi_grid')        
    
    res_dirs = list_dirs(folder)
    savepath = os.path.join(folder,'GTI_Plots')
    if 'GTI_Plots' not in res_dirs:
        os.mkdir(savepath)    
        
    res_dirs = list_dirs(savepath)
    savepath = os.path.join(savepath,'Scatter')
    if 'Scatter' not in res_dirs:
        os.mkdir(savepath)
        
    years = ["mk_" + campaign.split('_')[1] for campaign in pvrad_config["calibration_source"]]    
    stations_label = '_'.join(["".join(s.split('_')) for s in list_stations])
    T_labels = ["linear","non-linear"]
    
    for timeres in pvrad_config["timeres_comparison"]:
    
        fig, axs = plt.subplots(len(T_models),len(years),sharex='all',sharey='all')
        #cbar_ax = fig.add_axes([.27, .75, .43,.015]) 
        
        print(f"Plotting combined frequency scatter plot for {timeres}...please wait....")
        #plot_data = []
        max_gti = 0. #; min_z = 500.; max_z = 0.
        for i, ax in enumerate(axs.flatten()):            
            year = years[np.fmod(i,2)]
            T_model = T_models[int((i - np.fmod(i,2))/2)%2]
            T_label = T_labels[int((i - np.fmod(i,2))/2)%2]
            
            gti_data = dict_stats[T_model][year][f"df_delta_all_{timeres}"].stack()\
                .loc[:,["GTI_PV_inv","GTI_Pyr_ref"]].stack().dropna(how='any')
            
            gti_ref = gti_data["GTI_Pyr_ref"].values.flatten()
            gti_inv = gti_data["GTI_PV_inv"].values.flatten()
            xy = np.vstack([gti_ref,gti_inv])
            z = gaussian_kde(xy)(xy)
            idx = z.argsort()                        
            
            max_gti = np.max([max_gti,gti_ref.max(),gti_inv.max()])
            max_gti = np.ceil(max_gti/100)*100    
            
            sc = ax.scatter(gti_ref[idx], gti_inv[idx], s=5, c=z[idx], 
                            cmap="plasma")
            
            ax.plot([0, 1], [0, 1], ls = '--', transform=ax.transAxes,c='k')
            ax.annotate(f"{T_label}",fontsize=10,
                            xy=(1.,0.01),xycoords='axes fraction',
                            horizontalalignment='right')
            
            print(f"Using {dict_stats[T_model][year][timeres]['n_delta']} data points for {year}, {T_model} plot")
            ax.annotate(rf"rMBE = {dict_stats[T_model][year][timeres]['rMBE_GTI_%']:.1f} %" "\n" \
                        rf"rRMSE = {dict_stats[T_model][year][timeres]['rRMSE_GTI_%']:.1f} %" "\n"\
                            r"$\langle G_\mathrm{ref} \rangle$ =" \
                            rf" {dict_stats[T_model][year][timeres][f'mean_GTI_Wm2']:.2f} W m$^{{-2}}$" "\n"\
                        rf"n = ${dict_stats[T_model][year][timeres]['n_delta']:.0f}$",
                     xy=(0.05,0.75),xycoords='axes fraction',fontsize=8,bbox = dict(facecolor='lightgrey',edgecolor='k', alpha=0.5),
                     horizontalalignment='left',multialignment='left')     
            # ax.annotate(rf"RMSE = {dict_stats[year][timeres]['RMSE_GTI_Wm2']:.2f} W/m$^2$",
            #          xy=(0.05,0.85),xycoords='axes fraction',fontsize=10,bbox = dict(facecolor='lightgrey', alpha=0.3))  
            # ax.annotate(rf"n = {dict_stats[year][timeres]['n_delta']:.0f}",
            #          xy=(0.05,0.78),xycoords='axes fraction',fontsize=10,bbox = dict(facecolor='lightgrey', alpha=0.3))                  
            #ax.set_xticks([0,400,800,1200])
                
            ax.set_xlim([0.,max_gti])
            ax.set_ylim([0.,max_gti])            
            ax.set_aspect('equal')
            ax.grid(color='gray',linestyle=':')
            # if max_gti < 1400:
            #     ax.set_xticks([0,400,800,1200])
            # else:
            #     ax.set_xticks([0,400,800,1200,1400])
            if i == 2:                
                ax.set_ylabel(r"$G_\mathrm{{tot,inv}}^{{\angle}}$ (W m$^{-2}$)",position=(0,1.1))            
                ax.set_xlabel(r"$G_\mathrm{tot,pyranometer}^{\angle}$ (W m$^{-2}$)",position=(1.1,0))
                
            if i == 0 or i == 1:
                ax.set_title(f"{year.split('_')[1]}",y=0.98,fontsize=14)
        
        #print(max_gti)
        fig.subplots_adjust(wspace=-0.38,hspace=0.15)    
        cb = fig.colorbar(sc,ticks=[np.min(z),np.max(z)],ax=axs, shrink=0.6, location = 'top', 
                            aspect=20)    
        cb.set_ticklabels(["Low", "High"])  # horizontal colorbar
        cb.set_label("PDF",labelpad=-10, fontsize=16)         
        
        # fig.add_subplot(111, frameon=False)
        # plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        # plt.grid(False)
        
        plt.savefig(os.path.join(savepath,f"gti_scatter_hist_combo_all_{timeres}_modelcombo_{stations_label}.png"),bbox_inches = 'tight')  

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
   
config_filename = "config_PV2RAD_MetPVNet_messkampagne.yaml" #os.path.abspath(args.configfile)
 
config = load_yaml_configfile(config_filename)

#Load PV configuration
pvcal_config = load_yaml_configfile(config["pvcal_configfile"])

#Load PV configuration
pvrad_config = load_yaml_configfile(config["pvrad_configfile"])

#Load radiative transfer configuration
rt_config = load_yaml_configfile(config["rt_configfile"])

homepath = os.path.expanduser('~')

if args.station:
    station_list = args.station
    if station_list[0] == 'all':
        station_list = 'all'
else:
    #Stations for which to perform inversion
    station_list = ["PV_11","PV_12"] #pyr_config["stations"]
    
if type(station_list) != list:
    station_list = [station_list]
if station_list[0] == "all":
    station_list = list(pvrad_config["pv_stations"].keys())
    
stations_label = '_'.join(["".join(s.split('_')) for s in station_list])

if args.campaign:
    campaigns = args.campaign
    if type(campaigns) != list:
        campaigns = [campaigns]
else:
    campaigns = config["description"]  
    
pvsys_stats = {}
folder_label = generate_folder_names_pv2poarad(rt_config,pvcal_config)                

results_path = os.path.join(homepath,pvrad_config["results_path"]["main"],
                                pvrad_config["results_path"]["inversion"]) 

T_models = ["Tamizhmani","Faiman"]

for T_model in T_models:        
    filename = f"gti_combo_results_stats_{T_model}_{stations_label}.data"
    with open(os.path.join(results_path,folder_label,T_model,filename), 'rb') as filehandle:  
                        # read the data as binary data stream
        dict_stats, dummy, dummy, dummy = pd.read_pickle(filehandle)   

    pvsys_stats.update({T_model:dict_stats})
    
plot_all_combo_scatter(pvsys_stats,station_list,pvrad_config,T_models,results_path)