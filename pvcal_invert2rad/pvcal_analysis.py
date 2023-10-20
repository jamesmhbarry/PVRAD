#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 14:36:46 2018

@author: james


"""

#%% Preamble
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from file_handling_functions import *
from rt_functions import *
from pvcal_forward_model import azi_shift

#%%Functions
def load_inversion_results(rt_config,pv_config,station_list,home):
    """
    Load results from Bayesion Inversion
    
    args:    
    :param rt_config: dictionary with current RT configuration
    :param pv_config: dictionary with current calibration configuration    
    :param station_list: list of stations
    :param home: string with homepath
    
    out:
    :return pv_systems: dictionary of PV systems with data
    :return rt_config: dictionary with current RT configuration
    :return pv_config: dictionary with current calibration configuration    
    :return folder_label: string with folder for results
    """
    
    mainpath = os.path.join(home,pv_config['results_path']['main'])
    
    #atmosphere model
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

    model = pv_config["inversion"]["power_model"]
    eff_model = pv_config["eff_model"]
    T_model = pv_config["T_model"]["model"]
    sza_label = "SZA_" + str(int(pv_config["sza_max"]["disort"]))
    
    folder_label = os.path.join(mainpath,atm_geom_folder,wvl_folder_label,
                                disort_folder_label,atm_folder_label,
                                aero_folder_label,model,eff_model,T_model,sza_label)
    
    if len(pv_config["calibration_source"]) > 1:
        infos = '_'.join(pv_config["calibration_source"])
    else:
        infos = pv_config["calibration_source"][0]
    
    filename = filename + infos + '_disortres_' + theta_res + '_' + phi_res + '_'
    
    pv_systems = {}    
        
    #Choose which stations to load                
    if type(station_list) != list:
        station_list = [station_list]
        if station_list[0] == "all":
            station_list = list(pv_config["pv_stations"].keys())
    
    for station in station_list:                
        #Read in binary file that was saved from pvcal_radsim_disort
        filename_stat = filename + station + '.data'
        #print(station)
        try:
            with open(os.path.join(folder_label,filename_stat), 'rb') as filehandle:  
                # read the data as binary data stream
                (pvstat, rt_config, pv_config) = pd.read_pickle(filehandle)            
            pv_systems.update({station:pvstat})
            print('Data for %s loaded from %s, %s' % (station,folder_label,filename))
        except IOError:
            print('There is no simulation for %s' % station)   
            
    return pv_systems, rt_config, pv_config, folder_label

def plot_pnorm_irrad(station,substats,folder):
    """
    Create plots of normalised power vs. irradiance
    
    args:
    :param station: string, name of station
    :param substats: dictionary with substations
    :param folder: string with folder to save plots
    """
    
    plt.ioff()
    #plt.close('all')
    plt.style.use("my_paper")
    
    res_dirs = list_dirs(folder)
    savepath = os.path.join(folder,'Analysis_Plots')
    if 'Analysis_Plots' not in res_dirs:
        os.mkdir(savepath)        
        
    stat_dirs = list_dirs(savepath)
    savepath = os.path.join(savepath,station)
    if station not in stat_dirs:
        os.mkdir(savepath)
    
    plottypes = ["measured","modelled"]
    for substat in substats:
        for plottype in plottypes:
            opt_pars = substats[substat]["opt_pars"]
            dataframe = substats[substat]["df_cal"]
            fig, ax = plt.subplots(figsize=(9,8))
            
            if plottype == "measured":
                pnorm = dataframe[('P_meas_W',substat)]/\
                                     dataframe[('Etotpoa_pv',substat)]/opt_pars[3][1]
                pnorm_label = r'$\frac{P_{\rm AC,meas}}{P_{\rm AC,STC}} \cdot \frac{G_{\rm STC}}{G_{\rm tot,[0.3,1.2]\mu m}^{\angle}}$'
            elif plottype == "modelled":
                pnorm = dataframe[('P_MAP',substat)]/\
                                     dataframe[('Etotpoa_pv',substat)]/opt_pars[3][1]
                pnorm_label = r'$\frac{P_{\rm AC,meas}}{P_{\rm AC,STC}} \cdot \frac{G_{\rm STC}}{G_{\rm tot,[0.3,1.2]\mu m}^{\angle}}$'
            
            if "T_module_meas_C" in dataframe.columns:
                temp_module = dataframe[('T_module_meas_C','PVTemp_1')]
                t_label = r'$T_{\rm module,meas}$ ($^\circ$C)'
            else:
                temp_module = dataframe[('T_module_inv_C',substat)]
                t_label = r'$T_{\rm module,mod}$ ($^\circ$C)'
            
            sc =ax.scatter(dataframe[('Etotpoa_pv',substat)],pnorm,
                                     c=temp_module,cmap='jet')
        
            ax.set_xlabel(r'$G_{\rm tot,[0.3,1.2]\mu m}^{\angle}$ (W/m$^2$)')
            ax.set_ylabel(pnorm_label)
            ax.set_title('Normalised ' + plottype + ' power vs. irradiance for ' + station)
            cb = plt.colorbar(sc)
            
            cb.set_label(r'Temperature ($^\circ$C)')
            cb.set_label(t_label)
            
            plt.savefig(os.path.join(savepath,'pnorm_' + plottype + '_irrad_'
                                     + station + '_' + substat))

def plot_ave_kernel(station,substats,pv_config,folder):
    """
    Plots the averaging kernel matrix in order to evaluate inversion results
    
    args:
    :param station: string, name of station (key)    
    :param substats: dictionary with substations and optimisation results 
    :param pv_config: dictionary with current calibration configuration    
    :param folder: string with folder to save plots


    """
    
    T_model = pv_config["T_model"]
    eff_model = pv_config["eff_model"]
    
    plt.ioff()
    #plt.close('all')
    plt.style.use("my_paper")

    res_dirs = list_dirs(folder)
    savepath = os.path.join(folder,'Analysis_Plots')
    if 'Analysis_Plots' not in res_dirs:
        os.mkdir(savepath)        
        
    stat_dirs = list_dirs(savepath)
    savepath = os.path.join(savepath,station)
    if station not in stat_dirs:
        os.mkdir(savepath)
        
    for substat in substats:     
        ticklabels = np.array([r'$\theta$',r'$\phi$',r'$n$',r'$s$',r'$\zeta$'])
        if eff_model == "Ransome":
            ticklabels = np.hstack([ticklabels,r'$c_3$',r'$c_6$'])
        if T_model == "Tamizhmani":
            ticklabels = np.hstack([ticklabels,r'$u_0$',r'$u_1$',r'$u_2$',r'$u_3$'])
        elif T_model == "Faiman":
            ticklabels = np.hstack([ticklabels,r'$u_1$',r'$u_2$',r'$u_3$'])                    
        
        #Delete ticklabels for those parameters that were fixed
        index_delete = []
        for i, par in enumerate(substats[substat]["opt_pars"]):
            if par[2] == 0:
                index_delete.append(i)
                
        ticklabels = np.delete(ticklabels,index_delete)
        
        opt_dict = substats[substat]["opt"]
        
        ave_kernel = opt_dict["A"]
        fig, ax = plt.subplots(figsize=(9,8))
        ax.semilogy(np.diag(ave_kernel),marker='o')      
        #ax.set_ylim([0,1])
        ax.set_xlim([0,len(ave_kernel) - 1])
        ax.set_xticks(range(len(ave_kernel)))        
        
        ax.xaxis.set_ticklabels(ticklabels)
    
        ax.set_title('Diagonal terms of averaging kernel for ' + station + ', ' + substat)
        
        plt.annotate(r'$d_s = tr(A)$ = ' + str(np.round((opt_dict['d_s']),2)),
                     xy=(0.2,0.2),xycoords='figure fraction',fontsize=14)    
        
        plt.savefig(os.path.join(savepath,"ave_kernel_" + station + "_" + substat + ".png"))
    
    
    
def plot_temp_eff(key,pv_station,pv_config,folder):
    """
    Plots the temperature-dependent efficiency
    
    args:
    :param key: string, name of PV station
    :param pv_station: dictionary with information and data on PV station    
    :param pv_config: dictionary with current calibration configuration    
    :param folder: string with folder to save plots
    """

    plt.ioff()
    plt.style.use('my_paper')
    
    res_dirs = list_dirs(folder)
    savepath = os.path.join(folder,'Analysis_Plots')
    if 'Analysis_Plots' not in res_dirs:
        os.mkdir(savepath)        
        
    stat_dirs = list_dirs(savepath)
    savepath = os.path.join(savepath,key)
    if key not in stat_dirs:
        os.mkdir(savepath)
    
    data_source = pv_config["pv_stations"][key]["input_data"]   

    df = pv_station["df_cal"]     
    
    for substat in pv_station["substations"]:
        df_substat = pv_station["substations"][substat]["df_cal"] 
        for iday in pv_station["substations"][substat]["cal_days"]:
            df_substat_day = df_substat.loc[iday]
            df_day = df.loc[iday]
            #df_test_day.dropna(axis='columns',inplace=True)
            if "2018" in iday:
                amb_temp_source = data_source["irrad"]["mk_2018"]
                mod_temp_source = data_source["temp_module"]["mk_2018"]
            elif "2019" in iday:
                amb_temp_source = data_source["irrad"]["mk_2019"]
                mod_temp_source = data_source["temp_module"]["mk_2019"]
            
            fig1, ax1 = plt.subplots(figsize=(9,8))
            
            df_substat_day[("P_meas_W",substat)].plot(ax=ax1,color='r',style = '-',legend=True)
            df_substat_day[("P_MAP",substat)].plot(color='r',style = '--',legend=True)
            
            ax1.tick_params('y', colors='r')        
            ax1.set_xlabel('Time (UTC)')
            ax1.set_ylabel('Power (W)', color='r')
            
            plt.legend((r'$P_{\rm AC,meas}$',r'$P_{\rm AC,mod}$'),loc=2)
            plt.title('AC power and temperature for ' + key + ' on ' + iday)
            plt.grid(False)   
            
            ax2 = ax1.twinx()        
            
            legendstring = []
            if amb_temp_source != "None":
                df_day[("T_ambient_pyr_C",amb_temp_source)].plot(ax=ax2,color='b',style = '.',legend=True)
                legendstring.append(r'$T_{\rm ambient,meas}$')
            df_day[("T_ambient_2M_C","cosmo")].plot(ax=ax2,color='b',style = '-',legend=True)
            df_substat_day[("T_module_inv_C",substat)].plot(ax=ax2,color='b',style = '*',legend=True)
            legendstring.extend([r'$T_{\rm ambient,COSMO}$',
                        r'$T_{\rm module,sim}$'])
            
            if mod_temp_source != "None":
                df_day[("T_module_meas_C",mod_temp_source)].plot(ax=ax2,color='b',style = '-.',legend=True)
                legendstring.append(r'$T_{\rm module,meas}$')
            
            ax2.tick_params('y', colors='b')
            ax2.set_ylabel(r'Temperature ($^{\circ}$C)', color='b')
            plt.legend(legendstring,loc=1)
            plt.grid(False)   
            
            fig1.tight_layout()
            plt.savefig(os.path.join(savepath,'irrad_fit_temp_' + key + '_' + iday + '.png'))
            plt.close(fig1)        
            
            fig2, ax1 = plt.subplots(figsize=(9,8))
            
            df_substat_day[("P_meas_W",substat)].plot(ax=ax1,color='r',style = '-',legend=True)
            df_substat_day[("P_MAP",substat)].plot(color='r',style = '--',legend=True)
            
            ax1.tick_params('y', colors='r')        
            ax1.set_xlabel('Time (UTC)')
            ax1.set_ylabel('Power (W)', color='r')
            
            plt.legend((r'$P_{\rm AC,meas}$',r'$P_{\rm AC,mod}$'),loc=2)
            plt.title('AC power and module efficiency for ' + key + ' on ' + iday)
            plt.grid(False)   
            
            ax2 = ax1.twinx()        
            
            df_substat_day[("eff_temp_inv",substat)].plot(ax=ax2,color='b',style = '-',legend=False)
            ax2.set_ylabel(r'$\eta_{\rm module}$',color='b')
            
            ax2.tick_params('y', colors='b')
            plt.grid(False)   
            
            fig2.tight_layout()
            plt.savefig(os.path.join(savepath,'irrad_fit_eff_modul_' + key
                                     + '_' + iday + '.png'))
            plt.close(fig2)   


#%%Main Program 
#######################################################################
###                         MAIN PROGRAM                            ###
#######################################################################
#def main():
import argparse

parser = argparse.ArgumentParser()
#parser.add_argument("configfile", help="yaml file containing config")
parser.add_argument("-s","--station",nargs='+',help="station for which to perform inversion")
parser.add_argument("-c","--campaign",nargs='+',help="measurement campaign for which to perform simulation")
args = parser.parse_args()
    
config_filename = "config_PVCAL_MetPVNet_messkampagne.yaml" #os.path.abspath(args.configfile)
 
config = load_yaml_configfile(config_filename)

#Load data configuration
#data_config = load_yaml_configfile(config["data_configfile"])

#Load PV configuration
pv_config = load_yaml_configfile(config["pv_configfile"])

#Load radiative transfer configuration
rt_config = load_yaml_configfile(config["rt_configfile"])

homepath = os.path.expanduser('~') # #"/media/luke" #

if args.station:
    stations = args.station
    if stations[0] == 'all':
        stations = 'all'
else:
    #Stations for which to perform inversion
    stations = "all" #["MS_02","PV_01"] #"all" #pv_config["pv_stations"]
  
theta_res = str(rt_config["disort_rad_res"]["theta"]).replace('.','-')
phi_res = str(rt_config["disort_rad_res"]["phi"]).replace('.','-')
res_string_label = '_' + theta_res + '_' + phi_res + '_'


#%%Load ground truth angles
file_angles = "/mnt/bigdata/share/00_Projects/1_MetPVNet/Messkampagne/01_Dokumentation/PV_Calibration/Old_Results/20190626_Anlagen_Winkel_Vergleich_JB.xlsx"
# #file_angles = os.path.join(homepath,"MetPVNet/Data/Messkampagne/20190626_Anlagen_Winkel_Vergleich_JB.xlsx")

real_angles = pd.read_excel(file_angles,sheet_name="Finale Winkel",index_col=0,usecols=[0,1,2])        
real_angles.rename(columns={real_angles.columns[0]:'Theta_act',real_angles.columns[1]:'Phi_act'},inplace=True)

print('Analysis of inversion results for %s' % pv_config["stations"])
print('Loading results from Bayesian inversion')
print('Model: %s, efficiency model: %s, temperature model: %s' %(pv_config["inversion"]["power_model"],
                                                                 pv_config["eff_model"],
                                                                 pv_config["T_model"]))

#%%Load inversion results
pvsys, dummy, dummy, results_folder = load_inversion_results(rt_config,pv_config,stations,homepath)

#%%Plots for analysis
#for key in pvsys:
    #Plot the normalised performance ratio
    # print('Plotting normalised performance ratio')
    # plot_pnorm_irrad(key,pvsys[key]['substations'],results_folder)
    
    # #Plot averaging kernel matrix
    # print('Plotting diagonal elements of averaging kernel matrix')
    # plot_ave_kernel(key,pvsys[key]['substations'],pv_config,results_folder)
      
    # #Plot temperature comparison
    # print('Plotting measured vs. modelled temperature')
    # plot_temp_eff(key,pvsys[key],pv_config,results_folder)
    
#%%Generate results table for scatter plots
opt_results_sza_80 = pd.DataFrame() #theta_results = {}
#theta_tropos = {}

for key in pvsys:
    for substat in pvsys[key]["substations"]:
        index = key + ' ' + substat
        if "opt_pars" in pvsys[key]["substations"][substat]:
            opt_pars = pvsys[key]["substations"][substat]["opt_pars"]
            
            #if not np.isnan(opt_pars['x_min']).any():
            opt_results_sza_80.loc[index,'theta_opt'] = np.rad2deg(opt_pars[0][1])
            opt_results_sza_80.loc[index,'phi_opt'] = np.rad2deg(azi_shift(opt_pars[1][1]))
            opt_results_sza_80.loc[index,'n_opt'] = opt_pars[2][1]
            opt_results_sza_80.loc[index,'s_opt'] = opt_pars[3][1]
            opt_results_sza_80.loc[index,'zeta_opt'] = opt_pars[4][1]
            opt_results_sza_80.loc[index,'u0_opt'] = opt_pars[5][1]
            opt_results_sza_80.loc[index,'u1_opt'] = opt_pars[6][1]
            opt_results_sza_80.loc[index,'u2_opt'] = opt_pars[7][1]
            opt_results_sza_80.loc[index,'u3_opt'] = opt_pars[8][1]
        else:
            opt_results_sza_80.loc[index,'theta_opt'] = np.nan 
            opt_results_sza_80.loc[index,'phi_opt'] = np.nan
            opt_results_sza_80.loc[index,'n_opt'] = np.nan
            opt_results_sza_80.loc[index,'s_opt'] = np.nan
            opt_results_sza_80.loc[index,'zeta_opt'] = np.nan
            opt_results_sza_80.loc[index,'u0_opt'] = np.nan
            opt_results_sza_80.loc[index,'u1_opt'] = np.nan
            opt_results_sza_80.loc[index,'u2_opt'] = np.nan
            opt_results_sza_80.loc[index,'u3_opt'] = np.nan
            

#        if theta_results[key + ' ' + substat] < 0:
#            del theta_results[key + ' ' + substat]
#            
        opt_results_sza_80.loc[index,'theta_TP'] = pv_config["pv_stations"][key]["substat"][substat]["tilt_ap"][0]
        opt_results_sza_80.loc[index,'phi_TP'] = pv_config["pv_stations"][key]["substat"][substat]["azimuth_ap"][0]
        
        if key == 'PV_11':
            if "auew" in substat:
                keystring = key + ' ' + substat.upper().replace('_',' ')
            else:
                keystring = index
        else:
            keystring = key
        
        opt_results_sza_80.loc[index,'theta_act'] = real_angles.loc[keystring,"Theta_act"]
        opt_results_sza_80.loc[index,'phi_act'] = real_angles.loc[keystring,"Phi_act"]
        
        opt_results_sza_80.loc[index,'dtheta'] = opt_results_sza_80.loc[index,'theta_opt'] - opt_results_sza_80.loc[index,'theta_act']
        opt_results_sza_80.loc[index,'dphi'] = opt_results_sza_80.loc[index,'phi_opt'] - opt_results_sza_80.loc[index,'phi_act']
        
        opt_results_sza_80.loc[index,'P_max'] = np.max(pvsys[key]['df_cal'][('P_kW',substat)])
        if 'eff_temp_inv' in pvsys[key]['df_cal'].columns.levels[0] and substat in pvsys[key]['df_cal'].columns.levels[1]:
            opt_results_sza_80.loc[index,'eff_temp_ave'] = np.mean(pvsys[key]['df_cal'][('eff_temp_inv',substat)])
        
if 'PV_06 auew_1' in opt_results_sza_80.index:
    opt_results_sza_80.drop('PV_06 auew_1',axis=0,inplace=True)

opt_results_sza_80.to_csv(os.path.join(results_folder,'opt_results_table_sza_80.csv'))
        
res_dirs = list_dirs(results_folder)
if 'Bias_Plots' not in res_dirs:
    os.mkdir(os.path.join(results_folder,'Bias_Plots'))
#%% Scatter plots: azimuth angle
plt.ion()
plt.close('all')
plt.style.use("my_paper") #"presi_grid")

fig, ax = plt.subplots(figsize=(9,9))
plt.title(r'Azimuth angle retrieval for SZA < 80$^\circ$')    
ax.scatter(opt_results_sza_80['phi_act'].filter(regex='auew'), 
           opt_results_sza_80['phi_opt'].filter(regex='auew'),color='r',marker='o',s=80)
ax.scatter(opt_results_sza_80['phi_act'].filter(regex='egrid'), 
           opt_results_sza_80['phi_opt'].filter(regex='egrid'),color='b',marker='o',s=80)
ax.scatter(opt_results_sza_80['phi_act'].filter(regex='WR'), 
           opt_results_sza_80['phi_opt'].filter(regex='WR'),color='g',marker='o',s=80)
ax.set_xlabel(r'$\phi_{\rm actual}\ (\circ)$')
ax.set_ylabel(r'$\phi_{\rm opt}\ (\circ)$')
ax.set_xlim([130,235])
ax.set_ylim([130,235])
ax.set_aspect('equal', 'box')
ax.legend(['AUEW 15 min', 'egrid 1 Hz', 'Inverter 5 min'])

for txt in opt_results_sza_80.index:
    name = txt.split(' ')
    number = name[0].split('_')
    if number[0] == 'PV':
        label = number[1]
    else:
        label = name[0]
    if name[0] in ['PV_11'] and ("auew" in name[1] or "egrid" in name[1]):
        label = number[1] + ',' + name[1][-1]
    if name[0] in ['MS_02'] and ("egrid" in name[1] or "WR" in name[1]):
        if name[1][-1] == "2":
            label = name[0]# + ',' + name[1][-1]
        else:
            label = ''
    
#    if abs(opt_results_sza_80.loc[txt,'phi_opt'] - 185) < 5:
#        label = ''
        
    ax.annotate(label, xy=(opt_results_sza_80.loc[txt,'phi_act'], opt_results_sza_80.loc[txt,'phi_opt']),
                #xytext=(opt_results_sza_80.loc[txt,'phi_act'] + 2, opt_results_sza_80.loc[txt,'phi_opt'] - 1),
                fontsize=12)

ax.plot(np.linspace(ax.get_xlim()[0],ax.get_xlim()[1],50),
        np.linspace(ax.get_xlim()[0],ax.get_xlim()[1],50),':k')
fig.tight_layout()
plt.savefig(os.path.join(results_folder,'Bias_Plots/calibration_phi_sza_80.png'))

#%%Azimuth angle, zoom plot
ax.set_xlim([160,200])
ax.set_ylim([160,200])
fig.tight_layout()
plt.savefig(os.path.join(results_folder,'Bias_Plots/calibration_phi_sza_80_zoom.png'))

#%%Scatter plots: Elevation angle
fig2, ax2 = plt.subplots(figsize=(9,9))
plt.title(r'Elevation angle retrieval for SZA < 80$^\circ$')    
ax2.scatter(opt_results_sza_80['theta_act'].filter(regex='auew'), 
            opt_results_sza_80['theta_opt'].filter(regex='auew'),color='r',marker='o',s=80)
ax2.scatter(opt_results_sza_80['theta_act'].filter(regex='egrid'), 
            opt_results_sza_80['theta_opt'].filter(regex='egrid'),color='b',marker='o',s=80)
ax2.scatter(opt_results_sza_80['theta_act'].filter(regex='WR'), 
            opt_results_sza_80['theta_opt'].filter(regex='WR'),color='g',marker='o',s=80)
ax2.set_xlabel(r'$\theta_{\rm actual}\ (\circ)$')
ax2.set_ylabel(r'$\theta_{\rm opt}\ (\circ)$')
ax2.set_xlim([0,70])
ax2.set_ylim([0,70])
ax2.set_aspect('equal', 'box')
ax2.legend(['AUEW 15 min', 'egrid 1 Hz', 'Inverter 5 min'])

for txt in opt_results_sza_80.index:
    name = txt.split(' ')
    #if name[0] in ['PV_21','PV_06','PV_11']:
    number = name[0].split('_')
    if number[0] == 'PV':
        label = number[1]
    else:
        label = name[0]
    if name[0] in ['PV_11'] and ("auew" in name[1] or "egrid" in name[1]):
        label = number[1] + ',' + name[1][-1] 
    if name[0] in ['MS_02'] and ("egrid" in name[1] or "WR" in name[1]):
        if name[1][-1] == "2":
            label = name[0]# + ',' + name[1][-1]
        else:
            label = ''
        
    ax2.annotate(label, (opt_results_sza_80.loc[txt,'theta_act'], opt_results_sza_80.loc[txt,'theta_opt']) 
                ,fontsize=12)
ax2.plot(np.linspace(ax2.get_xlim()[0],ax2.get_xlim()[1],50),
        np.linspace(ax2.get_xlim()[0],ax2.get_xlim()[1],50),':k')
fig2.tight_layout()
plt.savefig(os.path.join(results_folder,'Bias_Plots/calibration_theta_sza_80.png'))

#%%Grid scatter plots
fig, axs = plt.subplots(1, 2, figsize=(14,7))            
axvec = axs.flatten()
print(opt_results_sza_80)
axvec[0].scatter(opt_results_sza_80['theta_act'].filter(regex='auew'), 
            opt_results_sza_80['theta_opt'].filter(regex='auew'),color='r',marker='o',s=80)
axvec[0].scatter(opt_results_sza_80['theta_act'].filter(regex='egrid'), 
            opt_results_sza_80['theta_opt'].filter(regex='egrid'),color='b',marker='o',s=80)
# axvec[0].scatter(opt_results_sza_80['theta_act'].filter(regex='WR'), 
#             opt_results_sza_80['theta_opt'].filter(regex='WR'),color='g',marker='o',s=80)
axvec[0].set_xlabel(r'$\theta_{\rm actual}\ (\circ)$')
axvec[0].set_ylabel(r'$\theta_{\rm opt}\ (\circ)$')
axvec[0].set_xlim([0,70])
axvec[0].set_ylim([0,70])
axvec[0].set_aspect('equal', 'box')
axvec[0].legend(['AUEW 15 min', 'egrid 1 Hz']) #, 'Inverter 5 min'])

axvec[0].plot(np.linspace(axvec[0].get_xlim()[0],axvec[0].get_xlim()[1],50),
        np.linspace(axvec[0].get_xlim()[0],axvec[0].get_xlim()[1],50),':k')

xlims_inset = (18,30)
ylims_inset = (18,30)
axins = axvec[0].inset_axes(
    [0.58, 0.05, 0.35, 0.35],
    xlim=xlims_inset, ylim=ylims_inset, #xticklabels=[], yticklabels=[],
    xticks=[20,25,30],yticks=[20,25,30])
axins.tick_params(axis='both',labelsize=12)
axins.plot(np.linspace(axins.get_xlim()[0],axins.get_xlim()[1],50),
        np.linspace(axins.get_xlim()[0],axins.get_xlim()[1],50),':k')

#axins.grid(False)

axins.scatter(opt_results_sza_80['theta_act'].filter(regex='auew'), 
            opt_results_sza_80['theta_opt'].filter(regex='auew'),color='r',
            marker='o',s=80)
axins.scatter(opt_results_sza_80['theta_act'].filter(regex='egrid'), 
            opt_results_sza_80['theta_opt'].filter(regex='egrid'),color='b',
            marker='o',s=80)

axvec[0].indicate_inset_zoom(axins, edgecolor="black")


for txt in opt_results_sza_80.index:
    name = txt.split(' ')
    #if name[0] in ['PV_21','PV_06','PV_11']:
    number = name[0].split('_')
    if number[0] == 'PV':
        label = number[1]
    else:
        label = name[0].replace('_','')
    if name[0] in ['PV_11'] and ("auew" in name[1] or "egrid" in name[1]):
        label = number[1] + ',' + name[1][-1] 
    if name[0] == 'MS_02':# and ("egrid" in name[1]):
        if name[1] == "egrid_1" or "egrid_2" in name[1] or "auew" in name[1]: # and "egrid" not in name[1]:
            label = name[0].replace('_','')# + ',' + name[1][-1]
        else:
            label = ''
        
    if (opt_results_sza_80.loc[txt,'theta_act'] < xlims_inset[0] or  \
        opt_results_sza_80.loc[txt,'theta_act'] > xlims_inset[1]) or \
        (opt_results_sza_80.loc[txt,'theta_opt'] < ylims_inset[0] or \
        opt_results_sza_80.loc[txt,'theta_opt'] > ylims_inset[1]):
        
        #if label != "09":
        delx = 0.5
        dely = -0.5
        axvec[0].annotate(label, (opt_results_sza_80.loc[txt,'theta_act']+delx,
                                      opt_results_sza_80.loc[txt,'theta_opt']+dely) 
                  ,fontsize=11)
    #     else:
    #         axvec[0].annotate(label, (opt_results_sza_80.loc[txt,'theta_act'], 
    #                                   opt_results_sza_80.loc[txt,'theta_opt']-2) 
    #               ,fontsize=11)
    else:
        delx = 0.1
        dely=-0.1
        if label == "06":
            dely = -0.5
            delx = -1.5
        axins.annotate(label, (opt_results_sza_80.loc[txt,'theta_act']+delx, 
                                opt_results_sza_80.loc[txt,'theta_opt']+dely) 
                   ,fontsize=11,xycoords='data')

axvec[1].scatter(opt_results_sza_80['phi_act'].filter(regex='auew'), 
           opt_results_sza_80['phi_opt'].filter(regex='auew'),color='r',marker='o',s=80)
axvec[1].scatter(opt_results_sza_80['phi_act'].filter(regex='egrid'), 
           opt_results_sza_80['phi_opt'].filter(regex='egrid'),color='b',marker='o',s=80)
# axvec[1].scatter(opt_results_sza_80['phi_act'].filter(regex='WR'), 
#            opt_results_sza_80['phi_opt'].filter(regex='WR'),color='g',marker='o',s=80)
axvec[1].set_xlabel(r'$\phi_{\rm actual}\ (\circ)$')
axvec[1].set_ylabel(r'$\phi_{\rm opt}\ (\circ)$')
axvec[1].set_xlim([130,235])
axvec[1].set_ylim([130,235])
axvec[1].set_aspect('equal', 'box')
axvec[1].legend(['AUEW 15 min', 'egrid 1 Hz'])#, 'Inverter 5 min'])

axvec[1].plot(np.linspace(axvec[1].get_xlim()[0],axvec[1].get_xlim()[1],50),
        np.linspace(axvec[1].get_xlim()[0],axvec[1].get_xlim()[1],50),':k')

xlims_inset2 = (176,192)
ylims_inset2 = (176,192)
axins2 = axvec[1].inset_axes(
    [0.58, 0.05, 0.35, 0.35],
    xlim=xlims_inset2, ylim=ylims_inset2, #xticklabels=[], yticklabels=[],
    xticks=[180,185,190],yticks=[180,185,190])
axins2.tick_params(axis='both',labelsize=12)
axins2.plot(np.linspace(axins2.get_xlim()[0],axins2.get_xlim()[1],50),
        np.linspace(axins2.get_xlim()[0],axins2.get_xlim()[1],50),':k')

#axins.grid(False)

axins2.scatter(opt_results_sza_80['phi_act'].filter(regex='auew'), 
            opt_results_sza_80['phi_opt'].filter(regex='auew'),color='r',
            marker='o',s=80)
axins2.scatter(opt_results_sza_80['phi_act'].filter(regex='egrid'), 
            opt_results_sza_80['phi_opt'].filter(regex='egrid'),color='b',
            marker='o',s=80)

axvec[1].indicate_inset_zoom(axins2, edgecolor="black")

for txt in opt_results_sza_80.index:
    name = txt.split(' ')
    #if name[0] in ['PV_21','PV_06','PV_11']:
    number = name[0].split('_')
    if number[0] == 'PV':
        label = number[1]
    else:
        label = name[0].replace('_','')
    if name[0] in ['PV_11'] and ("auew" in name[1] or "egrid" in name[1]):
        label = number[1] + ',' + name[1][-1] 
        if "egrid_1" in name[1]:
            label = ""
    if name[0] == 'MS_02':# and ("egrid" in name[1]):
        if name[1] == "egrid_2" or "auew" in name[1]: # and "egrid" not in name[1]:
            label = name[0].replace('_','')# + ',' + name[1][-1]
        else:
            label = ''
    # if name[0] in ['MS_02'] and ("egrid" in name[1] or "WR" in name[1]):
    #     if name[1][-1] == "2":
    #         label = name[0].replace('_','')# + ',' + name[1][-1]
    #     # else:
        #     label = ''
        
    
    if (opt_results_sza_80.loc[txt,'phi_act'] < xlims_inset2[0] or  \
        opt_results_sza_80.loc[txt,'phi_act'] > xlims_inset2[1]) or \
        (opt_results_sza_80.loc[txt,'phi_opt'] < ylims_inset2[0] or \
        opt_results_sza_80.loc[txt,'phi_opt'] > ylims_inset2[1]):
        
        delx = 0.5
        dely = -0.5
        #if label == "08":
            
        axvec[1].annotate(label, (opt_results_sza_80.loc[txt,'phi_act']+delx,
                                  opt_results_sza_80.loc[txt,'phi_opt']+dely) 
                 ,fontsize=11)
    else:
        delx = 0.1
        dely = -0.1
        if label == "06":
            dely = -1.
            delx=-1.5
        axins2.annotate(label, (opt_results_sza_80.loc[txt,'phi_act']+delx, 
                                 opt_results_sza_80.loc[txt,'phi_opt']+dely) 
                   ,fontsize=12,xycoords='data')
    

fig.subplots_adjust(wspace=-0.15)

# fig.suptitle(r'Elevation ($\theta$) and azimuth ($\phi$) angle retrievals for SZA < 80$^\circ$')
# fig.subplots_adjust(top=0.82)   

fig.tight_layout()
plt.savefig(os.path.join(results_folder,'Bias_Plots/calibration_theta_phi_grid_sza_80.png'))

#%%Deviation plots
fig3, ax3 = plt.subplots(figsize=(9,9))
plt.title(r'Deviation in azimuth and elevation for SZA < 80$^\circ$')    
ax3.scatter(opt_results_sza_80['dphi'].filter(regex='auew'),
            opt_results_sza_80['dtheta'].filter(regex='auew'),color='r',marker='o',s=80)
ax3.scatter(opt_results_sza_80['dphi'].filter(regex='egrid'), 
            opt_results_sza_80['dtheta'].filter(regex='egrid'),color='b',marker='o',s=80)
ax3.scatter(opt_results_sza_80['dphi'].filter(regex='WR'), 
            opt_results_sza_80['dtheta'].filter(regex='WR'),color='g',marker='o',s=80)
ax3.set_ylabel(r'$\theta_{\rm opt} - \theta_{\rm actual}\ (\circ)$')
ax3.set_xlabel(r'$\phi_{\rm opt} - \phi_{\rm actual}\ (\circ)$')
ax3.set_xlim([-25,25])
ax3.set_ylim([-25,25])
ax3.set_aspect('equal', 'box')
ax3.legend(['AUEW 15 min', 'egrid 1 Hz', 'Inverter 5 min'])

for txt in opt_results_sza_80.index:
    name = txt.split(' ')
    #if name[0] in ['PV_21','PV_06','PV_11']:
    number = name[0].split('_')
    if number[0] == 'PV':
        label = number[1]
    else:
        label = name[0]
    if name[0] in ['PV_11'] and "auew" in name[1]:
        label = number[1] + ',' + name[1][-1] 
    ax3.annotate(label, (opt_results_sza_80.loc[txt,'dphi'], opt_results_sza_80.loc[txt,'dtheta']) 
                ,fontsize=12)
ax3.plot(np.linspace(ax3.get_xlim()[0],ax3.get_xlim()[1],50),
        np.linspace(ax3.get_xlim()[0],ax3.get_xlim()[1],50),':k')
fig3.tight_layout()
plt.savefig(os.path.join(results_folder,'Bias_Plots/deviation_theta_phi_sza_80.png'))

##%%
#fig = plt.figure(figsize=(9,9))
#ax = fig.add_subplot(111, projection = '3d')
#ax.scatter(opt_results_sza_80.u0_opt,opt_results_sza_80.u1_opt,opt_results_sza_80.u2_opt)

#%%Temperature model coefficients: ambient temp, wind speed
# plt.close('all')
# fig, ax = plt.subplots(figsize=(9,9))
# ax.scatter(opt_results_sza_80.u0_opt.filter(regex='auew'),opt_results_sza_80.u2_opt.filter(regex='auew'),
#            color='r',marker='o',s=80)
# ax.scatter(opt_results_sza_80.u0_opt.filter(regex='egrid'),opt_results_sza_80.u2_opt.filter(regex='egrid'),
#            color='b',marker='o',s=80)
# ax.set_xlabel(r'$u_0$')
# ax.set_ylabel(r'$u_2\ (^{\circ}C\,m^{-1}\,s)$')
# ax.set_title('Ambient temperature vs. wind speed coefficient')
# for txt in opt_results_sza_80.index:
#     name = txt.split(' ')
#     #if name[0] in ['PV_21','PV_06','PV_11']:
#     number = name[0].split('_')
#     if number[0] == 'PV':
#         label = number[1]
#     else:
#         label = name[0]
#     if name[0] in ['PV_11'] and "auew" in name[1]:
#         label = number[1] + ',' + name[1][-1] 
#     ax.annotate(label, (opt_results_sza_80.loc[txt,'u0_opt'], opt_results_sza_80.loc[txt,'u2_opt']) 
#                 ,fontsize=12)
# fig.tight_layout()
# plt.savefig(os.path.join(results_folder,'Bias_Plots/u0_u2_opt.png'))

#%%Ambient temperature, irradiance
# plt.close('all')
# fig, ax = plt.subplots(figsize=(9,9))
# ax.scatter(opt_results_sza_80.u0_opt.filter(regex='auew'),opt_results_sza_80.u1_opt.filter(regex='auew'),
#            color='r',marker='o',s=80)
# ax.scatter(opt_results_sza_80.u0_opt.filter(regex='egrid'),opt_results_sza_80.u1_opt.filter(regex='egrid'),
#            color='b',marker='o',s=80)
# ax.set_xlabel(r'$u_0$')
# ax.set_ylabel(r'$u_1\ (^{\circ}C\, m^2\, W^{-1})$')
# ax.set_title('Ambient temperature vs. irradiance coefficient')
# for txt in opt_results_sza_80.index:
#     name = txt.split(' ')
#     #if name[0] in ['PV_21','PV_06','PV_11']:
#     number = name[0].split('_')
#     if number[0] == 'PV':
#         label = number[1]
#     else:
#         label = name[0]
#     if name[0] in ['PV_11'] and "auew" in name[1]:
#         label = number[1] + ',' + name[1][-1] 
#     ax.annotate(label, (opt_results_sza_80.loc[txt,'u0_opt'], opt_results_sza_80.loc[txt,'u1_opt']) 
#                 ,fontsize=12)
# fig.tight_layout()
# plt.savefig(os.path.join(results_folder,'Bias_Plots/u0_u1_opt.png'))

#%%Deviation in tilt vs. s

# plt.close('all')
# fig = plt.figure(figsize=(9,9)) #, gridspec_kw = {'height_ratios': [3, 1]})
# gs = GridSpec(2, 1, height_ratios=[3, 1])
# ax1 = fig.add_subplot(gs[0])
# ax1.scatter(opt_results_sza_80.s_opt.filter(regex='auew'),opt_results_sza_80.dtheta.filter(regex='auew'),
#            color='r',marker='o',s=80)
# ax1.scatter(opt_results_sza_80.s_opt.filter(regex='egrid'),opt_results_sza_80.dtheta.filter(regex='egrid'),
#            color='b',marker='o',s=80)
# #ax.set_xlabel(r'$P_{\rm max}$ (kW)')
# #ax1.set_xlabel(r'$s_{\rm opt}$ (m$^2$)')
# ax1.set_ylabel(r'$\theta_{\rm opt} - \theta_{\rm actual}\ (\circ)$')
# ax1.set_title('Deviation in tilt angle vs. area scaling factor')
# ax1.set_ylim([-25,25])
# ax1.set_xlim([0,250])
# for txt in opt_results_sza_80.index:
#     name = txt.split(' ')
#     #if name[0] in ['PV_21','PV_06','PV_11']:
#     number = name[0].split('_')
#     if number[0] == 'PV':
#         label = number[1]
#     else:
#         label = name[0]
#     if name[0] in ['PV_11'] and "auew" in name[1]:
#         label = number[1] + ',' + name[1][-1] 
#     ax1.annotate(label, (opt_results_sza_80.loc[txt,'s_opt'], opt_results_sza_80.loc[txt,'dtheta']) 
#                 ,fontsize=12)
#fig.tight_layout()
#plt.savefig(os.path.join(results_folder,'Bias_Plots/dtheta_sopt.png'))



#plt.close('all')
#fig, ax = plt.subplots(figsize=(9,9))
# ax2 = fig.add_subplot(gs[1])
# ax2.scatter(opt_results_sza_80.eff_temp_ave.filter(regex='auew'),opt_results_sza_80.dtheta.filter(regex='auew'),
#            color='r',marker='o',s=80)
# ax2.scatter(opt_results_sza_80.eff_temp_ave.filter(regex='egrid'),opt_results_sza_80.dtheta.filter(regex='egrid'),
#            color='b',marker='o',s=80)
# ax2.set_xlabel(r'$\eta_{\rm eff}$')
# ax2.set_ylabel(r'$\theta_{\rm opt} - \theta_{\rm actual}\ (\circ)$')
# #ax2.set_title('Deviation in tilt angle vs. effective efficiency correction')
# ax2.set_ylim([-25,25])
# #ax.set_xlim([0,250])
# for txt in opt_results_sza_80.index:
#     name = txt.split(' ')
#     #if name[0] in ['PV_21','PV_06','PV_11']:
#     number = name[0].split('_')
#     if number[0] == 'PV':
#         label = number[1]
#     else:
#         label = name[0]
#     if name[0] in ['PV_11'] and "auew" in name[1]:
#         label = number[1] + ',' + name[1][-1] 
#     ax2.annotate(label, (opt_results_sza_80.loc[txt,'eff_temp_ave'], opt_results_sza_80.loc[txt,'dtheta']) 
#                 ,fontsize=12)
#fig.tight_layout()
#plt.savefig(os.path.join(results_folder,'Bias_Plots/dtheta_eff_temp.png'))

#%%Extra plots
#pv_config["sza_max"] = 75
#pvsys, sys_info = load_inversion_results(config["description"],rt_config,pv_config,data_config,homepath)
#
#opt_results_sza_75 = pd.DataFrame() #theta_results = {}
##theta_tropos = {}
#
#for key in pvsys:
#    for substat in pvsys[key]["substations"]:
#        opt_pars = pvsys[key]["substations"][substat]["opt"]
#        index = key + ' ' + substat
#        if not np.isnan(opt_pars['x_min']).any():
#            opt_results_sza_75.loc[index,'theta_opt'] = np.rad2deg(opt_pars['x_min'][0])
#            opt_results_sza_75.loc[index,'phi_opt'] = np.rad2deg(azi_shift(opt_pars['x_min'][1]))
#            
##        if theta_results[key + ' ' + substat] < 0:
##            del theta_results[key + ' ' + substat]
##            
#        opt_results_sza_75.loc[index,'theta_TP'] = pv_config["pv_stations"][key]["substat"][substat]["tilt_ap"][0]
#        opt_results_sza_75.loc[index,'phi_TP'] = pv_config["pv_stations"][key]["substat"][substat]["azimuth_ap"][0]
#        
#        if key == 'PV_11':
#            if "auew" in substat:
#                keystring = key + ' ' + substat.upper().replace('_',' ')
#            else:
#                keystring = index
#        else:
#            keystring = key
#        
#        opt_results_sza_75.loc[index,'theta_act'] = real_angles.loc[keystring,"Theta_act"]
#        opt_results_sza_75.loc[index,'phi_act'] = real_angles.loc[keystring,"Phi_act"]
#        
#        opt_results_sza_75.loc[index,'dtheta'] = opt_results_sza_75.loc[index,'theta_act'] - opt_results_sza_75.loc[index,'theta_opt']
#        opt_results_sza_75.loc[index,'dphi'] = opt_results_sza_75.loc[index,'phi_act'] - opt_results_sza_75.loc[index,'phi_opt']
#        
#opt_results_sza_75.drop('PV_06 auew_1',axis=0,inplace=True)
#
##%%
#plt.style.use("my_presi_grid")
#
#fig, ax = plt.subplots(figsize=(9,9))
#plt.title(r'Azimuth angle retrieval for SZA < 75$^\circ$')    
#ax.scatter(opt_results_sza_75['phi_act'], opt_results_sza_75['phi_opt'],color='r',marker='*',s=80)
#ax.set_xlabel(r'$\phi_{\rm actual}\ (^\circ)$')
#ax.set_ylabel(r'$\phi_{\rm opt}\ (^\circ)$')
#ax.set_xlim([130,230])
#ax.set_ylim([130,230])
#ax.set_aspect('equal', 'box')
##for txt in opt_results_sza_75.index:
##    ax.annotate(txt, xy=(opt_results_sza_75.loc[txt,'phi_act'], opt_results_sza_75.loc[txt,'phi_opt']),
##                #xytext=(opt_results_sza_75.loc[txt,'phi_act'] + 1, opt_results_sza_75.loc[txt,'phi_opt'] - 1),
##                fontsize=12)
#
#ax.plot(np.linspace(ax.get_xlim()[0],ax.get_xlim()[1],50),
#        np.linspace(ax.get_xlim()[0],ax.get_xlim()[1],50),':k')
#fig.tight_layout()
#plt.savefig('deviation_phi_sza_75.png')
#
#fig2, ax2 = plt.subplots(figsize=(9,9))
#plt.title(r'Elevation angle retrieval for SZA < 75$^\circ$')    
#ax2.scatter(opt_results_sza_75['theta_act'], opt_results_sza_75['theta_opt'],color='r',marker='*',s=80)
#ax2.set_xlabel(r'$\theta_{\rm actual}\ (^\circ)$')
#ax2.set_ylabel(r'$\theta_{\rm opt}\ (^\circ)$')
#ax2.set_xlim([0,70])
#ax2.set_ylim([0,70])
#ax2.set_aspect('equal', 'box')
#
##for txt in opt_results_sza_75.index:
##    ax2.annotate(txt.split(' ')[0], (opt_results_sza_75.loc[txt,'theta_act'], opt_results_sza_75.loc[txt,'theta_opt']) 
##                ,fontsize=12)
#ax2.plot(np.linspace(ax2.get_xlim()[0],ax2.get_xlim()[1],50),
#        np.linspace(ax2.get_xlim()[0],ax2.get_xlim()[1],50),':k')
#fig.tight_layout()
#plt.savefig('deviation_theta_sza_75.png')


       
