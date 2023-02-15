#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 15:37:52 2019

@author: james
"""

import numpy as np
import pandas as pd
import os
import yaml

#%%

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

def read_auew_excel_write_to_csv(path_source,savepath):
    """
    Read AUEW power data from Excel file and split into separate CSV files
    
    args:
    :param path_source: string, path where excel file is stored
    :param savepath: string, path where files should be saved (in folders)
    """

    #Read excel data from AÜW
    df = pd.read_excel(path_source,encoding="utf-8")    
    df.rename(columns={"Unnamed: 0":"MEZ","Anlage:":"MESZ"},inplace=True)    
        
    #Go through file and extract columns
    for colname in df.columns[2:]:
        #Check existing directories every time, since some have substations!
        station_dirs = list_dirs(savepath)
        #If substation does not exist
        if '.' not in colname:
            #Create folder for each station if not already created
            filepath = os.path.join(savepath,colname)
            if colname not in station_dirs:
                os.mkdir(filepath)
            
            #Create folder for power measurement            
            sensor_dirs = list_dirs(filepath)        
            filepath = os.path.join(filepath,"Leistungsmessung_AUEW")    
            if "Leistungsmessung_AUEW" not in sensor_dirs:
                os.mkdir(filepath)
                
            #Create filename
            filename = "Leistung_AUEW_" + colname + '.csv'
            #Write to file
            f = open(os.path.join(filepath,filename),'w')
            f.write(';Anlage_MetPVNet: ' + colname.replace('.','_') + '\n')
            auewname = df[colname][3].encode("utf-8")
            f.write(';Anlage_AUEW: ' + str(auewname) + '\n')                       
            f.write(';Sensor: ' + str(df[colname].iloc[0]) + '\n')
            f.write(';Einheit: ' + str(df[colname].iloc[2]) + '\n')
            f.write(df.columns[0] + ';' + df.columns[1] + ';Power_kW\n')                   
            df.iloc[4:][['MEZ','MESZ',colname]].to_csv(f,sep=';',index=False,header=False,
                   float_format="%.3f",na_rep='nan')
            f.close()
        else:  #In this case split up the measurement into two substations                      
            pvname = colname.split('.')[0]
            subname = colname.split('.')[1]
            filepath = os.path.join(savepath,pvname)
            if pvname not in station_dirs:
                os.mkdir(filepath)
            
            #Create folder for power measurement            
            sensor_dirs = list_dirs(filepath)        
            filepath = os.path.join(filepath,"Leistungsmessung_AUEW")    
            if "Leistungsmessung_AUEW" not in sensor_dirs:
                os.mkdir(filepath)
                
            #Create filename
            filename = "Leistung_AUEW_" + pvname + '_' + subname + '.csv'
            #Write to file
            f = open(os.path.join(filepath,filename),'w')
            f.write(';Anlage_MetPVNet: ' + pvname + ' substation ' + subname + '\n')
            auewname = df[colname][3].encode("utf-8")
            f.write(';Anlage_AUEW: ' + str(auewname) + '\n')                       
            f.write(';Sensor: ' + str(df[colname].iloc[0]) + '\n')
            f.write(';Einheit: ' + str(df[colname].iloc[2]) + '\n')
            f.write(df.columns[0] + ';' + df.columns[1] + ';Power_kW\n')                   
            df.iloc[4:][['MEZ','MESZ',colname]].to_csv(f,sep=';',index=False,header=False,
                   float_format="%.3f",na_rep='nan')
            f.close()            
                
def read_process_egrid_data(path_source,savepath):
    """
    Read in egrid CSV files and process them (add up AC power from three phases)
    then save as new CSV files
    
    args:
    :param path_source: string, path where measurement data is stored 
    """
    
    station_dirs = list_dirs(path_source)
    
    for station in station_dirs:        
        station_path = os.path.join(path_source,station)
        sensor_dirs = list_dirs(station_path)       
        if "egrid-Messung" in sensor_dirs:
            readpath = os.path.join(station_path,"egrid-Messung")
            
            files = list_files(readpath)
            print("Reading egrid 3 phase power data from %s" % station)
            dfs = [(filename,pd.read_csv(os.path.join(readpath,filename),header=0,sep=';',
                               decimal=',',usecols=[0,1,2,3,4,5,6,7],low_memory=False)) for filename in files]
               
            sensor_dirs = list_dirs(station_path)        
            savepath = os.path.join(station_path,"Leistungsmessung_egrid")    
            if "Leistungsmessung_egrid" not in sensor_dirs:
                os.mkdir(savepath)
                
            #See whether we need substations, if there are more than one inverter
            substats = list(set([filename.split('_')[1][-1] for (filename,df) in dfs]))
            substat_dirs = list_dirs(savepath)
            if len(substats) > 1:
                #Create substation path
                for substat in substats:
                    subpath = "Wechselrichter_" + str(substat)
                    if subpath not in substat_dirs:
                        os.path.join(savepath,subpath)
            
            print("Calculating power, writing egrid data from %s to CSV files" % station)
            for (filename,df) in dfs:
                #In this case the header is missing, need to reconstruct it and move data down!
                if df.columns[0] != "Datum": 
                    df = df.shift(periods=1)
                    df.iloc[0,0:2] = df.columns[0:2]
                    for i in range(2,5):
                        df.iloc[0,i] = float(df.columns[i].replace(',','.'))
                    #df.columns = ['Datum','Uhrzeit','PL1','PL2','PL3']
                    df.columns = ['Datum','Uhrzeit','UL1','UL2','UL3','IL1','IL2','IL3']
                    
                #Set index to be datetime object    
                df.index = pd.to_datetime(df.iloc[:,0] + ' ' + df.iloc[:,1],format='%Y-%m-%d %H:%M:%S',errors="coerce")
                df.drop(columns=["Datum","Uhrzeit"],inplace=True)
                df.index.rename('Time(UTC)',inplace=True)
                
                #Deal with bad data                
                for colname in df.columns:
                    df[colname] = [str(x).replace(',', '.') for x in df[colname]]
                    df[colname] = pd.to_numeric(df[colname], errors="coerce")
                    
                #Sum up three phase power to give total power                    
#                df['P[W]'] = df["PL1"] + df["PL2"] + df["PL3"] #*df["IL2"] + df["UL3"]*df["IL3"]
#                df.drop(columns=["PL1","PL2","PL3"],inplace=True)
                #df['P[W]'] = np.sqrt(3)*(df["UL1"]*df["IL1"] + df["UL2"]*df["IL2"] + df["UL3"]*df["IL3"])
                
                #Set constant cosphi due to measurement error
                cosphi = 0.987
                
                #Calculate power
                #Ptot = < ULi > sum_i ILi cosphi
                df['P[W]'] = df.loc[:,["UL1","UL2","UL3"]].mean(axis='columns')*cosphi*\
                (df["IL1"] + df["IL2"] + df["IL3"])
                df.drop(columns=["UL1","UL2","UL3","IL1","IL2","IL3"],inplace=True)
                
                if len(substats) > 1:                
                    substat = filename.split('_')[1][-1]
                    subpath = "Wechselrichter_" + str(substat)		                                    
                    subname = "_WR_" + str(substat)
                else:
                    substat = ''
                    subpath = ''
                    subname = ''
                    
                #Create filename                    
                newfilename = "MetPVNet_" + station + subname + "_eM_" +\
                                df.index[0].strftime('%Y-%m-%d') + '.csv'
                
                #Write to file
                f = open(os.path.join(savepath,subpath,newfilename),'w')
                f.write('#Anlage_MetPVNet: ' + station + '\n')                                
                f.write('#Sensor: eGrid Messbox WR' + substat + '\n')
                f.write('#Einheit: W\n')                
                df.to_csv(f,sep=',',header=True,float_format="%.2f",na_rep='nan',
                          date_format='%Y.%m.%d %H:%M:%S')
                f.close()                

#%%  
#######################################################################
###                         MAIN PROGRAM                            ###
#######################################################################
#def main():
#    import argparse
#    
#    parser = argparse.ArgumentParser()
#    parser.add_argument("configfile", help="yaml file containing config")
#    parser.add_argument("-d","--description",help="description of the data")    
#    args = parser.parse_args()

config_filename = "../examples/config_pvdata_2019_messkampagne.yaml" #os.path.abspath(args.configfile) #

#Read in values from configuration file
config = load_yaml_configfile(config_filename)

#Define path where CSV files should be saved
savepath = config["savepath"]

if "auew_data" in config:
    #Define path of Excel file
    auew_data = config["auew_data"]
    read_auew_excel_write_to_csv(auew_data,savepath)

if "egrid_path" in config:
    #Define path where egrid Data is found
    egrid_path = config["egrid_path"]
    #Read egrid data and process
    read_process_egrid_data(egrid_path,savepath)
