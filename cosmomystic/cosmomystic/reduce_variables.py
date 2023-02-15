#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 13:20:03 2022

@author: james
"""

import xarray as xr
import os

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

paths = {}
paths.update({"2018":{}})
paths.update({"2019":{}})

paths["2018"].update({"old":"/home/james/MetPVNet/Data/COSMO/raw_data/cosmo_kampagne_2018_all_new"})
paths["2018"].update({"new":"/home/james/MetPVNet/Data/COSMO/raw_data/cosmo_kampagne_2018_all_new_reducedsize"})
paths["2019"].update({"old":"/home/james/MetPVNet/Data/COSMO/raw_data/cosmo_kampagne_2019_all"})
paths["2019"].update({"new":"/home/james/MetPVNet/Data/COSMO/raw_data/cosmo_kampagne_2019_all_reducedsize"})

for year in paths:    
    for filename in list_files(paths[year]["old"]):
    
        print(f"Opening {filename} from {paths[year]['old']} and extracting relevant variables")
        data = xr.open_dataset(os.path.join(paths[year]["old"],filename))
        
        data_subset = data[["pres_generalVerticalLayer", "t_generalVerticalLayer",
                            "q","sp","2t","2r","HHL","10u","10v","SWDIR_S","SWDIFD_S",
                            "clwmr","rotated_pole"]]
        
        data_subset.to_netcdf(os.path.join(paths[year]["new"],filename))
        print(f"Reduced {filename} saved to {paths[year]['new']}")
        
        
    
    
    