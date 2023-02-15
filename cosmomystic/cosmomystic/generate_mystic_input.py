#!/usr/bin/python

import os
import itertools
import yaml
import numpy as np
import xarray as xr
#import pyresample
import datetime
import netCDF4 as nc4

from contextlib import closing
from scipy.interpolate import interp1d

from .unit_conversion import *





###############################################################
###   general functions to load and process (cosmo) data    ###
###############################################################


def load_data(path_cosmo, fname_cosmo, vardict):
    """
    loads COSMO data from netCDF file and returns 
    a dictionary of variables
    
    args:
    :param path_cosmo: string, absolute or relative path to directory
                       of the netCDF COSMO files
    :param fname_cosmo: string, full filename of the COSMO netCDF file
    :param vardict: dict, names of variables and dimensions
                    in COSMO netCDF file. Keys are used within the
                    program to identify variables.
    
    out:
    :return: dictionary, key are the variable and dimensonnames from varlist,
             values are the n-dimensional data for each variable: 
             eg. time-1D, lat/lon-2D, surface pressure-3D (time,lat,lon),
             level pressure-4D (time, lev, lat, lon)
    """
    
    data_dict = {}
    
    with closing(xr.open_dataset(os.path.join(path_cosmo,fname_cosmo), decode_times=False)) as ds:
        time = ds.time.data
        timeunit = ds.time.units
        # change varlist to vardict:  
        for key in vardict:
            var = vardict[key]
            if var == "time":
               timeunit = ds[var].units
               data_dict.update({"timeunit":timeunit})
            
            data = ds[var].data
            data_dict.update({key:data})
            
    return data_dict


def combine_lyr_sfc_properties_4d(q_lyr, T_lyr, p_lyr, rh_sfc, T_sfc, p_sfc, phase="water", hum_output="numdens"):
    """
    converts 4 dimensional fields of q to relative 
    humidity and combines sufrace and layer properties.
    Dimensions of layer-variables (4d): (time, z, lat, lon)
    Dimensions of surface-variables (3d): (time, lat, lon)
    
    args:
    :param q_lyr: specific humidity in [kg/kg] with 4 dimensions
    :param T_lyr: temperature in [K] with 4 dimensions
    :param p_lyr: total air pressure in [Pa] with 4 dimensions
    :param rh_sfc: relative humidity at the surface in percent 3 dimensions
    :param T_sfc: surface temperature in [K] with 3 dimensions
    :param p_sfc: surface total air pressure in [Pa] with 3 dimensions
    :param phase: keyword for the phase of water: water = liquid, ice = solid
    
    out:
    :return: dictionary of merged 4d arrays with the keys:
             'P' for pressure, 'RH' for relative humidity and 'T' for temperature
    """
    out_dict = {}
    
    # squeez surface variables which sometimes have an additional axiss
    rh_sfc = rh_sfc.squeeze()
    T_sfc  = T_sfc.squeeze()
    p_sfc  = p_sfc.squeeze()

    if hum_output == "numdens":
        n_h2o_lyr = specifichum2numdens(q_lyr,T_lyr, p_lyr)
        n_h2o_sfc = relhum2numdens(rh_sfc, T_sfc)
        n_h2o_4d = np.append(n_h2o_lyr, n_h2o_sfc[:,np.newaxis,...], axis=1)

        
        out_dict.update({"n_h2o":n_h2o_4d})
        
    elif hum_output == "relhum":
        rh_lyr = specifichum2RH(q_lyr, T_lyr, p_lyr, phase="water")
        rh_4d = np.append(n_h2o_lyr, rh_sfc[:,np.newaxis,...], axis=1)
        
        out_dict.update({"RH":rh_4d})
    
    else:
        print("ERROR in function 'combine_lyr_sfc_properties_4d'")
        print(("passed keyword for 'hum_output' is {}".format(hum_output)))
        print("keyword is not supported, please chose 'relhum' or 'numdens'")
        print("program exits")
        quit()
    
    T_4d  = np.append(T_lyr, T_sfc[:,np.newaxis,...], axis=1)
    p_4d  = np.append(p_lyr, p_sfc[:,np.newaxis,...], axis=1)
    
    out_dict.update({"T":T_4d, "P":p_4d})
    
    return out_dict
    
# pyresample is not available in python3 of module system
# therefore an alternative approach using haversine is implemented
# to determine the nearest neighbour gridpoint
def haversine_distance(lat_lon_start, lat_lon_dest):
    """
    calculate the shortest distance between 2 points on a sphere 
    (great-circle distance) using the haversine formula
   
    args:
    :param lat_lon_start: tuple, lat and lon of the start point (unit: degree)
    :param lat_lon_dest: tuple, lat and lon of the end point(s). lat and lon 
                           can be floats or nd-arrays (unit: degree)
  
    out:
    :return: float or nd-array, distance between start and end point(s) in meters
    """
    R = 6371e3 # earth mean radius in m (WGS84)
 
    # convert degree to radiant
    lat1, lon1, lat2, lon2 = list(map(np.deg2rad, [lat_lon_start[0], lat_lon_start[1], lat_lon_dest[0], lat_lon_dest[1]]))
 
    dlat = lat2-lat1
    dlon = lon2-lon1
 
    # apply haversine formula
    a = np.sin(dlat/2.)*np.sin(dlat/2.) + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.)*np.sin(dlon/2.)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = R * c
 
    return d

def find_nearest_cosmo_gridpoint(lat_grid, lon_grid, lat_lon_pos):
    """
    search the nearest neighbour in a lat-lon grid and return the indices
    of its position
    
    args:
    :param lat_grid: 2d array containing the latitude grid
    :param lon_grid: 2d array containing the longitude grid
    :param lat_lon_pos: dictionary with values of (lat, lon)
    
    out:
    :return idx_dict: dictionary with tuples of (lat, lon) indices 
                      of the nearest gridpoint 
    :return lat_lon_gp: dictionary with same keys like 'idx_dict' 
                        containing the actual lat and lon position 
                        of the nearest cosmo gridpoint
    """
    latlon_grid = (lat_grid, lon_grid)

    idx_dict = {}
    lat_lon_gp = {}
    for key in lat_lon_pos:
        lat_pos = np.array([lat_lon_pos[key][0]])
        lon_pos = np.array([lat_lon_pos[key][1]])
    
        d = haversine_distance((lat_pos,lon_pos), latlon_grid)

        # get 1d-index of minimal distance
        idx_1d = np.nanargmin(d)

        # calculate 2d idcs
        idcs_2d = np.unravel_index(idx_1d, np.shape(d))

        idx_dict.update({key:(idcs_2d[0], idcs_2d[1])})
    
        lat_gp = lat_grid[idcs_2d[0], idcs_2d[1]]
        lon_gp = lon_grid[idcs_2d[0], idcs_2d[1]]
        lat_lon_gp.update({key:(lat_gp, lon_gp)})
        
    return idx_dict, lat_lon_gp


#def find_nearest_cosmo_gridpoint(lat_grid, lon_grid, lat_lon_pos):
#    """
#    search the nearest neighbour in a lat-lon grid and return the indices
#    of its position
#    
#    args:
#    :param lat_grid: 2d array containing the latitude grid
#    :param lon_grid: 2d array containing the longitude grid
#    :param lat_lon_pos: dictionary with values of (lat, lon)
#    
#    out:
#    :return idx_dict: dictionary with tuples of (lat, lon) indices 
#                      of the nearest gridpoint 
#    :return lat_lon_gp: dictionary with same keys like 'idx_dict' 
#                        containing the actual lat and lon position 
#                        of the nearest cosmo gridpoint
#    """
#    
#    idx_dict = {}
#    lat_lon_gp = {}
#    for key in lat_lon_pos:
#        lat_pos = np.array([lat_lon_pos[key][0]])
#        lon_pos = np.array([lat_lon_pos[key][1]])
#        grid = pyresample.geometry.GridDefinition(lats=lat_grid, lons=lon_grid)
#        swath = pyresample.geometry.SwathDefinition(lons=lon_pos, lats=lat_pos)
#
#        _, _, index_array, distance_array = pyresample.kd_tree.get_neighbour_info(source_geo_def=grid, 
#                                                                            target_geo_def=swath, 
#                                                                            radius_of_influence=2000,
#                                                                            neighbours=1)
#
#        index_array_2d = np.unravel_index(index_array, grid.shape)
#        
#        idx_dict.update({key:(index_array_2d[0], index_array_2d[1])})
#    
#        lat_gp = lat_grid[index_array_2d[0], index_array_2d[1]]
#        lon_gp = lon_grid[index_array_2d[0], index_array_2d[1]]
#        lat_lon_gp.update({key:(lat_gp, lon_gp)})
#        
#    return idx_dict, lat_lon_gp
#

def generate_new_time_array(times, timeunit, timeres):
    """
    args:
    :param times_old: array with cosmo times
    :param timeunit_old: string describing the timeunit of the cosmo timestamps
    :param timres: desired timeresolution in minutes
    
    out:
    :return: array with new time grid in the same unit like 'times_old'
    """
    times_filt = filter_times(times, timeunit)
    
    times_datetime = nc4.num2date(times_filt, timeunit, only_use_cftime_datetimes=False)
    times_epochms = nc4.date2num(times_datetime, "microseconds since 1970-01-01")
   
    timeres_ms = timeres*60.*1e6
    dt_ms = times_epochms[-1] - times_epochms[0]
    n = int(np.floor(dt_ms / timeres_ms))
    eff_tres_ms = dt_ms / n
    
    if np.around(eff_tres_ms/60./1e6, decimals=4)  != timeres:
        print(("to provide a perfect linear time grid, timeresolution is set to {} minutes".format(eff_tres_ms/60./1e6)))
   
    newtimes_epochms = [times_epochms[0]+(i*eff_tres_ms) for i in range(n)]
    newtimes_epochms.append(times_epochms[-1])
    
    newtimes_datetime = nc4.num2date(newtimes_epochms, "microseconds since 1970-01-01", only_use_cftime_datetimes=False)
    newtimes_out = nc4.date2num(newtimes_datetime, timeunit)
    
    return newtimes_out 


def isclose(a, b, abs_dev=1e-3):
    """
    function is used to check if 2 values are
    at least as close together as the given deviation
    
    args:
    :param a: float
    :param b: float
    :param abs_dev: float, maximum allowed absolute 
                    deviation between a and b.
                    Default is 1e-3
    out:
    :return: bool, True if a and b are close enough and 
             false if the deviation between a and b is larger
             than abs_dev
    
    """
    
    return abs(a-b) <= abs_dev



def interpolate_profiles_in_time(data_4d, times_old, times_new, method="linear"):
    """
    interpolates 4 dimensional cosmo data (time, z, lat, lon)
    on a abitrary timegrid
    
    args:
    :param data_4d: can be passed in 3 differents formats:
                    1. single 4d array (time, z, lat, lon)
                    2. list of 4d arrays
                    3. dictionary with single 4d array as value of each key
    :param times_old: current time grid of the cosmo data in form of integer
                      or float values e.g. epoch time
    :param times_new: new time grid the current cosmo data will be interpolated on
                      need to be in the same time unit 'like times_old'
    :param method: keyword argument describing the method used for interpolation
                   default is 'linear', for options see scipy.interpolate.interp1d
                   and the keyord 'kind'
    
    out:
    :return: interpolated values along the timeaxis
             outputformat is same as input format

    """
    if isclose(times_old[-1], times_new[-1]) and times_old[-1] < times_new[-1]:
        times_new_cor = times_new[:-1]
        append_last = True
        
    elif isclose(times_old[-1], times_new[-1]) == False and times_old[-1] < times_new[-1]:
        times_new_cor = times_new[:-1]
        append_last = False
    
    else:
        append_last = False
        times_new_cor = np.copy(times_new)
    
    if type(data_4d) == np.ndarray:
        f = interp1d(times_old, data_4d, kind=method, axis=0)
        out = f(times_new_cor)
        if append_last:
            out = np.append(out, data_4d[-1, np.newaxis,...], axis=0)
        return out
    
    elif type(data_4d) == list:
        out_list = []
        
        for arr in data_4d:
            f = interp1d(times_old, arr, kind=method, axis=0)
            out = f(times_new_cor)
            if append_last:
                out = np.append(out, arr[-1, np.newaxis,...], axis=0)
            out_list.append(out)
        
        return out_list
    
    elif type(data_4d) == dict:
        out_dict = {}
        
        for key in data_4d:
            f = interp1d(times_old, data_4d[key], axis=0, kind=method)
            out = f(times_new_cor)
            if append_last:
                out = np.append(out,data_4d[key][-1, np.newaxis,...], axis=0)
            out_dict.update({key:out})
        
        return out_dict
    
    else:
        print("ERROR in function 'interpolate_profiles_in_time'")
        print(("type(data_4d)= '{}' is not supported".format(type(data_4d))))
        print("program exits")
        quit()
        

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


def filter_times(time, timeunit):
    """
    if timesteps do not belong to the considered day they are
    removed from the original time array
    
    args:
    :param time: array or list of integers or float representing
                 time, eg. seconds since 1970-01-01 (epochtime)
    :param timeunit: string, describing the unit of the time array
                     eg. 'seconds since 1970-01-01' for epochtime
    out:
    :return: filtered time array
    """
    
    ref_time = nc4.num2date(np.mean(time), timeunit, only_use_cftime_datetimes=False)
    time_filt = [t for t in time if nc4.num2date(t, timeunit, only_use_cftime_datetimes=False).strftime("%Y%m%d") == ref_time.strftime("%Y%m%d")]
    
    return time_filt


def find_equal_values_in_dict(lat_lon_gp):
    """
    find keys with equal values in a dictionary
    
    args:
    :param lat_lon_gp: dictionary with tuples of (lat, lon) as values
    
    out:
    :return: dictionary with keys of the form 'lat_lon'. Values are
             lists containing the keys of equal (lat, lon)
    """
    out_dict = {}
    for key in lat_lon_gp:
        temp = lat_lon_gp.copy()
        del temp[key]
        values = list(temp.values())
        if lat_lon_gp[key] in values:
            new_key = "{}_{}".format(lat_lon_gp[key][0], lat_lon_gp[key][1])
            out_dict.update({new_key:[key]})
            for key2 in temp:
                if temp[key2] == lat_lon_gp[key]:
                    out_dict[new_key].append(key2)
    
    return out_dict
       
def get_cosmo_filenames(path, dates):
    """
    get cosmo filenames form dates
    """
    if type(dates[0]) == list:
        dates = create_daterange(dates)
    
    dates_str = [datetime.datetime.strftime(d, "%Y%m%d") for d in dates] 

    files = list_files(path)
    files_good = []

    for d in dates_str:
        for f in files:
            if d in f:
                files_good.append(f)
                files.remove(f)

    return files_good



#######################################################################
###   file and dirname generation, setup of dir/file infrastructure ###
#######################################################################
        

def generate_file_and_dir_names(lat_lon_gp, time, timeunit, filetype="atmofile"):
    """
    generate names for main directories (names contain lat lon position of COSMO gridpoint), 
    sub-directories (names contain date of each dataset) and atmosphere/radiosonde files
    (names contain UTC time of esch timestep)
    
    args:
    :param lat_lon_gp: dictionary which contains cosmo grid point(s) as  (lat, lon) tuples as values 
    :param time: list or array which contains the times of each cosmo profile as a float or integer 
                 number (eg. epoch time)
    :param timeunit: string containing the unit of the time array (e.g. 'seconds since 1970-01-01')
    :param filetype: string, controls the generated filename to identify the file content by the filename. 
                     Options are 'radiosonde' and 'atmofile'.
    
    :out
    :return main_dir: dictionary with names (strings) of the main directory as values containing 
                      the lat and lon position of the cosmo gridpoint to a precision of 2 digits. 
    :return sub_dir: string containing the date of the cosmo dataset in the form YYYYmmdd eg. 20160810
    :return fnames: list of strings with filenames for each timestep in the cosmo dataset.
                    The filename contains the timestep in the form HHMMSS eg. 14:23:14 --> 142314
    """

    main_dir_dict = {}

    for key in lat_lon_gp:
        lat_lon = lat_lon_gp[key]
        lat = float(lat_lon[0])
        lon = float(lat_lon[1])
        
        main_dir = "lat_{0:.2f}_lon_{1:.2f}".format(lat, lon)
        main_dir_dict.update({key:main_dir})
        
    time_filt = filter_times(time, timeunit)
    time_datetime = nc4.num2date(sorted(time_filt), timeunit, only_use_cftime_datetimes=False)
   
    sub_dir = datetime.datetime.strftime(time_datetime[0], "%Y%m%d") 
    if filetype == "radiosonde":
        fnames = ["{}_radiosonde.dat".format(datetime.datetime.strftime(t, "%H%M%S")) for t in time_datetime]
    elif filetype == "atmofile":
        fnames = ["{}_atmofile.dat".format(datetime.datetime.strftime(t, "%H%M%S")) for t in time_datetime]
    
    else: 
        print("ERROR in function 'generate_file_and_dir_names'")
        print(("passed argument for 'filetype' is {} which is no valid input".format(filetype)))
        print("chose 'radiosonde' or 'atmofile'")
        print("program exits")
        quit()
    
    return (main_dir_dict, sub_dir, fnames)
       

def check_existing_files_and_dirs(save_path, lat_lon_gp, time, timeunit, filetype="atmofile"):
    """
    args:
    :param save_path: string with path to the directory where the main directories of   
                      the folder structure will be generated
    :param lat_lon_gp: dictionary, keys are  values are (lat, lon) position of the pv station
    :param time: list or array which contains the times of each cosmo profile as a float or integer 
                 number (eg. epoch time)
    :param timeunit: string containingthe unit of the time array (e.g. 'seconds since 1970-01-01')
    :param filetype:
    """
    # 1. get dir and filenames
    main_dirs, sub_dir, fnames = generate_file_and_dir_names(lat_lon_gp, time, timeunit, filetype=filetype)
    
    # 2. check for existing main_dir and if main_dir exists, for existing subdir and fnames
    dirs = list_dirs(save_path)
    
    mdir_exist = {}
    sdir_exist = {}
    files_exist = {}
    for key in main_dirs:
        files_exist.update({key:[]})
        if main_dirs[key] in dirs:
            mdir_exist.update({key:True})
            sdirs = list_dirs(os.path.join(save_path, main_dirs[key]))
            if sub_dir in sdirs:
                sdir_exist.update({key:True})
                files = list_files(os.path.join(save_path, main_dirs[key], sub_dir))
                for f in fnames:
                    if f in files:
                        files_exist[key].append(True)
                    else:
                        files_exist[key].append(False)
                
            else:
                sdir_exist.update({key:False})
                files_exist[key] = files_exist[key] + [False] * len(fnames)
                
            
        else:
            mdir_exist.update({key:False})
            sdir_exist.update({key:False})
            files_exist[key] = files_exist[key] + [False] * len(fnames)
        
    return mdir_exist, sdir_exist, files_exist 


def check_write_lat_lon_to_known_stations(path, lat_lon, name):
    """
    checks if lat lon a of certain station is already listed in the 
    'known_stations.dat' file. In case the file doesnt exists,
    it is created
    
    args:
    :param path: string with path to the directory where the sub- 
                 directories of the folder structure are located
    :param lat_lon: tuple, array or list of lat and lon position 
                    of a specific station
    :param name: str, (unique) name or ID of location. E.g. name
                 of  measurement site.
    """
    fname = "known_stations.dat"
    header = "name, latitude and longitude of solar power plants with the same nearest COSMOgridpoint\nname, lat, lon"
    
    files = list_files(path)
    if fname not in files:
        data = np.array([(name, lat_lon[0], lat_lon[1])], dtype=[("name", 'U6'), ("lat", "float32"), ("lon", "float32")])
        np.savetxt(os.path.join(path,fname), data, fmt=["%s" , "%.5f", "%.5f"], header=header)
    
    else:
        with open(os.path.join(path, fname)) as f:
            lines = [line.strip() for line in f.readlines() if line[0] != "#"]
        # check if empty
        if len(lines) == 0:
            data = np.array([(name, lat_lon[0], lat_lon[1])], dtype=[("name", 'U6'), ("lat", "float32"), ("lon", "float32")])
            np.savetxt(os.path.join(path,fname), data, fmt=["%s" , "%.5f", "%.5f"], header=header)
        
        # if not empty
        elif len(lines) != 0:
            #try:
            ds = np.genfromtxt(os.path.join(path, fname), dtype=[("name", 'U6'), ("lat", "float32"), ("lon", "float32")], encoding=None)
            
            names = np.array([ds["name"]])
            lat   = np.array([ds["lat"]])
            lon   = np.array([ds["lon"]])
            
            if len(names.shape) == 1:
                names = list(names)
                lat   = list(lat)
                lon   = list(lon)

            elif len(names.shape) == 2:
                names = list(names.squeeze())
                lat   = list(lat.squeeze())
                lon   = list(lon.squeeze())

            lat_lon_tup  = tuple([np.round(x, decimals=5) for x in lat_lon])
            file_lat_lon = list(zip(np.round(np.array(lat), decimals=5), np.round(np.array(lon), decimals=5)))
          
            abs_dev = 1e-5
            for loc in file_lat_lon:
                lat_ref = loc[0]
                lon_ref = loc[1]
                
                latlon_exist = all([isclose(lat_ref, lat_lon_tup[0], abs_dev=abs_dev), isclose(lon_ref, lat_lon_tup[1], abs_dev=abs_dev)])
                
                if latlon_exist:
                    break
                
            if name in names and latlon_exist:
                print(("---"*20))
                print("INFO from function check_write_lat_lon_to_known_stations():")
                print(("location with name {}, lat {:.5f} and lon {:.5f} is already in known_stations!".format(name, lat_lon[0], lat_lon[1])))
                print(("---"*20))

            elif name not in names and not latlon_exist:

                names.append(name) 
                lat.append(lat_lon[0])
                lon.append(lat_lon[1])

                data = np.array(list(zip(names, lat, lon)), dtype=[("name", 'U6'), ("lat", "float32"), ("lon", "float32")])
                np.savetxt(os.path.join(path, fname), data, fmt=["%s", "%.5f", "%.5f"], header=header)

            else:
                print(("---"*20))
                print("ERROR function check_write_lat_lon_to_known_stations():")
                print(("location with name {}, lat {:.5f} and lon {:.5f}  already exists".format(name, lat_lon[0], lat_lon[1])))
                print("but name and position do not match!!")
                print("program exits...")
                print(("---"*20))
                quit()
                    
            #except Exception as e:
               #print "ERROR in check_write_lat_lon_to_known_stations: ", e
               #raise e

               #if str(e) == 'need more than 0 values to unpack':
                   #data = np.array([(name, lat_lon[0], lat_lon[1])], dtype=[("name", '|S50'), ("lat", "float32"), ("lon", "float32")])
                   #np.savetxt(os.path.join(path,fname), data, fmt=["%s" , "%.5f", "%.5f"], header=header)
                

#######################################################################
###       specific functions to create and save radiosondfiles      ###
#######################################################################


def write_radiosondefile(path, fnames, data_dict, lat_lon_idx, time_idx):
    """
    creates radiosond profiles for all required timesteps and saves them
    to a textfile
    
    args:
    :param path: string, full path to the directory where the files are saved 
    :param fnames: list'of filenames (only files missing in the directory)
    :param data_dict: dictionary with attmospheric parameters as values. Temperature (T)
                      and pressure (P) are required as input. Other parameters like humidity
                      or other trace gas profiles can also be contained in the dictionary.
                      The values must have the dimensions (time, zlev, lat, lon).
    :param lat_lon_idx: tuple with the latitue and longitude index which will be selected
                        from each 4d value of the data_dict.
    :param time_idx: list with index of timesteps which will be selected from the data_dict values
    
    """
    
    lat, lon = lat_lon_idx
    for t in time_idx:
        nlev = list(data_dict.values())[0].shape[1]
        save_data = np.zeros((nlev, 2))
        for key in data_dict:
            if key == "P":
                save_data[:,0] = data_dict[key][t,:,lat,lon]
            elif key == "T":
                save_data[:,1] = data_dict[key][t,:,lat,lon]
            else:
                save_data = np.hstack((save_data, data_dict[key][t,:,lat,lon].reshape(nlev,1)))
        
        np.savetxt(os.path.join(path, fnames[t]), save_data)
 
 
 

def save_profiles(data_dict, save_path, idx_dict, lat_lon_gp, stations_lat_lon, time, timeunit):
    """
    creates main and sub directories and writes radiosond files
    
    args:
    :param data_dict: dictionary containing the COSMO variables which will be written to the 
                      radiosonde file. The values of the dictionary must be 4 dimensional
                      with de dimensions (time, z, lat, lon)
    :param save_path: string, path to the directory where the main folders will be created
    :param idx_dict: dictionary, keys represent the different stations and the values are
                     the indices of latitude and longitude
    :param lat_lon_gp: dictionary, keys represent the different stations and values are
                       tuples of (lat, lon) with the position of the nearest COSMO grid point
                       (Note: variable probably unnecessary)
    :param time: array of COSMO timesteps (or interpolated timesteps), the timeunit hast
                 to be a integer or floating point number
    :param timeunit: string, timeunit of the time array, eg. 'seconds since 1970-01-01' (epoch)
    """
    # 1. check for existing files and dirs
    mdir_exist, sdir_exist, files_exist = check_existing_files_and_dirs(save_path, lat_lon_gp, time, timeunit, filetype="radiosonde")
    
    # 2. generate file and dir names
    mdir_namedict, sub_dir, filename_lst = generate_file_and_dir_names(lat_lon_gp, time, timeunit, filetype="radiosonde")
    val_set = []
    
    for key in lat_lon_gp:
        if lat_lon_gp[key] in val_set:
            check_write_lat_lon_to_known_stations(os.path.join(save_path, mdir_namedict[key]), stations_lat_lon[key], key)
        
        else:
            val_set.append(lat_lon_gp[key])
            if mdir_exist[key]:
                check_write_lat_lon_to_known_stations(os.path.join(save_path, mdir_namedict[key]), stations_lat_lon[key], key)
                if sdir_exist[key]:
                    if any(files_exist[key]):
                        time_idx = list(itertools.compress(list(range(len(files_exist[key]))), files_exist[key]))
                        filenames_sel = list(filename_lst[i] for i in time_idx)
                        path = os.path.join(save_path, mdir_namedict[key], sub_dir)
                        write_radiosondefile(path, filenames_sel, data_dict, idx_dict[key], time_idx)
                    else:
                        time_idx = list(range(len(files_exist[key])))
                        path = os.path.join(save_path, mdir_namedict[key], sub_dir)
                        write_radiosondefile(path, filename_lst, data_dict, idx_dict[key], time_idx)
                        
                        
                else:
                    os.mkdir(os.path.join(save_path, mdir_namedict[key], sub_dir))
                    time_idx = list(range(len(files_exist[key])))
                    path = os.path.join(save_path, mdir_namedict[key], sub_dir)
                    write_radiosondefile(path, filename_lst, data_dict, idx_dict[key], time_idx)
            else:
                os.mkdir(os.path.join(save_path,mdir_namedict[key]))
                check_write_lat_lon_to_known_stations(os.path.join(save_path, mdir_namedict[key]), stations_lat_lon[key], key)
                os.mkdir(os.path.join(save_path, mdir_namedict[key], sub_dir))
                time_idx = list(range(len(files_exist[key])))
                path = os.path.join(save_path, mdir_namedict[key], sub_dir)
                write_radiosondefile(path, filename_lst, data_dict, idx_dict[key], time_idx)
                

def create_radiosonde_profiles(path_cosmo, fname_cosmo, varlist, stations_lat_lon, save_path, timeres=None):
    """
    'main' function calling all necessary functions to create and save radiosonde files
    
    args:
    :param path_cosmo: string, absolute or relative path to directory of the netCDF COSMO files
    :param fname_cosmo: string, full filename of the COSMO netCDF file
    :param varlist: list of strings, names of variables and dimensionsin COSMO netCDF file 
    :param stations_lat_lon: dictionary, keys represent the different solar power plants,
                             values are tuples of their position in (lat,lon) coordinates
    :param save_path: string, path to the directory where the Radiosonde files and 
                      the according folder structure will be saved
    :param timeres: float, desired time resolution of the COSMO data in minutes, 
                    default value is 'None' which means that the original time 
                    resolution of the COSMO data is used (1 hour)
    
    """
    
    # 1. load data
    print("load data ...")
    data_dict = load_data(path_cosmo, fname_cosmo, varlist)
    #print data_dict
    # 2. combine surface and level data
    print("combine COSMO surface and level data...")
    comb_dict = combine_lyr_sfc_properties_4d(data_dict["humidity_lev"], data_dict["Temperature_lev"], 
                                                data_dict["pressure_lev"], data_dict["humidity_sfc"], 
                                                data_dict["Temperature_sfc"], data_dict["pressure_sfc"])
   
    # 3. extract station gridpoints
    print("find nearest neighbour to station gridpoints...")
    idx_nearest_gp, lat_lon_nearest_gp = find_nearest_cosmo_gridpoint(data_dict["lat"], data_dict["lon"], stations_lat_lon)
 
    # 4. option: change time resolution
    if timeres != None:
        print(("interpolate COSMO data to higher time resolution ({} minutes)...".format(timeres)))
        new_times = generate_new_time_array(data_dict["time"], data_dict["timeunit"], timeres)
        interp_dict = interpolate_profiles_in_time(comb_dict, data_dict["time"], new_times)
        
        print("save profiles...")
        save_profiles(interp_dict, save_path, idx_nearest_gp, lat_lon_nearest_gp, stations_lat_lon, new_times, data_dict["timeunit"])

    else:
        print("save profiles...")
        save_profiles(comb_dict, save_path, idx_nearest_gp, lat_lon_nearest_gp, stations_lat_lon, data_dict["time"], data_dict["timeunit"])
        
        
        
        
        

#######################################################################
###       specific functions to create and save atmosphere files    ###
#######################################################################

def load_libradtran_atmosphere_file(path, fname, varnames, get_header=True):
    """
    loads libradtran standars atmosphere file (textfile) to dictioary
    of numpy arrays
    
    args:
    :param path: string, path to atmosphere file
    :param fname: string, filename of atmosphere file
    :param varnames: list of strings, keys which are assigned 
                     to each column array in the output dictionary. 
                     Order of varnames has to match the order of 
                     the variables in the atmosphere file along 
                     the 1st axis (columns) to receive correct results.
    :param get_header: bool, if True the header of the atmosphere file 
                       is extracted and returned. Default value is True.
                       
    out:
    :return vardict: dictionary, keys are the description of the variable
                     values are the column arrays from the standard 
                     atmosphere file.
    :return header: string, header of the atmosphere file. Will only be 
                    returned if 'get_header' is True
    """
    data = np.loadtxt(os.path.join(path,fname))
    
    nlev, ncol = data.shape
    vardict = {}
    
    for i in range(ncol):
        vardict.update({varnames[i]: data[:,i]})
   
    if get_header:
        with open(os.path.join(path, fname)) as f:
            header_lines = [line.strip() for line in f.readlines() if line[0] == "#"]

        header = "{}\n{}".format(header_lines[0][2:], header_lines[1][2:])
        
        return vardict, header
    
    else:
        return vardict


def cosmo_zlev_2_zlyr(z_lev):
    """
    Calculates COSMO layer center altitude from level altitude.
    The lowes altitude of the output array is surface altitude.
    
    args:
    :param z_lev: array, altitude of COSMO levels
    
    out:
    :return: array, COSMO altitude in the center of each layer, 
             where most of the variables like pressure and 
             temperature are defined. The surface altitude is 
             included in the layer altitude array
    """
    
    z_0 = z_lev[-1,...]
    z_i =  z_lev[:-1,...]/2. + z_lev[1:,...]/2.
    z_lyr = np.append(z_i, z_0[np.newaxis, ...], axis=0) 
    
    return z_lyr


def reduce_atmosphere_zresolution(profiles):
    """
    reduce vertical resoloution of atmosphere to 
    speed up Mystic calculation. Column concentration will 
    be kept equal through comparing the integrad column 
    density of original and coarser resolution profiles
    
    :param profiles: 2d array, high resoloution
                     atmosphere with all species
                     
    :return: profiles interpolated on coarser zgrid
    """
    
    # get z and p
    z_old = profiles[:,0]
    p_old = profiles[:,1]
    # znew
    z_new = np.array([ 0. ,  0.25,  0.5,  0.75,  1. ,  1.5,  2. ,  2.5,  3. ,  3.5,
        4. ,  4.5,  5. ,  6. ,  7. ,  8. ,  9. , 10. , 11. , 12. , 13. ,
       14. , 15. , 16. , 17. , 18. , 19. , 20. , 21. , 22. , 23. , 24. ,
       25. , 27.5, 30. , 32.5, 35. , 37.5, 40. , 42.5, 45. , 47.5, 50. ,
       55. , 60. , 65. , 70. , 75. , 80. ])[::-1]
    # calculate new profiles
    
    f = interp1d(z_old, p_old, kind="quadratic")
    p_new =  f(z_new)
    
    # create new profile array
    new_profiles = np.zeros((len(z_new), profiles.shape[1]))
    
    new_profiles[:,0] = z_new
    new_profiles[:,1] = p_new
    
    # interpolate others
    for i in range(2,profiles.shape[1]):        
        var_old = profiles[:,i]

        f = interp1d(p_old, var_old, kind="linear") 
        var_new = f(p_new)
        
        if i >= 3:
            int_old = np.trapz(var_old[::-1], x=z_old[::-1]*1e5)
            int_new = np.trapz(var_new[::-1], x=z_new[::-1]*1e5)
            var_new = var_new * int_old/int_new
        
        new_profiles[:,i] = var_new
    
    return new_profiles
    

    
def new_atmosphere_from_cosmo_data(p_cosmo, T_cosmo, z_cosmo, n_h2o_cosmo, vardict, reduce_zres=False,
                                   mol_modify_O3=None, co2=404.98, interp_method="linear"):
    """
    merges cosmo data and standard atmosphere data, variables which are not provided by COSMO 
    are taken from the standard atmosphere and are interpolated on onto the merged pressure grid
    
    args:
    :param p_cosmo: array, combinde surface and level pressure in [Pa] from COSMO
    :param T_cosmo: array, combinde surface and level temperature in [K] from COSMO
    :param z_cosmo: array, layer heights including surface in [m above sea level] from COSMO
    :param n_h2o_cosmo: array, number density of water vapor in [#/cm3] from COSMO
    :param vardict: dictionary, keys are the variables contained in us standard atmosphere, 
                    values are the original profiles of the atmosphere file
    :param mol_modify_O3: None or float, sets the ozone column abundance in DU (Dopson Units).
                          To convert column number density to DU, T=273.25 K and p=101325 Pa
                          are used as standard temperature and standard pressure. 
                          If value is None (default) the concentration in the standard atmosphere is used.
                          Note: through interpolation on the COSMO pressure grid ozone column 
                          will be generally lower than in the original us standard atmosphere 
                          as p0_std > p0_cosmo (dDU ~ 2-3 DU)!!
    :param co2: None or float, concentration of CO2 in [ppmv], if None the concentration of the standard
                atmosphere will be used (330 ppmv). Default is 404.98 ppmv which is the global average 
                annual mean CO2 concentration of 2017 from NOAA/ESRL
                (source: ftp://aftp.cmdl.noaa.gov/products/trends/co2/co2_annmean_gl.txt)
                (accessed: 28.06.2018)
    :param interp_method: string, default is linear, for all options see python documentation of
                          the scipy function 'interp1d'
    
    out:
    :return: array, new profiles in same order like in US standard atmosphere file. 
             Therefore the output array can directly be written to a new atmosphere file.
    
    """
    
    # 1. write elements from dictonary to variables and convert units if necessary
    
    p_sta     = vardict["p"] * 100. # convert to pascal
    z_sta     = vardict["z"] * 1000. # convert to meters 
    T_sta     = vardict["T"]
    n_air_sta = vardict["air"]
    n_o2_sta  = vardict["O2"]
    n_o3_sta  = vardict["O3"]
    n_no2_sta = vardict["NO2"]
    n_co2_sta = vardict["CO2"]
    n_h2o_sta = vardict["H2O"]
    
    # calculate number density of air for new p and T profiles
    n_air_cosmo = ideal_gas_p2numdens(p_cosmo, T_cosmo)
    
    # calculate concentration of O2 using the ratio of o2-USstd/air-USstd
    # assume mixing ratio is constant from sfc to zmax_comso
    n_o2_cosmo = vardict["O2"][-1]/vardict["air"][-1] * n_air_cosmo
    
    # subtract lowest cosmo level to start at altitude of 0m
    z_cosmo_norm = z_cosmo - np.min(z_cosmo)
    
    # extract preliminary top part of standard atmosphere  
    p_top_sta = p_sta[p_sta<np.min(p_cosmo)]
    
    # extract preliminary border index of the top part of standard atmosphere
    idx = np.argmin(np.abs(p_sta-np.max(p_top_sta)))
    
    if z_sta[idx] > np.max(z_cosmo_norm):
        z_top_sta = z_sta[p_sta < np.min(p_cosmo)]
        T_top_sta = T_sta[p_sta < np.min(p_cosmo)]
        n_air_top_sta = n_air_sta[p_sta < np.min(p_cosmo)]
        n_o2_top_sta  = n_o2_sta[p_sta < np.min(p_cosmo)]
        n_h2o_top_sta = n_h2o_sta[p_sta < np.min(p_cosmo)]
        
        p_merge = np.append(p_top_sta, p_cosmo) # unit is pascal
        z_merge = np.append(z_top_sta, z_cosmo_norm) # unit is meters
        T_merge = np.append(T_top_sta, T_cosmo) # unit is Kelvin
        n_air_merge = np.append(n_air_top_sta, n_air_cosmo)
        n_o2_merge  = np.append(n_o2_top_sta, n_o2_cosmo)
        n_h2o_merge = np.append(n_h2o_top_sta, n_h2o_cosmo)
        
    else:
        z_top_sta = z_sta[p_sta < p_top_sta[-1]]
        T_top_sta = T_sta[p_sta < p_top_sta[-1]]
        n_air_top_sta = n_air_sta[p_sta < p_top_sta[-1]]
        n_o2_top_sta  = n_o2_sta[p_sta < p_top_sta[-1]]
        n_h2o_top_sta  = n_h2o_sta[p_sta < p_top_sta[-1]]
       
        p_merge = np.append(p_top_sta[:-1], p_cosmo) # unit is pascal
        z_merge = np.append(z_top_sta, z_cosmo_norm) # unit is meters
        T_merge = np.append(T_top_sta, T_cosmo) # unit is Kelvin 
        n_air_merge = np.append(n_air_top_sta, n_air_cosmo)
        n_o2_merge = np.append(n_o2_top_sta, n_o2_cosmo) 
        n_h2o_merge = np.append(n_h2o_top_sta, n_h2o_cosmo) 
    
    # interpolate O3 and NO2 onto new pressure grid 
    f = interp1d(p_sta, n_o3_sta, kind=interp_method, bounds_error=False, fill_value=(n_o3_sta[0], n_o3_sta[-1]))
    n_o3_merge = f(p_merge)
    
    # if O3 needs to be modified
    if mol_modify_O3 != None:
        int_o3 = np.trapz(n_o3_merge[::-1],z_merge[::-1]*1e5) # convert zgrid from km to cm 
        dz_o3 = int_o3 * k_boltzmann * 273.15 * 1e6 / 101325. # dz in cm
        DU_o3 = dz_o3 * 1e3 # cm to Dobson Units (DU)
        
        scaling = mol_modify_O3 / DU_o3
        n_o3_merge = n_o3_merge * scaling # scaling the profile by a single factor changes the integral accordingly

    f = interp1d(p_sta, n_no2_sta, kind=interp_method, bounds_error=False, fill_value=(n_no2_sta[0], n_no2_sta[-1]))
    n_no2_merge = f(p_merge)
    
    # calculate new co2 number density, take care of vertical profile
    f = interp1d(p_sta, n_co2_sta, kind=interp_method, bounds_error=False, fill_value=(n_co2_sta[0], n_co2_sta[-1]))
    n_co2_merge = f(p_merge)
    
    if co2 != None:
        co2_sfc_ppm = n_co2_sta[-1] / n_air_sta[-1] * 1e6
        vert_scaling = np.around(n_co2_sta / n_air_sta / co2_sfc_ppm * 1e6, decimals=2)
    
        f = interp1d(p_sta, vert_scaling, bounds_error=False, fill_value=(vert_scaling[0], vert_scaling[-1]))
        vert_scaling_merge = f(p_merge)
        n_co2_merge = co2*1e-6 * n_air_merge * vert_scaling_merge
        
    # write new profiles to 2d array
    new_profiles = np.zeros((len(z_merge), len(vardict)))
    
    new_profiles[:,0] = z_merge/1000. # convert back to km
    new_profiles[:,1] = p_merge/100.  # convert back to hPa/mbar
    new_profiles[:,2] = T_merge
    new_profiles[:,3] = n_air_merge
    new_profiles[:,4] = n_o3_merge
    new_profiles[:,5] = n_o2_merge
    new_profiles[:,6] = n_h2o_merge
    new_profiles[:,7] = n_co2_merge 
    new_profiles[:,8] = n_no2_merge
    
    if reduce_zres:
        new_profiles = reduce_atmosphere_zresolution(new_profiles)
    
    return new_profiles


def write_atmofile(path, fnames, data_dict, z_cosmo, std_atmo_dict, lat_lon_idx, 
                   time_idx, header, reduce_zres=False, mol_modify_O3=None):
    """
    writes atmosphere text-files created from standars atmosphere, COSMO data and concentrations
    scaling of tracegases (CO2 ad O3) for all required timesteps.
    
    args:
    :param path: string, full path to the directory where the files are saved 
    :param fnames: list of filenames (only files missing in the directory)
    :param data_dict: dictionary with atmospheric parameters: temperature (T in [K]), 
                      pressure (P in [Pa]) and water vapor number density [n_h2o in #/cm3]
                      The values must have the dimensions (time, zlev, lat, lon).
    :param z_cosmo: array, layer heights including surface in [m above sea level] from COSMO
    :param std_atmo_dict: dictionary of all profiles contained in the US standard atmosphere file,
                          including z, p and T.
    :param lat_lon_idx: tuple with the latitue and longitude index which will be selected
                        from each 4d value of the data_dict.
    :param time_idx: list with indices of timesteps which will be selected from the data_dict values
    :param header: string, contains the header of the standard atmosphere file
    :param mol_modify_O3: None or float, sets the ozone column abundance in DU (Dopson Units).
                          To convert column number density to DU, T=273.25 K and p=101325 Pa
                          are used as standard temperature and standard pressure. 
                          If value is None (default) the concentration in the standard atmosphere is used.
                          Note: through interpolation on the COSMO pressure grid ozone column 
                          will be generally lower than in the original us standard atmosphere 
                          as p0_std > p0_cosmo (dDU ~ 2-3 DU)!!
    
    """
    fmt = "%11.3f %12.5f %10.3f %.6E %.6E %.6E %.6E %.6E %.6E"
    if mol_modify_O3 != None:
        hdr_add1 = "atmosphere file with modified pressure and temperature from COSMO, modified CO2 (404.98 ppmv), modified O3 ({} DU)".format(mol_modify_O3)
    else:
        hdr_add1 = "atmosphere file with modified pressure and temperature from COSMO and modified CO2 (404.98 ppmv)"
    
    hdr_add2 = "other trace gas concentrations were interpolated onto the COSMO pressure levels"
    
    header_new = "{}\n{}\n{}".format(hdr_add1, hdr_add2, header)
    
    lat, lon = lat_lon_idx

    for t, fname in zip(time_idx,fnames):
        p_cosmo = data_dict["P"][t,:,lat,lon]
        T_cosmo = data_dict["T"][t,:,lat,lon]
        n_h2o_cosmo = data_dict["n_h2o"][t,:,lat,lon]
        z_cosmo_gp = z_cosmo[:,lat,lon] 
        save_data = new_atmosphere_from_cosmo_data(p_cosmo, T_cosmo, z_cosmo_gp, n_h2o_cosmo, std_atmo_dict, 
                                                   reduce_zres=reduce_zres, mol_modify_O3=mol_modify_O3)

        np.savetxt(os.path.join(path, fname), save_data, header=header_new, fmt=fmt)
 
    
    
    
def save_atmofiles(save_path, data_dict, std_atmo_dict, z_cosmo, idx_dict, lat_lon_gp, stations_lat_lon, 
                    time, timeunit, header, reduce_zres=False, mol_modify_O3=None):
    """
    creates main and sub directories and writes radiosond files
    
    args:
    :param data_dict: dictionary containing the COSMO variables which will be written to the 
                      radiosonde file. The values of the dictionary must be 4 dimensional
                      with de dimensions (time, z, lat, lon)
    :param save_path: string, path to the directory where the main folders will be created
    :param idx_dict: dictionary, keys represent the different stations and the values are
                     the indices of latitude and longitude
    :param lat_lon_gp: dictionary, keys represent the different stations and values are
                       tuples of (lat, lon) with the position of the nearest COSMO grid point
                       (Note: variable probably unnecessary)
    :param time: array of COSMO timesteps (or interpolated timesteps), the timeunit hast
                 to be a integer or floating point number
    :param timeunit: string, timeunit of the time array, eg. 'seconds since 1970-01-01' (epoch)
    :param header: string, contains the header of the standard atmosphere file
    :param mol_modify_O3: None or float, sets the ozone column abundance in DU (Dopson Units).
                          To convert column number density to DU, T=273.25 K and p=101325 Pa
                          are used as standard temperature and standard pressure. 
                          If value is None (default) the concentration in the standard atmosphere is used.
                          Note: through interpolation on the COSMO pressure grid ozone column 
                          will be generally lower than in the original us standard atmosphere 
                          as p0_std > p0_cosmo (dDU ~ 2-3 DU)!!
    """
    # 1. check for existing files and dirs
    mdir_exist, sdir_exist, files_exist = check_existing_files_and_dirs(save_path, lat_lon_gp, time, timeunit, filetype="atmofile")

    # 2. generate file and dir names
    mdir_namedict, sub_dir, filename_lst = generate_file_and_dir_names(lat_lon_gp, time, timeunit, filetype="atmofile")
    val_set = []
    
    for key in lat_lon_gp:
        if lat_lon_gp[key] in val_set:
            check_write_lat_lon_to_known_stations(os.path.join(save_path, mdir_namedict[key]), stations_lat_lon[key], key)
        
        else:
            val_set.append(lat_lon_gp[key])
            if mdir_exist[key]:
                check_write_lat_lon_to_known_stations(os.path.join(save_path, mdir_namedict[key]), stations_lat_lon[key], key)
                if sdir_exist[key]:
                    if any(files_exist[key]):
                        time_idx = list(itertools.compress(list(range(len(files_exist[key]))), np.invert(files_exist[key])))
                        filenames_sel = list(filename_lst[i] for i in time_idx)
                        path = os.path.join(save_path, mdir_namedict[key], sub_dir)
                        write_atmofile(path, filenames_sel, data_dict, z_cosmo, std_atmo_dict, idx_dict[key], time_idx, 
                                       header, reduce_zres=reduce_zres, mol_modify_O3=mol_modify_O3)
                    else:
                        time_idx = list(range(len(files_exist[key])))
                        path = os.path.join(save_path, mdir_namedict[key], sub_dir)
                        write_atmofile(path, filename_lst, data_dict, z_cosmo, std_atmo_dict, idx_dict[key], time_idx, 
                                       header, reduce_zres=reduce_zres, mol_modify_O3=mol_modify_O3)
                else:
                    os.mkdir(os.path.join(save_path, mdir_namedict[key], sub_dir))
                    time_idx = list(range(len(files_exist[key])))
                    path = os.path.join(save_path, mdir_namedict[key], sub_dir)
                    write_atmofile(path, filename_lst, data_dict, z_cosmo, std_atmo_dict, idx_dict[key], time_idx, 
                                   header, reduce_zres=reduce_zres, mol_modify_O3=mol_modify_O3)
            else:
                os.mkdir(os.path.join(save_path,mdir_namedict[key]))
                check_write_lat_lon_to_known_stations(os.path.join(save_path, mdir_namedict[key]), stations_lat_lon[key], key)
                os.mkdir(os.path.join(save_path, mdir_namedict[key], sub_dir))
                time_idx = list(range(len(files_exist[key])))
                path = os.path.join(save_path, mdir_namedict[key], sub_dir)
                write_atmofile(path, filename_lst, data_dict, z_cosmo, std_atmo_dict, idx_dict[key], time_idx, 
                               header, reduce_zres=reduce_zres, mol_modify_O3=mol_modify_O3)



def create_atmosphere_files(path_cosmo, fname_cosmo, path_std_atmo, fname_std_atmo,  
                            varlist_cosmo, varlist_std_atmo, stations_lat_lon, save_path, 
                            reduce_zres=False, timeres=None, mol_modify_O3=None):
    """
    Create libradtran atmosphere files based on US Standard Atmosphere (Anderson, 1976). The standard atmosphere
    is partly replaced by COSMO model output like temperature, pressure and humidity. The gas concentrations are 
    interpolated onto the new pressure levels of COSMO. CO2 is scaled to the current concentration of 404.98 ppmv (NOAA)
    and there is the option to modify the ozon column density. Furthermore the hourly COSMO output can be interpolated to 
    a higher time resolutionThe z-axis is for the COSMO (half-)levels is taken from HHL variable and the height of the 
    layer center where variables like temperature and humidity are defined is calculated as the mean height of adjacent levels. 
    Existing atmosphere files are not overwritten, only missing ones are added. If files schold be replaced they need to be 
    removed manually from their directory
    
    args:
    
    :param path_cosmo: string, absolute or relative path to directory
                       of the netCDF COSMO files
    :param fname_cosmo: string, full filename of the COSMO netCDF file
    :param path_std_atmo: string, path to atmosphere file
    :param fname_std_atmo: string, filename of atmosphere file
    :param varlist_cosmo: list of strings, names of variables and dimensions
                          in COSMO netCDF file
    :param varlist_std_atmo: list of strings, keys which are assigned 
                             to each column array in the output dictionary. 
                             Order of varnames has to match the order of 
                             the variables in the atmosphere file along 
                             the 1st axis (columns) to receive correct results.
    :param stations_lat_lon: dictionary, keys represent the different stations and values are
                             tuples of (lat, lon) with the position of the PV power plants
    :param save_path: string, path to the directory where the main folders will be created
    :param timeres: float, desired time resolution of the COSMO data in minutes, 
                    default value is 'None' which means that the original time 
                    resolution of the COSMO data is used (1 hour)
    :param mol_modify_O3: None or float, sets the ozone column abundance in DU (Dopson Units).
                          To convert column number density to DU, T=273.25 K and p=101325 Pa
                          are used as standard temperature and standard pressure. 
                          If value is None (default) the concentration in the standard atmosphere is used.
                          Note: through interpolation on the COSMO pressure grid ozone column 
                          will be generally lower than in the original us standard atmosphere 
                          as p0_std > p0_cosmo (dDU ~ 2-3 DU)!!
    """
    
    # 1. load data
    print(("load COSMO data file '{}'...".format(fname_cosmo)))
    data_dict = load_data(path_cosmo, fname_cosmo, varlist_cosmo)
    
    # 2. combine surface and level data
    print("combine COSMO surface and level data...")
    comb_dict = combine_lyr_sfc_properties_4d(data_dict["humidity_lev"], data_dict["Temperature_lev"], 
                                                data_dict["pressure_lev"], data_dict["humidity_sfc"], 
                                                data_dict["Temperature_sfc"], data_dict["pressure_sfc"])
   
    # 3. extract station gridpoints
    print("find nearest neighbour to station gridpoints...")
    idx_nearest_gp, lat_lon_nearest_gp = find_nearest_cosmo_gridpoint(data_dict["lat"], data_dict["lon"], stations_lat_lon)
    
    # 4. load afglus atmosphere
    print("load libradtran atmosphere file...")
    std_atmo_dict, header = load_libradtran_atmosphere_file(path_std_atmo, fname_std_atmo, varlist_std_atmo)
    
    # 5. convert COSMO z_levels (HHL) to z_layer
    z_cosmo = cosmo_zlev_2_zlyr(data_dict["level_alt"])

    # 6. option: change time resolution
    if timeres != None:
        print(("interpolate COSMO data to higher time resolution ({} minutes)...".format(timeres)))
        new_times = generate_new_time_array(data_dict["time"], data_dict["timeunit"], timeres)
        interp_dict = interpolate_profiles_in_time(comb_dict, data_dict["time"], new_times)
        
        print("saving profiles...")
        save_atmofiles(save_path, interp_dict, std_atmo_dict, z_cosmo, idx_nearest_gp, lat_lon_nearest_gp, 
                       stations_lat_lon, new_times, data_dict["timeunit"], header, reduce_zres=reduce_zres, mol_modify_O3=mol_modify_O3)
        
    else:
        print("saving profiles...")
        save_atmofiles(save_path, comb_dict, std_atmo_dict, z_cosmo, idx_nearest_gp, lat_lon_nearest_gp, 
                       stations_lat_lon, data_dict["time"], data_dict["timeunit"], header, reduce_zres=reduce_zres, mol_modify_O3=mol_modify_O3)
        


#######################################################################
###                         load config file                        ###
#######################################################################



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
            config = yaml.safe_load(ds)
        except yaml.YAMLError as exc:
            print(exc)
    return config  
    
def create_daterange(dates):
    """
    :param dates: list of 2 element list with
    tin and tend of datetrange. tin/tend are expected
    to be of type datetime.date or datetime.datetime.
    """
    datelist = []
    for tin, tend in dates:
        tin  = datetime.datetime(tin.year, tin.month, tin.day)
        tend = datetime.datetime(tend.year, tend.month, tend.day)
        dt = tend - tin
        drange = [tin+datetime.timedelta(days=i) for i in range(dt.days+1)]
        datelist += drange

    return sorted(datelist)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("configfile", help="yaml file containing config")
    args = parser.parse_args()

    config_filename = os.path.abspath(args.configfile)
     
    config = load_yaml_configfile(config_filename)
    
    output_filetype = config["output_filetype"]
     
    path_cosmo     = config["path_cosmo"]
    fnames_cosmo   = config["fnames_cosmo"]
     
    if fnames_cosmo == "from_dates":
        dates = config["dates"]
        fnames_cosmo = get_cosmo_filenames(path_cosmo, dates)
    
    vardict_cosmo  = config["variable_names_cosmo"]
    
    reduce_zres = config["reduce_vertical_resolution"]
    modify_O3 = config["modify_ozone"]
    
    if modify_O3 == False:
        modify_O3 = None
    
    #varlist_cosmo = config["variable_names_cosmo"].values()
    
    stations_lat_lon = config["stations_lat_lon"]
    for key in stations_lat_lon:
        stations_lat_lon[key] = tuple(stations_lat_lon[key])
    
    timeres = None if config["timeresolution"] == "None" else config["timeresolution"]
        
    if output_filetype == "radiosonde":
        save_path = config["path_radiosondefiles"]
    
        for fname in fnames_cosmo:
             create_radiosonde_profiles(path_cosmo, fname, vardict_cosmo, stations_lat_lon, save_path, timeres=timeres)
     
    elif output_filetype == "atmofile":
        save_path = config["path_atmofiles"]
       
        path_usstd  = config["path_usstd"]
        fname_usstd = config["fname_usstd"]
        variable_names_usstd = config["variable_names_usstd"]
    
        for fname in fnames_cosmo:
            create_atmosphere_files(path_cosmo, fname, path_usstd, fname_usstd,  
                                    vardict_cosmo, variable_names_usstd, stations_lat_lon, 
                                    save_path, reduce_zres=reduce_zres,  timeres=timeres,
                                    mol_modify_O3=modify_O3)
            
    else:
        print("ERROR in main():")
        print(("the passed argument for 'output_filetype' is {}".format(output_filetype)))
        print("arguments supported are: 'radiosonde' and 'atmofile'")
  
if __name__ == "__main__":
    main()

















