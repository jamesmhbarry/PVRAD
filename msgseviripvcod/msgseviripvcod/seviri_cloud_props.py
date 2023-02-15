
import os
import yaml
import numpy as np
import xarray as xr
import pyresample
import netCDF4 as nc4
import datetime

from contextlib import closing
from scipy.interpolate import interp1d


###############################################################
###   general functions to load and process (cosmo) data    ###
###############################################################

def load_data(path, fname, vardict, coordfile):
    """
    loads satellite data from netCDF file and returns 
    a dictionary of variables
    
    args:
    :param path: string, absolute or relative path to directory
                       of the netCDF files
    :param fname: string, full filename of the netCDF file
    :param vardict: dict, names of variables and dimensions
                    in SEVIRI netCDF file. Keys are used within the
                    program to identify variables.
    :param coordfile: string, name of coordinate file
    
    out:
    :return: dictionary, key are the variable and dimensonnames from varlist,
             values are the n-dimensional data for each variable: 
             eg. time-1D, lat/lon-2D, surface pressure-3D (time,lat,lon),
             level pressure-4D (time, lev, lat, lon)
    """
    
    data_dict = {}
    
    with closing(xr.open_dataset(os.path.join(path,fname), decode_times=False)) as ds:
        time = ds.time.data
        timeunit = ds.time.units
        # change varlist to vardict:  
        for key in vardict:
            var = vardict[key]
            if var == "time":             
               timeunit = ds[var].units
               data_dict.update({"timeunit":timeunit})
                        
            if not isinstance(var, dict):        
                data = ds[var].data
                data_dict.update({key:data})
                
    with closing(xr.open_dataset(os.path.join(path,coordfile), decode_times=False)) as crds:
        for key in vardict:
            var = vardict[key]
            if isinstance(var, dict):
                for name, coord in var.items():
                    data = crds[coord].data
                    data_dict.update({name:data})
            
    return data_dict

# def load_data(path_cosmo, fname_cosmo, foldername, vardict):
#     """
#     loads COSMO data from netCDF file and returns 
#     a dictionary of variables
    
#     args:
#     :param path_cosmo: string, absolute or relative path to directory
#                        of the netCDF COSMO files
#     :param fname_cosmo: dictionary, with lists of file for each date
#     :param foldername: string, name of folder (date)    
#     :param vardict: dict, names of variables and dimensions
#                     in COSMO netCDF file. Keys are used within the
#                     program to identify variables.
    
#     out:
#     :return: dictionary, key are the variable and dimensonnames from varlist,
#              values are the n-dimensional data for each variable: 
#              eg. time-1D, lat/lon-2D, surface pressure-3D (time,lat,lon),
#              level pressure-4D (time, lev, lat, lon)
#     """
    
#     data_dict = {}
#     data_dict.update({"time":[]})
#     for key in vardict:
#         data_dict.update({key:[]})
            
#     for filename in fname_cosmo[foldername]:            
#         with closing(xr.open_dataset(os.path.join(path_cosmo,foldername,filename), decode_times=False)) as ds:
#             timestring = ' '.join([filename.split('_')[2],filename.split('_')[3][0:2]+"00"])
#             time = datetime.datetime.strptime(timestring, '%Y%m%d %H%M')        
#             data_dict["time"].append(time)
#             # change varlist to vardict:  
#             for key in vardict:
#                 var = vardict[key]            
#                 data = ds[var].data
#                 data_dict[key].append(data)
    
#     #Same coordinates in each file!
#     for key in ["lat","lon"]:
#         data_dict[key] = np.mean(np.array([data for data in data_dict[key]]),axis=0)
        
#     for key in ["cod_water","cod_ice"]:
#         data_dict[key] = np.array([data for data in data_dict[key]])
                
#     return data_dict

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

def get_filenames(path, dates):
    """
    get filenames from dates

    Parameters
    ----------
    path : string
        path where COSMO files are kept
    dates : list
        list of dates

    Returns
    -------
    files_good : list
        list of filenames

    
    """
    
    if type(dates[0]) == list:
        dates = create_daterange(dates)
    
    dates_str = [datetime.datetime.strftime(d, "%Y-%m-%d") for d in dates] 

    files = list_files(path)
    files_good = []

    for d in dates_str:
        for f in files:
            if d in f:
                files_good.append(f)
                files.remove(f)

    return files_good
    
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
    lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat_lon_start[0], lat_lon_start[1], lat_lon_dest[0], lat_lon_dest[1]])
 
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


def generate_new_time_array(times, timeunit, timeres, datatype):
    """
    args:
    :param times_old: array with cosmo times
    :param timeunit_old: string describing the timeunit of the cosmo timestamps
    :param timres: desired timeresolution in minutes
    :param datatype: string describing data type
    
    out:
    :return: array with new time grid in the same unit like 'times_old'
    """
    times_filt = filter_times(times, timeunit)
    
    times_datetime = nc4.num2date(times_filt, timeunit)
    times_epochms = nc4.date2num(times_datetime, "microseconds since 1970-01-01")
   
    timeres_ms = timeres*60.*1e6
    dt_ms = times_epochms[-1] - times_epochms[0]
    n = int(np.floor(dt_ms / timeres_ms))
    eff_tres_ms = dt_ms / n
    
    if np.around(eff_tres_ms/60./1e6, decimals=4)  != timeres:
        print("to provide a perfect linear time grid, timeresolution is set to {} minutes".format(eff_tres_ms/60./1e6))
   
    newtimes_epochms = [times_epochms[0]+(i*eff_tres_ms) for i in range(n)]
    newtimes_epochms.append(times_epochms[-1])
    
    newtimes_datetime = nc4.num2date(newtimes_epochms, "microseconds since 1970-01-01")
    newtimes_out = nc4.date2num(newtimes_datetime, timeunit)
#    if datatype == "irradiance":
#        newtimes_out = newtimes_out - timeres/30
    
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
        print("type(data_4d)= '{}' is not supported".format(type(data_4d)))
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
    removed from the originan time array
    
    args:
    :param time: array or list of integers or float representing
                 time, eg. seconds since 1970-01-01 (epochtime)
    :param timeunit: string, describing the unit of the time array
                     eg. 'seconds since 1970-01-01' for epochtime
    out:
    :return: filtered time array
    """
    
    ref_time = nc4.num2date(np.mean(time), timeunit)
    time_filt = [t for t in time if nc4.num2date(t, timeunit).strftime("%Y%m%d") == ref_time.strftime("%Y%m%d")]
    
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


#######################################################################
###   file and dirname generation, setup of dir/file infrastructure ###
#######################################################################
        

def generate_file_and_dir_names(lat_lon_gp, fname, filetype):
    """
    generate names for main directories (names contain lat lon position of COSMO gridpoint), 
    names contain date of each day
    
    args:
    :param lat_lon_gp: dictionary which contains cosmo grid point(s) as  (lat, lon) tuples as values 
    :param fname: string, gives the filename from the date
    :param filetype: string, controls the generated filename to identify the file content by the filename. 
                     Options are 'surface' and 'irradiance'
    
    :out
    :return main_dir: dictionary with names (strings) of the main directory as values containing 
                      the lat and lon position of the cosmo gridpoint to a precision of 2 digits. 
    :return fnames: list of strings with filenames for each date in the cosmo dataset.
                    The filename contains the date in the form YYYYmmDD eg. 2018.10.05 --> 20181005
    """
    
    main_dir_dict = {}

    for key in lat_lon_gp:
        lat_lon = lat_lon_gp[key]
        lat = float(lat_lon[0])
        lon = float(lat_lon[1])
        
        main_dir = "lat_{0:.2f}_lon_{1:.2f}".format(lat, lon)
        main_dir_dict.update({key:main_dir})
                
    day = ''.join([s for s in fname if s.isdigit()])
    if filetype == "cod":
        fname = day + "_seviri_cod.dat"
    
    return (main_dir_dict, fname)
       

def check_existing_files_and_dirs(save_path, lat_lon_gp, fname, filetype):
    """
    args:
    :param save_path: string with path to the directory where the main directories of   
                      the folder structure will be generated
    :param lat_lon_gp: dictionary, keys are  values are (lat, lon) position of the pv station
    :param fname: list of file names
    """

    # 1. get dir and filenames
    main_dirs, fname = generate_file_and_dir_names(lat_lon_gp, fname, filetype)
    
    # 2. check for existing main_dir and if main_dir exists, for existing subdir and fnames
    dirs = list_dirs(save_path)
    
    mdir_exist = {}
    files_exist = {}
    
    for key in main_dirs:
        files_exist.update({key:[]})
        if main_dirs[key] in dirs:
            mdir_exist.update({key:True})
            files = list_files(os.path.join(save_path, main_dirs[key]))
            if fname in files:
                files_exist[key].append(True)
            else:
                files_exist[key].append(False)                    
                    
        else:
            mdir_exist.update({key:False})
            files_exist.update({key:False})
        
    return mdir_exist, files_exist 

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
                print("---"*20)
                print("INFO from function check_write_lat_lon_to_known_stations():")
                print("location with name {}, lat {:.5f} and lon {:.5f} is already in known_stations!".format(name, lat_lon[0], lat_lon[1]))
                print("---"*20)

            elif name not in names and not latlon_exist:

                names.append(name) 
                lat.append(lat_lon[0])
                lon.append(lat_lon[1])

                data = np.array(list(zip(names, lat, lon)), dtype=[("name", 'U6'), ("lat", "float32"), ("lon", "float32")])
                np.savetxt(os.path.join(path, fname), data, fmt=["%s", "%.5f", "%.5f"], header=header)

            else:
                print("---"*20)
                print("ERROR function check_write_lat_lon_to_known_stations():")
                print("location with name {}, lat {:.5f} and lon {:.5f}  already exists".format(name, lat_lon[0], lat_lon[1]))
                print("but name and position do not match!!")
                print("program exits...")
                print("---"*20)
                quit()
                    
            #except Exception as e:
               #print "ERROR in check_write_lat_lon_to_known_stations: ", e
               #raise e

               #if str(e) == 'need more than 0 values to unpack':
                   #data = np.array([(name, lat_lon[0], lat_lon[1])], dtype=[("name", '|S50'), ("lat", "float32"), ("lon", "float32")])
                   #np.savetxt(os.path.join(path,fname), data, fmt=["%s" , "%.5f", "%.5f"], header=header)              

#######################################################################
###       specific functions to create and save surface files       ###
#######################################################################
                    
def write_cod_file(path, fname, data_dict, lat_lon_idx, time, timeunit, header, pixel_avg):
    """
    writes COSMO cloud parameters for all required timesteps.
    
    args:
    :param path: string, full path to the directory where the files are saved 
    :param fname: filename for surface file
    :param data_dict: dictionary with atmospheric parameters: temperature (T in [K]), 
                      pressure (P in [Pa]) and water vapor number density [n_h2o in #/cm3]
                      The values must have the dimensions (time, zlev, lat, lon).    
    :param lat_lon_idx: tuple with the latitue and longitude index which will be selected
                        from each 4d value of the data_dict.
    :param time: array of COSMO timesteps (or interpolated timesteps), the timeunit hast
                 to be a integer or floating point number    
    :param timeunit: string, timeunit of the time array, eg. 'seconds since 1970-01-01' (epoch)
    :param header: string, contains the header of the standard atmosphere file    
    :param pixel_avg: dictionary, number of pixels to average in lat and lon direction

    
    """
       
    hdr_add = "Optical depth at 500nm of water clouds from SEVIRI-HRV, matched to nearest grid point"
    
    header_new = "{}\n{:^20} {}".format(hdr_add, "time", "{:>15} {:>15} {:>15} {:>15}".format(header[0],
                                                          header[1],header[2],header[3]))
            
    lat, lon = lat_lon_idx    
    
    dlat = pixel_avg["lat"]
    dlon = pixel_avg["lon"]

    cod_seviri_exact = data_dict["cod"][:,lat,lon]
    cod_seviri_mean = np.mean(data_dict["cod"][:,lat-dlat:lat+dlat+1,lon-dlon:lon+dlon+1],axis=(1,2))
    cod_seviri_iqr = np.subtract(*np.percentile(data_dict["cod"][:,lat-dlat:lat+dlat+1,lon-dlon:lon+dlon+1],
                                                [75, 25],axis=(1,2)))
    d_cod_seviri = data_dict["d_cod"][:,lat,lon]    
    timestamp = nc4.num2date(sorted(time), timeunit)
    
    save_data = np.column_stack(([dt.strftime(" %d.%m.%Y;%H:%M:%S") for dt in timestamp],
                                 ["%15.3f" % codw for codw in cod_seviri_exact],
                                 ["%15.3f" % dcod for dcod in d_cod_seviri],
                                 ["%15.3f" % cod for cod in cod_seviri_mean],
                                 ["%15.3f" % cod for cod in cod_seviri_iqr]))    
        
    np.savetxt(os.path.join(path, fname), save_data, header=header_new, fmt="%s")

# Pandas approach, without comments in file        
#    save_data = np.column_stack((windspeed_cosmo,winddir_cosmo,temp_cosmo))  
#    
#    df = pd.DataFrame(save_data,index=timestamp,columns=header)    
#    df.to_csv(os.path.join(path, fname), sep='\t', float_format="%20.3f",
#              date_format='%d.%m.%Y %H:%M:%S')    

    
def save_cloud_files(save_path, fname, data_dict, idx_dict, lat_lon_gp, 
                     pixel_avg, stations_lat_lon, time, timeunit, header):
    """
    creates main and sub directories and writes cloud files
    
    args:
    :param save_path: string, path to the directory where the files will be created
    :param fname: sting with filename
    :param data_dict: dictionary containing the COSMO variables which will be written to the 
                      surface properties file.     
    :param idx_dict: dictionary, keys represent the different stations and the values are
                     the indices of latitude and longitude
    :param lat_lon_gp: dictionary, keys represent the different stations and values are
                       tuples of (lat, lon) with the position of the nearest COSMO grid point
                       (Note: variable probably unnecessary)
    :param pixel_avg: dictionary, number of pixels to average in lat and lon direction
    :param stations_lat_lon: dictionary, keys represent the different solar power plants,
                             values are tuples of their position in (lat,lon) coordinates
    :param time: array of COSMO timesteps (or interpolated timesteps), the timeunit hast
                 to be a integer or floating point number
    :param timeunit: string, timeunit of the time array, eg. 'seconds since 1970-01-01' (epoch)
    :param header: string, contains the header for the surface properties file
    
    """
    # 1. check for existing files and dirs
    mdir_exist, files_exist = check_existing_files_and_dirs(save_path, lat_lon_gp, 
                                            fname, filetype="cod")

    # 2. generate file and dir names
    mdir_namedict, filename = generate_file_and_dir_names(lat_lon_gp, 
                                            fname, filetype="cod")
    val_set = []
    
    for key in lat_lon_gp:
        if lat_lon_gp[key] in val_set:
            check_write_lat_lon_to_known_stations(os.path.join(save_path, mdir_namedict[key]), stations_lat_lon[key], key)
        
        else:
            val_set.append(lat_lon_gp[key])
            if mdir_exist[key]:
                check_write_lat_lon_to_known_stations(os.path.join(save_path, mdir_namedict[key]), stations_lat_lon[key], key)
                
                if any(files_exist[key]):
                    path = os.path.join(save_path, mdir_namedict[key])
                    write_cod_file(path, filename, data_dict, idx_dict[key], time, timeunit, header, pixel_avg)
                else:
                    path = os.path.join(save_path, mdir_namedict[key])
                    write_cod_file(path, filename, data_dict, idx_dict[key], time, timeunit, header, pixel_avg)
                
            else:
                os.mkdir(os.path.join(save_path,mdir_namedict[key]))
                check_write_lat_lon_to_known_stations(os.path.join(save_path, mdir_namedict[key]), stations_lat_lon[key], key)
                path = os.path.join(save_path, mdir_namedict[key])
                write_cod_file(path, filename, data_dict, idx_dict[key], time, timeunit, header, pixel_avg)

    
def create_cod_files(path, fname, vardict, coordfile, stations_lat_lon, save_path_cod,
                                    pixel_avg, timeres=None):
    """
    'main' function calling all necessary functions to create and save cloud property files
    
    
    args:
    :param path: string, absolute or relative path to directory of the netCDF MSGCPP files
    :param fname: dictionary, with lists of file for each date    
    :param varlist: list of strings, names of variables and dimensions in SEVIRI netCDF file 
    :param coordfile: string, name of coordinate file
    :param stations_lat_lon: dictionary, keys represent the different solar power plants,
                             values are tuples of their position in (lat,lon) coordinates
    :param save_path_cod: string, path to the directory where the cloud property files will be saved    
    :param pixel_avg: dictionary, number of pixels to average in lat and lon direction
    :param timeres: float, desired time resolution of the SEVIRI data in minutes, 
                    default value is 'None' which means that the original time 
                    resolution of the data is used (5 minutes)
    
    """
    #Load data           
    data_dict = load_data(path,fname,vardict,coordfile)
    
    #Extract surface data
    cloud_dict = {k: data_dict[k] for k in ("cod","d_cod")}
        
    #Find nearest gridpoint corresponding to PV systems
    idx_nearest_gp, lat_lon_nearest_gp = find_nearest_cosmo_gridpoint(data_dict["lat"], data_dict["lon"], stations_lat_lon)
    
    header_cod = ["tau_500","d_tau_500","mean_tau_500","iqr_tau_500"]        
    
    #Interpolate and calculate wind speed and direction
    if timeres != None:
        print("This version does not interpolate times")
        # print("interpolate COSMO data ({}) to higher time resolution ({} minutes)...".format(fname_cosmo,timeres))
        # new_times = generate_new_time_array(data_dict["time"], data_dict["timeunit"], timeres, datatype = "surface")
        # interp_dict = interpolate_profiles_in_time(surface_dict, data_dict["time"], new_times)
        
        # #Calculate wind speed (and rotate to get directions)
        # surface_interp_dict = wind_from_components(interp_dict['Zonal wind'],
        #                         interp_dict['Meridional wind'],data_dict['lat'],
        #                         data_dict['lon'],data_dict['grid_north_pole_latitude'],
        #                         data_dict['grid_north_pole_longitude'])
        
        # #Add interpolated temperature        
        # surface_interp_dict.update({"T_ambient":interp_dict["Temperature_sfc" ]})
        
        # print("saving surface properties to file...")
        # save_surface_files(save_path_surf, fname_cosmo, surface_interp_dict, idx_nearest_gp, lat_lon_nearest_gp, 
        #                stations_lat_lon, new_times, data_dict["timeunit"], header_surface)
        
        # new_times = generate_new_time_array(data_dict["time"], data_dict["timeunit"], timeres, datatype = "irradiance")
        # #Interpolate irradiance profiles
        # irrad_interp_dict = interpolate_profiles_in_time(irradiance_dict, data_dict["time"], new_times)
            
        # print("saving irradiance properties to file...")
        # save_irradiance_files(save_path_irrad, fname_cosmo, irrad_interp_dict, idx_nearest_gp, lat_lon_nearest_gp, 
        #                stations_lat_lon, new_times, data_dict["timeunit"], header_irrad)
         
    else:                
        print("saving cloud properties to file...")
        save_cloud_files(save_path_cod, fname, cloud_dict, idx_nearest_gp, lat_lon_nearest_gp, 
                       pixel_avg, stations_lat_lon, data_dict["time"], data_dict["timeunit"], header_cod)
        
        
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
            config = yaml.load(ds,Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)
    return config  
    
#%%
def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("configfile", help="yaml file containing config")
    args = parser.parse_args()

    config_filename = os.path.abspath(args.configfile) # #"../examples/config_SEVIRI2PVCOD_2018_messkampagne.yaml" #
    
    config = load_yaml_configfile(config_filename)
    
    homepath = os.path.expanduser('~')
    
    path_seviri    = os.path.join(homepath,config["path_seviri"])
    fnames_seviri   = config["fnames_seviri"]
    
    if fnames_seviri == "from_dates":
        dates = config["dates"]
        fnames_seviri = get_filenames(path_seviri, dates)
        
    vardict_seviri = config["variable_names_seviri"] #.values()
    
    coord_file = config["coord_file"]
    
    stations_lat_lon = config["stations_lat_lon"]
    for key in stations_lat_lon:
        stations_lat_lon[key] = tuple(stations_lat_lon[key])
    
    timeres = None if config["timeresolution"] == "None" else config["timeresolution"]
    
    pixel_avg = config["pixel_avg"]
        
    save_path_cod = os.path.join(homepath,config["path_cod_files"])
    
    for fname in fnames_seviri:
        create_cod_files(path_seviri, fname, vardict_seviri, coord_file, stations_lat_lon, 
                                        save_path_cod, pixel_avg, timeres=timeres)
   
if __name__ == "__main__":
    main()