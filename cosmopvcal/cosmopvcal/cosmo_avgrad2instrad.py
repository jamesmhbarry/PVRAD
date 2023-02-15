# calculate instantaneous radiation from averaged radiation 
# add instantaneous radiation to netcdf file and save to new file
import os
import datetime
import numpy as np
import netCDF4 as nc4
import xarray as xr

from contextlib import closing



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

def datetime64_to_datetime(dt64_arr):

    if "ns" in str(dt64_arr.dtype):
        unixtime = dt64_arr.astype("int64")/1e9
        datetime = nc4.num2date(unixtime, "seconds since 1970-01-01")
    elif "us" in str(dt64_arr.dtype):
        unixtime = dt64_arr.astype("int64")/1e6
        datetime = nc4.num2date(unixtime, "seconds since 1970-01-01")

    elif "ms" in str(dt64_arr.dtype):
        unixtime = dt64_arr.astype("int64")/1e3
        datetime = nc4.num2date(unixtime, "seconds since 1970-01-01")

    else:
        raise TypeError("provided date object is of wrong type")

    return datetime



def add_inst_rad2ds(ds, varnames, new_varnames, tin_fc):
    """
    Calculate instantaneous irradiance from cumulative values

    Parameters
    ----------
    ds : xarray
        dataset with COSMO data
    varnames : list
        list of variable names from COSMO
    new_varnames : list
        list of new variable names
    tin_fc : int
        hour of first value in files

    Returns
    -------
    ds : xarray
        dataset with modified COSMO irradiance data

    """

    time = datetime64_to_datetime(ds.time.data)

    if tin_fc == 0:
        t0 = datetime.datetime(time[0].year, time[0].month, time[0].day, 0, 0)
    elif tin_fc == 12:
        t0 = datetime.datetime(time[0].year, time[0].month, time[0].day, 12, 0)
    tis = np.array([(t-t0).total_seconds() for t in time])
    #print("--------------------")
    #print("tis", tis.shape)
    
    tis = tis.reshape(-1, 1, 1)
    dt = np.diff(tis, axis=0)


    for var, newvar in zip(varnames, new_varnames):
        inst_rad =  ((ds[var].data[1:] * tis[1:]) - (ds[var].data[:-1] * tis[:-1])) / dt
        
        # inst rad has one element less than time coord --> append 0 at the beginning
        # valid fix for forecast init at 0 UTC, not valid for other init times
        # or other regions of the world
        first = np.zeros(tuple([1]+list(ds[var].data.shape[1:])))
        inst_rad = np.append(first, inst_rad, axis=0)
        coords = [ds[dim].data for dim in list(ds[var].dims)] 
        ds[newvar] = xr.DataArray(inst_rad.astype("float32"), coords=coords, dims=list(ds[var].dims))

    return ds


def write_instrad2COSMO(path_read, path_write, avg_rad_names, inst_rad_names, tin_fc):
    """
    calculate instantaneous radiation from standard COSMO output,
    add it to the dataset and write the dataset to a new file

    Parameters
    ----------
    path_read : string
        path where data is stored
    path_write : string
        path where new data is to be saved
    avg_rad_names : list
        list of variable names
    inst_rad_names : list
        list of new variable names
    tin_fc : int
                hour of first value in file

    Returns
    -------
    None.

    """
    
    files = list_files(path_read)

    for f in files:
        print("processing file {} ...".format(f))
        with closing(xr.open_dataset(os.path.join(path_read,f))) as ds:
            ds_new = add_inst_rad2ds(ds, avg_rad_names, inst_rad_names, tin_fc)
            print("writing new extended dataset to file ...")
            ds_new.to_netcdf(path=os.path.join(path_write,f))

    print("done!")


def main():
    # define stuff
    path_read = "/mnt/bigdata/share/00_Projects/2_Solarwatt/Data/COSMO/raw_data/cosmo_solarwatt"
    path_write = "/mnt/bigdata/share/00_Projects/2_Solarwatt/Data/COSMO/raw_data/cosmo_solarwatt_new" #tin_00UTC_rad"
    tin_fc = 0
    avg_rad_names = ["ASWDIFD_S", "ASWDIFU_S", "ASWDIR_S"]
    inst_rad_names = ["SWDIFD_S", "SWDIFU_S", "SWDIR_S"]

    write_instrad2COSMO(path_read, path_write, avg_rad_names, inst_rad_names, tin_fc)


    


if __name__ == "__main__":
    main()
#tis = [0, 3600, 7200., ....] # time in seconds since forecast start
#dt = 3600. # timestep between forecast outputs
#cosmo_avgrad = [....] # ASWDIR_S, ASWDIFD_S: averaged diffuse and direct downward radiation from cosmo
#inst_rad =  (( cosmo_avgrad[1:] * tis[1:]) - (cosmo_avgrad[:-1] * tis[:-1])) / dt
