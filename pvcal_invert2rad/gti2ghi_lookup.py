
import xarray as xr
import numpy as np

from contextlib import closing
from scipy.interpolate import interpn


class GTI2GHI(object):
    def __init__(self, fname_lut):
        self.a, self.b, self.grid, self.dims_order = self.read_lut(fname_lut)
    
    def read_lut(self, fname_lut):
        with closing(xr.open_dataset(fname_lut)) as ds:
            a = ds["a"].data
            b = ds["b"].data
            grid = tuple([ds[d].data for d in ds["a"].dims])
            return a, b, grid, ds["a"].dims

    def get_ghi(self, points, gti):
        """
        IMPORTANT: Order of dimensions is: [sza, rel_azi, tilt, cf]
        :param points: array_like
                       can be a list of list or list of tuples where each sublist/tuple
                       contains values for the 4 dimensions. 3rd option is an array of
                       shape (n x n_dims).
        :param gti: float or array, gti in W/m2

        """
        self.check_points_valid(points)
        a_interp, b_interp = self.interpolate_lut(points)
        
        if not isinstance(gti, (np.ndarray, float)):
            return self.gti2ghi(a_interp, b_interp, np.array(gti))
        
        else:
            return self.gti2ghi(a_interp, b_interp, gti)
    
    def interpolate_lut(self, points):
        a_interp = interpn(self.grid, self.a, points)
        b_interp = interpn(self.grid, self.b, points)
        return a_interp, b_interp
        
    def lin_func(self, a, b, x):
        return a*x + b
    
    def gti2ghi(self, a, b, gti):
        return self.lin_func(a,b, gti)
    
    def check_points_valid(self, points):
        dims_min = np.array([np.min(d) for d in self.grid])
        dims_max = np.array([np.max(d) for d in self.grid])
        
        if isinstance(points, (list, tuple)) and isinstance(points[0], (list, tuple)):
            tmp = np.stack(points, axis=0)
            max_vals = np.max(tmp, axis=0)
            min_vals = np.min(tmp, axis=0)
            if np.any(np.concatenate((max_vals > dims_max, min_vals < dims_min))):
                self.boundary_error(dims_min, dims_max)
        
        elif isinstance(points, np.ndarray) and len(points.shape) == 2:
            max_vals = np.max(points, axis=0)
            min_vals = np.min(points, axis=0)
            if np.any(np.concatenate((max_vals > dims_max, min_vals < dims_min))):
                self.boundary_error(dims_min, dims_max)
        
        else:
            check = np.any(np.diff(np.array(list(zip(dims_min, points, dims_max))), axis=1) < 0)
            if check:
                self.boundary_error(dims_min, dims_max)
    
    def boundary_error(self, dims_min, dims_max):
        bounds_msg = []
        bounds_msg.append("#"*30)
        bounds_msg.append("Boundaries of the lookup table:")
        for d, mi, ma in zip(self.dims_order, dims_min, dims_max):
            bounds_msg.append("{:.2f} <= {} <= {:.2f}".format(mi, d, ma))
        bounds_msg = "\n".join(bounds_msg)
        msg = "one or more of your requested points are outside the boundaries of the lookup table which are listed below" +"\n"+ bounds_msg
        raise ValueError(msg)

    def print_dims_order(self):
        print("Order of dimensions in LUT:{}".format(self.dims_order))
    