#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 13:42:18 2021

@author: james

Functions commonly used for radiative transfer simulation

"""

import numpy as np
import pandas as pd
import os

def define_disort_grid (res_dict):
    """
    Create grid for DISORT simulation of diffuse radiances
    
    args:
    :param res_dict: dictionary containing desired resolution for DISORT grid
    
    out:
    :return: dictionary containing disort grid:
             "theta" ("umu"): arrays of zenith angles (cosine theta) 
             "phi": array of azimuth angles
             "umustring", "phistring": strings for libradtran
    
    """
    
    disort_grid_dict = {}
    
    # Zenith angles
    num_theta_bins = int(180/res_dict['theta'] + 1)
    theta  = np.linspace(0,np.pi,num_theta_bins,endpoint=True)
    umu = - np.cos(theta)
    umu[int(num_theta_bins/2)] = 0.000001
    umustring = ' '.join([format(v,".6f") for v in umu])
    
    disort_grid_dict.update({"theta":theta})
    disort_grid_dict.update({"umu":umu})
    disort_grid_dict.update({"umustring":umustring})
    
    # azimuth angles (libradtran convention, south is 0)
    num_phi_bins = int(360/res_dict['phi'] + 1)
    phi = np.linspace(0.0,360.0,num_phi_bins,endpoint=True)
    phistring = ' '.join([format(v,".1f") for v in phi])
    
    disort_grid_dict.update({"phi":phi})
    disort_grid_dict.update({"phistring":phistring})

    return disort_grid_dict

def int_2d_diff(rad_mu, cos_factor, mu, phi):
    """
    Integrate radiance field over the mu = cos(theta) and phi angles to get the 
    total irradiance
    
    args:
    :param rad_mu: array of floats, radiance field as function of mu and phi
    :param cos_factor: array or vector of floats, cosine factor from Euler 
                       transformation, equals mu in the case of downward direction
    :param mu: float, cosine of zenith angle theta
    :param phi: float, azimuth angle
    
    out:
    :return: float, diffuse irradiance
    """
    umu_integral = np.zeros(len(phi))
    
    #in this case the cos_factor has azimuth dependence
    if len(cos_factor.shape) == 2: 
        rad_mu = rad_mu*cos_factor
    
        for iphi in range(len(phi)):
            umu_integral[iphi] = -np.trapz(rad_mu[:,iphi],mu)
    
    #in this case the cos_factor is independent of azimuth
    else:        
        for iphi in range(len(phi)):
            umu_integral[iphi] = -np.trapz(rad_mu[:,iphi]*cos_factor,mu)
    
    total_integral = np.trapz(umu_integral,phi)
    
    return total_integral

def read_lrt_atmosphere(filename,skiprows=1):
    """
    

    Parameters
    ----------
    filename : string
        name of file
    skiprows : integer
        number of rows to skip

    Returns
    -------
    df_lrt_atm : dataframe, libradtran atmosphere

    """
    
    df_lrt_atm = pd.read_csv(filename,skiprows=skiprows,header=0,sep='\s+')
    
    cols = df_lrt_atm.columns.to_list()
    cols_new = cols[1:]
    cols_new.append('dummy')
    
    df_lrt_atm.rename(columns=dict(zip(cols,cols_new)),inplace=True)
    df_lrt_atm.drop(columns='dummy',inplace=True)
    
    return df_lrt_atm

def calc_precipitable_water(n_h20_col_cm3,z_col_km):
    """
    

    Parameters
    ----------
    n_h20_col_cm3 : array of float
        concentration of water in molecules per cm^3
    z_col : array of float
        height in km

    Returns
    -------
    h2o_col_mm : float, precipitable water in kg / m^2

    """
    
    #Constants for calculation
    N_A = 6.02214858e23 # Avogadro constant, units: [1/mol]
    M_H2O = 18.01528    # molar mass water, units: [mol]

    #Convert number concentration and altitude to metres
    n_h20_col_m3 = n_h20_col_cm3*1e6
    z_col_m = z_col_km*1000.
    
    m_h2o_kg = M_H2O/N_A/1000.

    h2o_col_mm = np.trapz(n_h20_col_m3,z_col_m)*m_h2o_kg
    
    return h2o_col_mm