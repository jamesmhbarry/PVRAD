#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 16:45:25 2018

Collection of functions for the PV power forward model

@author: james
"""
import numpy as np
from numpy import sin, cos, tan, arcsin, arccos, exp, sqrt, log, nan
from math import pi
import diode_models as dm
from scipy.interpolate import interp1d


#######################################################################
###                         MODEL FUNCTIONS                               
#######################################################################

def azi_shift (lrt_azimuth):
    """
    Shift azimuth angles in radians for libradtran convention
    
    args:
        :param lrt_azimuth: float, azimuth angle in radians
        
    out:
        :return: float azimuth angle in radians shifted by pi radians
    """
    result = np.fmod(lrt_azimuth + pi, 2*pi)
    
    return result

def d_rperp_dn (x,n):
    """
    Calculate the derivatives of the reflection term for perpendicularly
    polarised light with respect to the refractive index n
    
    args:
    :param x: float, cosine of angle of incidence
    :param n: index of refraction
    
    out:
    :return: float, derivative of perpendicular polarised term
    """
    
    #Numerator
    rperp_1 = 1. - ((x**2 - 1.)/n - x*sqrt((x**2 - 1.)/n**2 + 1.))**2
    
    #Denominator
    rperp_2 = 1. - ((x**2 - 1.)/n + x*sqrt((x**2 - 1.)/n**2 + 1.))**2
    
    #Derivatives
    d_rperp_1 = -(2*(x**2 - 1.)/n**2 - 2*x*(x**2 - 1.)/\
                  (n**3*sqrt(1. + (x**2 - 1.)/n**2)))\
                *(x*sqrt(1. + (x**2 - 1.)/n**2) - (x**2 - 1.)/n)
    
    d_rperp_2 = -(-2*(x**2 - 1.)/n**2 - 2*x*(x**2 - 1.)/\
                  (n**3*sqrt(1. + (x**2 - 1.)/n**2)))\
                *(x*sqrt(1. + (x**2 - 1.)/n**2) + (x**2 - 1.)/n)
    
    #Total derivative with quotient rule                
    diff_result = (d_rperp_1*rperp_2 - rperp_1*d_rperp_2)/rperp_2**2
    
    return diff_result

def d_rpar_dn (x,n):
    """
    Calculate the derivatives of the reflection term for parallel
    polarised light with respect to the refractive index n
    
    args:
    :param x: float, cosine of angle of incidence
    :param n: index of refraction
    
    out:
    :return: float, derivative of parallel polarised term
    """
    #Numerator
    rpar_1 = 1./((x**2 - 1.)/n - x*sqrt((x**2 - 1.)/n**2 + 1))**2 - 1.
    
    #Denominator
    rpar_2 = 1./((x**2 - 1.)/n + x*sqrt((x**2 - 1.)/n**2 + 1))**2 - 1.
    
    #Derivatives
    d_rpar_1 = (2*(x**2 - 1.)/n**2 - 2*x*(x**2 - 1.)/(n**3*sqrt(1. + (x**2 - 1.)/n**2)))\
                /(-x*sqrt(1. + (x**2 - 1.)/n**2) + (x**2 - 1.)/n)**3
                
    d_rpar_2 = (2*(x**2 - 1.)/n**2 + 2*x*(x**2 - 1.)/(n**3*sqrt(1. + (x**2 - 1.)/n**2)))\
                /(x*sqrt(1. + (x**2 - 1.)/n**2) + (x**2 - 1.)/n)**3

    #Total derivative with quotient rule                
    diff_result = (d_rpar_1*rpar_2 - rpar_1*d_rpar_2)/rpar_2**2   
    
    return diff_result

def d_taupv_dn (x, n, K, L):
    """
    Calculate the total derivative of the transmission function with 
    respect to the refractive index n
    
    args:
    :param x: float, cosine of angle of incidence
    :param n: index of refraction
    :param K: extinction coefficient in m^-1
    :param L: thickness of glazing in m
    
    out:
    :return: float, derivative of transmission function
    """
    
    #Reflection terms written as functions of x and n
    rperp_1 = 1. - ((x**2 - 1)/n - x*sqrt((x**2 - 1.)/n**2 + 1.))**2
    rperp_2 = 1. - ((x**2 - 1)/n + x*sqrt((x**2 - 1.)/n**2 + 1.))**2
    rpar_1 = 1./((x**2 - 1)/n - x*sqrt((x**2 - 1.)/n**2 + 1.))**2 - 1.
    rpar_2 = 1./((x**2 - 1)/n + x*sqrt((x**2 - 1.)/n**2 + 1.))**2 - 1.
    
    #Cosine of angle of refraction and its derivative
    cos_thetar = sqrt(1. + (x**2 - 1.)/n**2)
    d_costhetar_dn = -(x**2 - 1.)/(n**3*sqrt(1 + (x**2 - 1.)/n**2))
    
    #Transmission for zero angle of incidence
    tau0 = exp(-K*L)*(4*n/((1. + n)**2))
    dtau0_dn = exp(-K*L)*4*(-n + 1.)/(n + 1.)**3
    
    #Transmission for general case
    exp_factor = exp(-K*L/cos_thetar)
    tau = exp_factor*(1. - (rperp_1/rperp_2 + rpar_1/rpar_2)/2)
    
    #Derivative of general case
    dtau_dn = (K*L/cos_thetar**2)*exp_factor*d_costhetar_dn*(1.\
              - (rperp_1/rperp_2 + rpar_1/rpar_2)/2)\
                  - exp_factor/2*(d_rperp_dn(x,n) + d_rpar_dn(x,n))
    
    #Total derivative using quotient rule
    diff_result = (dtau_dn*tau0 - tau*dtau0_dn)/tau0**2
    
    return diff_result

def d_rperp_dcostheta (x,n):
    """
    Calculate the derivatives of the reflection term for perpendicularly
    polarised light with respect to the cosine of incidence angle Theta
    
    args:
    :param x: float, cosine of angle of incidence
    :param n: index of refraction
    
    out:
    :return: float, derivative of perpendicular polarised term
    """
    
    #Numerator
    rperp_1 = 1. - ((x**2 - 1.)/n - x*sqrt((x**2 - 1.)/n**2 + 1.))**2
    
    #Denominator
    rperp_2 = 1. - ((x**2 - 1.)/n + x*sqrt((x**2 - 1.)/n**2 + 1.))**2
    
    #Derivatives
    d_rperp_1 = 2*((x**2 - 1.)/n - x*sqrt((x**2 - 1.)/n**2 + 1.))*\
                (sqrt((x**2 - 1.)/n**2 + 1.) - 2*x/n +\
                 x**2/(n**2*sqrt((x**2 - 1.)/n**2 + 1.)))
                
    d_rperp_2 = -2*((x**2 - 1.)/n + x*sqrt((x**2 - 1.)/n**2 + 1.))*\
                (sqrt((x**2 - 1.)/n**2 + 1.) + 2*x/n +\
                 x**2/(n**2*sqrt((x**2 - 1.)/n**2 + 1.)))
    
    #Total derivatives with quotient rule                
    diff_result = (d_rperp_1*rperp_2 - rperp_1*d_rperp_2)/rperp_2**2
    
    return diff_result

def d_rpar_dcostheta (x,n):
    """
    Calculate the derivatives of the reflection term for parallel
    polarised light with respect to the cosine of incidence angle Theta
    
    args:
    :param x: float, cosine of angle of incidence
    :param n: index of refraction
    
    out:
    :return: float, derivative of perpendicular polarised term
    """
    
    #Numerator
    rpar_1 = 1./((x**2 - 1.)/n - x*sqrt((x**2 - 1.)/n**2 + 1.))**2 - 1.
    
    #Denominator
    rpar_2 = 1./((x**2 - 1.)/n + x*sqrt((x**2 - 1.)/n**2 + 1.))**2 - 1.
    
    #Derivatives
    d_rpar_1 = 2*(sqrt((x**2 - 1.)/n**2 + 1.) - 2*x/n + x**2/\
                  (n**2*sqrt((x**2 - 1.)/n**2 + 1.)))\
                /((x**2 - 1.)/n - x*sqrt((x**2 - 1.)/n**2 + 1.))**3
                
    d_rpar_2 = -2*(sqrt((x**2 - 1.)/n**2 + 1.) + 2*x/n + x**2/\
                   (n**2*sqrt((x**2 - 1.)/n**2 + 1.)))\
                /((x**2 - 1.)/n + x*sqrt((x**2 - 1.)/n**2 + 1.))**3
    
    #Total derivative with quotient rule                                
    diff_result = (d_rpar_1*rpar_2 - rpar_1*d_rpar_2)/rpar_2**2   
    
    return diff_result

def d_taupv_dcostheta (x, n, K, L):
    """
    Calculate the total derivative of the transmission function with 
    respect to the cosine of incidence angle Theta
    
    args:
    :param x: float, cosine of angle of incidence
    :param n: index of refraction
    :param K: extinction coefficient in m^-1
    :param L: thickness of glazing in m
    
    out:
    :return: float, derivative of transmission function
    """
    
    #Reflection terms written as functions of x and n
    rperp_1 = 1. - ((x**2 - 1.)/n - x*sqrt((x**2 - 1.)/n**2 + 1.))**2
    rperp_2 = 1. - ((x**2 - 1.)/n + x*sqrt((x**2 - 1.)/n**2 + 1.))**2
    rpar_1 = 1./((x**2 - 1.)/n - x*sqrt((x**2 - 1.)/n**2 + 1.))**2 - 1.
    rpar_2 = 1./((x**2 - 1.)/n + x*sqrt((x**2 - 1.)/n**2 + 1.))**2 - 1.
    
    #Cosine of angel of refraction and its derivative
    cos_thetar = sqrt(1. + (x**2 - 1.)/n**2)
    d_cos_thetar = x/(n**2*sqrt(1 + (x**2 - 1.)/n**2))
    
    #Transmission for zero angle of incidence
    tau0 = exp(-K*L)*(4*n/((1. + n)**2))
    
    #Derivative of transmission function in the general case
    exp_factor = exp(-K*L/cos_thetar)
    diff_result = ((K*L/cos_thetar**2)*exp_factor*d_cos_thetar*\
                   (1. - (rperp_1/rperp_2 + rpar_1/rpar_2)/2)\
                  - exp_factor/2*(d_rperp_dcostheta(x,n)\
                                  + d_rpar_dcostheta(x,n)))/tau0
    
    return diff_result

def ang_response(theta, n, K, L):
    """
    define the angular response of the PV system with an optical model, as taken
    from De Soto, et al. 2006
    
    args:
    :param theta: float, angle of incoming radiation beam in radians
    :param n: float, refractive index of glass
    :param K: float, absorption coefficient of glass in m^⁻1
    :param L: float, thickness of glass in m
    
    out:
    :return: float, angular response function 
    """
    
    #Calculate refractive index
    theta_r = arcsin(1./n*sin(theta)) #air has refractive index of 1
    #Transmission at normal incidence
    tau0 = exp(-K*L)*(4.*n/((1.+n)**2)) 
    
    if type(theta) == np.ndarray:        
        theta[theta == 0.] = nan
        
        tau = exp(-K*L/cos(theta_r))*(1. - (sin(theta_r - theta)**2/ 
        sin(theta_r + theta)**2 + tan(theta_r - theta)**2/
        tan(theta_r + theta)**2)/2.)
        
        tau[np.isnan(theta)] = tau0
        
        tau[tau/tau0 < 0.00000001] = 0.0
        
    elif type(theta) == np.float64 or type(theta) == np.float:
        if theta == 0:
            tau = tau0
        else:
            tau  = exp(-K*L/cos(theta_r))*(1. - (sin(theta_r - theta)**2/ 
            sin(theta_r + theta)**2 + tan(theta_r - theta)**2/
            tan(theta_r + theta)**2)/2.)
        
#        if tau/tau0 < 0.000001:
#            tau = 0
            
    return tau/tau0

def cos_incident_angle(theta0,phi0,theta,phi):
    """
    Euler transformation that defines the scalar product between the normal vector
    to the PV system and the direction of the incoming radiation beam
    
    args:
    :param theta0: float, zenith angle of incoming radiation beam in radians
    :param phi0: float, azimuth angle of incoming radiation beam in radians
    :param theta: float, elevation angle of PV sytem
    :param phi: float, azimuth angle of PV system
    
    out:
    :return: float, Cosine of incidence angle
    """
    cos_ia = cos(theta0)*cos(theta) + sin(theta0)*sin(theta)*cos(phi0-phi)
    
    if type(cos_ia) != np.float64:
        cos_ia[cos_ia < 0] = 0
    else:
        if cos_ia < 0:
            cos_ia = 0
    
    return cos_ia

def int_2d_diff(rad_mu, cos_factor, mu, phi):
    """
    Integrate radiance field over the mu = cos(theta) and phi angles to get the 
    total irradiance
    
    args:
    :param rad_mu: series of floats, radiance field as function of mu and phi
    :param cos_factor: float, cosine factor from Euler transformation, equals
                       mu in the case of downward direction
    :param mu: float, cosine of zenith angle theta
    :param phi: float, azimuth angle
    
    out:
    :return: float, irradiance
    """
    umu_integral = np.zeros(len(phi))
    
    #in this case the cos_factor has azimuth dependence
    if len(cos_factor.shape) == 2: 
        rad_mu = rad_mu*cos_factor
    
        # for iphi in range(len(phi)):
        #     umu_integral[iphi] = -np.trapz(rad_mu[:,iphi],mu)
        
        umu_integral = -np.trapz(rad_mu,mu,axis=-2)
    #in this case the cos_factor is independent of azimuth
    else:        
        # for iphi in range(len(phi)):
        #     umu_integral[iphi] = -np.trapz(rad_mu[:,iphi]*cos_factor,mu)
        umu_integral = -np.trapz(rad_mu*cos_factor,mu,axis=-2)
    
    total_integral = np.trapz(umu_integral,phi)
    
    return total_integral

def dcos_ia_dtheta(theta0, phi0, theta, phi):
    """
    Derivative of Euler transform by elevation angle
    
    args:
    :param theta0: float, zenith angle of incoming radiation beam in radians
    :param phi0: float, azimuth angle of incoming radiation beam in radians
    :param theta: float, elevation angle of PV sytem
    :param phi: float, azimuth angle of PV system
    
    out:
        :return: float, derivative
    """
    
    cos_ia = cos(theta0)*cos(theta) + sin(theta0)*sin(theta)*cos(phi0-phi)
    diff_result = -cos(theta0)*sin(theta) + sin(theta0)*cos(theta)*cos(phi0-phi)
    
    diff_result[cos_ia < 0] = 0
    
    return diff_result

def dcos_ia_dphi(theta0, phi0, theta, phi):
    """
    Derivative of Euler transform by azimuth angle
    
    args:
    :param theta0: float, zenith angle of incoming radiation beam in radians
    :param phi0: float, azimuth angle of incoming radiation beam in radians
    :param theta: float, elevation angle of PV sytem
    :param phi: float, azimuth angle of PV system
    
    out:
        :return: float, derivative
    """
    
    cos_ia = cos(theta0)*cos(theta) + sin(theta0)*sin(theta)*cos(phi0-phi)
    diff_result = sin(theta0)*sin(theta)*sin(phi0-phi)
    
    diff_result[cos_ia < 0] = 0
    
    return diff_result

def d_Ediff_dtheta(Idiff,angles,tilt,azimuth,d_tilt,optics,n,optical_model_flag):
    """
    Numerical derivative of diffuse radiance by elevation angles
    
    args:
    :param Idiff, 3D array of floats, (time,theta,phi), diffuse radiance field
    :param angles, named tuple containing angular grid for integration
    :param tilt, float defining tilt of PV array
    :param azimuth, float defining azimuth of PV array
    :param d_tilt, float defining step change in tilt for differentiation
    :param optics, named tuple with optical model parameters
    :param n, float, refractive index
    :param optical_model_flag, boolean for optical model
    
    out:
        :return: vector of derivaties for each time point
    """
    
    theta_array = angles.theta
    phi_array = angles.phi
    umu = angles.umu
    
    cos_poa_diff_1 = np.zeros((len(theta_array),len(phi_array)))
    cos_poa_diff_2 = np.zeros((len(theta_array),len(phi_array)))
    trans_diff_1 = np.zeros((len(theta_array),len(phi_array)))
    trans_diff_2 = np.zeros((len(theta_array),len(phi_array)))
            
    #cos factor for rotated diffuse radiance field
    for itheta in range(len(theta_array)):
        for iphi in range(len(phi_array)):
            cos_poa_diff_1[itheta,iphi] = cos_incident_angle(theta_array[itheta],
               phi_array[iphi], tilt - d_tilt, azimuth)
            if optical_model_flag:
                trans_diff_1[itheta,iphi] = ang_response(arccos(cos_poa_diff_1[itheta,iphi]),
                      n,optics.kappa,optics.L)
            else:
                trans_diff_1[itheta,iphi] = 1.0
            
            cos_poa_diff_2[itheta,iphi] = cos_incident_angle(theta_array[itheta],
               phi_array[iphi], tilt + d_tilt, azimuth)
            if optical_model_flag:
                trans_diff_2[itheta,iphi] = ang_response(arccos(cos_poa_diff_2[itheta,iphi]),
                      n,optics.kappa,optics.L)
            else:
                trans_diff_2[itheta,iphi] = 1.0

    Ediff_1 = np.zeros(len(Idiff))                                      
    Ediff_2 = np.zeros(len(Idiff))                                      
    for idiff in range(len(Idiff)):
        Ediff_1[idiff] = int_2d_diff(Idiff[idiff,:,:],-cos_poa_diff_1*trans_diff_1,umu,phi_array)
        Ediff_2[idiff] = int_2d_diff(Idiff[idiff,:,:],-cos_poa_diff_2*trans_diff_2,umu,phi_array)
            
    diff_result = (Ediff_2 - Ediff_1)/(2*d_tilt)

    return diff_result

def d_Ediff_dphi(Idiff,angles,tilt,azimuth,d_azi,optics,n,optical_model_flag):
    """
    Numerical derivative of diffuse radiance by elevation angles
    
    args:
    :param Idiff, 3D array of floats, (time,theta,phi), diffuse radiance field
    :param angles, named tuple containing angular grid for integration
    :param tilt, float defining tilt of PV array
    :param azimuth, float defining azimuth of PV array
    :param d_azi, float defining step change in azimuth for differentiation
    :param optics, named tuple with optical model parameters
    :param n, float, refractive index
    :param optical_model_flag, boolean for optical model
    
    out:
        :return: vector of derivatives for each time point
    """
    
    theta_array = angles.theta
    phi_array = angles.phi
    umu = angles.umu

    cos_poa_diff_1 = np.zeros((len(theta_array),len(phi_array)))
    cos_poa_diff_2 = np.zeros((len(theta_array),len(phi_array)))
    trans_diff_1 = np.zeros((len(theta_array),len(phi_array)))
    trans_diff_2 = np.zeros((len(theta_array),len(phi_array)))
            
    #cos factor for rotated diffuse radiance field
    for itheta in range(len(theta_array)):
        for iphi in range(len(phi_array)):
            cos_poa_diff_1[itheta,iphi] = cos_incident_angle(theta_array[itheta],
               phi_array[iphi], tilt, azimuth - d_azi)
            if optical_model_flag:
                trans_diff_1[itheta,iphi] = ang_response(arccos(cos_poa_diff_1[itheta,iphi]),
                      n,optics.kappa,optics.L)
            else:
                trans_diff_1[itheta,iphi] = 1.0
            
            cos_poa_diff_2[itheta,iphi] = cos_incident_angle(theta_array[itheta],
               phi_array[iphi], tilt, azimuth + d_azi)
            if optical_model_flag:
                trans_diff_2[itheta,iphi] = ang_response(arccos(cos_poa_diff_2[itheta,iphi]),
                      n,optics.kappa,optics.L)
            else:
                trans_diff_2[itheta,iphi] = 1.0
       
    Ediff_1 = np.zeros(len(Idiff))                                      
    Ediff_2 = np.zeros(len(Idiff))                                      
    for idiff in range(len(Idiff)):
        Ediff_1[idiff] = int_2d_diff(Idiff[idiff,:,:],-cos_poa_diff_1*trans_diff_1,umu,phi_array)
        Ediff_2[idiff] = int_2d_diff(Idiff[idiff,:,:],-cos_poa_diff_2*trans_diff_2,umu,phi_array)
            
    diff_result = (Ediff_2 - Ediff_1)/(2*d_azi)

    return diff_result

def d_Ediff_dn(Idiff,angles,tilt,azimuth,d_n,n,optics):
    """
    Numerical derivative of diffuse radiance by elevation angles
    
    args:
    :param Idiff, 3D array of floats, (time,theta,phi), diffuse radiance field
    :param angles, named tuple containing angular grid for integration
    :param tilt, float defining tilt of PV array
    :param azimuth, float defining azimuth of PV array
    :param d_n, float defining step change in n for differentiation
    :param optics, named tuple with optical model parameters
    :param n, float, refractive index
    
    out:
        :return: vector of derivatives for each time point
    """
    
    theta_array = angles.theta
    phi_array = angles.phi
    umu = angles.umu

    cos_poa_diff = np.zeros((len(theta_array),len(phi_array)))
    trans_diff_1 = np.zeros((len(theta_array),len(phi_array)))
    trans_diff_2 = np.zeros((len(theta_array),len(phi_array)))
            
    #cos factor for rotated diffuse radiance field
    for itheta in range(len(theta_array)):
        for iphi in range(len(phi_array)):
            cos_poa_diff[itheta,iphi] = cos_incident_angle(theta_array[itheta],
               phi_array[iphi], tilt, azimuth)
            
            trans_diff_1[itheta,iphi] = ang_response(arccos(cos_poa_diff[itheta,iphi]),
                      n - d_n,optics.kappa,optics.L)
            trans_diff_2[itheta,iphi] = ang_response(arccos(cos_poa_diff[itheta,iphi]),
                      n + d_n,optics.kappa,optics.L)
       
    Ediff_1 = np.zeros(len(Idiff))                                      
    Ediff_2 = np.zeros(len(Idiff))                                      
    for idiff in range(len(Idiff)):
        Ediff_1[idiff] = int_2d_diff(Idiff[idiff,:,:],-cos_poa_diff*trans_diff_1,umu,phi_array)
        Ediff_2[idiff] = int_2d_diff(Idiff[idiff,:,:],-cos_poa_diff*trans_diff_2,umu,phi_array)
            
    diff_result = (Ediff_2 - Ediff_1)/(2*d_n)

    return diff_result

def d_E_dtheta(state,Edirdown,Idiff,sun,angles,d_tilt,optics,optical_model_flag):
    """
    Derivative of total plane-of-array irradiance Etotpoa 
    with respect to tilt angle theta
    
    args:
        :param state, array of floats representing parameter state space
        :param Edirdown, vector of floats, direct downward irradiance
        :param Idiff, 3D array of floats, (time,theta,phi), diffuse radiance field
        :param alb, float, surface albedo
        :param sun, named tuple containing sun position vectors
        :param angles, named tuple containing angular grid for integration
        :param d_tilt, delta in tilt angle for numerical differentiation
        :param optics, named tuple containing optical parameters
        :param optical_model_flag, boolean for optical model
        
    out:
        :return: dictionary of vectors of floats
                 'dEtot_dtheta': derivative of Etotpoa wrt tilt angle theta
                 'dEdir_dtheta': derivative of Edirpoa wrt tilt angle theta
                 'dEdiff_dtheta': derivative of Ediffpoa wrt tilt angle theta
    """
    
    #Extract parameters from state variable x
    tilt = state[0]
    azimuth = state[1]
    if optical_model_flag:
        n = state[2]
    else:
        n = 1.0
        
    
    #Sun position angles
    theta0 = sun.sza
    phi0 = sun.azimuth
    
    #Cosine factor for direct beam
    cos_factor = cos_incident_angle(theta0,phi0,tilt,azimuth)
    
    #Derivative of Edirpoa with respect to theta
    if optical_model_flag:
        d_direct = Edirdown*dcos_ia_dtheta(theta0,phi0,tilt,azimuth)/cos(theta0)*\
                ang_response(arccos(cos_factor),n,optics.kappa,optics.L)\
                + cos_factor/cos(theta0)*Edirdown*\
                d_taupv_dcostheta(cos_factor,n,optics.kappa,optics.L)*\
                dcos_ia_dtheta(theta0,phi0,tilt,azimuth)
    else:
        d_direct = Edirdown*dcos_ia_dtheta(theta0,phi0,tilt,azimuth)/cos(theta0)

    #Numerical derivative of Ediffpoa with respect to theta                
    d_diffuse = d_Ediff_dtheta(Idiff,angles,tilt,azimuth,d_tilt,optics,n,optical_model_flag)
    
#    #Derivative of reflected component
#    if optical_model_flag:
#        angle_refl = pi/2 - 0.5788*tilt + rad2deg(0.002693)*tilt**2
#        trans_refl = ang_response(angle_refl,n,optics.kappa,optics.L)
#        d_refl = alb*sin(tilt)*Edirdown*trans_refl/2.0 + \
#             alb*(1-cos(tilt))/2.0*Edirdown*(-0.5788 + 2*rad2deg(0.002693)*tilt)
#    else:
#        d_refl = alb*sin(tilt)*Edirdown/2.0
#    
    #Total derivative
    diff_result = d_direct + d_diffuse #+ d_refl
    
    return {'dEtot_dtheta': diff_result, 'dEdir_dtheta': d_direct, 
            'dEdiff_dtheta': d_diffuse} #, 'dErefl_dtheta' : d_refl}

def d_E_dphi(state,Edirdown,Idiff,sun,angles,d_azi,optics,optical_model_flag):
    """
    Derivative of total plane-of-array irradiance Etotpoa 
    with respect to tilt angle theta
    
    args:
        :param state, array of floats representing parameter state space
        :param Edirdown, vector of floats, direct downward irradiance
        :param Idiff, 3D array of floats, (time,theta,phi), diffuse radiance field        
        :param sun, named tuple containing sun position vectors
        :param angles, named tuple containing angular grid for integration
        :param d_azi, delta in azimuth angle for numerical differentiation
        :param optics, named tuple containing optical parameters
        :param optical_model_flag, boolean for optical model
        
    out:
        :return: dictionary of vectors of floats
                 'dEtot_dphi': derivative of Etotpoa wrt tilt angle phi
                 'dEdir_dphi': derivative of Edirpoa wrt tilt angle phi
                 'dEdiff_dphi': derivative of Ediffpoa wrt tilt angle phi
    """
    
    #Extract parameters from state variable x
    tilt = state[0]
    azimuth = state[1]
    if optical_model_flag:
        n = state[2]
    else:
        n = 1.0
    
    #Sun position angles
    theta0 = sun.sza
    phi0 = sun.azimuth
    
    #Cosine factor for direct beam
    cos_factor = cos_incident_angle(theta0,phi0,tilt,azimuth)
    
    #Derivative of Edirpoa with respect to phi
    if optical_model_flag:
        d_direct = Edirdown*dcos_ia_dphi(theta0,phi0,tilt,azimuth)/cos(theta0)*\
               ang_response(arccos(cos_factor),n,optics.kappa,optics.L)\
               + cos_factor/cos(theta0)*Edirdown*\
               d_taupv_dcostheta(cos_factor,n,optics.kappa,optics.L)*\
               dcos_ia_dphi(theta0,phi0,tilt,azimuth)
    else:
       d_direct = Edirdown*dcos_ia_dphi(theta0,phi0,tilt,azimuth)/cos(theta0)   
    
    #Numerical derivative of Ediffpoa with respect to phi
    d_diffuse = d_Ediff_dphi(Idiff,angles,tilt,azimuth,d_azi,optics,n,optical_model_flag)
    
    #Derivative of reflection term is independent of azimuth
    #d_refl = 0
    
    #Total derivative
    diff_result = d_direct + d_diffuse #+ d_refl
    
    return {'dEtot_dphi': diff_result, 'dEdir_dphi': d_direct, 
            'dEdiff_dphi': d_diffuse} #, 'dErefl_dphi' : d_refl}
    
def d_E_dn(state,Edirdown,Idiff,sun,angles,d_n,optics):
    """
    Derivative of total plane-of-array irradiance Etotpoa 
    with respect to angle of refraction
    
    args:
        :param state: array of floats representing parameter state space
        :param Edirdown: vector of floats, direct downward irradiance
        :param Idiff: 3D array of floats, (time,theta,phi), diffuse radiance field
        :param sun: named tuple containing sun position vectors
        :param angles: named tuple containing angular grid for integration
        :param d_n: delta in n for numerical differentiation
        :param optics: named tuple containing optical parameters
        
    out:
        :return: dictionary of vectors of floats
                 'dEtot_dn': derivative of Etotpoa wrt refractive index n
                 'dEdir_dn': derivative of Edirpoa wrt refractive index n
                 'dEdiff_dn': derivative of Ediffpoa wrt refractive index n
    """
    
    #Extract parameters from state
    tilt = state[0]
    azimuth = state[1]
    n = state[2]
    
    #Sun position angles
    theta0 = sun.sza
    phi0 = sun.azimuth
    
    #Cosine factor for direct beam
    cos_factor = cos_incident_angle(theta0,phi0,tilt,azimuth)
    
    #Derivative of Edirpoa with respect to n
    d_direct = Edirdown*cos_factor/cos(theta0)\
               *d_taupv_dn(cos_factor,n,optics.kappa,optics.L)
               
    #Numerical derivative of Ediffpoa with respect to phi
    d_diffuse = d_Ediff_dn(Idiff,angles,tilt,azimuth,d_n,n,optics)
    
#    angle_refl = pi/2 - 0.5788*tilt + rad2deg(0.002693)*tilt**2
#    d_refl = alb*(1-cos(tilt))/2*Edirdown*\
#             d_taupv_dn(cos(angle_refl),n,optics.kappa,optics.L)
#               
    diff_result = d_direct + d_diffuse #+ d_refl
    
    return {'dEtot_dn': diff_result, 'dEdir_dn': d_direct, 
            'dEdiff_dn': d_diffuse} #, 'dErefl_dphi' : d_refl}

def E_poa_calc(state,Edirdown,Idiff,sun,angles,optics,deltas=None,optical_model_flag=True):
    """
    Calculate the irradiance in the plane-of-array for direct, diffuse and 
    reflected (direct part) components
    
    args:
    :param state: state vector of x parameter space for optimisation
    :param Edirdown: array of floats, downward irradiance from libradtran simulation
    :param Idiff: 3D array, diffuse radiance distribution    
    :param sun: collections.namedtuple with sun position    
    :param angles: collections.namedtuple with angles for Idiff integration
    :param optics: collections.namedtuple with optical properties
    :param deltas: collections.namedtuple with increments for numerical differentiation,
                   if None then do not perform differentiation
    :param optical_model flag: boolean to determine whether optical model is on or off
    
     out:
    :return: dictionary of vectors of floats and differential dictionaries
             'Etotpoa': vector, Total plane-of-array irradiance
             'Edirpoa': vector,Direct plane-of-array irradiance
             'Ediffpoa': vector, Diffuse plane-of-array irradiance
             'Etotpoa_pv': vector, Total plane-of-array irradiance on PV cell
             'Edirpoa_pv': vector, Direct plane-of-array irradiance on PV cell
             'Ediffpoa_pv': vector, Diffuse plane-of-array irradiance on PV cell
             'dEs_theta': dictionary of derivatives wrt. theta
             'dEs_phi': dictionary of derivatives wrt. phi
             'dEs_n': dictionary of derivatives wrt. n
             'trans_dir': vector of transmission function for direct irradiance
             'cos_poa_dir' : vector of cosine of incident angles
    """
    
    #Extract parameters from the state variable x
    tilt = state[0]
    azimuth = state[1]
    
    #Sun position angles
    theta0 = sun.sza
    phi0 = sun.azimuth
    
    #Angles for integration of Idiff over a 2D grid
    theta_array = angles.theta
    phi_array = angles.phi
    umu = angles.umu
    
    #Small changes for numerical differentiation
    if deltas:
        d_theta = deltas.theta
        d_phi = deltas.phi
    
    if optical_model_flag:
        n = state[2]
        if deltas:
            d_n = deltas.n    
            #Calculate derivatives of irradiance wrt n    
            diffs_n = d_E_dn(state,Edirdown,Idiff,sun,angles,d_n,optics)
        else:
            diffs_n = None
    else:
        diffs_n = None
    
    #rotate direct irradiance into the plane-of-array
    cos_poa_dir = cos_incident_angle(theta0,phi0,tilt,azimuth)
    
    #Calculate transmission function
    if optical_model_flag:
        trans_dir = ang_response(arccos(cos_poa_dir),n,optics.kappa,optics.L)
    else:
        trans_dir = 1.0
    
    #Calculate direct irradiance in the plane-of-array (before optical model)
    if len(Edirdown.shape) == 2:
        Edirpoa = cos_poa_dir[:,np.newaxis]/cos(theta0)[:,np.newaxis]*Edirdown
    else:
        Edirpoa = cos_poa_dir/cos(theta0)*Edirdown
    
    #Direct irradiance in the plane-of-array (after optical model)
    if len(Edirpoa.shape) == 2 and type(trans_dir) != float:
        Edirpoa_pv = Edirpoa*trans_dir[:,np.newaxis]
    else:
        Edirpoa_pv = Edirpoa*trans_dir
        
    #cos factor for rotated diffuse radiance field
    cos_poa_diff = np.zeros((len(theta_array),len(phi_array)))
    trans_diff = np.zeros((len(theta_array),len(phi_array)))
    
    #Cosine factor and transmission function for diffuse field
    for itheta in range(len(theta_array)):
        for iphi in range(len(phi_array)):
            cos_poa_diff[itheta,iphi] = cos_incident_angle(theta_array[itheta],
               phi_array[iphi],tilt,azimuth)
            if optical_model_flag:
                trans_diff[itheta,iphi] = ang_response(arccos(cos_poa_diff[itheta,iphi]),
                      n,optics.kappa,optics.L)
            else:
                trans_diff[itheta,iphi] = 1.0
    
    #Calculate diffuse component by integrating over theta and umu
    #For before and after optical model
    Ediffpoa = int_2d_diff(Idiff,-cos_poa_diff*1.0,umu,phi_array) 
    Ediffpoa_pv = int_2d_diff(Idiff,-cos_poa_diff*trans_diff,umu,phi_array) 
        
    # for idiff in range(len(Idiff)):
    #     Ediffpoa[idiff] = int_2d_diff(Idiff[idiff,:,:],-cos_poa_diff*1.0,umu,phi_array) 
    #     Ediffpoa_pv[idiff] = int_2d_diff(Idiff[idiff,:,:],-cos_poa_diff*trans_diff,umu,phi_array) 
    
#   Don't need this part since it is included in the diffuse radiance distribution from DISORT,
#   since we calculated for the entire sphere, including reflection from the ground
#    #Angle for transmission function, from Duffie & Beckman
#    angle_refl = pi/2 - 0.5788*tilt + rad2deg(0.002693)*tilt**2
#    #Transmission function for reflected component
#    if optical_model_flag:
#        trans_refl = ang_response(angle_refl,n,optics.kappa,optics.L)
#    else:
#        trans_refl = 1.0
#    
#    #Reflected component of direct beam
#    Ereflpoa = alb*(1-cos(tilt))/2*Edirdown*trans_refl
    
    #Total plane-of-array irradiance before optical model
    Etotpoa = Edirpoa + Ediffpoa # +  Ereflpoa
    
    #Total plane-of-array irradiance after optical model
    Etotpoa_pv = Edirpoa_pv + Ediffpoa_pv # +  Ereflpoa
    
    if deltas:
        #Calculate derivatives of irradiance wrt theta
        diffs_theta = d_E_dtheta(state,Edirdown,Idiff,sun,angles,d_theta,optics,optical_model_flag)
    else:
        diffs_theta = None
        
    if deltas:
        #Calculate derivatives of irradiance wrt phi
        diffs_phi = d_E_dphi(state,Edirdown,Idiff,sun,angles,d_phi,optics,optical_model_flag)
    else:
        diffs_phi = None
        
    
    return {'Etotpoa': Etotpoa, 'Edirpoa': Edirpoa, 'Ediffpoa': Ediffpoa, #'Ereflpoa': Ereflpoa, 
            'Etotpoa_pv': Etotpoa_pv, 'Edirpoa_pv': Edirpoa_pv, 'Ediffpoa_pv': Ediffpoa_pv,
                  'dEs_theta': diffs_theta, 'dEs_phi': diffs_phi, 'dEs_n': diffs_n,
                  'trans_dir': trans_dir, 'cos_poa_dir' : cos_poa_dir}
    
def temp_model_tamizhmani(params,temp_amb,irrad_total,vwind,temp_sky):
    """
    Temperature model as taken from the Tamizhmani et. al. paper 
    args:
        :param params, array of floats representing parameter state space        
        :param temp_amb, vector of floats, ambient temperature in Celsisus
        :param irrad, vector of floats, total irradiance in W/m^2
        :param vwind, vector of floats, windspeed in m/s    
        :param temp_sky, vector of floats, sky temperature in Celsisus
        
    out:
        :return: dictionary of modelled temperature and derivatives    
    """
        
    u0 = params[0]
    u1 = params[1]
    u2 = params[2]
    u3 = params[3]
    
    T_module = u0*temp_amb + u1*irrad_total + u2*vwind + u3*temp_sky #Module temperature
    
    d_Tmod_d_u0 = temp_amb
    
    d_Tmod_d_u1 = irrad_total
    
    d_Tmod_d_u2 = vwind
    
    d_Tmod_d_u3 = temp_sky
   
    d_Tmod_d_E = u1
    
    return {'T_mod':T_module, 'diff_u0':d_Tmod_d_u0,'diff_u1':d_Tmod_d_u1,
            'diff_u2':d_Tmod_d_u2, 'diff_E': d_Tmod_d_E,'diff_u3':d_Tmod_d_u3}
    
def temp_model_faiman(params,temp_amb,irrad_total,vwind,temp_sky):
    """
    Temperature model as taken from Faiman
    args:
        :param params, array of floats representing parameter state space        
        :param temp_amb, vector of floats, ambient temperature in Celsisus
        :param irrad, vector of floats, total irradiances
        :param vwind, vector of floats, windspeed in m/s     
        :param temp_sky, vector of floats, sky temperature in Celsisus
        
    out:
        :return: dictionary of modelled temperature and derivatives    
    """
        
    u1 = params[0]
    u2 = params[1]       
    u3 = params[2]     
    
    #Faiman model for module temperature
    T_module = temp_amb + irrad_total/(u1 + u2*vwind) + u3*(temp_sky - temp_amb)
    
    d_Tmod_d_u1 = -irrad_total/(u1 + u2*vwind)**2
    
    d_Tmod_d_u2 = -irrad_total*vwind/(u1 + u2*vwind)**2
    
    d_Tmod_d_u3 = temp_sky - temp_amb
    
    d_Tmod_d_E = 1./(u1 + u2*vwind)
    
    return {'T_mod':T_module, 'diff_u1':d_Tmod_d_u1,'diff_u2':d_Tmod_d_u2, 
            'diff_u3':d_Tmod_d_u3, 'diff_E': d_Tmod_d_E}
    
def temp_model_king(params,temp,irrad_total,vwind):
    """
    Temperature model as taken from King
    args:
        :param params, array of floats representing parameter state space        
        :param temp, vector of floats, ambient temperature in Celsisus
        :param irrad_total, vector of floats, total irradiances
        :param vwind, vector of floats, windspeed in m/s        
        
    out:
        :return: dictionary of modelled temperature and derivatives    
    """
        
    a = params[0]
    b = params[1]        
    dT = params[2]
    
    #King's formula for the cell temperature
    T_module = temp + irrad_total*np.exp(a + b*vwind) + irrad_total/1000*dT
    
    d_Tmod_d_a = T_module - temp
    
    d_Tmod_d_b = vwind*(T_module - temp)
    
    d_Tmod_d_E = np.exp(a + b*vwind) + dT/1000
    
    d_Tmod_d_dT = irrad_total/1000
    
    return {'T_mod':T_module, 'diff_a':d_Tmod_d_a,'diff_b':d_Tmod_d_b, 
            'diff_dT': d_Tmod_d_dT, 'diff_E': d_Tmod_d_E}
            
def P_mod_simple_cal(state,Edirdown_pv,Edirdown_pyr,Idiff_pv,Idiff_pyr,alb,n_h2o,
                     df_spectral_fit,temp_module,temp_amb,wind,temp_sky,sun,
                     angles,optics,invdict,K,T_model,eff_model):
    """
    Define a simple model of PV power as a function of plane-of-array irradiance and temperature
    P = Pdcn*eff_temp(Etotpoa,Tamb,vwind)*Etotpoa
    The function P_mod_simple takes the direct irradiance and diffuse radiance field 
    from libradtran and calculates the irradiance in the plane-of-array, Etotpoa.
    Together with the temperature this gives the modelled power in Watts
    
    args:
        :param state, array of floats representing parameter state space
        :param Edirdown_pv, vector of floats, direct downward irradiance for PV range
        :param Edirdown_pyr, vector of floats, direct downward irradiance for broadband range
        :param Idiff_pv, 3D array of floats, (time,theta,phi), diffuse radiance field for PV range
        :param Idiff_pyr, 3D array of floats, (time,theta,phi), diffuse radiance field for broadband range
        :param alb, float, surface albedo
        :param n_h2o, vector of floats, precipitable water vapour from COSMO
        :param df_spectral_fit, dataframe with spectral mismatch fit
        :param temp_module, vector of floats, module temperature in Celsisus
        :param temp_amb, vector of floats, ambient temperature in Celsisus
        :param wind, vector of floats, windspeed in m/s
        :param temp_sky, vector of floats, sky temperature in Celsisus
        :param sun, named tuple containing sun position vectors
        :param angles, named tuple containing angular grid for integration
        :param optics, named tuple containing optical parameters
        :param invdict, dictionary with info for inversion
        :param K, Jacobian matrix
        :param T_model: string, temperature model
        :param eff_model: string, efficiency model
        
    out:
        :return: dictionary of vectors of floats
                 'P_mod': modelled PV power in Watts
                 'Etotpoa': total plane-of-array irradiance
                 'Edirpoa': direct irradiance in plane-of-array
                 'Ediffpoa': diffuse irradiance in plane-of-array
                 'Etotpoa_pv': vector, Total plane-of-array irradiance on PV cell
                 'Edirpoa_pv': vector, Direct plane-of-array irradiance on PV cell
                 'Ediffpoa_pv': vector, Diffuse plane-of-array irradiance on PV cell
                 'diffPE': derivative of P with respect to Etotpoa
                 'T_module': vector, module temperature
                  'eff_temp': vector, temperature dependent efficiency 
                  'K_mat': matrix, Jacobian matrix 
                  'pars': array of all model params
    """
    
    #Extract parameters from the state variable x 
    irrad_params = []
    eff_params = []
    tmod_params = []
    c_par = 0
    #Here we take parameters either from state update or keep them fixed
    #Depending on the uncertainty in config file
    for i, par in enumerate(invdict['pars']):
        if par[0] in ['theta','phi','n']:
            if par[2] == 0:
                irrad_params.append(par[1])
            else:
                irrad_params.append(state[c_par])
                c_par = c_par + 1
        if par[0] == 'pdcn':
            if par[2] == 0:
                pdcn = par[1]
            else:    
                pdcn = state[c_par]
                c_par = c_par + 1
        if par[0] == 'zeta':
            if par[2] == 0:
                zeta = par[1]
            else:        
                zeta = state[c_par]
                c_par = c_par + 1
        
        if i >= 5:
            if eff_model == "Beyer":
                if 5 <= i <= 7:
                    if par[2] == 0:
                        eff_params.append(par[1])
                    else:
                        eff_params.append(state[c_par])
                        c_par = c_par + 1
                else:
                    if par[2] == 0:
                        tmod_params.append(par[1])
                    else:
                        tmod_params.append(state[c_par])
                        c_par = c_par + 1
                                    
            elif eff_model == "Ransome":
                if 5 <= i <= 6:
                    if par[2] == 0:
                        eff_params.append(par[1])
                    else:
                        eff_params.append(state[c_par])
                        c_par = c_par + 1
                else:
                    if par[2] == 0:
                        tmod_params.append(par[1])
                    else:
                        tmod_params.append(state[c_par])
                        c_par = c_par + 1
                            
            else:    
                if par[2] == 0:
                    tmod_params.append(par[1])
                else:
                    tmod_params.append(state[c_par])
                    c_par = c_par + 1

    params = np.hstack([irrad_params,pdcn,zeta,eff_params,tmod_params])
    
    #named tuple containing deltas for numerical differentiation
    deltas = invdict["diffs"]
    
    #Calculate plane-of-array irradiance, including optical model for glass surfauce of panel
    #1. This irradiance is the photovoltaic irradiance, i.e. for the spectral range of PV
    irrad_pv = E_poa_calc(irrad_params,Edirdown_pv,Idiff_pv,sun,angles,optics,deltas
                       ,optical_model_flag=True)
    
    #2. This is the broadband irradiance, for the temperature model 
    irrad_temp = E_poa_calc(irrad_params,Edirdown_pyr,Idiff_pyr,sun,angles,optics,deltas
                       ,optical_model_flag=True)
                
    #Model the module temperature as a function of ambient T, irradiance and windspeed
    if T_model == "Tamizhmani":
        temp_model = temp_model_tamizhmani(tmod_params,temp_amb,irrad_temp['Etotpoa_pv'],wind,temp_sky)        
    elif T_model == "King":
        temp_model = temp_model_king(tmod_params,temp_amb,irrad_temp['Etotpoa_pv'],wind)        
    elif T_model == "Faiman" or T_model == "Barry":
        temp_model = temp_model_faiman(tmod_params,temp_amb,irrad_temp['Etotpoa_pv'],wind,temp_sky)    
    #In this case the temperature is measured
    elif T_model == "Dynamic_or_Measured":
        temp_model = {'T_mod':temp_module,'diff_E':0.0}
    
    #Calculate the efficiency correction due to module temperature
    if eff_model == "Evans":
        eff_temp_corr = 1.0 - zeta*(temp_model['T_mod'] - 25.0)    
        #Derivative of effiency wrt irradiance
        d_eff_dE_all = -zeta*temp_model['diff_E']
        d_eff_dE_pv = 0.
        
    elif eff_model == "Beyer":
        a1 = eff_params[0]
        a2 = eff_params[1]
        a3 = eff_params[2]
        eff_mpp = (a1 + a2*irrad_pv["Etotpoa_pv"] + a3*np.log(irrad_pv["Etotpoa_pv"]))
        eff_temp_corr = eff_mpp*(1.0 - zeta*(temp_model['T_mod'] - 25.0))            
        d_eff_dE = (a2 + a3/irrad_pv["Etotpoa_pv"])*(1.0 - zeta*(temp_model['T_mod'] - 25.0))\
                   - zeta*temp_model['diff_E']*eff_mpp
                   
    elif eff_model == "Ransome":
        c3 = eff_params[0]
        c6 = eff_params[1]
        eff_temp_corr = 1.0 - zeta*(temp_model['T_mod'] - 25.0)\
            + c3*np.log(irrad_pv["Etotpoa_pv"]) + c6/irrad_pv["Etotpoa_pv"]
        d_eff_dE = -zeta*temp_model['diff_E'] + c3/irrad_pv["Etotpoa_pv"]\
            - c6/(irrad_pv["Etotpoa_pv"]**2)
    
    #Final modelled PV power
    P_model = pdcn*eff_temp_corr*irrad_pv["Etotpoa_pv"]
    
    #Derivative of P with respect to module temperature
    dP_d_Tmod = -pdcn*zeta*irrad_pv["Etotpoa_pv"]
    
    #Derivative of P with respect to Etotpoa (both all and PV)
    dP_dE_all = pdcn*(d_eff_dE_all)*irrad_pv["Etotpoa_pv"]
    
    dP_dE_pv = pdcn*(d_eff_dE_pv*irrad_pv["Etotpoa_pv"] + eff_temp_corr)
    
    #Calculate columns of K matrix, derivative of P with respect to ...
    K_full = np.zeros((len(P_model),len(invdict['pars'])))    
    
    #Elevation angle theta
    K_full[:,0] = dP_dE_pv*irrad_pv['dEs_theta']['dEtot_dtheta']\
                    + dP_dE_all*irrad_temp['dEs_theta']['dEtot_dtheta']
    #Azimuth angle phi            
    K_full[:,1] = dP_dE_pv*irrad_pv['dEs_phi']['dEtot_dphi']\
                   + dP_dE_all*irrad_temp['dEs_phi']['dEtot_dphi']
    #Refractive index n
    K_full[:,2] = dP_dE_pv*irrad_pv['dEs_n']['dEtot_dn']\
                    + dP_dE_all*irrad_temp['dEs_n']['dEtot_dn']
    #Constant s
    K_full[:,3] = irrad_pv['Etotpoa_pv']*eff_temp_corr
    #Constant zeta
    K_full[:,4] = dP_d_Tmod*(temp_model['T_mod'] - 25)
    
    if eff_model == "Beyer":
        K_full[:,5] = pdcn*eff_temp_corr/eff_mpp*irrad_pv["Etotpoa_pv"]
        K_full[:,6] = pdcn*eff_temp_corr/eff_mpp*irrad_pv["Etotpoa_pv"]**2
        K_full[:,7] = pdcn*eff_temp_corr/eff_mpp*irrad_pv["Etotpoa_pv"]*\
        np.log(irrad_pv["Etotpoa_pv"])
        col_index = 8
    elif eff_model == "Ransome":
        K_full[:,5] = pdcn*irrad_pv["Etotpoa_pv"]*np.log(irrad_pv["Etotpoa_pv"])
        K_full[:,6] = pdcn
        col_index = 7
    else:
        col_index = 5
    
    if T_model == "Tamizhmani":
        #Constant u0 
        K_full[:,col_index] = dP_d_Tmod*temp_model['diff_u0']
        #Constant u1
        K_full[:,col_index + 1] = dP_d_Tmod*temp_model['diff_u1']
        #Constant u2
        K_full[:,col_index + 2] = dP_d_Tmod*temp_model['diff_u2']
        #Constant u3
        K_full[:,col_index + 3] = dP_d_Tmod*temp_model['diff_u3']
                
    elif T_model == "King":
        K_full[:,col_index] = dP_d_Tmod*temp_model['diff_a']        
        K_full[:,col_index + 1] = dP_d_Tmod*temp_model['diff_b']
        K_full[:,col_index + 2] = dP_d_Tmod*temp_model['diff_dT']        
        
    elif T_model == "Faiman" or T_model == "Barry":
        K_full[:,col_index] = dP_d_Tmod*temp_model['diff_u1']        
        K_full[:,col_index + 1] = dP_d_Tmod*temp_model['diff_u2']   
        K_full[:,col_index + 2] = dP_d_Tmod*temp_model['diff_u3']   
        
    #Remove columns of K for parameters that are fixed
    K = np.delete(K_full,[i for i, par in enumerate(invdict["pars"]) if par[2] == 0],axis=1)
            
    return {'P_mod': P_model, 'Etotpoa': irrad_pv['Etotpoa'], 'Edirpoa': irrad_pv['Edirpoa'],
                  'Ediffpoa': irrad_pv['Ediffpoa'], 'Etotpoa_pv': irrad_pv['Etotpoa_pv'], 
                  'Edirpoa_pv': irrad_pv['Edirpoa_pv'], 'Ediffpoa_pv': irrad_pv['Ediffpoa_pv'], 
                  'diffPE': dP_dE_pv, 'T_module': temp_model['T_mod'], 
                  'eff_temp': eff_temp_corr, 'K_mat': K, 'pars': params}
    
def I_mod_simple_diode_cal(state,Edirdown_pv,Edirdown_pyr,Idiff_pv,Idiff_pyr,alb,
                           temp_module,temp_amb,wind,temp_sky,sun,
                           angles,optics,invdict,K,T_model):
    """

    Using the photocurrent term from the diode model equation, in order to extract irradiance
    I_DC = Etotpoa_pv/1000*impn*(1. + ki*(T_module - 25.))*npp
    where Etotpoa_pv is the irradiance under the glass surface
    impn is the short current at STC
    ki is the current-based temperature coefficient in 1/C
    T_module is the module temperature
    npp is the number of modules in parallel
    
    args:
        :param state, array of floats representing parameter state space
        :param Edirdown_pv, vector of floats, direct downward irradiance for PV range
        :param Edirdown_pyr, vector of floats, direct downward irradiance for broadband range
        :param Idiff_pv, 3D array of floats, (time,theta,phi), diffuse radiance field for PV range
        :param Idiff_pyr, 3D array of floats, (time,theta,phi), diffuse radiance field for broadband range
        :param alb, float, surface albedo        
        :param temp_module, vector of floats, module temperature in Celsisus
        :param temp_amb, vector of floats, ambient temperature in Celsisus
        :param wind, vector of floats, windspeed in m/s
        :param temp_sky, vector of floats, sky temperature in Celsisus
        :param sun, named tuple containing sun position vectors
        :param angles, named tuple containing angular grid for integration
        :param optics, named tuple containing optical parameters
        :param invdict, dictionary with info for inversion
        :param K, Jacobian matrix
        :param T_model: string, temperature model


    out:
        :return: dictionary of vectors of floats
                 'I_mod': modelled PV current in Amps
                 'Etotpoa': total plane-of-array irradiance
                 'Edirpoa': direct irradiance in plane-of-array
                 'Ediffpoa': diffuse irradiance in plane-of-array
                 'Etotpoa_pv': vector, Total plane-of-array irradiance on PV cell
                 'Edirpoa_pv': vector, Direct plane-of-array irradiance on PV cell
                 'Ediffpoa_pv': vector, Diffuse plane-of-array irradiance on PV cell
                 'diffIE': derivative of I with respect to Etotpoa
                 'T_module': vector, module temperature                  
                  'K_mat': matrix, Jacobian matrix 
                  'pars': array of all model params


    """
    
    #Extract parameters from the state variable x 
    irrad_params = []    
    tmod_params = []
    c_par = 0
    #Here we take parameters either from state update or keep them fixed
    #Depending on the uncertainty in config file
    for i, par in enumerate(invdict['pars']):
        if par[0] in ['theta','phi','n']:
            if par[2] == 0:
                irrad_params.append(par[1])
            else:
                irrad_params.append(state[c_par])
                c_par = c_par + 1
        
        if par[0] == 'impn':
            if par[2] == 0:
                impn = par[1]
            else:    
                impn = state[c_par]
                c_par = c_par + 1
        if par[0] == 'npp':
            if par[2] == 0:
                npp = par[1]
            else:        
                npp = state[c_par]
                c_par = c_par + 1
        if par[0] == 'ki':
            if par[2] == 0:
                ki = par[1]
            else:        
                ki = state[c_par]
                c_par = c_par + 1
        
        if i >= 6:
            if par[2] == 0:
                tmod_params.append(par[1])
            else:
                tmod_params.append(state[c_par])
                c_par = c_par + 1

    params = np.hstack([irrad_params,impn,npp,ki,tmod_params])
    
    #named tuple containing deltas for numerical differentiation
    deltas = invdict["diffs"]
    
    #Calculate plane-of-array irradiance
    irrad_pv = E_poa_calc(irrad_params,Edirdown_pv,Idiff_pv,sun,angles,optics,deltas
                       ,optical_model_flag=True)
        
    irrad_temp = E_poa_calc(irrad_params,Edirdown_pyr,Idiff_pyr,sun,angles,optics,deltas
                       ,optical_model_flag=True)
    
    #Model the module temperature as a function of ambient T, irradiance and windspeed
    if T_model == "Tamizhmani":
        temp_model = temp_model_tamizhmani(tmod_params,temp_amb,irrad_temp['Etotpoa_pv'],wind,temp_sky)        
    elif T_model == "King":
        temp_model = temp_model_king(tmod_params,temp_amb,irrad_temp['Etotpoa_pv'],wind)        
    elif T_model == "Faiman" or T_model == "Barry":
        temp_model = temp_model_faiman(tmod_params,temp_amb,irrad_temp['Etotpoa_pv'],wind,temp_sky)    
    #In this case the temperature is measured
    elif T_model == "Dynamic_or_Measured":
        temp_model = {'T_mod':temp_module,'diff_E':0.0}
       
    #Final modelled PV current
    I_model = irrad_pv['Etotpoa_pv']/1000.*impn*npp*(1. + ki*(temp_model["T_mod"] - 25.))
    
    #Derivative of P with respect to module temperature
    dI_d_Tmod = irrad_pv['Etotpoa_pv']/1000.*impn*npp*ki
    
    #Derivative of P with respect to Etotpoa    
    dI_dE_pv = impn*npp*(1. + ki*(temp_model["T_mod"] - 25.))/1000.
    
    dI_dE_all = dI_d_Tmod*temp_model["diff_E"]
    
    #Calculate columns of K matrix, derivative of P with respect to ...
    K_full = np.zeros((len(I_model),len(invdict['pars'])))      
    
    #Elevation angle theta
    K_full[:,0] = dI_dE_pv*irrad_pv['dEs_theta']['dEtot_dtheta'] +\
        dI_dE_all*irrad_temp['dEs_theta']['dEtot_dtheta']
    #Azimuth angle phi            
    K_full[:,1] = dI_dE_pv*irrad_pv['dEs_phi']['dEtot_dphi'] + \
        dI_dE_all*irrad_temp['dEs_phi']['dEtot_dphi']
    #Refractive index n
    K_full[:,2] = dI_dE_pv*irrad_pv['dEs_n']['dEtot_dn'] + \
        dI_dE_all*irrad_temp['dEs_n']['dEtot_dn']
    #Isc
    K_full[:,3] = irrad_pv['Etotpoa_pv']/1000.*npp*(1. + ki*(temp_model["T_mod"] - 25.))
    #npp
    K_full[:,4] = irrad_pv['Etotpoa_pv']/1000.*impn*(1. + ki*(temp_model["T_mod"] - 25.))
    #ki
    K_full[:,5] = irrad_pv['Etotpoa_pv']/1000.*impn*npp*temp_model["T_mod"]
            
    if T_model == "Tamizhmani":
        #Constant u0 
        K_full[:,6] = dI_d_Tmod*temp_model['diff_u0']
        #Constant u1
        K_full[:,7] = dI_d_Tmod*temp_model['diff_u1']
        #Constant u2
        K_full[:,8] = dI_d_Tmod*temp_model['diff_u2']
        #Constant u3
        K_full[:,9] = dI_d_Tmod*temp_model['diff_u3']
                
    elif T_model == "King":
        K_full[:,6] = dI_d_Tmod*temp_model['diff_a']        
        K_full[:,7] = dI_d_Tmod*temp_model['diff_b']
        K_full[:,8] = dI_d_Tmod*temp_model['diff_dT']        
        
    elif T_model == "Faiman" or T_model == "Barry":
        K_full[:,6] = dI_d_Tmod*temp_model['diff_u1']        
        K_full[:,7] = dI_d_Tmod*temp_model['diff_u2']   
        K_full[:,8] = dI_d_Tmod*temp_model['diff_u3']   
        
    #Remove columns of K that are fixed
    K = np.delete(K_full,[i for i, par in enumerate(invdict["pars"]) if par[2] == 0],axis=1)
            
    return {'I_mod': I_model, 'Etotpoa': irrad_pv['Etotpoa'], 'Edirpoa': irrad_pv['Edirpoa'],
                  'Ediffpoa': irrad_pv['Ediffpoa'], 'Etotpoa_pv': irrad_pv['Etotpoa_pv'], 
                  'Edirpoa_pv': irrad_pv['Edirpoa_pv'], 'Ediffpoa_pv': irrad_pv['Ediffpoa_pv'], 
                  'diffIE': dI_dE_pv, 'T_module': temp_model['T_mod'], 
                  'K_mat': K, 'pars': params}
    
    
