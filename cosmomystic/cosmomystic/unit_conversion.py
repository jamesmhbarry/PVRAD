import numpy as np

from .constants import *






########################################################
###             Temperature conversion               ###
########################################################

def K2C(T):
    """
    args:
    :param T: Temperature in deg Kelvin
    
    out:
    :return: Temperature in deg Celsius
    """
    
    return T - 273.15


def C2K(T):
    """
    args:
    :param T: Temperature in deg Celsius
    
    out:
    :return: Temperature in Kelvin
    """
    
    return T + 273.15



########################################################
###         pascal to hektopascal and back           ###
########################################################

def hPa2Pa(p):
    return p*100.

def Pa2hPa(p):
    return p/100.


########################################################
###         watervapor and humidity measures         ###
########################################################

def magnus(T, phase="water"):
    """
    args:
    :param T: Temperature in Kelvin
    :param phase: options are ice and water to calculate 
    the vapor pressure above water or ice surfaces
    out:
    :return: saturation vapor pressure above liquid water/ice in pascal
    
    """
    Tdeg = K2C(T)
    
    if phase == "water":
        return 611.2 * np.exp(17.62*Tdeg / (243.12+Tdeg))
    
    elif phase == "ice":
        return 611.2 * np.exp(22.46*Tdeg / (272.62+Tdeg))
    
    else:
        print("ERROR in function magnus!!")
        print("given argument for phase is not supported")
        print("check input!")
        print("program exits!")
        quit()
        
########################################################
###               virtual Temperature                ###
########################################################


def virtual_temperature(T, p, moist_meas, measure="relhum", phase="water"):
    """
    args:
    :param T: air Temperature in [K] (float or array)
    :param p: air pressure in [Pa] 
    :param moist_meas: one moisture measure: currently supported:
                       dewpoint temperature in [K]and relative humidity in [pct]
                       --> give associated keyword for measure
    :param measure: keyword argument for different measures of moisture: 
                    dewpoint, relhum ; relhum is default
    :param phase: options are ice and water to calculate 
    the vapor pressure above water or ice surfaces
    
    out:
    :return T_v: Virtual temperature
    """
    
    if measure == "dewpoint":
        e = magnus(moist_meas, phase=phase)
        T_v = T / (1. - e/p * (1. - R_A/R_V))
    
    elif measure == "relhum":
        e = moist_meas/100. * magnus(T, phase=phase)
        T_v = T / (1. - e/p * (1. - R_A/R_V))
    
    elif measure == "spechum":
        
        T_v = T *(1. + (R_V/R_A - 1.) * moist_meas)
        # iterative approach
        #Tv = np.copy(T)
        #e = p * moist_meas * R_V/R_A * Tv/T
        #de = 1.
        #while (np.max(de) > 1e-10):
            #e_old = np.copy(e)
            #T_v = T / (1. - e/p * (1. - R_A/R_V))
            #e = p * moist_meas * R_V/R_A * Tv/T
            #de = e_old-e
            
    else:
        print("ERROR in function virtual_temperature!")
        print("temperature measure is not supported or wrong!")
        
    
    return T_v



########################################################
###               specific humidity                  ###
########################################################


def specific_humidity(T, p, moist_meas, measure="relhum", phase="water"):
    """
    converts given moisture measure into specific humidity 
    
    args:
    :param T: air Temperature in [K] (float or array)
    :param p: air pressure in [Pa] 
    :param moist_meas: one moisture measure: currently supported:
                       dewpoint temperature in [K]and relative humidity in [pct]
                       --> give associated keyword for measure
    :param measure: keyword argument for different measures of moisture: 
                    dewpoint, relhum ; relhum is default
    :param phase: options are ice and water to calculate 
    the specific humidity with respect to liquid water or ice
    
    out:
    :return q: specific humidity in units of kg/kg
    """
    
    Tv = virtual_temperature(T, p, moist_meas, measure=measure)
    
    if measure == "dewpoint":
        e = magnus(moist_meas, phase=phase)
    
    elif measure == "relhum":
        e = moist_meas/100. * magnus(T, phase=phase) 
        
    else:
        print("ERROR in function specific_humidity!")
        print("temperature measure is not supported or wrong!")
        print("program exits")
        quit()
    
    return e/p * R_A/R_V * Tv/T

########################################################
###    relative humidity from specific humidity      ###
########################################################

def specifichum2RH(q, T, p, phase="water"):
    """
    converts specific humidity (kg water vapor per kg moist air)
    to relative humidity in percent
     
    args:
    :param q: specific humidity in kg/kg
    :param T: (mean) temperature in [K] of the layer
    :param p: total air pressure in [Pa]
    :param phase: phase of water: water/ice
    out:
    :return : relative humidity in percent [%]
    
    """
    
    Tv = virtual_temperature(T, p, q, measure="spechum")
    
    e = q * p * R_V/R_A * Tv/T

    es = magnus(T, phase=phase)
    
    return e/es * 100.

########################################################
###       number density from specific humidity      ###
########################################################

def specifichum2numdens(q, T, p, phase="water"):
    """
    converts specific humidity (kg water vapor per kg moist air)
    to number density in #/cm3
     
    args:
    :param q: specific humidity in kg/kg
    :param T: (mean) temperature in [K] of the layer
    :param p: total air pressure in [Pa]
    :param phase: phase of water: water/ice
    out:
    :return : number density [#/cm3]
    
    """
    
    Tv = virtual_temperature(T, p, q, measure="spechum")
    
    e = q * p * R_V/R_A * Tv/T

    return e / k_boltzmann / T * 1e-6

########################################################
###       number density from specific humidity      ###
########################################################

def relhum2numdens(rh, T,phase="water"):
    """
    converts relative humidity (in percent)
    to number density in #/cm3
     
    args:
    :param rh: relative humidity in [%]
    :param T: (mean) temperature in [K] of the layer
    :param phase: phase of water: water/ice
    out:
    :return : number density [#/cm3]
    
    """
    
    es = magnus(T)
    e = rh/100. * es

    return e / k_boltzmann / T * 1e-6
    
    
    
########################################################
###          hydrostatic p to z conversion           ###
########################################################

def hydrostat_p2dz(T, p, p0):
    """
    calculates vertical extent between two pressure levels
    
    args:
    :param T: (mean) temperature in [K] of the layer
    :param p: pressure in [Pa] at upper boundary
    :param p0: pressure in [Pa] at lower boundary
    
    out:
    :return dz: vertical distance in [m] beteween p and p0
    """
    
    dz = - R_A * T / g_earth * np.log(p/p0)
    
    return dz
    
    

########################################################
###       ideal gas pressure to number density       ###
########################################################

def ideal_gas_p2numdens(p, T):
    """
    calculates numberdensity for given pressure and 
    given temperature using the ideal gas law
    
    args:
    :poaram p: array or float with pressure in units of [Pa]
    :param T: array or float with temperature in units of [K]
    
    out:
    :return: numberdensity in [molecules per cm3] (unit of libradtran)
    """
    
    return p / k_boltzmann / T * 1e-6



########################################################
###       ideal gas pressure to number density       ###
########################################################

    
    
