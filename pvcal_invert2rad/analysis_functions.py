#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 09:32:53 2022

@author: james
"""

import numpy as np

def v_index(clearness_index_series,delta=1):
    """
    

    Parameters
    ----------
    clearness_index_series : pandas series
        series with clearness index
    delta : int
        integer for determining finite difference

    Returns
    -------
    None.

    """
        
    v_index = np.sqrt((clearness_index_series.diff(delta)**2).sum()/len(clearness_index_series))
    
    return v_index

def overshoot_index(clearness_index_series,threshold):
    """
    

    Parameters
    ----------
    clearness_index_series : pandas series
        series with clearness index
    threshold : float
        threshold for determining overshoots

    Returns
    -------
    None.

    """
    
    result = len(clearness_index_series.loc[clearness_index_series > threshold])/\
        len(clearness_index_series)
        
    return result
    