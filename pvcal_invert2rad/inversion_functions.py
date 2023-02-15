#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 16:52:48 2018

@author: james

These are the different functions needed to perform non-linear inversion, as taken
from Rodgers

"""

import numpy as np

#######################################################################
###                         INVERSION FUNCTIONS                               
#######################################################################
def x_post_lin(x_a,y,K,S_a,S_eps):
    """
    Solution to Gaussian minimisation in the linear case, see Eq. 2.30 in 
    Rodgers
    
    args:
        :param x_a: float, vector of a-priori values
        :param y: float, vector of measurement values
        :param K: float, matrix of Jacobians
        :param S_a: float, matrix of a-priori covariance
        :param S_eps: float, matrix of measurement covariance
        
    out:
        :return x_post: posterior solution vector
    """
    
    #Linear solution of Gaussian case (Maximum-a-Priori method)
    x_post = x_a + np.dot(np.dot(S_a,np.transpose(K)),np.linalg.solve(
            np.dot(np.dot(K,S_a),np.transpose(K)) + S_eps,(y - np.dot(K,x_a))))
    
    return x_post

def S_post(S_a,K,S_eps):
    """
    Covariance matrix of posterior distribution in Gaussian case, see Eq. 4.7
    and 4.8 of Rodgers
    
    
    Shat = (Khat^T.S_eps^-1.Khat + S_a^-1)^-1
         = S_a - S_a.Khat^T.(S_eps + Khat.S_a.K^T)^-1.Khat.S_a
    
    args:
        :param S_a: float, n x n matrix of a-priori covariance
        :param K: float, m x n matrix of Jacobians
        :param S_eps: float, m x m matrix of measurement covariance
        
    out:
        :return S_post: error covatiance matrix of posterior distribution
    """
    
    #Check whether n < m
    if len(S_a) < len(S_eps):
        #n-form
        S_post = np.linalg.inv(np.dot(np.dot(np.transpose(K),np.linalg.inv(S_eps)),K) + 
            np.linalg.inv(S_a))
    else:
        #m-form
        S_post = S_a - np.dot(np.dot(S_a,np.transpose(K)),np.linalg.solve(
            np.dot(np.dot(K,S_a),np.transpose(K)) + S_eps,np.dot(K,S_a)))
    
    return S_post

def G_matrix (K,S_a,S_eps,gamma):
    """
    Gain matrix, see Eq. 4.40 of Rodgers
    
    G = (S_a^-1 + K^T.S_eps^-1.K)^-1.K^T.S_eps^-1
      = S_a.K^T.(K.S_a.K^T + S_eps)^-1
    
    args:
        :param K: float, matrix of Jacobians
        :param S_a: float, matrix of a-priori covariance
        :param S_eps: float, matrix of measurement covariance
        :param gamma: LM parameter
        
    out:
        :return gain: gain matrix
    """
    
    #Check whether n < m
    if len(S_a) < len(S_eps):
        #n-form
        gain = np.linalg.solve((1. + gamma)*np.linalg.inv(S_a)
               + np.dot(np.dot(np.transpose(K),np.linalg.inv(S_eps)),K),\
               np.dot(np.transpose(K),np.linalg.inv(S_eps)))
    else:    
        #m-form
        gain = np.dot(np.dot(S_a,np.transpose(K)),np.linalg.solve(
            np.dot(np.dot(K,S_a),np.transpose(K)) + S_eps,np.eye(len(S_eps))))   
    
    return gain

def A_kernel(S_a,K,S_eps):
    """
    Averaging kernel matrix in Gaussian case
    
    args:
        :param S_a: float, matrix of a-priori covariance
        :param K: float, matrix of Jacobians
        :param S_eps: float, matrix of measurement covariance
        
    out:
        :return A_matrix: float, averaging kernel matrix
    """
    
    A_matrix = np.dot(np.dot(S_a,np.transpose(K)),np.linalg.solve(
            np.dot(np.dot(K,S_a),np.transpose(K)) + S_eps,K))
        
    return A_matrix

def H_info(A_k):
    """
    Shannon information content for a linear Gaussian distribution
    
    args:
        :param A_k: float, averaging kernel matrix
        
    out:
        :return H: Shannon information content
    """
    #Get dimension of matrix
    n = A_k.shape[0]
    
    #Calculate information content
    H = -0.5*np.log(np.linalg.det(np.identity(n)-A_k))
    
    return H

def chi_square_fun(x,x_a,y,F,S_a,S_eps):
    """
    Chi-squared function in the Gaussian case
    
    args:
        :param x: float, vector of state space values
        :param x_a: float, vector of a-priori values
        :param y: float, vector of measurement values
        :param F: float, vector of model values (= Kx in linear case)
        :param S_a: float, matrix of a-priori covariance
        :param S_eps: float, matrix of measurement covariance
        
    out:
        :return chi_squared: float, chi squared function (scalar)
    """
    
    #Calculate chi-squared function
    chi_squared = np.dot(np.transpose(y - F),np.linalg.solve(S_eps,y - F))\
                  + np.dot(np.transpose(x - x_a),np.linalg.solve(S_a,x - x_a)) 
        
    return chi_squared

def grad_chi_square (x_i,x_a,y,F,K,S_a,S_eps):
    """
    Gradient of Chi-squared function in the Gaussian case
    
    args:
        :param x_i: n-dim vector of state space values for step i
        :param x: float, vector of state space values
        :param x_a: float, vector of a-priori values
        :param y: float, vector of measurement values
        :param F: float, vector of model values (= Kx in linear case)
        :param S_a: float, matrix of a-priori covariance
        :param S_eps: float, matrix of measurement covariance
        
    out:
        :return grad_chi_squared: float, chi squared function (scalar)
    """
    
    #Calculate chi-squared function
    grad_chi_sq = -np.dot(np.dot(np.transpose(K),np.linalg.inv(S_eps)),y - F) #\
                  #+ np.dot(np.linalg.inv(S_a),x_i - x_a)
        
    return grad_chi_sq

def x_non_lin (x_i,x_a,y,F,K,S_a,S_eps,gamma):
    """
    Iteration of state variable for Levenberg-Marquardt optimisation
    
    args:
        :param x_i: n-dim vector of state space values for step i
        :param x_a: float, n-dim vector of a-priori values
        :param y: float, m-dim vector of measurement values
        :param F: float, m-dim vector of model values (= Kx in linear case)
        :param K: float, m x n matrix of Jacobians
        :param S_a: float, n x n matrix of a-priori covariance
        :param S_eps: float, m x m matrix of measurement covariance
        :param gamma: LM parameter to switch between Gauss-Newton & steepest descent
        
    out:
        :return x_j: vector of state space values for next step j = i + 1
    """
    
    #Check whether n < m
    if len(S_a) < len(S_eps):
        #n-form
#        x_j = x_a + np.linalg.solve((1 + gamma)*np.linalg.inv(S_a) + np.dot(np.dot(np.transpose(K),
#            np.linalg.inv(S_eps)),K),np.dot(np.dot(np.transpose(K),np.linalg.inv(S_eps)),
#            y - F + np.dot(K,x_i - x_a)))
        x_j = x_a + np.dot(G_matrix(K,S_a,S_eps,gamma),y - F + np.dot(K,x_i - x_a))
        
#        x_j = x_i + np.linalg.solve((1 + gamma)*np.linalg.inv(S_a) + np.dot(np.dot(np.transpose(K),
#            np.linalg.inv(S_eps)),K),np.dot(np.dot(np.transpose(K),np.linalg.inv(S_eps)),
#            y - F) - np.dot(np.linalg.inv(S_a),x_i - x_a))
    else:    
        #m-form
        x_j = x_a + np.dot(np.dot(S_a,np.transpose(K)),np.linalg.solve(
            np.dot(np.dot(K,S_a),np.transpose(K)) + S_eps,(y - F + np.dot(K,x_i - x_a))))   
    
    return x_j

def d_i_sq (x_i, x_i1, x_a, y, F, K, S_a, S_eps):
    """
    
    The parameter d_i^2 is used as a test of convergence when n < m
    

    Parameters
    ----------
    x_i : n-dim vector of state space values for step i
    x_i1 : n-dim vector of state space values for step i+1
    x_a : float, n-dim vector of a-priori values
    y : float, m-dim vector of measurement values
    F : float, m-dim vector of model values (= Kx in linear case)
    K : float, m x n matrix of Jacobians
    S_a : float, n x n matrix of a-priori covariance
    S_eps : float, m x m matrix of measurement covariance

    Returns
    -------
    d : float, parameter d_i^2

    """
    
    
    d = np.dot(np.transpose(x_i1 - x_i),np.dot(np.transpose(K),np.linalg.solve(S_eps,y - F))
        - np.linalg.solve(S_a,x_i - x_a))
    
    return d

def S_del_y (K_hat, S_a, S_eps):
    """
    Covariance matrix of the difference between the model fit and the measurement, see
    Eq. 5.27 of Rodgers
    S_dely = S_eps.(Khat.S_a.Khat^T + S_eps)^(-1).S_eps
    
    args:
        :param K_hat: float, m x n matrix of Jacobians at the solution
        :param S_a: float, n x n matrix of a-priori covariance
        :param S_eps: float, m x m matrix of measurement covariance
        
    out:
        :return S_dely: m x m covariance matrix
    
    """
    S_dely = np.dot(S_eps,np.linalg.solve(np.dot(K_hat,np.dot(S_a,np.transpose(K_hat))) 
            + S_eps,S_eps))
    
    return S_dely

def chi_sq_retrieval (y, F_hat, K_hat, S_a, S_eps):
    """
    Chi-squared function of the retrieval, see Eq. 5.32 of Rodgers
    chi^2 = (y - F(x_hat))^T.S_dely^(-1).(y - F(x_hat))
    
    args:
        :param y: float, m-dim vector of measurement values
        :param F_hat: float, m-dim vector of model values at solution xhat
        :param K_hat: float, m x n matrix of Jacobians at solution
        :param S_a: float, n x n matrix of a-priori covariance
        :param S_eps: float, m x m matrix of measurement covariance
        
    out:
        :return S_dely: m x m covariance matrix
    
    """
    chi_squared = np.dot(np.transpose(y - F_hat),
                         np.linalg.solve(S_del_y(K_hat,S_a,S_eps),y - F_hat))
    
    return chi_squared

def d_s(A):
    """
    Degrees of freedom for signal, Eq. 2.80 or Rodgers
    
    args:
        :param A: float, averaging kernel matrix
        
    out:
        :return: d_s, scalar giving degrees of freedom
    """
    
    d_s = np.trace(A)
    
    return d_s
    
    
