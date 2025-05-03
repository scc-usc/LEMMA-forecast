import numpy as np
from numpy import *
import scipy.optimize
import scipy.stats
from functools import partial
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from scipy import signal
from config_param import season_start_day
import pandas as pd

import warnings

def var_ind_beta_un(data_4, passengerFlow, alpha_l, k_l, un_fact, popu, jp_l, ret_conf, S, compute_region=None, window_size=None, extra_imm=None):
    
    maxt = np.size(data_4, 1)
    
    if compute_region is None:
        compute_region = np.ones((np.size(popu), 0))
    
    if window_size is None:
        window_size = maxt*np.ones((np.size(data_4, 0), 1)); # By default, use all data to fit parameters
    
    if extra_imm is None:
        extra_imm = np.zeros((np.size(data_4, 0), maxt)) + S; # By default, use all data to fit parameters
    #addition of S
    
    F = passengerFlow
    beta_all_cell = [None] * np.size(popu)
    fittedC = [None] * np.size(popu)
    ci = [None] * np.size(popu)
    nn = np.size(popu)
    
    if np.isscalar(un_fact):
        un_fact = np.ones(len(popu)) * un_fact

    if np.isscalar(jp_l):
        jp_l = np.ones(len(popu)) * jp_l

    if np.isscalar(k_l):
        k_l = np.ones(len(popu)) * k_l

    if np.isscalar(alpha_l):
        alpha_l = np.ones(len(popu)) * alpha_l

    if np.isscalar(window_size):
        window_size = np.ones(nn) * window_size
    
    deldata = np.diff(data_4, 1)

    
    for j in range(1, np.size(popu) + 1):
        jp = int(jp_l[j - 1])
        k = int(k_l[j - 1])
        alpha = alpha_l[j - 1]
        jk = int(jp*k)
        
        beta_all_cell[j - 1] = np.zeros((k+1, 1))
        ci[j - 1] = np.zeros((k+1, 2))
        fittedC[j - 1] = np.zeros((1, 2))
        # update with start date
        # if data_4.size == 0 or data_4.iloc[j - 1].size == 0 or data_4.iloc[j - 1, -1] < 1 or compute_region[j - 1] < 1:

        # if compute_region[j - 1] < 1: # if explictly told not to compute
        #     continue

        
        skip_days = int(maxt - window_size[j - 1])
        
        if season_start_day + 15 < maxt:
            skip_days = season_start_day
            
        if skip_days < 0:
            skip_days = 0
        
        ex = (np.arange(maxt - skip_days - jk - 1, 0, -1)).transpose()
        alphavec = np.power(alpha, ex)
        alphamat = np.tile(alphavec, np.concatenate(([k+1], [1]))).transpose()
      
        tdim = maxt - jk - skip_days - 1
        if tdim<0:
            tdim = 0
        y = np.zeros((tdim, 1))
        X = np.zeros((tdim, k+1))
        Ikt = np.zeros((1,k))
        
        
        for t in range(skip_days+jk+1, maxt):
            Ikt1 = deldata[j - 1, t - jk - 1:t-1]
            S = (1-(extra_imm[j - 1, t - 1] + (un_fact[j - 1] * data_4[j - 1,t - 1])) / popu[j-1])
            for kk in range(1, k+1):
                m0 = Ikt1[(kk-1)*jp : kk*jp]
                Ikt[0, kk - 1] = S*sum(m0, 0)
            
            if F is None or (np.isscalar(F) or len(np.shape(F)) == 0) or np.size(F, 0) != np.size(popu):
                incoming_travel = np.zeros((1, 1))
            else:
                incoming_travel = np.transpose(F[:, j - 1] / popu) @ sum(Ikt1, 1)

            X[(t-1) - jk-skip_days, :] = np.concatenate((Ikt, incoming_travel), 1)
            y[(t-1) - jk-skip_days] = np.transpose(deldata[j - 1, t - 1])
        
        X = alphamat * X
        y = alphavec.transpose() * y.flatten()


        if np.size(X) != 0 and np.size(y) != 0:
            a = np.concatenate((np.ones(k), [np.inf]), 0)
            if ret_conf is None:   # If confidence intervals are not required, we will run this as this seems to be faster
                beta_vec = scipy.optimize.lsq_linear(X, y, bounds=(np.zeros(k+1), a)).x
                beta_CI = np.zeros((k+1, 2))
                beta_CI[:, 0] = beta_vec
                beta_CI[:, 1] = beta_vec
                
            else:
                def sigmoid(X, *w):
                    w1 = np.array(w).flatten()
                    return X @ (1 / (1 + np.exp(-w1)))
             
                
                k1 = X.shape[1]
                w0 = np.zeros(k1)
                popt, pcov = scipy.optimize.curve_fit(sigmoid, X, y, p0=w0, method="dogbox")

                n = len(y)
                mse = np.sum((y - sigmoid(X, popt)) ** 2) / (n - k1)
                se = np.sqrt(np.diag(pcov) * mse)
                tval = scipy.stats.t.ppf((1 + ret_conf) / 2, n - k1)

                CI_lower = popt - tval * se
                CI_upper = popt + tval * se

                beta_vec = 1 / (1 + np.exp(-popt))
                beta_CI = 1 / (1 + np.exp(-np.vstack((CI_lower, CI_upper))))
                
               
            ci[j - 1] = beta_CI    
            beta_all_cell[j - 1] = beta_vec
    return (beta_all_cell, fittedC, ci)
