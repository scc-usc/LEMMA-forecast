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

"""
Affects each time series (row) separately

beta_all_cell (python list of 1-d numpy arrays): List with each element representing infection parameters.
k_l: List with each element representing number of beta all cell params for the corresponding region
jp_l: List with each element representing a hyperparameter for binning
data_4: (2-D matrix of real numbers n x T) (should just be named data) cumulative reported cases matrix (each row is a region, each column is timestamp)
passengerFlowDarpa: set it to 0. Originally intended to be n x n matrix to capture inter-region mobility
popu (nx1): Population of each region
horizon: Number of time units to simulate starting from T_start
un_fact: under-reporting factor (>= 1)
base_infec: cumulative cases per region at the time of simulation start 
vac: vaccinations given (n x (T + horizon)) 
"""

def var_simulate_pred_un(data_4, passengerFlowDarpa, beta_all_cell, popu, k_l, horizon, jp_l, un_fact, base_infec=None, vac=None, rate_change=None):
    num_countries = size(data_4, 0)
    infec = zeros((num_countries, horizon))
    F = passengerFlowDarpa

    
    if rate_change is None:
        rate_change = ones((num_countries, horizon))

    if base_infec is None:
        base_infec = data_4[:, -1:]

    if vac is None:
        vac = zeros((num_countries, horizon))


    if size(rate_change) == 0:
        rate_change = ones((num_countries, horizon))

    if size(un_fact) == 1:
        un_fact = un_fact*ones((size(popu), 1))

    if size(jp_l) == 1:
        jp_l = ones((size(popu), 1))*jp_l

    if size(k_l) == 1:
        k_l = ones((size(popu), 1))*k_l

    for j in range(0, len(beta_all_cell)):
        this_beta = beta_all_cell[j]
        if size(this_beta)==k_l[j]:
            beta_all_cell[j] = np.append(this_beta, 0)

            #beta_all_cell[j] = concatenate((this_beta, [[0]]), axis=0)
            pass

    data_4_s = data_4
    temp = data_4_s
    deltemp = diff(temp, axis=1)
    
    if isinstance(F, (int, float)) or F is None:    # Optimized code when mobility not considered
        for j in range(0, size(popu)):
            lastinfec = base_infec[j]

            if sum(beta_all_cell[j]) == 0:
                infec[j, :] = lastinfec
               # print(infec.shape, "a")
                continue

            jp = int(jp_l[j])
            k = int(k_l[j])
            jk = int(jp*k)
            Ikt1 = deltemp[j, -(jk):]
            Ikt = zeros((k, 1))
            
            for t in range(0, horizon):
                true_infec = un_fact[j]*lastinfec
                S = 1 - true_infec/popu[j] - ((1-true_infec/popu[j])*vac[j, t]) / popu[j]
                for kk in range(1, k+1):
                    Ikt[kk - 1] = sum(Ikt1[(kk - 1)*jp : kk*jp], 0)
                Xt = concatenate(((S*Ikt), [[0]]), axis=0)
                yt = rate_change[j, t]*sum(transpose(beta_all_cell[j]) @ Xt, axis=0)
                yt = max(yt, 0)
                
                lastinfec = lastinfec + yt
                infec[j, t] = lastinfec
               # print(infec.shape, "b")
                Ikt1 = np.append(Ikt1, yt)  # Append yt to Ikt1
                Ikt1 = np.delete(Ikt1, 0)

    return infec