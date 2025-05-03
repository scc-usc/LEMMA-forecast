import numpy as np
from datetime import datetime

import pandas as pd
import config_model
import shared_utils.utils
############################
# offset = 11
zero_date = datetime(2021, 9, 1)
days_back = 0
bin_size = 7
#### model determining
weeks_ahead = 4
smooth_factor = 7
####
num_dh_rates_sample = 1
season_start = datetime(2023, 9, 30)
season_end = zero_date
season_start_day = (season_start - season_end).days

##############################
rlags = np.array([0])
rlag_list = np.arange(1, len(rlags) + 1)
un_list = np.array([100.0])
halpha_list = np.arange(0.98, 0.92, -0.02)
S = np.array([0.0])
hyperparams_lists = [halpha_list, rlag_list, un_list, S]
hk = 2
hjp = 7
########################

npredictors = (len(S) * len(halpha_list) * len(un_list) * len(rlag_list))*(weeks_ahead)
horizon = (weeks_ahead+1)*bin_size 


decay_factor = 0.99
wks_back = 1
# default_n_estimators = 100     #Moved to Config_model
# quantiles = [0.010, 0.025, 0.050, 0.100, 0.150, 0.200, 0.250, 0.300, 0.350, 0.400, 0.450, 0.500, 0.550, 0.600, 0.650, 0.700, 0.750, 0.800, 0.850, 0.900, 0.950, 0.975, 0.990]
quantiles = np.array([0.025, 0.5, 0.975])


#Retro Lookback
# predictor_generation_lookbacks = 20:30
# predictor_predict = 27:30



# range = np.arange(start, end)
# split = int(0.8 * len(range))
# retro_train, retro_test  = range[:split], range[split:]
# retro_lookback = np.concatenate((retro_train, retro_test))

hosp_cumu_s_org= np.loadtxt('data/hosp_cumu_s.csv', delimiter=',')

hosp_dat = pd.read_csv('data/hosp_dat.csv', delimiter=',').to_numpy()


popu = np.loadtxt('data/us_states_population_data.txt')

alpha = 1
beta = 1


start_train = 25
end_train = 32
retro_lookback = np.array([101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113])

start_test = 33
end_test = 40
test_lookback = np.array([105, 104, 103, 102, 101])

predictor_progress = 0
# start_train_list = [29]
# start_test_list = [29]
# end_train_list = [33]
# end_test_list = [32]
# retro_lookback = np.concatenate([np.arange(s, e + 1) for s, e in zip(start_train_list, end_train_list)])
# test_lookback = np.concatenate([np.arange(s, e + 1) for s, e in zip(start_test_list, end_test_list)])



# retro_lookback = np.array([29, 30, 31, 32, 33, 41, 42, 43, 44, 45])
# test_lookback = np.array([30, 31, 32, 41, 42, 43])
