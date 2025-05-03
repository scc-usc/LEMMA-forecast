
import numpy as np
import config_param
import warnings
import sys
sys.path.append('/home/lemma/Downloads/NewForecast-Generation')
from input_models.var_ind_beta_un import *
from input_models.var_simulate_pred_un import *
# from cy_var_ind import var_ind_beta_un
# from cy_var_simulate import var_simulate_pred_un

def process_scenario(args, hosp_cumu_s, hosp_cumu, popu, config_param, base_hosp):
    
    warnings.filterwarnings("ignore")
    temp_res = np.zeros((config_param.num_dh_rates_sample, hosp_cumu_s.shape[0], config_param.horizon))

    simnum, scenario_details = args
    halpha, rlag, un, s = scenario_details
    rr = config_param.rlags[rlag.astype(int) - 1]
    
    #un_array =popu[:, None] * 0 + config_param.un_list
    #un = un_array[:, un_array.astype(int) - 1]
    #un = config_param.un_list[1]
    if rr == 0:
        sliced_array = hosp_cumu_s
    else:
        sliced_array = hosp_cumu_s[:, :rr]
    
    # hosp_rate = np.zeros((popu.shape[0], config_param.hk+1))
    # ci_h = np.zeros((popu.shape[0], config_param.hk+1, 2))
    hosp_rate, fC, ci_h = var_ind_beta_un(sliced_array, 0, halpha, config_param.hk, un, popu, config_param.hjp, 0.95, s, None, 80)
    
    
    
    for rr in range(config_param.num_dh_rates_sample):
        this_rate = hosp_rate.copy()
        
        if rr != (config_param.num_dh_rates_sample + 1) // 2 - 1:
            for cid in range(hosp_cumu_s.shape[0]):
                this_rate[cid] = ci_h[cid][:, 0] + (ci_h[cid][:, 1] - ci_h[cid][:, 0]) * (rr) / (config_param.num_dh_rates_sample - 1)
        else:
            for cid in range(hosp_cumu_s.shape[0]):
                this_rate[cid] = (ci_h[cid][:, 1] + ci_h[cid][:, 0]) /2
        
        pred_hosps = var_simulate_pred_un(hosp_cumu_s, 0, this_rate, popu, config_param.hk, config_param.horizon, config_param.hjp, un, base_hosp)
      
        #pred_hosps = np.zeros((hosp_cumu_s.shape[0], config_param.horizon))
        h_start = 0
        temp_res[rr, :, :] = pred_hosps[:, h_start:h_start+config_param.horizon] - base_hosp.reshape(-1, 1)

    return temp_res