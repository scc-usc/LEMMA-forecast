import pandas as pd
import numpy as np
import config_param
from approaches import get_approach
from process_scenario import *
from math import ceil
from preprocess.util_function import smooth_epidata
from itertools import product
#from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor, as_completed
from loky import ProcessPoolExecutor, as_completed

def generate_predictors(hosp_cumu_s_org, hosp_dat, popu, config_param, retro_lookback, progress_callback = None):

    maxt = hosp_dat.shape[1]
    days = maxt
    hosp_cumu_orig = np.nancumsum(np.nan_to_num(hosp_dat), axis=1)
    num_states = len(popu)

    all_preds = {}
    progress_count = 0
    for x in retro_lookback:
    
        wks_back = x 
        hosp_cumu_s = hosp_cumu_s_org
        hosp_cumu = hosp_cumu_orig
        T_full = days - wks_back * config_param.bin_size  # Computing T_full by subtracting 7*wks_back from days
        thisday = T_full  
        ns = hosp_cumu_s.shape[0]  # Getting the number of rows in hosp_cumu_s
        
        if wks_back == 0:
            # if wks_back is zero, use the entire array
            hosp_cumu_s = hosp_cumu_s[:, :]
            hosp_cumu = hosp_cumu[:, :]
        else:
            # if wks_back is non-zero, then proceed with the original slicing
            hosp_cumu_s = hosp_cumu_s[:, :-(wks_back*config_param.bin_size)]
            hosp_cumu = hosp_cumu[:, :-(wks_back*config_param.bin_size)]
        
        
        # Build scenarios using the selected approach
        approach_name = getattr(config_param, "selected_approach", "SIKJalpha Basic")
        approach = get_approach(approach_name)
        scen_list = approach.build_scenarios(config_param)
        
        # One aggregate per scenario; approaches handle any internal sampling and we take mean
        net_hosp_A = np.empty((len(scen_list), ns, config_param.horizon))
        net_hosp_A[:] = np.nan
        
        net_h_cell = [None] * len(scen_list)  
        
        base_hosp = hosp_cumu[:, T_full-1]


        # Let the approach declare its required configuration
        config_params = approach.make_config(config_param)
        
        # Create a list of tasks, each task is a tuple (simnum, scenario)
        tasks = [(simnum, scen_list[simnum]) for simnum in range(scen_list.shape[0])]
        #'''
        # Parallel version
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(process_scenario, task, hosp_cumu_s, hosp_cumu, popu, config_params, base_hosp): task[0] for task in tasks}
            
            for future in as_completed(futures):
                simnum = futures[future]
                net_h_cell[simnum] = future.result()

        '''
        # Serial version
        
        for task in tasks:
            simnum, scenario = task
            #try:
            # Process the scenario and store the result in net_h_cell at the index corresponding to simnum
            net_h_cell[simnum] = process_scenario(task, hosp_cumu_s, hosp_cumu, popu, config_params, base_hosp)
            # except Exception as exc:
            #     print(f"Scenario {simnum} generated an exception: {exc}")
            #     net_h_cell[simnum] = -1000    
        '''
        
        for simnum in range(scen_list.shape[0]):
            net_hosp_A[simnum , :, :] = np.nanmean(net_h_cell[simnum], axis=0)
        # Use all scenarios produced by the selected approach; keep 3D shape
        p = net_hosp_A  # shape: (n_predictors, n_locations, horizon)
        predictors = np.diff(p, axis=2)
        lo = predictors[:, :, 0:config_param.weeks_ahead*config_param.bin_size]
        all_preds[x] = lo
        
        
        progress_count += 1
        try:
            progress_callback(progress_count / len(retro_lookback))
        except:
            pass

        print(f"Done for lookback {x}. Progress: {config_param.predictor_progress:.2%}")
    return all_preds