import pandas as pd
import numpy as np
import config_param
import config_model
from predictors import generate_predictors
import shared_utils
from shared_utils.utils import bin_array
import matplotlib.pyplot as plt


def generate_all_preds(progress_callback=None):
    # hosp_cumu_s_org= np.loadtxt('data/hosp_cumu_s.csv', delimiter=',')

    # hosp_dat = pd.read_csv('data/hosp_dat.csv')
    # popu = np.loadtxt('data/us_states_population_data.txt') #population of each state
    hosp_cumu_s_org = config_param.hosp_cumu_s_org
    hosp_dat = config_param.hosp_dat
    popu = config_param.popu
    retro_lookback = config_param.retro_lookback
    all_preds = generate_predictors(hosp_cumu_s_org, hosp_dat, popu, config_param, retro_lookback, progress_callback)    
    return all_preds, hosp_dat


def generate_preds(all_preds, hosp_dat):
    retro_lookback = config_param.retro_lookback
    X_train, Y_train, W_train, ID = shared_utils.utils.regression_prep(all_preds, hosp_dat, retro_lookback)
    #print(f"ID shape: {ID.shape}")
    #print(ID)
    for col in ID:
        print(f"Column: {col}, Unique values: {ID[col].unique()}")
    models = config_model.get_models(config_param.weeks_ahead)
    valid_train = ~ID['lookback'].isin(config_param.test_lookback)
    for i, model in enumerate(models):
        model.fit(X_train[valid_train,:], Y_train[valid_train, i], sample_weight=W_train[valid_train])
    
    test_lookback_array =  config_param.test_lookback # Remeber to add this in config_param

    #test_location = 2
    quantiles = config_param.quantiles
    all_test_preds = {}
    
    for test_lookback in test_lookback_array:
        for test_location in range(hosp_dat.shape[0]):
            #if test_location in [52,54,56]:
            #    continue
            try:
                rows = ID[(ID['lookback'] == test_lookback) & (ID['location'] == test_location)].index
                preds = np.zeros((len(models), len(quantiles)))
                
                for i, model in enumerate(models):
                    preds[i, :] = shared_utils.utils.predict_quantiles(model, X_train[rows], quantiles)
                
                all_test_preds[(test_lookback, test_location)] = preds*config_param.popu[test_location] # Check this
            except:
                print(f"Error in test_location {test_location} for test_lookback {test_lookback}")
                continue
    
    return all_test_preds