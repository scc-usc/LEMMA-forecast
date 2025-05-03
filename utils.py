import pandas as pd
import numpy as np
import config_param

def prepare_data_for_model(start_col, end_col, target_col_index, old_data, data_train, weights):
    X_old_data = old_data.iloc[:, [0, 1] + list(range(start_col, end_col)) + [-1]]  # Using -1 for the last column
    X_data_train = data_train.iloc[:, [0, 1] + list(range(start_col, end_col)) + [-1]]
    X = pd.concat([X_old_data, X_data_train])

    # Prepare the target vector
    y_old_data = old_data.iloc[:, target_col_index]
    y_data_train = data_train.iloc[:, target_col_index]
    y = pd.concat([y_old_data, y_data_train])

    # Combine, clean, and separate X and y
    combined = pd.concat([X.iloc[:, 2:], y.rename('target')], axis=1)
    non_nan_rows = ~combined.isnull().any(axis=1)
    combined_clean = combined[non_nan_rows]
    X_clean = combined_clean.iloc[:, :-1]
    y_clean = combined_clean.iloc[:, -1]
    weights_clean = weights[non_nan_rows].ravel()

    return X_clean, y_clean, weights_clean

def train_model(model, X, y, weights):
    model.fit(X, y, sample_weight=weights)
    return model

def predict_quantiles(model, X, quantiles):
    predictions = []
    for q in quantiles:
        model.q = q
        predictions.append(model.predict(X))
    return np.array(predictions).T  # Transpose to have shape (n_samples, n_quantiles)


def getmean(model, X):
    model.q = 0.5
    return model.predict(X)


def aggregate_quartiles(mpgQuartiles):
    indices = np.arange(0, int(np.floor(mpgQuartiles.shape[0] / 7) * 7), 7)
    pred = np.sum(np.array([mpgQuartiles[indices + i, :] for i in range(7)]), axis=0)
    return pred

def aggregate_means(mpgMean):
    indices = np.arange(0, int(np.floor(mpgMean.shape[0] / 7) * 7), 7)
    pred_ms = np.sum(np.array([mpgMean[indices + i] for i in range(7)]), axis=0)
    return pred_ms


def predict_modely(model_num, data_test, trained_models_dict, quantiles, start_col, end_col):
    X_test = data_test.iloc[:, [0, 1] + list(range(start_col, end_col + 1)) + [-1]]
    X_test.columns = X_test.columns.astype(str)
    X_test = X_test.iloc[:, 2:]  # Drop the first two columns
    model_key = f'model_{model_num}'
    print(type(trained_models_dict))
    print(trained_models_dict[model_key])
    model = trained_models_dict[model_key]
    if X_test.shape[1] != model.n_features_in_:
        raise ValueError(f"Incorrect number of features. Expected: {model.n_features_in_}, got: {X_test.shape[1]}")
    
    predictions = []
    for q in quantiles:
        trained_models_dict[model_key].q = q
        predictions.append(trained_models_dict[model_key].predict(X_test))
    #mpgQuartiles = predict_quantiles(trained_models_dict[model_key], X_test, quantiles)
    mpgQuartiles = predictions
    #mpgMean = getmean(trained_models_dict[model_key], X_test)
    trained_models_dict[model_key].q = 0.5
    mpgMean = trained_models_dict[model_key].predict(X_test)
    # Adjust the index for pmf_class if required by your logic
    pmf_index = model_num - 1 if model_num < 5 else 2
    pmf = pmf_class(trained_models_dict[model_key], X_test, pmf_index)
    return model_key, mpgQuartiles, mpgMean, pmf

def pmf_class(model, X, wk):
    """
    Calculate the Probability Mass Function (PMF) based on the rate of change in predictions
    from a RandomForest model (as a stand-in for TreeBaggerModel).
    
    Parameters:
    - model: Trained RandomForestRegressor model.
    - X: Input features as a DataFrame.
    - wk: Week parameter to adjust classification criteria.
    
    Returns:
    - pmf: Probability Mass Function for each classification category.
    """
    n_trees = len(model.estimators_)
    predictions = np.array([tree.predict(X) for tree in model.estimators_]).T
    pop = X.iloc[:, -1].values[::7]
    weekly_predictions = predictions.reshape((predictions.shape[0]//7, 7, predictions.shape[1])).sum(axis=1)
    XW = X.iloc[:, 1].values.reshape((X.shape[0]//7, 7)).sum(axis=1)
    RateChange = weekly_predictions - XW[:, None]
    RateChangeUN = np.abs(RateChange * (pop[:, None]/100000))
    RateChangeUN = np.append(RateChangeUN, RateChangeUN.sum(axis=0)[None, :], axis=0)
    RateChange = np.append(RateChange, (RateChange.sum(axis=0) * 100000 / pop.sum())[None, :], axis=0)

    # Classification criteria based on wk and RateChange
    def classify(rate, rate_un, wk):
        if wk == -1:
            thresholds = [(1, 2), (10, np.inf)]
        elif wk == 0:
            thresholds = [(1, 3), (10, np.inf)]
        elif wk == 1:
            thresholds = [(2, 4), (10, np.inf)]
        else:
            thresholds = [(2.5, 5), (10, np.inf)]

        if np.abs(rate) < thresholds[0][0] or rate_un < thresholds[1][0]:
            return 'Stable'
        elif 0 < rate < thresholds[0][1]:
            return 'Increase'
        elif rate >= thresholds[0][1]:
            return 'Large Increase'
        elif -thresholds[0][1] < rate < 0:
            return 'Decrease'
        elif rate <= -thresholds[0][1]:
            return 'Large Decrease'
        else:
            return 'Stable'

    classifications = np.array([[classify(rate, rate_un, wk) for rate, rate_un in zip(rc, rcun)] for rc, rcun in zip(RateChange, RateChangeUN)])

    # Unique classifications
    unique_classifications = ["Stable", "Increase", "Large Increase", "Decrease", "Large Decrease"]
    pmf = np.zeros((classifications.shape[0], len(unique_classifications)))

    for i, classification in enumerate(unique_classifications):
        pmf[:, i] = np.mean(classifications == classification, axis=1)

    return pmf




