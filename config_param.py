import numpy as np
from datetime import datetime

import pandas as pd

import config_model
import shared_utils.utils
import preprocess.util_function as pp
import user_config as user


# ============================
# Internal Compatibility Layer
# ============================
# This module maps user-facing settings in user_config.py to legacy variable
# names used throughout the codebase.


def _as_np(v):
    return v if isinstance(v, np.ndarray) else np.array(v)


def _to_datetime(v):
    if isinstance(v, datetime):
        return v
    return pd.Timestamp(v).to_pydatetime()


# Time semantics
base_time_step_unit = getattr(user, "base_time_step_unit", "day")
forecast_bin_unit = getattr(user, "forecast_bin_unit", "week")

zero_date = getattr(user, "reference_date", datetime(2021, 9, 1))
days_back = 0
bin_size = int(getattr(user, "timesteps_per_bin", 7))
weeks_ahead = int(getattr(user, "bins_ahead", getattr(user, "forecast_horizon_bins", 4)))
smooth_factor = int(getattr(user, "smoothing_window_timesteps", 7))

num_dh_rates_sample = 1
season_start = getattr(user, "season_start_date", datetime(2023, 9, 30))
season_end = zero_date
season_start_day = (season_start - season_end).days

# Model and approach settings
selected_approach = getattr(user, "predictor_approach", "ARIMA")
ensemble_method = getattr(user, "ensemble_method", "Random Forest")

quantiles = _as_np(getattr(user, "quantiles", np.array([0.0, 0.5, 1.0])))

# ARIMA aliases
ar_p_list = _as_np(getattr(user, "arima_autoregressive_orders", np.array([7.0, 14.0])))
d_list = _as_np(getattr(user, "arima_differencing_orders", np.array([0.0, 1.0, 2.0])))

# Flatline aliases
flat_k_list = _as_np(getattr(user, "flatline_lag_timesteps", np.array([0.0, 7.0, 14.0])))

# SIKJalpha-style aliases
rlags = _as_np(getattr(user, "rlags", np.array([0])))
rlag_list = _as_np(getattr(user, "retro_lag_bins", np.array([1.0])))
un_list = _as_np(getattr(user, "un_list", np.array([50.0])))
halpha_list = _as_np(getattr(user, "halpha_list", np.arange(0.98, 0.92, -0.02)))
S = _as_np(getattr(user, "S", np.array([0.0])))
hyperparams_lists = [halpha_list, rlag_list, un_list, S]
hk = int(getattr(user, "hk", 2))
hjp = int(getattr(user, "hjp", 7))

# Output settings
cli_output_path = getattr(user, "forecast_output_path", "outputs/predictions.csv")
cli_output_format = getattr(user, "forecast_output_format", "csv")

# Data sources
ts_dat = getattr(user, "target_data_path", "data/ts_dat.csv")
location_dat_path = getattr(user, "location_metadata_path", "data/location_dat.csv")

hosp_dat = pd.read_csv(ts_dat, delimiter=",", header=None).to_numpy()
location_dat = pd.read_csv(location_dat_path, delimiter=",")

alpha = 1
beta = 1

# Train/forecast range defaults (in bin units)
def _base_steps_between(start_dt, end_dt):
    delta = end_dt - start_dt
    unit = str(base_time_step_unit).lower().strip()
    if unit in {"day", "days", "d"}:
        return int(delta.days)
    if unit in {"week", "weeks", "w"}:
        return int(delta.days // 7)
    return int(delta.days)


def _bin_index_from_date(date_value):
    date_dt = _to_datetime(date_value)
    steps = _base_steps_between(zero_date, date_dt)
    return int(steps // bin_size)


n_binned_steps = max(1, hosp_dat.shape[1] // bin_size)
max_origin_bin = max(0, n_binned_steps - 1)

# Defaults requested for user-facing windows:
# - training start: 10 bins before last trainable bin (or 0)
# - training end: last trainable bin given bins_ahead
# - forecast start/end: last available origin bin
default_train_end_bin = max(0, max_origin_bin - weeks_ahead)
default_train_start_bin = max(0, default_train_end_bin - 10)
default_forecast_start_bin = max_origin_bin
default_forecast_end_bin = max_origin_bin


training_window_start_date = getattr(user, "training_window_start_date", None)
training_window_end_date = getattr(user, "training_window_end_date", None)
forecast_window_start_date = getattr(user, "forecast_window_start_date", None)
forecast_window_end_date = getattr(user, "forecast_window_end_date", None)

if training_window_start_date is not None:
    start_train = _bin_index_from_date(training_window_start_date)
else:
    start_train = int(getattr(user, "training_window_start_bin", default_train_start_bin))

if training_window_end_date is not None:
    end_train = _bin_index_from_date(training_window_end_date)
else:
    end_train = int(getattr(user, "training_window_end_bin", default_train_end_bin))

if forecast_window_start_date is not None:
    start_test = _bin_index_from_date(forecast_window_start_date)
else:
    start_test = int(getattr(user, "forecast_window_start_bin", default_forecast_start_bin))

if forecast_window_end_date is not None:
    end_test = _bin_index_from_date(forecast_window_end_date)
else:
    end_test = int(getattr(user, "forecast_window_end_bin", default_forecast_end_bin))

start_train = int(np.clip(start_train, 0, max_origin_bin))
end_train = int(np.clip(end_train, 0, max_origin_bin))
start_test = int(np.clip(start_test, 0, max_origin_bin))
end_test = int(np.clip(end_test, 0, max_origin_bin))


# Derived values
npredictors = (len(S) * len(halpha_list) * len(un_list) * len(rlag_list)) * weeks_ahead
horizon = (weeks_ahead + 1) * bin_size


def _build_lookbacks_from_ranges(start_train_v, end_train_v, start_test_v, end_test_v):
    max_t_binned = hosp_dat.shape[1] // bin_size
    train_days = np.arange(start_train_v, end_train_v + 2)
    test_days = np.arange(start_test_v, end_test_v + 1)

    train_lookback_v = [max_t_binned - i for i in train_days]
    test_lookback_v = [max_t_binned - i for i in test_days]
    retro_lookback_v = sorted(set(train_lookback_v + test_lookback_v))
    return np.array(retro_lookback_v), np.array(test_lookback_v)


retro_lookback, test_lookback = _build_lookbacks_from_ranges(
    start_train, end_train, start_test, end_test
)

predictor_progress = 0
wks_back = 1
decay_factor = 0.99

hosp_dat_cumu = pp.smooth_epidata(np.cumsum(hosp_dat, axis=1), 1)
hosp_dat = np.diff(hosp_dat_cumu, axis=1)
hosp_dat = np.concatenate((hosp_dat[:, 0:1], hosp_dat), axis=1)
hosp_cumu_s_org = pp.smooth_epidata(np.cumsum(hosp_dat, axis=1))

popu = location_dat["population"].to_numpy()
state_abbr = location_dat["location_name"].to_list()


def validate_config():
    issues = []

    if bin_size <= 0:
        issues.append("timesteps_per_bin/bin_size must be > 0")
    if weeks_ahead <= 0:
        issues.append("forecast_horizon_bins/weeks_ahead must be > 0")
    if np.any((quantiles < 0) | (quantiles > 1)):
        issues.append("quantiles must be in [0, 1]")
    if str(cli_output_format).lower() not in {"csv", "json"}:
        issues.append("forecast_output_format/cli_output_format must be 'csv' or 'json'")
    if hosp_dat.shape[0] != len(popu):
        issues.append("target data rows must match population rows")
    if hosp_dat.shape[0] != len(state_abbr):
        issues.append("target data rows must match location_name rows")

    max_t_binned = hosp_dat.shape[1] // bin_size
    if np.any(retro_lookback < 0) or np.any(retro_lookback > max_t_binned):
        issues.append("retro_lookback values must be between 0 and max available bins")
    if np.any(test_lookback < 0) or np.any(test_lookback > max_t_binned):
        issues.append("test_lookback values must be between 0 and max available bins")

    return issues
