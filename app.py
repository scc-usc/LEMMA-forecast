import streamlit as st
import time
import config_param
import importlib
import numpy as np
import gen_predictions
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import sys
import os
from datetime import timedelta
#sys.path.append('~/code/lemma-repo/LEMMA')
from shared_utils.utils import bin_array
from approaches import get_approach, approaches as model_config_approaches
if "forecast_ready" not in st.session_state:
    st.session_state["forecast_ready"] = False

st.set_page_config(initial_sidebar_state= "collapsed", layout="wide", page_title="Forecast Generator", page_icon="💡")

CONFIG_FILE = "user_config.py"

CONFIG_KEY_MAP = {
    "selected_approach": "predictor_approach",
    "weeks_ahead": "bins_ahead",
    "bin_size": "timesteps_per_bin",
    "start_train": "training_window_start_bin",
    "end_train": "training_window_end_bin",
    "start_test": "forecast_window_start_bin",
    "end_test": "forecast_window_end_bin",
    "ts_dat": "target_data_path",
    "location_dat_path": "location_metadata_path",
    "ar_p_list": "arima_autoregressive_orders",
    "d_list": "arima_differencing_orders",
    "flat_k_list": "flatline_lag_timesteps",
    "rlag_list": "retro_lag_bins",
}

EDITABLE_PARAMS = ["weeks_ahead", "bin_size", "start_train", "end_train", "start_test", "end_test", "quantiles"]
DISPLAY_PARAM_NAMES = {
    "weeks_ahead": "bins_ahead",
    "bin_size": "timesteps_per_bin",
}

# EDITABLE_PARAMS = PARAMS + hyper_params
st.markdown("👈 **Open the sidebar to update predictors**")

st.markdown(
    """
    <h1 style="text-align: center;">Forecast Generation 💡</h1>
    <style>
        .block-container {
            padding-top: 4rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)


def get_parameters():
    return {attr: getattr(config_param, attr) for attr in dir(config_param) if not attr.startswith("__")}


def _timedelta_from_timesteps(num_steps):
    unit = str(getattr(config_param, "base_time_step_unit", "day")).lower().strip()
    if unit in {"day", "days", "d"}:
        return timedelta(days=int(num_steps))
    if unit in {"week", "weeks", "w"}:
        return timedelta(weeks=int(num_steps))
    return timedelta(days=int(num_steps))


def _timedelta_from_bins(num_bins):
    return _timedelta_from_timesteps(int(num_bins) * int(config_param.bin_size))


def _steps_between_dates(start_ts, end_ts):
    delta_days = (pd.Timestamp(end_ts) - pd.Timestamp(start_ts)).days
    unit = str(getattr(config_param, "base_time_step_unit", "day")).lower().strip()
    if unit in {"day", "days", "d"}:
        return int(delta_days)
    if unit in {"week", "weeks", "w"}:
        return int(delta_days // 7)
    return int(delta_days)


def _date_to_bin_index(date_value, reference_date, max_bin_idx):
    steps = _steps_between_dates(reference_date, pd.Timestamp(date_value))
    bin_idx = int(steps // int(config_param.bin_size))
    return int(np.clip(bin_idx, 0, max_bin_idx))


def validate_window_bins(train_start_bin, train_end_bin, forecast_start_bin, forecast_end_bin):
    errors = []
    if train_end_bin < train_start_bin:
        errors.append("Training End Date must be on or after Training Start Date.")
    if forecast_end_bin < forecast_start_bin:
        errors.append("Forecast End Date must be on or after Forecast Start Date.")
    return errors


def reload_runtime_config():
    """Reload user-facing and derived config modules so changes apply immediately."""
    importlib.invalidate_caches()

    user_module = sys.modules.get("user_config")
    if user_module is not None:
        importlib.reload(user_module)
    else:
        importlib.import_module("user_config")

    global config_param
    config_module = sys.modules.get("config_param")
    if config_module is not None:
        config_param = importlib.reload(config_module)
    else:
        config_param = importlib.import_module("config_param")


# FOR ADDING PARAMETERS THAT DO NOT EXIST IN THE CONFIG FILE
def update_config_file(updated_keys):
    mapped_updates = {
        CONFIG_KEY_MAP.get(key, key): value for key, value in updated_keys.items()
    }

    with open(CONFIG_FILE, "r") as f:
        lines = f.readlines()

    new_lines = []
    found_keys = set()

    for line in lines:
        if "=" not in line:
            new_lines.append(line)
            continue

        key = line.split("=")[0].strip()

        if key in mapped_updates:
            value = mapped_updates[key]
            found_keys.add(key)
        else:
            new_lines.append(line)
            continue

        # Format value
        if key == "hosp_dat":
            formatted_value = f"pd.read_csv('{value}')"
        elif key == "location_dat":
            formatted_value = f"pd.read_csv('{value}')"
        elif key == "popu":
            formatted_value = f"np.loadtxt('{value}')"
        elif key == "halpha_list":
            try:
                start = round(float(value[0]), 5)
                step = round(float(value[1] - value[0]), 5)
                stop = round(float(value[-1] + step), 5)
                formatted_value = f"np.arange({start}, {stop}, {step})"
            except Exception:
                formatted_value = f"np.array({value.tolist()})"
        elif isinstance(value, np.ndarray):
            formatted_value = f"np.array({value.tolist()})"
        elif isinstance(value, list):
            formatted_value = str(value)
        elif isinstance(value, str):
            formatted_value = f'"{value}"'
        else:
            formatted_value = value

        new_lines.append(f"{key} = {formatted_value}\n")

    # Add any missing (new) keys
    for key in mapped_updates.keys() - found_keys:
        value = mapped_updates[key]

        if key == "hosp_dat":
            formatted_value = f"pd.read_csv('{value}')"
        elif key == "location_dat":
            formatted_value = f"pd.read_csv('{value}')"
        elif key == "popu":
            formatted_value = f"np.loadtxt('{value}')"
        elif key == "halpha_list":
            try:
                start = round(float(value[0]), 5)
                step = round(float(value[1] - value[0]), 5)
                stop = round(float(value[-1] + step), 5)
                formatted_value = f"np.arange({start}, {stop}, {step})"
            except Exception:
                formatted_value = f"np.array({value.tolist()})"
        elif isinstance(value, np.ndarray):
            formatted_value = f"np.array({value.tolist()})"
        elif isinstance(value, list):
            formatted_value = str(value)
        elif isinstance(value, str):
            formatted_value = f'"{value}"'
        else:
            formatted_value = value

        new_lines.append(f"{key} = {formatted_value}\n")

    # Atomic write to avoid partial writes.
    tmp_path = CONFIG_FILE + ".tmp"
    with open(tmp_path, "w") as f:
        f.writelines(new_lines)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, CONFIG_FILE)





params = get_parameters()
updated_params = {}

st.sidebar.header("🔬 Predictor Settings")
st.sidebar.markdown("### 🔧 Approach Selection")

approach_options = list(model_config_approaches.keys())
config_approach = getattr(config_param, "selected_approach", "ARIMA")
default_approach = config_approach if config_approach in approach_options else "ARIMA"
default_approach_index = approach_options.index(default_approach) if default_approach in approach_options else 0
selected_approach_label = st.sidebar.selectbox("Approach", approach_options, index=default_approach_index)

# Persist and write to config so back-end uses it
st.session_state["selected_approach"] = selected_approach_label
updated_params["selected_approach"] = selected_approach_label
st.markdown("---")

st.sidebar.markdown("### 🔧 Hyperparameters")

model_hyperparams = model_config_approaches.get(selected_approach_label, [])

for param in model_hyperparams:
    default = None
    input_type = "text"
    param_key = f"hyperparam_{param}"  # Unique key per param

    if hasattr(config_param, param):
        default = getattr(config_param, param)
        if isinstance(default, int):
            input_type = "int"
        elif isinstance(default, float):
            input_type = "float"
        elif isinstance(default, (list, np.ndarray)):
            input_type = "array"

    # Render input field with unique key
    if input_type == "int":
        val = st.sidebar.number_input(
            param,
            value=default if default is not None else 0,
            step=1,
            key=param_key
        )
    elif input_type == "float":
        val = st.sidebar.number_input(
            param,
            value=default if default is not None else 0.0,
            format="%.5f",
            key=param_key
        )
    elif input_type == "array":
        val_str = st.sidebar.text_input(
            param,
            value=", ".join(map(str, default)) if default is not None else "",
            key=param_key
        )
        val = np.array([float(v.strip()) for v in val_str.split(",") if v.strip()])
    else:
        val = st.sidebar.text_input(
            param,
            value=str(default) if default is not None else "",
            key=param_key
        )
        try:
            val_eval = eval(val)
            val = val_eval
        except Exception:
            pass

    updated_params[param] = val
with st.sidebar.expander("ℹ️ How to configure hyperparameters"):
    try:
        approach_mod = get_approach(selected_approach_label)
        help_text = getattr(approach_mod, "HYPERPARAMS_DOC", None)
        if not help_text:
            help_text = (
                f"No detailed hyperparameter description provided for '{selected_approach_label}'.\n\n"
                "Tip: Add HYPERPARAMS_DOC = \"\"\"...\"\"\" in the approach module to populate this help."
            )
        st.markdown(help_text)
    except Exception as _e:
        st.markdown(
            "No detailed hyperparameter description available."
        )


if "train_ranges" not in st.session_state:
    st.session_state.train_ranges = [(params["start_train"], params["end_train"])]
if "test_ranges" not in st.session_state:
    st.session_state.test_ranges = [(params["start_test"], params["end_test"])]

def render_range_inputs(label, ranges_key):
    st.markdown(f"### {label} Ranges")

    # Initialize add/remove trackers in session_state
    if f"{ranges_key}_add_clicked" not in st.session_state:
        st.session_state[f"{ranges_key}_add_clicked"] = False
    if f"{ranges_key}_remove_clicked" not in st.session_state:
        st.session_state[f"{ranges_key}_remove_clicked"] = None

    # Handle add clicked
    if st.session_state[f"{ranges_key}_add_clicked"]:
        st.session_state[ranges_key].append((0, 0))
        st.session_state[f"{ranges_key}_add_clicked"] = False  # reset

    # Handle remove clicked
    remove_idx = st.session_state[f"{ranges_key}_remove_clicked"]
    if remove_idx is not None and 0 <= remove_idx < len(st.session_state[ranges_key]):
        st.session_state[ranges_key].pop(remove_idx)
        st.session_state[f"{ranges_key}_remove_clicked"] = None  # reset

    new_ranges = []

    for i, (start, end) in enumerate(st.session_state[ranges_key]):
        cols = st.columns([3, 3, 1])
        with cols[0]:
            start_val = st.number_input(f"{label} Start {i+1}", value=start, key=f"{ranges_key}_start_{i}")
        with cols[1]:
            end_val = st.number_input(f"{label} End {i+1}", value=end, key=f"{ranges_key}_end_{i}", min_value=start_val)
        with cols[2]:
            if st.button("➖", key=f"{ranges_key}_remove_{i}"):
                st.session_state[f"{ranges_key}_remove_clicked"] = i

        new_ranges.append((start_val, end_val))

    st.session_state[ranges_key] = new_ranges

    # Add range button
    if st.button(f"➕ Add {label} Range", key=f"add_{ranges_key}"):
        st.session_state[f"{ranges_key}_add_clicked"] = True

    return new_ranges



left_col, right_col = st.columns(2)  # Adjust width

with left_col:
    st.markdown("### 📂 Input Files (Source Files)")
    configured_hv = str(getattr(config_param, "hubverse_input_path", "") or "").strip()
    default_mode = "Hubverse Target Data" if configured_hv else "Matrix + Location/Population"
    input_mode = st.radio(
        "Observed Data Input Mode",
        ["Hubverse Target Data", "Matrix + Location/Population"],
        index=0 if default_mode == "Hubverse Target Data" else 1,
        horizontal=True,
    )

    if input_mode == "Hubverse Target Data":
        st.caption("Upload Hubverse observed target-data CSV (location/target_end_date + weekly_rate or observation).")
        hubverse_observed_file = st.file_uploader("Upload Hubverse Target Data (CSV)", type=["csv"])
        hubverse_location_file = st.file_uploader(
            "Upload Location, Population Data (optional)",
            type=["csv"],
            help="Optional in Hubverse mode. If provided, it is used to map populations by location_name.",
        )
        hubverse_target_value = st.text_input(
            "Hubverse target to use (optional)",
            value=str(getattr(config_param, "hubverse_target", "") or ""),
            help="If left empty and the file has a target column, each (location, target) pair is treated as a distinct location row.",
        )

        if hubverse_location_file:
            try:
                location_dat = pd.read_csv(hubverse_location_file, delimiter=',')
                location_dat.to_csv("data/location_dat.csv", index=False)
                updated_params["location_dat_path"] = "data/location_dat.csv"
                update_config_file(updated_params)
                reload_runtime_config()
                st.success("✅ Optional location/population data updated for Hubverse mode!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error loading optional location file: {e}")

        active_hubverse_path = configured_hv if configured_hv else None
        if hubverse_observed_file:
            hubverse_sig = f"{hubverse_observed_file.name}:{hubverse_observed_file.size}"
            hubverse_path = "data/hubverse_target_data.csv"
            active_hubverse_path = hubverse_path
            if st.session_state.get("hubverse_upload_sig") != hubverse_sig:
                try:
                    with open(hubverse_path, "wb") as f:
                        f.write(hubverse_observed_file.getbuffer())

                    updated_params["hubverse_input_path"] = hubverse_path
                    updated_params["hubvsereInput"] = hubverse_path
                    updated_params["hubverse_target"] = hubverse_target_value.strip() or None
                    update_config_file(updated_params)
                    reload_runtime_config()
                    st.session_state["hubverse_upload_sig"] = hubverse_sig
                    st.success("✅ Hubverse observed target-data input enabled!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error loading Hubverse file: {e}")

        if active_hubverse_path is not None:
            current_target = str(getattr(config_param, "hubverse_target", "") or "").strip()
            desired_target = hubverse_target_value.strip()
            if desired_target != current_target:
                try:
                    updated_params["hubverse_input_path"] = active_hubverse_path
                    updated_params["hubvsereInput"] = active_hubverse_path
                    updated_params["hubverse_target"] = desired_target or None
                    update_config_file(updated_params)
                    reload_runtime_config()
                    st.success("✅ Hubverse target updated!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error applying Hubverse target: {e}")

    else:
        hosp_dat_file = st.file_uploader("Upload Target Data (CSV)", type=["csv"])
        if hosp_dat_file:
            matrix_sig = f"{hosp_dat_file.name}:{hosp_dat_file.size}"
            if st.session_state.get("matrix_upload_sig") != matrix_sig:
                hosp_dat = pd.read_csv(hosp_dat_file, header=None)
                try:
                    hosp_dat.to_csv("data/ts_dat.csv", index=False, header=False)
                    updated_params["ts_dat"] = "data/ts_dat.csv"
                    updated_params["hubverse_input_path"] = None
                    updated_params["hubvsereInput"] = None
                    updated_params["hubverse_target"] = None
                    update_config_file(updated_params)
                    reload_runtime_config()
                    st.session_state["matrix_upload_sig"] = matrix_sig
                    st.success("✅ Target matrix data updated!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error loading file: {e}")

        location_file = st.file_uploader("Upload Location, Population Data (CSV)", type=["csv"])
        if location_file:
            try:
                location_dat = pd.read_csv(location_file, delimiter=',')
                location_dat.to_csv("data/location_dat.csv", index=False)
                updated_params["location_dat_path"] = "data/location_dat.csv"
                update_config_file(updated_params)
                reload_runtime_config()
                st.success("✅ Location data updated!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error loading file: {e}")
         

    st.markdown("---")
    st.markdown("### 🎯 Target Parameters")
    st.subheader("Forecast Target Configurations 🔧")

    n_binned_steps = len(bin_array(config_param.hosp_dat[0, :], 0, config_param.bin_size, 0))
    max_bin_idx = max(0, n_binned_steps - 1)
    reference_date = pd.Timestamp(config_param.zero_date)
    min_date = reference_date.date()
    max_date = (reference_date + _timedelta_from_bins(max_bin_idx)).date()

    train_col_1, train_col_2 = st.columns(2)
    with train_col_1:
        train_start_default = (reference_date + _timedelta_from_bins(max(0, min(int(config_param.start_train), max_bin_idx)))).date()
        train_start_date = st.date_input(
            "Training Start Date",
            value=train_start_default,
            min_value=min_date,
            max_value=max_date,
        )
    with train_col_2:
        train_end_default = (reference_date + _timedelta_from_bins(max(0, min(int(config_param.end_train), max_bin_idx)))).date()
        train_end_date = st.date_input(
            "Training End Date",
            value=train_end_default,
            min_value=min_date,
            max_value=max_date,
        )

    forecast_col_1, forecast_col_2 = st.columns(2)
    forecast_mode = st.radio(
        "Forecast Origin Mode",
        options=["Single origin date", "Date range"],
        horizontal=True,
        help="Use a single origin date for one-time forecast generation, or a range for multiple origins.",
    )

    with forecast_col_1:
        forecast_start_default = (reference_date + _timedelta_from_bins(max(0, min(int(config_param.start_test), max_bin_idx)))).date()
        forecast_start_date = st.date_input(
            "Forecast Start Date",
            value=forecast_start_default,
            min_value=min_date,
            max_value=max_date,
        )
    with forecast_col_2:
        if forecast_mode == "Single origin date":
            forecast_end_date = forecast_start_date
            st.date_input(
                "Forecast End Date",
                value=forecast_end_date,
                min_value=min_date,
                max_value=max_date,
                disabled=True,
            )
        else:
            forecast_end_default = (reference_date + _timedelta_from_bins(max(0, min(int(config_param.end_test), max_bin_idx)))).date()
            forecast_end_date = st.date_input(
                "Forecast End Date",
                value=forecast_end_default,
                min_value=min_date,
                max_value=max_date,
            )

    train_start_bin = _date_to_bin_index(train_start_date, reference_date, max_bin_idx)
    train_end_bin = _date_to_bin_index(train_end_date, reference_date, max_bin_idx)
    forecast_start_bin = _date_to_bin_index(forecast_start_date, reference_date, max_bin_idx)
    forecast_end_bin = _date_to_bin_index(forecast_end_date, reference_date, max_bin_idx)

    train_start_label = pd.Timestamp(train_start_date).date().isoformat()
    train_end_label = pd.Timestamp(train_end_date).date().isoformat()
    forecast_start_label = pd.Timestamp(forecast_start_date).date().isoformat()
    forecast_end_label = pd.Timestamp(forecast_end_date).date().isoformat()

    updated_params["training_window_start_date"] = train_start_label
    updated_params["training_window_end_date"] = train_end_label
    updated_params["forecast_window_start_date"] = forecast_start_label
    updated_params["forecast_window_end_date"] = forecast_end_label

    # Merge lookbacks
    train_days = sorted(set(np.arange(train_start_bin, train_end_bin + 2)))
    train_lookback = [n_binned_steps - i for i in train_days]
    test_days = sorted(set(np.arange(forecast_start_bin, forecast_end_bin + 1)))
    test_lookback = [n_binned_steps - i for i in test_days]
    retro_lookback = sorted(set(train_lookback + test_lookback))

    # st.markdown("#### Preview")
    # st.code(f"Run the predictors for:  {retro_lookback}")
    # st.code(f"Generate forecasts for: {test_lookback}")




    skip_keys = {"start_train", "end_train", "start_test", "end_test"}
    for idx, key in enumerate(EDITABLE_PARAMS):
        if key in skip_keys:
            continue
        
        if key not in params:
            continue

        value = params[key]
        new_value = value  # Default
        display_name = DISPLAY_PARAM_NAMES.get(key, key)


        label_col, input_col = st.columns([2, 3])
        with label_col:
            st.write(f"**{display_name}**")

        with input_col:
            if isinstance(value, int):
                new_value = st.number_input(f"{display_name}", value=value, step=1, label_visibility="collapsed", min_value=0)
            elif isinstance(value, float):
                new_value = st.number_input(f"{display_name}", value=value, step=0.01, label_visibility="collapsed", min_value=0.0)
            elif isinstance(value, str):
                new_value = st.text_input(f"{display_name}", value=value, label_visibility="collapsed")
            elif isinstance(value, list):
                new_value = st.text_area(f"{display_name} (Comma-separated)", value=", ".join(map(str, value)))
                new_value = [v.strip() for v in new_value.split(",") if v.strip()]
                try:
                    new_value = [float(v) for v in new_value]
                    if any(v < 0 for v in new_value):
                        st.error(f"❌ {display_name} cannot contain negative values.")
                        new_value = value  # Revert to old value
                except ValueError:
                    st.error(f"❌ Invalid input for {display_name}. Please enter numbers only.")
                    new_value = value  # Revert to old value

            elif isinstance(value, np.ndarray):
                new_value = st.text_area(f"{display_name} (Comma-separated)", value=", ".join(map(str, value.tolist())))
                try:
                    new_value = np.array([float(v.strip()) for v in new_value.split(",") if v.strip()])
                    if (new_value < 0).any():
                        st.error(f"❌ {display_name} cannot contain negative values.")
                        new_value = value  # Revert to old value
                except ValueError:
                    st.error(f"❌ Invalid input for {display_name}. Please enter numbers only.")
                    new_value = value

            # Store updates only if changed
        if isinstance(value, np.ndarray) and not np.array_equal(new_value, value):
            updated_params[key] = new_value
        elif isinstance(value, list) and new_value != value:
            updated_params[key] = new_value
        elif not isinstance(value, (list, np.ndarray)) and new_value != value:
            updated_params[key] = new_value


with right_col:
   

    st.subheader("📊 Forecast Results")
    # Choose ensemble strategy (two options)
    ensemble_options = ["Random Forest", "Basic"]
    config_ensemble = getattr(config_param, "ensemble_method", "Random Forest")
    default_ensemble_index = ensemble_options.index(config_ensemble) if config_ensemble in ensemble_options else 0
    ensemble_choice = st.selectbox(
        "Ensemble method",
        options=ensemble_options,
        index=default_ensemble_index,
        help="Choose how to combine predictor outputs into final forecasts"
    )
    st.session_state["ensemble_method"] = ensemble_choice
    updated_params["ensemble_method"] = ensemble_choice
    # Button logic: always show Save & Run; show Rerun Ensemble if predictors exist
    has_preds = ("all_preds" in st.session_state) and isinstance(st.session_state["all_preds"], dict) and len(st.session_state["all_preds"]) > 0

    if st.button("💾 Save & Run Forecasts", use_container_width=True):
        window_errors = validate_window_bins(train_start_bin, train_end_bin, forecast_start_bin, forecast_end_bin)
        if window_errors:
            for msg in window_errors:
                st.error(f"❌ {msg}")
            st.stop()

        if updated_params:
            try:
                update_config_file(updated_params)
                reload_runtime_config()
                st.toast("✅ Configuration Updated!")
            except Exception as e:
                st.error(f"❌ Error applying configuration: {e}")
                st.stop()

        progress_bar = st.progress(0.0)
        progress_text = st.empty()  # Create an empty container for text

        def update_progress(progress):
            progress_bar.progress(progress)
            progress_text.info(f"Progress: {int(progress * 100)}%")

        st.session_state["all_preds"], st.session_state["hosp_dat"] = gen_predictions.generate_all_preds(update_progress)

        st.toast("✅ Predictors generated successfully!")

        # Build ensemble instance from selection
        import config_model as _cfg_model
        ensemble = _cfg_model.create_ensemble(st.session_state.get("ensemble_method", ensemble_choice))

        with st.spinner('Generating "ensemble" forecasts...'):
            st.session_state["preds"] = gen_predictions.generate_preds(
                st.session_state["all_preds"], st.session_state["hosp_dat"], ensemble=ensemble
            )
            st.session_state["preds_df"] = gen_predictions.predictions_to_long_df(st.session_state["preds"])
        st.toast("✅ Predictions generated successfully!")
        st.session_state["forecast_ready"] = True

    if has_preds:
        if st.button("🔁 Rerun Ensemble", use_container_width=True):
            window_errors = validate_window_bins(train_start_bin, train_end_bin, forecast_start_bin, forecast_end_bin)
            if window_errors:
                for msg in window_errors:
                    st.error(f"❌ {msg}")
                st.stop()

            # Optional: update only ensemble-related params (e.g., quantiles). For now, apply any updates.
            if updated_params:
                try:
                    update_config_file(updated_params)
                    reload_runtime_config()
                    st.toast("✅ Ensemble Configuration Updated!")
                except Exception as e:
                    st.error(f"❌ Error applying configuration: {e}")
                    st.stop()

            import config_model as _cfg_model
            ensemble = _cfg_model.create_ensemble(st.session_state.get("ensemble_method", ensemble_choice))
            with st.spinner("Re-generating ensemble predictions..."):
                st.session_state["preds"] = gen_predictions.generate_preds(
                    st.session_state["all_preds"], st.session_state["hosp_dat"], ensemble=ensemble
                )
                st.session_state["preds_df"] = gen_predictions.predictions_to_long_df(st.session_state["preds"])
            st.toast("✅ Ensemble re-generated successfully!")
            st.session_state["forecast_ready"] = True

    state_abbr = config_param.state_abbr
    
    state_select = st.selectbox("Select State", state_abbr)
    cid = state_abbr.index(state_select)    
    hosp_dat = config_param.hosp_dat
    maxt = hosp_dat.shape[1]
    observed_data = bin_array((config_param.hosp_dat[cid, :]), 0, config_param.bin_size, 0)
    reference_date = pd.Timestamp(config_param.zero_date)
    observed_dates = [
        (reference_date + _timedelta_from_bins(i)).date().isoformat()
        for i in range(len(observed_data))
    ]



# Create the Plotly figure
    fig = go.Figure()

    # Add observed data trace
    fig.add_trace(go.Scatter(
        x=observed_dates,
        y=observed_data,
        mode='lines',
        name='Observed'
    ))

    fig.update_layout(
        title=f"📊 Ground Truth for {state_select}",
        xaxis_title="Date",
        yaxis_title="Values",
        template="plotly_white"
    )

    if st.session_state.get("forecast_ready"):
        preds = st.session_state["preds"]

        if "preds_df" not in st.session_state:
            st.session_state["preds_df"] = gen_predictions.predictions_to_long_df(preds)

        preds_df = st.session_state["preds_df"]
        required_pred_cols = {"origin_date", "location", "horizon", "output_type_id", "value"}
        if preds_df.empty or not required_pred_cols.issubset(set(preds_df.columns)):
            st.warning("No forecasts available for the current selection. Try an earlier forecast origin date or broader forecast window.")
            st.plotly_chart(fig, use_container_width=True)
            st.stop()

        st.download_button(
            label="⬇️ Download All Forecasts (CSV)",
            data=preds_df.to_csv(index=False).encode("utf-8"),
            file_name="lemma_forecasts.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.caption(f"Rows available for download: {len(preds_df)}")

        state_preds = preds_df[preds_df["location"] == state_select]
        origin_options = sorted(state_preds["origin_date"].astype(str).unique().tolist())

        if not origin_options:
            st.warning("Prediction not available for this selection.")
        else:
            origin_ts_list = sorted(pd.to_datetime(origin_options).tolist())
            origin_pick = st.date_input(
                "Select Forecast Origin Date",
                value=origin_ts_list[-1].date(),
                min_value=origin_ts_list[0].date(),
                max_value=origin_ts_list[-1].date(),
            )
            origin_ts = pd.Timestamp(origin_pick)
            nearest_origin = min(origin_ts_list, key=lambda d: abs((d - origin_ts).days))
            origin_date = nearest_origin.date().isoformat()

            selected_preds = state_preds[state_preds["origin_date"].astype(str) == origin_date]
            quantile_ids = sorted(selected_preds["output_type_id"].unique().tolist())
            origin_ts = pd.Timestamp(origin_date)

            for q in quantile_ids:
                q_preds = selected_preds[selected_preds["output_type_id"] == q].sort_values("horizon")
                pred_dates = [
                    (origin_ts + _timedelta_from_bins(int(h))).date().isoformat()
                    for h in q_preds["horizon"].tolist()
                ]
                fig.add_trace(go.Scatter(
                    x=pred_dates,
                    y=q_preds["value"].to_numpy(),
                    mode='lines',
                    name=f"Predicted ({q})"
                ))

            fig.update_layout(
                title=f"📊 Ground Truth and Forecasts for {state_select}",
            )

    # Render the Plotly chart
    st.plotly_chart(fig, use_container_width=True)