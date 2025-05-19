# LEMMA

## Overview
LEMMA is a project designed to streamline and automate the process of generating accurate and reliable forecasts. This codebase provides tools and utilities for data preprocessing, model training, and forecast generation.

## Streamlit Web Interface
The project includes a Streamlit web interface that allows users to interact with the forecasting models easily. Users can upload their datasets, select models, and visualize the generated forecasts.
The web interface is designed to be user-friendly and intuitive, making it accessible to users with varying levels of technical expertise.

### Steps to Run the Streamlit App
1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Streamlit app:
   ```bash
    streamlit run app.py
    ```
3. Open your web browser and navigate to `http://localhost:8501` to access the app.
4. Upload your dataset, select the desired forecasting model, and generate forecasts.
5. Visualize the results and download the generated forecasts.
6. Explore the various features and functionalities of the app to enhance your forecasting experience.

### Features
- **Upload Input Files**:
    In the 📂 Input Files (Source Files) section:
    - Upload the Target Data (CSV) file for hospitalization data.
    - Upload the Population Data (TXT) file for population data. <br>
    Once uploaded, the app will process and save the files, updating the configuration.
- **Set Forecast Target Configurations**:
    In the 🎯 Target Parameters section: <br>
    Configure Training Steps and Forecasting Steps by adding or removing ranges.
    Adjust other editable parameters like weeks_ahead, start_train, end_train, etc.    
- **Select Forecasting Models**:
    The User can open the Sidebar and select the desired forecasting model from the list of available models. <br>
    The app will display the selected model's hyperparameters and allow the user to adjust them as needed.
- **Generate Forecasts**:
    In the 📊 Forecast Results section: <br>
    - Click the "💾 Save & Run Forecasts" button to to save the updated configuration and generate forecasts. 
    - A progress bar will display the status of the forecast generation.Once completed, you will see a success message.
- **View and Analyze Results**:
    - Select a State from the dropdown to view its data.
    - Use the Select Forecast Day dropdown to choose a specific forecast day.
    - The app will display a Plotly chart showing:
        - Observed data.
        - Predicted data with quantiles.
- **Re-run Predictions**:
    - Click the "Re-run Predictions" button to re-run the predictions with the updated configuration.
    - The app will display a success message once the predictions are completed.

    
## How to Add New Models
To add new models to the LEMMA project, follow these steps:
1. **Place your new model.py file in the input_models Folder**
2. **Define the Model's Parameters** <br> Inside the new model file, define the required parameters and functions
3. **Update model_config.py**
Open the *model_config.py* file in the input_models folder.
Add your new model to the estimation_models or simulation_models dictionary and define its parameters.
4. **Integrate the Model in the Streamlit App**
Open the app.py file.
Ensure the new model appears in the dropdowns for Estimation Model and Simulation Model:    
```python
est_model = st.sidebar.selectbox("Estimation Model", list(model_config.estimation_models.keys()))
sim_model = st.sidebar.selectbox("Simulation Model", list(model_config.simulation_models.keys()))
```
5. **Replace the function call in *process_scenario.py***
6. **Test the Model** <br> Run the Streamlit app and test 
```python
streamlit run app.py
```

## Purpose of Code Files
- **app.py**: The main entry point for the Streamlit web application. It handles user interactions, file uploads, and model selection.
- **model_config.py**: Contains the configuration for different forecasting models, including their parameters and settings.
- **config_param.py**: Serves as the configuration hub for the codebase. It defines key parameters, constants, and data inputs required throughout the forecasting pipeline. These configurations are used by various modules to ensure consistency and flexibility in the forecasting process.
- **gen_predictions.py**: This file is responsible for generating predictors and forecasts. It has two main functions:
    - generate_all_preds:<br>
    Prepares predictors using hospitalization data, population data, and retroactive lookback periods.
    Calls the generate_predictors function to compute predictors.
    Returns the generated predictors (all_preds) and hospitalization data (hosp_dat).
    - generate_preds:<br>
    Prepares training data for regression using the regression_prep function.
    Trains forecasting models using valid training data.
    Generates forecasts for specified test lookback periods and locations.
    Stores predictions in a dictionary (all_test_preds) scaled by population.

- **predictors.py**: The file generates time-series predictors for different retroactive lookback periods and scenarios, which are essential inputs for training forecasting models. It leverages parallel processing to improve performance and handles large-scale scenario simulations efficiently.

- **process_scenario.py**: The file processes individual scenarios by simulating hospitalization forecasts based on parameterized models. It is a core component of the forecasting pipeline, enabling the generation of scenario-specific predictions for use in broader analyses.

- **utils.py**: The utils.py file provides essential preprocessing functions to aggregate data, prepare regression inputs, and clean datasets for model training. These utilities streamline the data pipeline for forecasting models in the LEMMA project.
Functions Summaries: 
    - **bin_array**: Aggregates data into bins along a specified axis.
    - **regression_prep**: Prepares training data for regression models by converting hospitalization data into a numpy array, processes predictors for each lookback scenario, bins predictors and target data, normalizes them by the population and applies decay factors. Collects training features (X_train), targets (Y_train), weights (W_train), and metadata (ID).
    - **prepare_data_for_model**: Prepares data for model training by combining old and new datasets by extracting  relevant columns. Combines them into a single dataset and removes rows with missing values.
    Separates and returns features (X_clean), targets (y_clean), and weights (weights_clean).
    
