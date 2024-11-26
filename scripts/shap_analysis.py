import pandas as pd
import os


def run_shap_analysis(species, cluster, prepared_data, model_path, scaler_path):
    """
    Perform SHAP analysis on prepared data for a specific species and cluster.

    Args:
        species (str): Species name
        cluster (int): Cluster label
        prepared_data (pd.DataFrame): Prepared DataFrame for analysis.
        model_path (str): Path to saved model.
        scaler_path (str): Path to saved scaler.

    Returns:
        dict: Analysis results with species, cluster, and mean SHAP values.
    """
    import numpy as np
    import joblib
    import shap
    from tensorflow.keras.models import load_model
    

    # Features used in the model (same as in `prepare_future_data`)
    features = [
        'landed_w_kg', 'Cluster_Label', 'mean_temp_30m', 'mean_temp_10m',
        'thetao_sfc=6', 'thetao_sfc=7.92956018447876', 'thetao_sfc=9.572997093200684',
        'thetao_sfc=11.40499973297119', 'thetao_sfc=13.46714019775391',
        'thetao_sfc=15.8100700378418', 'thetao_sfc=18.49555969238281',
        'thetao_sfc=21.59881973266602', 'thetao_sfc=25.21141052246094',
        'thetao_sfc=29.44473075866699'
    ]

    try:
        # Load model and scaler
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)

        # Scale the features
        prepared_data_scaled = scaler.transform(prepared_data[features])
        prepared_data_unscaled = scaler.inverse_transform(prepared_data_scaled)

        # Ensure sufficient data for SHAP background
        look_back = 6
        if len(prepared_data_unscaled) < look_back:
            raise ValueError("Insufficient data for SHAP analysis background set.")

        # Create sliding window view for background
        background_unscaled = np.lib.stride_tricks.sliding_window_view(
            prepared_data_unscaled, (look_back, len(features))
        )

        # Create SHAP explainer
        explainer = shap.GradientExplainer(model, background_unscaled)

        # Calculate SHAP values
        X_inputs_unscaled = background_unscaled.reshape(-1, look_back, len(features))
        shap_values = explainer.shap_values(X_inputs_unscaled)

        # Compute mean SHAP values
        shap_mean_values = {
            feature: np.mean(shap_values[0][:, i])
            for i, feature in enumerate(features)
        }

        # Construct and return result
        return {
            'species': species,
            'cluster': cluster,
            **shap_mean_values
        }

    except Exception as e:
        raise RuntimeError(f"SHAP analysis failed: {str(e)}") from e
    


def evaluate_all_models(models_folder, future_data_file, shap_analysis_function):
    """
    Evaluate all models in a specified folder and generate a list of SHAP values for each model.

    Args:
        models_folder (str): Path to the folder containing model files.
        future_data_file (str): Path to the future data CSV.
        shap_analysis_function (callable): Function to perform SHAP analysis.

    Returns:
        pd.DataFrame: DataFrame containing cluster, species, and SHAP values for each feature.
    """
    import os
    import pandas as pd

    # List all model files in the folder
    model_files = [f for f in os.listdir(models_folder) if f.endswith('.h5')]

    # Initialize a list to store results
    shap_results = []

    # Loop through each model file
    for model_file in model_files:
        try:
            # Extract species and cluster from the filename
            parts = model_file.split('_')
            species = parts[0]
            cluster = int(parts[2].replace('cluster', '').replace('.h5', ''))

            # Paths for model and scaler
            model_path = os.path.join(models_folder, model_file)

            # Eliminar "_model" del nombre y generar el path del escalador
            scaler_path = model_path.replace('_model', '').replace('.h5', '_scaler.pkl')

            # Check if scaler exists
            if not os.path.exists(scaler_path):
                print(f"Scaler file missing for model {model_file}. Skipping...")
                continue

            # Load and prepare future data
            future_data = pd.read_csv(future_data_file)
            prepared_data, _ = prepare_future_data(future_data, cluster)

            # Perform SHAP analysis
            result = shap_analysis_function(
                species=species,
                cluster=cluster,
                prepared_data=prepared_data,
                model_path=model_path,
                scaler_path=scaler_path
            )

            # Append the result to the list
            shap_results.append(result)

        except Exception as e:
            print(f"Error processing model {model_file}: {e}")

    # Convert the results to a DataFrame
    return pd.DataFrame(shap_results)



def prepare_future_data(future_data, cluster, column_name_mapping=None):
    """
    Prepares future data for the SHAP analysis by renaming columns, adding missing columns, and selecting features.

    Args:
        future_data (pd.DataFrame): DataFrame containing future data.
        cluster (int): Cluster label to assign to the data.
        column_name_mapping (dict, optional): Dictionary for renaming columns. If None, uses a default mapping.

    Returns:
        pd.DataFrame: Prepared DataFrame with renamed columns and added default values.
        list: List of features used in the model.
    """
    if column_name_mapping is None:
        column_name_mapping = {
            'cluster': 'Cluster_Label',
            'depth_mean_30m': 'mean_temp_30m',
            'depth_mean_10m': 'mean_temp_10m',
            'depth_6.00': 'thetao_sfc=6',
            'depth_7.93': 'thetao_sfc=7.92956018447876',
            'depth_9.57': 'thetao_sfc=9.572997093200684',
            'depth_11.40': 'thetao_sfc=11.40499973297119',
            'depth_13.47': 'thetao_sfc=13.46714019775391',
            'depth_15.81': 'thetao_sfc=15.8100700378418',
            'depth_18.50': 'thetao_sfc=18.49555969238281',
            'depth_21.60': 'thetao_sfc=21.59881973266602',
            'depth_25.21': 'thetao_sfc=25.21141052246094',
            'depth_29.44': 'thetao_sfc=29.44473075866699'
        }

    # Rename columns in future data
    future_data.rename(columns=column_name_mapping, inplace=True)

    # Add missing columns with default values
    future_data['Cluster_Label'] = cluster
    future_data['landed_w_kg'] = 0  # Default value for landed weight

    # Define features used in the model
    features = [
        'landed_w_kg', 'Cluster_Label', 'mean_temp_30m', 'mean_temp_10m',
        'thetao_sfc=6', 'thetao_sfc=7.92956018447876', 'thetao_sfc=9.572997093200684',
        'thetao_sfc=11.40499973297119', 'thetao_sfc=13.46714019775391',
        'thetao_sfc=15.8100700378418', 'thetao_sfc=18.49555969238281',
        'thetao_sfc=21.59881973266602', 'thetao_sfc=25.21141052246094',
        'thetao_sfc=29.44473075866699'
    ]

    # Return prepared data and feature list
    return future_data, features
