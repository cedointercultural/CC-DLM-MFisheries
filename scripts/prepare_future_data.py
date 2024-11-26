import pandas as pd

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
