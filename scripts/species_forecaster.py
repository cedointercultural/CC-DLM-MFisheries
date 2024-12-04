import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class ForecastConfig:
    """Configuration class for forecast parameters"""
    look_back: int = 6
    n_bootstrap: int = 1
    alpha: float = 0.01
    features: List[str] = None

    def __post_init__(self):
        if self.features is None:
            self.features = [
                'landed_w_kg', 'Cluster_Label', 'mean_temp_30m', 'mean_temp_10m',
                *[f'thetao_sfc={temp}' for temp in [
                    6, 7.92956018447876, 9.572997093200684, 11.40499973297119,
                    13.46714019775391, 15.8100700378418, 18.49555969238281,
                    21.59881973266602, 25.21141052246094, 29.44473075866699
                ]]
            ]

class SpeciesForecaster:
    def __init__(self, model_path: Path, scaler_path: Path, config: ForecastConfig):
        """
        Initialize the SpeciesForecaster with model, scaler and configuration.
        
        Args:
            model_path (Path): Path to the saved model
            scaler_path (Path): Path to the saved scaler
            config (ForecastConfig): Configuration object with forecast parameters
        """
        self.model = tf.keras.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        self.config = config
        
    def filter_data(self, data: pd.DataFrame, species_name: str, cluster: int) -> pd.DataFrame:
        """
        Filter data for specific species and cluster
        
        Args:
            data (pd.DataFrame): Input DataFrame containing all data
            species_name (str): Name of the species to filter
            cluster (int): Cluster number to filter
            
        Returns:
            pd.DataFrame: Filtered DataFrame containing only the specified species and cluster
            
        Raises:
            ValueError: If no data is found for the specified species and cluster
        """
        # Filtrar por especie y cluster
        filtered_data = data[
            (data['species'] == species_name) & 
            (data['Cluster_Label'] == cluster)
        ].copy()
        
        # Verificar si hay datos después del filtrado
        if filtered_data.empty:
            raise ValueError(
                f"No se encontraron datos para la especie '{species_name}' "
                f"en el cluster {cluster}"
            )
        
        # Ordenar por fecha si existe una columna de fecha
        if 'date' in filtered_data.columns:
            filtered_data.sort_values('date', inplace=True)
        
        # Asegurarse de que todas las columnas necesarias estén presentes
        missing_features = [f for f in self.config.features if f not in filtered_data.columns]
        if missing_features:
            raise ValueError(
                f"Faltan las siguientes columnas en los datos: {missing_features}"
            )
            
        return filtered_data
        
    def _prepare_input(self, data: np.ndarray) -> np.ndarray:
        """
        Prepare input data for model prediction
        
        Args:
            data (np.ndarray): Input data to be reshaped
        
        Returns:
            np.ndarray: Reshaped input data
        """
        return np.reshape(data, (1, data.shape[0], data.shape[1]))

    def _get_future_temp_values(self, future_temps: pd.DataFrame, idx: int) -> np.ndarray:
        """
        Extract temperature values from future temperatures dataframe
        
        Args:
            future_temps (pd.DataFrame): DataFrame containing future temperature data
            idx (int): Index to extract values from
        
        Returns:
            np.ndarray: Array of temperature values
        """
        temp_columns = [col for col in self.config.features if 'temp' in col or 'thetao' in col]
        return future_temps.iloc[idx][temp_columns].values.reshape(1, -1).astype(np.float32)

    def _create_new_record(self, prediction: np.ndarray, cluster_label: float, 
                          future_temp_values: np.ndarray) -> np.ndarray:
        """
        Create a new record for the next prediction step
        
        Args:
            prediction (np.ndarray): Current prediction
            cluster_label (float): Cluster label value
            future_temp_values (np.ndarray): Temperature values for the next step
        
        Returns:
            np.ndarray: Combined new record
        """
        return np.hstack((prediction, 
                         np.array([[cluster_label]], dtype=np.float32),
                         future_temp_values))

    def make_predictions(self, group: pd.DataFrame, 
                        future_temps: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions with confidence intervals using bootstrapping
        
        Args:
            group (pd.DataFrame): Input data group
            future_temps (pd.DataFrame): Future temperature data
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Predictions, lower bound, and upper bound
        """
        group_scaled = self.scaler.transform(group[self.config.features])
        
        predictions = self._make_single_prediction_sequence(group_scaled, future_temps)
        
        bootstrap_predictions = [
            self._make_single_prediction_sequence(group_scaled, future_temps)
            for _ in range(self.config.n_bootstrap)
        ]
        
        bootstrap_predictions = np.array(bootstrap_predictions)
        lower_bound = np.percentile(bootstrap_predictions, 100 * self.config.alpha / 2, axis=0)
        upper_bound = np.percentile(bootstrap_predictions, 100 * (1 - self.config.alpha / 2), axis=0)
        
        return predictions, lower_bound, upper_bound

    def _make_single_prediction_sequence(self, group_scaled: np.ndarray, 
                                       future_temps: pd.DataFrame) -> np.ndarray:
        """
        Make a single sequence of predictions
        
        Args:
            group_scaled (np.ndarray): Scaled input data
            future_temps (pd.DataFrame): Future temperature data
        
        Returns:
            np.ndarray: Sequence of predictions
        """
        X_input = self._prepare_input(group_scaled[-self.config.look_back:].astype(np.float32))
        predictions = []
        
        for i in range(len(future_temps)):
            pred = self.model.predict(X_input, verbose=0)
            predictions.append(pred[0][0])
            
            future_temp_values = self._get_future_temp_values(future_temps, i)
            new_record = self._create_new_record(
                pred, future_temps['Cluster_Label'].iloc[0], future_temp_values)
            
            new_record_scaled = self.scaler.transform(new_record).reshape((1, 1, -1))
            X_input = np.append(X_input[:, 1:, :], new_record_scaled, axis=1)
        
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(
            np.hstack((predictions, np.zeros((len(predictions), group_scaled.shape[1] - 1))))
        )[:, 0]
        
        return predictions



# Example usage
if __name__ == "__main__":
    # Configuration
    config = ForecastConfig()
    
    # Paths
    SPECIES = 'BANDERA'
    CLUSTER = 1
    model_path = Path(f'modelos_moe/{SPECIES}_cluster_{CLUSTER}_moe_model.h5')
    scaler_path = Path(f'modelos_moe/{SPECIES}_cluster_{CLUSTER}_moe_scaler.pkl')
    
    # Initialize forecaster
    forecaster = SpeciesForecaster(
        model_path=model_path,
        scaler_path=scaler_path,
        config=config
    )
    
    # Load and filter data
    try:
        data = pd.read_csv('data/data.csv', low_memory=False)
        filtered_data = forecaster.filter_data(data, SPECIES, CLUSTER)
        print(f"Datos filtrados exitosamente. Shape: {filtered_data.shape}")
        
        # Load and prepare future temperatures
        future_temps = pd.read_csv('future_temp.csv')
        future_temps = future_temps[future_temps['Cluster_Label'] == CLUSTER].copy()
        future_temps['date'] = pd.to_datetime(
            future_temps['year'].astype(str) + '-' + future_temps['month'].astype(str)
        )
        future_temps.sort_values(by='date', inplace=True)
        
        # Make predictions
        predictions, lower_bound, upper_bound = forecaster.make_predictions(
            filtered_data, future_temps
        )
        print("Predicciones realizadas exitosamente")
        
    except Exception as e:
        print(f"Error: {str(e)}")