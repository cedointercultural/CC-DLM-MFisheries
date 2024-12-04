from scripts.species_forecaster import ForecastConfig, SpeciesForecaster

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