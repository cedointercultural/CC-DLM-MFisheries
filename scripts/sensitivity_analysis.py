import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List
from scripts.species_forecaster import ForecastConfig, SpeciesForecaster

def get_models_info(models_dir: Path) -> pd.DataFrame:
    """
    Extract species and cluster information from model files
    
    Args:
        models_dir (Path): Directory containing the model files
        
    Returns:
        pd.DataFrame: DataFrame with species and cluster information
    """
    models_info = []
    pattern = r'(.+)_cluster_(\d+)_moe_model\.h5'
    
    for file in os.listdir(models_dir):
        if file.endswith('_moe_model.h5'):
            match = re.match(pattern, file)
            if match:
                species, cluster = match.groups()
                models_info.append({
                    'species': species,
                    'cluster': int(cluster),
                    'model_file': file
                })
    
    return pd.DataFrame(models_info)

def run_batch_analysis(models_dir: Path = Path('modelos_moe'),
                      save_results: bool = True):
    """
    Run sensitivity analysis for all models
    
    Args:
        models_dir (Path): Directory containing the model files
        save_results (bool): Whether to save results
    """
    # Get models information
    models_df = get_models_info(models_dir)
    print(f"Found {len(models_df)} models to analyze")
    
    results_all = []
    
    # Run analysis for each model
    for _, row in models_df.iterrows():
        species = row['species']
        cluster = row['cluster']
        
        print(f"\nAnalyzing {species} - Cluster {cluster}")
        try:
            # Run sensitivity analysis
            results = run_analysis(
                species=species,
                cluster=cluster,
                base_dir=Path('.'),
                save_results=save_results
            )
            
            # Add species and cluster information to results
            results['species'] = species
            results['cluster'] = cluster
            results_all.append(results)
            
            print(f"Analysis completed for {species} - Cluster {cluster}")
            
        except Exception as e:
            print(f"Error analyzing {species} - Cluster {cluster}: {str(e)}")
    
    # Combine all results
    if results_all:
        all_results = pd.concat(results_all, ignore_index=True)
        
        if save_results:
            results_dir = Path('sensitivity_analysis')
            results_dir.mkdir(exist_ok=True)
            all_results.to_csv(results_dir / 'sensitivity_analysis_all.csv', index=False)
            
        return all_results
    else:
        print("No results generated")
        return None

# Ejemplo de uso
if __name__ == "__main__":
    results = run_batch_analysis()
    if results is not None:
        print("\nAnálisis completo para todos los modelos")
        print(f"Total de análisis realizados: {len(results['species'].unique())}")


class SensitivityAnalyzer:
    def __init__(self, base_temp: float = 15, max_temp: float = 32, step: float = 1):
        self.base_temp = base_temp
        self.max_temp = max_temp
        self.step = step
        self.temp_range = np.arange(base_temp, max_temp + step, step)
    
    def create_base_scenario(self, cluster: int, n_months: int = 12) -> pd.DataFrame:
        """
        Create base scenario template for temperature analysis
        """
        # Crear fechas para los próximos 12 meses
        dates = pd.date_range(start='2024-01-01', periods=n_months, freq='M')
        
        # Crear DataFrame base
        scenario = pd.DataFrame({
            'date': dates,
            'year': dates.year,
            'month': dates.month,
            'Cluster_Label': cluster
        })
        
        # Añadir columnas de temperatura con valores temporales
        temp_columns = [
            'mean_temp_30m', 'mean_temp_10m',
            'thetao_sfc=6', 'thetao_sfc=7.92956018447876', 
            'thetao_sfc=9.572997093200684', 'thetao_sfc=11.40499973297119',
            'thetao_sfc=13.46714019775391', 'thetao_sfc=15.8100700378418', 
            'thetao_sfc=18.49555969238281', 'thetao_sfc=21.59881973266602',
            'thetao_sfc=25.21141052246094', 'thetao_sfc=29.44473075866699'
        ]
        
        for col in temp_columns:
            scenario[col] = 0  # Valor temporal que será reemplazado
            
        return scenario
    
    def create_temperature_scenarios(self, cluster: int) -> List[pd.DataFrame]:
        """
        Create different temperature scenarios for sensitivity analysis
        """
        base_scenario = self.create_base_scenario(cluster)
        scenarios = []
        
        temp_columns = [col for col in base_scenario.columns 
                       if 'temp' in col or 'thetao' in col]
        
        for temp in self.temp_range:
            scenario = base_scenario.copy()
            for col in temp_columns:
                scenario[col] = temp
            scenario['scenario_temp'] = temp
            scenarios.append(scenario)
            
        return scenarios
    
    def analyze_sensitivity(self, 
                          forecaster: SpeciesForecaster,
                          data: pd.DataFrame,
                          species: str,
                          cluster: int) -> pd.DataFrame:
        """
        Perform sensitivity analysis
        """
        filtered_data = forecaster.filter_data(data, species, cluster)
        scenarios = self.create_temperature_scenarios(cluster)
        
        results = []
        for scenario in scenarios:
            predictions, lower_bound, upper_bound = forecaster.make_predictions(
                filtered_data, scenario
            )
            
            results.append({
                'temperature': scenario['scenario_temp'].iloc[0],
                'mean_prediction': np.mean(predictions),
                'max_prediction': np.max(predictions),
                'min_prediction': np.min(predictions),
                'std_prediction': np.std(predictions)
            })
        
        return pd.DataFrame(results)
    
    def plot_sensitivity(self, 
                        results: pd.DataFrame,
                        species: str,
                        cluster: int,
                        save_path: Path = None):
        """
        Plot sensitivity analysis results
        """
        plt.figure(figsize=(12, 6))
        
        plt.plot(results['temperature'], results['mean_prediction'], 
                'b-', label='Predicción Media')
        plt.fill_between(results['temperature'],
                        results['mean_prediction'] - results['std_prediction'],
                        results['mean_prediction'] + results['std_prediction'],
                        alpha=0.2, color='b', label='Desviación Estándar')
        
        plt.plot(results['temperature'], results['max_prediction'], 
                'r--', label='Máximo', alpha=0.5)
        plt.plot(results['temperature'], results['min_prediction'], 
                'g--', label='Mínimo', alpha=0.5)
        
        plt.title(f'Análisis de Sensibilidad a la Temperatura\n{species} (Cluster {cluster})')
        plt.xlabel('Temperatura (°C)')
        plt.ylabel('Landed_w_kg Predicho')
        plt.grid(True)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()

def run_analysis(species: str, 
                cluster: int,
                base_dir: Path = Path('.'),
                save_results: bool = True):
    """
    Run complete sensitivity analysis for a species and cluster
    """
    # Initialize objects
    config = ForecastConfig()
    forecaster = SpeciesForecaster(
        model_path=base_dir / f'modelos_moe/{species}_cluster_{cluster}_moe_model.h5',
        scaler_path=base_dir / f'modelos_moe/{species}_cluster_{cluster}_moe_scaler.pkl',
        config=config
    )
    analyzer = SensitivityAnalyzer(base_temp=15, max_temp=32, step=1)
    
    # Load data
    data = pd.read_csv(base_dir / 'data/data.csv', low_memory=False)
    
    # Run analysis
    results = analyzer.analyze_sensitivity(
        forecaster, data, species, cluster
    )
    
    # Save results if requested
    if save_results:
        results_dir = base_dir / 'sensitivity_analysis'
        results_dir.mkdir(exist_ok=True)
        
        results.to_csv(
            results_dir / f'sensitivity_{species}_cluster_{cluster}.csv',
            index=False
        )
        
        analyzer.plot_sensitivity(
            results,
            species,
            cluster,
            save_path=results_dir / f'sensitivity_{species}_cluster_{cluster}.png'
        )
    else:
        analyzer.plot_sensitivity(results, species, cluster)
    
    return results

if __name__ == "__main__":
    species = 'BANDERA'
    cluster = 1
    results = run_analysis(species, cluster)
    print("Análisis completado.")