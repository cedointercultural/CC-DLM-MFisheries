import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Definir la lista de especies y clústeres
species_cluster_data = {
    'species': [ 'CAMARON', 'RAYA Y SIMILARES', 'LOBINA', 'BAQUETA', 'MOJARRA', 'LOBINA', 'JAIBA', 'TIBURON', 'CAMARON',
                'CAMARON', 'SARDINA', 'JAIBA', 'SIERRA', 'JAIBA', 'CAMARON', 'BERRUGATA', 'PARGO', 'MOJARRA', 'RAYA Y SIMILARES','GUACHINANGO','GUACHINANGO'],
    'cluster': [ 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 6, 6, 6, 7, 7, 7,1,3]
}

species_cluster_df = pd.DataFrame(species_cluster_data)

def make_predictions_with_confidence_intervals(model, scaler, group, future_temps, look_back=6, n_bootstrap=1, alpha=0.01, n_forecasts_per_month=1):
    features = ['landed_w_kg','Cluster_Label', 'mean_temp_30m','mean_temp_10m','thetao_sfc=6',
                'thetao_sfc=7.92956018447876','thetao_sfc=9.572997093200684','thetao_sfc=11.40499973297119',
                'thetao_sfc=13.46714019775391','thetao_sfc=15.8100700378418','thetao_sfc=18.49555969238281',
                'thetao_sfc=21.59881973266602','thetao_sfc=25.21141052246094','thetao_sfc=29.44473075866699']
    
    group_scaled = scaler.transform(group[features])
    
    # Preparar X_input manteniendo la forma correcta (1, look_back, num_features)
    X_input = group_scaled[-look_back:].astype(np.float32)
    X_input = np.reshape(X_input, (1, X_input.shape[0], X_input.shape[1]))

    predictions = []
    for i in range(len(future_temps)):
        monthly_predictions = []
        for _ in range(n_forecasts_per_month):
            pred = model.predict(X_input)
            monthly_predictions.append(pred[0][0])
            
            future_temp_values = future_temps.iloc[i][['mean_temp_30m', 'mean_temp_10m', 'thetao_sfc=6', 
                                                       'thetao_sfc=7.92956018447876', 'thetao_sfc=9.572997093200684',
                                                       'thetao_sfc=11.40499973297119', 'thetao_sfc=13.46714019775391',
                                                       'thetao_sfc=15.8100700378418', 'thetao_sfc=18.49555969238281',
                                                       'thetao_sfc=21.59881973266602', 'thetao_sfc=25.21141052246094',
                                                       'thetao_sfc=29.44473075866699']].values.reshape(1, -1).astype(np.float32)
            
            new_record = np.hstack((pred, np.array([[group['Cluster_Label'].iloc[0]]], dtype=np.float32), future_temp_values))
            new_record_df = pd.DataFrame(new_record, columns=features)
            new_record_scaled = scaler.transform(new_record_df)  # Escalar el nuevo registro
            
            # Mantener la forma correcta al agregar el nuevo registro
            new_record_scaled = np.reshape(new_record_scaled, (1, 1, -1))
            X_input = np.append(X_input[:, 1:, :], new_record_scaled, axis=1)

        predictions.extend(monthly_predictions)
    
    predictions = np.array(predictions).reshape(-1, 1)
    # Crear un array de ceros con las mismas dimensiones que el grupo original escalado
    predictions_full = np.hstack((predictions, np.zeros((len(predictions), group_scaled.shape[1] - 1))))
    
    # Convertirlo a DataFrame con los nombres de columnas correctos antes de desescalar
    predictions_full_df = pd.DataFrame(predictions_full, columns=features)
    predictions_descaled = scaler.inverse_transform(predictions_full_df)[:, 0]
    
    # Bootstrap para bandas de confianza
    bootstrap_predictions = []
    for _ in range(n_bootstrap):
        X_input_bootstrap = group_scaled[-look_back:].astype(np.float32)
        X_input_bootstrap = np.reshape(X_input_bootstrap, (1, X_input_bootstrap.shape[0], X_input_bootstrap.shape[1]))

        bootstrap_pred = []
        for i in range(len(future_temps)):
            monthly_bootstrap_predictions = []
            for _ in range(n_forecasts_per_month):
                pred_boot = model.predict(X_input_bootstrap)
                monthly_bootstrap_predictions.append(pred_boot[0][0])
                
                future_temp_values = future_temps.iloc[i][['mean_temp_30m', 'mean_temp_10m', 'thetao_sfc=6', 
                                                           'thetao_sfc=7.92956018447876', 'thetao_sfc=9.572997093200684',
                                                           'thetao_sfc=11.40499973297119', 'thetao_sfc=13.46714019775391',
                                                           'thetao_sfc=15.8100700378418', 'thetao_sfc=18.49555969238281',
                                                           'thetao_sfc=21.59881973266602', 'thetao_sfc=25.21141052246094',
                                                           'thetao_sfc=29.44473075866699']].values.reshape(1, -1).astype(np.float32)
                
                new_record_boot = np.hstack((pred_boot, np.array([[group['Cluster_Label'].iloc[0]]], dtype=np.float32), future_temp_values))
                new_record_boot_df = pd.DataFrame(new_record_boot, columns=features)
                new_record_boot_scaled = scaler.transform(new_record_boot_df)  # Escalar el nuevo registro
                
                new_record_boot_scaled = np.reshape(new_record_boot_scaled, (1, 1, -1))
                X_input_bootstrap = np.append(X_input_bootstrap[:, 1:, :], new_record_boot_scaled, axis=1)

            bootstrap_pred.extend(monthly_bootstrap_predictions)
        
        bootstrap_pred = np.array(bootstrap_pred).reshape(-1, 1)
        bootstrap_pred_full = np.hstack((bootstrap_pred, np.zeros((len(bootstrap_pred), group_scaled.shape[1] - 1))))
        bootstrap_pred_full_df = pd.DataFrame(bootstrap_pred_full, columns=features)
        bootstrap_pred_descaled = scaler.inverse_transform(bootstrap_pred_full_df)[:, 0]
        bootstrap_predictions.append(bootstrap_pred_descaled)
    
    bootstrap_predictions = np.array(bootstrap_predictions)
    lower_bound = np.percentile(bootstrap_predictions, 100 * alpha / 2, axis=0)
    upper_bound = np.percentile(bootstrap_predictions, 100 * (1 - alpha / 2), axis=0)
    
    return predictions_descaled, lower_bound, upper_bound

def process_species_cluster(specie, cluster_label, data, future_temps, results_dir='Resultados',
                            look_back=6, n_bootstrap=1, alpha=0.01, n_forecasts_per_month=1):
    print(f"Procesando especie: {specie}, clúster: {cluster_label}")
    
    # Definir rutas
    model_path = f'modelos_moe/{specie}_cluster_{cluster_label}_moe_model.h5'
    scaler_path = f'modelos_moe/{specie}_cluster_{cluster_label}_moe_scaler.pkl'
    
    # Comprobar existencia de archivos
    if not os.path.exists(model_path):
        print(f"Modelo no encontrado para {specie} en clúster {cluster_label}. Saltando...")
        return
    if not os.path.exists(scaler_path):
        print(f"Scaler no encontrado para {specie} en clúster {cluster_label}. Saltando...")
        return
    
    # Cargar modelo y scaler
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    
    # Filtrar datos para la especie y el clúster
    specie_data = data[(data['species'] == specie) & (data['Cluster_Label'] == cluster_label)]
    if specie_data.empty:
        print(f"No hay datos históricos para {specie} en clúster {cluster_label}. Saltando...")
        return
    
    # Filtrar future_temps para el clúster
    future_temps_cluster = future_temps[future_temps['Cluster_Label'] == cluster_label]
    if future_temps_cluster.empty:
        print(f"No hay datos de temperatura futuros para clúster {cluster_label}. Saltando...")
        return
    
    # Realizar predicciones
    predictions, lower_bound, upper_bound = make_predictions_with_confidence_intervals(
        model, scaler, specie_data, future_temps_cluster, look_back, n_bootstrap, alpha, n_forecasts_per_month
    )
    
    # Preparar datos para guardar
    expanded_dates = future_temps_cluster['date'].repeat(n_forecasts_per_month).reset_index(drop=True)
    results_df = pd.DataFrame({
        'date': expanded_dates,
        'predictions': predictions,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    })
    
    # Crear directorio si no existe
    os.makedirs(results_dir, exist_ok=True)
    
    # Guardar CSV
    csv_path = os.path.join(results_dir, f'predicciones_{specie}_cluster_{cluster_label}.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"Resultados guardados en {csv_path}")

def generate_plots(specie, cluster_label, data, results_dir='Resultados', output_dir='Graficas'):
    print(f"Generando gráficos para especie: {specie}, clúster: {cluster_label}")
    
    # Definir rutas
    prediction_file = os.path.join(results_dir, f'predicciones_{specie}_cluster_{cluster_label}.csv')
    
    # Comprobar existencia de archivo de predicciones
    if not os.path.exists(prediction_file):
        print(f"Archivo de predicciones no encontrado para {specie} en clúster {cluster_label}. Saltando...")
        return
    
    # Cargar datos
    predictions = pd.read_csv(prediction_file)
    predictions['date'] = pd.to_datetime(predictions['date'])
    predictions['year'] = predictions['date'].dt.year
    predictions['type'] = 'Pronóstico'
    
    # Datos históricos
    historical_data = data[(data['species'] == specie) & (data['Cluster_Label'] == cluster_label)].copy()
    if historical_data.empty:
        print(f"No hay datos históricos para {specie} en clúster {cluster_label}. Saltando gráfico...")
        return
    historical_data['date'] = pd.to_datetime(historical_data['date'])
    historical_data['year'] = historical_data['date'].dt.year
    historical_data['type'] = 'Histórico'
    historical_data.rename(columns={'landed_w_kg': 'value'}, inplace=True)
    
    # Preparar datos combinados
    predictions.rename(columns={'predictions': 'value'}, inplace=True)
    combined_data = pd.concat([historical_data[['year', 'value', 'type']], predictions[['year', 'value', 'type']]], ignore_index=True)
    
    # Crear gráfico
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='year', y='value', hue='type', data=combined_data, palette={'Histórico': 'lightblue', 'Pronóstico': 'lightgreen'},showfliers=False)
    plt.title(f'Boxplot of Predictions and Historical Data for  {specie} - Cluster {cluster_label}')
    plt.xlabel('Year')
    plt.ylabel('Landed Weight (kg)')
    plt.legend(title='Type')
    plt.tight_layout()
    plt.xticks(rotation=45, ha='right')
    
    # Crear directorio si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar gráfico
    plot_path = os.path.join(output_dir, f'boxplot_{specie}_cluster_{cluster_label}.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Gráfico guardado en {plot_path}")

def main():
    # Cargar datos generales
    data = pd.read_csv('data/data.csv', low_memory=False)
    future_temps = pd.read_csv('future_temp.csv')
    future_temps['date'] = pd.to_datetime(future_temps['year'].astype(str) + '-' + future_temps['month'].astype(str))
    future_temps = future_temps.sort_values(by='date')
    
    # Iterar sobre cada especie y clúster
    for _, row in species_cluster_df.iterrows():
        specie = row['species']
        cluster_label = row['cluster']
        print(specie,cluster_label)
        
        # Procesar predicciones
        process_species_cluster(specie, cluster_label, data, future_temps)
        
        # Generar gráficos
        generate_plots(specie, cluster_label, data)
        
if __name__ == '__main__':
    main()
