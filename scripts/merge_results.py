import os
import pandas as pd

# Ruta de la carpeta que contiene los archivos CSV
input_folder = r'C:\Users\ricar\OneDrive\Documentos\Proyectos\CC-DLM-MFisheries\Resultados'
output_file = r'C:\Users\ricar\OneDrive\Documentos\Proyectos\CC-DLM-MFisheries\resultado_unificado.csv'

# Crear una lista para almacenar los dataframes
dataframes = []

# Iterar sobre los archivos en la carpeta
for file_name in os.listdir(input_folder):
    if file_name.endswith('.csv'):
        # Extraer especie y cluster del nombre del archivo
        parts = file_name.split('_')
        if len(parts) >= 3:
            especie = parts[1]
            cluster = parts[3]
        else:
            print(f'Nombre de archivo no válido: {file_name}')
            continue

        # Leer el archivo CSV
        file_path = os.path.join(input_folder, file_name)
        df = pd.read_csv(file_path)

        # Agregar columnas de especie y cluster
        df['Especie'] = especie
        df['Cluster'] = cluster

        # Añadir el dataframe a la lista
        dataframes.append(df)

# Concatenar todos los dataframes
resultado_df = pd.concat(dataframes, ignore_index=True)

# Guardar el dataframe unificado en un archivo CSV
resultado_df.to_csv(output_file, index=False)

print(f'Archivo unificado guardado en: {output_file}')
