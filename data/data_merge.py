import os
import pandas as pd

# Especifica la carpeta que contiene los archivos
folder_path = 'Data_GLORY'

# Lista para almacenar los DataFrames
dataframes = []

# Listar todos los archivos en la carpeta
for file_name in os.listdir(folder_path):
    # Crear la ruta completa del archivo
    file_path = os.path.join(folder_path, file_name)
    # Verificar si el archivo es un archivo CSV
    if file_name.endswith('.csv'):
        # Leer el archivo CSV y agregarlo a la lista de DataFrames
        df = pd.read_csv(file_path)
        dataframes.append(df)

# Concatenar todos los DataFrames en uno solo
merged_df = pd.concat(dataframes, ignore_index=True)

# Guardar el DataFrame resultante en un archivo CSV llamado data.csv
merged_df.to_csv('data.csv', index=False)

print("Archivos combinados y guardados en 'data.csv'.")
