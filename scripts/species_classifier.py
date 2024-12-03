import pandas as pd

def classify_species(file_path, habitat_groups, output_path='classified_species.csv', species_column='species'):
    """
    Clasifica las especies en un archivo según el diccionario de hábitats proporcionado.

    Parameters:
    - file_path (str): Ruta del archivo CSV con las especies.
    - habitat_groups (dict): Diccionario que mapea especies a sus hábitats.
    - output_path (str): Ruta donde se guardará el archivo clasificado (por defecto: 'classified_species.csv').
    - species_column (str): Nombre de la columna que contiene las especies (por defecto: 'species').

    Returns:
    - pd.DataFrame: DataFrame con las especies clasificadas y sus hábitats.
    """
    try:
        # Cargar los datos del archivo
        data = pd.read_csv(file_path)

        # Verificar si la columna de especies existe
        if species_column not in data.columns:
            raise ValueError(f"La columna '{species_column}' no existe en el archivo.")

        # Crear una nueva columna 'habitat' clasificando las especies
        data['habitat'] = data[species_column].map(habitat_groups)

        # Guardar el DataFrame clasificado en el archivo de salida
        data.to_csv(output_path, index=False)

        print(f"Archivo clasificado guardado en: {output_path}")
        return data

    except Exception as e:
        print(f"Error al clasificar las especies: {e}")
        return None
