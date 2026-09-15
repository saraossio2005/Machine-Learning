import os
import pandas as pd
import numpy as np

def clean_candy_data(input_path: str) -> pd.DataFrame:
    """
    Carga y limpia el dataset de Candy.
    """
    print(f"-> Procesando Candy Data desde: {input_path}")
    df = pd.read_csv(input_path)
    
    # 1. Eliminar duplicados
    df = df.drop_duplicates()
    
    # 2. Eliminar columna de nombres si existe
    if 'competitorname' in df.columns:
        df = df.drop(columns=['competitorname'])
        
    # 3. Imputación de valores nulos
    if df.isnull().sum().sum() > 0:
        df = df.fillna(df.median(numeric_only=True))
        
    # 4. Normalizar nombres de columnas
    df.columns = df.columns.str.strip().str.lower()
    
    return df


def clean_wine_data(input_path: str) -> pd.DataFrame:
    """
    Carga y limpia el dataset de Wine Quality.
    Intenta leer con coma (',') y si detecta 1 sola columna intenta con punto y coma (';').
    """
    print(f"-> Procesando Wine Quality Data desde: {input_path}")
    
    # Intenta leer primero por coma
    df = pd.read_csv(input_path, sep=',')
    
    # Si detecta solo 1 columna, reintenta usando punto y coma
    if df.shape[1] == 1:
        df = pd.read_csv(input_path, sep=';')
    
    # 1. Eliminar duplicados
    df = df.drop_duplicates()
    
    # 2. Imputación de valores nulos
    if df.isnull().sum().sum() > 0:
        df = df.fillna(df.median(numeric_only=True))
        
    # 3. Normalizar nombres de columnas (espacios por guiones bajos)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    return df


def run_cleaning_pipeline(raw_folder: str = 'data/raw', processed_folder: str = 'data/processed'):
    """
    Ejecuta el pipeline de limpieza y guarda los archivos limpios en data/processed/.
    """
    os.makedirs(processed_folder, exist_ok=True)
    
    # 1. Limpieza de Candy
    candy_raw_path = os.path.join(raw_folder, 'candy-data.csv')
    if os.path.exists(candy_raw_path):
        df_candy_clean = clean_candy_data(candy_raw_path)
        candy_out_path = os.path.join(processed_folder, 'candy_clean.csv')
        df_candy_clean.to_csv(candy_out_path, index=False)
        print(f"   [OK] Candy guardado en: {candy_out_path} (Filas: {len(df_candy_clean)}, Cols: {len(df_candy_clean.columns)})\n")
    else:
        print(f"   [!] ADVERTENCIA: No se encontró {candy_raw_path}\n")

    # 2. Limpieza de Wine Quality
    wine_raw_path = os.path.join(raw_folder, 'winequality-red.csv')
    if os.path.exists(wine_raw_path):
        df_wine_clean = clean_wine_data(wine_raw_path)
        wine_out_path = os.path.join(processed_folder, 'winequality_clean.csv')
        df_wine_clean.to_csv(wine_out_path, index=False)
        print(f"   [OK] Wine Quality guardado en: {wine_out_path} (Filas: {len(df_wine_clean)}, Cols: {len(df_wine_clean.columns)})\n")
    else:
        print(f"   [!] ADVERTENCIA: No se encontró {wine_raw_path}\n")


if __name__ == "__main__":
    run_cleaning_pipeline()