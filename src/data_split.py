import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def split_and_scale(df: pd.DataFrame, target_column: str, test_size: float = 0.2, random_state: int = 42):
    """
    Separa X e y, realiza la división Train/Test y escala X con StandardScaler.
    """
    if target_column not in df.columns:
        raise KeyError(f"La columna objetivo '{target_column}' no existe en el DataFrame. Columnas disponibles: {list(df.columns)}")

    # 1. Separar X e y
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # 2. División Train/Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # 3. Escalado de datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convertir a DataFrames conservando nombres
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    return X_train_scaled_df, X_test_scaled_df, y_train.reset_index(drop=True), y_test.reset_index(drop=True)


def run_split_pipeline(processed_folder: str = 'data/processed'):
    """
    Ejecuta la partición y escalado para Candy y Wine Quality.
    """
    # 1. Candy Data
    candy_path = os.path.join(processed_folder, 'candy_clean.csv')
    if os.path.exists(candy_path):
        df_candy = pd.read_csv(candy_path)
        candy_target = 'winpercent' if 'winpercent' in df_candy.columns else df_candy.columns[-1]
        
        X_tr_c, X_te_c, y_tr_c, y_te_c = split_and_scale(df_candy, target_column=candy_target)
        
        X_tr_c.to_csv(os.path.join(processed_folder, 'candy_X_train.csv'), index=False)
        X_te_c.to_csv(os.path.join(processed_folder, 'candy_X_test.csv'), index=False)
        y_tr_c.to_csv(os.path.join(processed_folder, 'candy_y_train.csv'), index=False)
        y_te_c.to_csv(os.path.join(processed_folder, 'candy_y_test.csv'), index=False)
        print(f"   [OK] Particiones de Candy creadas exitosamente.")
    else:
        print(f"   [!] ADVERTENCIA: No se encontró {candy_path}.")

    # 2. Wine Quality Data
    wine_path = os.path.join(processed_folder, 'winequality_clean.csv')
    if os.path.exists(wine_path):
        df_wine = pd.read_csv(wine_path)
        wine_target = 'quality'
        
        X_tr_w, X_te_w, y_tr_w, y_te_w = split_and_scale(df_wine, target_column=wine_target)
        
        X_tr_w.to_csv(os.path.join(processed_folder, 'wine_X_train.csv'), index=False)
        X_te_w.to_csv(os.path.join(processed_folder, 'wine_X_test.csv'), index=False)
        y_tr_w.to_csv(os.path.join(processed_folder, 'wine_y_train.csv'), index=False)
        y_te_w.to_csv(os.path.join(processed_folder, 'wine_y_test.csv'), index=False)
        print(f"   [OK] Particiones de Wine Quality creadas exitosamente.")
    else:
        print(f"   [!] ADVERTENCIA: No se encontró {wine_path}.")


if __name__ == "__main__":
    run_split_pipeline()