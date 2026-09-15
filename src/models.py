import os
import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# ==========================================
# FUNCIONES AUXILIARES DE EVALUACIÓN
# ==========================================

def evaluate_model(model, X_test, y_test):
    """Calcula las métricas estándar de evaluación para regresión."""
    y_pred = model.predict(X_test)
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE': mean_absolute_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred)
    }
    return metrics


def save_model(model, model_name: str, dataset_name: str, output_folder: str = 'models_saved'):
    """Guarda el modelo entrenado en la carpeta correspondiente."""
    path = os.path.join(output_folder, dataset_name)
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, f"{model_name}.joblib")
    joblib.dump(model, file_path)
    print(f"   [+] Modelo guardado en: {file_path}")


# ==========================================
# DEFINICIÓN DE LOS 6 MODELOS
# ==========================================

# --- DATASET 1: CANDY DATA ---

def train_candy_dt(X_train, y_train, max_depth: int = 5):
    """Modelo 1: Árbol de Decisión para Candy"""
    print("-> Entrenando M1: Árbol de Decisión (Candy)...")
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train.values.ravel())
    return model


def train_candy_svm(X_train, y_train, C: float = 1.0, kernel: str = 'rbf'):
    """Modelo 2: Support Vector Machine (SVR) para Candy"""
    print("-> Entrenando M2: SVM (Candy)...")
    model = SVR(C=C, kernel=kernel)
    model.fit(X_train, y_train.values.ravel())
    return model


def train_candy_mlp(X_train, y_train):
    """Modelo 3: Red Neuronal Perceptrón Multicapa (MLP) para Candy"""
    print("-> Entrenando M3: Perceptrón Multicapa (Candy)...")
    model = MLPRegressor(
        hidden_layer_sizes=(32, 16),
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train, y_train.values.ravel())
    return model


# --- DATASET 2: WINE QUALITY DATA ---

def train_wine_dt(X_train, y_train, max_depth: int = 6):
    """Modelo 4: Árbol de Decisión para Wine Quality"""
    print("-> Entrenando M4: Árbol de Decisión (Wine Quality)...")
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train.values.ravel())
    return model


def train_wine_poly(X_train, y_train, degree: int = 2):
    """Modelo 5: Regresión Polinómica (Grado 2) para Wine Quality"""
    print("-> Entrenando M5: Regresión Polinómica (Grado 2) (Wine Quality)...")
    model = Pipeline([
        ('poly_features', PolynomialFeatures(degree=degree, include_bias=False)),
        ('linear_regression', LinearRegression())
    ])
    model.fit(X_train, y_train.values.ravel())
    return model


def train_wine_mlp(X_train, y_train):
    """Modelo 6: Red Neuronal Perceptrón Multicapa (MLP) para Wine Quality"""
    print("-> Entrenando M6: Perceptrón Multicapa (Wine Quality)...")
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train, y_train.values.ravel())
    return model


# ==========================================
# PIPELINE COMPLETO DE ENTRENAMIENTO
# ==========================================

def run_models_pipeline(processed_folder: str = 'data/processed'):
    """Carga los conjuntos de datos, entrena los 6 modelos y reporta métricas."""
    results = []

    # 1. EJECUCIÓN GRUPO CANDY
    print("\n=================== ENTRENANDO MODELOS: CANDY ===================")
    try:
        X_tr_c = pd.read_csv(os.path.join(processed_folder, 'candy_X_train.csv'))
        X_te_c = pd.read_csv(os.path.join(processed_folder, 'candy_X_test.csv'))
        y_tr_c = pd.read_csv(os.path.join(processed_folder, 'candy_y_train.csv'))
        y_te_c = pd.read_csv(os.path.join(processed_folder, 'candy_y_test.csv'))

        # M1: Árbol
        m1 = train_candy_dt(X_tr_c, y_tr_c)
        met1 = evaluate_model(m1, X_te_c, y_te_c)
        save_model(m1, '01_arbol_decision', 'candy')
        results.append({'Dataset': 'Candy', 'Modelo': 'M1: Árbol de Decisión', **met1})

        # M2: SVM
        m2 = train_candy_svm(X_tr_c, y_tr_c)
        met2 = evaluate_model(m2, X_te_c, y_te_c)
        save_model(m2, '02_svm', 'candy')
        results.append({'Dataset': 'Candy', 'Modelo': 'M2: SVM (SVR)', **met2})

        # M3: MLP
        m3 = train_candy_mlp(X_tr_c, y_tr_c)
        met3 = evaluate_model(m3, X_te_c, y_te_c)
        save_model(m3, '03_mlp_red_neuronal', 'candy')
        results.append({'Dataset': 'Candy', 'Modelo': 'M3: Perceptrón Multicapa', **met3})

    except FileNotFoundError:
        print("   [!] Error: Archivos procesados de Candy no encontrados. Corre primero data_split.py.")

    # 2. EJECUCIÓN GRUPO WINE QUALITY
    print("\n================ ENTRENANDO MODELOS: WINE QUALITY ================")
    try:
        X_tr_w = pd.read_csv(os.path.join(processed_folder, 'wine_X_train.csv'))
        X_te_w = pd.read_csv(os.path.join(processed_folder, 'wine_X_test.csv'))
        y_tr_w = pd.read_csv(os.path.join(processed_folder, 'wine_y_train.csv'))
        y_te_w = pd.read_csv(os.path.join(processed_folder, 'wine_y_test.csv'))

        # M4: Árbol
        m4 = train_wine_dt(X_tr_w, y_tr_w)
        met4 = evaluate_model(m4, X_te_w, y_te_w)
        save_model(m4, '04_arbol_decision', 'redwine')
        results.append({'Dataset': 'Wine', 'Modelo': 'M4: Árbol de Decisión', **met4})

        # M5: Regresión Polinómica
        m5 = train_wine_poly(X_tr_w, y_tr_w, degree=2)
        met5 = evaluate_model(m5, X_te_w, y_te_w)
        save_model(m5, '05_regresion_polinomica', 'redwine')
        results.append({'Dataset': 'Wine', 'Modelo': 'M5: Regresión Polinómica (G2)', **met5})

        # M6: MLP
        m6 = train_wine_mlp(X_tr_w, y_tr_w)
        met6 = evaluate_model(m6, X_te_w, y_te_w)
        save_model(m6, '06_mlp_red_neuronal', 'redwine')
        results.append({'Dataset': 'Wine', 'Modelo': 'M6: Perceptrón Multicapa', **met6})

    except FileNotFoundError:
        print("   [!] Error: Archivos procesados de Red Wine no encontrados. Corre primero data_split.py.")

    # 3. MOSTRAR TABLA FINAL
    if results:
        df_results = pd.DataFrame(results)
        print("\n================ RESUMEN DE MÉTRICAS (6 MODELOS) ================")
        print(df_results.to_string(index=False))
        
        # Guardar en carpeta reports
        os.makedirs('reports', exist_ok=True)
        df_results.to_csv('reports/metrics_summary.csv', index=False)
        print("\n[OK] Resumen de métricas guardado en reports/metrics_summary.csv")


if __name__ == "__main__":
    run_models_pipeline()