"""
Módulo de definición, entrenamiento, evaluación y persistencia de los 6 modelos.
Cumple con la Fase B (entrenamiento) y Fase C (evaluación, guardado y reporte).
"""

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
# FUNCIONES AUXILIARES
# ==========================================

def evaluate_model(model, X_test, y_test, dataset: str):
    """
    Calcula las métricas según lo que pide la Fase C:
    - Candy (M1, M2, M3): R2 y RMSE
    - Red Wine (M4, M5, M6): R2, MAE y RMSE
    """
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    if dataset == "Candy":
        return {"R2": round(r2, 4), "RMSE": round(rmse, 4)}
    else:  # Red Wine
        return {"R2": round(r2, 4), "MAE": round(mae, 4), "RMSE": round(rmse, 4)}


def save_model(model, model_name: str, dataset_name: str, output_folder: str = "models_saved"):
    """Guarda el modelo entrenado como .joblib en la subcarpeta correspondiente."""
    path = os.path.join(output_folder, dataset_name)
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, f"{model_name}.joblib")
    joblib.dump(model, file_path)
    print(f"   [+] Modelo guardado en: {file_path}")


# ==========================================
# DEFINICIÓN DE LOS 6 MODELOS
# ==========================================

# --- DATASET 1: CANDY ---

def train_candy_dt(X_train, y_train, max_depth: int = 5):
    """M1: Árbol de Decisión para Candy."""
    print("-> Entrenando M1: Árbol de Decisión (Candy)...")
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train.values.ravel())
    return model


def train_candy_svm(X_train, y_train, C: float = 1.0, kernel: str = "rbf"):
    """M2: SVM (SVR) para Candy."""
    print("-> Entrenando M2: SVM (Candy)...")
    model = SVR(C=C, kernel=kernel)
    model.fit(X_train, y_train.values.ravel())
    return model


def train_candy_mlp(X_train, y_train):
    """M3: Red Neuronal MLP para Candy."""
    print("-> Entrenando M3: Perceptrón Multicapa (Candy)...")
    model = MLPRegressor(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=42,
    )
    model.fit(X_train, y_train.values.ravel())
    return model


# --- DATASET 2: WINE QUALITY ---

def train_wine_dt(X_train, y_train, max_depth: int = 5):
    """M4: Árbol de Decisión para Wine Quality."""
    print("-> Entrenando M4: Árbol de Decisión (Wine Quality)...")
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train.values.ravel())
    return model


def train_wine_poly(X_train, y_train, degree: int = 2):
    """
    M5: Regresión Polinómica (Grado 2) para Wine Quality.
    Pipeline: PolynomialFeatures(degree=2) + LinearRegression()
    """
    print("-> Entrenando M5: Regresión Polinómica (Grado 2) (Wine Quality)...")
    model = Pipeline([
        ("poly_features", PolynomialFeatures(degree=degree, include_bias=False)),
        ("linear_regression", LinearRegression()),
    ])
    model.fit(X_train, y_train.values.ravel())
    return model


def train_wine_mlp(X_train, y_train):
    """
    M6: Red Neuronal MLP más profunda para Wine Quality.
    Parámetros exactos del documento: (64, 32), relu, adam, max_iter=1000.
    """
    print("-> Entrenando M6: Perceptrón Multicapa (Wine Quality)...")
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=1000,
        random_state=42,
    )
    model.fit(X_train, y_train.values.ravel())
    return model


# ==========================================
# PIPELINE COMPLETO: FASE B + FASE C
# ==========================================

def run_models_pipeline(processed_folder: str = "data/processed"):
    """
    Entrena los 6 modelos, calcula métricas, guarda los .joblib
    y genera los reportes tabulares separados por dataset.
    """
    results = []

    # ---------------- GRUPO CANDY (M1, M2, M3) ----------------
    print("\n=================== ENTRENANDO MODELOS: CANDY ===================")
    try:
        X_tr_c = pd.read_csv(os.path.join(processed_folder, "candy_X_train.csv"))
        X_te_c = pd.read_csv(os.path.join(processed_folder, "candy_X_test.csv"))
        y_tr_c = pd.read_csv(os.path.join(processed_folder, "candy_y_train.csv"))
        y_te_c = pd.read_csv(os.path.join(processed_folder, "candy_y_test.csv"))

        # M1
        m1 = train_candy_dt(X_tr_c, y_tr_c)
        met1 = evaluate_model(m1, X_te_c, y_te_c, dataset="Candy")
        save_model(m1, "01_arbol_decision", "candy")
        results.append({"Dataset": "Candy", "Modelo": "M1: Árbol de Decisión", **met1})

        # M2
        m2 = train_candy_svm(X_tr_c, y_tr_c)
        met2 = evaluate_model(m2, X_te_c, y_te_c, dataset="Candy")
        save_model(m2, "02_svm", "candy")
        results.append({"Dataset": "Candy", "Modelo": "M2: SVM (SVR)", **met2})

        # M3
        m3 = train_candy_mlp(X_tr_c, y_tr_c)
        met3 = evaluate_model(m3, X_te_c, y_te_c, dataset="Candy")
        save_model(m3, "03_mlp_red_neuronal", "candy")
        results.append({"Dataset": "Candy", "Modelo": "M3: Perceptrón Multicapa", **met3})

    except FileNotFoundError:
        print("   [!] Error: Archivos procesados de Candy no encontrados.")

    # ---------------- GRUPO WINE (M4, M5, M6) ----------------
    print("\n================ ENTRENANDO MODELOS: WINE QUALITY ================")
    try:
        X_tr_w = pd.read_csv(os.path.join(processed_folder, "wine_X_train.csv"))
        X_te_w = pd.read_csv(os.path.join(processed_folder, "wine_X_test.csv"))
        y_tr_w = pd.read_csv(os.path.join(processed_folder, "wine_y_train.csv"))
        y_te_w = pd.read_csv(os.path.join(processed_folder, "wine_y_test.csv"))

        # M4
        m4 = train_wine_dt(X_tr_w, y_tr_w)
        met4 = evaluate_model(m4, X_te_w, y_te_w, dataset="Wine")
        save_model(m4, "04_arbol_decision", "redwine")
        results.append({"Dataset": "Wine", "Modelo": "M4: Árbol de Decisión", **met4})

        # M5
        m5 = train_wine_poly(X_tr_w, y_tr_w, degree=2)
        met5 = evaluate_model(m5, X_te_w, y_te_w, dataset="Wine")
        save_model(m5, "05_regresion_polinomica", "redwine")
        results.append({"Dataset": "Wine", "Modelo": "M5: Regresión Polinómica (G2)", **met5})

        # M6
        m6 = train_wine_mlp(X_tr_w, y_tr_w)
        met6 = evaluate_model(m6, X_te_w, y_te_w, dataset="Wine")
        save_model(m6, "06_mlp_red_neuronal", "redwine")
        results.append({"Dataset": "Wine", "Modelo": "M6: Perceptrón Multicapa", **met6})

    except FileNotFoundError:
        print("   [!] Error: Archivos procesados de Red Wine no encontrados.")

    # ---------------- FASE C: REPORTE TABULAR ----------------
    if results:
        df_results = pd.DataFrame(results)
        os.makedirs("reports", exist_ok=True)

        # Separar por dataset
        df_candy = df_results[df_results["Dataset"] == "Candy"][["Modelo", "R2", "RMSE"]]
        df_wine = df_results[df_results["Dataset"] == "Wine"][["Modelo", "R2", "MAE", "RMSE"]]

        # Tabla 1: Candy
        print("\n" + "=" * 60)
        print("TABLA 1: COMPARATIVA DATASET CANDY (M1, M2, M3)")
        print("=" * 60)
        print(df_candy.to_string(index=False))

        # Tabla 2: Red Wine
        print("\n" + "=" * 60)
        print("TABLA 2: COMPARATIVA DATASET RED WINE (M4, M5, M6)")
        print("=" * 60)
        print(df_wine.to_string(index=False))

        # Guardar reportes
        df_candy.to_csv("reports/candy_metrics_comparison.csv", index=False)
        df_wine.to_csv("reports/redwine_metrics_comparison.csv", index=False)
        df_results.to_csv("reports/metrics_summary.csv", index=False)

        print("\n[OK] Reportes guardados en 'reports/':")
        print("     - candy_metrics_comparison.csv")
        print("     - redwine_metrics_comparison.csv")
        print("     - metrics_summary.csv")


if __name__ == "__main__":
    run_models_pipeline()