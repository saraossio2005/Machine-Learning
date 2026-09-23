"""Extensiones para las tareas asignadas a Nayver, Mijail y Jhon.

Este módulo no modifica ni elimina los módulos existentes. Genera nuevos
artefactos con nombres explícitos para que el trabajo previo se conserve.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models_saved"
REPORTS = ROOT / "reports"
RANDOM_STATE = 42


def augment_candy_to_1500(source: pd.DataFrame, target_size: int = 1500) -> pd.DataFrame:
    """Crea ejemplos sintéticos reproducibles sin alterar el CSV original.

    Candy tiene 85 observaciones reales. Para cumplir el mínimo de 1.500 se
    hace sobremuestreo con reemplazo y se agrega una perturbación pequeña a
    sus variables continuas. Las variables binarias se conservan como 0/1.
    El archivo resultante queda marcado como sintético/augmentado.
    """
    if len(source) >= target_size:
        return source.sample(n=target_size, random_state=RANDOM_STATE).reset_index(drop=True)

    rng = np.random.default_rng(RANDOM_STATE)
    sampled = source.iloc[rng.integers(0, len(source), size=target_size)].copy()
    numeric_continuous = ["sugarpercent", "pricepercent", "winpercent"]
    for column in numeric_continuous:
        std = source[column].std(ddof=0)
        noise = rng.normal(loc=0, scale=std * 0.03, size=target_size)
        sampled[column] = (sampled[column] + noise).clip(source[column].min(), source[column].max())

    return sampled.reset_index(drop=True)


def prepare_candy_1500():
    """Genera Candy ampliado, divide 80/20 y estandariza sus entradas."""
    PROCESSED.mkdir(parents=True, exist_ok=True)
    candy_clean = pd.read_csv(PROCESSED / "candy_clean.csv")
    candy_1500 = augment_candy_to_1500(candy_clean, target_size=1500)
    candy_1500.to_csv(PROCESSED / "candy_augmented_1500.csv", index=False)

    X = candy_1500.drop(columns=["winpercent"])
    y = candy_1500["winpercent"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X.columns
    )
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
    X_train_scaled.to_csv(PROCESSED / "candy1500_X_train.csv", index=False)
    X_test_scaled.to_csv(PROCESSED / "candy1500_X_test.csv", index=False)
    y_train.reset_index(drop=True).to_csv(PROCESSED / "candy1500_y_train.csv", index=False)
    y_test.reset_index(drop=True).to_csv(PROCESSED / "candy1500_y_test.csv", index=False)
    joblib.dump(scaler, MODELS / "candy" / "candy1500_standard_scaler.joblib")
    return X_train_scaled, X_test_scaled, y_train.reset_index(drop=True), y_test.reset_index(drop=True)


def metrics(model, X_test, y_test):
    prediction = model.predict(X_test)
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_test, prediction))),
        "MAE": float(mean_absolute_error(y_test, prediction)),
        "R2": float(r2_score(y_test, prediction)),
    }


def train_candy_regression(X_train, X_test, y_train, y_test):
    """Tarea de Mijail: regresión lineal para predecir winpercent."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    path = MODELS / "candy" / "03_regresion_lineal_candy1500.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return {"Tarea": "Mijail", "Dataset": "Candy 1500", "Modelo": "Regresión lineal", **metrics(model, X_test, y_test)}


def train_wine_svr():
    """Tarea de Jhon: SVR para predecir quality en Red Wine."""
    X_train = pd.read_csv(PROCESSED / "wine_X_train.csv")
    X_test = pd.read_csv(PROCESSED / "wine_X_test.csv")
    y_train = pd.read_csv(PROCESSED / "wine_y_train.csv").squeeze("columns")
    y_test = pd.read_csv(PROCESSED / "wine_y_test.csv").squeeze("columns")
    model = SVR(kernel="rbf", C=1.0)
    model.fit(X_train, y_train)
    path = MODELS / "redwine" / "07_svr_redwine.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return {"Tarea": "Jhon", "Dataset": "Red Wine", "Modelo": "SVR (kernel RBF)", **metrics(model, X_test, y_test)}


def run_assigned_tasks():
    """Ejecuta exclusivamente las tres tareas nuevas, preservando el trabajo previo."""
    REPORTS.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = prepare_candy_1500()
    results = [
        {
            "Tarea": "Nayver",
            "Dataset": "Candy 1500",
            "Modelo": "Ampliación y StandardScaler",
            "RMSE": "-",
            "MAE": "-",
            "R2": "-",
        },
        train_candy_regression(X_train, X_test, y_train, y_test),
        train_wine_svr(),
    ]
    report = pd.DataFrame(results)
    report.to_csv(REPORTS / "tareas_nayver_mijail_jhon.csv", index=False)
    print("\n============= TAREAS NUEVAS DEL EQUIPO =============")
    print(report.to_string(index=False))
    print("\n[OK] Reporte guardado en reports/tareas_nayver_mijail_jhon.csv")
    print("[OK] Candy ampliado guardado en data/processed/candy_augmented_1500.csv")

