import sys
import os
import pandas as pd

# Asegurar que la carpeta raíz esté en el path de Python
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_cleaning import run_cleaning_pipeline
from src.data_split import run_split_pipeline
from src.models import (
    modelo_1_arbol_decision_candy,
    modelo_2_svm_candy,
    run_models_pipeline,
)

def legacy_main():
    print("==================================================")
    print("       PROCESAMIENTO Y PREPARACIÓN DE DATOS       ")
    print("==================================================\n")

    # ----------------------------------------------------
    # PASO 1: LIMPIEZA Y PREPROCESAMIENTO DE DATOS
    # ----------------------------------------------------
    print("[PASO 1/3] Ejecutando limpieza de datasets...")
    try:
        run_cleaning_pipeline()
    except Exception as e:
        print(f"   [!] Error en el Paso 1 (Limpieza): {e}")
        return

    # ----------------------------------------------------
    # PASO 2: DIVISIÓN DE DATOS Y ESCALADO (StandardScaler)
    # ----------------------------------------------------
    print("[PASO 2/3] Dividiendo y escalando datasets (Train/Test)...")
    try:
        run_split_pipeline()
    except Exception as e:
        print(f"   [!] Error en el Paso 2 (División/Escalado): {e}")
        return

    # ----------------------------------------------------
    # PASO 3: ENTRENAMIENTO Y EVALUACIÓN DE MODELOS
    # ----------------------------------------------------
    print("[PASO 3/3] Entrenando y evaluando modelos...")
    try:
        run_models_pipeline()
    except Exception as e:
        print(f"   [!] Error en el Paso 3 (Modelos): {e}")
        return

    print("\n==================================================")
    print("  ¡PROCESAMIENTO DE DATOS COMPLETADO EXITOSAMENTE!")
    print("  - Datasets limpios y particionados en: data/processed/")
    print("==================================================")

def main():
    """Entrena los dos modelos solicitados con los CSV Candy ya procesados."""
    print("==================================================")
    print("          ENTRENAMIENTO DE MODELOS CANDY          ")
    print("==================================================\n")

    # Los CSV procesados ya están limpios y escalados.
    print("[PASO 1/1] Entrenando y evaluando modelos...")
    try:
        resultados = [
            modelo_1_arbol_decision_candy(),
            modelo_2_svm_candy(),
        ]
    except Exception as error:
        print(f"   [!] Error durante el entrenamiento: {error}")
        return

    # Unir las métricas devueltas por ambos modelos en una tabla comparativa.
    tabla_metricas = pd.DataFrame(resultados)
    print("\n================ TABLA COMPARATIVA ================")
    print(tabla_metricas.to_string(index=False))

    print("\n==================================================")
    print("  MODELOS CANDY ENTRENADOS EXITOSAMENTE")
    print("  - Modelos guardados en: models_saved/candy/")
    print("==================================================")


if __name__ == "__main__":
    main()
