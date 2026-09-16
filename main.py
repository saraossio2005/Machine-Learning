import sys
import os
import pandas as pd

# Asegurar que la carpeta raíz esté en el path de Python
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_cleaning import run_cleaning_pipeline
from src.data_split import run_split_pipeline
from src.models import run_models_pipeline


def main():
    print("==================================================")
    print("   PIPELINE COMPLETO: 6 MODELOS (CANDY + WINE)   ")
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
    print("\n[PASO 2/3] Dividiendo y escalando datasets (Train/Test)...")
    try:
        run_split_pipeline()
    except Exception as e:
        print(f"   [!] Error en el Paso 2 (División/Escalado): {e}")
        return

    # ----------------------------------------------------
    # PASO 3: ENTRENAMIENTO Y EVALUACIÓN DE LOS 6 MODELOS
    # ----------------------------------------------------
    print("\n[PASO 3/3] Entrenando y evaluando los 6 modelos...")
    try:
        run_models_pipeline()
    except Exception as e:
        print(f"   [!] Error en el Paso 3 (Modelos): {e}")
        return

    print("\n==================================================")
    print("  ¡PIPELINE COMPLETADO EXITOSAMENTE!")
    print("  - Modelos guardados en: models_saved/")
    print("  - Reportes guardados en: reports/")
    print("==================================================")


if __name__ == "__main__":
    main()