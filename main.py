import sys
import os

# Asegurar que la carpeta raíz esté en el path de Python
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_cleaning import run_cleaning_pipeline
from src.data_split import run_split_pipeline

def main():
    print("==================================================")
    print("       PROCESAMIENTO Y PREPARACIÓN DE DATOS       ")
    print("==================================================\n")

    # ----------------------------------------------------
    # PASO 1: LIMPIEZA Y PREPROCESAMIENTO DE DATOS
    # ----------------------------------------------------
    print("[PASO 1/2] Ejecutando limpieza de datasets...")
    try:
        run_cleaning_pipeline()
    except Exception as e:
        print(f"   [!] Error en el Paso 1 (Limpieza): {e}")
        return

    # ----------------------------------------------------
    # PASO 2: DIVISIÓN DE DATOS Y ESCALADO (StandardScaler)
    # ----------------------------------------------------
    print("[PASO 2/2] Dividiendo y escalando datasets (Train/Test)...")
    try:
        run_split_pipeline()
    except Exception as e:
        print(f"   [!] Error en el Paso 2 (División/Escalado): {e}")
        return

    print("\n==================================================")
    print("  ¡PROCESAMIENTO DE DATOS COMPLETADO EXITOSAMENTE!")
    print("  - Datasets limpios y particionados en: data/processed/")
    print("==================================================")

if __name__ == "__main__":
    main()