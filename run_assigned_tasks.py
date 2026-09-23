"""Ejecutor adicional: conserva main.py y añade las tareas de Nayver, Mijail y Jhon."""

from src.data_cleaning import run_cleaning_pipeline
from src.data_split import run_split_pipeline
from src.assigned_tasks import run_assigned_tasks


def main():
    print("==================================================")
    print("      TAREAS: NAYVER, MIJAIL Y JHON              ")
    print("==================================================")
    print("[1/3] Limpieza de los datasets originales...")
    run_cleaning_pipeline()
    print("[2/3] División y escalado del pipeline existente...")
    run_split_pipeline()
    print("[3/3] Ampliación Candy 1500, regresión y SVR Wine...")
    run_assigned_tasks()


if __name__ == "__main__":
    main()
