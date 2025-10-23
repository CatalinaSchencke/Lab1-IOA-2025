# Laboratorio 1 — Clasificación de Melanoma con MobileNetV3 y Optimización Bayesiana

> Laboratorio Realizado para la asignatura ICI5142 - *Investigación de Operaciones Avanzadas* a cargo del profesor **Marcelo Becerra**, con integrantes:
> - Francisco Molinas
> - Rodrigo Molina
> - Rodrigo Fernandez
> - Matias Andrade
> - Catalina Schencke

Este proyecto implementa un flujo completo de entrenamiento, validación y optimización de hiperparámetros para un modelo de **detección binaria de melanoma (Mel / NoMel)** utilizando **PyTorch**, **torchvision** y **Optuna**.

El trabajo se estructura en un **notebook explicativo** (`lab_1.ipynb`) y dos scripts en Python (`train_mnv3.py` y `hpo_optuna.py`) que automatizan el entrenamiento y la búsqueda de hiperparámetros.

---

## Estructura del proyecto

El proyecto sigue la siguiente organización:

```
Lab1_IOA_MNV3/
├── .gitignore
├── data/
│ ├── sample_submission.csv
│ ├── test/
│ ├── test.csv
│ ├── train/
│ │ ├── mel/
│ │ └── nomel/
│ └── val/
│ ├── mel/
│ └── nomel/
├── lab_1.ipynb ← Notebook explicativo (todo el desarrollo)
├── outputs/ ← Modelos entrenados y resultados
├── requirements.txt ← Paquetes necesarios
└── src/
├── train_mnv3.py ← Entrenamiento principal (transfer learning)
└── hpo_optuna.py ← Optimización de hiperparámetros (Optuna)
```



## Descripción general

El notebook `lab_1.ipynb` documenta **todo el proceso del proyecto**, incluyendo:

1. **Exploración y descripción del dataset**  
2. **Preprocesamiento y data augmentation**  
3. **Entrenamiento con MobileNetV3 (transfer learning)**  
4. **Optimización de hiperparámetros con Optuna**  
5. **Evaluación y visualización de resultados**

Los scripts dentro de la carpeta `src/` son los que ejecutan las fases prácticas del entrenamiento.

---

## Instalación de dependencias


Ejecuta el siguiente comando para instalar todas las dependencias:

```bash
pip install -r requirements.txt
```
Se recomienda usar `Python 3.10+` y un entorno virtual (`venv` o `conda`).

## Ejecución del proyecto

1. Entrenamiento base:
```bash
python ./src/train_mnv3.py --data_dir ./data --img_size 224 --batch 8 --epochs_head 2 --epochs_ft 15 --patience 5 --lr 1e-3 --wd 1e-4 --workers 2
```
2. Búsqueda de hiperparámetros (Optuna)
```bash
python ./src/hpo_optuna.py --data_dir ./data --trials 10
```
3. Entrenamiento final con los hiperparámetros óptimos

Finalmente, utiliza el comando que entrega Optuna al terminar. Esto ejecuta el entrenamiento completo con los mejores hiperparámetros encontrados, generando el modelo final `(mnv3_best_f1.pth)` y el archivo de resultados `sample_submission.csv`.

##  Resultados esperados

| Archivo | Descripción |
|----------|--------------|
| `outputs/mnv3_best_f1.pth` | Modelo entrenado con el mejor F1 Score alcanzado durante el entrenamiento. |
| `data/sample_submission.csv` | Archivo de predicciones finales en formato de entrega (0 = No Melanoma, 1 = Melanoma). |
| `outputs/optuna_study.db` | Base de datos de experimentos generada por Optuna con el historial de hiperparámetros y resultados. |
