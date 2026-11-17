# PML Forecast 7D

Proyecto de pronóstico horario del Precio Marginal Local (PML) para el nodo **01AAN-85**, con un horizonte directo de 168 horas (7 días).

El repositorio contiene el código de preparación, auditoría, análisis exploratorio, entrenamiento y evaluación de modelos. Los datos originales, conjuntos de entrenamiento derivados y artefactos pesados se mantienen fuera de Git.

## Objetivo

Evaluar modelos de pronóstico horario a siete días y compararlos con referencias ingenuas, usando una separación temporal entre entrenamiento, validación y prueba.

## Enfoque

- Serie horaria del nodo 01AAN-85.
- 16.056 observaciones entre 2025-01-01 00:00 y 2026-10-31 23:00.
- Auditoría de integridad temporal y análisis exploratorio.
- Pronóstico directo multi-horizonte: 168 modelos independientes, uno por cada hora futura.
- 21 variables predictoras, incluyendo rezagos, estadísticas móviles calculadas con desplazamiento temporal y variables cíclicas de calendario.
- Comparación de persistencia, naive semanal, RBF y XGBoost.

## Resultados principales

En el conjunto de prueba, XGBoost obtuvo los mejores resultados globales entre los modelos evaluados:

| Modelo | MAE | RMSE | sMAPE |
|---|---:|---:|---:|
| Persistencia | 484.7281 | 916.8685 | 41.2889% |
| Naive semanal | 343.3138 | 701.7936 | 29.4506% |
| RBF | 290.5234 | 620.6937 | 27.6318% |
| **XGBoost** | **271.8229** | **580.8672** | **25.6840%** |

Las métricas corresponden al conjunto de prueba y se reportan como resultados del experimento; no constituyen una garantía de desempeño futuro. Consulta [docs/results.md](docs/results.md) para el análisis y las limitaciones.

## Estructura del proyecto

```text
PML_Forecast_7D/
├── data/                  # Datos locales; excluidos de Git
├── models/                # Artefactos locales; excluidos de Git
├── notebooks/
├── reports/               # Figuras y resultados seleccionados
├── src/
│   ├── 01_validate_raw_data.py
│   ├── 02_analyze_data_coverage.py
│   ├── 03_audit_target_node.py
│   ├── 04_exploratory_data_analysis.py
│   ├── 05_temporal_integrity_audit.py
│   ├── 06_modeling_dataset.py
│   ├── 07_naive_baseline.py
│   ├── 08_xgboost_forecast.py
│   ├── 09_rbf_forecast.py
│   ├── 10_model_comparison.py
│   ├── 11_model_comparison_r2.py
│   └── 12_horizon_block_analysis.py
├── docs/
├── .gitignore
└── requirements.txt
```

## Entorno

Se validó el proyecto con Python 3.13.9 y las dependencias directas indicadas en `requirements.txt`.

En Windows PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Si PowerShell bloquea la activación del entorno, para la sesión actual puede ejecutarse:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Ejecución

Los scripts numerados de `src/` representan las etapas del flujo de trabajo. Ejecutarlos en orden, desde la raíz del proyecto, después de disponer de los datos en la estructura local esperada. Los datos no se distribuyen en este repositorio; por tanto, las rutas y los archivos de entrada deben configurarse según la fuente de datos autorizada.

```powershell
python src/01_validate_raw_data.py
```

Continúa con los scripts siguientes según la etapa que se quiera reproducir. Revisa los argumentos y rutas definidos en cada script antes de ejecutar etapas de entrenamiento.

## Datos y reproducibilidad

Los archivos originales CENACE MTR y los conjuntos derivados no se incluyen. Esta decisión evita publicar varios gigabytes de datos y resultados intermedios, y deja pendiente que cada usuario obtenga los datos desde una fuente autorizada. Antes de compartir el repositorio, verifica las condiciones de uso y redistribución de la fuente.

## Documentación

- [Metodología](docs/methodology.md)
- [Resultados y limitaciones](docs/results.md)
- [Estimación preliminar de costos de producción](docs/production_costs.md)

## Alcance y limitaciones

Los resultados corresponden a un nodo, periodo y partición temporal concretos. El desempeño puede cambiar ante cambios de distribución, eventos extremos o modificaciones del mercado. La evaluación histórica no sustituye monitoreo, validación operativa ni recalibración periódica.
