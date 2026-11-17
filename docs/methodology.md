# Metodología

## Serie objetivo y cobertura

La variable objetivo es el Precio Marginal Local (PML) del nodo **01AAN-85**. El conjunto analizado contiene 16.056 observaciones horarias, desde 2025-01-01 00:00 hasta 2026-10-31 23:00.

La auditoría reportada encontró continuidad horaria del 100%, sin marcas temporales faltantes, duplicadas o inválidas en la serie evaluada.

## Análisis exploratorio y auditoría

El flujo contempla validación de datos, cobertura, auditoría del nodo objetivo, análisis exploratorio y revisión de integridad temporal. El análisis exploratorio identificó asimetría positiva y valores extremos, además de patrones diarios, semanales y autocorrelación. Los valores extremos se conservaron en el análisis.

## Formulación del pronóstico

Se utiliza pronóstico directo para un horizonte de 168 horas. Se entrena un modelo independiente para cada horizonte (h=1,...,168), en lugar de generar las predicciones de forma recursiva.

## Variables predictoras

El conjunto descrito utiliza 21 variables predictoras:
- Rezagos: 1, 2, 3, 6, 12, 24, 48, 72 y 168 horas.
- Media y desviación estándar móviles para ventanas de 24 y 168 horas, calculadas con desplazamiento (`shift(1)`) para evitar incorporar información futura.
- Variables cíclicas de calendario.

## Separación temporal y evaluación

Los datos se separan cronológicamente en entrenamiento, validación y prueba. La validación se utiliza para decisiones de selección; el conjunto de prueba se reserva para la evaluación final reportada.

## Modelos

- Persistencia.
- Naive semanal.
- RBF, con estandarización ajustada sobre entrenamiento, 128 centros MiniBatchKMeans y regresión Ridge con selección de alpha usando validación.
- XGBoost, con un modelo por horizonte.

## Métricas

Se reportan MAE, RMSE y sMAPE. También se realizó comparación por horizonte y por bloques de 24 horas, además de análisis de R² calculado sobre los pares reales-predichos agrupados.

Los detalles de las implementaciones se encuentran en los scripts numerados de `src/`.
