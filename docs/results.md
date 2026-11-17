# Resultados y limitaciones

## Resultados globales en prueba

| Modelo | MAE | RMSE | sMAPE |
|---|---:|---:|---:|
| Persistencia | 484.7281 | 916.8685 | 41.2889% |
| Naive semanal | 343.3138 | 701.7936 | 29.4506% |
| RBF | 290.5234 | 620.6937 | 27.6318% |
| **XGBoost** | **271.8229** | **580.8672** | **25.6840%** |

Frente al naive semanal, XGBoost redujo el MAE aproximadamente 20,8%, el RMSE 17,2% y el sMAPE 12,8% en prueba.

## Validación

| Modelo | MAE | RMSE | sMAPE |
|---|---:|---:|---:|
| Persistencia | 215.8185 | 545.4029 | 24.9995% |
| Naive semanal | 207.2989 | 502.1359 | 24.7850% |
| XGBoost | 220.7840 | 379.8465 | 29.6595% |
| RBF | 236.7162 | 397.2433 | 31.5737% |

La clasificación depende de la métrica: XGBoost obtuvo el menor RMSE de validación, mientras que el naive semanal obtuvo menor MAE y sMAPE. Por ello, no debe afirmarse que XGBoost dominó todas las métricas en validación.

## Comparación por horizonte y bloques

En prueba, XGBoost obtuvo el menor RMSE en los 168 horizontes. Ganó en MAE en 164 de 168 horizontes y en sMAPE en 160 de 168. En el análisis agregado por bloques de 24 horas, XGBoost tuvo el menor MAE en los siete bloques diarios.

## R²

El R² global agrupado en prueba fue:
- Persistencia: -0.938622
- Naive semanal: -0.135790
- RBF: 0.111548
- XGBoost: 0.221905

El R² positivo de XGBoost indica una mejora respecto al predictor constante basado en la media, pero su magnitud también muestra que la varianza explicada sigue siendo limitada. Los R² negativos de los modelos de referencia indican que, bajo esta agregación, fueron peores que ese predictor constante.

El R² se calculó agrupando pares reales-predichos de los 168 horizontes; estos pares comparten marcas temporales y no deben interpretarse como observaciones independientes.

## Cambio de distribución

La dispersión de los valores reales en prueba fue considerablemente mayor que en validación. Esto sugiere un cambio de distribución entre periodos y es una razón para interpretar con cautela las diferencias entre validación y prueba.

## Conclusión

XGBoost es el mejor modelo global en el conjunto de prueba para las métricas reportadas. La conclusión debe limitarse al periodo, nodo y protocolo evaluados. Se recomienda monitoreo de error, análisis de cambios de distribución y reevaluación periódica antes de cualquier uso operativo.
