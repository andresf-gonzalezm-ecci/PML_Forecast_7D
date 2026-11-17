# Estimación preliminar de costos de producción

## Escenario de referencia

Estimación orientativa para una ejecución batch diaria que carga modelos previamente entrenados y genera 168 pronósticos horarios para el nodo 01AAN-85.

Supuestos:
- Una ejecución diaria.
- Máquina Linux de 2 vCPU y 4 GB RAM.
- Una hora de cómputo por día (30 horas al mes).
- 30 GB de disco persistente y 20 GB de almacenamiento de objetos.
- Sin GPU y sin egreso de datos significativo.
- No se incluye reentrenamiento diario.

## Estimación mensual indicativa

| Nube | Estimación mensual |
|---|---:|
| AWS | USD 4.11 |
| Azure | USD 4.47 |
| Google Cloud | USD 2.61 |

Son cifras aproximadas de un escenario económico, no cotizaciones. Dependen de región, sistema operativo, tarifas vigentes, almacenamiento, impuestos y configuración. Deben verificarse en las calculadoras oficiales antes de presupuestar.

## Máquina encendida continuamente

Como referencia, mantener la máquina encendida 730 horas al mes elevaría el costo estimado de cómputo a aproximadamente USD 30.51 en AWS, USD 30.37 en Azure y USD 24.46 en Google Cloud, antes de considerar otros componentes. Programar y detener la máquina puede reducir sustancialmente el costo de cómputo.

## Consideraciones

- La estimación no contempla servicios de monitoreo avanzados, alta disponibilidad, backups adicionales, soporte, redes privadas ni ambientes separados.
- Si se requiere reentrenamiento, mayor frecuencia, múltiples nodos o mayor volumen, debe recalcularse.
- Una alternativa es evaluar servicios serverless de ejecución batch, comparando sus costos y complejidad operacional.
- Validar precios y región objetivo con las calculadoras oficiales de cada proveedor antes de tomar decisiones.
