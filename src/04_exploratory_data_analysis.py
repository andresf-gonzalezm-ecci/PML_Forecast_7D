
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURACIÓN
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "reports" / "target_node_01AAN-85.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "eda"

NODE = "01AAN-85"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["figure.figsize"] = (15, 6)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25

# ============================================================
# CARGA Y PREPARACIÓN
# ============================================================

print("=" * 70)
print("ANÁLISIS EXPLORATORIO - PML FORECAST 7D")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

df["FechaHora"] = pd.to_datetime(df["FechaHora"])

df = df.sort_values("FechaHora").reset_index(drop=True)

df["PML"] = pd.to_numeric(df["PML"], errors="coerce")

df = df.dropna(subset=["FechaHora", "PML"]).copy()

df["Hora"] = df["FechaHora"].dt.hour
df["DiaSemana"] = df["FechaHora"].dt.dayofweek
df["NombreDia"] = df["FechaHora"].dt.day_name()
df["Mes"] = df["FechaHora"].dt.month
df["Periodo"] = df["FechaHora"].dt.to_period("M").astype(str)
df["Fecha"] = df["FechaHora"].dt.date

print(f"\nNodo: {NODE}")
print(f"Observaciones: {len(df):,}")
print(f"Fecha inicial: {df['FechaHora'].min()}")
print(f"Fecha final: {df['FechaHora'].max()}")

# ============================================================
# 1. ESTADÍSTICAS DESCRIPTIVAS
# ============================================================

print("\n" + "=" * 70)
print("1. ESTADÍSTICAS DESCRIPTIVAS")
print("=" * 70)

stats = df["PML"].describe(
    percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
)

print(stats)

additional_stats = {
    "Asimetría": df["PML"].skew(),
    "Curtosis": df["PML"].kurtosis(),
    "Valores negativos": int((df["PML"] < 0).sum()),
    "Valores iguales a cero": int((df["PML"] == 0).sum()),
    "Valores superiores a P99": int(
        (df["PML"] > df["PML"].quantile(0.99)).sum()
    ),
}

for name, value in additional_stats.items():
    print(f"{name}: {value}")

stats.to_csv(OUTPUT_DIR / "descriptive_statistics.csv")

# ============================================================
# 2. SERIE TEMPORAL COMPLETA
# ============================================================

print("\nGenerando gráfica de evolución temporal...")

plt.figure()

plt.plot(
    df["FechaHora"],
    df["PML"],
    linewidth=0.5
)

plt.title(f"Evolución histórica del PML - Nodo {NODE}")
plt.xlabel("Fecha")
plt.ylabel("PML ($/MWh)")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "01_time_series.png",
    dpi=160
)

plt.close()

# ============================================================
# 3. DISTRIBUCIÓN DE PRECIOS
# ============================================================

print("Generando distribución de precios...")

plt.figure()

plt.hist(
    df["PML"],
    bins=100
)

plt.title(f"Distribución del PML - Nodo {NODE}")
plt.xlabel("PML ($/MWh)")
plt.ylabel("Frecuencia")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "02_price_distribution.png",
    dpi=160
)

plt.close()

# Distribución sin extremos visuales, conservando todos los datos.
p01 = df["PML"].quantile(0.01)
p99 = df["PML"].quantile(0.99)

plt.figure()

plt.hist(
    df["PML"],
    bins=100,
    range=(p01, p99)
)

plt.title(f"Distribución del PML entre P1 y P99 - Nodo {NODE}")
plt.xlabel("PML ($/MWh)")
plt.ylabel("Frecuencia")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "03_price_distribution_p01_p99.png",
    dpi=160
)

plt.close()

# ============================================================
# 4. COMPORTAMIENTO POR HORA DEL DÍA
# ============================================================

print("Analizando comportamiento horario...")

hourly = df.groupby("Hora")["PML"].agg(
    ["mean", "median", "std", "min", "max", "count"]
)

hourly.to_csv(OUTPUT_DIR / "hourly_statistics.csv")

plt.figure()

plt.plot(
    hourly.index,
    hourly["mean"],
    marker="o",
    label="Media"
)

plt.plot(
    hourly.index,
    hourly["median"],
    marker="o",
    label="Mediana"
)

plt.title("Comportamiento del PML por hora del día")
plt.xlabel("Hora del día")
plt.ylabel("PML ($/MWh)")
plt.xticks(range(24))
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "04_hourly_behavior.png",
    dpi=160
)

plt.close()

# ============================================================
# 5. COMPORTAMIENTO POR DÍA DE LA SEMANA
# ============================================================

print("Analizando comportamiento semanal...")

day_order = [
    "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday"
]

weekly = df.groupby("NombreDia")["PML"].agg(
    ["mean", "median", "std", "count"]
).reindex(day_order)

weekly.to_csv(OUTPUT_DIR / "weekly_statistics.csv")

plt.figure()

plt.bar(
    weekly.index,
    weekly["mean"]
)

plt.title("PML promedio por día de la semana")
plt.xlabel("Día")
plt.ylabel("PML promedio ($/MWh)")
plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "05_weekly_behavior.png",
    dpi=160
)

plt.close()

# ============================================================
# 6. COMPORTAMIENTO MENSUAL
# ============================================================

print("Analizando comportamiento mensual...")

monthly = df.groupby("Periodo")["PML"].agg(
    ["mean", "median", "std", "min", "max", "count"]
)

monthly.to_csv(OUTPUT_DIR / "monthly_statistics.csv")

plt.figure()

plt.plot(
    monthly.index,
    monthly["mean"],
    marker="o",
    label="Media"
)

plt.plot(
    monthly.index,
    monthly["median"],
    marker="o",
    label="Mediana"
)

plt.title("Evolución mensual del PML promedio")
plt.xlabel("Mes")
plt.ylabel("PML ($/MWh)")
plt.xticks(rotation=60)
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "06_monthly_behavior.png",
    dpi=160
)

plt.close()

# ============================================================
# 7. VOLATILIDAD Y PROMEDIO MÓVIL
# ============================================================

print("Calculando volatilidad...")

df["MediaMovil24h"] = df["PML"].rolling(24).mean()
df["DesviacionMovil24h"] = df["PML"].rolling(24).std()

plt.figure()

plt.plot(
    df["FechaHora"],
    df["PML"],
    alpha=0.35,
    linewidth=0.5,
    label="PML"
)

plt.plot(
    df["FechaHora"],
    df["MediaMovil24h"],
    linewidth=1,
    label="Media móvil 24h"
)

plt.title("PML y media móvil de 24 horas")
plt.xlabel("Fecha")
plt.ylabel("PML ($/MWh)")
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "07_moving_average_24h.png",
    dpi=160
)

plt.close()

# ============================================================
# 8. AUTOCORRELACIÓN
# ============================================================

print("Calculando autocorrelación...")

lags = [1, 2, 3, 6, 12, 24, 48, 72, 168]

autocorrelation = pd.DataFrame({
    "lag_horas": lags,
    "autocorrelacion": [
        df["PML"].autocorr(lag=lag)
        for lag in lags
    ]
})

autocorrelation.to_csv(
    OUTPUT_DIR / "autocorrelation.csv",
    index=False
)

print("\nAutocorrelación:")
print(autocorrelation.to_string(index=False))

plt.figure()

plt.bar(
    autocorrelation["lag_horas"].astype(str),
    autocorrelation["autocorrelacion"]
)

plt.title("Autocorrelación del PML por rezago")
plt.xlabel("Rezago (horas)")
plt.ylabel("Autocorrelación")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "08_autocorrelation.png",
    dpi=160
)

plt.close()

# ============================================================
# 9. RESUMEN FINAL
# ============================================================

print("\n" + "=" * 70)
print("RESUMEN DEL ANÁLISIS EXPLORATORIO")
print("=" * 70)

print(f"Media: {df['PML'].mean():.2f}")
print(f"Mediana: {df['PML'].median():.2f}")
print(f"Desviación estándar: {df['PML'].std():.2f}")
print(f"Mínimo: {df['PML'].min():.2f}")
print(f"Máximo: {df['PML'].max():.2f}")
print(f"Asimetría: {df['PML'].skew():.4f}")
print(f"Curtosis: {df['PML'].kurtosis():.4f}")

print("\nAutocorrelaciones:")
for _, row in autocorrelation.iterrows():
    print(
        f"Lag {int(row['lag_horas']):>3}h: "
        f"{row['autocorrelacion']:.4f}"
    )

print("\nGráficas y estadísticas guardadas en:")
print(OUTPUT_DIR)

print("\nANÁLISIS EXPLORATORIO FINALIZADO")