
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "cenace_mtr"
REPORTS_DIR = PROJECT_ROOT / "reports"

TARGET_NODE = "01AAN-85"

frames = []

print(f"Auditando nodo: {TARGET_NODE}")

for file_path in sorted(DATA_DIR.glob("*.csv")):

    print(f"Revisando: {file_path.name}")

    df = pd.read_csv(
        file_path,
        skiprows=7,
        dtype=str,
        encoding="utf-8-sig"
    )

    df.columns = [c.strip().strip('"') for c in df.columns]

    df["Clave del nodo"] = df["Clave del nodo"].str.strip()

    df = df[df["Clave del nodo"] == TARGET_NODE].copy()

    if df.empty:
        continue

    df["Fecha"] = pd.to_datetime(
        df["Fecha"].str.strip(),
        format="%Y-%m-%d",
        errors="coerce"
    )

    df["Hora"] = pd.to_numeric(
        df["Hora"].str.strip(),
        errors="coerce"
    )

    df["FechaHora"] = (
        df["Fecha"] +
        pd.to_timedelta(df["Hora"] - 1, unit="h")
    )

    df["PML"] = pd.to_numeric(
        df["Precio marginal local ($/MWh)"].str.strip(),
        errors="coerce"
    )

    frames.append(
        df[["FechaHora", "Clave del nodo", "PML"]]
    )

if not frames:
    raise ValueError(f"No se encontraron registros para {TARGET_NODE}")

data = pd.concat(frames, ignore_index=True)

data = data.sort_values("FechaHora").reset_index(drop=True)

# ============================================================
# AUDITORÍA
# ============================================================

expected_index = pd.date_range(
    start="2025-01-01 00:00:00",
    end="2026-10-31 23:00:00",
    freq="h"
)

observed_index = pd.DatetimeIndex(data["FechaHora"].dropna())

missing_hours = expected_index.difference(observed_index)
unexpected_hours = observed_index.difference(expected_index)

data["Diferencia_horas"] = (
    data["FechaHora"].diff().dt.total_seconds() / 3600
)

gaps = data[data["Diferencia_horas"] > 1]

print("\n" + "=" * 65)
print("RESULTADOS DE AUDITORÍA")
print("=" * 65)

print(f"Nodo: {TARGET_NODE}")
print(f"Registros: {len(data):,}")
print(f"Valores PML nulos: {data['PML'].isna().sum():,}")
print(f"Fechas iniciales: {data['FechaHora'].min()}")
print(f"Fechas finales: {data['FechaHora'].max()}")
print(f"Horas esperadas: {len(expected_index):,}")
print(f"Horas observadas únicas: {observed_index.nunique():,}")
print(f"Horas faltantes: {len(missing_hours):,}")
print(f"Horas fuera del periodo: {len(unexpected_hours):,}")
print(f"Saltos temporales mayores a 1 hora: {len(gaps):,}")

print("\nESTADÍSTICAS DEL PML")
print(data["PML"].describe())

# Guardar auditoría
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

data.to_csv(
    REPORTS_DIR / "target_node_01AAN-85.csv",
    index=False,
    encoding="utf-8-sig"
)

pd.DataFrame({
    "fecha_hora_faltante": missing_hours
}).to_csv(
    REPORTS_DIR / "target_node_missing_hours.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nReportes generados:")
print("reports/target_node_01AAN-85.csv")
print("reports/target_node_missing_hours.csv")
