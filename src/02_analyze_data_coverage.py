
from pathlib import Path
import pandas as pd

# ============================================================
# CONFIGURACIÓN
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "cenace_mtr"
REPORTS_DIR = PROJECT_ROOT / "reports"

CHUNK_SIZE = 100_000

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

csv_files = sorted(DATA_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(f"No hay CSV en {DATA_DIR}")

# ============================================================
# ACUMULADORES
# ============================================================

total_rows = 0
min_date = None
max_date = None

unique_nodes = set()
hours_count = {hour: 0 for hour in range(1, 25)}

monthly_stats = {}
node_stats = {}

# ============================================================
# PROCESAMIENTO
# ============================================================

print("=" * 70)
print("ANÁLISIS DE COBERTURA TEMPORAL - PML MTR")
print("=" * 70)
print(f"Archivos encontrados: {len(csv_files)}")

for file_number, file_path in enumerate(csv_files, start=1):

    print(f"\n[{file_number}/{len(csv_files)}] {file_path.name}")

    reader = pd.read_csv(
        file_path,
        skiprows=7,
        dtype=str,
        encoding="utf-8-sig",
        chunksize=CHUNK_SIZE
    )

    for chunk in reader:

        chunk.columns = [
            col.strip().strip('"')
            for col in chunk.columns
        ]

        chunk["Fecha"] = pd.to_datetime(
            chunk["Fecha"].str.strip(),
            format="%Y-%m-%d",
            errors="coerce"
        )

        chunk["Hora"] = pd.to_numeric(
            chunk["Hora"].str.strip(),
            errors="coerce"
        )

        chunk["Clave del nodo"] = (
            chunk["Clave del nodo"].str.strip()
        )

        # Filtrar registros válidos para el análisis.
        valid = chunk[
            chunk["Fecha"].notna()
            & chunk["Hora"].between(1, 24)
            & chunk["Clave del nodo"].notna()
            & chunk["Clave del nodo"].ne("")
        ].copy()

        if valid.empty:
            continue

        total_rows += len(valid)

        # Fechas mínima y máxima.
        chunk_min = valid["Fecha"].min()
        chunk_max = valid["Fecha"].max()

        min_date = (
            chunk_min if min_date is None
            else min(min_date, chunk_min)
        )

        max_date = (
            chunk_max if max_date is None
            else max(max_date, chunk_max)
        )

        # Nodos únicos.
        unique_nodes.update(
            valid["Clave del nodo"].unique()
        )

        # Distribución por hora.
        hour_counts = valid["Hora"].value_counts()

        for hour, count in hour_counts.items():
            hours_count[int(hour)] += int(count)

        # Distribución mensual.
        valid["Periodo"] = valid["Fecha"].dt.to_period("M").astype(str)

        monthly_counts = valid["Periodo"].value_counts()

        for period, count in monthly_counts.items():
            monthly_stats[period] = (
                monthly_stats.get(period, 0) + int(count)
            )

        # Registros por nodo.
        node_counts = valid["Clave del nodo"].value_counts()

        for node, count in node_counts.items():
            node_stats[node] = (
                node_stats.get(node, 0) + int(count)
            )

# ============================================================
# RESULTADOS
# ============================================================

print("\n" + "=" * 70)
print("RESUMEN GENERAL")
print("=" * 70)

print(f"Registros analizados: {total_rows:,}")
print(f"Nodos únicos: {len(unique_nodes):,}")
print(f"Fecha inicial: {min_date}")
print(f"Fecha final: {max_date}")

if min_date is not None and max_date is not None:
    expected_days = (max_date - min_date).days + 1
    print(f"Días calendario del periodo: {expected_days:,}")

# ============================================================
# REPORTE MENSUAL
# ============================================================

monthly_report = pd.DataFrame(
    [
        {"periodo": period, "registros": count}
        for period, count in sorted(monthly_stats.items())
    ]
)

monthly_report.to_csv(
    REPORTS_DIR / "monthly_coverage.csv",
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# REPORTE HORARIO
# ============================================================

hourly_report = pd.DataFrame(
    [
        {"hora": hour, "registros": count}
        for hour, count in hours_count.items()
    ]
)

hourly_report.to_csv(
    REPORTS_DIR / "hourly_coverage.csv",
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# REPORTE POR NODO
# ============================================================

node_report = pd.DataFrame(
    [
        {"nodo": node, "registros": count}
        for node, count in node_stats.items()
    ]
)

node_report = node_report.sort_values(
    "registros",
    ascending=False
)

node_report.to_csv(
    REPORTS_DIR / "node_coverage.csv",
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# RESULTADOS EN CONSOLA
# ============================================================

print("\nREGISTROS POR MES")
print("-" * 50)
print(monthly_report.to_string(index=False))

print("\nDISTRIBUCIÓN POR HORA")
print("-" * 50)
print(hourly_report.to_string(index=False))

print("\n10 NODOS CON MÁS REGISTROS")
print("-" * 50)
print(node_report.head(10).to_string(index=False))

print("\n" + "=" * 70)
print("REPORTES GENERADOS")
print("=" * 70)

print("reports/monthly_coverage.csv")
print("reports/hourly_coverage.csv")
print("reports/node_coverage.csv")

print("\nANÁLISIS DE COBERTURA FINALIZADO")
