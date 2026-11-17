
from pathlib import Path
import sqlite3
import pandas as pd

# ============================================================
# CONFIGURACIÓN
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "cenace_mtr"
REPORTS_DIR = PROJECT_ROOT / "reports"

EXPECTED_FILES = 44
CHUNK_SIZE = 100_000

REQUIRED_COLUMNS = [
    "Fecha",
    "Hora",
    "Clave del nodo",
    "Precio marginal local ($/MWh)",
    "Componente de energia ($/MWh)",
    "Componente de perdidas ($/MWh)",
    "Componente de congestion ($/MWh)",
]

# ============================================================
# PREPARACIÓN
# ============================================================

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

csv_files = sorted(DATA_DIR.glob("*.csv"))

print("=" * 65)
print("VALIDACIÓN DE DATOS CENACE - PML MTR")
print("=" * 65)
print(f"Carpeta de datos: {DATA_DIR}")
print(f"Archivos encontrados: {len(csv_files)}")
print(f"Archivos esperados: {EXPECTED_FILES}")

if not csv_files:
    raise FileNotFoundError(
        f"No se encontraron archivos CSV en: {DATA_DIR}"
    )

if len(csv_files) != EXPECTED_FILES:
    print(
        f"ADVERTENCIA: se esperaban {EXPECTED_FILES} archivos, "
        f"pero se encontraron {len(csv_files)}."
    )

# ============================================================
# BASE DE DATOS TEMPORAL PARA DETECTAR DUPLICADOS
# ============================================================

db_path = REPORTS_DIR / "validation_temp.sqlite"

if db_path.exists():
    db_path.unlink()

connection = sqlite3.connect(db_path)

connection.execute("""
    CREATE TABLE registros (
        fecha_hora TEXT NOT NULL,
        nodo TEXT NOT NULL,
        PRIMARY KEY (fecha_hora, nodo)
    )
""")

# ============================================================
# CONTADORES
# ============================================================

total_rows = 0
total_nulls = 0
invalid_dates = 0
invalid_hours = 0
duplicate_keys = 0
files_with_errors = 0

file_results = []

# ============================================================
# PROCESAMIENTO DE ARCHIVOS
# ============================================================

for file_number, file_path in enumerate(csv_files, start=1):

    print("\n" + "-" * 65)
    print(f"[{file_number}/{len(csv_files)}] Procesando: {file_path.name}")

    file_rows = 0
    file_nulls = 0
    file_invalid_dates = 0
    file_invalid_hours = 0
    file_duplicates = 0
    file_errors = []

    try:
        # Leer por bloques para evitar cargar todo en RAM.
        reader = pd.read_csv(
            file_path,
            skiprows=7,
            dtype=str,
            encoding="utf-8-sig",
            chunksize=CHUNK_SIZE
        )

        first_chunk = True

        for chunk in reader:

            # Normalizar nombres de columnas.
            chunk.columns = [
                column.strip().strip('"')
                for column in chunk.columns
            ]

            if first_chunk:
                missing_columns = [
                    col for col in REQUIRED_COLUMNS
                    if col not in chunk.columns
                ]

                if missing_columns:
                    raise ValueError(
                        f"Faltan columnas: {missing_columns}"
                    )

                first_chunk = False

            # Limpiar espacios y detectar campos vacíos.
            for column in REQUIRED_COLUMNS:
                chunk[column] = chunk[column].str.strip()

            null_mask = (
                chunk[REQUIRED_COLUMNS]
                .isna()
                | chunk[REQUIRED_COLUMNS].eq("")
            )

            null_count = int(null_mask.sum().sum())

            file_nulls += null_count
            total_nulls += null_count

            # Validar fechas.
            parsed_dates = pd.to_datetime(
                chunk["Fecha"],
                format="%Y-%m-%d",
                errors="coerce"
            )

            invalid_date_mask = parsed_dates.isna()

            invalid_date_count = int(invalid_date_mask.sum())

            file_invalid_dates += invalid_date_count
            invalid_dates += invalid_date_count

            # Validar horas (1 a 24).
            numeric_hours = pd.to_numeric(
                chunk["Hora"],
                errors="coerce"
            )

            invalid_hour_mask = (
                numeric_hours.isna()
                | ~numeric_hours.between(1, 24)
                | (numeric_hours % 1 != 0)
            )

            invalid_hour_count = int(invalid_hour_mask.sum())

            file_invalid_hours += invalid_hour_count
            invalid_hours += invalid_hour_count

            # Construir fecha-hora normalizada.
            fecha_hora = (
                parsed_dates
                + pd.to_timedelta(
                    numeric_hours - 1,
                    unit="h"
                )
            )

            # Identificar claves inválidas.
            valid_mask = (
                parsed_dates.notna()
                & ~invalid_hour_mask
                & chunk["Clave del nodo"].notna()
                & chunk["Clave del nodo"].ne("")
            )

            valid_records = pd.DataFrame({
                "fecha_hora": fecha_hora[valid_mask].dt.strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "nodo": chunk.loc[
                    valid_mask, "Clave del nodo"
                ].values
            })

            # Insertar claves. Las repetidas no se insertan.
            records = list(
                valid_records.itertuples(index=False, name=None)
            )

            before = connection.total_changes

            connection.executemany(
                """
                INSERT OR IGNORE INTO registros
                (fecha_hora, nodo)
                VALUES (?, ?)
                """,
                records
            )

            inserted = connection.total_changes - before
            duplicates = len(records) - inserted

            file_duplicates += duplicates
            duplicate_keys += duplicates

            rows_in_chunk = len(chunk)

            file_rows += rows_in_chunk
            total_rows += rows_in_chunk

        connection.commit()

        if (
            file_nulls
            or file_invalid_dates
            or file_invalid_hours
            or file_duplicates
        ):
            files_with_errors += 1

        print(f"Registros: {file_rows:,}")
        print(f"Valores nulos/vacíos: {file_nulls:,}")
        print(f"Fechas inválidas: {file_invalid_dates:,}")
        print(f"Horas inválidas: {file_invalid_hours:,}")
        print(f"Claves duplicadas: {file_duplicates:,}")

    except Exception as error:
        files_with_errors += 1
        file_errors.append(str(error))

        print(f"ERROR: {error}")

    file_results.append({
        "archivo": file_path.name,
        "registros": file_rows,
        "nulos_vacios": file_nulls,
        "fechas_invalidas": file_invalid_dates,
        "horas_invalidas": file_invalid_hours,
        "claves_duplicadas": file_duplicates,
        "errores": " | ".join(file_errors)
    })

# ============================================================
# RESULTADOS FINALES
# ============================================================

unique_keys = connection.execute(
    "SELECT COUNT(*) FROM registros"
).fetchone()[0]

connection.close()

report = pd.DataFrame(file_results)

report_path = REPORTS_DIR / "raw_validation_report.csv"

report.to_csv(
    report_path,
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 65)
print("RESULTADO GENERAL")
print("=" * 65)

print(f"Archivos procesados: {len(csv_files)}")
print(f"Registros totales: {total_rows:,}")
print(f"Claves únicas fecha-hora-nodo: {unique_keys:,}")
print(f"Valores nulos/vacíos: {total_nulls:,}")
print(f"Fechas inválidas: {invalid_dates:,}")
print(f"Horas inválidas: {invalid_hours:,}")
print(f"Claves duplicadas: {duplicate_keys:,}")
print(f"Archivos con errores/incidencias: {files_with_errors}")

print("\nReporte generado:")
print(report_path)

print("\n" + "=" * 65)
print("VALIDACIÓN FINALIZADA")
print("=" * 65)
