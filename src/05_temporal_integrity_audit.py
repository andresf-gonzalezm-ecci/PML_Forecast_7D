from pathlib import Path
import sys

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURACIÓN
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "reports" / "target_node_01AAN-85.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "temporal_audit"

NODE = "01AAN-85"
TIMESTAMP_COL = "FechaHora"
TARGET_COL = "PML"

# Rezagos que utilizamos o consideramos para el modelado.
LAGS_HOURS = [1, 2, 3, 6, 12, 24, 48, 72, 168]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_csv(dataframe: pd.DataFrame, filename: str) -> None:
    """Guarda un CSV incluso cuando el resultado está vacío."""
    dataframe.to_csv(OUTPUT_DIR / filename, index=False, encoding="utf-8-sig")


def main() -> int:
    print("=" * 72)
    print("AUDITORÍA DE INTEGRIDAD TEMPORAL - PML FORECAST 7D")
    print("=" * 72)
    print(f"Nodo: {NODE}")
    print(f"Archivo: {INPUT_FILE}")

    if not INPUT_FILE.exists():
        print("\nERROR: No se encontró el archivo de entrada.")
        print(f"Ruta esperada: {INPUT_FILE}")
        return 1

    # --------------------------------------------------------
    # 1. CARGA Y VALIDACIÓN BÁSICA
    # --------------------------------------------------------
    raw = pd.read_csv(INPUT_FILE)

    required_columns = [TIMESTAMP_COL, TARGET_COL]
    missing_columns = [col for col in required_columns if col not in raw.columns]
    if missing_columns:
        print(f"\nERROR: Faltan columnas requeridas: {missing_columns}")
        print(f"Columnas encontradas: {list(raw.columns)}")
        return 1

    raw["_FechaHora_parseada"] = pd.to_datetime(
        raw[TIMESTAMP_COL], errors="coerce"
    )
    raw["_PML_parseado"] = pd.to_numeric(raw[TARGET_COL], errors="coerce")

    invalid_datetime = raw[raw["_FechaHora_parseada"].isna()].copy()
    invalid_pml = raw[
        raw["_FechaHora_parseada"].notna() & raw["_PML_parseado"].isna()
    ].copy()

    invalid_records = pd.concat(
        [
            invalid_datetime.assign(Motivo="FechaHora inválida"),
            invalid_pml.assign(Motivo="PML nulo o no numérico"),
        ],
        ignore_index=True,
    )
    save_csv(invalid_records, "invalid_records.csv")

    valid_mask = raw["_FechaHora_parseada"].notna() & raw["_PML_parseado"].notna()

    # Construimos el dataframe de trabajo usando las columnas originales
    # y reemplazando sus valores por las versiones ya convertidas.
    # Así evitamos duplicar los nombres FechaHora y PML al renombrar.
    valid = raw.loc[valid_mask, [TIMESTAMP_COL, TARGET_COL]].copy()
    valid[TIMESTAMP_COL] = raw.loc[valid_mask, "_FechaHora_parseada"].to_numpy()
    valid[TARGET_COL] = raw.loc[valid_mask, "_PML_parseado"].to_numpy()

    if valid.empty:
        print("\nERROR: No hay registros válidos para auditar.")
        return 1

    valid = valid.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    # --------------------------------------------------------
    # 2. ALINEACIÓN HORARIA Y DUPLICADOS
    # --------------------------------------------------------
    off_hour_mask = (
        (valid[TIMESTAMP_COL].dt.minute != 0)
        | (valid[TIMESTAMP_COL].dt.second != 0)
        | (valid[TIMESTAMP_COL].dt.microsecond != 0)
    )
    off_hour = valid.loc[off_hour_mask, [TIMESTAMP_COL, TARGET_COL]].copy()
    save_csv(off_hour, "timestamps_off_hour.csv")

    duplicate_mask = valid.duplicated(subset=[TIMESTAMP_COL], keep=False)
    duplicate_timestamps = (
        valid.loc[duplicate_mask]
        .groupby(TIMESTAMP_COL)
        .agg(
            cantidad_registros=(TARGET_COL, "size"),
            pml_minimo=(TARGET_COL, "min"),
            pml_maximo=(TARGET_COL, "max"),
            pml_valores_distintos=(TARGET_COL, "nunique"),
        )
        .reset_index()
        .sort_values(TIMESTAMP_COL)
    )
    save_csv(duplicate_timestamps, "duplicate_timestamps.csv")

    exact_duplicate_rows = int(valid.duplicated().sum())

    # Se usa una sola marca temporal por instante para auditar cobertura.
    # No se elimina ni modifica ningún registro del CSV original.
    unique_times = pd.DatetimeIndex(
        valid[TIMESTAMP_COL].drop_duplicates().sort_values()
    )

    start_time = unique_times.min()
    end_time = unique_times.max()
    expected_times = pd.date_range(start=start_time, end=end_time, freq="h")

    missing_times = expected_times.difference(unique_times)
    missing_hours = pd.DataFrame({TIMESTAMP_COL: missing_times})
    save_csv(missing_hours, "missing_hours.csv")

    # --------------------------------------------------------
    # 3. INTERVALOS ENTRE MARCAS TEMPORALES CONSECUTIVAS
    # --------------------------------------------------------
    interval_hours = (
        (unique_times[1:] - unique_times[:-1]).total_seconds() / 3600
    )
    irregular_mask = interval_hours != 1

    irregular_intervals = pd.DataFrame(
        {
            "FechaHora_anterior": unique_times[:-1][irregular_mask],
            "FechaHora_siguiente": unique_times[1:][irregular_mask],
            "intervalo_horas": interval_hours[irregular_mask],
            "horas_faltantes_estimadas": np.maximum(
                interval_hours[irregular_mask] - 1, 0
            ),
        }
    )
    save_csv(irregular_intervals, "irregular_intervals.csv")

    # --------------------------------------------------------
    # 4. DISPONIBILIDAD REAL DE LOS REZAGOS
    # --------------------------------------------------------
    available_times = set(unique_times)
    lag_rows = []

    for lag in LAGS_HOURS:
        expected_prior_times = unique_times - pd.Timedelta(hours=lag)
        available = np.array(
            [timestamp in available_times for timestamp in expected_prior_times],
            dtype=bool,
        )
        valid_pairs = int(available.sum())
        total_targets = len(unique_times)

        lag_rows.append(
            {
                "lag_horas": lag,
                "pares_temporales_disponibles": valid_pairs,
                "objetivos_evaluables": total_targets,
                "cobertura_pct": (
                    valid_pairs / total_targets * 100 if total_targets else 0
                ),
            }
        )

    lag_availability = pd.DataFrame(lag_rows)
    save_csv(lag_availability, "lag_availability.csv")

    # --------------------------------------------------------
    # 5. COBERTURA POR MES
    # --------------------------------------------------------
    expected_by_month = pd.Series(expected_times).groupby(
        expected_times.to_period("M")
    ).size()
    observed_by_month = pd.Series(unique_times).groupby(
        unique_times.to_period("M")
    ).size()

    monthly_coverage = pd.DataFrame(
        {
            "periodo": expected_by_month.index.astype(str),
            "horas_esperadas": expected_by_month.values,
            "horas_observadas": observed_by_month.reindex(
                expected_by_month.index, fill_value=0
            ).values,
        }
    )
    monthly_coverage["horas_faltantes"] = (
        monthly_coverage["horas_esperadas"]
        - monthly_coverage["horas_observadas"]
    )
    monthly_coverage["cobertura_pct"] = (
        monthly_coverage["horas_observadas"]
        / monthly_coverage["horas_esperadas"]
        * 100
    )
    save_csv(monthly_coverage, "monthly_coverage.csv")

    # --------------------------------------------------------
    # 6. RESUMEN Y DIAGNÓSTICO
    # --------------------------------------------------------
    expected_count = len(expected_times)
    observed_unique_count = len(unique_times)
    missing_count = len(missing_times)
    duplicate_timestamp_count = len(duplicate_timestamps)
    off_hour_count = len(off_hour)

    coverage_pct = (
        observed_unique_count / expected_count * 100 if expected_count else 0
    )

    # La continuidad estricta exige una observación única por hora,
    # todas las marcas alineadas a la hora y ningún dato inválido.
    strict_continuity = (
        missing_count == 0
        and duplicate_timestamp_count == 0
        and off_hour_count == 0
        and len(invalid_records) == 0
    )

    summary_lines = [
        "=" * 72,
        "RESUMEN DE AUDITORÍA TEMPORAL",
        "=" * 72,
        f"Nodo: {NODE}",
        f"Archivo: {INPUT_FILE}",
        f"Registros originales: {len(raw):,}",
        f"Registros válidos: {len(valid):,}",
        f"Registros con fecha inválida: {len(invalid_datetime):,}",
        f"Registros con PML inválido: {len(invalid_pml):,}",
        f"Fecha inicial: {start_time}",
        f"Fecha final: {end_time}",
        f"Horas esperadas en el intervalo: {expected_count:,}",
        f"Marcas temporales únicas observadas: {observed_unique_count:,}",
        f"Horas faltantes: {missing_count:,}",
        f"Marcas temporales duplicadas: {duplicate_timestamp_count:,}",
        f"Filas duplicadas exactas: {exact_duplicate_rows:,}",
        f"Marcas fuera de la hora exacta: {off_hour_count:,}",
        f"Intervalos distintos de una hora: {len(irregular_intervals):,}",
        f"Cobertura temporal: {coverage_pct:.4f}%",
        f"Continuidad horaria estricta: {'SÍ' if strict_continuity else 'NO'}",
        "",
        "Disponibilidad de rezagos:",
        lag_availability.to_string(index=False),
        "",
        "Archivos generados:",
        "- invalid_records.csv",
        "- timestamps_off_hour.csv",
        "- duplicate_timestamps.csv",
        "- missing_hours.csv",
        "- irregular_intervals.csv",
        "- lag_availability.csv",
        "- monthly_coverage.csv",
        "- temporal_audit_summary.txt",
        "",
        f"Directorio de salida: {OUTPUT_DIR}",
    ]

    summary = "\n".join(summary_lines)
    (OUTPUT_DIR / "temporal_audit_summary.txt").write_text(
        summary, encoding="utf-8"
    )

    print("\n" + summary)
    print("\nAUDITORÍA FINALIZADA")

    return 0


if __name__ == "__main__":
    sys.exit(main())
