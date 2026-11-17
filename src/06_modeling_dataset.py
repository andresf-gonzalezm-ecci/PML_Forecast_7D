from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURACIÓN
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "reports" / "target_node_01AAN-85.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "modeling_dataset"

NODE = "01AAN-85"
TIMESTAMP_COL = "FechaHora"
TARGET_COL = "PML"

HORIZON_HOURS = 168
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15

LAGS_HOURS = [1, 2, 3, 6, 12, 24, 48, 72, 168]
ROLLING_WINDOWS = [24, 168]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def cyclic_features(values: np.ndarray, period: int, prefix: str):
    angle = 2.0 * np.pi * values / period
    return {
        f"{prefix}_sin": np.sin(angle),
        f"{prefix}_cos": np.cos(angle),
    }


def main() -> int:
    print("=" * 76)
    print("RECONSTRUCCIÓN DEL DATASET - PML FORECAST 7D")
    print("=" * 76)
    print(f"Nodo: {NODE}")
    print(f"Entrada: {INPUT_FILE}")

    if not INPUT_FILE.exists():
        print("\nERROR: No se encontró el CSV de entrada.")
        return 1

    df = pd.read_csv(INPUT_FILE)

    missing_columns = [
        col for col in [TIMESTAMP_COL, TARGET_COL] if col not in df.columns
    ]
    if missing_columns:
        print(f"\nERROR: Faltan columnas requeridas: {missing_columns}")
        return 1

    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df.dropna(subset=[TIMESTAMP_COL, TARGET_COL])
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    if df.empty:
        print("\nERROR: No hay observaciones válidas.")
        return 1

    # Revalidación defensiva: el dataset debe ser estrictamente horario.
    timestamps = pd.DatetimeIndex(df[TIMESTAMP_COL])
    expected = pd.date_range(timestamps.min(), timestamps.max(), freq="h")

    if timestamps.has_duplicates:
        print("\nERROR: Hay marcas temporales duplicadas. Ejecuta primero la auditoría.")
        return 1
    if not timestamps.equals(expected):
        print("\nERROR: La serie no tiene continuidad horaria perfecta.")
        print(f"Observadas: {len(timestamps):,} | Esperadas: {len(expected):,}")
        return 1

    if not np.all(np.isfinite(df[TARGET_COL].to_numpy(dtype=float))):
        print("\nERROR: El objetivo contiene valores no finitos.")
        return 1

    # --------------------------------------------------------
    # 1. VARIABLES HISTÓRICAS DISPONIBLES EN EL ORIGEN
    # --------------------------------------------------------
    data = pd.DataFrame(index=df.index)
    y = df[TARGET_COL].astype(float)

    for lag in LAGS_HOURS:
        data[f"PML_lag_{lag}"] = y.shift(lag)

    for window in ROLLING_WINDOWS:
        # shift(1) garantiza que el precio de la hora objetivo no
        # participe en las estadísticas disponibles en el origen.
        historical = y.shift(1).rolling(window=window, min_periods=window)
        data[f"PML_mean_{window}h"] = historical.mean()
        data[f"PML_std_{window}h"] = historical.std()

    # Variables temporales del instante que se desea predecir.
    # Son conocidas de antemano y se calculan para cada horizonte.
    timestamps_series = df[TIMESTAMP_COL]
    total_rows = len(df)

    # El origen debe disponer de todo el histórico de 168 horas.
    first_origin = max(LAGS_HOURS + ROLLING_WINDOWS)
    # Para cada horizonte, el target debe existir en el histórico.
    last_origin = total_rows - HORIZON_HOURS - 1

    if last_origin < first_origin:
        print("\nERROR: No hay suficientes observaciones para crear ejemplos.")
        return 1

    origins = np.arange(first_origin, last_origin + 1, dtype=int)
    origin_timestamps = timestamps_series.iloc[origins].reset_index(drop=True)

    # Particiones cronológicas por FECHA OBJETIVO, no aleatorias.
    # Así cada target pertenece a una sola partición en todos los horizontes.
    target_timestamps_all = timestamps_series.iloc[
        origins[0] + 1 : origins[-1] + HORIZON_HOURS + 1
    ]
    target_start = timestamps_series.iloc[first_origin + 1]
    target_end = timestamps_series.iloc[-1]

    total_target_hours = len(timestamps_series) - first_origin - 1
    train_end_position = first_origin + int(
        total_target_hours * TRAIN_FRACTION
    )
    validation_end_position = train_end_position + int(
        total_target_hours * VALIDATION_FRACTION
    )

    train_end_time = timestamps_series.iloc[min(train_end_position, total_rows - 1)]
    validation_end_time = timestamps_series.iloc[
        min(validation_end_position, total_rows - 1)
    ]

    split_metadata = {
        "train_target_end_exclusive": str(train_end_time),
        "validation_target_end_exclusive": str(validation_end_time),
        "test_target_end_inclusive": str(target_end),
        "fractions_requested": {
            "train": TRAIN_FRACTION,
            "validation": VALIDATION_FRACTION,
            "test": TEST_FRACTION,
        },
        "split_rule": "Cronológico por timestamp del objetivo; sin shuffle.",
    }

    # Características base: rezagos y estadísticas calculados en el origen.
    feature_names_base = list(data.columns)
    base_matrix = data.iloc[origins].to_numpy(dtype=np.float32)

    # Una carpeta/archivo por horizonte. Cada archivo contiene matrices
    # train/validation/test, objetivo real y dos baselines Naive.
    horizon_summaries = []

    print(f"\nObservaciones originales: {total_rows:,}")
    print(f"Orígenes utilizables: {len(origins):,}")
    print(f"Primer origen: {origin_timestamps.iloc[0]}")
    print(f"Último origen: {origin_timestamps.iloc[-1]}")
    print(f"Fin de entrenamiento (objetivo): {train_end_time}")
    print(f"Fin de validación (objetivo): {validation_end_time}")
    print("\nGenerando 168 datasets directos...")

    for horizon in range(1, HORIZON_HOURS + 1):
        target_indices = origins + horizon
        target_times = timestamps_series.iloc[target_indices].reset_index(drop=True)
        target_values = y.iloc[target_indices].to_numpy(dtype=np.float32)

        # Calendario del timestamp futuro objetivo.
        target_hour = target_times.dt.hour.to_numpy()
        target_dayofweek = target_times.dt.day_of_week.to_numpy()
        target_month = target_times.dt.month.to_numpy()
        target_dayofyear = target_times.dt.day_of_year.to_numpy()

        calendar = {
            **cyclic_features(target_hour, 24, "target_hour"),
            **cyclic_features(target_dayofweek, 7, "target_dayofweek"),
            **cyclic_features(target_month - 1, 12, "target_month"),
            **cyclic_features(target_dayofyear - 1, 365.25, "target_dayofyear"),
        }

        calendar_matrix = np.column_stack(
            [calendar[name] for name in calendar]
        ).astype(np.float32)

        X = np.column_stack([base_matrix, calendar_matrix]).astype(np.float32)
        feature_names = feature_names_base + list(calendar.keys())

        # Naive semanal: mismo precio de la misma hora de la semana anterior.
        # Naive persistente: último precio conocido en el origen.
        naive_week = y.iloc[target_indices - 168].to_numpy(dtype=np.float32)
        naive_persistence = y.iloc[origins].to_numpy(dtype=np.float32)

        train_mask = target_times < train_end_time
        validation_mask = (
            (target_times >= train_end_time)
            & (target_times < validation_end_time)
        )
        test_mask = target_times >= validation_end_time

        horizon_file = OUTPUT_DIR / f"horizon_{horizon:03d}.npz"
        np.savez_compressed(
            horizon_file,
            X_train=X[train_mask],
            y_train=target_values[train_mask],
            X_validation=X[validation_mask],
            y_validation=target_values[validation_mask],
            X_test=X[test_mask],
            y_test=target_values[test_mask],
            naive_week_validation=naive_week[validation_mask],
            naive_week_test=naive_week[test_mask],
            naive_persistence_validation=naive_persistence[validation_mask],
            naive_persistence_test=naive_persistence[test_mask],
            # dtype=str fuerza arreglos Unicode (<U...), no object.
            # Así np.load() puede leer el NPZ con allow_pickle=False.
            origin_train=np.asarray(origin_timestamps[train_mask].astype(str), dtype=str),
            origin_validation=np.asarray(origin_timestamps[validation_mask].astype(str), dtype=str),
            origin_test=np.asarray(origin_timestamps[test_mask].astype(str), dtype=str),
            target_train=np.asarray(target_times[train_mask].astype(str), dtype=str),
            target_validation=np.asarray(target_times[validation_mask].astype(str), dtype=str),
            target_test=np.asarray(target_times[test_mask].astype(str), dtype=str),
            feature_names=np.array(feature_names, dtype=str),
        )

        horizon_summaries.append(
            {
                "horizonte_horas": horizon,
                "train": int(train_mask.sum()),
                "validation": int(validation_mask.sum()),
                "test": int(test_mask.sum()),
                "features": len(feature_names),
                "archivo": horizon_file.name,
            }
        )

        if horizon in [1, 24, 48, 72, 120, 168]:
            print(
                f"  Horizonte {horizon:3d}: "
                f"train={train_mask.sum():,}, "
                f"validation={validation_mask.sum():,}, "
                f"test={test_mask.sum():,}"
            )

    summary_df = pd.DataFrame(horizon_summaries)
    summary_df.to_csv(
        OUTPUT_DIR / "dataset_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "project": "PML Forecast 7D",
        "node": NODE,
        "input_file": str(INPUT_FILE),
        "observations": int(total_rows),
        "date_start": str(timestamps.min()),
        "date_end": str(timestamps.max()),
        "forecast_horizon_hours": HORIZON_HOURS,
        "forecast_strategy": "Direct: un dataset y un modelo por horizonte",
        "lags_hours": LAGS_HOURS,
        "rolling_windows_hours": ROLLING_WINDOWS,
        "base_features": feature_names_base,
        "calendar_features": [
            "target_hour_sin", "target_hour_cos",
            "target_dayofweek_sin", "target_dayofweek_cos",
            "target_month_sin", "target_month_cos",
            "target_dayofyear_sin", "target_dayofyear_cos",
        ],
        "target": TARGET_COL,
        "naive_baselines": [
            "naive_week: PML en target_timestamp - 168 horas",
            "naive_persistence: último PML conocido en el origen",
        ],
        "first_origin": str(origin_timestamps.iloc[0]),
        "last_origin": str(origin_timestamps.iloc[-1]),
        "split": split_metadata,
        "leakage_controls": [
            "Todas las variables lag se calculan en el origen.",
            "Las estadísticas móviles usan shift(1).",
            "Las variables calendario corresponden al timestamp objetivo y son conocidas.",
            "La partición es cronológica por timestamp objetivo.",
            "No se usa shuffle.",
        ],
        "storage": "Un archivo NPZ comprimido por horizonte, con train/validation/test.",
    }

    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 76)
    print("RECONSTRUCCIÓN FINALIZADA")
    print("=" * 76)
    print(f"Horizontes generados: {len(summary_df)}")
    print(f"Variables por horizonte: {summary_df['features'].iloc[0]}")
    print(f"Directorio de salida: {OUTPUT_DIR}")
    print("Archivos principales:")
    print("- horizon_001.npz ... horizon_168.npz")
    print("- dataset_summary.csv")
    print("- metadata.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
