"""
07_naive_baseline.py

Evalúa dos modelos de referencia para los 168 horizontes:
1. Naive semanal: PML observado 168 horas antes del instante objetivo.
2. Persistencia: último PML observado en el origen de predicción.

Lee los archivos horizon_001.npz ... horizon_168.npz generados por
06_modeling_dataset.py y guarda métricas, predicciones, gráficos e informe.

Ejecución desde la raíz del proyecto:
    python .\src\07_naive_baseline.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "reports" / "modeling_dataset"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "naive_baseline"

HORIZONS = range(1, 169)
MODELS = {
    "naive_week": "Naive semanal (t-168)",
    "naive_persistence": "Persistencia (último valor observado)",
}


def load_split(data: np.lib.npyio.NpzFile, split: str) -> dict[str, np.ndarray]:
    """Carga los campos requeridos de un split y comprueba sus dimensiones."""
    fields = {
        "actual": f"y_{split}",
        "weekly": f"naive_week_{split}",
        "persistence": f"naive_persistence_{split}",
        "origin_time": f"origin_{split}",
        "target_time": f"target_{split}",
    }

    result = {key: np.asarray(data[field]) for key, field in fields.items()}
    lengths = {key: len(value) for key, value in result.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Longitudes inconsistentes en {split}: {lengths}")

    for key in ("actual", "weekly", "persistence"):
        result[key] = result[key].astype(float)

    return result


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Calcula métricas robustas ante precios negativos y valores cero."""
    valid = np.isfinite(actual) & np.isfinite(predicted)
    y = actual[valid]
    yhat = predicted[valid]

    if y.size == 0:
        return {
            "n": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "median_absolute_error": np.nan,
            "smape_pct": np.nan,
            "smape_valid_n": 0,
        }

    errors = y - yhat
    absolute_errors = np.abs(errors)
    denominator = np.abs(y) + np.abs(yhat)
    smape_mask = denominator > 0

    smape = (
        200.0 * absolute_errors[smape_mask] / denominator[smape_mask]
        if np.any(smape_mask)
        else np.array([], dtype=float)
    )

    return {
        "n": int(y.size),
        "mae": float(np.mean(absolute_errors)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "median_absolute_error": float(np.median(absolute_errors)),
        "smape_pct": float(np.mean(smape)) if smape.size else np.nan,
        "smape_valid_n": int(smape.size),
    }


def main() -> int:
    if not INPUT_DIR.exists():
        print(f"ERROR: No existe el directorio de entrada:\n{INPUT_DIR}")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    loaded_horizons = 0

    print("=" * 76)
    print("EVALUACIÓN DE MODELOS NAIVE — PML FORECAST 7D")
    print("=" * 76)

    for horizon in HORIZONS:
        file_path = INPUT_DIR / f"horizon_{horizon:03d}.npz"
        if not file_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo: {file_path}")

        # allow_pickle=False conserva la carga segura de los archivos NPZ.
        with np.load(file_path, allow_pickle=False) as data:
            loaded_horizons += 1

            for split in ("validation", "test"):
                values = load_split(data, split)
                actual = values["actual"]

                predictions = {
                    "naive_week": values["weekly"],
                    "naive_persistence": values["persistence"],
                }

                for model_key, predicted in predictions.items():
                    metrics = calculate_metrics(actual, predicted)
                    metric_rows.append({
                        "horizon": horizon,
                        "split": split,
                        "model": model_key,
                        "model_name": MODELS[model_key],
                        **metrics,
                    })

                    prediction_frames.append(pd.DataFrame({
                        "horizon": horizon,
                        "split": split,
                        "model": model_key,
                        "model_name": MODELS[model_key],
                        "origin_time": values["origin_time"].astype(str),
                        "target_time": values["target_time"].astype(str),
                        "actual": actual,
                        "predicted": predicted,
                        "absolute_error": np.abs(actual - predicted),
                        "error": actual - predicted,
                    }))

        if horizon == 1 or horizon % 24 == 0 or horizon == 168:
            print(f"Horizonte {horizon:03d}/168 procesado")

    metrics_df = pd.DataFrame(metric_rows)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)

    # Métricas por horizonte y split.
    metrics_path = OUTPUT_DIR / "baseline_metrics_by_horizon.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    # Resumen pooled: cada fila horizonte-instante cuenta como una observación.
    summary_rows = []
    for split in ("validation", "test"):
        for model_key, model_name in MODELS.items():
            subset = predictions_df[
                (predictions_df["split"] == split)
                & (predictions_df["model"] == model_key)
            ]
            pooled = calculate_metrics(
                subset["actual"].to_numpy(dtype=float),
                subset["predicted"].to_numpy(dtype=float),
            )
            summary_rows.append({
                "split": split,
                "model": model_key,
                "model_name": model_name,
                **pooled,
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / "baseline_metrics_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    predictions_path = OUTPUT_DIR / "baseline_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False, encoding="utf-8-sig")

    # Gráficos de métricas por horizonte para validación y prueba.
    plot_metrics = [
        ("mae", "MAE por horizonte", "MAE (unidad de PML)", "01_mae_by_horizon.png"),
        ("rmse", "RMSE por horizonte", "RMSE (unidad de PML)", "02_rmse_by_horizon.png"),
        ("smape_pct", "sMAPE por horizonte", "sMAPE (%)", "03_smape_by_horizon.png"),
    ]

    for metric, title, ylabel, filename in plot_metrics:
        fig, ax = plt.subplots(figsize=(12, 6))
        for split, linestyle in (("validation", "-"), ("test", "--")):
            for model_key, model_name in MODELS.items():
                subset = metrics_df[
                    (metrics_df["split"] == split)
                    & (metrics_df["model"] == model_key)
                ].sort_values("horizon")
                ax.plot(
                    subset["horizon"],
                    subset[metric],
                    linestyle=linestyle,
                    linewidth=1.5,
                    label=f"{model_name} — {split}",
                )

        ax.set_title(title)
        ax.set_xlabel("Horizonte de predicción (horas)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(1, 168)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / filename, dpi=160)
        plt.close(fig)

    # Informe de texto.
    report_lines = [
        "PML FORECAST 7D — NAIVE BASELINE",
        "=" * 76,
        f"Horizontes procesados: {loaded_horizons}",
        f"Directorio de entrada: {INPUT_DIR}",
        f"Directorio de salida: {OUTPUT_DIR}",
        "",
        "Definición de modelos:",
        "- Naive semanal: utiliza el PML de la misma hora de la semana anterior (t-168).",
        "- Persistencia: utiliza el último PML observado en el origen de predicción.",
        "",
        "Métricas pooled por split y modelo:",
        "(Cada combinación horizonte-instante se contabiliza como una observación.)",
        "",
    ]

    for _, row in summary_df.iterrows():
        report_lines.extend([
            f"{row['split'].upper()} — {row['model_name']}",
            f"  N: {int(row['n']):,}",
            f"  MAE: {row['mae']:.6f}",
            f"  RMSE: {row['rmse']:.6f}",
            f"  Mediana del error absoluto: {row['median_absolute_error']:.6f}",
            f"  sMAPE: {row['smape_pct']:.6f}%",
            f"  N usado para sMAPE: {int(row['smape_valid_n']):,}",
            "",
        ])

    report_lines.extend([
        "Notas metodológicas:",
        "- sMAPE = promedio de 200*|real-predicho|/(|real|+|predicho|).",
        "- Los casos donde ambos valores son cero se excluyen únicamente del sMAPE.",
        "- MAE y RMSE incluyen precios negativos y cero; no se eliminan por ser extremos.",
        "- Las observaciones de distintos horizontes pueden solaparse y no son independientes.",
        "- El conjunto de prueba debe reservarse para la comparación final; la selección de modelos",
        "  e hiperparámetros debe hacerse con entrenamiento y validación.",
        "",
    ])

    report_path = OUTPUT_DIR / "baseline_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("\n" + "=" * 76)
    print("RESUMEN DE RESULTADOS")
    print("=" * 76)
    print(summary_df[
        ["split", "model_name", "n", "mae", "rmse", "smape_pct"]
    ].to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

    print("\n" + "=" * 76)
    print("EVALUACIÓN FINALIZADA")
    print("=" * 76)
    print(f"Horizontes procesados: {loaded_horizons}")
    print(f"Directorio de salida: {OUTPUT_DIR}")
    print("Archivos generados:")
    print("- baseline_metrics_by_horizon.csv")
    print("- baseline_metrics_summary.csv")
    print("- baseline_predictions.csv")
    print("- 01_mae_by_horizon.png")
    print("- 02_rmse_by_horizon.png")
    print("- 03_smape_by_horizon.png")
    print("- baseline_report.txt")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"\nERROR durante la evaluación: {exc}")
        raise
