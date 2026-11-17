"""
11_model_comparison_r2.py

Calcula R² a partir de predicciones individuales para:
Naive semanal, Persistencia, XGBoost y RBF.

Fuentes:
- reports/naive_baseline/baseline_predictions.csv
- reports/xgboost/xgboost_predictions.csv
- reports/rbf/rbf_predictions.csv

No reentrena modelos ni modifica datasets/modelos existentes.

Ejecución desde la raíz del proyecto:
    python .\src\11_model_comparison_r2.py

Salidas:
- reports/model_comparison_r2/r2_metrics_by_horizon.csv
- reports/model_comparison_r2/r2_summary.csv
- reports/model_comparison_r2/prediction_metrics_by_horizon.csv
- gráficos y r2_report.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASELINE_FILE = ROOT / "reports" / "naive_baseline" / "baseline_predictions.csv"
XGB_FILE = ROOT / "reports" / "xgboost" / "xgboost_predictions.csv"
RBF_FILE = ROOT / "reports" / "rbf" / "rbf_predictions.csv"
OUT_DIR = ROOT / "reports" / "model_comparison_r2"

MODEL_LABELS = {
    "naive_week": "Naive semanal",
    "naive_persistence": "Persistencia",
    "xgboost": "XGBoost",
    "rbf": "RBF",
}
MODEL_ORDER = ["naive_week", "naive_persistence", "xgboost", "rbf"]
SPLITS = ["validation", "test"]
KEYS = ["horizon", "split", "origin_time", "target_time"]


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")


def read_predictions() -> pd.DataFrame:
    """Lee cada fuente y la convierte a un formato largo y homogéneo."""
    for path in (BASELINE_FILE, XGB_FILE, RBF_FILE):
        require_file(path)

    # Baselines: este archivo es la fuente única para ambos modelos naive.
    base_cols = [
        "horizon", "split", "model", "origin_time", "target_time",
        "actual", "predicted",
    ]
    baseline = pd.read_csv(BASELINE_FILE, usecols=base_cols)
    baseline = baseline[
        baseline["model"].isin(["naive_week", "naive_persistence"])
    ].copy()
    baseline = baseline.rename(columns={"predicted": "prediction"})
    baseline = baseline[KEYS + ["model", "actual", "prediction"]]

    # XGBoost: se conserva únicamente su predicción propia.
    xgb_cols = [
        "horizon", "split", "origin_time", "target_time",
        "actual", "xgboost_prediction",
    ]
    xgb = pd.read_csv(XGB_FILE, usecols=xgb_cols)
    xgb = xgb.rename(columns={"xgboost_prediction": "prediction"})
    xgb["model"] = "xgboost"
    xgb = xgb[KEYS + ["model", "actual", "prediction"]]

    # RBF: se conserva únicamente su predicción propia.
    rbf_cols = [
        "horizon", "split", "origin_time", "target_time",
        "actual", "rbf_prediction",
    ]
    rbf = pd.read_csv(RBF_FILE, usecols=rbf_cols)
    rbf = rbf.rename(columns={"rbf_prediction": "prediction"})
    rbf["model"] = "rbf"
    rbf = rbf[KEYS + ["model", "actual", "prediction"]]

    frames = [baseline, xgb, rbf]
    for frame in frames:
        frame["horizon"] = pd.to_numeric(frame["horizon"], errors="raise").astype(int)
        frame["actual"] = pd.to_numeric(frame["actual"], errors="coerce")
        frame["prediction"] = pd.to_numeric(frame["prediction"], errors="coerce")
        frame["split"] = frame["split"].astype(str).str.lower()

    data = pd.concat(frames, ignore_index=True)
    data = data[data["split"].isin(SPLITS)].copy()

    # Eliminar únicamente filas sin actual/predicción; reportar cuántas.
    before = len(data)
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["actual", "prediction"])
    dropped = before - len(data)
    if dropped:
        print(f"Advertencia: se excluyeron {dropped:,} filas con valores no válidos.")

    data["model_name"] = data["model"].map(MODEL_LABELS)

    # Cada clave debe aparecer una sola vez por modelo.
    duplicate_mask = data.duplicated(KEYS + ["model"], keep=False)
    if duplicate_mask.any():
        sample = data.loc[duplicate_mask, KEYS + ["model"]].head(10)
        raise ValueError(
            "Se encontraron predicciones duplicadas por clave y modelo:\n"
            + sample.to_string(index=False)
        )

    # Verificar cobertura de 168 horizontes × 2 splits × 4 modelos.
    counts = data.groupby(["split", "model"]).size()
    expected_pairs = set((s, m) for s in SPLITS for m in MODEL_ORDER)
    actual_pairs = set(counts.index.tolist())
    if actual_pairs != expected_pairs:
        raise ValueError(
            f"Combinaciones split/model inesperadas. Encontradas: {sorted(actual_pairs)}"
        )

    for split in SPLITS:
        for model in MODEL_ORDER:
            subset = data[(data["split"] == split) & (data["model"] == model)]
            horizons = set(subset["horizon"].unique())
            expected_horizons = set(range(1, 169))
            if horizons != expected_horizons:
                missing = sorted(expected_horizons - horizons)
                extra = sorted(horizons - expected_horizons)
                raise ValueError(
                    f"Cobertura incorrecta para {split}/{model}. "
                    f"Faltantes={missing[:10]}, extras={extra[:10]}"
                )

    return data


def calculate_metrics(group: pd.DataFrame) -> dict:
    """Calcula R², MAE, RMSE y sMAPE desde los pares actual/predicción."""
    y = group["actual"].to_numpy(dtype=np.float64)
    pred = group["prediction"].to_numpy(dtype=np.float64)
    n = len(y)

    if n == 0:
        return {"n": 0, "r2": np.nan, "mae": np.nan, "rmse": np.nan,
                "smape_pct": np.nan, "actual_std": np.nan}

    errors = y - pred
    ss_res = float(np.sum(errors ** 2))
    centered = y - float(np.mean(y))
    ss_tot = float(np.sum(centered ** 2))

    # R² no está definido si el valor real es constante.
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    denom = np.abs(y) + np.abs(pred)
    valid = denom > 0
    smape = (
        float(np.mean(200.0 * np.abs(errors[valid]) / denom[valid]))
        if valid.any() else np.nan
    )

    return {
        "n": int(n),
        "r2": float(r2),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "smape_pct": smape,
        "actual_std": float(np.std(y, ddof=0)),
    }


def build_metrics(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_horizon_rows = []
    summary_rows = []

    for (split, model, horizon), group in data.groupby(
        ["split", "model", "horizon"], sort=True
    ):
        metrics = calculate_metrics(group)
        by_horizon_rows.append({
            "split": split,
            "model": model,
            "model_name": MODEL_LABELS[model],
            "horizon": int(horizon),
            **metrics,
        })

    for (split, model), group in data.groupby(["split", "model"], sort=True):
        metrics = calculate_metrics(group)
        summary_rows.append({
            "split": split,
            "model": model,
            "model_name": MODEL_LABELS[model],
            **metrics,
            "mean_horizon_r2": float(
                pd.DataFrame(by_horizon_rows)
                .query("split == @split and model == @model")["r2"].mean()
            ),
        })

    by_horizon = pd.DataFrame(by_horizon_rows).sort_values(
        ["split", "horizon", "model"]
    )
    summary = pd.DataFrame(summary_rows).sort_values(["split", "model"])
    return by_horizon, summary


def save_plots(metrics: pd.DataFrame) -> None:
    for split in SPLITS:
        fig, ax = plt.subplots(figsize=(13, 6))
        subset = metrics[metrics["split"] == split]
        for model in MODEL_ORDER:
            curve = subset[subset["model"] == model].sort_values("horizon")
            ax.plot(
                curve["horizon"], curve["r2"],
                linewidth=1.5, label=MODEL_LABELS[model]
            )
        ax.axhline(0, linewidth=1, linestyle="--")
        ax.set_title(f"R² por horizonte — {split.upper()}")
        ax.set_xlabel("Horizonte de predicción (horas)")
        ax.set_ylabel("R²")
        ax.set_xlim(1, 168)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"r2_{split}_by_horizon.png", dpi=160)
        plt.close(fig)

    test = metrics[metrics["split"] == "test"]
    values = [
        test.loc[test["model"] == model, "r2"].dropna().to_numpy()
        for model in MODEL_ORDER
    ]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(values, tick_labels=[MODEL_LABELS[m] for m in MODEL_ORDER])
    ax.axhline(0, linewidth=1, linestyle="--")
    ax.set_title("Distribución de R² por horizonte — TEST")
    ax.set_ylabel("R²")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "r2_test_distribution.png", dpi=160)
    plt.close(fig)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Leyendo predicciones individuales...")
    data = read_predictions()

    print(f"Predicciones válidas consolidadas: {len(data):,}")
    print("\nRegistros por conjunto y modelo:")
    print(data.groupby(["split", "model"]).size().to_string())

    print("\nCalculando R² y métricas desde actual/prediction...")
    by_horizon, summary = build_metrics(data)

    by_horizon.to_csv(
        OUT_DIR / "r2_metrics_by_horizon.csv",
        index=False, encoding="utf-8-sig"
    )
    summary.to_csv(
        OUT_DIR / "r2_summary.csv",
        index=False, encoding="utf-8-sig"
    )

    # Métricas recalculadas desde predicciones para contrastar el pipeline previo.
    data["absolute_error"] = np.abs(data["actual"] - data["prediction"])
    data["squared_error"] = (data["actual"] - data["prediction"]) ** 2
    metrics_check = (
        data.groupby(["split", "model", "model_name"], as_index=False)
        .agg(
            n=("actual", "size"),
            mae=("absolute_error", "mean"),
            mse=("squared_error", "mean"),
        )
    )
    metrics_check["rmse"] = np.sqrt(metrics_check.pop("mse"))
    metrics_check.to_csv(
        OUT_DIR / "prediction_metrics_summary.csv",
        index=False, encoding="utf-8-sig"
    )

    save_plots(by_horizon)

    lines = [
        "PML FORECAST 7D — EVALUACIÓN CON R²",
        "=" * 78,
        "R² calculado desde predicciones individuales (actual y prediction).",
        "R² global se calcula sobre todos los pares de predicción de cada split/modelo.",
        "Los horizontes pueden compartir observaciones reales; el agregado representa",
        "el conjunto de pares horizonte-observación y no observaciones independientes.",
        "R² por horizonte se calcula separadamente para cada uno de los 168 horizontes.",
        "Un R² negativo es válido e indica desempeño inferior al predictor constante",
        "basado en la media de los valores reales del conjunto evaluado.",
        "",
        "R² GLOBAL — VALIDATION Y TEST",
        summary[["split", "model_name", "n", "r2", "mean_horizon_r2",
                 "mae", "rmse", "smape_pct"]]
        .to_string(index=False, float_format=lambda v: f"{v:,.6f}"),
        "",
        "MÉTRICAS RECALCULADAS DESDE LAS PREDICCIONES",
        metrics_check.to_string(index=False, float_format=lambda v: f"{v:,.6f}"),
        "",
        "Nota: el R² global no es el promedio de los R² por horizonte.",
    ]
    (OUT_DIR / "r2_report.txt").write_text("\n".join(lines), encoding="utf-8")

    print("\nRESUMEN R²")
    print(summary[["split", "model_name", "n", "r2", "mean_horizon_r2",
                   "mae", "rmse"]].to_string(
        index=False, float_format=lambda v: f"{v:,.6f}"
    ))
    print(f"\nResultados guardados en: {OUT_DIR}")
    print("Informe principal: r2_report.txt")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"\nERROR en evaluación R²: {exc}")
        raise
