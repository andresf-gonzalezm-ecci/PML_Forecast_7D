"""
12_horizon_block_analysis.py

Analiza el rendimiento por bloques de 24 horas (7 días) para:
Naive semanal, Persistencia, XGBoost y RBF.

Calcula MAE, RMSE, sMAPE y R² directamente desde las predicciones individuales.
También resume ganadores por horizonte dentro de cada bloque y compara la
variabilidad de los valores reales entre VALIDATION y TEST.

No reentrena modelos.

Ejecución desde la raíz del proyecto:
    python .\src\12_horizon_block_analysis.py
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
OUT_DIR = ROOT / "reports" / "horizon_block_analysis"

MODEL_LABELS = {
    "naive_week": "Naive semanal",
    "naive_persistence": "Persistencia",
    "xgboost": "XGBoost",
    "rbf": "RBF",
}
MODEL_ORDER = ["naive_week", "naive_persistence", "xgboost", "rbf"]
SPLITS = ["validation", "test"]
KEYS = ["horizon", "split", "origin_time", "target_time"]

BLOCKS = [
    (1, 24, "Día 1"),
    (25, 48, "Día 2"),
    (49, 72, "Día 3"),
    (73, 96, "Día 4"),
    (97, 120, "Día 5"),
    (121, 144, "Día 6"),
    (145, 168, "Día 7"),
]


def load_predictions() -> pd.DataFrame:
    for path in (BASELINE_FILE, XGB_FILE, RBF_FILE):
        if not path.exists():
            raise FileNotFoundError(f"No se encontró: {path}")

    baseline = pd.read_csv(
        BASELINE_FILE,
        usecols=["horizon", "split", "model", "origin_time",
                 "target_time", "actual", "predicted"],
    )
    baseline = baseline[
        baseline["model"].isin(["naive_week", "naive_persistence"])
    ].copy().rename(columns={"predicted": "prediction"})
    baseline = baseline[KEYS + ["model", "actual", "prediction"]]

    xgb = pd.read_csv(
        XGB_FILE,
        usecols=["horizon", "split", "origin_time", "target_time",
                 "actual", "xgboost_prediction"],
    ).rename(columns={"xgboost_prediction": "prediction"})
    xgb["model"] = "xgboost"
    xgb = xgb[KEYS + ["model", "actual", "prediction"]]

    rbf = pd.read_csv(
        RBF_FILE,
        usecols=["horizon", "split", "origin_time", "target_time",
                 "actual", "rbf_prediction"],
    ).rename(columns={"rbf_prediction": "prediction"})
    rbf["model"] = "rbf"
    rbf = rbf[KEYS + ["model", "actual", "prediction"]]

    data = pd.concat([baseline, xgb, rbf], ignore_index=True)
    data["horizon"] = pd.to_numeric(data["horizon"], errors="raise").astype(int)
    data["actual"] = pd.to_numeric(data["actual"], errors="coerce")
    data["prediction"] = pd.to_numeric(data["prediction"], errors="coerce")
    data["split"] = data["split"].astype(str).str.lower()
    data = data[data["split"].isin(SPLITS)].copy()
    data = data.replace([np.inf, -np.inf], np.nan)
    invalid = data[["actual", "prediction"]].isna().any(axis=1).sum()
    if invalid:
        print(f"Advertencia: {invalid:,} filas inválidas excluidas.")
        data = data.dropna(subset=["actual", "prediction"])

    dup = data.duplicated(KEYS + ["model"], keep=False)
    if dup.any():
        raise ValueError(
            "Hay predicciones duplicadas:\n"
            + data.loc[dup, KEYS + ["model"]].head(10).to_string(index=False)
        )

    # Confirmar que están los siete bloques y los cuatro modelos en ambos splits.
    expected_horizons = set(range(1, 169))
    for split in SPLITS:
        for model in MODEL_ORDER:
            sub = data[(data["split"] == split) & (data["model"] == model)]
            if set(sub["horizon"].unique()) != expected_horizons:
                raise ValueError(f"Cobertura incompleta: {split}/{model}")

    return data


def metrics(group: pd.DataFrame) -> dict:
    y = group["actual"].to_numpy(dtype=np.float64)
    p = group["prediction"].to_numpy(dtype=np.float64)
    err = y - p
    n = len(y)
    ss_tot = float(np.sum((y - np.mean(y)) ** 2)) if n else 0.0
    ss_res = float(np.sum(err ** 2)) if n else 0.0
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    denom = np.abs(y) + np.abs(p)
    valid = denom > 0
    smape = (
        float(np.mean(200 * np.abs(err[valid]) / denom[valid]))
        if valid.any() else np.nan
    )
    return {
        "n_forecast_pairs": n,
        "mae": float(np.mean(np.abs(err))) if n else np.nan,
        "rmse": float(np.sqrt(np.mean(err ** 2))) if n else np.nan,
        "smape_pct": smape,
        "r2": r2,
        "actual_mean": float(np.mean(y)) if n else np.nan,
        "actual_std": float(np.std(y, ddof=0)) if n else np.nan,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Cargando predicciones individuales...")
    data = load_predictions()

    block_rows = []
    for split in SPLITS:
        for start, end, label in BLOCKS:
            for model in MODEL_ORDER:
                sub = data[
                    (data["split"] == split)
                    & (data["model"] == model)
                    & data["horizon"].between(start, end)
                ]
                block_rows.append({
                    "split": split,
                    "block": label,
                    "horizon_start": start,
                    "horizon_end": end,
                    "model": model,
                    "model_name": MODEL_LABELS[model],
                    **metrics(sub),
                })

    block_metrics = pd.DataFrame(block_rows)
    block_metrics.to_csv(
        OUT_DIR / "metrics_by_24h_block.csv",
        index=False, encoding="utf-8-sig"
    )

    # Métricas por horizonte: permite ver si el comportamiento interno del bloque
    # es homogéneo o si el promedio del bloque oculta horizontes concretos.
    horizon_rows = []
    for (split, model, horizon), sub in data.groupby(
        ["split", "model", "horizon"], sort=True
    ):
        horizon_rows.append({
            "split": split,
            "model": model,
            "model_name": MODEL_LABELS[model],
            "horizon": int(horizon),
            **metrics(sub),
        })
    by_horizon = pd.DataFrame(horizon_rows)
    by_horizon.to_csv(
        OUT_DIR / "metrics_by_horizon.csv",
        index=False, encoding="utf-8-sig"
    )

    # Conteo de horizontes ganados por modelo dentro de cada bloque y split.
    wins = []
    for split in SPLITS:
        for start, end, label in BLOCKS:
            subset = by_horizon[
                (by_horizon["split"] == split)
                & by_horizon["horizon"].between(start, end)
            ]
            for metric in ["mae", "rmse", "smape_pct"]:
                idx = subset.groupby("horizon")[metric].idxmin()
                winners = subset.loc[idx, ["horizon", "model"]]
                counts = winners["model"].value_counts()
                for model in MODEL_ORDER:
                    wins.append({
                        "split": split,
                        "block": label,
                        "metric": metric,
                        "model": model,
                        "model_name": MODEL_LABELS[model],
                        "horizons_won": int(counts.get(model, 0)),
                    })
    wins_df = pd.DataFrame(wins)
    wins_df.to_csv(
        OUT_DIR / "horizon_wins_by_block.csv",
        index=False, encoding="utf-8-sig"
    )

    # Comparación de dispersión de precios reales por split y bloque.
    actual_rows = []
    for split in SPLITS:
        for start, end, label in BLOCKS:
            # Los valores reales son los mismos para cada modelo en las mismas
            # claves. Usamos una sola fuente/modelo para no multiplicarlos.
            sub = data[
                (data["split"] == split)
                & (data["model"] == "naive_week")
                & data["horizon"].between(start, end)
            ]
            actual_rows.append({
                "split": split,
                "block": label,
                "horizon_start": start,
                "horizon_end": end,
                "n_forecast_pairs": len(sub),
                "actual_mean": sub["actual"].mean(),
                "actual_std": sub["actual"].std(ddof=0),
                "actual_min": sub["actual"].min(),
                "actual_median": sub["actual"].median(),
                "actual_max": sub["actual"].max(),
            })
    actual_distribution = pd.DataFrame(actual_rows)
    actual_distribution.to_csv(
        OUT_DIR / "actual_distribution_by_block.csv",
        index=False, encoding="utf-8-sig"
    )

    # Gráficos por métrica: un panel por split, líneas por modelo y siete bloques.
    for split in SPLITS:
        sub = block_metrics[block_metrics["split"] == split]
        for metric, ylabel, filename in [
            ("mae", "MAE", f"{split}_mae_by_block.png"),
            ("rmse", "RMSE", f"{split}_rmse_by_block.png"),
            ("smape_pct", "sMAPE (%)", f"{split}_smape_by_block.png"),
            ("r2", "R²", f"{split}_r2_by_block.png"),
        ]:
            fig, ax = plt.subplots(figsize=(12, 6))
            for model in MODEL_ORDER:
                curve = sub[sub["model"] == model].copy()
                curve["block_order"] = curve["horizon_start"]
                curve = curve.sort_values("block_order")
                ax.plot(
                    curve["block"], curve[metric],
                    marker="o", linewidth=1.7,
                    label=MODEL_LABELS[model]
                )
            if metric == "r2":
                ax.axhline(0, linewidth=1, linestyle="--")
            ax.set_title(f"{ylabel} por bloque de 24 horas — {split.upper()}")
            ax.set_xlabel("Día del horizonte de pronóstico")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(OUT_DIR / filename, dpi=160)
            plt.close(fig)

    # Reporte de lectura rápida.
    test = block_metrics[block_metrics["split"] == "test"]
    lines = [
        "PML FORECAST 7D — ANÁLISIS POR BLOQUES DE 24 HORAS",
        "=" * 86,
        "Las métricas se calculan sobre pares horizonte-observación dentro de cada bloque.",
        "R² se calcula de nuevo sobre todos los pares del bloque; no es promedio de R² horario.",
        "Los horizontes pueden compartir observaciones reales; interpretar los pares como",
        "pronósticos evaluados, no como observaciones independientes.",
        "",
        "RESULTADOS TEST POR BLOQUE",
        test[["block", "model_name", "n_forecast_pairs", "mae", "rmse",
              "smape_pct", "r2"]].to_string(
            index=False, float_format=lambda v: f"{v:,.5f}"
        ),
        "",
        "HORIZONTES GANADOS POR BLOQUE — TEST",
        wins_df[wins_df["split"] == "test"].to_string(index=False),
        "",
        "DISTRIBUCIÓN DE VALORES REALES POR BLOQUE",
        actual_distribution.to_string(
            index=False, float_format=lambda v: f"{v:,.5f}"
        ),
    ]
    (OUT_DIR / "block_analysis_report.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    print(f"\nPredicciones analizadas: {len(data):,}")
    print("\nMÉTRICAS TEST POR BLOQUE")
    print(
        test[["block", "model_name", "mae", "rmse", "smape_pct", "r2"]]
        .to_string(index=False, float_format=lambda v: f"{v:,.5f}")
    )
    print(f"\nResultados guardados en: {OUT_DIR}")
    print("Informe principal: block_analysis_report.txt")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"\nERROR en análisis por bloques: {exc}")
        raise
