"""
10_model_comparison.py

Comparación consolidada de Naive semanal, Persistencia, XGBoost y RBF.

IMPORTANTE:
- baseline_metrics_by_horizon.csv es la fuente única de ambos baselines.
- Los registros baseline incluidos en los CSV de XGBoost y RBF se descartan
  para evitar duplicados.
- Selección por horizonte con menor MAE en VALIDATION.
- La selección se evalúa después en TEST, sin usar TEST para elegir.
- No reentrena modelos ni modifica datasets/modelos existentes.

Ejecución:
    python .\src\10_model_comparison.py
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
BASELINE_FILE = ROOT / "reports" / "naive_baseline" / "baseline_metrics_by_horizon.csv"
XGB_FILE = ROOT / "reports" / "xgboost" / "xgboost_metrics_by_horizon.csv"
RBF_FILE = ROOT / "reports" / "rbf" / "rbf_metrics_by_horizon.csv"
OUT_DIR = ROOT / "reports" / "model_comparison"

MODEL_LABELS = {
    "naive_week": "Naive semanal",
    "naive_persistence": "Persistencia",
    "xgboost": "XGBoost",
    "rbf": "RBF",
}
MODEL_ORDER = ["naive_week", "naive_persistence", "xgboost", "rbf"]
METRICS = ["mae", "rmse", "smape_pct"]
REQUIRED = {"horizon", "split", "model", "model_name", "n", *METRICS}


def read_csv_checked(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")
    df = pd.read_csv(path)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: faltan columnas {sorted(missing)}")
    df["horizon"] = pd.to_numeric(df["horizon"], errors="raise").astype(int)
    df["n"] = pd.to_numeric(df["n"], errors="raise").astype(int)
    for col in METRICS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def pooled_metrics(df: pd.DataFrame) -> dict:
    n = df["n"].to_numpy(dtype=float)
    result = {"n": int(n.sum())}
    if n.sum() == 0:
        return {**result, "mae": np.nan, "rmse": np.nan, "smape_pct": np.nan}

    result["mae"] = float(np.average(df["mae"], weights=n))
    result["rmse"] = float(np.sqrt(np.average(df["rmse"] ** 2, weights=n)))

    if "smape_valid_n" in df.columns:
        sw = pd.to_numeric(df["smape_valid_n"], errors="coerce").fillna(0).to_numpy(float)
    else:
        sw = n
    valid = np.isfinite(df["smape_pct"].to_numpy(float)) & (sw > 0)
    result["smape_pct"] = (
        float(np.average(df.loc[valid, "smape_pct"], weights=sw[valid]))
        if valid.any() else np.nan
    )
    return result


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    baseline_all = read_csv_checked(BASELINE_FILE)
    xgb_all = read_csv_checked(XGB_FILE)
    rbf_all = read_csv_checked(RBF_FILE)

    # Baselines se toman únicamente del archivo baseline.
    baseline = baseline_all[
        baseline_all["model"].isin(["naive_week", "naive_persistence"])
    ].copy()

    # De los archivos de modelos se conserva exclusivamente el modelo propio.
    xgb = xgb_all[xgb_all["model"] == "xgboost"].copy()
    rbf = rbf_all[rbf_all["model"] == "rbf"].copy()

    combined = pd.concat([baseline, xgb, rbf], ignore_index=True, sort=False)
    combined["model_name"] = combined["model"].map(MODEL_LABELS)

    # Verificación estricta: 168 horizontes × 2 splits × 4 modelos.
    expected_keys = pd.MultiIndex.from_product(
        [range(1, 169), ["validation", "test"], MODEL_ORDER],
        names=["horizon", "split", "model"],
    )
    actual_keys = pd.MultiIndex.from_frame(combined[["horizon", "split", "model"]])

    duplicates = combined.duplicated(["horizon", "split", "model"], keep=False)
    if duplicates.any():
        sample = combined.loc[duplicates, ["horizon", "split", "model"]].head(20)
        raise ValueError("Persisten duplicados tras depurar fuentes:\n" + sample.to_string(index=False))

    missing_keys = expected_keys.difference(actual_keys)
    extra_keys = actual_keys.difference(expected_keys)
    if len(missing_keys) or len(extra_keys):
        raise ValueError(
            f"Estructura incompleta. Faltantes={len(missing_keys)}, extras={len(extra_keys)}. "
            f"Ejemplos faltantes: {list(missing_keys[:8])}"
        )

    combined = combined.sort_values(["split", "horizon", "model"]).reset_index(drop=True)
    combined.to_csv(OUT_DIR / "all_models_metrics_by_horizon.csv",
                    index=False, encoding="utf-8-sig")

    # Resumen pooled: MAE ponderado por n; RMSE reconstruido desde MSE;
    # sMAPE ponderado por el número de denominadores válidos.
    summary_rows = []
    for split in ("validation", "test"):
        for model in MODEL_ORDER:
            sub = combined[(combined["split"] == split) & (combined["model"] == model)]
            summary_rows.append({
                "split": split,
                "model": model,
                "model_name": MODEL_LABELS[model],
                **pooled_metrics(sub),
                "mean_horizon_mae": float(sub["mae"].mean()),
                "mean_horizon_rmse": float(sub["rmse"].mean()),
                "mean_horizon_smape_pct": float(sub["smape_pct"].mean()),
            })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "all_models_metrics_summary.csv",
                   index=False, encoding="utf-8-sig")

    # Tabla amplia por horizonte y mejoras porcentuales contra Naive semanal.
    wide_parts = []
    for split in ("validation", "test"):
        sub = combined[combined["split"] == split]
        part = sub.pivot(index="horizon", columns="model", values=METRICS)
        part.columns = [f"{metric}_{model}" for metric, model in part.columns]
        part = part.reset_index()
        part["split"] = split
        wide_parts.append(part)
    wide = pd.concat(wide_parts, ignore_index=True)

    for metric in METRICS:
        baseline_col = f"{metric}_naive_week"
        for model in ("naive_persistence", "xgboost", "rbf"):
            model_col = f"{metric}_{model}"
            wide[f"{metric}_improvement_vs_weekly_{model}"] = (
                100 * (wide[baseline_col] - wide[model_col]) / wide[baseline_col]
            )
    wide.to_csv(OUT_DIR / "horizon_comparison_and_improvements.csv",
                index=False, encoding="utf-8-sig")

    # Ganador por horizonte y métrica. Un solo ganador por grupo.
    winner_rows = []
    for split in ("validation", "test"):
        for metric in METRICS:
            sub = combined[combined["split"] == split]
            idx = sub.groupby("horizon")[metric].idxmin()
            winners = sub.loc[idx, ["horizon", "model", "model_name", metric]].copy()
            winners["split"] = split
            winners["metric"] = metric
            winners = winners.rename(columns={metric: "winning_metric_value"})
            winner_rows.append(winners)
    winners_df = pd.concat(winner_rows, ignore_index=True)
    winners_df.to_csv(OUT_DIR / "winners_by_horizon.csv",
                      index=False, encoding="utf-8-sig")

    # Selección por horizonte basada solo en VALIDATION MAE; evaluación en TEST.
    val = combined[combined["split"] == "validation"]
    test = combined[combined["split"] == "test"]
    selected_rows = []
    for horizon in range(1, 169):
        val_h = val[val["horizon"] == horizon]
        selected = val_h.loc[val_h["mae"].idxmin()]
        chosen_model = selected["model"]
        test_h = test[
            (test["horizon"] == horizon) & (test["model"] == chosen_model)
        ]
        if len(test_h) != 1:
            raise ValueError(f"No hay exactamente una fila test para H{horizon}, {chosen_model}")
        test_row = test_h.iloc[0]
        selected_rows.append({
            "horizon": horizon,
            "selected_model": chosen_model,
            "selected_model_name": MODEL_LABELS[chosen_model],
            "validation_mae": float(selected["mae"]),
            "test_mae": float(test_row["mae"]),
            "test_rmse": float(test_row["rmse"]),
            "test_smape_pct": float(test_row["smape_pct"]),
            "test_n": int(test_row["n"]),
        })
    selected_df = pd.DataFrame(selected_rows)
    selected_df.to_csv(OUT_DIR / "validation_selected_model_by_horizon_test.csv",
                       index=False, encoding="utf-8-sig")

    selection_counts = (
        selected_df["selected_model"].value_counts()
        .reindex(MODEL_ORDER, fill_value=0)
        .rename_axis("model").reset_index(name="horizons_selected")
    )
    selection_counts["model_name"] = selection_counts["model"].map(MODEL_LABELS)
    selection_counts.to_csv(OUT_DIR / "validation_selection_counts.csv",
                            index=False, encoding="utf-8-sig")

    # Gráficos por horizonte en TEST.
    test_metrics = combined[combined["split"] == "test"]
    for metric, title, ylabel, filename in (
        ("mae", "MAE por horizonte — conjunto de prueba", "MAE (unidad de PML)", "01_test_mae_by_horizon.png"),
        ("rmse", "RMSE por horizonte — conjunto de prueba", "RMSE (unidad de PML)", "02_test_rmse_by_horizon.png"),
        ("smape_pct", "sMAPE por horizonte — conjunto de prueba", "sMAPE (%)", "03_test_smape_by_horizon.png"),
    ):
        fig, ax = plt.subplots(figsize=(13, 6))
        for model in MODEL_ORDER:
            sub = test_metrics[test_metrics["model"] == model].sort_values("horizon")
            ax.plot(sub["horizon"], sub[metric], linewidth=1.6,
                    label=MODEL_LABELS[model])
        ax.set_title(title)
        ax.set_xlabel("Horizonte de predicción (horas)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(1, 168)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT_DIR / filename, dpi=160)
        plt.close(fig)

    # Resumen de ganadores.
    winner_counts = (
        winners_df.groupby(["split", "metric", "model"])
        .size().rename("horizons_won").reset_index()
    )

    lines = [
        "PML FORECAST 7D — COMPARACIÓN CONSOLIDADA",
        "=" * 78,
        "Baselines tomados únicamente de baseline_metrics_by_horizon.csv.",
        "Registros duplicados de baseline en CSV XGBoost/RBF descartados.",
        f"Registros consolidados: {len(combined)} (esperados: 672).",
        "MAE pooled ponderado por n; RMSE reconstruido desde MSE; sMAPE ponderado por smape_valid_n.",
        "Los horizontes comparten observaciones solapadas; no son observaciones independientes.",
        "",
        "RESUMEN POOLED",
        summary[["split", "model_name", "n", "mae", "rmse", "smape_pct"]]
        .to_string(index=False, float_format=lambda v: f"{v:,.4f}"),
        "",
        "HORIZONTES GANADOS POR MODELO",
        winner_counts.to_string(index=False),
        "",
        "SELECCIÓN POR HORIZONTE SEGÚN VALIDATION MAE",
        selection_counts[["model_name", "horizons_selected"]].to_string(index=False),
        "",
        "RESULTADO EN TEST DE LA SELECCIÓN",
        f"MAE medio por horizonte: {selected_df['test_mae'].mean():.6f}",
        f"RMSE medio por horizonte: {selected_df['test_rmse'].mean():.6f}",
        f"sMAPE medio por horizonte: {selected_df['test_smape_pct'].mean():.6f}%",
        "",
        "Nota: la selección se realizó con validation. Test se utilizó solo para evaluar.",
    ]
    (OUT_DIR / "comparison_report.txt").write_text("\n".join(lines), encoding="utf-8")

    print("=" * 78)
    print("COMPARACIÓN CONSOLIDADA — PML FORECAST 7D")
    print("=" * 78)
    expected_rows = 168 * 2 * 4
    print(f"Registros consolidados: {len(combined)} / {expected_rows} esperados")
    print("\nRESUMEN POOLED")
    print(summary[["split", "model_name", "n", "mae", "rmse", "smape_pct"]]
          .to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    print("\nGANADORES POR HORIZONTE")
    print(winner_counts.to_string(index=False))
    print("\nSELECCIÓN SEGÚN VALIDATION MAE")
    print(selection_counts[["model_name", "horizons_selected"]].to_string(index=False))
    print(f"\nResultados guardados en: {OUT_DIR}")
    print("Archivo principal: comparison_report.txt")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"\nERROR en comparación: {exc}")
        raise
