"""
08_xgboost_forecast.py

Entrena 168 modelos XGBoost independientes (uno por horizonte) para PML Forecast 7D.
Usa train para ajustar, validation para early stopping y test solo para evaluación final.

Ejecución desde la raíz del proyecto:
    python .\src\08_xgboost_forecast.py

Dependencia:
    python -m pip install xgboost
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import xgboost as xgb
except ImportError:
    print("ERROR: XGBoost no está instalado en este entorno.")
    print("Instálalo con: python -m pip install xgboost")
    raise


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "reports" / "modeling_dataset"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "xgboost"

HORIZONS = range(1, 169)

# Configuración inicial controlada; no es una búsqueda exhaustiva.
XGB_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "max_depth": 6,
    "eta": 0.05,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "min_child_weight": 5,
    "lambda": 5.0,
    "alpha": 0.0,
    "seed": 42,
    "nthread": -1,
    "verbosity": 0,
}
NUM_BOOST_ROUND = 1500
EARLY_STOPPING_ROUNDS = 75


def load_npz_split(data: np.lib.npyio.NpzFile, split: str) -> dict[str, np.ndarray]:
    """Carga X/y para train y añade baselines/metadatos en validation y test."""
    fields = {
        "X": f"X_{split}",
        "y": f"y_{split}",
    }
    if split in ("validation", "test"):
        fields.update({
            "weekly": f"naive_week_{split}",
            "persistence": f"naive_persistence_{split}",
            "origin_time": f"origin_{split}",
            "target_time": f"target_{split}",
        })

    result = {key: np.asarray(data[field]) for key, field in fields.items()}

    if result["X"].ndim != 2:
        raise ValueError(f"X_{split} debe ser una matriz 2D; forma recibida: {result['X'].shape}")
    n = len(result["y"])
    if result["X"].shape[0] != n:
        raise ValueError(f"X_{split} e y_{split} tienen distinto número de filas.")

    if split in ("validation", "test"):
        for key in ("weekly", "persistence", "origin_time", "target_time"):
            if len(result[key]) != n:
                raise ValueError(f"{key} no tiene la misma longitud que y_{split}.")
        result["weekly"] = result["weekly"].astype(float)
        result["persistence"] = result["persistence"].astype(float)

    result["y"] = result["y"].astype(np.float32)
    result["X"] = result["X"].astype(np.float32)
    return result


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(actual) & np.isfinite(predicted)
    y = actual[valid].astype(float)
    yhat = predicted[valid].astype(float)
    if y.size == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "median_absolute_error": np.nan,
                "smape_pct": np.nan, "smape_valid_n": 0}

    abs_error = np.abs(y - yhat)
    denom = np.abs(y) + np.abs(yhat)
    smape_mask = denom > 0
    smape_values = 200.0 * abs_error[smape_mask] / denom[smape_mask]
    return {
        "n": int(y.size),
        "mae": float(np.mean(abs_error)),
        "rmse": float(np.sqrt(np.mean((y - yhat) ** 2))),
        "median_absolute_error": float(np.median(abs_error)),
        "smape_pct": float(np.mean(smape_values)) if smape_values.size else np.nan,
        "smape_valid_n": int(smape_values.size),
    }


def predict_best(model: xgb.Booster, matrix: xgb.DMatrix) -> np.ndarray:
    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is not None:
        return model.predict(matrix, iteration_range=(0, int(best_iteration) + 1))
    return model.predict(matrix)


def main() -> int:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"No existe el dataset: {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR = OUTPUT_DIR / "models"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    metrics_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    training_rows: list[dict] = []

    print("=" * 78)
    print("XGBOOST — PML FORECAST 7D (168 MODELOS DIRECTOS)")
    print("=" * 78)
    print(f"Versión XGBoost: {xgb.__version__}")
    print(f"Parámetros: {json.dumps(XGB_PARAMS, ensure_ascii=False)}")
    print(f"Máximo de rondas: {NUM_BOOST_ROUND} | Early stopping: {EARLY_STOPPING_ROUNDS}")
    print("El conjunto TEST no participa en el entrenamiento ni en early stopping.\n")

    start_all = time.time()

    for horizon in HORIZONS:
        file_path = INPUT_DIR / f"horizon_{horizon:03d}.npz"
        if not file_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo: {file_path}")

        with np.load(file_path, allow_pickle=False) as data:
            train = load_npz_split(data, "train")
            validation = load_npz_split(data, "validation")
            test = load_npz_split(data, "test")

        dtrain = xgb.DMatrix(train["X"], label=train["y"])
        dvalidation = xgb.DMatrix(validation["X"], label=validation["y"])
        dtest = xgb.DMatrix(test["X"])

        evals_result: dict = {}
        model = xgb.train(
            params=XGB_PARAMS,
            dtrain=dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            evals=[(dtrain, "train"), (dvalidation, "validation")],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            evals_result=evals_result,
            verbose_eval=False,
        )

        # Predicciones: validación y prueba usando la mejor iteración seleccionada
        # exclusivamente mediante el conjunto de validación.
        pred_validation = predict_best(model, dvalidation)
        pred_test = predict_best(model, dtest)

        model_path = MODEL_DIR / f"xgboost_horizon_{horizon:03d}.json"
        model.save_model(str(model_path))

        best_iteration = int(getattr(model, "best_iteration", NUM_BOOST_ROUND - 1))
        best_score = float(getattr(model, "best_score", np.nan))

        for split_name, actual, predicted, weekly, persistence, origin_time, target_time in (
            ("validation", validation["y"], pred_validation, validation["weekly"],
             validation["persistence"], validation["origin_time"], validation["target_time"]),
            ("test", test["y"], pred_test, test["weekly"],
             test["persistence"], test["origin_time"], test["target_time"]),
        ):
            model_metrics = metrics(actual, predicted)
            metrics_rows.append({
                "horizon": horizon,
                "split": split_name,
                "model": "xgboost",
                "model_name": "XGBoost",
                "best_iteration": best_iteration,
                "best_validation_rmse": best_score,
                **model_metrics,
            })

            for baseline_key, baseline_name, baseline_prediction in (
                ("naive_week", "Naive semanal", weekly),
                ("naive_persistence", "Persistencia", persistence),
            ):
                baseline_metrics = metrics(actual, baseline_prediction)
                metrics_rows.append({
                    "horizon": horizon,
                    "split": split_name,
                    "model": baseline_key,
                    "model_name": baseline_name,
                    "best_iteration": np.nan,
                    "best_validation_rmse": np.nan,
                    **baseline_metrics,
                })

            prediction_frames.append(pd.DataFrame({
                "horizon": horizon,
                "split": split_name,
                "origin_time": np.asarray(origin_time).astype(str),
                "target_time": np.asarray(target_time).astype(str),
                "actual": actual,
                "xgboost_prediction": predicted,
                "naive_week_prediction": weekly,
                "naive_persistence_prediction": persistence,
                "xgboost_absolute_error": np.abs(actual - predicted),
                "naive_week_absolute_error": np.abs(actual - weekly),
                "naive_persistence_absolute_error": np.abs(actual - persistence),
            }))

        training_rows.append({
            "horizon": horizon,
            "train_rows": len(train["y"]),
            "validation_rows": len(validation["y"]),
            "test_rows": len(test["y"]),
            "best_iteration": best_iteration,
            "best_validation_rmse": best_score,
            "model_path": str(model_path.relative_to(PROJECT_ROOT)),
        })

        if horizon == 1 or horizon % 12 == 0 or horizon == 168:
            elapsed = time.time() - start_all
            print(
                f"Horizonte {horizon:03d}/168 | best_iteration={best_iteration} "
                f"| validation RMSE={best_score:.3f} | elapsed={elapsed / 60:.1f} min"
            )

    metrics_df = pd.DataFrame(metrics_rows)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    training_df = pd.DataFrame(training_rows)

    metrics_df.to_csv(OUTPUT_DIR / "xgboost_metrics_by_horizon.csv",
                      index=False, encoding="utf-8-sig")
    predictions_df.to_csv(OUTPUT_DIR / "xgboost_predictions.csv",
                          index=False, encoding="utf-8-sig")
    training_df.to_csv(OUTPUT_DIR / "xgboost_training_log.csv",
                       index=False, encoding="utf-8-sig")

    # Resumen pooled por split y modelo. Las filas entre horizontes pueden solaparse.
    summary_rows = []
    for split_name in ("validation", "test"):
        split_predictions = predictions_df[predictions_df["split"] == split_name]
        for column, model_key, model_name in (
            ("xgboost_prediction", "xgboost", "XGBoost"),
            ("naive_week_prediction", "naive_week", "Naive semanal"),
            ("naive_persistence_prediction", "naive_persistence", "Persistencia"),
        ):
            pooled = metrics(
                split_predictions["actual"].to_numpy(dtype=float),
                split_predictions[column].to_numpy(dtype=float),
            )
            summary_rows.append({
                "split": split_name,
                "model": model_key,
                "model_name": model_name,
                **pooled,
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_DIR / "xgboost_metrics_summary.csv",
                      index=False, encoding="utf-8-sig")

    # Gráficos por horizonte.
    for metric, title, ylabel, filename in (
        ("mae", "MAE por horizonte — XGBoost y baselines", "MAE (unidad de PML)", "01_mae_comparison.png"),
        ("rmse", "RMSE por horizonte — XGBoost y baselines", "RMSE (unidad de PML)", "02_rmse_comparison.png"),
        ("smape_pct", "sMAPE por horizonte — XGBoost y baselines", "sMAPE (%)", "03_smape_comparison.png"),
    ):
        fig, ax = plt.subplots(figsize=(13, 6))
        for split_name, linestyle in (("validation", "-"), ("test", "--")):
            for model_key, model_name in (
                ("xgboost", "XGBoost"),
                ("naive_week", "Naive semanal"),
                ("naive_persistence", "Persistencia"),
            ):
                subset = metrics_df[
                    (metrics_df["split"] == split_name)
                    & (metrics_df["model"] == model_key)
                ].sort_values("horizon")
                ax.plot(subset["horizon"], subset[metric], linestyle=linestyle,
                        linewidth=1.5, label=f"{model_name} — {split_name}")
        ax.set_title(title)
        ax.set_xlabel("Horizonte de predicción (horas)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(1, 168)
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / filename, dpi=160)
        plt.close(fig)

    report = [
        "PML FORECAST 7D — XGBOOST",
        "=" * 78,
        f"XGBoost version: {xgb.__version__}",
        f"Modelos entrenados: {len(training_df)}",
        f"Parámetros: {json.dumps(XGB_PARAMS, ensure_ascii=False)}",
        f"Num boost rounds máximo: {NUM_BOOST_ROUND}",
        f"Early stopping rounds: {EARLY_STOPPING_ROUNDS}",
        "",
        "Resumen pooled por split y modelo:",
        "(Las observaciones de diferentes horizontes pueden solaparse.)",
        "",
    ]
    for _, row in summary_df.iterrows():
        report.extend([
            f"{row['split'].upper()} — {row['model_name']}",
            f"  N: {int(row['n']):,}",
            f"  MAE: {row['mae']:.6f}",
            f"  RMSE: {row['rmse']:.6f}",
            f"  Mediana error absoluto: {row['median_absolute_error']:.6f}",
            f"  sMAPE: {row['smape_pct']:.6f}%",
            "",
        ])
    report.extend([
        "Notas:",
        "- Cada horizonte tiene un modelo independiente.",
        "- El conjunto de validación se usa para early stopping.",
        "- El conjunto test no se usa para ajustar parámetros ni seleccionar iteraciones.",
        "- sMAPE excluye casos donde real y predicho son ambos cero.",
        "- MAE y RMSE conservan precios negativos y valores extremos.",
    ])
    (OUTPUT_DIR / "xgboost_report.txt").write_text("\n".join(report), encoding="utf-8")

    print("\n" + "=" * 78)
    print("RESUMEN POOLED")
    print("=" * 78)
    print(summary_df[["split", "model_name", "n", "mae", "rmse", "smape_pct"]]
          .to_string(index=False, float_format=lambda value: f"{value:,.4f}"))

    print("\n" + "=" * 78)
    print("ENTRENAMIENTO FINALIZADO")
    print("=" * 78)
    print(f"Modelos guardados: {MODEL_DIR}")
    print(f"Resultados: {OUTPUT_DIR}")
    print("Archivos: xgboost_metrics_by_horizon.csv, xgboost_metrics_summary.csv,")
    print("          xgboost_predictions.csv, xgboost_training_log.csv,")
    print("          01_mae_comparison.png, 02_rmse_comparison.png,")
    print("          03_smape_comparison.png, xgboost_report.txt")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"\nERROR durante XGBoost: {exc}")
        raise
