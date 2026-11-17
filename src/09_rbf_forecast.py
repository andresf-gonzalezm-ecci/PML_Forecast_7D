"""
09_rbf_forecast.py

Red neuronal RBF (Radial Basis Function) con 168 modelos directos,
uno por horizonte horario.

Metodología:
- Estandarización de las 21 variables usando SOLO el conjunto train.
- Centros RBF obtenidos con MiniBatchKMeans sobre train.
- Función gaussiana sobre distancias a los centros.
- Capa de salida lineal con regularización Ridge.
- Selección de alpha usando RMSE de VALIDATION.
- TEST se utiliza exclusivamente para evaluación final.

Ejecución:
    python .\src\09_rbf_forecast.py

Dependencia:
    python -m pip install scikit-learn
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
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import pairwise_distances
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print("ERROR: scikit-learn no está instalado en este entorno.")
    print("Instálalo con: python -m pip install scikit-learn")
    raise


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "reports" / "modeling_dataset"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "rbf"

HORIZONS = range(1, 169)

# Configuración inicial controlada.
N_CENTERS = 128
KMEANS_BATCH_SIZE = 2048
KMEANS_MAX_ITER = 300
KMEANS_N_INIT = 3
RANDOM_STATE = 42
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]


def load_split(data: np.lib.npyio.NpzFile, split: str) -> dict[str, np.ndarray]:
    fields = {"X": f"X_{split}", "y": f"y_{split}"}
    if split in ("validation", "test"):
        fields.update({
            "weekly": f"naive_week_{split}",
            "persistence": f"naive_persistence_{split}",
            "origin_time": f"origin_{split}",
            "target_time": f"target_{split}",
        })

    result = {key: np.asarray(data[field]) for key, field in fields.items()}
    if result["X"].ndim != 2 or result["X"].shape[0] != len(result["y"]):
        raise ValueError(f"Dimensiones inválidas en split {split}.")

    result["X"] = result["X"].astype(np.float64)
    result["y"] = result["y"].astype(np.float64)

    if split in ("validation", "test"):
        for key in ("weekly", "persistence", "origin_time", "target_time"):
            if len(result[key]) != len(result["y"]):
                raise ValueError(f"Longitud inconsistente: {key}")
        result["weekly"] = result["weekly"].astype(float)
        result["persistence"] = result["persistence"].astype(float)
    return result


def gaussian_rbf(X: np.ndarray, centers: np.ndarray, gamma: float) -> np.ndarray:
    distances_sq = pairwise_distances(X, centers, metric="sqeuclidean")
    return np.exp(-gamma * distances_sq)


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(actual) & np.isfinite(predicted)
    y = actual[valid].astype(float)
    yhat = predicted[valid].astype(float)
    if y.size == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan,
                "median_absolute_error": np.nan, "smape_pct": np.nan,
                "smape_valid_n": 0}

    abs_error = np.abs(y - yhat)
    denom = np.abs(y) + np.abs(yhat)
    smape_mask = denom > 0
    smape = 200.0 * abs_error[smape_mask] / denom[smape_mask]
    return {
        "n": int(y.size),
        "mae": float(abs_error.mean()),
        "rmse": float(np.sqrt(np.mean((y - yhat) ** 2))),
        "median_absolute_error": float(np.median(abs_error)),
        "smape_pct": float(smape.mean()) if smape.size else np.nan,
        "smape_valid_n": int(smape.size),
    }


def fit_ridge_closed_form(Phi: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Ridge con intercepto no penalizado."""
    design = np.column_stack([np.ones(Phi.shape[0]), Phi])
    penalty = np.eye(design.shape[1], dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    lhs = design.T @ design + penalty
    rhs = design.T @ y
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(lhs, rhs, rcond=None)[0]


def predict_ridge(Phi: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return coefficients[0] + Phi @ coefficients[1:]


def main() -> int:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"No existe el dataset: {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR = OUTPUT_DIR / "models"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    training_rows: list[dict] = []

    print("=" * 78)
    print("RBF NEURAL NETWORK — PML FORECAST 7D (168 MODELOS DIRECTOS)")
    print("=" * 78)
    print(f"Centros RBF: {N_CENTERS}")
    print(f"Alphas Ridge: {RIDGE_ALPHAS}")
    print("Escalado y centros se ajustan exclusivamente con TRAIN.")
    print("VALIDATION selecciona alpha; TEST queda reservado para evaluación final.\n")

    start_all = time.time()

    for horizon in HORIZONS:
        file_path = INPUT_DIR / f"horizon_{horizon:03d}.npz"
        if not file_path.exists():
            raise FileNotFoundError(f"No se encontró: {file_path}")

        with np.load(file_path, allow_pickle=False) as data:
            train = load_split(data, "train")
            validation = load_split(data, "validation")
            test = load_split(data, "test")

        if len(train["y"]) < N_CENTERS:
            raise ValueError(f"H{horizon}: no hay suficientes filas para {N_CENTERS} centros.")

        scaler = StandardScaler()
        X_train = scaler.fit_transform(train["X"])
        X_validation = scaler.transform(validation["X"])
        X_test = scaler.transform(test["X"])

        kmeans = MiniBatchKMeans(
            n_clusters=N_CENTERS,
            random_state=RANDOM_STATE,
            batch_size=KMEANS_BATCH_SIZE,
            max_iter=KMEANS_MAX_ITER,
            n_init=KMEANS_N_INIT,
            reassignment_ratio=0.01,
        )
        kmeans.fit(X_train)
        centers = kmeans.cluster_centers_

        # Escala gaussiana basada en la mediana de la distancia al centro más cercano
        # entre los propios centros. Evita depender de datos de validación/test.
        center_dist_sq = pairwise_distances(centers, centers, metric="sqeuclidean")
        np.fill_diagonal(center_dist_sq, np.inf)
        nearest_center_dist = np.sqrt(np.min(center_dist_sq, axis=1))
        sigma = float(np.median(nearest_center_dist))
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = 1.0
        gamma = 1.0 / (2.0 * sigma ** 2)

        Phi_train = gaussian_rbf(X_train, centers, gamma)
        Phi_validation = gaussian_rbf(X_validation, centers, gamma)
        Phi_test = gaussian_rbf(X_test, centers, gamma)

        # Selección de regularización exclusivamente con validation RMSE.
        alpha_results = []
        best = None
        for alpha in RIDGE_ALPHAS:
            coefficients = fit_ridge_closed_form(Phi_train, train["y"], alpha)
            pred_val = predict_ridge(Phi_validation, coefficients)
            val_rmse = calculate_metrics(validation["y"], pred_val)["rmse"]
            alpha_results.append({"alpha": alpha, "validation_rmse": val_rmse})
            if best is None or val_rmse < best["validation_rmse"]:
                best = {
                    "alpha": alpha,
                    "validation_rmse": val_rmse,
                    "coefficients": coefficients,
                }

        assert best is not None
        pred_validation = predict_ridge(Phi_validation, best["coefficients"])
        pred_test = predict_ridge(Phi_test, best["coefficients"])

        # Guardar parámetros del modelo de forma reproducible.
        model_path = MODEL_DIR / f"rbf_horizon_{horizon:03d}.npz"
        np.savez_compressed(
            model_path,
            scaler_mean=scaler.mean_,
            scaler_scale=scaler.scale_,
            centers=centers,
            gamma=np.asarray(gamma),
            sigma=np.asarray(sigma),
            alpha=np.asarray(best["alpha"]),
            coefficients=best["coefficients"],
        )

        for split_name, split_data, predicted in (
            ("validation", validation, pred_validation),
            ("test", test, pred_test),
        ):
            for model_key, model_name, baseline_pred in (
                ("rbf", "RBF", predicted),
                ("naive_week", "Naive semanal", split_data["weekly"]),
                ("naive_persistence", "Persistencia", split_data["persistence"]),
            ):
                metric_rows.append({
                    "horizon": horizon,
                    "split": split_name,
                    "model": model_key,
                    "model_name": model_name,
                    "selected_alpha": best["alpha"] if model_key == "rbf" else np.nan,
                    **calculate_metrics(split_data["y"], baseline_pred),
                })

            prediction_frames.append(pd.DataFrame({
                "horizon": horizon,
                "split": split_name,
                "origin_time": np.asarray(split_data["origin_time"]).astype(str),
                "target_time": np.asarray(split_data["target_time"]).astype(str),
                "actual": split_data["y"],
                "rbf_prediction": predicted,
                "naive_week_prediction": split_data["weekly"],
                "naive_persistence_prediction": split_data["persistence"],
                "rbf_absolute_error": np.abs(split_data["y"] - predicted),
            }))

        training_rows.append({
            "horizon": horizon,
            "train_rows": len(train["y"]),
            "validation_rows": len(validation["y"]),
            "test_rows": len(test["y"]),
            "n_centers": N_CENTERS,
            "sigma": sigma,
            "gamma": gamma,
            "selected_alpha": best["alpha"],
            "best_validation_rmse": best["validation_rmse"],
            "kmeans_inertia": float(kmeans.inertia_),
            "model_path": str(model_path.relative_to(PROJECT_ROOT)),
            "alpha_validation_results": json.dumps(alpha_results),
        })

        if horizon == 1 or horizon % 12 == 0 or horizon == 168:
            elapsed = time.time() - start_all
            print(
                f"Horizonte {horizon:03d}/168 | alpha={best['alpha']:g} "
                f"| validation RMSE={best['validation_rmse']:.3f} "
                f"| elapsed={elapsed / 60:.1f} min"
            )

    metrics_df = pd.DataFrame(metric_rows)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    training_df = pd.DataFrame(training_rows)

    metrics_df.to_csv(OUTPUT_DIR / "rbf_metrics_by_horizon.csv",
                      index=False, encoding="utf-8-sig")
    predictions_df.to_csv(OUTPUT_DIR / "rbf_predictions.csv",
                          index=False, encoding="utf-8-sig")
    training_df.to_csv(OUTPUT_DIR / "rbf_training_log.csv",
                       index=False, encoding="utf-8-sig")

    summary_rows = []
    for split_name in ("validation", "test"):
        subset = predictions_df[predictions_df["split"] == split_name]
        for prediction_col, model_key, model_name in (
            ("rbf_prediction", "rbf", "RBF"),
            ("naive_week_prediction", "naive_week", "Naive semanal"),
            ("naive_persistence_prediction", "naive_persistence", "Persistencia"),
        ):
            summary_rows.append({
                "split": split_name,
                "model": model_key,
                "model_name": model_name,
                **calculate_metrics(
                    subset["actual"].to_numpy(dtype=float),
                    subset[prediction_col].to_numpy(dtype=float),
                ),
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_DIR / "rbf_metrics_summary.csv",
                      index=False, encoding="utf-8-sig")

    for metric, title, ylabel, filename in (
        ("mae", "MAE por horizonte — RBF y baselines", "MAE (unidad de PML)", "01_mae_comparison.png"),
        ("rmse", "RMSE por horizonte — RBF y baselines", "RMSE (unidad de PML)", "02_rmse_comparison.png"),
        ("smape_pct", "sMAPE por horizonte — RBF y baselines", "sMAPE (%)", "03_smape_comparison.png"),
    ):
        fig, ax = plt.subplots(figsize=(13, 6))
        for split_name, linestyle in (("validation", "-"), ("test", "--")):
            for model_key, model_name in (
                ("rbf", "RBF"),
                ("naive_week", "Naive semanal"),
                ("naive_persistence", "Persistencia"),
            ):
                sub = metrics_df[
                    (metrics_df["split"] == split_name)
                    & (metrics_df["model"] == model_key)
                ].sort_values("horizon")
                ax.plot(sub["horizon"], sub[metric], linestyle=linestyle,
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
        "PML FORECAST 7D — RBF",
        "=" * 78,
        f"Centros RBF: {N_CENTERS}",
        f"Alphas candidatos: {RIDGE_ALPHAS}",
        "Escalador y centros ajustados solo con train.",
        "Alpha seleccionado con validation RMSE.",
        "Test reservado para evaluación.",
        "",
        "Resumen pooled (los horizontes pueden solaparse):",
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
    (OUTPUT_DIR / "rbf_report.txt").write_text("\n".join(report), encoding="utf-8")

    print("\n" + "=" * 78)
    print("RESUMEN POOLED")
    print("=" * 78)
    print(summary_df[["split", "model_name", "n", "mae", "rmse", "smape_pct"]]
          .to_string(index=False, float_format=lambda value: f"{value:,.4f}"))
    print("\nENTRENAMIENTO RBF FINALIZADO")
    print(f"Modelos guardados: {MODEL_DIR}")
    print(f"Resultados: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"\nERROR durante RBF: {exc}")
        raise
