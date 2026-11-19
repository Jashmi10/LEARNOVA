
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# 1. Paths and experiment setup
# --------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_DIR / "processed" / "early_engagement_4weeks.csv"
OUTPUT_DIR = PROJECT_DIR / "results" / "unit3"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

mlflow.set_tracking_uri(f"sqlite:///{PROJECT_DIR / 'mlflow.db'}")
mlflow.set_experiment("LEARNOVA_Unit3_Clustering")

RANDOM_STATE = 42
CLUSTER_COUNTS = [2, 3, 4, 5, 6]
HIERARCHICAL_SAMPLE_SIZE = 5000
EVALUATION_SIZE = 3000


def evaluate_clusters(X_eval, labels):
    """Calculate internal clustering quality metrics."""
    return {
        "silhouette": float(silhouette_score(X_eval, labels)),
        "davies_bouldin": float(davies_bouldin_score(X_eval, labels)),
        "calinski_harabasz": float(
            calinski_harabasz_score(X_eval, labels)
        ),
    }


def log_run(algorithm, k, metrics, extra_params=None):
    """Record one clustering experiment in MLflow."""
    with mlflow.start_run(run_name=f"{algorithm}_k{k}"):
        mlflow.set_tag("unit", "Unit 3")
        mlflow.set_tag("algorithm", algorithm)
        mlflow.log_param("n_clusters", k)
        mlflow.log_param("random_state", RANDOM_STATE)

        if extra_params:
            mlflow.log_params(extra_params)

        mlflow.log_metrics(metrics)

    print(
        f"{algorithm:15s} k={k} | "
        f"Silhouette={metrics['silhouette']:.4f} | "
        f"DB={metrics['davies_bouldin']:.4f} | "
        f"CH={metrics['calinski_harabasz']:.2f}"
    )


def main():
    # --------------------------------------------------
    # 2. Load and validate the prepared dataset
    # --------------------------------------------------
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"{INPUT_FILE} not found. Run prepare_data.py first."
        )

    df = pd.read_csv(INPUT_FILE)

    weekly_features = [
        "week_1_clicks",
        "week_2_clicks",
        "week_3_clicks",
        "week_4_clicks",
    ]

    required = weekly_features + ["final_result"]

    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Do not use final_result as a clustering input.
    # Log transformation reduces the influence of very high click counts.
    X_raw = df[weekly_features].astype(float)

    if not np.isfinite(X_raw.to_numpy()).all():
        raise ValueError("Non-finite values found in engagement features.")

    X_log = np.log1p(X_raw)

    # Standardise features so each week is on a comparable scale.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)

    print(f"Students loaded: {len(df):,}")
    print(f"Features used: {weekly_features}")
    print("Final outcome excluded from clustering inputs.")

    # Use a fixed evaluation sample shared by all three algorithms.
    rng = np.random.default_rng(RANDOM_STATE)
    sample_size = min(HIERARCHICAL_SAMPLE_SIZE, len(df))
    sample_indices = np.sort(
        rng.choice(len(df), size=sample_size, replace=False)
    )

    eval_size = min(EVALUATION_SIZE, sample_size)
    eval_indices = sample_indices[:eval_size]

    X_eval = X_scaled[eval_indices]
    X_hier = X_scaled[sample_indices]

    results = []
    best_models = {}

    # --------------------------------------------------
    # 3. K-Means: fit on all students
    # --------------------------------------------------
    best_models["KMeans"] = {"silhouette": -1}

    for k in CLUSTER_COUNTS:
        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=10,
        )
        model.fit(X_scaled)

        eval_labels = model.predict(X_eval)
        metrics = evaluate_clusters(X_eval, eval_labels)

        log_run(
            "KMeans",
            k,
            metrics,
            {"n_init": 10, "input_rows": len(df)},
        )

        results.append({"algorithm": "KMeans", "k": k, **metrics})

        if metrics["silhouette"] > best_models["KMeans"]["silhouette"]:
            best_models["KMeans"] = {
                "silhouette": metrics["silhouette"],
                "k": k,
                "labels": model.labels_.copy(),
            }

    # --------------------------------------------------
    # 4. Gaussian Mixture Model: fit on all students
    # --------------------------------------------------
    best_models["GMM"] = {"silhouette": -1}

    for k in CLUSTER_COUNTS:
        model = GaussianMixture(
            n_components=k,
            covariance_type="diag",
            random_state=RANDOM_STATE,
            n_init=2,
            max_iter=200,
        )
        model.fit(X_scaled)

        eval_labels = model.predict(X_eval)
        metrics = evaluate_clusters(X_eval, eval_labels)

        log_run(
            "GMM",
            k,
            metrics,
            {
                "covariance_type": "diag",
                "n_init": 2,
                "max_iter": 200,
                "input_rows": len(df),
            },
        )

        results.append({"algorithm": "GMM", "k": k, **metrics})

        if metrics["silhouette"] > best_models["GMM"]["silhouette"]:
            best_models["GMM"] = {
                "silhouette": metrics["silhouette"],
                "k": k,
                "labels": model.predict(X_scaled),
            }

    # --------------------------------------------------
    # 5. Hierarchical clustering: fit on a 5,000-row sample
    # --------------------------------------------------
    best_models["Hierarchical"] = {"silhouette": -1}

    for k in CLUSTER_COUNTS:
        model = AgglomerativeClustering(
            n_clusters=k,
            linkage="ward",
        )
        sample_labels = model.fit_predict(X_hier)

        # Evaluate the same evaluation rows used by K-Means and GMM.
        eval_labels = sample_labels[:eval_size]
        metrics = evaluate_clusters(X_eval, eval_labels)

        log_run(
            "Hierarchical",
            k,
            metrics,
            {
                "linkage": "ward",
                "fit_rows": len(sample_indices),
                "evaluation_rows": eval_size,
            },
        )

        results.append(
            {"algorithm": "Hierarchical", "k": k, **metrics}
        )

        if metrics["silhouette"] > best_models["Hierarchical"]["silhouette"]:
            best_models["Hierarchical"] = {
                "silhouette": metrics["silhouette"],
                "k": k,
                "labels": sample_labels.copy(),
            }

    # --------------------------------------------------
    # 6. Save experiment comparison
    # --------------------------------------------------
    results_df = pd.DataFrame(results).sort_values(
        "silhouette", ascending=False
    )

    results_path = OUTPUT_DIR / "clustering_comparison.csv"
    results_df.to_csv(results_path, index=False)

    print("\n--- CLUSTERING COMPARISON ---")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # --------------------------------------------------
    # 7. Save best cluster assignments and profiles
    # --------------------------------------------------
    id_columns = [
        "code_module",
        "code_presentation",
        "id_student",
    ]

    for algorithm, best in best_models.items():
        if algorithm == "Hierarchical":
            output_df = df.iloc[sample_indices][id_columns].copy()
        else:
            output_df = df[id_columns].copy()

        output_df["cluster"] = best["labels"]

        # Outcome is used only after clustering to interpret the groups.
        output_df["final_result"] = df.iloc[
            sample_indices if algorithm == "Hierarchical"
            else np.arange(len(df))
        ]["final_result"].to_numpy()

        output_path = OUTPUT_DIR / f"{algorithm.lower()}_best_clusters.csv"
        output_df.to_csv(output_path, index=False)

        profile = (
            output_df.groupby("cluster")["final_result"]
            .value_counts(normalize=True)
            .rename("proportion")
            .reset_index()
        )
        profile.to_csv(
            OUTPUT_DIR / f"{algorithm.lower()}_outcome_profile.csv",
            index=False,
        )

        print(
            f"\nBest {algorithm}: k={best['k']}, "
            f"Silhouette={best['silhouette']:.4f}"
        )

    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("MLflow experiment: LEARNOVA_Unit3_Clustering")


if __name__ == "__main__":
    main()
