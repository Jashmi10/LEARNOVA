
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit


# --------------------------------------------------
# 1. PROJECT PATHS AND SETTINGS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "processed"
OUTPUT_DIR = PROJECT_DIR / "results" / "unit5"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = DATA_DIR / "early_engagement_4weeks.csv"
RANDOM_STATE = 42

TARGET = "final_result"
GROUP = "id_student"

CLICK_FEATURES = [
    "week_1_clicks",
    "week_2_clicks",
    "week_3_clicks",
    "week_4_clicks",
]

ENGINEERED_FEATURES = [
    "total_clicks",
    "active_weeks",
    "peak_week_clicks",
    "week4_minus_week1",
]

FEATURES = CLICK_FEATURES + ENGINEERED_FEATURES


# --------------------------------------------------
# 2. LOAD AND VALIDATE DATA
# --------------------------------------------------

def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Run python src/prepare_data.py first."
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = FEATURES + [TARGET, GROUP]
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing columns in the dataset: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    df = df.dropna(subset=FEATURES + [TARGET, GROUP]).copy()

    if df.empty:
        raise ValueError("No valid rows remain after removing missing values.")

    if (df[CLICK_FEATURES + ["total_clicks", "peak_week_clicks"]] < 0).any().any():
        raise ValueError("Click-count features must not contain negative values.")

    print("\n========== UNIT 5: ENSEMBLE LEARNING ==========")
    print(f"Dataset shape: {df.shape}")
    print(f"Number of records: {len(df):,}")
    print(f"Unique students: {df[GROUP].nunique():,}")

    print("\nFinal outcome distribution:")
    print(df[TARGET].value_counts())

    return df


# --------------------------------------------------
# 3. PREPARE FEATURES AND SPLIT THE DATA
# --------------------------------------------------

def prepare_data(df):
    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    groups = df[GROUP].copy()

    # Log-transform click counts to reduce the effect of extreme values.
    log_features = CLICK_FEATURES + [
        "total_clicks",
        "peak_week_clicks",
    ]

    X[log_features] = np.log1p(X[log_features])

    # Keep week4_minus_week1 unchanged because it can be negative.
    # Keep active_weeks unchanged because it is already a small count.

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    train_idx, test_idx = next(
        splitter.split(X, y, groups=groups)
    )

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    train_students = set(groups.iloc[train_idx])
    test_students = set(groups.iloc[test_idx])

    assert train_students.isdisjoint(test_students), (
        "Data leakage detected: a student appears in both splits."
    )

    print(f"\nTraining records: {len(X_train):,}")
    print(f"Testing records: {len(X_test):,}")
    print(f"Training students: {len(train_students):,}")
    print(f"Testing students: {len(test_students):,}")
    print("Student-level train/test overlap: 0")

    return X_train, X_test, y_train, y_test, test_idx


# --------------------------------------------------
# 4. TRAIN AND COMPARE ENSEMBLE MODELS
# --------------------------------------------------

def train_and_compare(df, X_train, X_test, y_train, y_test, test_idx):
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    comparison = []
    predictions_by_model = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        predictions_by_model[name] = predictions

        accuracy = accuracy_score(y_test, predictions)
        balanced_accuracy = balanced_accuracy_score(y_test, predictions)
        macro_f1 = f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        )

        comparison.append({
            "model": name,
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "macro_f1": macro_f1,
        })

        print(f"\n--- {name}: CLASSIFICATION REPORT ---")
        print(
            classification_report(
                y_test,
                predictions,
                labels=["Distinction", "Pass", "Fail", "Withdrawn"],
                zero_division=0,
            )
        )

        # Save the confusion matrix.
        labels = ["Distinction", "Pass", "Fail", "Withdrawn"]

        cm = confusion_matrix(
            y_test,
            predictions,
            labels=labels,
        )

        cm_df = pd.DataFrame(
            cm,
            index=[f"Actual_{label}" for label in labels],
            columns=[f"Predicted_{label}" for label in labels],
        )

        safe_name = name.lower().replace(" ", "_")
        cm_path = OUTPUT_DIR / f"{safe_name}_confusion_matrix.csv"
        cm_df.to_csv(cm_path)

        # Save feature importance.
        importance_df = pd.DataFrame({
            "feature": FEATURES,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)

        importance_df.to_csv(
            OUTPUT_DIR / f"{safe_name}_feature_importance.csv",
            index=False,
        )

    # Save predictions from both models.
    prediction_df = df.iloc[test_idx][
        ["code_module", "code_presentation", GROUP, TARGET]
    ].copy()

    prediction_df = prediction_df.rename(
        columns={TARGET: "actual_outcome"}
    )

    for name, predictions in predictions_by_model.items():
        prediction_df[
            f"{name.lower().replace(' ', '_')}_prediction"
        ] = predictions

    prediction_df.to_csv(
        OUTPUT_DIR / "test_predictions.csv",
        index=False,
    )

    # Compare model-level metrics.
    comparison_df = pd.DataFrame(comparison).sort_values(
        "macro_f1",
        ascending=False,
    )

    comparison_path = OUTPUT_DIR / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    print("\n========== MODEL COMPARISON ==========")
    print(comparison_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Select the best model by macro F1.
    best_model = comparison_df.iloc[0]

    print("\n========== BEST MODEL CONFIGURATION ==========")
    print(f"Best model: {best_model['model']}")
    print(f"Accuracy: {best_model['accuracy']:.4f}")
    print(f"Balanced accuracy: {best_model['balanced_accuracy']:.4f}")
    print(f"Macro F1-score: {best_model['macro_f1']:.4f}")
    print("Selection criterion: Highest macro F1-score")
    print("=============================================")


# --------------------------------------------------
# 5. MAIN
# --------------------------------------------------

def main():
    df = load_data()

    X_train, X_test, y_train, y_test, test_idx = prepare_data(df)

    train_and_compare(
        df,
        X_train,
        X_test,
        y_train,
        y_test,
        test_idx,
    )

    print(f"\nUnit 5 results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
