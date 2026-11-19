
from pathlib import Path

import pandas as pd


# Locate the project folder, regardless of where the script is launched.
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "processed"

OUTPUT_DIR.mkdir(exist_ok=True)

KEYS = ["code_module", "code_presentation", "id_student"]

VLE_FILE = DATA_DIR / "studentVle.csv"
INFO_FILE = DATA_DIR / "studentInfo.csv"


def main():
    # 1. Check that the required input files exist.
    for file_path in [VLE_FILE, INFO_FILE]:
        if not file_path.exists():
            raise FileNotFoundError(f"Required file not found: {file_path}")

    print("Loading student VLE activity...")
    activity = pd.read_csv(VLE_FILE)

    print("Loading student information...")
    students = pd.read_csv(INFO_FILE)

    # 2. Validate the required columns.
    required_activity = KEYS + ["date", "sum_click"]
    required_students = KEYS + ["final_result"]

    missing_activity = set(required_activity) - set(activity.columns)
    missing_students = set(required_students) - set(students.columns)

    if missing_activity:
        raise ValueError(
            f"studentVle.csv is missing columns: {sorted(missing_activity)}"
        )

    if missing_students:
        raise ValueError(
            f"studentInfo.csv is missing columns: {sorted(missing_students)}"
        )

    # 3. Validate student-course records before merging.
    if students[KEYS].isna().any().any():
        raise ValueError("Student identifiers contain missing values.")

    if students.duplicated(KEYS).any():
        raise ValueError(
            "Duplicate student-course records found in studentInfo.csv. "
            "Inspect these records before proceeding."
        )

    if students["final_result"].isna().any():
        raise ValueError("Some student records have no final_result.")

    # 4. Validate and clean activity fields.
    for column in ["id_student", "date", "sum_click"]:
        activity[column] = pd.to_numeric(activity[column], errors="raise")

    if activity[KEYS].isna().any().any():
        raise ValueError("Activity records contain missing student identifiers.")

    if (activity["sum_click"] < 0).any():
        raise ValueError("Negative click counts found; inspect the source data.")

    # Negative dates represent activity before the course start.
    before_start = int((activity["date"] < 0).sum())
    activity = activity.loc[activity["date"] >= 0].copy()

    # A week starts at day 0: days 0-6 are week 1, days 7-13 week 2, etc.
    activity["week"] = (activity["date"] // 7 + 1).astype(int)

    # 5. Aggregate daily/site-level activity into weekly student activity.
    weekly = (
        activity.groupby(KEYS + ["week"], as_index=False)["sum_click"]
        .sum()
        .rename(columns={"sum_click": "weekly_clicks"})
    )

    weekly = weekly.sort_values(KEYS + ["week"])

    # Save all recorded weeks for sequential modelling in Unit 4.
    weekly.to_csv(OUTPUT_DIR / "weekly_activity.csv", index=False)

    # 6. Create one row per student-course, including students with no clicks.
    student_outcomes = students[KEYS + ["final_result"]].copy()

    first_four = weekly.loc[weekly["week"].between(1, 4)].copy()

    if first_four.empty:
        raise ValueError("No activity records were found in weeks 1-4.")

    early_wide = first_four.pivot(
        index=KEYS,
        columns="week",
        values="weekly_clicks",
    )

    early_wide = early_wide.reindex(columns=[1, 2, 3, 4])
    early_wide.columns = [f"week_{week}_clicks" for week in early_wide.columns]
    early_wide = early_wide.fillna(0).reset_index()

    early = student_outcomes.merge(
        early_wide,
        on=KEYS,
        how="left",
        validate="one_to_one",
    )

    feature_columns = [f"week_{week}_clicks" for week in range(1, 5)]
    early[feature_columns] = early[feature_columns].fillna(0).astype(int)

    # Helpful engagement features for clustering and later model comparisons.
    early["total_clicks"] = early[feature_columns].sum(axis=1)
    early["active_weeks"] = (early[feature_columns] > 0).sum(axis=1)
    early["peak_week_clicks"] = early[feature_columns].max(axis=1)
    early["week4_minus_week1"] = (
        early["week_4_clicks"] - early["week_1_clicks"]
    )

    early.to_csv(OUTPUT_DIR / "early_engagement_4weeks.csv", index=False)

    # 7. Print a summary so we can verify the outputs.
    print("\n--- DATA PREPARATION SUMMARY ---")
    print(f"Student VLE rows loaded: {len(activity) + before_start:,}")
    print(f"Rows excluded for negative dates: {before_start:,}")
    print(f"Weekly activity records: {len(weekly):,}")
    print(f"Unique student-course records: {len(student_outcomes):,}")
    print(f"First-four-week dataset shape: {early.shape}")
    print(f"Students with zero clicks in weeks 1-4: "f"{int((early['total_clicks'] == 0).sum()):,}")
    print("\nFinal-result counts:")
    print(early["final_result"].value_counts(dropna=False))
    print(f"\nSaved outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
