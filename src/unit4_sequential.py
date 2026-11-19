
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from hmmlearn.hmm import CategoricalHMM


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "processed"
OUTPUT_DIR = PROJECT_DIR / "results" / "unit4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

KEYS = ["code_module", "code_presentation", "id_student"]
N_WEEKS = 12
N_ENGAGEMENT_STATES = 4
HMM_SAMPLE_SIZE = 8000
RANDOM_STATE = 42

mlflow.set_tracking_uri(f"sqlite:///{PROJECT_DIR / 'mlflow.db'}")
mlflow.set_experiment("LEARNOVA_Unit4_Sequential")


def prepare_sequences():
    """Create complete weekly sequences, filling missing activity with zero."""
    weekly_path = DATA_DIR / "weekly_activity.csv"
    early_path = DATA_DIR / "early_engagement_4weeks.csv"

    if not weekly_path.exists() or not early_path.exists():
        raise FileNotFoundError(
            "Processed data is missing. Run src/prepare_data.py first."
        )

    weekly = pd.read_csv(weekly_path)
    students = pd.read_csv(early_path)[KEYS + ["final_result"]]

    weekly = weekly[weekly["week"].between(1, N_WEEKS)].copy()

    wide = weekly.pivot(
        index=KEYS,
        columns="week",
        values="weekly_clicks",
    )

    # Include every student-course record, even those with no recorded clicks.
    full_index = pd.MultiIndex.from_frame(students[KEYS])
    wide = wide.reindex(
        index=full_index,
        columns=range(1, N_WEEKS + 1),
    ).fillna(0)

    wide.index.names = KEYS
    wide = wide.reset_index()

    click_cols = []
    for week in range(1, N_WEEKS + 1):
        column = f"week_{week}_clicks"
        wide = wide.rename(columns={week: column})
        click_cols.append(column)

    # Observable engagement states:
    # 0 = no clicks, 1 = low, 2 = medium, 3 = high.
    values = wide[click_cols].to_numpy(dtype=float)
    positive = values[values > 0]

    if len(positive) == 0:
        raise ValueError("No positive weekly click counts were found.")

    q33, q66 = np.quantile(positive, [1 / 3, 2 / 3])

    states = np.zeros(values.shape, dtype=int)
    states[(values > 0) & (values <= q33)] = 1
    states[(values > q33) & (values <= q66)] = 2
    states[values > q66] = 3

    state_cols = []
    for week in range(1, N_WEEKS + 1):
        column = f"week_{week}_state"
        wide[column] = states[:, week - 1]
        state_cols.append(column)

    sequences = wide.merge(
        students,
        on=KEYS,
        how="left",
        validate="one_to_one",
    )

    sequences.to_csv(
        OUTPUT_DIR / "weekly_engagement_states.csv",
        index=False,
    )

    print(f"Student-course sequences: {len(sequences):,}")
    print(f"Weeks per sequence: {N_WEEKS}")
    print(f"Positive-click state thresholds: {q33:.1f}, {q66:.1f}")
    print("State meanings: 0=no clicks, 1=low, 2=medium, 3=high")

    return sequences, state_cols, q33, q66


def run_markov_chain(sequences, state_cols, q33, q66):
    """Estimate week-to-week transition probabilities."""
    matrix = sequences[state_cols].to_numpy(dtype=int)

    counts = np.zeros(
        (N_ENGAGEMENT_STATES, N_ENGAGEMENT_STATES),
        dtype=int,
    )

    for week in range(N_WEEKS - 1):
        np.add.at(
            counts,
            (matrix[:, week], matrix[:, week + 1]),
            1,
        )

    row_totals = counts.sum(axis=1, keepdims=True)

    probabilities = np.divide(
        counts,
        row_totals,
        out=np.zeros_like(counts, dtype=float),
        where=row_totals != 0,
    )

    labels = ["no_clicks", "low", "medium", "high"]

    transition_df = pd.DataFrame(
        probabilities,
        index=labels,
        columns=labels,
    )

    counts_df = pd.DataFrame(
        counts,
        index=labels,
        columns=labels,
    )

    transition_path = OUTPUT_DIR / "markov_transition_matrix.csv"
    counts_path = OUTPUT_DIR / "markov_transition_counts.csv"

    transition_df.to_csv(transition_path)
    counts_df.to_csv(counts_path)

    with mlflow.start_run(run_name="Markov_Chain_12_weeks"):
        mlflow.set_tag("unit", "Unit 4")
        mlflow.set_tag("algorithm", "Markov Chain")

        mlflow.log_params({
            "weeks": N_WEEKS,
            "number_of_observed_states": N_ENGAGEMENT_STATES,
            "state_0": "no recorded clicks",
            "low_click_threshold": float(q33),
            "medium_click_threshold": float(q66),
            "students": len(sequences),
        })

        mlflow.log_metric(
            "probability_no_clicks_next_week",
            float(probabilities[0, 0]),
        )
        mlflow.log_metric(
            "probability_low_to_no_clicks",
            float(probabilities[1, 0]),
        )
        mlflow.log_metric(
            "probability_high_to_no_clicks",
            float(probabilities[3, 0]),
        )

        mlflow.log_artifact(str(transition_path))
        mlflow.log_artifact(str(counts_path))

    print("\n--- MARKOV TRANSITION PROBABILITIES ---")
    print(transition_df.round(3))


def run_hmm(sequences, state_cols):
    """Compare HMMs with different numbers of hidden states."""
    sample_size = min(HMM_SAMPLE_SIZE, len(sequences))

    sample = sequences.sample(
        n=sample_size,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    observed = sample[state_cols].to_numpy(dtype=int)
    X = observed.reshape(-1, 1)
    lengths = [N_WEEKS] * len(sample)

    comparison = []

    for n_hidden in [2, 3, 4]:
        print(f"\nTraining HMM with {n_hidden} hidden states...")

        model = CategoricalHMM(
            n_components=n_hidden,
            n_features=N_ENGAGEMENT_STATES,
            n_iter=100,
            random_state=RANDOM_STATE,
            verbose=False,
        )

        model.fit(X, lengths=lengths)

        total_log_likelihood = model.score(X, lengths=lengths)
        average_log_likelihood = total_log_likelihood / len(X)

        # Decode the most likely hidden-state sequence for each student.
        hidden = model.predict(X, lengths=lengths)
        hidden_matrix = hidden.reshape(sample_size, N_WEEKS)

        output = sample[KEYS + ["final_result"]].copy()

        for week in range(N_WEEKS):
            output[f"hidden_state_week_{week + 1}"] = (
                hidden_matrix[:, week]
            )

        hidden_path = (
            OUTPUT_DIR / f"hmm_hidden_states_{n_hidden}.csv"
        )
        output.to_csv(hidden_path, index=False)

        with mlflow.start_run(
            run_name=f"HMM_{n_hidden}_hidden_states"
        ):
            mlflow.set_tag("unit", "Unit 4")
            mlflow.set_tag("algorithm", "Hidden Markov Model")

            mlflow.log_params({
                "hidden_states": n_hidden,
                "observation_states": N_ENGAGEMENT_STATES,
                "weeks": N_WEEKS,
                "training_students": sample_size,
                "iterations": int(model.monitor_.iter),
                "converged": bool(model.monitor_.converged),
                "random_state": RANDOM_STATE,
            })

            mlflow.log_metric(
                "average_log_likelihood_per_week",
                float(average_log_likelihood),
            )
            mlflow.log_artifact(str(hidden_path))

        comparison.append({
            "hidden_states": n_hidden,
            "average_log_likelihood_per_week": average_log_likelihood,
            "iterations": model.monitor_.iter,
            "converged": model.monitor_.converged,
        })

        print(
            "Average log-likelihood per week: "
            f"{average_log_likelihood:.4f}"
        )
        print(f"Iterations: {model.monitor_.iter}")
        print(f"Converged: {model.monitor_.converged}")

    # Save and display the comparison of all three HMM configurations.
    comparison_df = pd.DataFrame(comparison)

    comparison_path = OUTPUT_DIR / "hmm_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    print("\n--- HMM COMPARISON ---")
    print(comparison_df.to_string(index=False))

    # Select the best configuration using the highest log-likelihood.
    best_hmm = comparison_df.loc[
        comparison_df["average_log_likelihood_per_week"].idxmax()
    ]

    print("\n========== BEST HMM CONFIGURATION ==========")
    print(f"Best number of hidden states: {int(best_hmm['hidden_states'])}")
    print(
        "Average log-likelihood per week: "
        f"{best_hmm['average_log_likelihood_per_week']:.4f}"
    )
    print(f"Iterations: {int(best_hmm['iterations'])}")
    print(f"Converged: {best_hmm['converged']}")
    print("============================================")


def main():
    sequences, state_cols, q33, q66 = prepare_sequences()

    run_markov_chain(sequences, state_cols, q33, q66)
    run_hmm(sequences, state_cols)

    print(f"\nUnit 4 outputs saved to: {OUTPUT_DIR}")
    print("MLflow experiment: LEARNOVA_Unit4_Sequential")


if __name__ == "__main__":
    main()
