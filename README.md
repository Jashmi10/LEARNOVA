# LEARNOVA — Student Learning Analytics Using Machine Learning

## Overview

**LEARNOVA** is a machine learning project that analyses student engagement and academic performance using the [Open University Learning Analytics Dataset (OULAD)](https://analyse.kmi.open.ac.uk/open_dataset).

The project explores supervised learning, unsupervised clustering, sequential modelling, and ensemble learning to understand engagement patterns and predict academic outcomes from student learning activity.

## Objectives

- **Unit 1 — Academic Risk Classification:** Classify students as at risk (Fail/Withdrawn) or successful (Pass/Distinction) using Logistic Regression and RBF Support Vector Machine (SVM).
- **Unit 2 — Assessment Score Prediction:** Estimate assessment scores using Ridge Regression and Support Vector Regression (SVR).
- **Unit 3 — Student Engagement Clustering:** Identify engagement groups using K-Means, Gaussian Mixture Models (GMM), and Agglomerative Clustering.
- **Unit 4 — Sequential Engagement Modelling:** Analyse weekly engagement transitions using Markov models and Hidden Markov Models (HMMs).
- **Unit 5 — Ensemble-Based Outcome Prediction:** Predict Distinction, Pass, Fail, and Withdrawn outcomes using Random Forest and Extra Trees.

## Dataset

The project uses OULAD, which contains student demographic information, course registrations, assessment records, and virtual learning environment (VLE) interactions.

Early engagement features include weekly click counts, active weeks, total activity, and other activity-based measures. Different units use different observation windows according to their modelling objectives.

**Original dataset source:** https://analyse.kmi.open.ac.uk/open_dataset

### Dataset Setup Instructions

The raw dataset files are maintained separately in the [LEARNOVA VLE repository](https://github.com/Jashmi10/VLE). They are not included in this repository.

Before running the project, clone the VLE repository into a folder named `data` inside the LEARNOVA project directory.

#### Step 1 — Clone LEARNOVA

```bash
git clone https://github.com/Jashmi10/LEARNOVA.git
cd LEARNOVA
```

#### Step 2 — Download the dataset

From the LEARNOVA project root, run:

```bash
git clone https://github.com/Jashmi10/VLE.git data
```

This creates the `data/` directory and downloads the contents of the VLE repository into it.

#### Step 3 — Verify the directory structure

After downloading the dataset, the project should have the following structure:

```text
LEARNOVA/
├── data/
│   └── Dataset files from the VLE repository
├── processed/
│   ├── early_engagement_4weeks.csv
│   └── weekly_activity.csv
├── results/
│   ├── unit3/
│   ├── unit4/
│   └── unit5/
├── src/
│   ├── prepare_data.py
│   ├── unit3_clustering.py
│   ├── unit4_sequential.py
│   └── unit5_ensemble.py
├── requirements.txt
└── README.md
```

#### Step 4 — Set up the Python environment

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment using:

```bash
.venv\Scripts\activate
```

#### Step 5 — Run the project

Run the Python scripts from the LEARNOVA project root, following the execution order and file paths expected by the scripts.

Ensure that the required raw dataset files are present inside `data/` before running data preparation or modelling scripts.

**Important:** The `data/` directory is excluded from the LEARNOVA Git repository to avoid uploading large raw dataset files. Each user must obtain the dataset from the separate VLE repository before running the project.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- hmmlearn
- MLflow for Unit 4 sequential experiment tracking

## Methodology

### 1. Supervised Learning

Academic risk classification and assessment score regression are performed using early student activity and assessment-related features. Models are evaluated using classification and regression metrics.

### 2. Unsupervised Learning

K-Means, GMM, and Agglomerative Clustering are applied to the first four weeks of student engagement. The resulting clusters are evaluated using the Silhouette Score and Davies–Bouldin Index.

### 3. Sequential Modelling

Weekly engagement is represented using four observed categories: no clicks, low engagement, medium engagement, and high engagement. Markov models capture observed-state transitions, while HMMs investigate latent engagement patterns over 12 weeks.

### 4. Ensemble Learning

Random Forest and Extra Trees classifiers use the first four weeks of engagement and engineered activity features to predict final academic outcomes.

### 5. Experiment Tracking

MLflow is used to track Unit 4 sequential modelling experiments, including comparisons of HMM configurations with different numbers of hidden states.

## Results

### Unit 1 — Academic Risk Classification

| Model | Accuracy | F1-score | ROC-AUC |
|---|---:|---:|---:|
| RBF SVM | 0.7807 | 0.7901 | 0.8451 |

### Unit 2 — Assessment Score Prediction

| Model | RMSE | R² |
|---|---:|---:|
| Support Vector Regression (SVR) | 9.5821 | 0.6671 |

### Unit 3 — Student Engagement Clustering

| Algorithm | Clusters | Silhouette Score | Davies–Bouldin Index |
|---|---:|---:|---:|
| K-Means | 2 | 0.4369 | 0.9119 |
| GMM | 2 | 0.4351 | 0.9159 |
| Agglomerative Clustering | 2 | 0.4032 | 0.9695 |

K-Means achieved the highest Silhouette Score among the tested configurations.

### Unit 4 — Sequential Engagement Modelling

The HMM configurations were compared using average in-sample log-likelihood per week.

| Hidden States | Average Log-Likelihood per Week |
|---|---:|
| 2 | -1.104088 |
| 3 | -1.103750 |
| 4 | -1.003285 |

The four-hidden-state configuration achieved the highest in-sample log-likelihood among the tested models.

### Unit 5 — Ensemble-Based Outcome Prediction

| Model | Accuracy | Balanced Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Random Forest | 0.4465 | 0.3763 | 0.3780 |
| Extra Trees | 0.4743 | 0.3657 | 0.3567 |

Random Forest was selected based on its higher Macro-F1 score, while Extra Trees achieved higher overall accuracy.

## Key Findings

- Early engagement features provide useful information for academic risk classification and assessment score prediction.
- Clustering revealed two broad engagement groups with different activity levels and academic outcome distributions.
- Markov models and HMMs provide complementary ways to represent engagement transitions and latent patterns.
- Ensemble models demonstrate the challenges of predicting multiple academic outcome classes from limited early activity data.
- Evaluation metrics must be interpreted according to the modelling task and class distribution.

## Limitations

- Early engagement data may not capture all factors influencing academic outcomes.
- Minority outcome classes remain challenging to predict accurately.
- The HMM comparison reflects in-sample fit and does not establish generalisation performance.
- Predictive associations do not establish causation or demonstrate that interventions improve student performance.

## Future Work

- Explore additional engagement and assessment features.
- Improve prediction of minority outcome classes.
- Investigate longer engagement sequences and alternative temporal models.
- Add model explainability and visual analytics.
- Evaluate generalisation using independent datasets.

## References

1. Kuzilek, J., Hlosta, M., and Zdrahal, Z., “Open University Learning Analytics dataset,” *Scientific Data*, vol. 4, article 170171, 2017. https://doi.org/10.1038/sdata.2017.171
2. Open University Learning Analytics Dataset: https://analyse.kmi.open.ac.uk/open_dataset
3. Scikit-learn documentation: https://scikit-learn.org/
4. hmmlearn documentation: https://hmmlearn.readthedocs.io/

## Disclaimer

LEARNOVA is an academic machine learning project developed for educational and analytical purposes. Its predictions are exploratory and should not be treated as definitive assessments of individual students.
