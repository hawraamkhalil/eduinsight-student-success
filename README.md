# EduInsight: Student Engagement and Early-Support Analytics

[![CI](https://github.com/hawraamkhalil/eduinsight-student-success/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/hawraamkhalil/eduinsight-student-success/actions/workflows/ci.yml)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://eduinsight-student-success.streamlit.app/)
[![GitHub Release](https://img.shields.io/github/v/release/hawraamkhalil/eduinsight-student-success?display_name=tag)](https://github.com/hawraamkhalil/eduinsight-student-success/releases/latest)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

EduInsight is an end-to-end data science portfolio project that explores how student activity observed during the first 30 days of a course is associated with later academic outcomes.

The project demonstrates a complete workflow covering multi-table data preparation, exploratory analysis, leakage-aware feature engineering, machine-learning model development, automated testing, continuous integration, model persistence, and deployment through a multipage Streamlit application.

> **Release status:** Version 1.0.0 — initial portfolio release.

## Application preview

### Project overview
![EduInsight home dashboard](images/home-dashboard.png)
### Engagement analysis
![EduInsight engagement analysis](images/engagement-analysis.png)
### Early-support demonstration
![EduInsight early-support demonstration](images/early-support-demo.png)

## Live demonstration

The deployed Streamlit application is available through the **Streamlit App** badge above.

The application contains:

* a project and dataset overview;
* interactive student-engagement analysis;
* final-outcome comparisons;
* a hypothetical early-support model demonstration;
* clear ethical and usage limitations.

> The application is a portfolio and learning-analytics demonstration. It must not be used as the sole basis for real academic decisions.

## Research question

**Can information available during the first 30 course days help identify student profiles that may benefit from additional academic support?**

The project does not attempt to automatically judge students or replace educators. Its purpose is to demonstrate how early activity data can be prepared, analysed, modelled, evaluated, and presented responsibly.

## Why this project

EduInsight was designed to demonstrate more than a single machine-learning notebook.

It showcases:

* multi-table data preparation;
* validated database-style joins;
* memory-conscious processing of a large activity dataset;
* reusable and testable Python modules;
* leakage-aware feature engineering;
* exploratory data analysis and interpretation;
* class-imbalance-aware model evaluation;
* cross-validated model comparison;
* final evaluation on an untouched holdout set;
* model persistence with metadata;
* a multipage Streamlit application;
* automated testing with pytest;
* continuous integration with GitHub Actions;
* professional project documentation and release management.

## Project workflow

```text
OULAD raw CSV tables
        ↓
File, schema, and key validation
        ↓
Course-presentation selection
        ↓
Day-30 eligibility filtering
        ↓
Early engagement feature engineering
        ↓
Early assessment feature engineering
        ↓
Student-level processed dataset
        ↓
Exploratory data analysis
        ↓
Training and holdout split
        ↓
Cross-validated model comparison
        ↓
Untouched holdout evaluation
        ↓
Persisted model pipeline
        ↓
Multipage Streamlit application
```

## Dataset

This project uses the **Open University Learning Analytics Dataset**, commonly known as **OULAD**.

OULAD contains linked information about:

* courses and course presentations;
* student registrations;
* student demographic and study information;
* assessments and deadlines;
* student assessment submissions;
* virtual learning environment resources;
* student interactions with online learning resources.

The original raw files are not committed to this repository. They must be downloaded from the official source and placed in:

```text
data/raw/
```

Expected files:

```text
courses.csv
assessments.csv
studentAssessment.csv
studentInfo.csv
studentRegistration.csv
studentVle.csv
vle.csv
```

The large `studentVle.csv` file is processed in chunks to reduce memory consumption.

### Dataset citation

Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2015).
*Open University Learning Analytics Dataset*.
UCI Machine Learning Repository.
https://doi.org/10.24432/C5KK69

Dataset licence: **Creative Commons Attribution 4.0 International — CC BY 4.0**.

## Experiment scope

The first project release focuses on one course presentation:

| Configuration              | Value              |
| -------------------------- | ------------------ |
| Module                     | `BBB`              |
| Presentation               | `2013J`            |
| Observation window         | Course days `0–30` |
| Prepared modelling records | `1,825`            |
| Training records           | `1,460`            |
| Holdout records            | `365`              |
| Holdout percentage         | `20%`              |
| Random state               | `42`               |

Focusing on one course presentation creates a controlled first experiment. It does not imply that the trained model will generalise to other courses, institutions, or time periods.

## Target definition

The binary modelling target is:

```text
0 = Pass or Distinction
1 = Fail or Withdrawn
```

In the application, class `1` is described as the **support flag**.

The term “support flag” is used deliberately. It represents a modelling category, not a diagnosis, permanent label, or automatic academic decision.

## Day-30 eligibility rules

The model is intended to simulate an estimate made on course day 30.

A student is included only when:

* the student belongs to the selected module and presentation;
* the student registered by day 30;
* the student had not withdrawn on or before day 30.

Students who withdrew on or before day 30 are excluded because their withdrawal was already known at the intended prediction time.

Students who withdrew after day 30 remain eligible because their later outcome is part of what the experiment attempts to identify.

## Leakage controls

The project includes explicit controls against data leakage.

The modelling workflow:

* uses only virtual-learning activity from days 0 through 30;
* uses only assessments submitted by day 30;
* excludes withdrawals already known by day 30;
* excludes future withdrawal dates from model inputs;
* does not use `final_result` as an input feature;
* creates the holdout set before model comparison;
* performs model selection only through cross-validation on the training partition;
* evaluates the selected model once on the untouched holdout partition;
* keeps imputation and preprocessing inside scikit-learn pipelines.

These controls help ensure that model evaluation more closely represents the intended day-30 scenario.

## Engineered features

The model uses the following features:

| Feature                        | Description                                                      |
| ------------------------------ | ---------------------------------------------------------------- |
| `num_of_prev_attempts`         | Number of previous attempts associated with the student record   |
| `studied_credits`              | Number of credits being studied                                  |
| `date_registration`            | Registration date relative to the course start                   |
| `total_clicks_30`              | Total virtual-learning-environment clicks during days 0–30       |
| `active_days_30`               | Number of distinct active days during days 0–30                  |
| `unique_resources_30`          | Number of distinct learning resources accessed                   |
| `avg_clicks_per_active_day_30` | Average clicks per active day                                    |
| `assessments_submitted_30`     | Number of distinct assessments submitted by day 30               |
| `avg_score_30`                 | Average score available by day 30                                |
| `late_submissions_30`          | Number of early submissions made after their assessment deadline |
| `has_early_assessment`         | Indicator showing whether early assessment information exists    |

The first model intentionally excludes variables such as:

* gender;
* age band;
* region;
* disability;
* deprivation band.

This keeps the initial experiment focused on early learning behaviour, registration timing, and study history while reducing unnecessary fairness concerns.

## Missing-value strategy

A missing assessment score is not automatically replaced with zero in the processed dataset.

These two situations have different meanings:

```text
Score equals zero
```

and:

```text
No early assessment score was available
```

The model pipeline therefore:

1. preserves missing scores in the processed feature table;
2. applies median imputation during model fitting;
3. adds a missingness indicator through `SimpleImputer(add_indicator=True)`.

This allows the pipeline to treat missing information consistently during training and prediction.

## Model candidates

Two classification models were compared:

1. **Logistic Regression**

   * interpretable linear baseline;
   * median imputation;
   * missing-value indicators;
   * feature scaling;
   * balanced class weights.

2. **Random Forest**

   * nonlinear ensemble model;
   * median imputation;
   * missing-value indicators;
   * balanced class weights;
   * minimum leaf-size regularisation.

## Cross-validation results

The models were compared using stratified five-fold cross-validation on the training partition.

| Model               | Mean CV F1 — support | Mean CV ROC-AUC | Mean CV average precision | Mean CV balanced accuracy |
| ------------------- | -------------------: | --------------: | ------------------------: | ------------------------: |
| Random Forest       |              `0.635` |         `0.740` |                   `0.696` |                   `0.691` |
| Logistic Regression |              `0.629` |         `0.748` |                   `0.702` |                   `0.686` |

The model-selection policy prioritised the mean cross-validated F1-score for the support class.

Based on that policy, **Random Forest** was selected for final fitting and holdout evaluation.

## Holdout results

The selected Random Forest pipeline was evaluated once on the untouched 20% holdout partition.

| Metric                   | Holdout result |
| ------------------------ | -------------: |
| Balanced accuracy        |        `0.666` |
| Precision — support flag |        `0.611` |
| Recall — support flag    |        `0.603` |
| F1-score — support flag  |        `0.607` |
| ROC-AUC                  |        `0.740` |
| Average precision        |        `0.678` |
| Decision threshold       |         `0.50` |

### Confusion matrix

|                     | Predicted no flag | Predicted support flag |
| ------------------- | ----------------: | ---------------------: |
| Actual no flag      |             `156` |                   `58` |
| Actual support flag |              `60` |                   `91` |

### Interpretation

The model correctly identified 91 students in the support-flag category and correctly classified 156 students in the no-flag category.

It also produced:

* 58 false-positive support flags;
* 60 missed support-category records.

These results demonstrate that the model is imperfect and should not be treated as an automated decision-maker.

The model output is intended to support exploration and discussion. Any real academic intervention would require educator review, contextual evidence, and careful fairness evaluation.

## Application preview

### Project overview

![EduInsight home dashboard](images/home-dashboard.png)

The home page presents:

* the number of analysed students;
* average early active days;
* average early clicks;
* the percentage of records in the support category;
* the final-outcome distribution;
* a summary of the project’s technical capabilities.

### Engagement analysis

![EduInsight engagement analysis](images/engagement-analysis.png)

The engagement page allows users to:

* filter records by final outcome;
* compare student counts;
* inspect median early clicks;
* inspect median active days;
* view the active-day distribution;
* compare median engagement indicators across outcome groups.

### Early-support demonstration

![EduInsight early-support demonstration](images/early-support-demo.png)

The demonstration page accepts a hypothetical day-30 profile containing:

* previous attempts;
* studied credits;
* registration timing;
* early clicks;
* active days;
* resources accessed;
* assessment participation;
* average early score;
* late submissions.

The application then displays:

* whether the profile is placed in the support flag;
* the model’s support score;
* the model name;
* the observation cutoff;
* the decision threshold;
* a clear usage disclaimer.

## Technologies

### Data science and application

* Python 3.12
* pandas
* NumPy
* Matplotlib
* scikit-learn
* Joblib
* Streamlit
* Jupyter

### Software quality and delivery

* pytest
* Git
* GitHub
* GitHub Actions
* virtual environments
* automated source compilation
* continuous integration
* Streamlit Community Cloud

## Repository structure

```text
eduinsight-student-success/
├── .github/
│   └── workflows/
│       └── ci.yml
├── data/
│   ├── README.md
│   ├── raw/
│   │   └── .gitkeep
│   └── processed/
│       └── student_features.csv
├── images/
│   ├── home-dashboard.png
│   ├── engagement-analysis.png
│   └── early-support-demo.png
├── models/
│   └── early_support_model.joblib
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_preparation.ipynb
│   ├── 03_exploratory_analysis.ipynb
│   └── 04_model_development.ipynb
├── pages/
│   ├── 1_Overview.py
│   ├── 2_Engagement_Analysis.py
│   └── 3_Early_Support_Demo.py
├── src/
│   ├── __init__.py
│   ├── data_preparation.py
│   ├── feature_engineering.py
│   └── model_training.py
├── tests/
│   ├── conftest.py
│   ├── test_data_preparation.py
│   ├── test_feature_engineering.py
│   └── test_model_training.py
├── .gitignore
├── app.py
├── CHANGELOG.md
├── LICENSE
├── README.md
└── requirements.txt
```

## Notebook workflow

The notebooks should be executed in the following order:

### 1. Data understanding

```text
notebooks/01_data_understanding.ipynb
```

This notebook:

* validates the raw data files;
* loads `studentInfo.csv`;
* inspects columns and data types;
* reviews missing values;
* checks exact duplicates;
* validates the composite student key;
* analyses final-result distributions;
* compares available module presentations;
* confirms the selected presentation.

### 2. Data preparation

```text
notebooks/02_data_preparation.ipynb
```

This notebook:

* applies day-30 eligibility rules;
* processes the large VLE table in chunks;
* creates early engagement features;
* creates early assessment features;
* merges the feature sets using validated keys;
* creates one row per eligible student;
* creates the binary support target;
* performs quality assertions;
* saves the processed feature table.

Generated output:

```text
data/processed/student_features.csv
```

### 3. Exploratory analysis

```text
notebooks/03_exploratory_analysis.ipynb
```

This notebook examines:

* final-outcome distributions;
* early clicks by outcome;
* active days by outcome;
* assessment participation;
* early assessment scores;
* correlations between numeric variables;
* observational findings and limitations.

### 4. Model development

```text
notebooks/04_model_development.ipynb
```

This notebook:

* defines the model feature policy;
* creates a stratified training and holdout split;
* compares Logistic Regression and Random Forest;
* performs stratified cross-validation;
* selects the model according to support-class F1;
* evaluates the selected model on the holdout set;
* saves the fitted pipeline and metadata.

Generated output:

```text
models/early_support_model.joblib
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/hawraamkhalil/eduinsight-student-success.git
cd eduinsight-student-success
```

### 2. Create a virtual environment

#### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 4. Install project dependencies

```bash
python -m pip install -r requirements.txt
```

## Running the application

The repository contains the processed deployment dataset and persisted model artifact required by the Streamlit application.

Run:

```bash
python -m streamlit run app.py
```

The application will normally open at:

```text
http://localhost:8501
```

## Reproducing the complete analysis

To recreate the processed dataset and model from the original source data:

1. Download OULAD from the official UCI Machine Learning Repository.
2. Extract the seven CSV files into `data/raw/`.
3. Activate the project virtual environment.
4. Select the `.venv` Python kernel in VS Code or Jupyter.
5. Execute the notebooks in order from Notebook 1 through Notebook 4.
6. Restart the kernel and use **Run All** for every notebook.

Every notebook defines its own imports, paths, and configuration. The notebooks do not rely on variables remaining in memory from previous notebooks.

## Running automated tests

Run:

```bash
python -m pytest -q
```

The test suite validates:

* presentation filtering;
* defensive DataFrame copying;
* duplicate-key detection;
* missing-value summaries;
* day-30 VLE filtering;
* assessment cutoff logic;
* student eligibility rules;
* missing-activity handling;
* target creation;
* model splitting;
* cross-validation;
* holdout evaluation.

## Continuous integration

The GitHub Actions workflow runs automatically for:

* pushes to `main`;
* pushes to `release/**` branches;
* pull requests targeting `main`.

The workflow:

1. checks out the repository;
2. configures Python 3.12;
3. installs the project dependencies;
4. compiles the Python source;
5. runs the automated tests;
6. verifies that the processed dataset exists;
7. verifies that the model artifact exists;
8. loads and validates the saved model artifact.

Workflow file:

```text
.github/workflows/ci.yml
```

## Model artifact

The persisted model file contains:

```text
model
feature_columns
metadata
metrics
feature_summary
```

This keeps the fitted preprocessing pipeline, model, feature order, experiment metadata, holdout metrics, and Streamlit input defaults together.

The application loads the complete artifact from:

```text
models/early_support_model.joblib
```

Model-persistence files should only be loaded from trusted sources.

## Ethical considerations

Student-support models can affect real people and must be developed and interpreted carefully.

This project therefore:

* avoids describing the output as a diagnosis;
* uses “support flag” rather than labelling students as failures;
* excludes selected demographic variables from the first predictive model;
* avoids claiming that engagement causes academic success;
* displays clear disclaimers in the application;
* documents false positives and false negatives;
* requires human review for any real intervention;
* does not present the model as suitable for institutional deployment.

## Limitations

* The model is trained on one historical OULAD course presentation.
* Results may not generalise to other presentations, institutions, countries, or learning platforms.
* The data is observational and cannot establish causal relationships.
* Platform activity does not fully represent motivation, personal circumstances, teaching quality, or access barriers.
* Fail and Withdrawn outcomes are combined into one support category.
* The default classification threshold has not been optimised for a real institutional cost policy.
* The model has not undergone external validation.
* The project does not evaluate long-term model drift.
* The model should not be used to make automatic decisions about students.

## Future improvements

Potential future work includes:

* evaluating temporal generalisation across additional course presentations;
* comparing performance across modules;
* adding probability-calibration analysis;
* evaluating alternative classification thresholds;
* reporting confidence intervals through repeated validation;
* investigating model fairness across student groups;
* adding explainability views for individual predictions;
* adding model-monitoring and drift checks;
* separating Fail and Withdrawn into different modelling tasks;
* testing whether weekly engagement trends improve performance;
* evaluating deployment through an API-based architecture.

These improvements are intentionally left outside version 1.0.0 so the initial release remains focused, understandable, and reproducible.

## Release information

The initial portfolio release includes:

* reproducible data-understanding notebooks;
* leakage-aware day-30 feature engineering;
* chunked VLE data processing;
* exploratory student-engagement analysis;
* cross-validated model comparison;
* untouched holdout evaluation;
* a persisted Random Forest pipeline;
* a multipage Streamlit application;
* an automated pytest suite;
* GitHub Actions continuous integration;
* application screenshots;
* professional documentation;
* dataset attribution and ethical limitations.

See:

```text
CHANGELOG.md
```

for release history.

## Licence

The code in this repository is released under the **MIT License**.

See:

```text
LICENSE
```

The OULAD dataset is a separate third-party resource and remains subject to its own **CC BY 4.0** licence and attribution requirements.

## Author

**Hawraa Khalil**

* GitHub: https://github.com/hawraamkhalil
* LinkedIn: https://www.linkedin.com/in/hawraakhalil

---

If this project is useful to you, consider starring the repository.
