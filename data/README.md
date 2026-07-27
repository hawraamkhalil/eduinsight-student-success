# Data directory

This project uses the **Open University Learning Analytics Dataset (OULAD)**.
The original CSV files are intentionally excluded from Git because `studentVle.csv`
is large and the dataset should be obtained from its official source.

## Expected raw files

Place these files in `data/raw/`:

- `courses.csv`
- `assessments.csv`
- `studentAssessment.csv`
- `studentInfo.csv`
- `studentRegistration.csv`
- `studentVle.csv`
- `vle.csv`

## Generated files

Running `notebooks/02_data_preparation.ipynb` creates:

- `data/processed/student_features.csv`

Running `notebooks/04_model_development.ipynb` creates:

- `models/early_support_model.joblib`

## Dataset citation

Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2015). *Open University
Learning Analytics Dataset*. UCI Machine Learning Repository.
https://doi.org/10.24432/C5KK69

License: Creative Commons Attribution 4.0 International (CC BY 4.0).