# EduInsight: Student Engagement and Early-Support Analytics

EduInsight is an end-to-end data science portfolio project that explores how
student activity observed during the first 30 days of a course relates to later
academic outcomes. It combines reproducible data preparation, exploratory
analysis, machine learning, testing, and a Streamlit dashboard.

> **Project status:** the repository contains the complete professional project
> structure and implementation. The original OULAD files must be downloaded
> locally before running the notebooks.

## Why this project

The project is designed to demonstrate more than a single notebook. It shows:

- multi-table data preparation and validated joins;
- chunked processing of a large activity file;
- leakage-aware feature engineering;
- a clean separation between notebooks and reusable Python modules;
- cross-validated model selection with a final holdout evaluation;
- automated tests and defensive error handling;
- a deployable Streamlit interface.

## Research question

**Can activity available during the first 30 course days help identify student
profiles that may benefit from additional academic support?**

The output is a portfolio demonstration, not an automated academic decision.

## Repository structure

```text
eduinsight-student-success/
├── app.py
├── README.md
├── requirements.txt
├── data/
│   ├── README.md
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_preparation.ipynb
│   ├── 03_exploratory_analysis.ipynb
│   └── 04_model_development.ipynb
├── src/
│   ├── __init__.py
│   ├── data_preparation.py
│   ├── feature_engineering.py
│   └── model_training.py
├── pages/
│   ├── 1_Overview.py
│   ├── 2_Engagement_Analysis.py
│   └── 3_Early_Support_Demo.py
├── models/
├── images/
└── tests/