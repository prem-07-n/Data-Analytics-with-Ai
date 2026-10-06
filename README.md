# Dream Housing Finance – Loan Eligibility Prediction

**Student:** Nadipati Prem Sai Kumar

## Project Description
A supervised machine-learning classification project that predicts home-loan eligibility (`Loan_Status`: Y/N) using customer demographic, financial, credit and property information.

## Workflow
Problem Statement → Hypothesis Generation → Data Loading → Data Understanding → EDA → Univariate Analysis → Bivariate Analysis → Missing/Outlier Treatment → Evaluation Metrics → Model Building Part 1 → Feature Engineering → Model Building Part 2 → Cross-Validation → Final Model → Test Prediction → Conclusion.

## Dataset
Dream Housing Finance / Loan Prediction Problem Dataset:
https://www.kaggle.com/competitions/loan-prediction-problem-dataset

Place `loan_sanction_train.csv` and `loan_sanction_test.csv` in the same folder as the notebook.

## Technologies
Python, Pandas, NumPy, Matplotlib, Scikit-learn, Jupyter Notebook.

## Models
Logistic Regression, SVM, KNN, Decision Tree, Random Forest, Extra Trees and Gradient Boosting.

## Key Features
Total Income, log-transformed income/loan amount, income per dependent, loan amount-to-income ratio, EMI proxy and missing-credit-history indicator.

## Setup
```bash
pip install -r requirements.txt
jupyter notebook
```

Open `NadipatiPremSaiKumar_LoanEligibilityPrediction.ipynb` and run all cells from top to bottom.

## Output
The notebook generates `loan_predictions.csv` containing `Loan_ID`, predicted `Loan_Status`, and approval probability.

## Key Finding
Credit history is the strongest predictor of loan eligibility in this dataset.

## Academic Note
This is an academic ML project and is not a production lending/credit decision system.
