import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from IPython.display import display
except ImportError:
    def display(value):
        print(value)

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
from sklearn.inspection import permutation_importance


TRAIN_FILE = "loan_sanction_train.csv"
TEST_FILE = "loan_sanction_test.csv"

train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)

print("Training data shape:", train.shape)
print("Test data shape:", test.shape)

display(train.head())


print(train.info())
display(train.describe(include="all").T)
display(train.isnull().sum().sort_values(ascending=False))
display(train["Loan_Status"].value_counts())


plt.figure(figsize=(6,4))
train["Loan_Status"].value_counts().plot(kind="bar")
plt.title("Loan Status Distribution")
plt.xlabel("Loan Status")
plt.ylabel("Number of Applicants")
plt.tight_layout()
plt.show()

categorical_columns = [
    "Gender", "Married", "Dependents",
    "Education", "Self_Employed",
    "Credit_History", "Property_Area"
]

for col in categorical_columns:
    if col in train.columns:
        plt.figure(figsize=(7,4))
        train[col].value_counts(dropna=False).plot(kind="bar")
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()


def approval_rate(column):
    return train.groupby(column)["Loan_Status"].apply(
        lambda x: (x == "Y").mean()
    )

for col in ["Credit_History", "Education", "Married", "Property_Area", "Self_Employed"]:
    if col in train.columns:
        print("\nApproval rate by", col)
        display(approval_rate(col).to_frame("Approval_Rate"))

        plt.figure(figsize=(7,4))
        approval_rate(col).plot(kind="bar")
        plt.title(f"Loan Approval Rate by {col}")
        plt.xlabel(col)
        plt.ylabel("Approval Rate")
        plt.tight_layout()
        plt.show()


numeric_columns = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term"
]

for col in numeric_columns:
    if col in train.columns:
        plt.figure(figsize=(7,4))
        plt.hist(train[col].dropna(), bins=30)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

        q1 = train[col].quantile(0.25)
        q3 = train[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = train[(train[col] < lower) | (train[col] > upper)]

        print(f"{col}: {len(outliers)} IQR outliers")


def engineer_features(df):
    data = df.copy()

    data["TotalIncome"] = (
        data["ApplicantIncome"] + data["CoapplicantIncome"]
    )

    data["TotalIncomeLog"] = np.log1p(data["TotalIncome"])
    data["ApplicantIncomeLog"] = np.log1p(data["ApplicantIncome"])
    data["CoapplicantIncomeLog"] = np.log1p(data["CoapplicantIncome"])
    data["LoanAmountLog"] = np.log1p(data["LoanAmount"])

    dependents = pd.to_numeric(
        data["Dependents"].replace("3+", 3),
        errors="coerce"
    ).fillna(0)

    data["IncomePerDependent"] = (
        data["TotalIncome"] / (dependents + 1)
    )

    data["LoanAmountPerIncome"] = (
        data["LoanAmount"] / (data["TotalIncome"] + 1)
    )

    data["EMI_Proxy"] = (
        data["LoanAmount"] /
        data["Loan_Amount_Term"].replace(0, np.nan)
    )

    data["Credit_History_Missing"] = (
        data["Credit_History"].isna().astype(int)
    )

    return data


X = engineer_features(train.drop(columns=["Loan_Status"]))
y = train["Loan_Status"].map({"Y": 1, "N": 0})

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X.select_dtypes(
    exclude=["object"]
).columns.tolist()

print("Numerical features:", numerical_features)
print("Categorical features:", categorical_features)


def create_preprocessor():
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("numerical", numerical_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])


def create_pipeline(model):
    return Pipeline([
        ("preprocessor", create_preprocessor()),
        ("model", model)
    ])


models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=42
    ),
    "SVM": SVC(
        probability=True,
        random_state=42
    ),
    "KNN": KNeighborsClassifier(
        n_neighbors=7
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=5,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=500,
        max_depth=6,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    )
}

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

holdout_results = []
trained_models = {}

for name, model in models.items():
    pipeline = create_pipeline(model)
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_valid)
    probabilities = pipeline.predict_proba(X_valid)[:, 1]

    holdout_results.append([
        name,
        accuracy_score(y_valid, predictions),
        precision_score(y_valid, predictions, zero_division=0),
        recall_score(y_valid, predictions, zero_division=0),
        f1_score(y_valid, predictions, zero_division=0),
        roc_auc_score(y_valid, probabilities)
    ])

    trained_models[name] = pipeline

holdout_results = pd.DataFrame(
    holdout_results,
    columns=[
        "Model", "Accuracy", "Precision",
        "Recall", "F1", "ROC-AUC"
    ]
).sort_values(
    ["F1", "Accuracy"],
    ascending=False
)

display(holdout_results)


best_holdout_model_name = holdout_results.iloc[0]["Model"]
best_holdout_model = trained_models[best_holdout_model_name]

holdout_predictions = best_holdout_model.predict(X_valid)

print("Best holdout model:", best_holdout_model_name)
print("\nClassification Report:")
print(
    classification_report(
        y_valid,
        holdout_predictions,
        target_names=["Rejected", "Approved"]
    )
)

print("Confusion Matrix:")
print(confusion_matrix(y_valid, holdout_predictions))


cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_results = []

for name, model in models.items():
    pipeline = create_pipeline(model)

    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=[
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc"
        ],
        n_jobs=-1
    )

    cv_results.append([
        name,
        scores["test_accuracy"].mean(),
        scores["test_precision"].mean(),
        scores["test_recall"].mean(),
        scores["test_f1"].mean(),
        scores["test_roc_auc"].mean()
    ])

cv_results = pd.DataFrame(
    cv_results,
    columns=[
        "Model",
        "CV Accuracy",
        "CV Precision",
        "CV Recall",
        "CV F1",
        "CV ROC-AUC"
    ]
).sort_values(
    ["CV F1", "CV Accuracy"],
    ascending=False
)

display(cv_results)


plt.figure(figsize=(9,5))
plt.bar(
    cv_results["Model"],
    cv_results["CV F1"]
)
plt.title("Model Comparison - Cross-Validated F1 Score")
plt.xlabel("Model")
plt.ylabel("F1 Score")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


final_model_name = cv_results.iloc[0]["Model"]

final_model = create_pipeline(
    models[final_model_name]
)

final_model.fit(X, y)

print("Final selected model:", final_model_name)


permutation = permutation_importance(
    best_holdout_model,
    X_valid,
    y_valid,
    scoring="roc_auc",
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": permutation.importances_mean
}).sort_values(
    "Importance",
    ascending=False
)

display(feature_importance.head(15))

plt.figure(figsize=(8,5))
top_features = feature_importance.head(10).sort_values("Importance")
plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)
plt.title("Top Feature Importance")
plt.xlabel("Permutation Importance")
plt.tight_layout()
plt.show()


test_features = engineer_features(test)

test_predictions = final_model.predict(test_features)
test_probabilities = final_model.predict_proba(test_features)[:, 1]

prediction_output = pd.DataFrame({
    "Loan_ID": test["Loan_ID"],
    "Loan_Status": np.where(
        test_predictions == 1,
        "Y",
        "N"
    ),
    "Approval_Probability": test_probabilities
})

prediction_output.to_csv(
    "loan_predictions.csv",
    index=False
)

display(prediction_output.head(20))

print(
    "Approved:",
    (prediction_output["Loan_Status"] == "Y").sum()
)

print(
    "Rejected:",
    (prediction_output["Loan_Status"] == "N").sum()
)

print("Prediction file saved as loan_predictions.csv")


print("PROJECT COMPLETED")
print("Student: Nadipati Prem Sai Kumar")
print("Final Model:", final_model_name)
print("Training Rows:", len(train))
print("Test Rows:", len(test))
print("Prediction File: loan_predictions.csv")
