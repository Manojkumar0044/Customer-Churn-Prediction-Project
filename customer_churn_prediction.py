# Customer Churn Prediction
# Run in Google Colab or locally.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

df = pd.read_csv("customer_churn_dataset.csv")

X = df.drop(columns=["CustomerID", "Churn"])
y = df["Churn"]

numeric_features = ["Age", "TenureMonths", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
categorical_features = ["Gender", "Contract", "InternetService", "PaymentMethod",
                        "TechSupport", "OnlineSecurity", "PaperlessBilling"]

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric_features),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_features)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=250, max_depth=10, min_samples_split=5,
        random_state=42, class_weight="balanced"
    )
}

for name, estimator in models.items():
    pipe = Pipeline([("prep", preprocessor), ("model", estimator)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]

    print("\n", "="*60)
    print(name)
    print("="*60)
    print("Accuracy :", round(accuracy_score(y_test, pred), 4))
    print("Precision:", round(precision_score(y_test, pred, zero_division=0), 4))
    print("Recall   :", round(recall_score(y_test, pred, zero_division=0), 4))
    print("F1 Score :", round(f1_score(y_test, pred, zero_division=0), 4))
    print("ROC-AUC  :", round(roc_auc_score(y_test, proba), 4))
    print(classification_report(y_test, pred, target_names=["Stayed", "Churned"], zero_division=0))
