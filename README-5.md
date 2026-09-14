# Customer Churn Prediction 📊

## Project Overview
Customer Churn Prediction is a machine-learning project that predicts whether a customer is likely to leave a service.

The project compares **Logistic Regression** and **Random Forest** models and evaluates them using Accuracy, Precision, Recall, F1 Score and ROC-AUC.

> **Dataset note:** The included dataset is synthetic and was created for educational/portfolio purposes. It is not real customer data.

## Objectives
- Analyze customer behavior and churn patterns.
- Clean and prepare numerical and categorical data.
- Build classification models.
- Compare model performance.
- Visualize churn trends and model results.
- Generate business insights that can support customer-retention strategies.

## Dataset
**Records:** 3,000

Main columns:
- `CustomerID`
- `Age`
- `Gender`
- `TenureMonths`
- `Contract`
- `InternetService`
- `PaymentMethod`
- `TechSupport`
- `OnlineSecurity`
- `PaperlessBilling`
- `MonthlyCharges`
- `TotalCharges`
- `SeniorCitizen`
- `Churn`

`Churn = 1` means the customer churned, while `Churn = 0` means the customer stayed.

## Technologies
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Google Colab
- Jupyter Notebook

## Machine Learning Workflow
1. Load dataset
2. Explore the data
3. Check missing values
4. Encode categorical features
5. Scale numerical features
6. Split into training and testing sets
7. Train Logistic Regression
8. Train Random Forest
9. Evaluate both models
10. Visualize results
11. Identify business insights

## Model Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6900 | 0.6862 | 0.6769 | 0.6815 | 0.7720 |
| Random Forest | 0.6783 | 0.6645 | 0.6939 | 0.6789 | 0.7516 |

**Best model by ROC-AUC:** Logistic Regression (0.7720)

## Graphs
- `01_churn_distribution.png` — overall churn distribution
- `02_churn_by_contract.png` — churn by contract type
- `03_monthly_charges_vs_churn.png` — monthly charges comparison
- `04_roc_curve.png` — model ROC comparison
- `05_confusion_matrix.png` — best-model confusion matrix

## Key Business Insights
- Month-to-month customers show higher churn risk in this synthetic dataset.
- Customers with longer tenure tend to be more stable.
- Higher monthly charges can increase churn probability.
- Customer-support and security services are useful features for churn prediction.

## How to Run in Google Colab
1. Open the `.ipynb` file in Google Colab.
2. Upload `customer_churn_dataset.csv` when prompted.
3. Run all cells from top to bottom.
4. Review the metrics, graphs and business insights.

## Project Structure
```text
customer_churn_prediction/
├── customer_churn_dataset.csv
├── Customer_Churn_Prediction_Colab.ipynb
├── customer_churn_prediction.py
├── model_results.csv
├── requirements.txt
├── README.md
├── LICENSE
├── report/
│   └── Customer_Churn_Project_Report.md
└── graphs/
    ├── 01_churn_distribution.png
    ├── 02_churn_by_contract.png
    ├── 03_monthly_charges_vs_churn.png
    ├── 04_roc_curve.png
    └── 05_confusion_matrix.png
```

## Future Improvements
- Use real company/customer data.
- Perform hyperparameter tuning.
- Add SHAP explainability.
- Build a Streamlit dashboard.
- Add probability-based customer-risk segments.
- Connect predictions to a retention campaign workflow.

## Author
**Manoj Kumar S**

This project is prepared as a student portfolio/internship project.
