# Customer Churn Prediction — Project Report

## 1. Introduction
Customer churn occurs when customers stop using a company's products or services. Predicting churn early allows organizations to focus retention efforts on customers who are more likely to leave.

## 2. Problem Statement
Build a machine-learning classification system that predicts whether a customer will churn based on demographic, service, billing and tenure-related information.

## 3. Objectives
- Understand customer churn patterns.
- Prepare structured customer data for machine learning.
- Train classification models.
- Compare model performance.
- Produce visualizations and actionable business insights.

## 4. Dataset
The project uses a synthetic dataset containing 3,000 customer records. It includes customer demographics, contract information, services, payment method, charges, tenure and churn status.

The dataset is synthetic and intended for educational/portfolio use.

## 5. Methodology
### Data Preparation
- Removed the identifier column from model training.
- Separated target variable `Churn`.
- Used median imputation and standardization for numerical features.
- Used most-frequent imputation and one-hot encoding for categorical features.

### Models
1. Logistic Regression — interpretable baseline classification model.
2. Random Forest — ensemble model capable of capturing nonlinear relationships.

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

## 6. Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6900 | 0.6862 | 0.6769 | 0.6815 | 0.7720 |
| Random Forest | 0.6783 | 0.6645 | 0.6939 | 0.6789 | 0.7516 |

The best model according to ROC-AUC is **Logistic Regression**, with an ROC-AUC of **0.7720**.

## 7. Key Findings
- Contract type is an important churn signal.
- Month-to-month customers are more likely to churn than customers with longer contracts.
- Monthly charges are associated with churn risk.
- Tenure is an important retention indicator.
- Support and security services can provide additional predictive information.

## 8. Business Recommendations
1. Provide retention offers to high-risk month-to-month customers.
2. Encourage customers to move to annual or longer-term contracts.
3. Monitor customers with high monthly charges.
4. Improve technical support and online-security service adoption.
5. Build a customer-risk dashboard for the retention team.

## 9. Limitations
- The dataset is synthetic.
- Results should not be interpreted as real business performance.
- Real-world deployment requires validation on representative production data.
- Model thresholds should be selected according to the cost of false positives and false negatives.

## 10. Conclusion
The project demonstrates an end-to-end customer churn prediction workflow using Python and machine learning. It covers data preparation, model training, evaluation, visualization and business recommendations. The solution can be extended into a real-time retention dashboard or customer-risk scoring application.

## 11. Tools Used
Python, Pandas, NumPy, Matplotlib, Scikit-learn and Google Colab.
