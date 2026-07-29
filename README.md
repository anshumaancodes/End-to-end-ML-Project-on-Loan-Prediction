# End-to-end-ML-Project-on-Loan-Prediction
Predicting loan approval using applicant demographic and financial data — data wrangling, EDA, feature engineering &amp; classification modeling (Logistic Regression, Random Forest, SVM).
## Dataset
Loan Prediction Dataset (Analytics Vidhya / Kaggle)
614 applicant records with demographic (Gender, Married, Dependents, Education, 
Self_Employed, Property_Area) and financial (ApplicantIncome, CoapplicantIncome, 
LoanAmount, Loan_Amount_Term, Credit_History) features.

## Project Structure
loan_prediction.csv — dataset
EDA_Loan_Prediction.ipynb — data cleaning, EDA, hypothesis testing
Loan_Model_EndToEnd.ipynb — feature engineering & model training/evaluation
Research_Report_Loan_Prediction.pdf — written research report
loan_model.pkl — trained model (for FastAPI deployment)
loan_scaler.pkl — fitted feature scaler
loan_feature_columns.pkl — expected feature column order
loan_scale_columns.pkl — columns requiring scaling
requirements.txt — dependencies
## Setup
1. Clone this repo
2. `pip install -r requirements.txt`
3. Open the notebook(s) with Jupyter or PyCharm
4. Run all cells top to bottom (Restart Kernel → Run All)

## Key Findings
- **Credit_History is the single strongest predictor** of loan approval — 
  applicants with a good credit history are approved at a dramatically higher rate.
- Engineered features like `Loan_Income_Ratio` and `TotalIncome` add signal beyond 
  the raw income/loan columns.
- Best model: **Logistic Regression**, ROC-AUC ≈ 0.879 — also the most interpretable, 
  which matters for explaining credit decisions to applicants and regulators.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.854 | 0.832 | 0.988 | 0.903 | 0.879 |
| Random Forest | 0.870 | 0.888 | 0.929 | 0.908 | 0.858 |
| SVM | 0.846 | 0.824 | 0.988 | 0.898 | 0.853 |
| Naive Bayes | 0.862 | 0.854 | 0.965 | 0.906 | 0.844 |

## Next Steps
- FastAPI deployment for real-time predictions
- Streamlit dashboard for visualizing results and model functionality

## Author
anshumaancodes
