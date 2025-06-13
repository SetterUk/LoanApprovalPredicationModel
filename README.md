# My Loan Approval Prediction System

I developed this end-to-end machine learning model to predict whether a loan application would be approved using the Loan Prediction Dataset from Kaggle.

## My Project Overview

I used XGBoost to predict loan approval based on applicant information such as income, loan amount, credit score, and assets. My model achieved high accuracy and provided interpretable results through feature importance analysis and SHAP values.

## My Features

- **Data Preprocessing**: I handled categorical variables and created engineered features.
- **Model Training**: I used XGBoost for classification with hyperparameter tuning.
- **Model Evaluation**: I provided accuracy, confusion matrix, ROC AUC, and a classification report.
- **Interactive Dashboard**: I visualized model performance and allowed "what-if" analysis.
- **Model Interpretability**: I showed feature importance and SHAP values.

## My Files

- `loan_prediction_model.py`: This was my script for data preprocessing, model training, and evaluation.
- `loan_dashboard.py`: This was my Streamlit dashboard for model visualization and interaction.
- `check_dataset.py`: This was a utility script I used to examine the dataset structure.
- `loan_prediction_model.pkl`: This was my saved trained model.
- `model_data.pkl`: This was my saved data for dashboard visualization.
- `feature_importance.png`: This was a visualization of my feature importance.

## My Dataset

The dataset I used was the "Loan Approval Prediction Dataset" from Kaggle, which contained information about loan applicants including:

- Number of dependents
- Education level
- Self-employment status
- Income
- Loan amount and term
- Credit score (CIBIL score)
- Asset values (residential, commercial, luxury, bank)
- Loan approval status (target variable)

## My Model Performance

My XGBoost model achieved:
- Accuracy: 99.8%
- ROC AUC: 0.9999
- Precision and Recall: Nearly perfect for both classes

## My Top Features

The most important features for loan approval prediction in my model were:
1. CIBIL Score (credit score)
2. Loan Term
3. Loan-to-Income Ratio
4. Payment-to-Income Ratio
5. Loan-to-Assets Ratio

## How to Run My Project

1. I installed the required packages:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap streamlit kagglehub
   ```

2. Train the model:
   ```
   python loan_prediction_model.py
   ```

3. Run the dashboard:
   ```
   streamlit run loan_dashboard.py
   ```

## Dashboard Features

The interactive dashboard includes:
- Model performance metrics and visualizations
- Feature importance analysis
- SHAP value interpretation
- What-if analysis tool to test different applicant scenarios

## Future Improvements

- Implement more advanced feature engineering
- Try ensemble methods to potentially improve performance
- Add more detailed explanations for loan rejections
- Deploy the dashboard to a cloud platform for wider access