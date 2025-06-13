# First, I'll import all the libraries I need for my loan prediction model
# I need these for handling my data
import pandas as pd
import numpy as np

# These will help me create beautiful visualizations
import matplotlib.pyplot as plt
import seaborn as sns

# I'll use these for my machine learning tasks
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import xgboost as xgb  # This is my main model

# Some utility libraries I'll need
import kagglehub  # To get my dataset
import pickle     # To save my model
import os         # To handle file paths
import shap       # To explain my model's decisions

# I want my results to be reproducible, so I'll set a random seed
np.random.seed(42)

# Now I'll get my dataset from Kaggle
print("Downloading dataset...")
path = kagglehub.dataset_download("architsharma01/loan-approval-prediction-dataset")
print("Path to dataset files:", path)

# Let me load my dataset into a DataFrame
df = pd.read_csv(os.path.join(path, "loan_approval_dataset.csv"))

# I should clean up any spaces in my column names
df.columns = df.columns.str.strip()

# Let me check what my data looks like
print("\nDataset Information:")
print(df.info())

# I need to check if I have any missing values to deal with
print("\nMissing values in each column:")
print(df.isnull().sum())

# I'll look at how my loan approvals are distributed
print("\nDistribution of loan_status:")
print(df['loan_status'].value_counts())
print("\nPercentage distribution:")
print(df['loan_status'].value_counts(normalize=True).mul(100))

# Let me check for any duplicate entries
print("\nNumber of duplicate rows:", df.duplicated().sum())

# Time for data preprocessing!
print("\nPreprocessing data...")

# I'll convert my categorical variables into a format my model can understand
df_encoded = pd.get_dummies(df, columns=['education', 'self_employed'], drop_first=True)

# Let me see what values I have in loan_status
print("\nUnique values in loan_status:")
print(df_encoded['loan_status'].unique())

# I'll convert my loan status into binary values - this makes it easier for my model
df_encoded['loan_status_binary'] = df_encoded['loan_status'].apply(lambda x: 1 if x.strip() == 'Approved' else 0)

# Let me check the distribution of my binary target
print("\nDistribution of binary target:")
print(df_encoded['loan_status_binary'].value_counts())
print(df_encoded['loan_status_binary'].value_counts(normalize=True).mul(100))

# I don't need these columns anymore
df_encoded = df_encoded.drop(['loan_status', 'loan_id'], axis=1)

# Now for the exciting part - feature engineering!
print("\nPerforming feature engineering...")

# I'll create some new features that I think will help my model make better decisions
# First, I'll calculate the total value of all assets
df_encoded['total_assets_value'] = df_encoded['residential_assets_value'] + \
                                  df_encoded['commercial_assets_value'] + \
                                  df_encoded['luxury_assets_value'] + \
                                  df_encoded['bank_asset_value']

# This will tell me how big the loan is compared to income
df_encoded['loan_to_income_ratio'] = df_encoded['loan_amount'] / df_encoded['income_annum']

# And this shows how the loan compares to total assets
df_encoded['loan_to_assets_ratio'] = df_encoded['loan_amount'] / df_encoded['total_assets_value']

# Let me calculate the monthly payments
df_encoded['monthly_loan_payment'] = df_encoded['loan_amount'] / df_encoded['loan_term']

# And see how much of their income goes to payments
df_encoded['payment_to_income_ratio'] = (df_encoded['monthly_loan_payment'] * 12) / df_encoded['income_annum']

# Time to split my data for training!
X = df_encoded.drop('loan_status_binary', axis=1)
y = df_encoded['loan_status_binary']

# I'll use 80% for training and keep 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Now I'll set up my XGBoost model with the parameters I've chosen
model = xgb.XGBClassifier(
    n_estimators=100,      # I'll use 100 trees
    learning_rate=0.1,     # This controls how much I adjust for mistakes
    max_depth=5,           # My trees won't get too complex
    min_child_weight=1,    # This helps prevent overfitting
    gamma=0,               # Minimum loss reduction
    subsample=0.8,         # I'll use 80% of data for each tree
    colsample_bytree=0.8,  # And 80% of features
    objective='binary:logistic',  # Since I'm doing binary classification
    random_state=42,       # For reproducibility
    base_score=0.5         # Starting with neutral predictions
)

# Let's train my model!
model.fit(X_train, y_train)

# Time to see how well my model performs
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# I'll check various metrics to evaluate my model
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Let me calculate the ROC AUC score
roc_auc = roc_auc_score(y_test, y_pred_proba)
print("\nROC AUC Score:", roc_auc)

# Now I'll look at which features were most important
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

# I'll create a nice visualization of feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
plt.title('My Top 15 Most Important Features')
plt.tight_layout()
plt.savefig('feature_importance.png')

# Let me calculate SHAP values to understand my model's decisions
print("\nCalculating SHAP values...")
explainer = shap.Explainer(model)
shap_values = explainer(X_test)

# Finally, I'll save my model and all the important data
print("\nSaving my model and data...")
with open('loan_prediction_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# I'll save everything else I might need later
with open('model_data.pkl', 'wb') as f:
    pickle.dump({
        'X': X, 'y': y,
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'feature_importance': feature_importance,
        'shap_values': shap_values,
        'explainer': explainer
    }, f)

print("\nGreat! My model is trained and ready to go!")