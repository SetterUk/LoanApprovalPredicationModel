import streamlit as st
import pickle
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns # Import seaborn for confusion matrix heatmap
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score # Import confusion_matrix, roc_curve, roc_auc_score explicitly

# Set page config first, before any other Streamlit commands
st.set_page_config(
    page_title="Loan Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add a loading spinner while loading the model and data
with st.spinner('Loading model and data...'):
    try:
        # Load the saved model and data
        with open('loan_prediction_model.pkl', 'rb') as f:
            model = pickle.load(f)

        with open('model_data.pkl', 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        st.error("Error: Model or data files not found. Please ensure 'loan_prediction_model.pkl' and 'model_data.pkl' are in the same directory.")
        st.stop() # Stop the app if files are not found

# Title and description
st.title("Loan Prediction Dashboard")
st.write("This dashboard helps predict loan approval based on applicant information and visualize model insights.")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a Page", ["Model Overview", "Prediction", "Feature Importance", "SHAP Analysis"])

if page == "Model Overview":
    st.header("Model Performance Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Model Accuracy")
        y_pred = model.predict(data['X_test'])
        accuracy = np.mean(y_pred == data['y_test'])
        st.metric("Test Set Accuracy", f"{accuracy:.2%}")
    
    with col2:
        st.subheader("Features")
        st.metric("Number of Features", len(data['X'].columns))
    
    with col3:
        st.subheader("Dataset Size")
        st.metric("Training Samples", len(data['X_train']))

    st.subheader("Confusion Matrix")
    # Generate confusion matrix
    cm = confusion_matrix(data['y_test'], y_pred)
    
    # Plotting the confusion matrix
    fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Rejected (0)', 'Approved (1)'],
                yticklabels=['Actual Rejected (0)', 'Actual Approved (1)'],
                ax=ax_cm)
    ax_cm.set_xlabel('Predicted Label')
    ax_cm.set_ylabel('True Label')
    ax_cm.set_title('Confusion Matrix')
    st.pyplot(fig_cm)
    plt.clf() # Clear the figure

    st.subheader("ROC Curve")
    # Generate ROC Curve
    y_pred_proba = model.predict_proba(data['X_test'])[:, 1]
    fpr, tpr, thresholds = roc_curve(data['y_test'], y_pred_proba)
    roc_auc = roc_auc_score(data['y_test'], y_pred_proba)

    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax_roc.legend(loc="lower right")
    ax_roc.grid(True)
    st.pyplot(fig_roc)
    plt.clf() # Clear the figure

    st.subheader("Correlation Matrix")
    # Calculate correlation matrix for the numerical columns after encoding
    numerical_cols = data['X'].select_dtypes(include=np.number).columns
    corr_matrix = data['X'][numerical_cols].corr()

    fig_corr, ax_corr = plt.subplots(figsize=(14, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax_corr)
    ax_corr.set_title('Correlation Matrix of Features')
    plt.tight_layout()
    st.pyplot(fig_corr)
    plt.clf() # Clear the figure

elif page == "Prediction":
    st.header("Loan Application Prediction (What-If Analysis)")
    
    st.write("Adjust the input features below to see how the loan approval prediction changes.")

    # Create two columns for input fields
    col1, col2 = st.columns(2)
    
    # Dictionary to store input values, initialized with default values for derived features
    input_data_dict = {
        'total_assets_value': 0.0,
        'loan_to_income_ratio': 0.0,
        'loan_to_assets_ratio': 0.0,
        'monthly_loan_payment': 0.0,
        'payment_to_income_ratio': 0.0
    }
    
    with col1:
        st.subheader("Personal Information")
        # Income and employment
        input_data_dict['income_annum'] = st.number_input("Annual Income", min_value=0.0, value=50000.0, step=1000.0, format="%.2f")
        input_data_dict['self_employed_Yes'] = 1 if st.selectbox("Self Employed", ["No", "Yes"], key="self_employed_pred") == "Yes" else 0
        input_data_dict['education_Not Graduate'] = 1 if st.selectbox("Education", ["Graduate", "Not Graduate"], key="education_pred") == "Not Graduate" else 0
        
        # Add CIBIL Score and Number of Dependents
        input_data_dict['cibil_score'] = st.number_input("CIBIL Score", min_value=300, max_value=900, value=750, step=1)
        input_data_dict['no_of_dependents'] = st.number_input("Number of Dependents", min_value=0, value=0, step=1)

        # Asset values
        st.subheader("Asset Information")
        input_data_dict['residential_assets_value'] = st.number_input("Residential Assets Value", min_value=0.0, value=100000.0, step=10000.0, format="%.2f")
        input_data_dict['commercial_assets_value'] = st.number_input("Commercial Assets Value", min_value=0.0, value=50000.0, step=10000.0, format="%.2f")
        input_data_dict['luxury_assets_value'] = st.number_input("Luxury Assets Value", min_value=0.0, value=10000.0, step=5000.0, format="%.2f")
        input_data_dict['bank_asset_value'] = st.number_input("Bank Assets Value", min_value=0.0, value=25000.0, step=5000.0, format="%.2f")

    with col2:
        st.subheader("Loan Information")
        input_data_dict['loan_amount'] = st.number_input("Loan Amount", min_value=0.0, value=100000.0, step=10000.0, format="%.2f")
        input_data_dict['loan_term'] = st.number_input("Loan Term (months)", min_value=1, value=60, step=12) # min_value changed to 1
        
        # Calculate derived features using updated input values
        input_data_dict['total_assets_value'] = (input_data_dict['residential_assets_value'] + 
                                                  input_data_dict['commercial_assets_value'] + 
                                                  input_data_dict['luxury_assets_value'] + 
                                                  input_data_dict['bank_asset_value'])
        
        # Handle division by zero for derived features
        input_data_dict['loan_to_income_ratio'] = input_data_dict['loan_amount'] / input_data_dict['income_annum'] if input_data_dict['income_annum'] != 0 else 0.0
        input_data_dict['loan_to_assets_ratio'] = input_data_dict['loan_amount'] / input_data_dict['total_assets_value'] if input_data_dict['total_assets_value'] != 0 else 0.0
        input_data_dict['monthly_loan_payment'] = input_data_dict['loan_amount'] / input_data_dict['loan_term'] if input_data_dict['loan_term'] != 0 else 0.0
        input_data_dict['payment_to_income_ratio'] = (input_data_dict['monthly_loan_payment'] * 12) / input_data_dict['income_annum'] if input_data_dict['income_annum'] != 0 else 0.0

    # Ensure all columns expected by the model are present, fill missing with 0 if any
    # This creates a DataFrame with the order of columns matching X_train
    # It's crucial for the prediction to work correctly as column order matters for tree-based models
    input_df = pd.DataFrame([input_data_dict])
    
    # Reindex the input_df to match the columns of X_train, filling missing with 0
    # This handles cases where certain dummy variables might not be present in input_data_dict
    input_df = input_df.reindex(columns=data['X_train'].columns, fill_value=0)

    # Prediction button
    if st.button("Predict Loan Approval"):
        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]
        
        # Display result
        st.subheader("Prediction Result")
        if prediction == 1:
            st.success(f"**Approved!** Probability: {probability[1]:.2%}")
            st.info("Based on the input, the model predicts the loan is likely to be approved.")
        else:
            st.error(f"**Rejected!** Probability: {probability[0]:.2%}")
            st.info("Based on the input, the model predicts the loan is likely to be rejected.")
        
        # Display individual SHAP values for the prediction
        st.subheader("How this prediction was made (Local SHAP Values)")
        try:
            explainer = data['explainer']
            # Reshape input_df to match the shape expected by explainer.shap_values
            shap_values_individual = explainer.shap_values(input_df)
            
            # Create a force plot for the individual prediction
            # Using matplotlib to render SHAP plots as Streamlit doesn't directly support JS visualizations like force_plot
            st.set_option('deprecation.showPyplotGlobalUse', False) # Suppress warning
            fig_force = shap.force_plot(explainer.expected_value, shap_values_individual[0,:], input_df.iloc[0,:], matplotlib=True, show=False)
            st.pyplot(fig_force)
            plt.clf() # Clear the figure
        except Exception as e:
            st.warning(f"Could not generate individual SHAP explanation: {e}. Please ensure SHAP explainer is correctly loaded and the input data format is compatible.")

elif page == "Feature Importance":
    st.header("Feature Importance Analysis")
    
    # Plot feature importance
    fig_fi, ax_fi = plt.subplots(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=data['feature_importance'].head(15), ax=ax_fi) # Show top 15 as in original script
    ax_fi.set_title('Top 15 Most Important Features')
    ax_fi.set_xlabel('Importance Score')
    ax_fi.set_ylabel('Feature')
    plt.tight_layout()
    st.pyplot(fig_fi)
    plt.clf() # Clear the figure
    
    # Display feature importance table
    st.subheader("Feature Importance Table")
    st.dataframe(data['feature_importance']) # Show entire table

elif page == "SHAP Analysis":
    st.header("Global SHAP Value Analysis")
    st.write("The SHAP summary plot shows the impact of each feature on the model's output across the entire test dataset.")
    
    try:
        # Create SHAP summary plot (Beeswarm plot)
        fig_shap, ax_shap = plt.subplots(figsize=(12, 8))
        shap.summary_plot(data['shap_values'], data['X_test'], show=False)
        ax_shap.set_title('SHAP Summary Plot (Feature Impact on Model Output)')
        plt.tight_layout()
        st.pyplot(fig_shap)
        plt.clf() # Clear the figure
    except Exception as e:
        st.error(f"Error generating SHAP plot: {e}. This might be due to memory constraints or incompatible SHAP values.")
        st.info("Consider reducing the size of the test set for SHAP value calculation if this error persists.")





