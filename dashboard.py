import streamlit as st
import pickle
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

# Set page config first, before any other Streamlit commands
st.set_page_config(
    page_title="Loan Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add a loading spinner while loading the model and data
with st.spinner('Loading model and data...'):
    # Load the saved model and data
    with open('loan_prediction_model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('model_data.pkl', 'rb') as f:
        data = pickle.load(f)

# Title and description
st.title("Loan Prediction Dashboard")
st.write("This dashboard helps predict loan approval based on applicant information.")

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

elif page == "Prediction":
    st.header("Loan Application Prediction")
    
    # Create two columns for input fields
    col1, col2 = st.columns(2)
    
    # Dictionary to store input values
    input_data = {}
    
    with col1:
        st.subheader("Personal Information")
        # Income and employment
        input_data['income_annum'] = st.number_input("Annual Income", min_value=0.0, value=50000.0, step=1000.0)
        input_data['self_employed_Yes'] = 1 if st.selectbox("Self Employed", ["No", "Yes"]) == "Yes" else 0
        input_data['education_Not Graduate'] = 1 if st.selectbox("Education", ["Graduate", "Not Graduate"]) == "Not Graduate" else 0
        
        # Asset values
        st.subheader("Asset Information")
        input_data['residential_assets_value'] = st.number_input("Residential Assets Value", min_value=0.0, value=100000.0, step=10000.0)
        input_data['commercial_assets_value'] = st.number_input("Commercial Assets Value", min_value=0.0, value=50000.0, step=10000.0)
        input_data['luxury_assets_value'] = st.number_input("Luxury Assets Value", min_value=0.0, value=10000.0, step=5000.0)
        input_data['bank_asset_value'] = st.number_input("Bank Assets Value", min_value=0.0, value=25000.0, step=5000.0)

    with col2:
        st.subheader("Loan Information")
        input_data['loan_amount'] = st.number_input("Loan Amount", min_value=0.0, value=100000.0, step=10000.0)
        input_data['loan_term'] = st.number_input("Loan Term (months)", min_value=12, value=60, step=12)
        
        # Calculate derived features
        input_data['total_assets_value'] = (input_data['residential_assets_value'] + 
                                          input_data['commercial_assets_value'] + 
                                          input_data['luxury_assets_value'] + 
                                          input_data['bank_asset_value'])
        
        input_data['loan_to_income_ratio'] = input_data['loan_amount'] / input_data['income_annum']
        input_data['loan_to_assets_ratio'] = input_data['loan_amount'] / input_data['total_assets_value']
        input_data['monthly_loan_payment'] = input_data['loan_amount'] / input_data['loan_term']
        input_data['payment_to_income_ratio'] = (input_data['monthly_loan_payment'] * 12) / input_data['income_annum']

    # Prediction button
    if st.button("Predict Loan Approval"):
        # Create DataFrame with input data
        input_df = pd.DataFrame([input_data])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]
        
        # Display result
        st.subheader("Prediction Result")
        if prediction == 1:
            st.success(f"Loan Approval Probability: {probability[1]:.2%}")
        else:
            st.error(f"Loan Rejection Probability: {probability[0]:.2%}")

elif page == "Feature Importance":
    st.header("Feature Importance Analysis")
    
    # Plot feature importance
    fig, ax = plt.subplots(figsize=(12, 8))
    data['feature_importance'].head(10).plot(kind='barh', x='Feature', y='Importance', ax=ax)
    plt.title('Top 10 Most Important Features')
    st.pyplot(fig)
    
    # Display feature importance table
    st.subheader("Feature Importance Table")
    st.dataframe(data['feature_importance'].head(10))

elif page == "SHAP Analysis":
    st.header("SHAP Value Analysis")
    
    try:
        # Create SHAP summary plot
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.summary_plot(data['shap_values'], data['X_test'], show=False)
        st.pyplot(fig)
        plt.clf()
    except Exception as e:
        st.error("Error generating SHAP plot. This might be due to memory constraints or incompatible SHAP values.")

# Footer
st.markdown("---")
st.markdown("Loan Prediction Model Dashboard - Created with Streamlit")