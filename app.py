import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import xgboost as xgb

# Load the saved model and preprocessing objects
model_xgb = joblib.load('model_xgb.joblib')
scaler = joblib.load('scaler.joblib')
ohe = joblib.load('ohe.joblib')

# Extract the unique values for 'country' and 'sector' from the OneHotEncoder (ohe) object
unique_countries = ohe.categories_[0]  # Assuming 'country' is the first categorical feature
unique_sectors = ohe.categories_[1]  # Assuming 'sector' is the second categorical feature

# Define a mapping of encoded values to repayment interval labels
repayment_interval_mapping = {0: '🚅 Bullet Repayment Interval', 1: '🪙 Irregular Repayment Interval', 2: '📅 Monthly Repayment Interval'}

# Title with emojis and colors
st.markdown("<h1 style='text-align: center; color: blue;'>📊 Loan Repayment Interval Prediction 📈</h1>", unsafe_allow_html=True)

# Input Features Section with emojis and description
st.write("## 🎯 Input Features")
st.markdown("Here you can choose the variables to predict the repayment interval based on historical data. Please provide the following details:")

# User input fields using unique country and sector values from the OneHotEncoder object
country = st.selectbox('🌍 Country', unique_countries)
sector = st.selectbox('🏢 Sector', unique_sectors)
funded_amount = st.number_input('💰 Funded Amount', min_value=0, max_value=10000, value=1000, step=50)
lender_count = st.number_input('👥 Lender Count', min_value=1, max_value=100, value=2, step=1)

# Create a sample observation from the user input
sample_listing = pd.DataFrame({
    'country': [country],
    'sector': [sector],
    'funded_amount': [funded_amount],
    'lender_count': [lender_count],
})

# Separate categorical and numerical features
cat_features = ['country', 'sector']
num_features = ['funded_amount', 'lender_count']

# Get the feature names from the OneHotEncoder
ohe_feature_names = ohe.get_feature_names_out(cat_features)

# Combine numerical feature names with encoded categorical feature names
feature_names = np.concatenate([num_features, ohe_feature_names])

# One-hot encode categorical features
X_cat = pd.DataFrame(ohe.transform(sample_listing[cat_features]), columns=ohe_feature_names)

# Scale numerical features
X_num = pd.DataFrame(scaler.transform(sample_listing[num_features]), columns=num_features)

# Combine processed features
X_processed = pd.concat([X_num, X_cat], axis=1)

# Make a prediction (returns the encoded value)
predicted_encoded_repayment_interval = model_xgb.predict(X_processed)[0]

# Map the encoded value back to the actual repayment interval label
predicted_repayment_interval = repayment_interval_mapping.get(int(predicted_encoded_repayment_interval), "Unknown")

# Display the actual repayment interval label with more style
st.title("✅ Predicted Repayment Interval:")
st.markdown(f"<h2 style='color:green;'>{predicted_repayment_interval}</h2>", unsafe_allow_html=True)

# Explanation for SHAP force plots
st.write("## 🔍 SHAP Explanation")
st.markdown("The following SHAP plots explain the model's decision for each repayment interval type. These visualizations help you understand the key features that influenced the model's prediction.")

# SHAP explanations
explainer = shap.TreeExplainer(model_xgb)
shap_values = explainer.shap_values(X_processed)

# Function to add background color to SHAP plots
def add_background(html_content):
    white_background_style = "<style>body { background-color: white; }</style>"
    return white_background_style + html_content

# Generate and save SHAP force plot for Class 0 (Bullet)
st.write("### 🚅 SHAP Force Plot for Class 0: Bullet Repayment Interval")
st.markdown("This plot explains the factors influencing the Bullet repayment interval prediction. Bullet repayment means paying off the loan in one lump sum at the end of the loan period.")
shap_html_path_0 = "shap_force_plot_class_0.html"
shap.save_html(shap_html_path_0, shap.force_plot(
    explainer.expected_value[0],
    shap_values[0][:, 0],
    X_processed.iloc[0, :].values,
    feature_names,
    show=False,
    matplotlib=False
))
with open(shap_html_path_0, 'r', encoding='utf-8') as f:
    shap_html_0 = f.read()
st.components.v1.html(add_background(shap_html_0), height=130)

# Generate and save SHAP force plot for Class 1 (Irregular)
st.write("### 🪙 SHAP Force Plot for Class 1: Irregular Repayment Interval")
st.markdown("This plot explains the factors influencing the Irregular repayment interval prediction. Irregular repayment means paying off the loan at irregular intervals based on specific conditions.")
shap_html_path_1 = "shap_force_plot_class_1.html"
shap.save_html(shap_html_path_1, shap.force_plot(
    explainer.expected_value[1],
    shap_values[0][:, 1],
    X_processed.iloc[0, :].values,
    feature_names,
    show=False,
    matplotlib=False
))
with open(shap_html_path_1, 'r', encoding='utf-8') as f:
    shap_html_1 = f.read()
st.components.v1.html(add_background(shap_html_1), height=130)

# Generate and save SHAP force plot for Class 2 (Monthly)
st.write("### 📅 SHAP Force Plot for Class 2: Monthly Repayment Interval")
st.markdown("This plot explains the factors influencing the Monthly repayment interval prediction. Monthly repayment means paying off the loan in equal monthly installments.")
shap_html_path_2 = "shap_force_plot_class_2.html"
shap.save_html(shap_html_path_2, shap.force_plot(
    explainer.expected_value[2],
    shap_values[0][:, 2],
    X_processed.iloc[0, :].values,
    feature_names,
    show=False,
    matplotlib=False
))
with open(shap_html_path_2, 'r', encoding='utf-8') as f:
    shap_html_2 = f.read()
st.components.v1.html(add_background(shap_html_2), height=130)
