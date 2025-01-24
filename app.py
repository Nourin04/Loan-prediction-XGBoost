import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import streamlit as st
from sklearn.metrics import classification_report, accuracy_score

# Load data
data = pd.read_csv('loan_data_set.csv')

# Step 1: Data Cleaning
if 'Loan_ID' in data.columns:
    data = data.drop('Loan_ID', axis=1)

# Fill missing values
categorical_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History']
numerical_cols = ['LoanAmount', 'Loan_Amount_Term']
for col in categorical_cols:
    data[col] = data[col].fillna(data[col].mode()[0])
for col in numerical_cols:
    data[col] = data[col].fillna(data[col].median())

# Step 2: Data Preprocessing
label_encoders = {}
for col in data.select_dtypes(include='object').columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Feature Engineering
data['Total_Income'] = data['ApplicantIncome'] + data['CoapplicantIncome']
data['Debt_To_Income_Ratio'] = data['LoanAmount'] / (data['Total_Income'] + 1e-6)
data = data.drop(['ApplicantIncome', 'CoapplicantIncome'], axis=1)

# Outlier Removal
for col in ['LoanAmount', 'Total_Income']:
    q1 = data[col].quantile(0.25)
    q3 = data[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]

# Normalize numerical features
scaler = StandardScaler()
data[['LoanAmount', 'Loan_Amount_Term', 'Total_Income', 'Debt_To_Income_Ratio']] = scaler.fit_transform(
    data[['LoanAmount', 'Loan_Amount_Term', 'Total_Income', 'Debt_To_Income_Ratio']]
)

# Separate features and target variable
X = data.drop('Loan_Status', axis=1)
y = data['Loan_Status']

# Address class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Model training with XGBoost
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 6],
    'learning_rate': [0.01, 0.1],
}

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=StratifiedKFold(n_splits=5), scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# Streamlit App
st.title("Loan Eligibility Predictor")

st.sidebar.header("Input Applicant Details")
user_input = {
    'Gender': st.sidebar.selectbox("Gender", ['Male', 'Female']),
    'Married': st.sidebar.selectbox("Married", ['Yes', 'No']),
    'Dependents': st.sidebar.selectbox("Dependents", ['0', '1', '2', '3+']),
    'Education': st.sidebar.selectbox("Education", ['Graduate', 'Not Graduate']),
    'Self_Employed': st.sidebar.selectbox("Self Employed", ['Yes', 'No']),
    'ApplicantIncome': st.sidebar.number_input("Applicant Income", min_value=0),
    'CoapplicantIncome': st.sidebar.number_input("Coapplicant Income", min_value=0),
    'LoanAmount': st.sidebar.number_input("Loan Amount", min_value=0),
    'Loan_Amount_Term': st.sidebar.number_input("Loan Amount Term", min_value=0),
    'Credit_History': st.sidebar.selectbox("Credit History", [1.0, 0.0]),
    'Property_Area': st.sidebar.selectbox("Property Area", ['Urban', 'Semiurban', 'Rural']),
}

if st.button("Predict Loan Eligibility"):
    # Preprocess user input
    input_df = pd.DataFrame([user_input])
    for col, le in label_encoders.items():
        if col in input_df.columns:
            input_df[col] = le.transform(input_df[col])
    input_df['Total_Income'] = input_df['ApplicantIncome'] + input_df['CoapplicantIncome']
    input_df['Debt_To_Income_Ratio'] = input_df['LoanAmount'] / (input_df['Total_Income'] + 1e-6)
    input_df = input_df.drop(['ApplicantIncome', 'CoapplicantIncome'], axis=1)

    # Ensure correct feature order
    input_df = input_df.reindex(columns=X.columns, fill_value=0)
    input_df[['LoanAmount', 'Loan_Amount_Term', 'Total_Income', 'Debt_To_Income_Ratio']] = scaler.transform(
        input_df[['LoanAmount', 'Loan_Amount_Term', 'Total_Income', 'Debt_To_Income_Ratio']]
    )

    # Make prediction
    prediction = best_model.predict(input_df)
    result = "Loan Approved" if prediction[0] == 1 else "Loan Denied"
    st.success(f"Prediction: {result}")

