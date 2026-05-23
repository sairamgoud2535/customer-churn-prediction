import streamlit as st
import joblib
model=joblib.load("churn_model.pkl")
scaler=joblib.load("scaler.pkl")
encoder=joblib.load("encoder.pkl")
selected_features=joblib.load("feature_names.pkl")
cat_cols=joblib.load("cat_cols.pkl")
num_cols=joblib.load("num_cols.pkl")
st.title("Customer Churn Prediction")
tenure=st.number_input(
	"Tenure Months",
	min_value=0,
	value=12
)
monthly_charges=st.number_input(
	"Monthly Charges",
	min_value=0.0,
	value=50.0
)
gender=st.selectbox(
	"Gender",
	["Male","Female"]
)
Contract=st.selectbox(
	"Contract",
	["Month-to-Month","One year","Two Year"]
)
Internet_service=st.selectbox(
	"Internet Service",
	["DSL","Fiber optic","No"]
)
if st.button("Predict Sample Customer"):
	import pandas as pd
	def predict_churn(customer_data):
		encoded = encoder.transform(customer_data[cat_cols])
		encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))
		num_df = customer_data[num_cols].reset_index(drop=True)
		final_df = pd.concat([num_df, encoded_df], axis=1)
		final_df = final_df[selected_features]
		scaled_data = scaler.transform(final_df)
		prediction = model.predict(scaled_data)[0]
		probability = model.predict_proba(scaled_data)[0][1]
		return int(prediction), float(probability)
	customer = pd.DataFrame({
		"Country": ["united states"],
		"State": ["California"],
		"Gender": [gender],
		"Senior Citizen": ["No"],
		"Partner": ["Yes"],
		"Dependents": ["No"],
		"Phone Service": ["Yes"],
		"Multiple Lines": ["Yes"],
		"Internet Service": [Internet_service],
		"Online Security": ["Yes"],
		"Online Backup": ["No"],
		"Device Protection": ["Yes"],
		"Tech Support": ["Yes"],
		"Streaming TV": ["Yes"],
		"Streaming Movies": ["Yes"],
		"Contract": [Contract],
		"Paperless Billing": ["Yes"],
		"Payment Method": ["Electronic Check"],
		"Zip Code": [90001],
		"Latitude": [34.0],
		"Longitude": [-118.0],
		"Tenure Months": [tenure],
		"Monthly Charges": [monthly_charges],
		"Total Charges": [2148],
		"CLTV": [4000]})
	prediction,probability=predict_churn(customer)
	if prediction==1:
		st.error(f"High Risk of Churn({probability:.2%})")
	else:
		st.success(f"Low Risk of Churn({probability:.2%})")
