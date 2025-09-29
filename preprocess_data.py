import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, StandardScaler

# Load dataset
df = pd.read_csv('sample_water_health_data_balanced_disease.csv')

# Only keep columns present in the CSV
expected_cols = [
	'turbidity','pH','temperature','EC','DO','residual_chlorine','source_type','season',
	'location_lat','location_lon','recent_reports_in_area','age','gender','duration','symptoms',
	'person_name','water_safety_status','predicted_disease','outbreak_probability'
]
df = df[[col for col in expected_cols if col in df.columns]]

# Impute missing values for numeric columns
for col in ['turbidity','pH','temperature','EC','DO','residual_chlorine','age','duration','recent_reports_in_area']:
	if col in df.columns:
		df[col] = df[col].fillna(df[col].median())

# Encode categorical features if present
if 'source_type' in df.columns:
	le_source = LabelEncoder()
	df['source_type'] = le_source.fit_transform(df['source_type'])
if 'season' in df.columns:
	le_season = LabelEncoder()
	df['season'] = le_season.fit_transform(df['season'])
if 'gender' in df.columns:
	le_gender = LabelEncoder()
	df['gender'] = le_gender.fit_transform(df['gender'])
if 'village_id' in df.columns:
	le_village = LabelEncoder()
	df['village_id'] = le_village.fit_transform(df['village_id'])

# Symptoms: multi-label binarizer if present
if 'symptoms' in df.columns:
	mlb = MultiLabelBinarizer()
	df['symptoms'] = df['symptoms'].fillna('')
	df['symptoms_list'] = df['symptoms'].apply(lambda x: x.split(';') if x else [])
	symptoms_encoded = mlb.fit_transform(df['symptoms_list'])
	symptom_cols = [f'symptom_{s}' for s in mlb.classes_]
	symptoms_df = pd.DataFrame(symptoms_encoded, columns=symptom_cols)
	df = pd.concat([df, symptoms_df], axis=1)


# Advanced Feature Engineering
if 'turbidity' in df.columns and 'residual_chlorine' in df.columns:
	df['turbidity_x_chlorine'] = df['turbidity'] * df['residual_chlorine']
	df['turbidity_div_chlorine'] = df['turbidity'] / (df['residual_chlorine'] + 1e-3)
if 'temperature' in df.columns and 'DO' in df.columns:
	df['temp_x_DO'] = df['temperature'] * df['DO']
	df['temp_div_DO'] = df['temperature'] / (df['DO'] + 1e-3)
if 'turbidity' in df.columns and 'pH' in df.columns:
	df['turbidity_div_pH'] = df['turbidity'] / (df['pH'] + 1e-3)
if 'source_type' in df.columns and 'season' in df.columns:
	df['source_season'] = df['source_type'].astype(str) + '_' + df['season'].astype(str)

# Rolling average of recent reports (simulate by groupby village if timestamp available)
if 'timestamp' in df.columns and 'village_id' in df.columns and 'recent_reports_in_area' in df.columns:
	df['timestamp'] = pd.to_datetime(df['timestamp'])
	df = df.sort_values(['village_id','timestamp'])
	df['rolling_reports'] = df.groupby('village_id')['recent_reports_in_area'].transform(lambda x: x.rolling(window=2,min_periods=1).mean())
elif 'recent_reports_in_area' in df.columns:
	df['rolling_reports'] = df['recent_reports_in_area'].rolling(window=3, min_periods=1).mean()

# Risk score
if 'turbidity' in df.columns and 'pH' in df.columns and 'residual_chlorine' in df.columns:
	df['risk_score'] = df['turbidity'] + (7 - df['pH']).abs() + (0.2 - df['residual_chlorine']).abs()
else:
	df['risk_score'] = 0

# Example: Add a feature for recent outbreaks (if you have such data)
if 'recent_reports_in_area' in df.columns:
	df['recent_outbreaks_flag'] = (df['recent_reports_in_area'] > 2).astype(int)

# Days since last outbreak (placeholder)
df['days_since_last_outbreak'] = 0

# Feature scaling for numeric columns
from sklearn.preprocessing import StandardScaler
import joblib
num_cols = [col for col in ['turbidity','pH','temperature','EC','DO','residual_chlorine','age','duration','recent_reports_in_area','turbidity_x_chlorine','temp_x_DO','rolling_reports','risk_score','days_since_last_outbreak'] if col in df.columns]
if num_cols:
	scaler = StandardScaler()
	df[num_cols] = scaler.fit_transform(df[num_cols])
	joblib.dump(scaler, 'models/feature_scaler.pkl')

# Save preprocessed data
df.to_csv('preprocessed_data.csv', index=False)
print('Preprocessing complete. Saved to preprocessed_data.csv')

import joblib
if 'scaler' in locals():
	joblib.dump(scaler, 'models/feature_scaler.pkl')
# Save encoders for later use
if 'le_source' in locals():
	joblib.dump(le_source, 'models/le_source.pkl')
if 'le_season' in locals():
	joblib.dump(le_season, 'models/le_season.pkl')
if 'le_gender' in locals():
	joblib.dump(le_gender, 'models/le_gender.pkl')
if 'le_village' in locals():
	joblib.dump(le_village, 'models/le_village.pkl')
if 'mlb' in locals():
	joblib.dump(mlb, 'models/mlb_symptoms.pkl')
df.to_csv('preprocessed_data.csv', index=False)
print('Preprocessing complete. Saved as preprocessed_data.csv')
