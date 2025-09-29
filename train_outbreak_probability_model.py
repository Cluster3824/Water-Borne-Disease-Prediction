import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
import json

# Load preprocessed data
df = pd.read_csv('preprocessed_data.csv')

y = df['outbreak_probability'].fillna(0).astype(float) / 100.0
# Use the same features as water safety model for consistency
with open('models/water_safety_features.json') as f:
	prob_features = json.load(f)
X = df[[col for col in prob_features if col in df.columns]]
feature_scaler = joblib.load('models/feature_scaler.pkl')
num_cols = [col for col in ['turbidity','pH','temperature','EC','DO','residual_chlorine','age','duration','recent_reports_in_area','turbidity_x_chlorine','temp_x_DO','rolling_reports','risk_score','days_since_last_outbreak'] if col in X.columns]
X_scaled = X.copy()
if num_cols:
	X_scaled[num_cols] = feature_scaler.transform(X[num_cols])
y = df['outbreak_probability'].fillna(0).astype(float) / 100.0

# Train probability model (regression)
prob_model = LinearRegression()
prob_model.fit(X_scaled, y)
joblib.dump(prob_model, 'models/outbreak_probability_model.pkl')
# Save features used for probability model
with open('models/probability_model_features.json', 'w') as f:
	json.dump(list(X_scaled.columns), f)
print('Outbreak probability model trained and saved.')
