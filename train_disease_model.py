import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import json

# Load preprocessed data
df = pd.read_csv('preprocessed_data.csv')

# Load water safety model and scaler
safety_model = joblib.load('models/water_safety_model.pkl')
scaler = joblib.load('models/feature_scaler.pkl')

# Get features
with open('models/water_safety_features.json') as f:
    safety_features = json.load(f)
X = df[[col for col in safety_features if col in df.columns]].copy()

# Use the same scaler and numeric columns as in preprocessing
num_cols = [col for col in ['turbidity','pH','temperature','EC','DO','residual_chlorine','age','duration','recent_reports_in_area','turbidity_x_chlorine','temp_x_DO','rolling_reports','risk_score','days_since_last_outbreak'] if col in X.columns]
X_scaled = X.copy()
if num_cols:
    X_scaled[num_cols] = scaler.transform(X[num_cols])

# Add water safety prediction as feature
df['water_safety_pred'] = safety_model.predict(X_scaled)

# Disease label encoding

# Disease label encoding (all values are valid diseases)
le_disease = LabelEncoder()
df['disease_label'] = df['predicted_disease']
df['disease_label_enc'] = le_disease.fit_transform(df['disease_label'])

# Use all rows for training
X_disease = X.copy()
X_disease['water_safety_pred'] = df['water_safety_pred'].values
y_disease = df['disease_label_enc']

# Save disease model features
with open('models/disease_model_features.json', 'w') as f:
    json.dump(list(X_disease.columns), f)

# Train/test split
if len(X_disease) > 0:
    X_train, X_test, y_train, y_test = train_test_split(X_disease, y_disease, test_size=0.2, random_state=42)

    # Train model
    if len(set(y_train)) > 1:
        disease_model = RandomForestClassifier(random_state=42)
        disease_model.fit(X_train, y_train)
        joblib.dump(disease_model, 'models/disease_model.pkl')
        joblib.dump(le_disease, 'models/le_disease.pkl')
        print('Disease Prediction Classification Report:')
        y_pred = disease_model.predict(X_test)
        # Only use labels and target_names present in y_test
        unique_labels = sorted(set(y_test) | set(y_pred))
        target_names = [le_disease.classes_[i] for i in unique_labels]
        print(classification_report(y_test, y_pred, labels=unique_labels, target_names=target_names))
    else:
        print('Not enough disease data to train model.')
else:
    print('No disease-labeled data available for training.')
