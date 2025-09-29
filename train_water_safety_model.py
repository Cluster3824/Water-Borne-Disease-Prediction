import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


# Load preprocessed data
df = pd.read_csv('preprocessed_data.csv')

# Advanced Feature Engineering (should match preprocess_data.py)
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
if 'timestamp' in df.columns and 'village_id' in df.columns and 'recent_reports_in_area' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['village_id','timestamp'])
    df['rolling_reports'] = df.groupby('village_id')['recent_reports_in_area'].transform(lambda x: x.rolling(window=2,min_periods=1).mean())
elif 'recent_reports_in_area' in df.columns:
    df['rolling_reports'] = df['recent_reports_in_area'].rolling(window=3, min_periods=1).mean()
if 'turbidity' in df.columns and 'pH' in df.columns and 'residual_chlorine' in df.columns:
    df['risk_score'] = df['turbidity'] + (7 - df['pH']).abs() + (0.2 - df['residual_chlorine']).abs()
else:
    df['risk_score'] = 0
if 'recent_reports_in_area' in df.columns:
    df['recent_outbreaks_flag'] = (df['recent_reports_in_area'] > 2).astype(int)


status_map = {'Safe': 0, 'Unsafe': 1, 'safe': 0, 'unsafe': 1}
y = df['water_safety_status'].map(status_map)
target = 'water_safety_status'
drop_cols = ['water_safety_status','predicted_disease','outbreak_probability','person_name','timestamp','symptoms','symptoms_list','last_outbreak_date']
X = df.drop([col for col in drop_cols if col in df.columns], axis=1)

# Remove columns not present in the CSV
X = X[[col for col in X.columns if col in df.columns or col in ['turbidity_x_chlorine','temp_x_DO','rolling_reports','risk_score','days_since_last_outbreak']]]


# Class balancing (filter out NaN and ensure classes only present in y)
y_nonan = y.dropna()
classes = np.unique(y_nonan)
class_weights = compute_class_weight('balanced', classes=classes, y=y_nonan)
class_weight_dict = {k: v for k, v in zip(classes, class_weights)}


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Debug logging
print("[DEBUG] Training features shape:", X_train.shape)
print("[DEBUG] Training label distribution:", np.bincount(y_train.dropna().astype(int)) if hasattr(y_train, 'values') else y_train.value_counts())

# Try both RandomForest and GradientBoosting, use cross-validation to pick best
models = {
    'RandomForest': RandomForestClassifier(random_state=42, class_weight=class_weight_dict),
    'GradientBoosting': GradientBoostingClassifier(random_state=42)
}
best_score = 0
best_model = None
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
    print(f'{name} CV accuracy: {scores.mean():.3f}')
    if scores.mean() > best_score:
        best_score = scores.mean()
        best_model = model

# Train best model
best_model.fit(X_train, y_train)
import os
import json
joblib.dump(best_model, 'models/water_safety_model.pkl')
# Save feature columns for prediction consistency
with open('models/water_safety_features.json', 'w') as f:
    json.dump(list(X.columns), f)

# Evaluate
print('Water Safety Classification Report:')
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))
