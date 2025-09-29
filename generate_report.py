import pandas as pd
import joblib
import json
import numpy as np

# Input validation function (must be defined before use)
def validate_input(data):
    # Example: check for negative values, missing required fields, or out-of-range
    errors = []
    for k, v in data.items():
        if v is None or (isinstance(v, float) and np.isnan(v)):
            errors.append(f"Missing value for {k}")
        if isinstance(v, (int, float)) and v < 0 and k not in ['recent_reports_in_area']:
            errors.append(f"Negative value for {k}")
    return errors

# Load encoders and models
le_source = joblib.load('models/le_source.pkl')
le_season = joblib.load('models/le_season.pkl')
le_gender = joblib.load('models/le_gender.pkl')
le_village = joblib.load('models/le_village.pkl')
mlb = joblib.load('models/mlb_symptoms.pkl')
safety_model = joblib.load('models/water_safety_model.pkl')
with open('models/water_safety_features.json') as f:
    safety_features = json.load(f)
num_cols = ['turbidity','pH','temperature','EC','DO','residual_chlorine','age','duration','recent_reports_in_area','turbidity_x_chlorine','temp_x_DO','rolling_reports','risk_score','days_since_last_outbreak']
scaler = joblib.load('models/feature_scaler.pkl')
try:
    disease_model = joblib.load('models/disease_model.pkl')
    le_disease = joblib.load('models/le_disease.pkl')
    with open('models/disease_model_features.json') as f:
        disease_features = json.load(f)
except:
    disease_model = None
    le_disease = None
    disease_features = None

prob_model = joblib.load('models/outbreak_probability_model.pkl')
with open('models/probability_model_features.json') as f:
    prob_features = json.load(f)

# Symptom columns
symptom_cols = [f'symptom_{s}' for s in mlb.classes_]

# Report generation function
def generate_report(input_dict):
    input_df = pd.DataFrame([input_dict])
    print("[DEBUG] Raw input_df:")
    # Convert all numeric columns to int for output
    for col in input_df.columns:
        if input_df[col].dtype in [np.float64, np.int64, float, int]:
            try:
                input_df[col] = input_df[col].astype(int)
            except Exception:
                pass
    print(input_df)
    # Handle unseen labels for categorical encoders
    try:
        input_df['source_type'] = le_source.transform(input_df['source_type'])
    except Exception as e:
        print(f"[ERROR] Unknown source_type: {input_df['source_type'].values}. Please use one of: {list(le_source.classes_)}")
        exit(1)
    try:
        input_df['season'] = le_season.transform(input_df['season'])
    except Exception as e:
        print(f"[ERROR] Unknown season: {input_df['season'].values}. Please use one of: {list(le_season.classes_)}")
        exit(1)
    try:
        input_df['gender'] = le_gender.transform(input_df['gender'])
    except Exception as e:
        print(f"[ERROR] Unknown gender: {input_df['gender'].values}. Please use one of: {list(le_gender.classes_)}")
        exit(1)
    try:
        input_df['village_id'] = le_village.transform(input_df['village_id'])
    except Exception as e:
        print(f"[ERROR] Unknown village_id: {input_df['village_id'].values}. Please use one of: {list(le_village.classes_)}")
        exit(1)
    input_df['symptoms'] = input_df['symptoms'].fillna('')
    input_df['symptoms_list'] = input_df['symptoms'].apply(lambda x: x.split(';') if x else [])
    symptoms_encoded = mlb.transform(input_df['symptoms_list'])
    for i, col in enumerate(symptom_cols):
        input_df[col] = symptoms_encoded[:,i]
    # Drop unused and reorder to match training
    input_X = input_df.reindex(columns=safety_features, fill_value=0)
    print("[DEBUG] Model input_X (before scaling):")
    print(input_X)
    # Scale only numeric columns
    input_X_scaled = input_X.copy()
    input_X_scaled[num_cols] = scaler.transform(input_X[num_cols])
    print("[DEBUG] Model input_X_scaled (after scaling):")
    print(input_X_scaled)
    # Water safety
    safety_pred = safety_model.predict(input_X_scaled)[0]
    safety_label = 'Unsafe' if safety_pred == 1 else 'Safe'
    # Feature importance for water safety
    if hasattr(safety_model, 'feature_importances_'):
        importances = safety_model.feature_importances_
        feature_importance = sorted(zip(safety_features, importances), key=lambda x: -abs(x[1]))[:5]
        # Format as integer for specified features
        def format_feat(k, v):
            if k in ['DO', 'gender', 'turbidity', 'pH', 'residual_chlorine']:
                return f"{k}: {int(round(v*100))}"
            else:
                return f"{k}: {v:.2f}"
        feature_importance_str = ', '.join([format_feat(k, v) for k, v in feature_importance])
    else:
        feature_importance_str = 'N/A'
    # Disease
    input_X['water_safety_pred'] = safety_pred
    if disease_model and le_disease and disease_features:
        # Align features for disease model
        disease_X = input_X.reindex(columns=disease_features, fill_value=0)
        disease_pred = disease_model.predict(disease_X)[0]
        disease_label = le_disease.inverse_transform([disease_pred])[0]
    else:
        disease_label = 'Unknown'
    # Outbreak probability
    prob_X = input_X_scaled.reindex(columns=prob_features, fill_value=0)
    prob = prob_model.predict(prob_X)[0]
    prob_percent = int(prob * 100)
    # Risk factors (simple demo)
    risk_factors = []
    if 'turbidity' in input_dict and input_dict['turbidity'] > 5:
        risk_factors.append('High turbidity')
    if 'residual_chlorine' in input_dict and input_dict['residual_chlorine'] < 0.2:
        risk_factors.append('Low residual chlorine')
    if 'recent_reports_in_area' in input_dict and input_dict['recent_reports_in_area'] > 1:
        risk_factors.append('Multiple recent reports in area')
    # Recommendation
    if safety_label == 'Unsafe':
        recommendation = 'Immediate chlorination, awareness drive, medical screening'
    else:
        recommendation = 'Continue regular monitoring'
    # Report
    report = {
        'Water Safety': safety_label,
        'Predicted Disease': disease_label,
        'Outbreak Probability': f'{prob_percent}%',
        'Risk Factors': ', '.join(risk_factors) if risk_factors else 'None',
        'Recommendation': recommendation,
        'Top Contributing Features': feature_importance_str
    }
    return report

# Example usage
sample_input = {
    'turbidity': 6.0,
    'pH': 7.0,
    'temperature': 29.0,
    'EC': 360,
    'DO': 6.0,
    'residual_chlorine': 0.1,
    'source_type': 'well',
    'season': 'monsoon',
    'location_lat': 26.123,
    'location_lon': 92.456,
    'village_id': 'V001',
    'recent_reports_in_area': 3,
    'age': 30,
    'gender': 'M',
    'duration': 2,
    'symptoms': 'abdominal pain;vomiting',
    'person_name': 'Test User'
}


def prompt_input():
    print("\nEnter values for prediction:")
    # Define min/max for each field
    field_limits = {
        "turbidity": (0, 100),
        "pH": (0, 14),
        "temperature": (0, 60),
        "EC": (0, 2000),
        "DO": (0, 20),
        "residual_chlorine": (0, 5),
        "location_lat": (-90, 90),
        "location_lon": (-180, 180),
        "recent_reports_in_area": (0, 100),
        "age": (0, 100),
        "duration": (0, 30)
    }
    def ask(key, default, as_int=False, choices=None, minval=None, maxval=None):
        range_str = f" (range: {minval} to {maxval})" if minval is not None and maxval is not None else ""
        prompt = f"{key}{range_str}"
        if choices:
            prompt = f"{key} {choices}{range_str}"
        val = input(prompt + ": ")
        if not val:
            return default
        if as_int:
            try:
                ival = int(val)
                if minval is not None and ival < minval:
                    print(f"{key} must be >= {minval}. Setting to {minval}.")
                    return minval
                if maxval is not None and ival > maxval:
                    print(f"{key} must be <= {maxval}. Setting to {maxval}.")
                    return maxval
                return ival
            except Exception:
                print(f"Invalid integer for {key}. Using default {default}.")
                return default
        if choices:
            if val not in choices:
                print(f"Invalid value for {key}. Must be one of: {choices}")
                return ask(key, default, as_int, choices, minval, maxval)
            return val
        try:
            fval = type(default)(val)
            if minval is not None and fval < minval:
                print(f"{key} must be >= {minval}. Setting to {minval}.")
                return minval
            if maxval is not None and fval > maxval:
                print(f"{key} must be <= {maxval}. Setting to {maxval}.")
                return maxval
            return fval
        except Exception:
            return val
    seasons = ["monsoon", "summer", "winter"]
    fields = [
        ("turbidity", 6, True),
        ("pH", 7, True),
        ("temperature", 29, True),
        ("EC", 360, True),
        ("DO", 6, True),
        ("residual_chlorine", 0, True),
        ("source_type", "well", False),
        ("season", "monsoon", False, seasons),
        ("location_lat", 26, True),
        ("location_lon", 92, True),
        ("village_id", "V001", False),
        ("recent_reports_in_area", 3, True),
        ("age", 30, True),
        ("gender", "M", False),
        ("duration", 2, True),
        ("symptoms", "abdominal pain;vomiting", False),
        ("person_name", "Test User", False)
    ]
    user_input = {}
    for f in fields:
        if len(f) == 4:
            k, v, as_int, choices = f
            minval, maxval = field_limits.get(k, (None, None))
            user_input[k] = ask(k, v, as_int, choices, minval, maxval)
        else:
            k, v, as_int = f
            minval, maxval = field_limits.get(k, (None, None))
            user_input[k] = ask(k, v, as_int, None, minval, maxval)
    errors = validate_input(user_input)
    if errors:
        print("Input errors detected:")
        for err in errors:
            print("  -", err)
        print("Please re-run and fix the above errors.")
        exit(1)
    return user_input

if __name__ == "__main__":
    print("\n========== SMART COMMUNITY HEALTH REPORT ==========")
    try:
        user_input = prompt_input()
    except Exception as e:
        print("[WARN] Using default sample input due to error:", e)
        user_input = sample_input
    report = generate_report(user_input)
    print(f"Water Safety Status      : {report['Water Safety']}")
    print(f"Predicted Disease        : {report['Predicted Disease']}")
    print(f"Outbreak Probability     : {report['Outbreak Probability']}")
    print(f"Risk Factors             : {report['Risk Factors']}")
    print(f"Recommendation           : {report['Recommendation']}")
    print(f"Top Contributing Features: {report['Top Contributing Features']}")
    print("===================================================\n")
