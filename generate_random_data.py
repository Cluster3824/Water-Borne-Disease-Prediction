import pandas as pd
import numpy as np
import random

# Define possible values for categorical columns
source_types = ['borewell', 'piped_supply', 'river', 'spring', 'well','tap']
seasons = ['monsoon', 'summer', 'winter']
genders = ['M', 'F']
diseases = ['Cholera', 'Typhoid', 'Dysentery', 'Hepatitis', 'None']
symptoms_list = ['abdominal pain', 'vomiting', 'diarrhea', 'fever', 'headache', 'nausea']

# Number of rows to generate
N = 4777

def random_symptoms():
    count = random.randint(0, 3)
    return ';'.join(random.sample(symptoms_list, count)) if count > 0 else ''

data = {
    'turbidity': np.random.randint(0, 101, N),
    'pH': np.random.uniform(5.5, 9.5, N).round(2),
    'temperature': np.random.randint(10, 41, N),
    'EC': np.random.randint(50, 2001, N),
    'DO': np.random.uniform(2, 12, N).round(2),
    'residual_chlorine': np.random.uniform(0, 1, N).round(2),
    'source_type': np.random.choice(source_types, N),
    'season': np.random.choice(seasons, N),
    'location_lat': np.random.uniform(24, 28, N).round(5),
    'location_lon': np.random.uniform(90, 95, N).round(5),
    'village_id': [f'V{str(i%20+1).zfill(3)}' for i in range(N)],
    'recent_reports_in_area': np.random.randint(0, 6, N),
    'age': np.random.randint(1, 101, N),
    'gender': np.random.choice(genders, N),
    'duration': np.random.randint(1, 11, N),
    'symptoms': [random_symptoms() for _ in range(N)],
    'person_name': [f'Person_{i+1}' for i in range(N)],
    'water_safety_status': np.random.choice(['Safe', 'Unsafe'], N),
    'predicted_disease': np.random.choice(diseases, N),
    'outbreak_probability': np.random.randint(0, 101, N)
}

df = pd.DataFrame(data)
df.to_csv('sample_water_health_data_balanced_disease.csv', index=False)
print('Random data generated and saved to sample_water_health_data_balanced_disease.csv')
