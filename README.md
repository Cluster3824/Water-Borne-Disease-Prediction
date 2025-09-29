# Water-Borne Disease Prediction

This project predicts water safety and the risk of water-borne diseases using machine learning. It includes data preprocessing, model training, and an interactive CLI for generating health reports based on water quality and demographic inputs.

## Features
- Predicts water safety status (Safe/Unsafe)
- Predicts likely water-borne disease
- Estimates outbreak probability
- Provides risk factors and recommendations
- Interactive CLI for user input
- Handles real-world input ranges and unseen categories
- Debug logging for transparency

## How It Works
1. **Data Preprocessing**: Cleans and engineers features from raw water quality and health data.
2. **Model Training**: Trains machine learning models for water safety, disease prediction, and outbreak probability.
3. **Prediction/Reporting**: Accepts user input (with validation and real-world ranges) and generates a detailed health report.

## Getting Started

### Prerequisites
- Python 3.8+
- pip
- (Recommended) Virtual environment

### Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/Cluster3824/Water-Borne-Disease-Prediction.git
   cd Water-Borne-Disease-Prediction
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. (Optional) Activate your virtual environment.

### Usage
- **Interactive CLI**:
  ```sh
  python generate_report.py
  ```
  Enter values for each field as prompted. Ranges and valid options are shown for each input.

- **API (if available)**:
  ```sh
  python api_backend.py
  ```
  Then use a tool like Postman or `curl` to send POST requests to `/predict`.

- **Model Training**:
  ```sh
  python preprocess_data.py
  python train_water_safety_model.py
  python train_disease_model.py
  python train_outbreak_probability_model.py
  ```

## Project Structure
- `generate_report.py` — CLI for predictions and reporting
- `preprocess_data.py` — Data cleaning and feature engineering
- `train_water_safety_model.py` — Water safety model training
- `train_disease_model.py` — Disease prediction model training
- `train_outbreak_probability_model.py` — Outbreak probability model training
- `api_backend.py` — (Optional) FastAPI backend for web/API use
- `models/` — Saved models and encoders
- `data/` — Data files (CSV, etc.)

## Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](LICENSE)

## Authors
- Cluster3824
- Contributors welcome!

