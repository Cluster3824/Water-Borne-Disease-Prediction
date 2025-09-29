import subprocess
import sys

steps = [
    ("Preprocessing data", [sys.executable, "preprocess_data.py"]),
    ("Training water safety model", [sys.executable, "train_water_safety_model.py"]),
    ("Training disease model", [sys.executable, "train_disease_model.py"]),
    ("Training outbreak probability model", [sys.executable, "train_outbreak_probability_model.py"]),
    ("Generating sample report", [sys.executable, "generate_report.py"]),
    ("Sending alerts if needed", [sys.executable, "alert.py"])

]

for desc, cmd in steps:
    print(f"\n=== {desc} ===")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Error in step: {desc}")
        sys.exit(result.returncode)
print("\nAll steps completed successfully.")
