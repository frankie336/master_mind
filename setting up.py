import os

dirs = [
    "services/common",
    "services/fetch_service",
    "services/feature_engineering_service",
    "services/data_merger_service",
    "services/indicators_merger_service",
    "services/predictor_service",
    "configs",
    "data/processed",
    "data/raw",
    "tests/common_tests",
    "tests/fetch_service_tests",
    "tests/feature_engineering_tests",
    "tests/predictor_service_tests",
    "scripts",
    "logs",
    "volumes"
]

for dir in dirs:
    os.makedirs(dir, exist_ok=True)
    with open(os.path.join(dir, '__init__.py'), 'w') as f:  # Create __init__.py for packages
        pass

print("Directory structure has been created.")
