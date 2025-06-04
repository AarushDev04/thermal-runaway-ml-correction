"""
Configuration file for thermal runaway ML pipeline
"""
import os

# File paths - UPDATE THESE WITH YOUR ACTUAL PATHS
DATA_PATHS = {
    '0d_csv': 'data/0d_simulation_data.csv',  # ← Put your 0D CSV here
    '3d_csv': 'data/3d_simulation_data.csv'   # ← Put your 3D CSV here
}

# If using sample data, uncomment these lines:
# DATA_PATHS = {
#     '0d_csv': 'data/sample_0d_data.csv',
#     '3d_csv': 'data/sample_3d_data.csv'
# }

# Model parameters
MODEL_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 5
}

# Output directories
OUTPUT_DIRS = {
    'models': 'models/',
    'results': 'results/',
    'plots': 'plots/'
}

# Create directories if they don't exist
for dir_path in OUTPUT_DIRS.values():
    os.makedirs(dir_path, exist_ok=True)

print("✅ Configuration loaded successfully")
