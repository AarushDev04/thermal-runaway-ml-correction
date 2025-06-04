"""
Configuration file for thermal runaway ML pipeline
"""
import os

# SAMPLE DATA PATHS (Public)
DATA_PATHS = {
    '0d_csv': 'data/sample_0d_data.csv',
    '3d_csv': 'data/sample_3d_data.csv'
}

# FOR REAL DATA (Users must update these paths)
# DATA_PATHS = {
#     '0d_csv': 'path/to/your/licensed/0d_data.xlsx',
#     '3d_csv': 'path/to/your/licensed/3d_data.xlsx'
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

# Create directories
for dir_path in OUTPUT_DIRS.values():
    os.makedirs(dir_path, exist_ok=True)

print("✅ Configuration loaded - Using sample data")
print("⚠️  For real data: Update DATA_PATHS with your licensed files")

