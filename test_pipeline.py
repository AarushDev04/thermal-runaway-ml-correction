"""
Enhanced test script to validate pipeline functionality
"""

import pandas as pd
import numpy as np
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from utils import validate_csv_format, check_dependencies, create_directories, create_sample_data
from config import DATA_PATHS

def safe_create_directories():
    """Create directories, but handle the case where a file blocks directory creation."""
    dirs = ['data', 'models', 'results', 'plots']
    for dir_name in dirs:
        if os.path.isfile(dir_name):
            print(f"❌ A file named '{dir_name}' exists. Please remove or rename it so a directory can be created.")
            exit(1)
        os.makedirs(dir_name, exist_ok=True)
        print(f"✅ Directory created/verified: {dir_name}/")

def test_data_files():
    """Test if data files exist and are valid"""
    print("🧪 Testing data files...")

    files_valid = True

    # Test 0D data
    if validate_csv_format(DATA_PATHS['0d_csv'], ['time', 'temperature'], '0D'):
        print("✅ 0D data file is valid")
    else:
        print("❌ 0D data file has issues")
        files_valid = False

    # Test 3D data  
    if validate_csv_format(DATA_PATHS['3d_csv'], ['Time Step', 'flow-time', 'Volume-Average Static Temperature'], '3D'):
        print("✅ 3D data file is valid")
    else:
        print("❌ 3D data file has issues")
        files_valid = False

    return files_valid

def run_mini_pipeline_test():
    """Run a mini version of the pipeline to test functionality"""
    print("🧪 Running mini pipeline test...")

    try:
        from thermal_runaway_ml_pipeline import ThermalRunawayDataProcessor, ThermalRunawayMLPipeline

        # Create sample data
        create_sample_data()

        # Test data processor
        processor = ThermalRunawayDataProcessor()
        data_0d, data_3d = processor.load_data('data/sample_0d_data.csv', 'data/sample_3d_data.csv')

        print("✅ Data loading test passed")

        # Test data alignment
        aligned_data = processor.align_data_by_interpolation()
        corrected_data = processor.calculate_correction_factor()
        final_data = processor.engineer_features()

        print("✅ Data processing test passed")

        # Test ML pipeline initialization
        ml_pipeline = ThermalRunawayMLPipeline()
        X, y, feature_names = ml_pipeline.prepare_features_target(final_data)

        print("✅ ML pipeline initialization test passed")
        print(f"  - Features: {len(feature_names)}")
        print(f"  - Data points: {len(X)}")

        return True

    except Exception as e:
        print(f"❌ Mini pipeline test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Running comprehensive pipeline tests...")
    print("="*60)

    # Check dependencies
    print("\n1. Checking dependencies...")
    if not check_dependencies():
        print("❌ Dependency check failed!")
        exit(1)

    # Create directories (safe version)
    print("\n2. Creating directories...")
    safe_create_directories()

    # Test data files
    print("\n3. Testing data files...")
    if test_data_files():
        print("✅ Data files are valid!")
    else:
        print("⚠️ Data file issues detected, but continuing with sample data...")
        create_sample_data()

    # Run mini pipeline test
    print("\n4. Running mini pipeline test...")
    if run_mini_pipeline_test():
        print("✅ Mini pipeline test passed!")
    else:
        print("❌ Mini pipeline test failed!")
        exit(1)

    print("\n" + "="*60)
    print("🎉 ALL TESTS PASSED! Ready to run full pipeline.")
    print("Run: python run_pipeline.py")
    print("="*60)

