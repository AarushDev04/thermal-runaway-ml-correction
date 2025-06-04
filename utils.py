"""
Utility functions for the thermal runaway ML pipeline
"""

import pandas as pd
import numpy as np
import os

def validate_csv_format(csv_path, expected_columns, csv_type):
    """
    Validate CSV/Excel file format before processing

    Args:
        csv_path: Path to CSV/Excel file
        expected_columns: List of expected column names
        csv_type: Type of CSV ('0D' or '3D')

    Returns:
        bool: True if valid, False otherwise
    """
    try:
        # Check if file exists
        if not os.path.exists(csv_path):
            print(f"❌ File not found: {csv_path}")
            return False

        # Load file based on extension
        if csv_path.endswith('.xlsx'):
            df = pd.read_excel(csv_path)
        else:
            df = pd.read_csv(csv_path)

        print(f"📋 Validating {csv_type} file format...")
        print(f"  - File: {csv_path}")
        print(f"  - Shape: {df.shape}")
        print(f"  - Columns: {df.columns.tolist()}")

        # Check for empty data
        if len(df) == 0:
            print("❌ File is empty!")
            return False

        # Check if we have enough columns
        if len(df.columns) < len(expected_columns):
            print(f"❌ Insufficient columns. Expected at least {len(expected_columns)}, got {len(df.columns)}")
            return False

        # Check for numeric data in first few columns
        numeric_cols = 0
        for i in range(min(3, len(df.columns))):
            if pd.api.types.is_numeric_dtype(df.iloc[:, i]):
                numeric_cols += 1

        if numeric_cols < 2:
            print(f"❌ Insufficient numeric columns!")
            return False

        print(f"✅ {csv_type} file format is valid!")
        return True

    except Exception as e:
        print(f"❌ Error validating {csv_type} file: {e}")
        return False

def check_dependencies():
    """
    Check if all required packages are installed
    """
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn',
        'sklearn', 'xgboost', 'joblib'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            if package == 'sklearn':
                import sklearn
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"❌ Missing packages: {missing_packages}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    else:
        print("✅ All required packages are installed!")
        return True

def create_directories():
    """
    Create necessary directories for the project
    """
    dirs = ['data', 'models', 'results', 'plots']
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        print(f"✅ Directory created/verified: {dir_name}/")

def create_sample_data():
    """
    Create sample data for testing if real data is not available
    """
    print("🔧 Creating sample data for testing...")

    # Create sample 0D data
    time_0d = np.linspace(0, 100, 500)
    temp_0d = 25 + 50 * np.exp(time_0d/50) + np.random.normal(0, 2, len(time_0d))

    df_0d = pd.DataFrame({
        'time': time_0d,
        'temperature': temp_0d
    })

    # Create sample 3D data
    time_3d = np.linspace(0, 100, 480)
    temp_3d = 25 + 55 * np.exp(time_3d/48) + np.random.normal(0, 3, len(time_3d))

    df_3d = pd.DataFrame({
        'Time Step': range(len(time_3d)),
        'flow-time': time_3d,
        'Volume-Average Static Temperature': temp_3d
    })

    # Save sample data
    create_directories()
    df_0d.to_csv('data/sample_0d_data.csv', index=False)
    df_3d.to_csv('data/sample_3d_data.csv', index=False)

    print("✅ Sample data created:")
    print("  - data/sample_0d_data.csv")
    print("  - data/sample_3d_data.csv")

    return True

def predict_correction_factor(new_0d_data, trained_model, scaler, feature_names):
    """
    Use trained model to predict correction factors for new data

    Args:
        new_0d_data: DataFrame with columns ['time', 'temperature_0d']
        trained_model: The trained ML model
        scaler: The fitted feature scaler
        feature_names: List of feature names used in training

    Returns:
        DataFrame: Enhanced predictions with correction factors
    """

    print("\n" + "="*60)
    print("🔮 PREDICTING CORRECTION FACTORS FOR NEW DATA")
    print("="*60)

    try:
        processed_data = new_0d_data.copy()

        # Ensure correct column names
        if 'time' not in processed_data.columns or 'temperature_0d' not in processed_data.columns:
            print("❌ Error: New data must have columns 'time' and 'temperature_0d'")
            return None

        print(f"📊 Processing {len(processed_data)} new data points...")

        # Feature engineering (same as training)
        print("🔧 Engineering features...")

        # Time normalization
        processed_data['time_normalized'] = (
            (processed_data['time'] - processed_data['time'].min()) /
            (processed_data['time'].max() - processed_data['time'].min())
        )

        # Temperature normalization
        processed_data['temp_0d_normalized'] = (
            (processed_data['temperature_0d'] - processed_data['temperature_0d'].min()) /
            (processed_data['temperature_0d'].max() - processed_data['temperature_0d'].min())
        )

        # Temperature derivative
        processed_data['temp_0d_derivative'] = np.gradient(processed_data['temperature_0d'])

        # Rolling statistics
        window_size = max(2, min(5, len(processed_data) // 10))
        if window_size >= 2:
            processed_data['temp_0d_rolling_mean'] = (
                processed_data['temperature_0d'].rolling(window=window_size, center=True).mean()
            )
            processed_data['temp_0d_rolling_std'] = (
                processed_data['temperature_0d'].rolling(window=window_size, center=True).std()
            )

        # Fill NaN values
        processed_data = processed_data.fillna(method='bfill').fillna(method='ffill')

        # Temperature range indicators
        processed_data['temp_range_low'] = (processed_data['temperature_0d'] < 60).astype(int)
        processed_data['temp_range_medium'] = ((processed_data['temperature_0d'] >= 60) &
                                              (processed_data['temperature_0d'] < 120)).astype(int)
        processed_data['temp_range_high'] = (processed_data['temperature_0d'] >= 120).astype(int)

        # Interaction features
        processed_data['temp_time_interaction'] = (
            processed_data['temperature_0d'] * processed_data['time_normalized']
        )

        # Physics-based features
        activation_energy = 50000  # J/mol
        gas_constant = 8.314  # J/(mol·K)
        processed_data['arrhenius_factor'] = np.exp(-activation_energy /
                                                   (gas_constant * (processed_data['temperature_0d'] + 273.15)))

        # Select only the features used in training
        try:
            X_new = processed_data[feature_names]
        except KeyError as e:
            print(f"❌ Error: Missing feature {e}. Check if new data has same structure as training data.")
            return None

        # Scale features
        X_new_scaled = scaler.transform(X_new)

        # Make predictions
        print("🤖 Making predictions...")
        correction_factors = trained_model.predict(X_new_scaled)

        # Calculate corrected temperatures
        corrected_temperatures = processed_data['temperature_0d'] * correction_factors

        # Create results DataFrame
        results = processed_data[['time', 'temperature_0d']].copy()
        results['predicted_correction_factor'] = correction_factors
        results['predicted_temperature_3d'] = corrected_temperatures
        results['temperature_difference'] = corrected_temperatures - processed_data['temperature_0d']
        results['improvement_percentage'] = (results['temperature_difference'] / processed_data['temperature_0d'] * 100)

        print("✅ Predictions completed successfully!")
        print(f"📊 PREDICTION SUMMARY:")
        print(f"  - Average correction factor: {correction_factors.mean():.4f}")
        print(f"  - Correction factor range: {correction_factors.min():.4f} to {correction_factors.max():.4f}")
        print(f"  - Average temperature improvement: {results['temperature_difference'].mean():.2f}°C")

        return results

    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        return None

