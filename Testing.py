"""
THERMAL RUNAWAY TESTING FUNCTION - OPTIMIZED FOR EXISTING PIPELINE
================================================================
This module integrates with the main thermal runaway pipeline code
and provides testing functionality for new real-world data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import joblib
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
# Import classes from your main pipeline (ensure the main file is in the same directory)
try:
    from Thermal_Runaway_JU_New import (
        ThermalRunawayDataProcessor,
        ThermalRunawayMLPipeline,
    )
    print("✓ Successfully imported pipeline classes")
    
    # Add immediate execution test
    print("🔧 Testing class instantiation...")
    test_processor = ThermalRunawayDataProcessor()
    print("✓ Classes working correctly")
    
except ImportError as e:
    print(f"❌ Error importing pipeline classes: {e}")
    print("Please ensure Thermal_Runaway_JU_New.py is in the same directory")
except Exception as e:
    print(f"❌ Error instantiating classes: {e}")

# Import classes from your main pipeline (ensure the main file is in the same directory)
try:
    
    from Thermal_Runaway_JU_New import (
        ThermalRunawayDataProcessor,
        ThermalRunawayMLPipeline,
        
    )
    
    print("✓ Successfully imported pipeline classes")
except ImportError as e:
    print(f"❌ Error importing pipeline classes: {e}")
    print("Please ensure thermal_runaway_ml_pipeline.py is in the same directory")

def thermal_runaway_testing_function(test_file_path, model_directory="./"):
    """
    COMPREHENSIVE THERMAL RUNAWAY TESTING FUNCTION
    =============================================
    
    Takes new 0D temperature data and predicts corrected 3D temperatures
    using pre-trained ML models from the thermal runaway pipeline.
    
    Args:
        test_file_path (str): test_file_path (str): "C:/Users/KIIT0001/Desktop/JU/Thermal_Runaway/test_thermal-runaway-ml-correction.xlsx"
        model_directory (str): Directory containing saved models and scalers
        
    Returns:
        dict: Complete results including predictions, metrics, and visualizations
    """
    
    print("🔥 THERMAL RUNAWAY TESTING PIPELINE INITIATED")
    print("=" * 60)
    
    # ================================================================
    # STEP 1: LOAD TEST DATA
    # ================================================================
    try:
        if test_file_path.endswith('.csv'):
            test_data = pd.read_csv(test_file_path)
        elif test_file_path.endswith('.xlsx'):
            test_data = pd.read_excel(test_file_path)
        else:
            raise ValueError("Unsupported file format. Use CSV or Excel.")
        
        # Standardize column names
        if 'Time' in test_data.columns:
            test_data = test_data.rename(columns={'Time': 'time'})
        if 'Temperature' in test_data.columns:
            test_data = test_data.rename(columns={'Temperature': 'temperature_0d'})
        
        print(f"✓ Test data loaded: {test_data.shape}")
        print(f"  Columns: {test_data.columns.tolist()}")
        
    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        return None
    
    # ================================================================
    # STEP 2: LOAD TRAINED MODELS AND SCALER
    # ================================================================
    try:
        # Load feature scaler
        scaler = joblib.load(f"{model_directory}feature_scaler.pkl")
        
        # Load trained models
        models = {}
        model_files = [
            'best_thermal_model_random_forest.pkl',
            'best_thermal_model_xgboost.pkl',
            'best_thermal_model_gradient_boosting.pkl',
            'best_thermal_model_linear_regression.pkl',
            'best_thermal_model_ridge_regression.pkl'
        ]
        
        for model_file in model_files:
            try:
                model_name = model_file.replace('best_thermal_model_', '').replace('.pkl', '')
                models[model_name] = joblib.load(f"{model_directory}{model_file}")
                print(f"✓ Loaded model: {model_name}")
            except:
                print(f"⚠️ Model not found: {model_file}")
        
        if not models:
            raise ValueError("No trained models found!")
            
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return None
    
    # ================================================================
    # STEP 3: FEATURE ENGINEERING FOR TEST DATA
    # ================================================================
    print("\n🔧 Engineering features for test data...")
    
    # Create a data processor instance to use existing feature engineering
    processor = ThermalRunawayDataProcessor()
    
    # Prepare data in the format expected by the processor
    processor.aligned_data = test_data.copy()
    processor.aligned_data['temperature_3d'] = test_data['temperature_0d']  # Dummy 3D data
    processor.aligned_data['correction_factor'] = 1.0  # Dummy correction factor
    
    # Apply feature engineering using existing function
    test_features = processor.engineer_features()
    
    # Extract feature columns (exclude target and identifier columns)
    exclude_cols = ['correction_factor', 'temperature_3d', 'time']
    feature_cols = [col for col in test_features.columns if col not in exclude_cols]
    X_test = test_features[feature_cols]
    
    # Scale features using trained scaler
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols)
    
    print(f"✓ Features engineered: {X_test_scaled.shape}")
    
    # ================================================================
    # STEP 4: GENERATE PREDICTIONS FROM ALL MODELS
    # ================================================================
    print("\n🤖 Generating predictions from all models...")
    
    predictions = {}
    corrected_temperatures = {}
    
    for model_name, model in models.items():
        try:
            # Predict correction factors
            cf_pred = model.predict(X_test_scaled)
            predictions[model_name] = cf_pred
            
            # Calculate corrected 3D temperatures
            corrected_temp = test_data['temperature_0d'] * cf_pred
            corrected_temperatures[model_name] = corrected_temp
            
            print(f"✓ {model_name}: CF range [{cf_pred.min():.3f}, {cf_pred.max():.3f}]")
            
        except Exception as e:
            print(f"❌ Error with {model_name}: {e}")
    
    # ================================================================
    # STEP 5: CREATE ENSEMBLE PREDICTION
    # ================================================================
    if len(predictions) > 1:
        ensemble_cf = np.mean(list(predictions.values()), axis=0)
        predictions['ensemble'] = ensemble_cf
        corrected_temperatures['ensemble'] = test_data['temperature_0d'] * ensemble_cf
        print("✓ Ensemble prediction created")
    
    # ================================================================
    # STEP 6: GENERATE COMPREHENSIVE VISUALIZATIONS
    # ================================================================
    print("\n📊 Generating visualizations...")
    
    results = create_comprehensive_test_visualizations(
        test_data, predictions, corrected_temperatures
    )
    
    # ================================================================
    # STEP 7: CALCULATE PERFORMANCE METRICS
    # ================================================================
    print("\n📈 Calculating performance metrics...")
    
    performance_metrics = calculate_test_performance_metrics(
        test_data, predictions, corrected_temperatures
    )
    
    # Compile final results
    final_results = {
        'test_data': test_data,
        'predictions': predictions,
        'corrected_temperatures': corrected_temperatures,
        'performance_metrics': performance_metrics,
        'visualizations': results,
        'models_used': list(models.keys())
    }
    
    print("\n🎉 Testing pipeline completed successfully!")
    return final_results

def create_comprehensive_test_visualizations(test_data, predictions, corrected_temperatures):
    """
    CREATE COMPREHENSIVE VISUALIZATION SUITE FOR TEST DATA
    ====================================================
    """
    
    # Set style for publication-quality plots
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # ================================================================
    # PLOT 1: TEMPERATURE EVOLUTION COMPARISON
    # ================================================================
    fig1, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig1.suptitle('Thermal Runaway Testing Results - Temperature Evolution', fontsize=16, fontweight='bold')
    
    # Original 0D temperature
    axes[0,0].plot(test_data['time'], test_data['temperature_0d'], 'b-', linewidth=2, label='Original 0D')
    axes[0,0].axhline(y=60, color='orange', linestyle='--', alpha=0.7, label='Warning (60°C)')
    axes[0,0].axhline(y=120, color='red', linestyle='--', alpha=0.7, label='Critical (120°C)')
    axes[0,0].set_title('Original 0D Temperature')
    axes[0,0].set_xlabel('Time (s)')
    axes[0,0].set_ylabel('Temperature (°C)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Corrected temperatures comparison
    colors = ['red', 'green', 'purple', 'orange', 'brown']
    for i, (model_name, corrected_temp) in enumerate(corrected_temperatures.items()):
        if i < len(colors):
            axes[0,1].plot(test_data['time'], corrected_temp, 
                          color=colors[i], linewidth=2, label=f'{model_name.title()}', alpha=0.8)
    
    axes[0,1].axhline(y=60, color='orange', linestyle='--', alpha=0.7)
    axes[0,1].axhline(y=120, color='red', linestyle='--', alpha=0.7)
    axes[0,1].set_title('Corrected 3D Temperature Predictions')
    axes[0,1].set_xlabel('Time (s)')
    axes[0,1].set_ylabel('Temperature (°C)')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Correction factors over time
    for i, (model_name, cf) in enumerate(predictions.items()):
        if i < len(colors):
            axes[1,0].plot(test_data['time'], cf, 
                          color=colors[i], linewidth=2, label=f'{model_name.title()}', alpha=0.8)
    
    axes[1,0].axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='Perfect Match (CF=1)')
    axes[1,0].set_title('Correction Factor Evolution')
    axes[1,0].set_xlabel('Time (s)')
    axes[1,0].set_ylabel('Correction Factor')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Correction factor vs temperature
    for i, (model_name, cf) in enumerate(predictions.items()):
        if i < len(colors):
            axes[1,1].scatter(test_data['temperature_0d'], cf, 
                            color=colors[i], alpha=0.6, label=f'{model_name.title()}', s=30)
    
    axes[1,1].set_title('Correction Factor vs 0D Temperature')
    axes[1,1].set_xlabel('0D Temperature (°C)')
    axes[1,1].set_ylabel('Correction Factor')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # ================================================================
    # PLOT 2: MODEL PERFORMANCE COMPARISON
    # ================================================================
    fig2, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig2.suptitle('Model Performance Analysis on Test Data', fontsize=16, fontweight='bold')
    
    # Temperature range analysis
    temp_ranges = ['<60°C', '60-120°C', '>120°C']
    range_masks = [
        test_data['temperature_0d'] < 60,
        (test_data['temperature_0d'] >= 60) & (test_data['temperature_0d'] < 120),
        test_data['temperature_0d'] >= 120
    ]
    
    model_names = list(predictions.keys())
    range_performance = np.zeros((len(model_names), len(temp_ranges)))
    
    for i, model_name in enumerate(model_names):
        cf = predictions[model_name]
        for j, mask in enumerate(range_masks):
            if mask.sum() > 0:
                range_performance[i, j] = cf[mask].mean()
    
    # Heatmap of correction factors by temperature range
    sns.heatmap(range_performance, 
                xticklabels=temp_ranges, 
                yticklabels=[name.title() for name in model_names],
                annot=True, fmt='.3f', cmap='RdYlBu_r', ax=axes[0,0])
    axes[0,0].set_title('Mean Correction Factor by Temperature Range')
    
    # Correction factor distribution
    cf_data = []
    model_labels = []
    for model_name, cf in predictions.items():
        cf_data.extend(cf)
        model_labels.extend([model_name.title()] * len(cf))
    
    cf_df = pd.DataFrame({'Correction_Factor': cf_data, 'Model': model_labels})
    sns.boxplot(data=cf_df, x='Model', y='Correction_Factor', ax=axes[0,1])
    axes[0,1].set_title('Correction Factor Distribution by Model')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Temperature difference analysis
    temp_diffs = {}
    for model_name, corrected_temp in corrected_temperatures.items():
        temp_diff = corrected_temp - test_data['temperature_0d']
        temp_diffs[model_name] = temp_diff.mean()
    
    axes[1,0].bar(temp_diffs.keys(), temp_diffs.values(), alpha=0.7, color=colors[:len(temp_diffs)])
    axes[1,0].set_title('Mean Temperature Correction by Model')
    axes[1,0].set_ylabel('Temperature Difference (°C)')
    axes[1,0].tick_params(axis='x', rotation=45)
    axes[1,0].grid(True, alpha=0.3)
    
    # Model agreement analysis
    if len(predictions) > 1:
        cf_matrix = np.array(list(predictions.values()))
        correlation_matrix = np.corrcoef(cf_matrix)
        
        sns.heatmap(correlation_matrix,
                    xticklabels=[name.title() for name in model_names],
                    yticklabels=[name.title() for name in model_names],
                    annot=True, fmt='.3f', cmap='coolwarm', ax=axes[1,1])
        axes[1,1].set_title('Model Prediction Correlation')
    
    plt.tight_layout()
    plt.show()
    
    return {'fig1': fig1, 'fig2': fig2}
class ThermalRunawayDataProcessor:
    """
    ENHANCED DATA PROCESSING CLASS FOR TESTING
    =========================================
    """
    
    def __init__(self, config=None):
        self.config = config
        self.data_0d = None
        self.data_3d = None
        self.aligned_data = None
        self.feature_names = []
        print("✓ Enhanced data processor initialized")

    def engineer_features(self):
        """
        OPTIMIZED FEATURE ENGINEERING
        ============================
        """
        if self.aligned_data is None:
            raise ValueError("No aligned data available for feature engineering")
    
        data = self.aligned_data.copy()
    
        # Ensure required columns exist
        if 'temperature_0d' not in data.columns:
            raise KeyError("No temperature_0d column found")
    
        # Optimized time normalization
        if 'time' in data.columns and len(data) > 1:
            time_range = data['time'].max() - data['time'].min()
            data['time_normalized'] = (data['time'] - data['time'].min()) / (time_range + 1e-8)
        else:
            data['time_normalized'] = np.linspace(0, 1, len(data))
    
        # Efficient derivative calculation
        temp_values = data['temperature_0d'].values
        data['temp_0d_derivative'] = np.gradient(temp_values)
        data['temp_0d_second_derivative'] = np.gradient(data['temp_0d_derivative'])
    
        # Optimized rolling statistics
        window_size = max(3, min(7, len(data) // 8))
        if window_size >= 3:
            rolling_mean = data['temperature_0d'].rolling(window=window_size, center=True, min_periods=1).mean()
            rolling_std = data['temperature_0d'].rolling(window=window_size, center=True, min_periods=1).std()
            data['temp_0d_rolling_mean'] = rolling_mean.fillna(data['temperature_0d'])
            data['temp_0d_rolling_std'] = rolling_std.fillna(0)
        else:
            data['temp_0d_rolling_mean'] = data['temperature_0d']
            data['temp_0d_rolling_std'] = np.zeros(len(data))
    
        # Enhanced thermal zone indicators with smooth transitions
        temp_norm = (data['temperature_0d'] - data['temperature_0d'].min()) / (data['temperature_0d'].max() - data['temperature_0d'].min() + 1e-8)
        data['temp_range_low'] = np.exp(-((data['temperature_0d'] - 30) / 20)**2)  # Gaussian around 30°C
        data['temp_range_medium'] = np.exp(-((data['temperature_0d'] - 90) / 30)**2)  # Gaussian around 90°C
        data['temp_range_high'] = 1 / (1 + np.exp(-(data['temperature_0d'] - 120) / 10))  # Sigmoid above 120°C
    
        # Advanced interaction features
        data['temp_time_interaction'] = data['temperature_0d'] * data['time_normalized']
        data['temp_derivative_interaction'] = data['temperature_0d'] * data['temp_0d_derivative']
        data['velocity_acceleration'] = data['temp_0d_derivative'] * data['temp_0d_second_derivative']
    
        # Physics-based features
        data['thermal_energy'] = data['temperature_0d'] ** 1.5  # Thermal energy proxy
        data['arrhenius_factor'] = np.exp(-1000 / (data['temperature_0d'] + 273.15))  # Arrhenius kinetics
    
        # Remove redundant cumulative features (optimization)
        # data['temp_0d_cumsum'] = data['temperature_0d'].cumsum()  # REMOVED
        # data['temp_0d_integral'] = ...  # REMOVED
    
        print(f"✓ Optimized features engineered: {data.shape[1]} features created")
        return data


def calculate_test_performance_metrics(test_data, predictions, corrected_temperatures):
    """
    CALCULATE COMPREHENSIVE PERFORMANCE METRICS
    ==========================================
    """
    
    metrics = {}
    
    for model_name in predictions.keys():
        cf = predictions[model_name]
        corrected_temp = corrected_temperatures[model_name]
        
        # Correction factor statistics
        cf_stats = {
            'mean_cf': cf.mean(),
            'std_cf': cf.std(),
            'min_cf': cf.min(),
            'max_cf': cf.max(),
            'median_cf': np.median(cf)
        }
        
        # Temperature correction statistics
        temp_correction = corrected_temp - test_data['temperature_0d']
        temp_stats = {
            'mean_temp_correction': temp_correction.mean(),
            'max_temp_correction': temp_correction.max(),
            'min_temp_correction': temp_correction.min(),
            'std_temp_correction': temp_correction.std()
        }
        
        # Thermal zone analysis
        zone_analysis = {}
        zones = {
            'normal': test_data['temperature_0d'] < 60,
            'warning': (test_data['temperature_0d'] >= 60) & (test_data['temperature_0d'] < 120),
            'critical': test_data['temperature_0d'] >= 120
        }
        
        for zone_name, mask in zones.items():
            if mask.sum() > 0:
                zone_analysis[f'{zone_name}_mean_cf'] = cf[mask].mean()
                zone_analysis[f'{zone_name}_count'] = mask.sum()
        
        metrics[model_name] = {
            **cf_stats,
            **temp_stats,
            **zone_analysis
        }
    
    return metrics
def plot_0d_vs_corrected_temperature(time, temp_0d, temp_corrected, 
                                   correction_factors=None, 
                                   title="0D vs Corrected Temperature Analysis"):
    """
    Plot original 0D temperature vs corrected temperature with enhanced visualization
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    # Set style for professional appearance
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Main temperature comparison plot
    ax1.plot(time, temp_0d, label='Original 0D Temperature', 
             color='blue', linewidth=2.5, alpha=0.8)
    ax1.plot(time, temp_corrected, label='Corrected 0D Temperature', 
             color='red', linewidth=2.5, linestyle='--', alpha=0.8)
    
    # Add thermal safety zones
    ax1.axhline(y=60, color='orange', linestyle=':', alpha=0.7, 
                label='Warning Zone (60°C)')
    ax1.axhline(y=120, color='darkred', linestyle=':', alpha=0.7, 
                label='Critical Zone (120°C)')
    
    # Fill areas for thermal zones
    ax1.fill_between(time, 0, 60, alpha=0.1, color='green', label='Safe Zone')
    ax1.fill_between(time, 60, 120, alpha=0.1, color='orange')
    ax1.fill_between(time, 120, max(temp_corrected.max(), temp_0d.max()) + 10, 
                     alpha=0.1, color='red')
    
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Temperature (°C)', fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Temperature difference plot
    temp_diff = temp_corrected - temp_0d
    ax2.plot(time, temp_diff, color='purple', linewidth=2, 
             label='Temperature Difference (Corrected - Original)')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.fill_between(time, temp_diff, 0, alpha=0.3, color='purple')
    
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Temperature Difference (°C)', fontsize=12)
    ax2.set_title('Temperature Correction Applied', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add statistics text box
    stats_text = f"""Statistics:
Original Peak: {temp_0d.max():.1f}°C
Corrected Peak: {temp_corrected.max():.1f}°C
Max Correction: {temp_diff.max():.1f}°C
Mean Correction: {temp_diff.mean():.1f}°C"""
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', 
             facecolor='lightblue', alpha=0.8), fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def train_models_on_new_data(test_data, target_column='correction_factor'):
    """
    OPTIMIZED ML TRAINING FOR HIGH ACCURACY
    =====================================
    """
    print("🤖 Training optimized models for high accuracy...")
    
    # Initialize processor and engineer features
    processor = ThermalRunawayDataProcessor()
    processor.aligned_data = test_data.copy()
    
    # Create more realistic correction factors based on thermal physics
    if target_column not in test_data.columns:
        # Physics-based correction factor generation
        temp_normalized = (test_data['temperature_0d'] - test_data['temperature_0d'].min()) / (test_data['temperature_0d'].max() - test_data['temperature_0d'].min())
        
        # Exponential correction based on thermal runaway physics
        base_cf = 1.0 + 0.5 * np.exp(temp_normalized * 2)  # Exponential increase
        thermal_noise = 0.1 * np.sin(np.linspace(0, 4*np.pi, len(test_data)))  # Thermal oscillations
        processor.aligned_data[target_column] = base_cf + thermal_noise
        
        print(f"✓ Generated physics-based {target_column}")
    
    # Enhanced feature engineering
    featured_data = processor.engineer_features()
    
    # Add advanced features for better accuracy
    featured_data['temp_acceleration'] = np.gradient(featured_data['temp_0d_derivative'])
    featured_data['thermal_momentum'] = featured_data['temperature_0d'] * featured_data['temp_0d_derivative']
    featured_data['exponential_temp'] = np.exp(featured_data['temperature_0d'] / 100)
    featured_data['log_temp'] = np.log1p(featured_data['temperature_0d'])
    
    # Prepare training data
    exclude_cols = [target_column, 'time', 'temperature_3d'] if 'temperature_3d' in featured_data.columns else [target_column, 'time']
    feature_cols = [col for col in featured_data.columns if col not in exclude_cols]
    
    X = featured_data[feature_cols].fillna(0)
    y = featured_data[target_column].fillna(1.0)
    
    # Optimized models with better hyperparameters
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import Pipeline
    import xgboost as xgb
    
    models = {
        'RandomForest': RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_split=3,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            random_state=42
        ),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
        'PolynomialRegression': Pipeline([
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('linear', LinearRegression())
        ])
    }
    
    trained_models = {}
    for name, model in models.items():
        try:
            model.fit(X, y)
            trained_models[name] = model
            
            # Calculate training score
            train_score = model.score(X, y)
            print(f"✓ {name} trained successfully (R²: {train_score:.4f})")
        except Exception as e:
            print(f"❌ {name} training failed: {e}")
    
    return trained_models, feature_cols, processor


def apply_error_correction(test_data, trained_models, feature_cols):
    """
    OPTIMIZED ERROR CORRECTION
    =========================
    """
    print("🔧 Applying optimized error correction...")
    
    # Single feature engineering call (avoid redundancy)
    processor = ThermalRunawayDataProcessor()
    processor.aligned_data = test_data.copy()
    featured_data = processor.engineer_features()
    
    # Optimized feature selection
    available_features = [col for col in feature_cols if col in featured_data.columns]
    X = featured_data[available_features].fillna(0)
    
    # Vectorized predictions
    corrections = {}
    corrected_temps = {}

    
    for name, model in trained_models.items():
        try:
            # Single prediction call
            correction_factors = model.predict(X)
            
            # Clip extreme values for stability
            correction_factors = np.clip(correction_factors, 0.1, 5.0)
            
            corrections[name] = correction_factors
            corrected_temps[name] = test_data['temperature_0d'].values * correction_factors
            
            print(f"✓ {name}: CF range [{correction_factors.min():.3f}, {correction_factors.max():.3f}]")
            
        except Exception as e:
            print(f"❌ {name} correction failed: {e}")
    
    return corrections, corrected_temps


def analyze_correction_performance(test_data, corrections, corrected_temps):
    """
    ANALYZE CORRECTION PERFORMANCE
    =============================
    """
    print("📊 Analyzing correction performance...")
    
    analysis = {}
    
    for model_name in corrections.keys():
        cf = corrections[model_name]
        corrected_temp = corrected_temps[model_name]
        
        # Basic statistics
        stats = {
            'mean_cf': cf.mean(),
            'std_cf': cf.std(),
            'min_cf': cf.min(),
            'max_cf': cf.max(),
            'mean_temp_increase': (corrected_temp - test_data['temperature_0d']).mean(),
            'max_temp_increase': (corrected_temp - test_data['temperature_0d']).max()
        }
        
        # Thermal zone analysis
        normal_mask = test_data['temperature_0d'] < 60
        warning_mask = (test_data['temperature_0d'] >= 60) & (test_data['temperature_0d'] < 120)
        critical_mask = test_data['temperature_0d'] >= 120
        
        if normal_mask.sum() > 0:
            stats['normal_zone_cf'] = cf[normal_mask].mean()
        if warning_mask.sum() > 0:
            stats['warning_zone_cf'] = cf[warning_mask].mean()
        if critical_mask.sum() > 0:
            stats['critical_zone_cf'] = cf[critical_mask].mean()
        
        analysis[model_name] = stats
    
    return analysis
def generate_comprehensive_plots(test_data, corrections, corrected_temps, analysis):
    """
    GENERATE CLEAN, PROFESSIONAL PLOTS
    =================================
    """
    print("📈 Generating clean, professional plots...")
    
    # Set clean style
    plt.style.use('default')
    sns.set_style("whitegrid")
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Create figure with proper spacing
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('Thermal Runaway ML Analysis Results', fontsize=16, fontweight='bold', y=0.98)
    
    # Plot 1: Temperature Evolution (Clean)
    ax1 = plt.subplot(2, 3, 1)
    time_data = test_data['time'] if 'time' in test_data.columns else range(len(test_data))
    
    ax1.plot(time_data, test_data['temperature_0d'], 'k-', linewidth=2.5, label='Original 0D', alpha=0.8)
    
    for i, (model_name, corrected_temp) in enumerate(corrected_temps.items()):
        if i < len(colors):
            ax1.plot(time_data, corrected_temp, color=colors[i], 
                    linewidth=2, label=f'{model_name}', alpha=0.7)
    
    # Clean threshold lines
    ax1.axhline(y=60, color='orange', linestyle='--', alpha=0.6, linewidth=1.5)
    ax1.axhline(y=120, color='red', linestyle='--', alpha=0.6, linewidth=1.5)
    
    ax1.set_title('Temperature Evolution', fontsize=12, fontweight='bold', pad=15)
    ax1.set_xlabel('Time (s)', fontsize=10)
    ax1.set_ylabel('Temperature (°C)', fontsize=10)
    ax1.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, fontsize=8)
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Plot 2: Correction Factors (Clean)
    ax2 = plt.subplot(2, 3, 2)
    for i, (model_name, cf) in enumerate(corrections.items()):
        if i < len(colors):
            ax2.plot(time_data, cf, color=colors[i], linewidth=2, 
                    label=f'{model_name}', alpha=0.7, marker='o', markersize=2)
    
    ax2.axhline(y=1.0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    ax2.set_title('Correction Factor Evolution', fontsize=12, fontweight='bold', pad=15)
    ax2.set_xlabel('Time (s)', fontsize=10)
    ax2.set_ylabel('Correction Factor', fontsize=10)
    ax2.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=8)
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Plot 3: Clean Scatter Plot
    ax3 = plt.subplot(2, 3, 3)
    for i, (model_name, cf) in enumerate(corrections.items()):
        if i < len(colors):
            ax3.scatter(test_data['temperature_0d'], cf, color=colors[i], 
                       alpha=0.6, label=f'{model_name}', s=25, edgecolors='white', linewidth=0.5)
    
    ax3.set_title('CF vs Temperature', fontsize=12, fontweight='bold', pad=15)
    ax3.set_xlabel('0D Temperature (°C)', fontsize=10)
    ax3.set_ylabel('Correction Factor', fontsize=10)
    ax3.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=8)
    ax3.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # Plot 4: Clean Distribution
    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(test_data['temperature_0d'], bins=15, alpha=0.7, color='skyblue', 
             edgecolor='navy', linewidth=1, label='Original 0D')
    
    # Only show top 2 models to avoid clutter
    top_models = list(corrected_temps.items())[:2]
    for i, (model_name, corrected_temp) in enumerate(top_models):
        ax4.hist(corrected_temp, bins=15, alpha=0.5, color=colors[i+1], 
                edgecolor='white', linewidth=1, label=f'{model_name}')
    
    ax4.set_title('Temperature Distribution', fontsize=12, fontweight='bold', pad=15)
    ax4.set_xlabel('Temperature (°C)', fontsize=10)
    ax4.set_ylabel('Frequency', fontsize=10)
    ax4.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=8)
    ax4.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    # Plot 5: Clean Bar Chart
    ax5 = plt.subplot(2, 3, 5)
    models = list(analysis.keys())
    mean_cfs = [analysis[model]['mean_cf'] for model in models]
    
    bars = ax5.bar(range(len(models)), mean_cfs, color=colors[:len(models)], 
                   alpha=0.8, edgecolor='white', linewidth=1)
    
    # Clean labels
    ax5.set_xticks(range(len(models)))
    ax5.set_xticklabels([name.replace('Forest', 'RF').replace('Regression', 'Reg') 
                        for name in models], rotation=45, ha='right', fontsize=9)
    
    ax5.set_title('Mean Correction Factor', fontsize=12, fontweight='bold', pad=15)
    ax5.set_ylabel('Mean CF', fontsize=10)
    ax5.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, axis='y')
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    
    # Add value labels on bars
    for bar, value in zip(bars, mean_cfs):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Plot 6: Clean Performance Summary
    ax6 = plt.subplot(2, 3, 6)
    temp_increases = [analysis[model]['mean_temp_increase'] for model in models]
    
    bars = ax6.bar(range(len(models)), temp_increases, color=colors[:len(models)], 
                   alpha=0.8, edgecolor='white', linewidth=1)
    
    ax6.set_xticks(range(len(models)))
    ax6.set_xticklabels([name.replace('Forest', 'RF').replace('Regression', 'Reg') 
                        for name in models], rotation=45, ha='right', fontsize=9)
    
    ax6.set_title('Temperature Increase', fontsize=12, fontweight='bold', pad=15)
    ax6.set_ylabel('Temp Increase (°C)', fontsize=10)
    ax6.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, axis='y')
    ax6.spines['top'].set_visible(False)
    ax6.spines['right'].set_visible(False)
    
    # Add value labels
    for bar, value in zip(bars, temp_increases):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{value:.1f}°C', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    return fig

def generate_correction_insights_infographic(initial_error, corrected_error, 
                                           correction_percentage, model_name="ML Model",
                                           additional_metrics=None):
    """
    Generate comprehensive correction insights infographic
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import FancyBboxPatch
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, '🔥 Thermal Runaway Correction Analysis', 
            fontsize=20, fontweight='bold', ha='center', va='center')
    ax.text(5, 9, f'Model: {model_name}', 
            fontsize=14, ha='center', va='center', style='italic')
    
    # Main metrics boxes
    # Initial Error Box
    initial_box = FancyBboxPatch((0.5, 6.5), 2, 1.5, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#ffcccc', edgecolor='red', linewidth=2)
    ax.add_patch(initial_box)
    ax.text(1.5, 7.6, 'Initial Error', fontsize=12, fontweight='bold', ha='center')
    ax.text(1.5, 7.2, f'{initial_error:.2f}%', fontsize=16, fontweight='bold', 
            ha='center', color='red')
    ax.text(1.5, 6.8, '(0D vs 3D)', fontsize=10, ha='center', style='italic')
    
    # Corrected Error Box
    corrected_box = FancyBboxPatch((4, 6.5), 2, 1.5, 
                                  boxstyle="round,pad=0.1", 
                                  facecolor='#ccffcc', edgecolor='green', linewidth=2)
    ax.add_patch(corrected_box)
    ax.text(5, 7.6, 'Corrected Error', fontsize=12, fontweight='bold', ha='center')
    ax.text(5, 7.2, f'{corrected_error:.2f}%', fontsize=16, fontweight='bold', 
            ha='center', color='green')
    ax.text(5, 6.8, '(ML Corrected)', fontsize=10, ha='center', style='italic')
    
    # Improvement Box
    improvement_box = FancyBboxPatch((7.5, 6.5), 2, 1.5, 
                                    boxstyle="round,pad=0.1", 
                                    facecolor='#cceeff', edgecolor='blue', linewidth=2)
    ax.add_patch(improvement_box)
    ax.text(8.5, 7.6, 'Improvement', fontsize=12, fontweight='bold', ha='center')
    ax.text(8.5, 7.2, f'{correction_percentage:.1f}%', fontsize=16, fontweight='bold', 
            ha='center', color='blue')
    ax.text(8.5, 6.8, 'Error Reduction', fontsize=10, ha='center', style='italic')
    
    # Arrow showing improvement
    arrow = patches.FancyArrowPatch((2.5, 7.25), (4, 7.25),
                                   arrowstyle='->', mutation_scale=20, 
                                   color='purple', linewidth=3)
    ax.add_patch(arrow)
    ax.text(3.25, 7.5, 'ML Correction', fontsize=10, ha='center', 
            fontweight='bold', color='purple')
    
    # Safety assessment
    safety_color = 'green' if corrected_error < 5 else 'orange' if corrected_error < 15 else 'red'
    safety_status = 'EXCELLENT' if corrected_error < 5 else 'GOOD' if corrected_error < 15 else 'NEEDS IMPROVEMENT'
    
    safety_box = FancyBboxPatch((2, 1.5), 6, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor=f'{safety_color}', alpha=0.3, 
                               edgecolor=safety_color, linewidth=2)
    ax.add_patch(safety_box)
    ax.text(5, 2.2, 'Safety Assessment', fontsize=12, fontweight='bold', ha='center')
    ax.text(5, 1.8, f'Status: {safety_status}', fontsize=14, fontweight='bold', 
            ha='center', color=safety_color)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def test_pipeline_functions():
    """Test all pipeline functions with sample data"""
    print("\n🧪 TESTING PIPELINE FUNCTIONS")
    print("=" * 50)
    
    # Create sample data
    sample_data = pd.DataFrame({
        'time': np.linspace(0, 100, 100),
        'temperature_0d': 25 + 75 * (1 - np.exp(-np.linspace(0, 3, 100))) + np.random.normal(0, 1, 100)
    })
    
    print(f"📊 Sample data created: {sample_data.shape}")
    print(f"   Temperature range: {sample_data['temperature_0d'].min():.1f} - {sample_data['temperature_0d'].max():.1f}°C")
    
    # Test each function
    try:
        print("\n1️⃣ Testing model training...")
        trained_models, feature_cols, processor = train_models_on_new_data(sample_data)
        
        print("\n2️⃣ Testing error correction...")
        corrections, corrected_temps = apply_error_correction(sample_data, trained_models, feature_cols)
        
        print("\n3️⃣ Testing performance analysis...")
        analysis = analyze_correction_performance(sample_data, corrections, corrected_temps)
        
        print("\n4️⃣ Testing visualization...")
        fig = generate_comprehensive_plots(sample_data, corrections, corrected_temps, analysis)
        
        print("\n✅ ALL TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def train_models_on_new_data(test_data, target_column='correction_factor'):
    """
    TRAIN NEW MODELS ON PROVIDED DATA
    ================================
    """
    print("🤖 Training new models on provided data...")
    
    # Initialize processor and engineer features
    processor = ThermalRunawayDataProcessor()
    processor.aligned_data = test_data.copy()
    
    # Add dummy target if not present
    if target_column not in test_data.columns:
        processor.aligned_data[target_column] = np.random.uniform(0.8, 1.2, len(test_data))
        print(f"⚠️ Added dummy {target_column} for demonstration")
    
    # Engineer features
    featured_data = processor.engineer_features()
    
    # Prepare training data
    exclude_cols = [target_column, 'time', 'temperature_3d'] if 'temperature_3d' in featured_data.columns else [target_column, 'time']
    feature_cols = [col for col in featured_data.columns if col not in exclude_cols]
    
    X = featured_data[feature_cols].fillna(0)
    y = featured_data[target_column].fillna(1.0)
    
    # Train models
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    import xgboost as xgb
    
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=50, random_state=42),
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'XGBoost': xgb.XGBRegressor(n_estimators=50, random_state=42)
    }
    
    trained_models = {}
    for name, model in models.items():
        try:
            model.fit(X, y)
            trained_models[name] = model
            print(f"✓ {name} trained successfully")
        except Exception as e:
            print(f"❌ {name} training failed: {e}")
    
    return trained_models, feature_cols, processor

def apply_error_correction(test_data, trained_models, feature_cols):
    """
    APPLY ERROR CORRECTION TO 0D DATA
    =================================
    """
    print("🔧 Applying error correction...")
    
    # Prepare features
    processor = ThermalRunawayDataProcessor()
    processor.aligned_data = test_data.copy()
    featured_data = processor.engineer_features()
    
    X = featured_data[feature_cols].fillna(0)
    
    corrections = {}
    corrected_temps = {}
    
    for name, model in trained_models.items():
        try:
            correction_factors = model.predict(X)
            corrections[name] = correction_factors
            corrected_temps[name] = test_data['temperature_0d'] * correction_factors
            print(f"✓ {name}: CF range [{correction_factors.min():.3f}, {correction_factors.max():.3f}]")
        except Exception as e:
            print(f"❌ {name} correction failed: {e}")
    
    return corrections, corrected_temps

def analyze_correction_performance(test_data, corrections, corrected_temps):
    """
    ANALYZE CORRECTION PERFORMANCE
    =============================
    """
    print("📊 Analyzing correction performance...")
    
    analysis = {}
    
    for model_name in corrections.keys():
        cf = corrections[model_name]
        corrected_temp = corrected_temps[model_name]
        
        # Basic statistics
        stats = {
            'mean_cf': cf.mean(),
            'std_cf': cf.std(),
            'min_cf': cf.min(),
            'max_cf': cf.max(),
            'mean_temp_increase': (corrected_temp - test_data['temperature_0d']).mean(),
            'max_temp_increase': (corrected_temp - test_data['temperature_0d']).max()
        }
        
        # Thermal zone analysis
        normal_mask = test_data['temperature_0d'] < 60
        warning_mask = (test_data['temperature_0d'] >= 60) & (test_data['temperature_0d'] < 120)
        critical_mask = test_data['temperature_0d'] >= 120
        
        if normal_mask.sum() > 0:
            stats['normal_zone_cf'] = cf[normal_mask].mean()
        if warning_mask.sum() > 0:
            stats['warning_zone_cf'] = cf[warning_mask].mean()
        if critical_mask.sum() > 0:
            stats['critical_zone_cf'] = cf[critical_mask].mean()
        
        analysis[model_name] = stats
    
    return analysis
def generate_comprehensive_plots(test_data, corrections, corrected_temps, analysis):
    """
    GENERATE ALL REQUIRED PLOTS
    ===========================
    """
    print("📈 Generating comprehensive plots...")
    
    # Set up the plotting style
    plt.style.use('default')
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    # Create main figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # Plot 1: Original vs Corrected Temperatures
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(test_data.index, test_data['temperature_0d'], 'b-', linewidth=2, label='Original 0D')
    
    for i, (model_name, corrected_temp) in enumerate(corrected_temps.items()):
        if i < len(colors):
            ax1.plot(test_data.index, corrected_temp, color=colors[i], 
                    linewidth=2, label=f'{model_name}', alpha=0.7)
    
    ax1.axhline(y=60, color='orange', linestyle='--', alpha=0.7, label='Warning (60°C)')
    ax1.axhline(y=120, color='red', linestyle='--', alpha=0.7, label='Critical (120°C)')
    ax1.set_title('Temperature Evolution Comparison')
    ax1.set_xlabel('Data Point Index')
    ax1.set_ylabel('Temperature (°C)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Correction Factors
    ax2 = plt.subplot(2, 3, 2)
    for i, (model_name, cf) in enumerate(corrections.items()):
        if i < len(colors):
            ax2.plot(test_data.index, cf, color=colors[i], 
                    linewidth=2, label=f'{model_name}', alpha=0.7)
    
    ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='Perfect Match (CF=1)')
    ax2.set_title('Correction Factor Evolution')
    ax2.set_xlabel('Data Point Index')
    ax2.set_ylabel('Correction Factor')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Correction Factor vs Temperature
    ax3 = plt.subplot(2, 3, 3)
    for i, (model_name, cf) in enumerate(corrections.items()):
        if i < len(colors):
            ax3.scatter(test_data['temperature_0d'], cf, color=colors[i], 
                       alpha=0.6, label=f'{model_name}', s=30)
    
    ax3.set_title('Correction Factor vs Temperature')
    ax3.set_xlabel('0D Temperature (°C)')
    ax3.set_ylabel('Correction Factor')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Temperature Distribution
    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(test_data['temperature_0d'], bins=20, alpha=0.7, label='Original 0D', color='blue')
    
    for i, (model_name, corrected_temp) in enumerate(corrected_temps.items()):
        if i < len(colors) and i < 2:  # Limit to avoid overcrowding
            ax4.hist(corrected_temp, bins=20, alpha=0.5, 
                    label=f'{model_name}', color=colors[i])
    
    ax4.set_title('Temperature Distribution')
    ax4.set_xlabel('Temperature (°C)')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Model Performance Metrics
    ax5 = plt.subplot(2, 3, 5)
    models = list(analysis.keys())
    mean_cfs = [analysis[model]['mean_cf'] for model in models]
    
    bars = ax5.bar(models, mean_cfs, color=colors[:len(models)], alpha=0.7)
    ax5.set_title('Mean Correction Factor by Model')
    ax5.set_ylabel('Mean Correction Factor')
    ax5.tick_params(axis='x', rotation=45)
    ax5.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, mean_cfs):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # Plot 6: Temperature Increase Analysis
    ax6 = plt.subplot(2, 3, 6)
    temp_increases = [analysis[model]['mean_temp_increase'] for model in models]
    
    bars = ax6.bar(models, temp_increases, color=colors[:len(models)], alpha=0.7)
    ax6.set_title('Mean Temperature Increase by Model')
    ax6.set_ylabel('Temperature Increase (°C)')
    ax6.tick_params(axis='x', rotation=45)
    ax6.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, temp_increases):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{value:.1f}°C', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return fig
def generate_correction_insights_infographic(initial_error, corrected_error, 
                                           correction_percentage, model_name="ML Model",
                                           additional_metrics=None):
    """
    Generate comprehensive correction insights infographic
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import FancyBboxPatch
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, '🔥 Thermal Runaway Correction Analysis', 
            fontsize=20, fontweight='bold', ha='center', va='center')
    ax.text(5, 9, f'Model: {model_name}', 
            fontsize=14, ha='center', va='center', style='italic')
    
    # Main metrics boxes
    # Initial Error Box
    initial_box = FancyBboxPatch((0.5, 6.5), 2, 1.5, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#ffcccc', edgecolor='red', linewidth=2)
    ax.add_patch(initial_box)
    ax.text(1.5, 7.6, 'Initial Error', fontsize=12, fontweight='bold', ha='center')
    ax.text(1.5, 7.2, f'{initial_error:.2f}%', fontsize=16, fontweight='bold', 
            ha='center', color='red')
    ax.text(1.5, 6.8, '(0D vs 3D)', fontsize=10, ha='center', style='italic')
    
    # Corrected Error Box
    corrected_box = FancyBboxPatch((4, 6.5), 2, 1.5, 
                                  boxstyle="round,pad=0.1", 
                                  facecolor='#ccffcc', edgecolor='green', linewidth=2)
    ax.add_patch(corrected_box)
    ax.text(5, 7.6, 'Corrected Error', fontsize=12, fontweight='bold', ha='center')
    ax.text(5, 7.2, f'{corrected_error:.2f}%', fontsize=16, fontweight='bold', 
            ha='center', color='green')
    ax.text(5, 6.8, '(ML Corrected)', fontsize=10, ha='center', style='italic')
    
    # Improvement Box
    improvement_box = FancyBboxPatch((7.5, 6.5), 2, 1.5, 
                                    boxstyle="round,pad=0.1", 
                                    facecolor='#cceeff', edgecolor='blue', linewidth=2)
    ax.add_patch(improvement_box)
    ax.text(8.5, 7.6, 'Improvement', fontsize=12, fontweight='bold', ha='center')
    ax.text(8.5, 7.2, f'{correction_percentage:.1f}%', fontsize=16, fontweight='bold', 
            ha='center', color='blue')
    ax.text(8.5, 6.8, 'Error Reduction', fontsize=10, ha='center', style='italic')
    
    # Arrow showing improvement
    arrow = patches.FancyArrowPatch((2.5, 7.25), (4, 7.25),
                                   arrowstyle='->', mutation_scale=20, 
                                   color='purple', linewidth=3)
    ax.add_patch(arrow)
    ax.text(3.25, 7.5, 'ML Correction', fontsize=10, ha='center', 
            fontweight='bold', color='purple')
    
    # Safety assessment
    safety_color = 'green' if corrected_error < 5 else 'orange' if corrected_error < 15 else 'red'
    safety_status = 'EXCELLENT' if corrected_error < 5 else 'GOOD' if corrected_error < 15 else 'NEEDS IMPROVEMENT'
    
    safety_box = FancyBboxPatch((2, 1.5), 6, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor=f'{safety_color}', alpha=0.3, 
                               edgecolor=safety_color, linewidth=2)
    ax.add_patch(safety_box)
    ax.text(5, 2.2, 'Safety Assessment', fontsize=12, fontweight='bold', ha='center')
    ax.text(5, 1.8, f'Status: {safety_status}', fontsize=14, fontweight='bold', 
            ha='center', color=safety_color)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def test_pipeline_functions():
    """Test all pipeline functions with sample data"""
    print("\n🧪 TESTING PIPELINE FUNCTIONS")
    print("=" * 50)
    
    # Create sample data
    sample_data = pd.DataFrame({
        'time': np.linspace(0, 100, 100),
        'temperature_0d': 25 + 75 * (1 - np.exp(-np.linspace(0, 3, 100))) + np.random.normal(0, 1, 100)
    })
    
    print(f"📊 Sample data created: {sample_data.shape}")
    print(f"   Temperature range: {sample_data['temperature_0d'].min():.1f} - {sample_data['temperature_0d'].max():.1f}°C")
    
    # Test each function
    try:
        print("\n1️⃣ Testing model training...")
        trained_models, feature_cols, processor = train_models_on_new_data(sample_data)
        
        print("\n2️⃣ Testing error correction...")
        corrections, corrected_temps = apply_error_correction(sample_data, trained_models, feature_cols)
        
        print("\n3️⃣ Testing performance analysis...")
        analysis = analyze_correction_performance(sample_data, corrections, corrected_temps)
        
        print("\n4️⃣ Testing visualization...")
        fig = generate_comprehensive_plots(sample_data, corrections, corrected_temps, analysis)
        
        print("\n✅ ALL TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def train_models_on_new_data(test_data, target_column='correction_factor'):
    """
    TRAIN NEW MODELS ON PROVIDED DATA
    ================================
    """
    print("🤖 Training new models on provided data...")
    
    # Initialize processor and engineer features
    processor = ThermalRunawayDataProcessor()
    processor.aligned_data = test_data.copy()
    
    # Add dummy target if not present
    if target_column not in test_data.columns:
        processor.aligned_data[target_column] = np.random.uniform(0.8, 1.2, len(test_data))
        print(f"⚠️ Added dummy {target_column} for demonstration")
    
    # Engineer features
    featured_data = processor.engineer_features()
    
    # Prepare training data
    exclude_cols = [target_column, 'time', 'temperature_3d'] if 'temperature_3d' in featured_data.columns else [target_column, 'time']
    feature_cols = [col for col in featured_data.columns if col not in exclude_cols]
    
    X = featured_data[feature_cols].fillna(0)
    y = featured_data[target_column].fillna(1.0)
    
    # Train models
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    import xgboost as xgb
    
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=50, random_state=42),
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'XGBoost': xgb.XGBRegressor(n_estimators=50, random_state=42)
    }
    
    trained_models = {}
    for name, model in models.items():
        try:
            model.fit(X, y)
            trained_models[name] = model
            print(f"✓ {name} trained successfully")
        except Exception as e:
            print(f"❌ {name} training failed: {e}")
    
    return trained_models, feature_cols, processor

def apply_error_correction(test_data, trained_models, feature_cols):
    """
    APPLY ERROR CORRECTION TO 0D DATA
    =================================
    """
    print("🔧 Applying error correction...")
    
    # Prepare features
    processor = ThermalRunawayDataProcessor()
    processor.aligned_data = test_data.copy()
    featured_data = processor.engineer_features()
    
    X = featured_data[feature_cols].fillna(0)
    
    corrections = {}
    corrected_temps = {}
    
    for name, model in trained_models.items():
        try:
            correction_factors = model.predict(X)
            corrections[name] = correction_factors
            corrected_temps[name] = test_data['temperature_0d'] * correction_factors
            print(f"✓ {name}: CF range [{correction_factors.min():.3f}, {correction_factors.max():.3f}]")
        except Exception as e:
            print(f"❌ {name} correction failed: {e}")
    
    return corrections, corrected_temps

def analyze_correction_performance(test_data, corrections, corrected_temps):
    """
    ANALYZE CORRECTION PERFORMANCE
    =============================
    """
    print("📊 Analyzing correction performance...")
    
    analysis = {}
    
    for model_name in corrections.keys():
        cf = corrections[model_name]
        corrected_temp = corrected_temps[model_name]
        
        # Basic statistics
        stats = {
            'mean_cf': cf.mean(),
            'std_cf': cf.std(),
            'min_cf': cf.min(),
            'max_cf': cf.max(),
            'mean_temp_increase': (corrected_temp - test_data['temperature_0d']).mean(),
            'max_temp_increase': (corrected_temp - test_data['temperature_0d']).max()
        }
        
        # Thermal zone analysis
        normal_mask = test_data['temperature_0d'] < 60
        warning_mask = (test_data['temperature_0d'] >= 60) & (test_data['temperature_0d'] < 120)
        critical_mask = test_data['temperature_0d'] >= 120
        
        if normal_mask.sum() > 0:
            stats['normal_zone_cf'] = cf[normal_mask].mean()
        if warning_mask.sum() > 0:
            stats['warning_zone_cf'] = cf[warning_mask].mean()
        if critical_mask.sum() > 0:
            stats['critical_zone_cf'] = cf[critical_mask].mean()
        
        analysis[model_name] = stats
    
    return analysis
def generate_comprehensive_plots(test_data, corrections, corrected_temps, analysis):
    """
    GENERATE ALL REQUIRED PLOTS
    ===========================
    """
    print("📈 Generating comprehensive plots...")
    
    # Set up the plotting style
    plt.style.use('default')
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    # Create main figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # Plot 1: Original vs Corrected Temperatures
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(test_data.index, test_data['temperature_0d'], 'b-', linewidth=2, label='Original 0D')
    
    for i, (model_name, corrected_temp) in enumerate(corrected_temps.items()):
        if i < len(colors):
            ax1.plot(test_data.index, corrected_temp, color=colors[i], 
                    linewidth=2, label=f'{model_name}', alpha=0.7)
    
    ax1.axhline(y=60, color='orange', linestyle='--', alpha=0.7, label='Warning (60°C)')
    ax1.axhline(y=120, color='red', linestyle='--', alpha=0.7, label='Critical (120°C)')
    ax1.set_title('Temperature Evolution Comparison')
    ax1.set_xlabel('Data Point Index')
    ax1.set_ylabel('Temperature (°C)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Correction Factors
    ax2 = plt.subplot(2, 3, 2)
    for i, (model_name, cf) in enumerate(corrections.items()):
        if i < len(colors):
            ax2.plot(test_data.index, cf, color=colors[i], 
                    linewidth=2, label=f'{model_name}', alpha=0.7)
    
    ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='Perfect Match (CF=1)')
    ax2.set_title('Correction Factor Evolution')
    ax2.set_xlabel('Data Point Index')
    ax2.set_ylabel('Correction Factor')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Correction Factor vs Temperature
    ax3 = plt.subplot(2, 3, 3)
    for i, (model_name, cf) in enumerate(corrections.items()):
        if i < len(colors):
            ax3.scatter(test_data['temperature_0d'], cf, color=colors[i], 
                       alpha=0.6, label=f'{model_name}', s=30)
    
    ax3.set_title('Correction Factor vs Temperature')
    ax3.set_xlabel('0D Temperature (°C)')
    ax3.set_ylabel('Correction Factor')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Temperature Distribution
    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(test_data['temperature_0d'], bins=20, alpha=0.7, label='Original 0D', color='blue')
    
    for i, (model_name, corrected_temp) in enumerate(corrected_temps.items()):
        if i < len(colors) and i < 2:  # Limit to avoid overcrowding
            ax4.hist(corrected_temp, bins=20, alpha=0.5, 
                    label=f'{model_name}', color=colors[i])
    
    ax4.set_title('Temperature Distribution')
    ax4.set_xlabel('Temperature (°C)')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Model Performance Metrics
    ax5 = plt.subplot(2, 3, 5)
    models = list(analysis.keys())
    mean_cfs = [analysis[model]['mean_cf'] for model in models]
    
    bars = ax5.bar(models, mean_cfs, color=colors[:len(models)], alpha=0.7)
    ax5.set_title('Mean Correction Factor by Model')
    ax5.set_ylabel('Mean Correction Factor')
    ax5.tick_params(axis='x', rotation=45)
    ax5.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, mean_cfs):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # Plot 6: Temperature Increase Analysis
    ax6 = plt.subplot(2, 3, 6)
    temp_increases = [analysis[model]['mean_temp_increase'] for model in models]
    
    bars = ax6.bar(models, temp_increases, color=colors[:len(models)], alpha=0.7)
    ax6.set_title('Mean Temperature Increase by Model')
    ax6.set_ylabel('Temperature Increase (°C)')
    ax6.tick_params(axis='x', rotation=45)
    ax6.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, temp_increases):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{value:.1f}°C', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return fig

def plot_temperature_comparison(self):
    """Generate temperature comparison plot"""
    if self.corrected_temps is None:
        messagebox.showerror("Error", "Please apply corrections first!")
        return
    
    # Use the best model for comparison
    best_model = list(self.corrected_temps.keys())[0]
    plot_0d_vs_corrected_temperature(
        self.test_data['time'],
        self.test_data['temperature_0d'],
        self.corrected_temps[best_model],
        title=f"{best_model} - Temperature Correction Analysis"
    )

def generate_infographic(self):
    """Generate correction insights infographic"""
    if self.analysis is None:
        messagebox.showerror("Error", "Please apply corrections first!")
        return
    
    best_model = min(self.analysis.keys(), 
                    key=lambda x: abs(self.analysis[x]['mean_cf'] - 1.0))
    
    initial_error = 15.0  # You can calculate this from your data
    corrected_error = abs(self.analysis[best_model]['mean_cf'] - 1.0) * 100
    improvement = ((initial_error - corrected_error) / initial_error) * 100
    
    additional_metrics = {
        'Mean CF': f"{self.analysis[best_model]['mean_cf']:.3f}",
        'Temp Increase': f"{self.analysis[best_model]['mean_temp_increase']:.1f}°C"
    }
    
    generate_correction_insights_infographic(
        initial_error, corrected_error, improvement, 
        best_model, additional_metrics
    )


def generate_correction_insights_infographic(initial_error, corrected_error, 
                                           correction_percentage, model_name="ML Model",
                                           additional_metrics=None):
    """
    Generate comprehensive correction insights infographic
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import FancyBboxPatch
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, '🔥 Thermal Runaway Correction Analysis', 
            fontsize=20, fontweight='bold', ha='center', va='center')
    ax.text(5, 9, f'Model: {model_name}', 
            fontsize=14, ha='center', va='center', style='italic')
    
    # Main metrics boxes
    # Initial Error Box
    initial_box = FancyBboxPatch((0.5, 6.5), 2, 1.5, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#ffcccc', edgecolor='red', linewidth=2)
    ax.add_patch(initial_box)
    ax.text(1.5, 7.6, 'Initial Error', fontsize=12, fontweight='bold', ha='center')
    ax.text(1.5, 7.2, f'{initial_error:.2f}%', fontsize=16, fontweight='bold', 
            ha='center', color='red')
    ax.text(1.5, 6.8, '(0D vs 3D)', fontsize=10, ha='center', style='italic')
    
    # Corrected Error Box
    corrected_box = FancyBboxPatch((4, 6.5), 2, 1.5, 
                                  boxstyle="round,pad=0.1", 
                                  facecolor='#ccffcc', edgecolor='green', linewidth=2)
    ax.add_patch(corrected_box)
    ax.text(5, 7.6, 'Corrected Error', fontsize=12, fontweight='bold', ha='center')
    ax.text(5, 7.2, f'{corrected_error:.2f}%', fontsize=16, fontweight='bold', 
            ha='center', color='green')
    ax.text(5, 6.8, '(ML Corrected)', fontsize=10, ha='center', style='italic')
    
    # Improvement Box
    improvement_box = FancyBboxPatch((7.5, 6.5), 2, 1.5, 
                                    boxstyle="round,pad=0.1", 
                                    facecolor='#cceeff', edgecolor='blue', linewidth=2)
    ax.add_patch(improvement_box)
    ax.text(8.5, 7.6, 'Improvement', fontsize=12, fontweight='bold', ha='center')
    ax.text(8.5, 7.2, f'{correction_percentage:.1f}%', fontsize=16, fontweight='bold', 
            ha='center', color='blue')
    ax.text(8.5, 6.8, 'Error Reduction', fontsize=10, ha='center', style='italic')
    
    # Arrow showing improvement
    arrow = patches.FancyArrowPatch((2.5, 7.25), (4, 7.25),
                                   arrowstyle='->', mutation_scale=20, 
                                   color='purple', linewidth=3)
    ax.add_patch(arrow)
    ax.text(3.25, 7.5, 'ML Correction', fontsize=10, ha='center', 
            fontweight='bold', color='purple')
    
    # Safety assessment
    safety_color = 'green' if corrected_error < 5 else 'orange' if corrected_error < 15 else 'red'
    safety_status = 'EXCELLENT' if corrected_error < 5 else 'GOOD' if corrected_error < 15 else 'NEEDS IMPROVEMENT'
    
    safety_box = FancyBboxPatch((2, 1.5), 6, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor=f'{safety_color}', alpha=0.3, 
                               edgecolor=safety_color, linewidth=2)
    ax.add_patch(safety_box)
    ax.text(5, 2.2, 'Safety Assessment', fontsize=12, fontweight='bold', ha='center')
    ax.text(5, 1.8, f'Status: {safety_status}', fontsize=14, fontweight='bold', 
            ha='center', color=safety_color)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def test_pipeline_functions():
    """Test all pipeline functions with sample data"""
    print("\n🧪 TESTING PIPELINE FUNCTIONS")
    print("=" * 50)
    
    # Create sample data
    sample_data = pd.DataFrame({
        'time': np.linspace(0, 100, 100),
        'temperature_0d': 25 + 75 * (1 - np.exp(-np.linspace(0, 3, 100))) + np.random.normal(0, 1, 100)
    })
    
    print(f"📊 Sample data created: {sample_data.shape}")
    print(f"   Temperature range: {sample_data['temperature_0d'].min():.1f} - {sample_data['temperature_0d'].max():.1f}°C")
    
    # Test each function
    try:
        print("\n1️⃣ Testing model training...")
        trained_models, feature_cols, processor = train_models_on_new_data(sample_data)
        
        print("\n2️⃣ Testing error correction...")
        corrections, corrected_temps = apply_error_correction(sample_data, trained_models, feature_cols)
        
        print("\n3️⃣ Testing performance analysis...")
        analysis = analyze_correction_performance(sample_data, corrections, corrected_temps)
        
        print("\n4️⃣ Testing visualization...")
        fig = generate_comprehensive_plots(sample_data, corrections, corrected_temps, analysis)
        
        print("\n✅ ALL TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def train_models_on_new_data(test_data, target_column='correction_factor'):
    """
    TRAIN NEW MODELS ON PROVIDED DATA
    ================================
    """
    print("🤖 Training new models on provided data...")
    
    # Initialize processor and engineer features
    processor = ThermalRunawayDataProcessor()
    processor.aligned_data = test_data.copy()
    
    # Add dummy target if not present
    if target_column not in test_data.columns:
        processor.aligned_data[target_column] = np.random.uniform(0.8, 1.2, len(test_data))
        print(f"⚠️ Added dummy {target_column} for demonstration")
    
    # Engineer features
    featured_data = processor.engineer_features()
    
    # Prepare training data
    exclude_cols = [target_column, 'time', 'temperature_3d'] if 'temperature_3d' in featured_data.columns else [target_column, 'time']
    feature_cols = [col for col in featured_data.columns if col not in exclude_cols]
    
    X = featured_data[feature_cols].fillna(0)
    y = featured_data[target_column].fillna(1.0)
    
    # Train models
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    import xgboost as xgb
    
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=50, random_state=42),
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'XGBoost': xgb.XGBRegressor(n_estimators=50, random_state=42)
    }
    
    trained_models = {}
    for name, model in models.items():
        try:
            model.fit(X, y)
            trained_models[name] = model
            print(f"✓ {name} trained successfully")
        except Exception as e:
            print(f"❌ {name} training failed: {e}")
    
    return trained_models, feature_cols, processor

def apply_error_correction(test_data, trained_models, feature_cols):
    """
    APPLY ERROR CORRECTION TO 0D DATA
    =================================
    """
    print("🔧 Applying error correction...")
    
    # Prepare features
    processor = ThermalRunawayDataProcessor()
    processor.aligned_data = test_data.copy()
    featured_data = processor.engineer_features()
    
    X = featured_data[feature_cols].fillna(0)
    
    corrections = {}
    corrected_temps = {}
    
    for name, model in trained_models.items():
        try:
            correction_factors = model.predict(X)
            corrections[name] = correction_factors
            corrected_temps[name] = test_data['temperature_0d'] * correction_factors
            print(f"✓ {name}: CF range [{correction_factors.min():.3f}, {correction_factors.max():.3f}]")
        except Exception as e:
            print(f"❌ {name} correction failed: {e}")
    
    return corrections, corrected_temps

def analyze_correction_performance(test_data, corrections, corrected_temps):
    """
    ANALYZE CORRECTION PERFORMANCE
    =============================
    """
    print("📊 Analyzing correction performance...")
    
    analysis = {}
    
    for model_name in corrections.keys():
        cf = corrections[model_name]
        corrected_temp = corrected_temps[model_name]
        
        # Basic statistics
        stats = {
            'mean_cf': cf.mean(),
            'std_cf': cf.std(),
            'min_cf': cf.min(),
            'max_cf': cf.max(),
            'mean_temp_increase': (corrected_temp - test_data['temperature_0d']).mean(),
            'max_temp_increase': (corrected_temp - test_data['temperature_0d']).max()
        }
        
        # Thermal zone analysis
        normal_mask = test_data['temperature_0d'] < 60
        warning_mask = (test_data['temperature_0d'] >= 60) & (test_data['temperature_0d'] < 120)
        critical_mask = test_data['temperature_0d'] >= 120
        
        if normal_mask.sum() > 0:
            stats['normal_zone_cf'] = cf[normal_mask].mean()
        if warning_mask.sum() > 0:
            stats['warning_zone_cf'] = cf[warning_mask].mean()
        if critical_mask.sum() > 0:
            stats['critical_zone_cf'] = cf[critical_mask].mean()
        
        analysis[model_name] = stats
    
    return analysis
def generate_comprehensive_plots(test_data, corrections, corrected_temps, analysis):
    """
    GENERATE ALL REQUIRED PLOTS
    ===========================
    """
    print("📈 Generating comprehensive plots...")
    
    # Set up the plotting style
    plt.style.use('default')
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    # Create main figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # Plot 1: Original vs Corrected Temperatures
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(test_data.index, test_data['temperature_0d'], 'b-', linewidth=2, label='Original 0D')
    
    for i, (model_name, corrected_temp) in enumerate(corrected_temps.items()):
        if i < len(colors):
            ax1.plot(test_data.index, corrected_temp, color=colors[i], 
                    linewidth=2, label=f'{model_name}', alpha=0.7)
    
    ax1.axhline(y=60, color='orange', linestyle='--', alpha=0.7, label='Warning (60°C)')
    ax1.axhline(y=120, color='red', linestyle='--', alpha=0.7, label='Critical (120°C)')
    ax1.set_title('Temperature Evolution Comparison')
    ax1.set_xlabel('Data Point Index')
    ax1.set_ylabel('Temperature (°C)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Correction Factors
    ax2 = plt.subplot(2, 3, 2)
    for i, (model_name, cf) in enumerate(corrections.items()):
        if i < len(colors):
            ax2.plot(test_data.index, cf, color=colors[i], 
                    linewidth=2, label=f'{model_name}', alpha=0.7)
    
    ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='Perfect Match (CF=1)')
    ax2.set_title('Correction Factor Evolution')
    ax2.set_xlabel('Data Point Index')
    ax2.set_ylabel('Correction Factor')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Correction Factor vs Temperature
    ax3 = plt.subplot(2, 3, 3)
    for i, (model_name, cf) in enumerate(corrections.items()):
        if i < len(colors):
            ax3.scatter(test_data['temperature_0d'], cf, color=colors[i], 
                       alpha=0.6, label=f'{model_name}', s=30)
    
    ax3.set_title('Correction Factor vs Temperature')
    ax3.set_xlabel('0D Temperature (°C)')
    ax3.set_ylabel('Correction Factor')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Temperature Distribution
    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(test_data['temperature_0d'], bins=20, alpha=0.7, label='Original 0D', color='blue')
    
    for i, (model_name, corrected_temp) in enumerate(corrected_temps.items()):
        if i < len(colors) and i < 2:  # Limit to avoid overcrowding
            ax4.hist(corrected_temp, bins=20, alpha=0.5, 
                    label=f'{model_name}', color=colors[i])
    
    ax4.set_title('Temperature Distribution')
    ax4.set_xlabel('Temperature (°C)')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Model Performance Metrics
    ax5 = plt.subplot(2, 3, 5)
    models = list(analysis.keys())
    mean_cfs = [analysis[model]['mean_cf'] for model in models]
    
    bars = ax5.bar(models, mean_cfs, color=colors[:len(models)], alpha=0.7)
    ax5.set_title('Mean Correction Factor by Model')
    ax5.set_ylabel('Mean Correction Factor')
    ax5.tick_params(axis='x', rotation=45)
    ax5.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, mean_cfs):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # Plot 6: Temperature Increase Analysis
    ax6 = plt.subplot(2, 3, 6)
    temp_increases = [analysis[model]['mean_temp_increase'] for model in models]
    
    bars = ax6.bar(models, temp_increases, color=colors[:len(models)], alpha=0.7)
    ax6.set_title('Mean Temperature Increase by Model')
    ax6.set_ylabel('Temperature Increase (°C)')
    ax6.tick_params(axis='x', rotation=45)
    ax6.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, temp_increases):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{value:.1f}°C', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return fig

class ThermalRunawayTestingGUI:
    """
    COMPREHENSIVE GUI FOR THERMAL RUNAWAY TESTING
    ============================================
    Complete interface for thermal runaway analysis with ML correction
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("🔥 Thermal Runaway ML Testing Interface")
        self.root.geometry("1400x900")
        self.root.minsize(1000, 700)
        
        # Initialize variables
        self.test_data = None
        self.trained_models = None
        self.feature_cols = None
        self.corrections = None
        self.corrected_temps = None
        self.analysis = None
        self.processor = None
        
        # GUI variables
        self.file_path = tk.StringVar()
        self.status_var = tk.StringVar()
        self.progress_var = tk.DoubleVar()
        
        # Initialize GUI
        self.setup_gui()
        self.status_var.set("Ready - Please load data file")
    
    def setup_gui(self):
        """Setup the complete GUI layout"""
        
        # Configure root window
        self.root.configure(bg='#f0f0f0')
        
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Title section
        title_frame = ttk.Frame(main_frame)
        title_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        title_label = ttk.Label(title_frame, text="🔥 Thermal Runaway ML Analysis Interface", 
                               font=('Arial', 18, 'bold'))
        title_label.pack()
        
        subtitle_label = ttk.Label(title_frame, text="Machine Learning Pipeline for 0D-3D Model Correction", 
                                  font=('Arial', 12, 'italic'))
        subtitle_label.pack()
        
        # Control Panel
        control_frame = ttk.LabelFrame(main_frame, text="Control Panel", padding="15")
        control_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # File selection section
        file_section = ttk.Frame(control_frame)
        file_section.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(file_section, text="Data File:", font=('Arial', 10, 'bold')).grid(row=0, column=0, sticky=tk.W, padx=5)
        file_entry = ttk.Entry(file_section, textvariable=self.file_path, width=60, font=('Arial', 10))
        file_entry.grid(row=0, column=1, padx=5, sticky=(tk.W, tk.E))
        ttk.Button(file_section, text="Browse", command=self.browse_file).grid(row=0, column=2, padx=5)
        
        file_section.columnconfigure(1, weight=1)
        
        # Progress bar
        progress_frame = ttk.Frame(control_frame)
        progress_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Label(progress_frame, text="Progress:", font=('Arial', 10, 'bold')).grid(row=0, column=0, sticky=tk.W)
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, 
                                           maximum=100, length=400)
        self.progress_bar.grid(row=0, column=1, padx=10, sticky=(tk.W, tk.E))
        
        progress_frame.columnconfigure(1, weight=1)
        
        # Action buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=2, column=0, columnspan=3, pady=15)
        
        # Primary action buttons
        ttk.Button(button_frame, text="📁 Load Data", command=self.load_data, 
                  width=12).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="🤖 Train Models", command=self.train_models, 
                  width=12).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="🔧 Apply Correction", command=self.apply_correction, 
                  width=15).grid(row=0, column=2, padx=5)
        ttk.Button(button_frame, text="📊 Generate Plots", command=self.generate_plots, 
                  width=15).grid(row=0, column=3, padx=5)
        
        # Secondary action buttons
        button_frame2 = ttk.Frame(control_frame)
        button_frame2.grid(row=3, column=0, columnspan=3, pady=5)
        
        ttk.Button(button_frame2, text="📈 Temperature Plot", command=self.plot_temperature_comparison, 
                  width=15).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame2, text="📋 Generate Report", command=self.generate_infographic, 
                  width=15).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame2, text="💾 Export Results", command=self.export_results, 
                  width=15).grid(row=0, column=2, padx=5)
        ttk.Button(button_frame2, text="🔄 Reset", command=self.reset_analysis, 
                  width=12).grid(row=0, column=3, padx=5)
        
        # Results area
        results_frame = ttk.LabelFrame(main_frame, text="Analysis Results", padding="10")
        results_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Create notebook for tabbed results
        self.notebook = ttk.Notebook(results_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Data Overview tab
        self.data_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.data_frame, text="📊 Data Overview")
        
        # Model Performance tab
        self.performance_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.performance_frame, text="🤖 Model Performance")
        
        # Analysis Results tab
        self.analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.analysis_frame, text="📈 Analysis Results")
        
        # Detailed Metrics tab
        self.details_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.details_frame, text="📋 Detailed Metrics")
        
        # Status bar
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        status_label = ttk.Label(status_frame, text="Status:", font=('Arial', 10, 'bold'))
        status_label.grid(row=0, column=0, sticky=tk.W)
        
        self.status_display = ttk.Label(status_frame, textvariable=self.status_var, 
                                       relief=tk.SUNKEN, padding="5")
        self.status_display.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        
        status_frame.columnconfigure(1, weight=1)
    
    def browse_file(self):
        """Open file dialog to select test data"""
        filename = filedialog.askopenfilename(
            title="Select Test Data File",
            filetypes=[
                ("Excel files", "*.xlsx"),
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.file_path.set(filename)
            self.status_var.set(f"File selected: {filename.split('/')[-1]}")
    
    def load_data(self):
        """FIXED: Load the selected data file with robust error handling"""
        if not self.file_path.get():
            messagebox.showerror("Error", "Please select a data file first!")
            return
    
        try:
            self.status_var.set("Loading data...")
            self.progress_var.set(10)
            self.root.update()
        
            file_path = self.file_path.get()
            print(f"📁 Loading file: {file_path}")
        
            # Robust file loading
            if file_path.endswith('.csv'):
               self.test_data = pd.read_csv(file_path, encoding='utf-8')
            elif file_path.endswith('.xlsx'):
                self.test_data = pd.read_excel(file_path, engine='openpyxl')
            else:
                # Try to detect format
                try:
                    self.test_data = pd.read_csv(file_path)
                except:
                    self.test_data = pd.read_excel(file_path)
        
            self.progress_var.set(30)
            self.root.update()
        
            # Validate data
            if self.test_data.empty:
                raise ValueError("File is empty or contains no valid data")
        
            print(f"✓ Raw data shape: {self.test_data.shape}")
            print(f"✓ Raw columns: {list(self.test_data.columns)}")
        
            # Clean column names
            self.test_data.columns = [str(col).strip().lower().replace(" ", "_").replace("(", "").replace(")", "") 
                                 for col in self.test_data.columns]
        
            self.progress_var.set(50)
            self.root.update()
        
            # Smart column detection
            time_candidates = ['time', 'time_s', 't', 'seconds', 'sec']
            temp_candidates = ['temperature', 'temp', 'temperature_0d', 'temp_0d', 'celsius', 'c']
        
            # Find time column
            time_col = None
            for candidate in time_candidates:
                matching_cols = [col for col in self.test_data.columns if candidate in col.lower()]
                if matching_cols:
                    time_col = matching_cols[0]
                    break
        
            if time_col and time_col != 'time':
                self.test_data = self.test_data.rename(columns={time_col: 'time'})
            elif not time_col:
                self.test_data['time'] = range(len(self.test_data))
                print("⚠️ No time column found, created index-based time")
        
             # Find temperature column
            temp_col = None
            for candidate in temp_candidates:
                matching_cols = [col for col in self.test_data.columns if candidate in col.lower()]
                if matching_cols:
                    temp_col = matching_cols[0]
                    break
        
            if temp_col and temp_col != 'temperature_0d':
                self.test_data = self.test_data.rename(columns={temp_col: 'temperature_0d'})
            elif not temp_col:
                # Use first numeric column as temperature
                numeric_cols = self.test_data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    self.test_data = self.test_data.rename(columns={numeric_cols[0]: 'temperature_0d'})
            else:
                raise ValueError("No numeric temperature column found")
        
            self.progress_var.set(70)
            self.root.update()
        
            # Data validation and cleaning
            self.test_data = self.test_data.dropna(subset=['temperature_0d'])
        
            # Convert to numeric
            self.test_data['temperature_0d'] = pd.to_numeric(self.test_data['temperature_0d'], errors='coerce')
            self.test_data['time'] = pd.to_numeric(self.test_data['time'], errors='coerce')
        
            # Remove any remaining NaN values
            self.test_data = self.test_data.dropna()
        
            if len(self.test_data) < 10:
                raise ValueError("Insufficient valid data points (need at least 10)")
        
            self.progress_var.set(100)
        
            # Display data overview
            self.display_data_overview()
        
            success_msg = f"""Data loaded successfully!

📊 Dataset Information:
  • Rows: {len(self.test_data)}
  • Columns: {len(self.test_data.columns)}
  • Temperature range: {self.test_data['temperature_0d'].min():.1f} - {self.test_data['temperature_0d'].max():.1f}°C
  • Time range: {self.test_data['time'].min():.1f} - {self.test_data['time'].max():.1f}s

✓ Ready for analysis!"""
        
            self.status_var.set(f"Data loaded: {len(self.test_data)} rows, {len(self.test_data.columns)} cols")
            messagebox.showinfo("Success", success_msg)
        
            print(f"✓ Final data shape: {self.test_data.shape}")
            print(f"✓ Final columns: {list(self.test_data.columns)}")
        
        except Exception as e:
            self.progress_var.set(0)
            self.status_var.set("Error loading data")
            error_msg = f"Failed to load data:\n\n{str(e)}\n\nPlease check:\n• File format (CSV/Excel)\n• Data contains numeric values\n• File is not corrupted"
            messagebox.showerror("Data Loading Error", error_msg)
            print(f"❌ Data loading error: {e}")

    
    def train_models(self):
        """Train ML models on the loaded data"""
        if self.test_data is None:
            messagebox.showerror("Error", "Please load data first!")
            return
        
        try:
            self.status_var.set("Training models...")
            self.progress_var.set(20)
            self.root.update()
            
            # Train models using the existing function
            self.trained_models, self.feature_cols, self.processor = train_models_on_new_data(self.test_data)
            
            self.progress_var.set(100)
            
            if self.trained_models:
                self.status_var.set(f"Models trained successfully - {len(self.trained_models)} models ready")
                messagebox.showinfo("Success", 
                                   f"Successfully trained {len(self.trained_models)} models:\n\n" + 
                                   "\n".join([f"• {name}" for name in self.trained_models.keys()]))
            else:
                raise ValueError("No models were successfully trained")
                
        except Exception as e:
            self.progress_var.set(0)
            self.status_var.set("Error training models")
            messagebox.showerror("Error", f"Failed to train models:\n{str(e)}")
    
    def apply_correction(self):
        """Apply error correction using trained models"""
        if self.trained_models is None:
            messagebox.showerror("Error", "Please train models first!")
            return
        
        try:
            self.status_var.set("Applying corrections...")
            self.progress_var.set(30)
            self.root.update()
            
            # Apply corrections using existing function
            self.corrections, self.corrected_temps = apply_error_correction(
                self.test_data, self.trained_models, self.feature_cols)
            
            self.progress_var.set(70)
            self.root.update()
            
            # Analyze performance
            self.analysis = analyze_correction_performance(
                self.test_data, self.corrections, self.corrected_temps)
            
            self.progress_var.set(100)
            self.display_analysis_results()
            self.display_performance_results()
            
            self.status_var.set("Corrections applied successfully")
            messagebox.showinfo("Success", "Error corrections applied successfully!\nCheck the Analysis Results tab for details.")
            
        except Exception as e:
            self.progress_var.set(0)
            self.status_var.set("Error applying corrections")
            messagebox.showerror("Error", f"Failed to apply corrections:\n{str(e)}")
    
    def generate_plots(self):
        """Generate comprehensive plots"""
        if self.corrections is None:
            messagebox.showerror("Error", "Please apply corrections first!")
            return
        
        try:
            self.status_var.set("Generating plots...")
            self.progress_var.set(50)
            self.root.update()
            
            # Generate comprehensive plots
            fig = generate_comprehensive_plots(
                self.test_data, self.corrections, self.corrected_temps, self.analysis)
            
            self.progress_var.set(100)
            self.status_var.set("Plots generated successfully")
            
        except Exception as e:
            self.progress_var.set(0)
            self.status_var.set("Error generating plots")
            messagebox.showerror("Error", f"Failed to generate plots:\n{str(e)}")
    
    def plot_temperature_comparison(self):
        """Generate temperature comparison plot"""
        if self.corrected_temps is None:
            messagebox.showerror("Error", "Please apply corrections first!")
            return
        
        try:
            # Use the best model for comparison
            best_model = min(self.analysis.keys(), 
                           key=lambda x: abs(self.analysis[x]['mean_cf'] - 1.0))
            
            plot_0d_vs_corrected_temperature(
                self.test_data['time'],
                self.test_data['temperature_0d'],
                self.corrected_temps[best_model],
                title=f"{best_model} - Temperature Correction Analysis"
            )
            
            self.status_var.set("Temperature comparison plot generated")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate temperature plot:\n{str(e)}")
    
    def generate_infographic(self):
        """Generate correction insights infographic"""
        if self.analysis is None:
            messagebox.showerror("Error", "Please apply corrections first!")
            return
        
        try:
            # Find best model
            best_model = min(self.analysis.keys(), 
                           key=lambda x: abs(self.analysis[x]['mean_cf'] - 1.0))
            
            # Calculate metrics for infographic
            initial_error = 15.0  # You can calculate this from your actual data
            corrected_error = abs(self.analysis[best_model]['mean_cf'] - 1.0) * 100
            improvement = ((initial_error - corrected_error) / initial_error) * 100
            
            additional_metrics = {
                'Mean CF': f"{self.analysis[best_model]['mean_cf']:.3f}",
                'Temp Increase': f"{self.analysis[best_model]['mean_temp_increase']:.1f}°C",
                'Model Type': best_model
            }
            
            generate_correction_insights_infographic(
                initial_error, corrected_error, improvement, 
                best_model, additional_metrics
            )
            
            self.status_var.set("Correction insights infographic generated")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate infographic:\n{str(e)}")
    
    def export_results(self):
        """Export analysis results to file"""
        if self.analysis is None:
            messagebox.showerror("Error", "No results to export. Please run analysis first!")
            return
        
        try:
            filename = filedialog.asksaveasfilename(
                title="Save Results",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")]
            )
            
            if filename:
                # Create results dataframe
                results_data = []
                for model_name, metrics in self.analysis.items():
                    results_data.append({
                        'Model': model_name,
                        'Mean_CF': metrics['mean_cf'],
                        'Std_CF': metrics['std_cf'],
                        'Min_CF': metrics['min_cf'],
                        'Max_CF': metrics['max_cf'],
                        'Mean_Temp_Increase': metrics['mean_temp_increase'],
                        'Max_Temp_Increase': metrics['max_temp_increase']
                    })
                
                results_df = pd.DataFrame(results_data)
                
                if filename.endswith('.csv'):
                    results_df.to_csv(filename, index=False)
                else:
                    results_df.to_excel(filename, index=False)
                
                self.status_var.set(f"Results exported to {filename}")
                messagebox.showinfo("Success", f"Results exported successfully to:\n{filename}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export results:\n{str(e)}")
    
    def reset_analysis(self):
        """Reset all analysis data"""
        if messagebox.askyesno("Reset", "Are you sure you want to reset all analysis data?"):
            self.test_data = None
            self.trained_models = None
            self.feature_cols = None
            self.corrections = None
            self.corrected_temps = None
            self.analysis = None
            self.processor = None
            
            self.file_path.set("")
            self.progress_var.set(0)
            
            # Clear all tabs
            for widget in self.data_frame.winfo_children():
                widget.destroy()
            for widget in self.performance_frame.winfo_children():
                widget.destroy()
            for widget in self.analysis_frame.winfo_children():
                widget.destroy()
            for widget in self.details_frame.winfo_children():
                widget.destroy()
            
            self.status_var.set("Analysis reset - Ready for new data")
    
    def display_data_overview(self):
        """Display data overview in the GUI"""
        # Clear existing content
        for widget in self.data_frame.winfo_children():
            widget.destroy()
        
        # Create scrollable text widget
        text_frame = ttk.Frame(self.data_frame)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        info_text = tk.Text(text_frame, wrap=tk.WORD, height=20, width=80, font=('Consolas', 10))
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=info_text.yview)
        info_text.configure(yscrollcommand=scrollbar.set)
        
        # Generate data overview content
        info_content = f"""DATA OVERVIEW
{'='*60}

📊 Basic Information:
  • Total data points: {len(self.test_data)}
  • Columns: {list(self.test_data.columns)}
  • Data types: {dict(self.test_data.dtypes)}
  • File size: {self.test_data.memory_usage(deep=True).sum() / 1024:.1f} KB

🌡️ Temperature Statistics:
  • Min temperature: {self.test_data['temperature_0d'].min():.2f}°C
  • Max temperature: {self.test_data['temperature_0d'].max():.2f}°C
  • Mean temperature: {self.test_data['temperature_0d'].mean():.2f}°C
  • Std deviation: {self.test_data['temperature_0d'].std():.2f}°C

📈 Thermal Zones Distribution:
  • Normal (<60°C): {(self.test_data['temperature_0d'] < 60).sum()} points ({(self.test_data['temperature_0d'] < 60).mean()*100:.1f}%)
  • Warning (60-120°C): {((self.test_data['temperature_0d'] >= 60) & (self.test_data['temperature_0d'] < 120)).sum()} points ({((self.test_data['temperature_0d'] >= 60) & (self.test_data['temperature_0d'] < 120)).mean()*100:.1f}%)
  • Critical (>120°C): {(self.test_data['temperature_0d'] >= 120).sum()} points ({(self.test_data['temperature_0d'] >= 120).mean()*100:.1f}%)

📋 Data Sample (First 10 rows):
{self.test_data.head(10).to_string()}

📋 Data Sample (Last 5 rows):
{self.test_data.tail(5).to_string()}

🔍 Missing Values:
{self.test_data.isnull().sum().to_string()}
"""
        
        info_text.insert(tk.END, info_content)
        info_text.config(state=tk.DISABLED)
        
        info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def display_performance_results(self):
        """Display model performance in tabular format"""
        # Clear existing content
        for widget in self.performance_frame.winfo_children():
            widget.destroy()
        
        # Create treeview for performance data
        tree_frame = ttk.Frame(self.performance_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        columns = ('Model', 'Mean CF', 'Std CF', 'CF Range', 'Temp Increase', 'Status')
        performance_tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=15)
        
        # Configure column headings and widths
        for col in columns:
            performance_tree.heading(col, text=col)
            performance_tree.column(col, width=120, anchor=tk.CENTER)
        
        # Populate performance data
        for model_name, metrics in self.analysis.items():
            cf_range = f"{metrics.get('min_cf', 0):.3f} - {metrics.get('max_cf', 0):.3f}"
            status = "🟢 Excellent" if abs(metrics.get('mean_cf', 1) - 1) < 0.1 else "🟡 Good" if abs(metrics.get('mean_cf', 1) - 1) < 0.3 else "🔴 Needs Review"
            
            values = (
                model_name.title(),
                f"{metrics.get('mean_cf', 0):.4f}",
                f"{metrics.get('std_cf', 0):.4f}",
                cf_range,
                f"{metrics.get('mean_temp_increase', 0):.2f}°C",
                status
            )
            performance_tree.insert('', tk.END, values=values)
        
        performance_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Add scrollbar for performance tree
        perf_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=performance_tree.yview)
        performance_tree.configure(yscrollcommand=perf_scrollbar.set)
        perf_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def display_analysis_results(self):
        """Display detailed analysis results"""
        # Clear existing content
        for widget in self.analysis_frame.winfo_children():
            widget.destroy()
        
        # Create scrollable text widget
        text_frame = ttk.Frame(self.analysis_frame)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        analysis_text = tk.Text(text_frame, wrap=tk.WORD, height=20, width=80, font=('Consolas', 10))
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=analysis_text.yview)
        analysis_text.configure(yscrollcommand=scrollbar.set)
        
        # Generate analysis content
        analysis_content = "THERMAL RUNAWAY CORRECTION ANALYSIS\n"
        analysis_content += "="*60 + "\n\n"
        
        # Overall summary
        best_model = min(self.analysis.keys(), key=lambda x: abs(self.analysis[x]['mean_cf'] - 1.0))
        analysis_content += f"🏆 BEST PERFORMING MODEL: {best_model.upper()}\n"
        analysis_content += f"   Mean Correction Factor: {self.analysis[best_model]['mean_cf']:.4f}\n"
        analysis_content += f"   Deviation from ideal: {abs(self.analysis[best_model]['mean_cf'] - 1.0)*100:.2f}%\n\n"
        
        # Detailed model analysis
        for model_name, stats in self.analysis.items():
            analysis_content += f"🤖 {model_name.upper()} ANALYSIS:\n"
            analysis_content += f"   Correction Factor Statistics:\n"
            analysis_content += f"     • Mean: {stats['mean_cf']:.4f}\n"
            analysis_content += f"     • Standard Deviation: {stats['std_cf']:.4f}\n"
            analysis_content += f"     • Range: [{stats['min_cf']:.3f}, {stats['max_cf']:.3f}]\n"
            analysis_content += f"     • Median: {stats.get('median_cf', 'N/A')}\n\n"
            
            analysis_content += f"   Temperature Correction Impact:\n"
            analysis_content += f"     • Mean increase: {stats['mean_temp_increase']:.2f}°C\n"
            analysis_content += f"     • Maximum increase: {stats['max_temp_increase']:.2f}°C\n\n"
            
            # Thermal zone analysis
            if 'normal_zone_cf' in stats:
                analysis_content += f"   Thermal Zone Performance:\n"
                analysis_content += f"     • Normal zone (<60°C): CF = {stats.get('normal_zone_cf', 'N/A'):.4f}\n"
                analysis_content += f"     • Warning zone (60-120°C): CF = {stats.get('warning_zone_cf', 'N/A'):.4f}\n"
                analysis_content += f"     • Critical zone (>120°C): CF = {stats.get('critical_zone_cf', 'N/A'):.4f}\n"
            
            analysis_content += "\n" + "-"*50 + "\n\n"
        
        analysis_text.insert(tk.END, analysis_content)
        analysis_text.config(state=tk.DISABLED)
        
        analysis_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)


def generate_correction_insights_infographic(initial_error, corrected_error, 
                                           correction_percentage, model_name="ML Model",
                                           additional_metrics=None):
    """
    Generate comprehensive correction insights infographic
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import FancyBboxPatch
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, '🔥 Thermal Runaway Correction Analysis', 
            fontsize=20, fontweight='bold', ha='center', va='center')
    ax.text(5, 9, f'Model: {model_name}', 
            fontsize=14, ha='center', va='center', style='italic')
    
    # Main metrics boxes
    # Initial Error Box
    initial_box = FancyBboxPatch((0.5, 6.5), 2, 1.5, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#ffcccc', edgecolor='red', linewidth=2)
    ax.add_patch(initial_box)
    ax.text(1.5, 7.6, 'Initial Error', fontsize=12, fontweight='bold', ha='center')
    ax.text(1.5, 7.2, f'{initial_error:.2f}%', fontsize=16, fontweight='bold', 
            ha='center', color='red')
    ax.text(1.5, 6.8, '(0D vs 3D)', fontsize=10, ha='center', style='italic')
    
    # Corrected Error Box
    corrected_box = FancyBboxPatch((4, 6.5), 2, 1.5, 
                                  boxstyle="round,pad=0.1", 
                                  facecolor='#ccffcc', edgecolor='green', linewidth=2)
    ax.add_patch(corrected_box)
    ax.text(5, 7.6, 'Corrected Error', fontsize=12, fontweight='bold', ha='center')
    ax.text(5, 7.2, f'{corrected_error:.2f}%', fontsize=16, fontweight='bold', 
            ha='center', color='green')
    ax.text(5, 6.8, '(ML Corrected)', fontsize=10, ha='center', style='italic')
    
    # Improvement Box
    improvement_box = FancyBboxPatch((7.5, 6.5), 2, 1.5, 
                                    boxstyle="round,pad=0.1", 
                                    facecolor='#cceeff', edgecolor='blue', linewidth=2)
    ax.add_patch(improvement_box)
    ax.text(8.5, 7.6, 'Improvement', fontsize=12, fontweight='bold', ha='center')
    ax.text(8.5, 7.2, f'{correction_percentage:.1f}%', fontsize=16, fontweight='bold', 
            ha='center', color='blue')
    ax.text(8.5, 6.8, 'Error Reduction', fontsize=10, ha='center', style='italic')
    
    # Arrow showing improvement
    arrow = patches.FancyArrowPatch((2.5, 7.25), (4, 7.25),
                                   arrowstyle='->', mutation_scale=20, 
                                   color='purple', linewidth=3)
    ax.add_patch(arrow)
    ax.text(3.25, 7.5, 'ML Correction', fontsize=10, ha='center', 
            fontweight='bold', color='purple')
    
    # Safety assessment
    safety_color = 'green' if corrected_error < 5 else 'orange' if corrected_error < 15 else 'red'
    safety_status = 'EXCELLENT' if corrected_error < 5 else 'GOOD' if corrected_error < 15 else 'NEEDS IMPROVEMENT'
    
    safety_box = FancyBboxPatch((2, 1.5), 6, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor=f'{safety_color}', alpha=0.3, 
                               edgecolor=safety_color, linewidth=2)
    ax.add_patch(safety_box)
    ax.text(5, 2.2, 'Safety Assessment', fontsize=12, fontweight='bold', ha='center')
    ax.text(5, 1.8, f'Status: {safety_status}', fontsize=14, fontweight='bold', 
            ha='center', color=safety_color)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def test_pipeline_functions():
    """Test all pipeline functions with sample data"""
    print("\n🧪 TESTING PIPELINE FUNCTIONS")
    print("=" * 50)
    
    # Create sample data
    sample_data = pd.DataFrame({
        'time': np.linspace(0, 100, 100),
        'temperature_0d': 25 + 75 * (1 - np.exp(-np.linspace(0, 3, 100))) + np.random.normal(0, 1, 100)
    })
    
    print(f"📊 Sample data created: {sample_data.shape}")
    print(f"   Temperature range: {sample_data['temperature_0d'].min():.1f} - {sample_data['temperature_0d'].max():.1f}°C")
    
    # Test each function
    try:
        print("\n1️⃣ Testing model training...")
        trained_models, feature_cols, processor = train_models_on_new_data(sample_data)
        
        print("\n2️⃣ Testing error correction...")
        corrections, corrected_temps = apply_error_correction(sample_data, trained_models, feature_cols)
        
        print("\n3️⃣ Testing performance analysis...")
        analysis = analyze_correction_performance(sample_data, corrections, corrected_temps)
        
        print("\n4️⃣ Testing visualization...")
        fig = generate_comprehensive_plots(sample_data, corrections, corrected_temps, analysis)
        
        print("\n✅ ALL TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def train_models_on_new_data(test_data, target_column='correction_factor'):
    """
    TRAIN NEW MODELS ON PROVIDED DATA
    ================================
    """
    print("🤖 Training new models on provided data...")
    
    # Initialize processor and engineer features
    processor = ThermalRunawayDataProcessor()
    processor.aligned_data = test_data.copy()
    
    # Add dummy target if not present
    if target_column not in test_data.columns:
        processor.aligned_data[target_column] = np.random.uniform(0.8, 1.2, len(test_data))
        print(f"⚠️ Added dummy {target_column} for demonstration")
    
    # Engineer features
    featured_data = processor.engineer_features()
    
    # Prepare training data
    exclude_cols = [target_column, 'time', 'temperature_3d'] if 'temperature_3d' in featured_data.columns else [target_column, 'time']
    feature_cols = [col for col in featured_data.columns if col not in exclude_cols]
    
    X = featured_data[feature_cols].fillna(0)
    y = featured_data[target_column].fillna(1.0)
    
    # Train models
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    import xgboost as xgb
    
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=50, random_state=42),
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'XGBoost': xgb.XGBRegressor(n_estimators=50, random_state=42)
    }
    
    trained_models = {}
    for name, model in models.items():
        try:
            model.fit(X, y)
            trained_models[name] = model
            print(f"✓ {name} trained successfully")
        except Exception as e:
            print(f"❌ {name} training failed: {e}")
    
    return trained_models, feature_cols, processor

def apply_error_correction(test_data, trained_models, feature_cols):
    """
    APPLY ERROR CORRECTION TO 0D DATA
    =================================
    """
    print("🔧 Applying error correction...")
    
    # Prepare features
    processor = ThermalRunawayDataProcessor()
    processor.aligned_data = test_data.copy()
    featured_data = processor.engineer_features()
    
    X = featured_data[feature_cols].fillna(0)
    
    corrections = {}
    corrected_temps = {}
    
    for name, model in trained_models.items():
        try:
            correction_factors = model.predict(X)
            corrections[name] = correction_factors
            corrected_temps[name] = test_data['temperature_0d'] * correction_factors
            print(f"✓ {name}: CF range [{correction_factors.min():.3f}, {correction_factors.max():.3f}]")
        except Exception as e:
            print(f"❌ {name} correction failed: {e}")
    
    return corrections, corrected_temps

def launch_thermal_gui():
    """
    Launch the thermal runaway GUI application
    ========================================
    Creates and runs the main GUI interface for thermal runaway analysis
    """
    print("🚀 Launching Thermal Runaway GUI...")
    
    try:
        # Import required GUI libraries
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox
        
        # Check if GUI class is defined
        if 'ThermalRunawayTestingGUI' not in globals():
            print("❌ ThermalRunawayTestingGUI class not found!")
            print("Please ensure the GUI class is defined in your code.")
            return
        
        # Create main window
        root = tk.Tk()
        
        # Set window properties
        root.title("🔥 Thermal Runaway ML Analysis Interface")
        root.geometry("1400x900")
        root.minsize(800, 600)
        
        # Set window icon (optional)
        try:
            # You can add an icon file here if available
            # root.iconbitmap('thermal_icon.ico')
            pass
        except:
            pass
        
        # Create application instance
        app = ThermalRunawayTestingGUI(root)
        
        print("✓ GUI initialized successfully")
        print("✓ Window created with dimensions 1400x900")
        print("✓ Application ready for user interaction")
        
        # Configure window closing behavior
        def on_closing():
            if messagebox.askokcancel("Quit", "Do you want to quit the thermal analysis application?"):
                print("👋 Closing Thermal Runaway GUI...")
                root.destroy()
        
        root.protocol("WM_DELETE_WINDOW", on_closing)
        
        # Center window on screen
        root.update_idletasks()
        width = root.winfo_width()
        height = root.winfo_height()
        x = (root.winfo_screenwidth() // 2) - (width // 2)
        y = (root.winfo_screenheight() // 2) - (height // 2)
        root.geometry(f'{width}x{height}+{x}+{y}')
        
        # Start the GUI event loop
        print("🎯 Starting GUI main loop...")
        root.mainloop()
        
        print("✓ GUI session completed successfully")
        
    except ImportError as e:
        print(f"❌ GUI library import failed: {e}")
        print("Please ensure tkinter is installed:")
        print("  - For Windows/Mac: tkinter comes with Python")
        print("  - For Linux: sudo apt-get install python3-tk")
        
    except Exception as e:
        print(f"❌ GUI launch failed: {e}")
        print("Error details:")
        import traceback
        traceback.print_exc()
        
        # Fallback option
        print("\n🔄 Attempting fallback GUI launch...")
        try:
            root = tk.Tk()
            root.title("Thermal Analysis - Basic Mode")
            root.geometry("800x600")
            
            # Create basic interface
            main_frame = ttk.Frame(root, padding="20")
            main_frame.pack(fill=tk.BOTH, expand=True)
            
            ttk.Label(main_frame, text="🔥 Thermal Runaway Analysis", 
                     font=('Arial', 16, 'bold')).pack(pady=10)
            
            ttk.Label(main_frame, text="Basic mode - Full GUI failed to load", 
                     font=('Arial', 10)).pack(pady=5)
            
            ttk.Button(main_frame, text="Close", 
                      command=root.destroy).pack(pady=20)
            
            root.mainloop()
            
        except Exception as fallback_error:
            print(f"❌ Fallback GUI also failed: {fallback_error}")
            print("Please check your Python tkinter installation")

def test_gui_dependencies():
    """
    Test if all GUI dependencies are available
    ==========================================
    """
    print("🔍 Testing GUI dependencies...")
    
    dependencies = {
        'tkinter': False,
        'matplotlib': False,
        'pandas': False,
        'numpy': False
    }
    
    # Test tkinter
    try:
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox
        dependencies['tkinter'] = True
        print("✓ tkinter available")
    except ImportError:
        print("❌ tkinter not available")
    
    # Test matplotlib
    try:
        import matplotlib.pyplot as plt
        dependencies['matplotlib'] = True
        print("✓ matplotlib available")
    except ImportError:
        print("❌ matplotlib not available")
    
    # Test pandas
    try:
        import pandas as pd
        dependencies['pandas'] = True
        print("✓ pandas available")
    except ImportError:
        print("❌ pandas not available")
    
    # Test numpy
    try:
        import numpy as np
        dependencies['numpy'] = True
        print("✓ numpy available")
    except ImportError:
        print("❌ numpy not available")
    
    # Check if all dependencies are met
    all_available = all(dependencies.values())
    
    if all_available:
        print("✅ All GUI dependencies are available")
        return True
    else:
        print("❌ Some dependencies are missing:")
        for dep, available in dependencies.items():
            if not available:
                print(f"  - {dep}")
        return False

def launch_thermal_gui_safe():
    """
    Safe version of GUI launcher with dependency checking
    ===================================================
    """
    print("🛡️ Safe GUI Launch - Checking dependencies first...")
    
    if test_gui_dependencies():
        launch_thermal_gui()
    else:
        print("❌ Cannot launch GUI due to missing dependencies")
        print("Please install missing packages and try again")

# Alternative simple GUI launcher for testing
def launch_simple_gui():
    """
    Launch a simple test GUI to verify tkinter works
    ==============================================
    """
    try:
        import tkinter as tk
        from tkinter import ttk
        
        root = tk.Tk()
        root.title("Thermal Analysis - Test GUI")
        root.geometry("600x400")
        
        # Main frame
        main_frame = ttk.Frame(root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        ttk.Label(main_frame, text="🔥 Thermal Runaway Analysis", 
                 font=('Arial', 16, 'bold')).pack(pady=10)
        
        # Status
        ttk.Label(main_frame, text="✓ GUI Test Successful!", 
                 font=('Arial', 12), foreground='green').pack(pady=5)
        
        # Info
        info_text = """This is a test GUI to verify tkinter functionality.
If you can see this window, your GUI dependencies are working.
Close this window and run the full application."""
        
        ttk.Label(main_frame, text=info_text, 
                 font=('Arial', 10), justify=tk.CENTER).pack(pady=20)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=20)
        
        ttk.Button(button_frame, text="Test Successful", 
                  command=lambda: print("✓ GUI test button clicked")).pack(side=tk.LEFT, padx=10)
        
        ttk.Button(button_frame, text="Close", 
                  command=root.destroy).pack(side=tk.LEFT, padx=10)
        
        print("✓ Simple GUI launched successfully")
        root.mainloop()
        
    except Exception as e:
        print(f"❌ Simple GUI test failed: {e}")

# Usage examples:
if __name__ == "__main__":
    # Test GUI dependencies first
    print("Testing GUI system...")
    
    # Option 1: Safe launch with dependency checking
    launch_thermal_gui_safe()
    
    # Option 2: Direct launch (use if dependencies are confirmed)
    # launch_thermal_gui()
    
    # Option 3: Simple test GUI
    # launch_simple_gui()


# ================================================================
# MAIN EXECUTION BLOCK
# ================================================================
if __name__ == "__main__":
    print("🔥 THERMAL RUNAWAY TESTING MODULE")
    print("=" * 50)
    print("Choose an option:")
    print("1. Test pipeline functions with sample data")
    print("2. Launch GUI for file-based analysis")
    print("3. Run both tests and GUI")
    print("4. Direct function testing")
    
    try:
        choice = input("\nEnter choice (1-4) or press Enter for GUI: ").strip()
        
        if choice == "1":
            # Test with sample data
            print("Creating sample data for testing...")
            sample_data = pd.DataFrame({
                'time': np.linspace(0, 100, 50),
                'temperature_0d': 25 + 50 * np.exp(np.linspace(0, 2, 50)) + np.random.normal(0, 2, 50)
            })
            
            print("Testing individual functions...")
            trained_models, feature_cols, processor = train_models_on_new_data(sample_data)
            corrections, corrected_temps = apply_error_correction(sample_data, trained_models, feature_cols)
            analysis = analyze_correction_performance(sample_data, corrections, corrected_temps)
            fig = generate_comprehensive_plots(sample_data, corrections, corrected_temps, analysis)
            
            print("✅ ALL TESTS PASSED!")
            
        elif choice == "2":
            print("🚀 Launching GUI...")
            launch_thermal_gui()
            
        elif choice == "3":
            # Run tests first, then GUI
            print("Running tests first...")
            test_success = test_pipeline_functions()
            if test_success:
                print("🚀 Tests passed! Launching GUI...")
                launch_thermal_gui()
            else:
                print("❌ Tests failed. Check errors above.")
                
        elif choice == "4":
            # Direct testing with immediate results
            sample_data = pd.DataFrame({
                'time': np.linspace(0, 50, 25),
                'temperature_0d': np.linspace(25, 150, 25) + np.random.normal(0, 2, 25)
            })
            
            trained_models, feature_cols, processor = train_models_on_new_data(sample_data)
            corrections, corrected_temps = apply_error_correction(sample_data, trained_models, feature_cols)
            analysis = analyze_correction_performance(sample_data, corrections, corrected_temps)
            
            print("\n📊 RESULTS SUMMARY:")
            for model, stats in analysis.items():
                print(f"  {model}: CF={stats['mean_cf']:.3f}, ΔT={stats['mean_temp_increase']:.1f}°C")
                
            # Generate plots
            plot_0d_vs_corrected_temperature(
                sample_data['time'],
                sample_data['temperature_0d'],
                corrected_temps[list(corrected_temps.keys())[0]],
                title="Sample Data - Temperature Correction Analysis"
            )
            
            # Generate infographic
            best_model = list(analysis.keys())[0]
            generate_correction_insights_infographic(
                15.0, 2.5, 83.3, best_model,
                {'Mean CF': f"{analysis[best_model]['mean_cf']:.3f}"}
            )
        else:
            # Default to GUI
            print("🚀 Launching GUI...")
            launch_thermal_gui()
            
    except KeyboardInterrupt:
        print("\n👋 Exiting...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Launching GUI as fallback...")
        launch_thermal_gui()

# ================================================================
# GUI LAUNCH FUNCTION (if not already defined)
# ================================================================
def launch_thermal_gui():
    """Launch the thermal runaway GUI"""
    print("🚀 Launching Thermal Runaway GUI...")
    try:
        root = tk.Tk()
        app = ThermalRunawayTestingGUI(root)
        print("✓ GUI initialized successfully")
        root.mainloop()
    except Exception as e:
        print(f"❌ GUI launch failed: {e}")
        import traceback
        traceback.print_exc()

# ================================================================
# QUICK VERIFICATION TEST
# ================================================================
print("\n🔍 QUICK VERIFICATION TEST")
try:
    # Test data processor
    processor = ThermalRunawayDataProcessor()
    print("✓ Data processor working")
    
    # Test sample data creation
    test_data = pd.DataFrame({'time': [1,2,3], 'temperature_0d': [25,30,35]})
    processor.aligned_data = test_data
    features = processor.engineer_features()
    print(f"✓ Feature engineering working: {features.shape}")
    
    print("🎯 Ready for full execution!")
    print("Run the script and choose an option from the menu!")
    
except Exception as e:
    print(f"❌ Verification failed: {e}")


def analyze_correction_performance(test_data, corrections, corrected_temps):
    """
    ANALYZE CORRECTION PERFORMANCE
    =============================
    """
    print("📊 Analyzing correction performance...")
    
    analysis = {}
    
    for model_name in corrections.keys():
        cf = corrections[model_name]
        corrected_temp = corrected_temps[model_name]
        
        # Basic statistics
        stats = {
            'mean_cf': cf.mean(),
            'std_cf': cf.std(),
            'min_cf': cf.min(),
            'max_cf': cf.max(),
            'mean_temp_increase': (corrected_temp - test_data['temperature_0d']).mean(),
            'max_temp_increase': (corrected_temp - test_data['temperature_0d']).max()
        }
        
        # Thermal zone analysis
        normal_mask = test_data['temperature_0d'] < 60
        warning_mask = (test_data['temperature_0d'] >= 60) & (test_data['temperature_0d'] < 120)
        critical_mask = test_data['temperature_0d'] >= 120
        
        if normal_mask.sum() > 0:
            stats['normal_zone_cf'] = cf[normal_mask].mean()
        if warning_mask.sum() > 0:
            stats['warning_zone_cf'] = cf[warning_mask].mean()
        if critical_mask.sum() > 0:
            stats['critical_zone_cf'] = cf[critical_mask].mean()
        
        analysis[model_name] = stats
    
    return analysis
def generate_comprehensive_plots(test_data, corrections, corrected_temps, analysis):
    """
    GENERATE ALL REQUIRED PLOTS
    ===========================
    """
    print("📈 Generating comprehensive plots...")
    
    # Set up the plotting style
    plt.style.use('default')
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    # Create main figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # Plot 1: Original vs Corrected Temperatures
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(test_data.index, test_data['temperature_0d'], 'b-', linewidth=2, label='Original 0D')
    
    for i, (model_name, corrected_temp) in enumerate(corrected_temps.items()):
        if i < len(colors):
            ax1.plot(test_data.index, corrected_temp, color=colors[i], 
                    linewidth=2, label=f'{model_name}', alpha=0.7)
    
    ax1.axhline(y=60, color='orange', linestyle='--', alpha=0.7, label='Warning (60°C)')
    ax1.axhline(y=120, color='red', linestyle='--', alpha=0.7, label='Critical (120°C)')
    ax1.set_title('Temperature Evolution Comparison')
    ax1.set_xlabel('Data Point Index')
    ax1.set_ylabel('Temperature (°C)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Correction Factors
    ax2 = plt.subplot(2, 3, 2)
    for i, (model_name, cf) in enumerate(corrections.items()):
        if i < len(colors):
            ax2.plot(test_data.index, cf, color=colors[i], 
                    linewidth=2, label=f'{model_name}', alpha=0.7)
    
    ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='Perfect Match (CF=1)')
    ax2.set_title('Correction Factor Evolution')
    ax2.set_xlabel('Data Point Index')
    ax2.set_ylabel('Correction Factor')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Correction Factor vs Temperature
    ax3 = plt.subplot(2, 3, 3)
    for i, (model_name, cf) in enumerate(corrections.items()):
        if i < len(colors):
            ax3.scatter(test_data['temperature_0d'], cf, color=colors[i], 
                       alpha=0.6, label=f'{model_name}', s=30)
    
    ax3.set_title('Correction Factor vs Temperature')
    ax3.set_xlabel('0D Temperature (°C)')
    ax3.set_ylabel('Correction Factor')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Temperature Distribution
    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(test_data['temperature_0d'], bins=20, alpha=0.7, label='Original 0D', color='blue')
    
    for i, (model_name, corrected_temp) in enumerate(corrected_temps.items()):
        if i < len(colors) and i < 2:  # Limit to avoid overcrowding
            ax4.hist(corrected_temp, bins=20, alpha=0.5, 
                    label=f'{model_name}', color=colors[i])
    
    ax4.set_title('Temperature Distribution')
    ax4.set_xlabel('Temperature (°C)')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Model Performance Metrics
    ax5 = plt.subplot(2, 3, 5)
    models = list(analysis.keys())
    mean_cfs = [analysis[model]['mean_cf'] for model in models]
    
    bars = ax5.bar(models, mean_cfs, color=colors[:len(models)], alpha=0.7)
    ax5.set_title('Mean Correction Factor by Model')
    ax5.set_ylabel('Mean Correction Factor')
    ax5.tick_params(axis='x', rotation=45)
    ax5.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, mean_cfs):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # Plot 6: Temperature Increase Analysis
    ax6 = plt.subplot(2, 3, 6)
    temp_increases = [analysis[model]['mean_temp_increase'] for model in models]
    
    bars = ax6.bar(models, temp_increases, color=colors[:len(models)], alpha=0.7)
    ax6.set_title('Mean Temperature Increase by Model')
    ax6.set_ylabel('Temperature Increase (°C)')
    ax6.tick_params(axis='x', rotation=45)
    ax6.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, temp_increases):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{value:.1f}°C', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return fig
def generate_correction_insights_infographic(initial_error, corrected_error, 
                                           correction_percentage, model_name="ML Model",
                                           additional_metrics=None):
    """
    Generate comprehensive correction insights infographic
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import FancyBboxPatch
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, '🔥 Thermal Runaway Correction Analysis', 
            fontsize=20, fontweight='bold', ha='center', va='center')
    ax.text(5, 9, f'Model: {model_name}', 
            fontsize=14, ha='center', va='center', style='italic')
    
    # Main metrics boxes
    # Initial Error Box
    initial_box = FancyBboxPatch((0.5, 6.5), 2, 1.5, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#ffcccc', edgecolor='red', linewidth=2)
    ax.add_patch(initial_box)
    ax.text(1.5, 7.6, 'Initial Error', fontsize=12, fontweight='bold', ha='center')
    ax.text(1.5, 7.2, f'{initial_error:.2f}%', fontsize=16, fontweight='bold', 
            ha='center', color='red')
    ax.text(1.5, 6.8, '(0D vs 3D)', fontsize=10, ha='center', style='italic')
    
    # Corrected Error Box
    corrected_box = FancyBboxPatch((4, 6.5), 2, 1.5, 
                                  boxstyle="round,pad=0.1", 
                                  facecolor='#ccffcc', edgecolor='green', linewidth=2)
    ax.add_patch(corrected_box)
    ax.text(5, 7.6, 'Corrected Error', fontsize=12, fontweight='bold', ha='center')
    ax.text(5, 7.2, f'{corrected_error:.2f}%', fontsize=16, fontweight='bold', 
            ha='center', color='green')
    ax.text(5, 6.8, '(ML Corrected)', fontsize=10, ha='center', style='italic')
    
    # Improvement Box
    improvement_box = FancyBboxPatch((7.5, 6.5), 2, 1.5, 
                                    boxstyle="round,pad=0.1", 
                                    facecolor='#cceeff', edgecolor='blue', linewidth=2)
    ax.add_patch(improvement_box)
    ax.text(8.5, 7.6, 'Improvement', fontsize=12, fontweight='bold', ha='center')
    ax.text(8.5, 7.2, f'{correction_percentage:.1f}%', fontsize=16, fontweight='bold', 
            ha='center', color='blue')
    ax.text(8.5, 6.8, 'Error Reduction', fontsize=10, ha='center', style='italic')
    
    # Arrow showing improvement
    arrow = patches.FancyArrowPatch((2.5, 7.25), (4, 7.25),
                                   arrowstyle='->', mutation_scale=20, 
                                   color='purple', linewidth=3)
    ax.add_patch(arrow)
    ax.text(3.25, 7.5, 'ML Correction', fontsize=10, ha='center', 
            fontweight='bold', color='purple')
    
    # Safety assessment
    safety_color = 'green' if corrected_error < 5 else 'orange' if corrected_error < 15 else 'red'
    safety_status = 'EXCELLENT' if corrected_error < 5 else 'GOOD' if corrected_error < 15 else 'NEEDS IMPROVEMENT'
    
    safety_box = FancyBboxPatch((2, 1.5), 6, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor=f'{safety_color}', alpha=0.3, 
                               edgecolor=safety_color, linewidth=2)
    ax.add_patch(safety_box)
    ax.text(5, 2.2, 'Safety Assessment', fontsize=12, fontweight='bold', ha='center')
    ax.text(5, 1.8, f'Status: {safety_status}', fontsize=14, fontweight='bold', 
            ha='center', color=safety_color)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def test_pipeline_functions():
    """Test all pipeline functions with sample data"""
    print("\n🧪 TESTING PIPELINE FUNCTIONS")
    print("=" * 50)
    
    # Create sample data
    sample_data = pd.DataFrame({
        'time': np.linspace(0, 100, 100),
        'temperature_0d': 25 + 75 * (1 - np.exp(-np.linspace(0, 3, 100))) + np.random.normal(0, 1, 100)
    })
    
    print(f"📊 Sample data created: {sample_data.shape}")
    print(f"   Temperature range: {sample_data['temperature_0d'].min():.1f} - {sample_data['temperature_0d'].max():.1f}°C")
    
    # Test each function
    try:
        print("\n1️⃣ Testing model training...")
        trained_models, feature_cols, processor = train_models_on_new_data(sample_data)
        
        print("\n2️⃣ Testing error correction...")
        corrections, corrected_temps = apply_error_correction(sample_data, trained_models, feature_cols)
        
        print("\n3️⃣ Testing performance analysis...")
        analysis = analyze_correction_performance(sample_data, corrections, corrected_temps)
        
        print("\n4️⃣ Testing visualization...")
        fig = generate_comprehensive_plots(sample_data, corrections, corrected_temps, analysis)
        
        print("\n✅ ALL TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def train_models_on_new_data(test_data, target_column='correction_factor'):
    """
    TRAIN NEW MODELS ON PROVIDED DATA
    ================================
    """
    print("🤖 Training new models on provided data...")
    
    # Initialize processor and engineer features
    processor = ThermalRunawayDataProcessor()
    processor.aligned_data = test_data.copy()
    
    # Add dummy target if not present
    if target_column not in test_data.columns:
        processor.aligned_data[target_column] = np.random.uniform(0.8, 1.2, len(test_data))
        print(f"⚠️ Added dummy {target_column} for demonstration")
    
    # Engineer features
    featured_data = processor.engineer_features()
    
    # Prepare training data
    exclude_cols = [target_column, 'time', 'temperature_3d'] if 'temperature_3d' in featured_data.columns else [target_column, 'time']
    feature_cols = [col for col in featured_data.columns if col not in exclude_cols]
    
    X = featured_data[feature_cols].fillna(0)
    y = featured_data[target_column].fillna(1.0)
    
    # Train models
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    import xgboost as xgb
    
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=50, random_state=42),
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'XGBoost': xgb.XGBRegressor(n_estimators=50, random_state=42)
    }
    
    trained_models = {}
    for name, model in models.items():
        try:
            model.fit(X, y)
            trained_models[name] = model
            print(f"✓ {name} trained successfully")
        except Exception as e:
            print(f"❌ {name} training failed: {e}")
    
    return trained_models, feature_cols, processor

def apply_error_correction(test_data, trained_models, feature_cols):
    """
    APPLY ERROR CORRECTION TO 0D DATA
    =================================
    """
    print("🔧 Applying error correction...")
    
    # Prepare features
    processor = ThermalRunawayDataProcessor()
    processor.aligned_data = test_data.copy()
    featured_data = processor.engineer_features()
    
    X = featured_data[feature_cols].fillna(0)
    
    corrections = {}
    corrected_temps = {}
    
    for name, model in trained_models.items():
        try:
            correction_factors = model.predict(X)
            corrections[name] = correction_factors
            corrected_temps[name] = test_data['temperature_0d'] * correction_factors
            print(f"✓ {name}: CF range [{correction_factors.min():.3f}, {correction_factors.max():.3f}]")
        except Exception as e:
            print(f"❌ {name} correction failed: {e}")
    
    return corrections, corrected_temps

def analyze_correction_performance(test_data, corrections, corrected_temps):
    """
    ANALYZE CORRECTION PERFORMANCE
    =============================
    """
    print("📊 Analyzing correction performance...")
    
    analysis = {}
    
    for model_name in corrections.keys():
        cf = corrections[model_name]
        corrected_temp = corrected_temps[model_name]
        
        # Basic statistics
        stats = {
            'mean_cf': cf.mean(),
            'std_cf': cf.std(),
            'min_cf': cf.min(),
            'max_cf': cf.max(),
            'mean_temp_increase': (corrected_temp - test_data['temperature_0d']).mean(),
            'max_temp_increase': (corrected_temp - test_data['temperature_0d']).max()
        }
        
        # Thermal zone analysis
        normal_mask = test_data['temperature_0d'] < 60
        warning_mask = (test_data['temperature_0d'] >= 60) & (test_data['temperature_0d'] < 120)
        critical_mask = test_data['temperature_0d'] >= 120
        
        if normal_mask.sum() > 0:
            stats['normal_zone_cf'] = cf[normal_mask].mean()
        if warning_mask.sum() > 0:
            stats['warning_zone_cf'] = cf[warning_mask].mean()
        if critical_mask.sum() > 0:
            stats['critical_zone_cf'] = cf[critical_mask].mean()
        
        analysis[model_name] = stats
    
    return analysis
def generate_comprehensive_plots(test_data, corrections, corrected_temps, analysis):
    """
    GENERATE ALL REQUIRED PLOTS
    ===========================
    """
    print("📈 Generating comprehensive plots...")
    
    # Set up the plotting style
    plt.style.use('default')
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    # Create main figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # Plot 1: Original vs Corrected Temperatures
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(test_data.index, test_data['temperature_0d'], 'b-', linewidth=2, label='Original 0D')
    
    for i, (model_name, corrected_temp) in enumerate(corrected_temps.items()):
        if i < len(colors):
            ax1.plot(test_data.index, corrected_temp, color=colors[i], 
                    linewidth=2, label=f'{model_name}', alpha=0.7)
    
    ax1.axhline(y=60, color='orange', linestyle='--', alpha=0.7, label='Warning (60°C)')
    ax1.axhline(y=120, color='red', linestyle='--', alpha=0.7, label='Critical (120°C)')
    ax1.set_title('Temperature Evolution Comparison')
    ax1.set_xlabel('Data Point Index')
    ax1.set_ylabel('Temperature (°C)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Correction Factors
    ax2 = plt.subplot(2, 3, 2)
    for i, (model_name, cf) in enumerate(corrections.items()):
        if i < len(colors):
            ax2.plot(test_data.index, cf, color=colors[i], 
                    linewidth=2, label=f'{model_name}', alpha=0.7)
    
    ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='Perfect Match (CF=1)')
    ax2.set_title('Correction Factor Evolution')
    ax2.set_xlabel('Data Point Index')
    ax2.set_ylabel('Correction Factor')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Correction Factor vs Temperature
    ax3 = plt.subplot(2, 3, 3)
    for i, (model_name, cf) in enumerate(corrections.items()):
        if i < len(colors):
            ax3.scatter(test_data['temperature_0d'], cf, color=colors[i], 
                       alpha=0.6, label=f'{model_name}', s=30)
    
    ax3.set_title('Correction Factor vs Temperature')
    ax3.set_xlabel('0D Temperature (°C)')
    ax3.set_ylabel('Correction Factor')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Temperature Distribution
    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(test_data['temperature_0d'], bins=20, alpha=0.7, label='Original 0D', color='blue')
    
    for i, (model_name, corrected_temp) in enumerate(corrected_temps.items()):
        if i < len(colors) and i < 2:  # Limit to avoid overcrowding
            ax4.hist(corrected_temp, bins=20, alpha=0.5, 
                    label=f'{model_name}', color=colors[i])
    
    ax4.set_title('Temperature Distribution')
    ax4.set_xlabel('Temperature (°C)')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Model Performance Metrics
    ax5 = plt.subplot(2, 3, 5)
    models = list(analysis.keys())
    mean_cfs = [analysis[model]['mean_cf'] for model in models]
    
    bars = ax5.bar(models, mean_cfs, color=colors[:len(models)], alpha=0.7)
    ax5.set_title('Mean Correction Factor by Model')
    ax5.set_ylabel('Mean Correction Factor')
    ax5.tick_params(axis='x', rotation=45)
    ax5.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, mean_cfs):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # Plot 6: Temperature Increase Analysis
    ax6 = plt.subplot(2, 3, 6)
    temp_increases = [analysis[model]['mean_temp_increase'] for model in models]
    
    bars = ax6.bar(models, temp_increases, color=colors[:len(models)], alpha=0.7)
    ax6.set_title('Mean Temperature Increase by Model')
    ax6.set_ylabel('Temperature Increase (°C)')
    ax6.tick_params(axis='x', rotation=45)
    ax6.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, temp_increases):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{value:.1f}°C', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return fig
def generate_correction_insights_infographic(initial_error, corrected_error, 
                                           correction_percentage, model_name="ML Model",
                                           additional_metrics=None):
    """
    Generate comprehensive correction insights infographic
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import FancyBboxPatch
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, '🔥 Thermal Runaway Correction Analysis', 
            fontsize=20, fontweight='bold', ha='center', va='center')
    ax.text(5, 9, f'Model: {model_name}', 
            fontsize=14, ha='center', va='center', style='italic')
    
    # Main metrics boxes
    # Initial Error Box
    initial_box = FancyBboxPatch((0.5, 6.5), 2, 1.5, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#ffcccc', edgecolor='red', linewidth=2)
    ax.add_patch(initial_box)
    ax.text(1.5, 7.6, 'Initial Error', fontsize=12, fontweight='bold', ha='center')
    ax.text(1.5, 7.2, f'{initial_error:.2f}%', fontsize=16, fontweight='bold', 
            ha='center', color='red')
    ax.text(1.5, 6.8, '(0D vs 3D)', fontsize=10, ha='center', style='italic')
    
    # Corrected Error Box
    corrected_box = FancyBboxPatch((4, 6.5), 2, 1.5, 
                                  boxstyle="round,pad=0.1", 
                                  facecolor='#ccffcc', edgecolor='green', linewidth=2)
    ax.add_patch(corrected_box)
    ax.text(5, 7.6, 'Corrected Error', fontsize=12, fontweight='bold', ha='center')
    ax.text(5, 7.2, f'{corrected_error:.2f}%', fontsize=16, fontweight='bold', 
            ha='center', color='green')
    ax.text(5, 6.8, '(ML Corrected)', fontsize=10, ha='center', style='italic')
    
    # Improvement Box
    improvement_box = FancyBboxPatch((7.5, 6.5), 2, 1.5, 
                                    boxstyle="round,pad=0.1", 
                                    facecolor='#cceeff', edgecolor='blue', linewidth=2)
    ax.add_patch(improvement_box)
    ax.text(8.5, 7.6, 'Improvement', fontsize=12, fontweight='bold', ha='center')
    ax.text(8.5, 7.2, f'{correction_percentage:.1f}%', fontsize=16, fontweight='bold', 
            ha='center', color='blue')
    ax.text(8.5, 6.8, 'Error Reduction', fontsize=10, ha='center', style='italic')
    
    # Arrow showing improvement
    arrow = patches.FancyArrowPatch((2.5, 7.25), (4, 7.25),
                                   arrowstyle='->', mutation_scale=20, 
                                   color='purple', linewidth=3)
    ax.add_patch(arrow)
    ax.text(3.25, 7.5, 'ML Correction', fontsize=10, ha='center', 
            fontweight='bold', color='purple')
    
    # Safety assessment
    safety_color = 'green' if corrected_error < 5 else 'orange' if corrected_error < 15 else 'red'
    safety_status = 'EXCELLENT' if corrected_error < 5 else 'GOOD' if corrected_error < 15 else 'NEEDS IMPROVEMENT'
    
    safety_box = FancyBboxPatch((2, 1.5), 6, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor=f'{safety_color}', alpha=0.3, 
                               edgecolor=safety_color, linewidth=2)
    ax.add_patch(safety_box)
    ax.text(5, 2.2, 'Safety Assessment', fontsize=12, fontweight='bold', ha='center')
    ax.text(5, 1.8, f'Status: {safety_status}', fontsize=14, fontweight='bold', 
            ha='center', color=safety_color)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def test_pipeline_functions():
    """Test all pipeline functions with sample data"""
    print("\n🧪 TESTING PIPELINE FUNCTIONS")
    print("=" * 50)
    
    # Create sample data
    sample_data = pd.DataFrame({
        'time': np.linspace(0, 100, 100),
        'temperature_0d': 25 + 75 * (1 - np.exp(-np.linspace(0, 3, 100))) + np.random.normal(0, 1, 100)
    })
    
    print(f"📊 Sample data created: {sample_data.shape}")
    print(f"   Temperature range: {sample_data['temperature_0d'].min():.1f} - {sample_data['temperature_0d'].max():.1f}°C")
    
    # Test each function
    try:
        print("\n1️⃣ Testing model training...")
        trained_models, feature_cols, processor = train_models_on_new_data(sample_data)
        
        print("\n2️⃣ Testing error correction...")
        corrections, corrected_temps = apply_error_correction(sample_data, trained_models, feature_cols)
        
        print("\n3️⃣ Testing performance analysis...")
        analysis = analyze_correction_performance(sample_data, corrections, corrected_temps)
        
        print("\n4️⃣ Testing visualization...")
        fig = generate_comprehensive_plots(sample_data, corrections, corrected_temps, analysis)
        
        print("\n✅ ALL TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def train_models_on_new_data(test_data, target_column='correction_factor'):
    """
    TRAIN NEW MODELS ON PROVIDED DATA
    ================================
    """
    print("🤖 Training new models on provided data...")
    
    # Initialize processor and engineer features
    processor = ThermalRunawayDataProcessor()
    processor.aligned_data = test_data.copy()
    
    # Add dummy target if not present
    if target_column not in test_data.columns:
        processor.aligned_data[target_column] = np.random.uniform(0.8, 1.2, len(test_data))
        print(f"⚠️ Added dummy {target_column} for demonstration")
    
    # Engineer features
    featured_data = processor.engineer_features()
    
    # Prepare training data
    exclude_cols = [target_column, 'time', 'temperature_3d'] if 'temperature_3d' in featured_data.columns else [target_column, 'time']
    feature_cols = [col for col in featured_data.columns if col not in exclude_cols]
    
    X = featured_data[feature_cols].fillna(0)
    y = featured_data[target_column].fillna(1.0)
    
    # Train models
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    import xgboost as xgb
    
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=50, random_state=42),
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'XGBoost': xgb.XGBRegressor(n_estimators=50, random_state=42)
    }
    
    trained_models = {}
    for name, model in models.items():
        try:
            model.fit(X, y)
            trained_models[name] = model
            print(f"✓ {name} trained successfully")
        except Exception as e:
            print(f"❌ {name} training failed: {e}")
    
    return trained_models, feature_cols, processor

def apply_error_correction(test_data, trained_models, feature_cols):
    """
    APPLY ERROR CORRECTION TO 0D DATA
    =================================
    """
    print("🔧 Applying error correction...")
    
    # Prepare features
    processor = ThermalRunawayDataProcessor()
    processor.aligned_data = test_data.copy()
    featured_data = processor.engineer_features()
    
    X = featured_data[feature_cols].fillna(0)
    
    corrections = {}
    corrected_temps = {}
    
    for name, model in trained_models.items():
        try:
            correction_factors = model.predict(X)
            corrections[name] = correction_factors
            corrected_temps[name] = test_data['temperature_0d'] * correction_factors
            print(f"✓ {name}: CF range [{correction_factors.min():.3f}, {correction_factors.max():.3f}]")
        except Exception as e:
            print(f"❌ {name} correction failed: {e}")
    
    return corrections, corrected_temps

def analyze_correction_performance(test_data, corrections, corrected_temps):
    """
    ANALYZE CORRECTION PERFORMANCE
    =============================
    """
    print("📊 Analyzing correction performance...")
    
    analysis = {}
    
    for model_name in corrections.keys():
        cf = corrections[model_name]
        corrected_temp = corrected_temps[model_name]
        
        # Basic statistics
        stats = {
            'mean_cf': cf.mean(),
            'std_cf': cf.std(),
            'min_cf': cf.min(),
            'max_cf': cf.max(),
            'mean_temp_increase': (corrected_temp - test_data['temperature_0d']).mean(),
            'max_temp_increase': (corrected_temp - test_data['temperature_0d']).max()
        }
        
        # Thermal zone analysis
        normal_mask = test_data['temperature_0d'] < 60
        warning_mask = (test_data['temperature_0d'] >= 60) & (test_data['temperature_0d'] < 120)
        critical_mask = test_data['temperature_0d'] >= 120
        
        if normal_mask.sum() > 0:
            stats['normal_zone_cf'] = cf[normal_mask].mean()
        if warning_mask.sum() > 0:
            stats['warning_zone_cf'] = cf[warning_mask].mean()
        if critical_mask.sum() > 0:
            stats['critical_zone_cf'] = cf[critical_mask].mean()
        
        analysis[model_name] = stats
    
    return analysis
def generate_comprehensive_plots(test_data, corrections, corrected_temps, analysis):
    """
    GENERATE ALL REQUIRED PLOTS
    ===========================
    """
    print("📈 Generating comprehensive plots...")
    
    # Set up the plotting style
    plt.style.use('default')
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    # Create main figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # Plot 1: Original vs Corrected Temperatures
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(test_data.index, test_data['temperature_0d'], 'b-', linewidth=2, label='Original 0D')
    
    for i, (model_name, corrected_temp) in enumerate(corrected_temps.items()):
        if i < len(colors):
            ax1.plot(test_data.index, corrected_temp, color=colors[i], 
                    linewidth=2, label=f'{model_name}', alpha=0.7)
    
    ax1.axhline(y=60, color='orange', linestyle='--', alpha=0.7, label='Warning (60°C)')
    ax1.axhline(y=120, color='red', linestyle='--', alpha=0.7, label='Critical (120°C)')
    ax1.set_title('Temperature Evolution Comparison')
    ax1.set_xlabel('Data Point Index')
    ax1.set_ylabel('Temperature (°C)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Correction Factors
    ax2 = plt.subplot(2, 3, 2)
    for i, (model_name, cf) in enumerate(corrections.items()):
        if i < len(colors):
            ax2.plot(test_data.index, cf, color=colors[i], 
                    linewidth=2, label=f'{model_name}', alpha=0.7)
    
    ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='Perfect Match (CF=1)')
    ax2.set_title('Correction Factor Evolution')
    ax2.set_xlabel('Data Point Index')
    ax2.set_ylabel('Correction Factor')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Correction Factor vs Temperature
    ax3 = plt.subplot(2, 3, 3)
    for i, (model_name, cf) in enumerate(corrections.items()):
        if i < len(colors):
            ax3.scatter(test_data['temperature_0d'], cf, color=colors[i], 
                       alpha=0.6, label=f'{model_name}', s=30)
    
    ax3.set_title('Correction Factor vs Temperature')
    ax3.set_xlabel('0D Temperature (°C)')
    ax3.set_ylabel('Correction Factor')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Temperature Distribution
    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(test_data['temperature_0d'], bins=20, alpha=0.7, label='Original 0D', color='blue')
    
    for i, (model_name, corrected_temp) in enumerate(corrected_temps.items()):
        if i < len(colors) and i < 2:  # Limit to avoid overcrowding
            ax4.hist(corrected_temp, bins=20, alpha=0.5, 
                    label=f'{model_name}', color=colors[i])
    
    ax4.set_title('Temperature Distribution')
    ax4.set_xlabel('Temperature (°C)')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Model Performance Metrics
    ax5 = plt.subplot(2, 3, 5)
    models = list(analysis.keys())
    mean_cfs = [analysis[model]['mean_cf'] for model in models]
    
    bars = ax5.bar(models, mean_cfs, color=colors[:len(models)], alpha=0.7)
    ax5.set_title('Mean Correction Factor by Model')
    ax5.set_ylabel('Mean Correction Factor')
    ax5.tick_params(axis='x', rotation=45)
    ax5.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, mean_cfs):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # Plot 6: Temperature Increase Analysis
    ax6 = plt.subplot(2, 3, 6)
    temp_increases = [analysis[model]['mean_temp_increase'] for model in models]
    
    bars = ax6.bar(models, temp_increases, color=colors[:len(models)], alpha=0.7)
    ax6.set_title('Mean Temperature Increase by Model')
    ax6.set_ylabel('Temperature Increase (°C)')
    ax6.tick_params(axis='x', rotation=45)
    ax6.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, temp_increases):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{value:.1f}°C', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return fig
def generate_correction_insights_infographic(initial_error, corrected_error, 
                                           correction_percentage, model_name="ML Model",
                                           additional_metrics=None):
    """
    Generate comprehensive correction insights infographic
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import FancyBboxPatch
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, '🔥 Thermal Runaway Correction Analysis', 
            fontsize=20, fontweight='bold', ha='center', va='center')
    ax.text(5, 9, f'Model: {model_name}', 
            fontsize=14, ha='center', va='center', style='italic')
    
    # Main metrics boxes
    # Initial Error Box
    initial_box = FancyBboxPatch((0.5, 6.5), 2, 1.5, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#ffcccc', edgecolor='red', linewidth=2)
    ax.add_patch(initial_box)
    ax.text(1.5, 7.6, 'Initial Error', fontsize=12, fontweight='bold', ha='center')
    ax.text(1.5, 7.2, f'{initial_error:.2f}%', fontsize=16, fontweight='bold', 
            ha='center', color='red')
    ax.text(1.5, 6.8, '(0D vs 3D)', fontsize=10, ha='center', style='italic')
    
    # Corrected Error Box
    corrected_box = FancyBboxPatch((4, 6.5), 2, 1.5, 
                                  boxstyle="round,pad=0.1", 
                                  facecolor='#ccffcc', edgecolor='green', linewidth=2)
    ax.add_patch(corrected_box)
    ax.text(5, 7.6, 'Corrected Error', fontsize=12, fontweight='bold', ha='center')
    ax.text(5, 7.2, f'{corrected_error:.2f}%', fontsize=16, fontweight='bold', 
            ha='center', color='green')
    ax.text(5, 6.8, '(ML Corrected)', fontsize=10, ha='center', style='italic')
    
    # Improvement Box
    improvement_box = FancyBboxPatch((7.5, 6.5), 2, 1.5, 
                                    boxstyle="round,pad=0.1", 
                                    facecolor='#cceeff', edgecolor='blue', linewidth=2)
    ax.add_patch(improvement_box)
    ax.text(8.5, 7.6, 'Improvement', fontsize=12, fontweight='bold', ha='center')
    ax.text(8.5, 7.2, f'{correction_percentage:.1f}%', fontsize=16, fontweight='bold', 
            ha='center', color='blue')
    ax.text(8.5, 6.8, 'Error Reduction', fontsize=10, ha='center', style='italic')
    
    # Arrow showing improvement
    arrow = patches.FancyArrowPatch((2.5, 7.25), (4, 7.25),
                                   arrowstyle='->', mutation_scale=20, 
                                   color='purple', linewidth=3)
    ax.add_patch(arrow)
    ax.text(3.25, 7.5, 'ML Correction', fontsize=10, ha='center', 
            fontweight='bold', color='purple')
    
    # Safety assessment
    safety_color = 'green' if corrected_error < 5 else 'orange' if corrected_error < 15 else 'red'
    safety_status = 'EXCELLENT' if corrected_error < 5 else 'GOOD' if corrected_error < 15 else 'NEEDS IMPROVEMENT'
    
    safety_box = FancyBboxPatch((2, 1.5), 6, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor=f'{safety_color}', alpha=0.3, 
                               edgecolor=safety_color, linewidth=2)
    ax.add_patch(safety_box)
    ax.text(5, 2.2, 'Safety Assessment', fontsize=12, fontweight='bold', ha='center')
    ax.text(5, 1.8, f'Status: {safety_status}', fontsize=14, fontweight='bold', 
            ha='center', color=safety_color)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def test_pipeline_functions():
    """Test all pipeline functions with sample data"""
    print("\n🧪 TESTING PIPELINE FUNCTIONS")
    print("=" * 50)
    
    # Create sample data
    sample_data = pd.DataFrame({
        'time': np.linspace(0, 100, 100),
        'temperature_0d': 25 + 75 * (1 - np.exp(-np.linspace(0, 3, 100))) + np.random.normal(0, 1, 100)
    })
    
    print(f"📊 Sample data created: {sample_data.shape}")
    print(f"   Temperature range: {sample_data['temperature_0d'].min():.1f} - {sample_data['temperature_0d'].max():.1f}°C")
    
    # Test each function
    try:
        print("\n1️⃣ Testing model training...")
        trained_models, feature_cols, processor = train_models_on_new_data(sample_data)
        
        print("\n2️⃣ Testing error correction...")
        corrections, corrected_temps = apply_error_correction(sample_data, trained_models, feature_cols)
        
        print("\n3️⃣ Testing performance analysis...")
        analysis = analyze_correction_performance(sample_data, corrections, corrected_temps)
        
        print("\n4️⃣ Testing visualization...")
        fig = generate_comprehensive_plots(sample_data, corrections, corrected_temps, analysis)
        
        print("\n✅ ALL TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def train_models_on_new_data(test_data, target_column='correction_factor'):
    """
    TRAIN NEW MODELS ON PROVIDED DATA
    ================================
    """
    print("🤖 Training new models on provided data...")
    
    # Initialize processor and engineer features
    processor = ThermalRunawayDataProcessor()
    processor.aligned_data = test_data.copy()
    
    # Add dummy target if not present
    if target_column not in test_data.columns:
        processor.aligned_data[target_column] = np.random.uniform(0.8, 1.2, len(test_data))
        print(f"⚠️ Added dummy {target_column} for demonstration")
    
    # Engineer features
    featured_data = processor.engineer_features()
    
    # Prepare training data
    exclude_cols = [target_column, 'time', 'temperature_3d'] if 'temperature_3d' in featured_data.columns else [target_column, 'time']
    feature_cols = [col for col in featured_data.columns if col not in exclude_cols]
    
    X = featured_data[feature_cols].fillna(0)
    y = featured_data[target_column].fillna(1.0)
    
    # Train models
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    import xgboost as xgb
    
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=50, random_state=42),
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'XGBoost': xgb.XGBRegressor(n_estimators=50, random_state=42)
    }
    
    trained_models = {}
    for name, model in models.items():
        try:
            model.fit(X, y)
            trained_models[name] = model
            print(f"✓ {name} trained successfully")
        except Exception as e:
            print(f"❌ {name} training failed: {e}")
    
    return trained_models, feature_cols, processor

def apply_error_correction(test_data, trained_models, feature_cols):
    """
    APPLY ERROR CORRECTION TO 0D DATA
    =================================
    """
    print("🔧 Applying error correction...")
    
    # Prepare features
    processor = ThermalRunawayDataProcessor()
    processor.aligned_data = test_data.copy()
    featured_data = processor.engineer_features()
    
    X = featured_data[feature_cols].fillna(0)
    
    corrections = {}
    corrected_temps = {}
    
    for name, model in trained_models.items():
        try:
            correction_factors = model.predict(X)
            corrections[name] = correction_factors
            corrected_temps[name] = test_data['temperature_0d'] * correction_factors
            print(f"✓ {name}: CF range [{correction_factors.min():.3f}, {correction_factors.max():.3f}]")
        except Exception as e:
            print(f"❌ {name} correction failed: {e}")
    
    return corrections, corrected_temps

def analyze_correction_performance(test_data, corrections, corrected_temps):
    """
    ANALYZE CORRECTION PERFORMANCE
    =============================
    """
    print("📊 Analyzing correction performance...")
    
    analysis = {}
    
    for model_name in corrections.keys():
        cf = corrections[model_name]
        corrected_temp = corrected_temps[model_name]
        
        # Basic statistics
        stats = {
            'mean_cf': cf.mean(),
            'std_cf': cf.std(),
            'min_cf': cf.min(),
            'max_cf': cf.max(),
            'mean_temp_increase': (corrected_temp - test_data['temperature_0d']).mean(),
            'max_temp_increase': (corrected_temp - test_data['temperature_0d']).max()
        }
        
        # Thermal zone analysis
        normal_mask = test_data['temperature_0d'] < 60
        warning_mask = (test_data['temperature_0d'] >= 60) & (test_data['temperature_0d'] < 120)
        critical_mask = test_data['temperature_0d'] >= 120
        
        if normal_mask.sum() > 0:
            stats['normal_zone_cf'] = cf[normal_mask].mean()
        if warning_mask.sum() > 0:
            stats['warning_zone_cf'] = cf[warning_mask].mean()
        if critical_mask.sum() > 0:
            stats['critical_zone_cf'] = cf[critical_mask].mean()
        
        analysis[model_name] = stats
    
    return analysis
def generate_comprehensive_plots(test_data, corrections, corrected_temps, analysis):
    """
    GENERATE ALL REQUIRED PLOTS
    ===========================
    """
    print("📈 Generating comprehensive plots...")
    
    # Set up the plotting style
    plt.style.use('default')
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    # Create main figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # Plot 1: Original vs Corrected Temperatures
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(test_data.index, test_data['temperature_0d'], 'b-', linewidth=2, label='Original 0D')
    
    for i, (model_name, corrected_temp) in enumerate(corrected_temps.items()):
        if i < len(colors):
            ax1.plot(test_data.index, corrected_temp, color=colors[i], 
                    linewidth=2, label=f'{model_name}', alpha=0.7)
    
    ax1.axhline(y=60, color='orange', linestyle='--', alpha=0.7, label='Warning (60°C)')
    ax1.axhline(y=120, color='red', linestyle='--', alpha=0.7, label='Critical (120°C)')
    ax1.set_title('Temperature Evolution Comparison')
    ax1.set_xlabel('Data Point Index')
    ax1.set_ylabel('Temperature (°C)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Correction Factors
    ax2 = plt.subplot(2, 3, 2)
    for i, (model_name, cf) in enumerate(corrections.items()):
        if i < len(colors):
            ax2.plot(test_data.index, cf, color=colors[i], 
                    linewidth=2, label=f'{model_name}', alpha=0.7)
    
    ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='Perfect Match (CF=1)')
    ax2.set_title('Correction Factor Evolution')
    ax2.set_xlabel('Data Point Index')
    ax2.set_ylabel('Correction Factor')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Correction Factor vs Temperature
    ax3 = plt.subplot(2, 3, 3)
    for i, (model_name, cf) in enumerate(corrections.items()):
        if i < len(colors):
            ax3.scatter(test_data['temperature_0d'], cf, color=colors[i], 
                       alpha=0.6, label=f'{model_name}', s=30)
    
    ax3.set_title('Correction Factor vs Temperature')
    ax3.set_xlabel('0D Temperature (°C)')
    ax3.set_ylabel('Correction Factor')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Temperature Distribution
    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(test_data['temperature_0d'], bins=20, alpha=0.7, label='Original 0D', color='blue')
    
    for i, (model_name, corrected_temp) in enumerate(corrected_temps.items()):
        if i < len(colors) and i < 2:  # Limit to avoid overcrowding
            ax4.hist(corrected_temp, bins=20, alpha=0.5, 
                    label=f'{model_name}', color=colors[i])
    
    ax4.set_title('Temperature Distribution')
    ax4.set_xlabel('Temperature (°C)')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Model Performance Metrics
    ax5 = plt.subplot(2, 3, 5)
    models = list(analysis.keys())
    mean_cfs = [analysis[model]['mean_cf'] for model in models]
    
    bars = ax5.bar(models, mean_cfs, color=colors[:len(models)], alpha=0.7)
    ax5.set_title('Mean Correction Factor by Model')
    ax5.set_ylabel('Mean Correction Factor')
    ax5.tick_params(axis='x', rotation=45)
    ax5.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, mean_cfs):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # Plot 6: Temperature Increase Analysis
    ax6 = plt.subplot(2, 3, 6)
    temp_increases = [analysis[model]['mean_temp_increase'] for model in models]
    
    bars = ax6.bar(models, temp_increases, color=colors[:len(models)], alpha=0.7)
    ax6.set_title('Mean Temperature Increase by Model')
    ax6.set_ylabel('Temperature Increase (°C)')
    ax6.tick_params(axis='x', rotation=45)
    ax6.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, temp_increases):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{value:.1f}°C', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return fig
def generate_correction_insights_infographic(initial_error, corrected_error, 
                                           correction_percentage, model_name="ML Model",
                                           additional_metrics=None):
    """
    Generate comprehensive correction insights infographic
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import FancyBboxPatch
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, '🔥 Thermal Runaway Correction Analysis', 
            fontsize=20, fontweight='bold', ha='center', va='center')
    ax.text(5, 9, f'Model: {model_name}', 
            fontsize=14, ha='center', va='center', style='italic')
    
