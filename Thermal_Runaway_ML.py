# thermal_runaway_ml_pipeline.py
"""
COMPREHENSIVE MACHINE LEARNING PIPELINE FOR THERMAL RUNAWAY MODEL CORRECTION
===========================================================================

This pipeline integrates 0D and 3D simulation data to create correction factors
for improved thermal runaway prediction in lithium-ion batteries.

Based on research: "A lumped electrochemical-thermal model for simulating detection and
mitigation of thermal runaway in lithium-ion batteries under different ambient conditions"

Author: Your Name
Date: June 2025
"""

# ============================================================================
# IMPORT REQUIRED LIBRARIES
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Or 'Qt5Agg' if you have Qt installed
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib  # For saving/loading models
import warnings
warnings.filterwarnings('ignore')

# Set plotting style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("All required libraries imported successfully!")
print("="*80)

# ============================================================================
# DATA PROCESSING CLASS
# ============================================================================

class ThermalRunawayDataProcessor:
    """
    MAIN DATA PROCESSING CLASS
    =========================
    
    This class handles:
    1. Loading CSV files (0D and 3D simulation data)
    2. Data preprocessing and alignment
    3. Correction factor calculation
    4. Feature engineering for ML models
    5. Error percentage calculations
    """
    
    def __init__(self, config=None):
        """Initialize the data processor with empty scalers and feature storage"""
        self.config = config
        self.scaler_0d = StandardScaler()  # For normalizing 0D data
        self.scaler_3d = StandardScaler()  # For normalizing 3D data
        self.feature_names = []            # Store feature column names
        self.initial_error_percentage = 0  # Store initial 0D vs 3D error
        self.corrected_error_percentage = 0 # Store corrected error after ML

        print("✓ Data processor initialized successfully")

    def load_data(self, xlsx_0d_path, xlsx_3d_paths):
        """
        LOAD MULTI-SHEET/MULTI-FILE 0D AND 3D DATA
        ==========================================
        Args:
            xlsx_0d_path (str): Path to your 0D Excel file (all points in one file, different sheets or columns)
            xlsx_3d_paths (list): List of paths to your 3D Excel files (A, B, C, D)
        Returns:
            tuple: (0D DataFrame, 3D DataFrame)
        """

        print("\n" + "="*50)
        print("LOADING SIMULATION DATA FROM FILES")
        print("="*50)

        # --- Load 0D data for all points A, B, C, D from specific sheets ---
        print(f"Loading 0D data for points A, B, C, D from: {xlsx_0d_path}")
        try:
            xls = pd.ExcelFile(xlsx_0d_path)
            data_0d_list = []
            # Use the correct sheet names as per your Excel file
            for point, sheet in zip(['A', 'B', 'C', 'D'], ['POINT-A', 'POINT-B', 'POINT-C', 'POINT-D']):
                if sheet in xls.sheet_names:
                    df = pd.read_excel(xlsx_0d_path, sheet_name=sheet)
                    df['point'] = point
                    data_0d_list.append(df)
                else:
                    print(f"⚠️ Sheet '{sheet}' not found in 0D Excel file.")
            if data_0d_list:
                self.data_0d = pd.concat(data_0d_list, ignore_index=True)

                # Ensure correct columns and numeric types
                self.data_0d = self.data_0d.rename(columns={self.data_0d.columns[0]: 'time', self.data_0d.columns[1]: 'temperature_0d'})

                self.data_0d['time'] = pd.to_numeric(self.data_0d['time'], errors='coerce')
                self.data_0d['temperature_0d'] = pd.to_numeric(self.data_0d['temperature_0d'], errors='coerce')
                self.data_0d = self.data_0d.dropna(subset=['time', 'temperature_0d'])

                print(f"✓ 0D data loaded for points A, B, C, D! Shape: {self.data_0d.shape}")
                print(f"  - Columns: {self.data_0d.columns.tolist()}")
                print(self.data_0d.head())
            else:
                raise ValueError("No 0D sheets found. Please check sheet names in your Excel file.")
        except Exception as e:
            print(f"❌ ERROR loading 0D data: {e}")
            raise

        # --- Load 3D data for all points from separate Excel files ---
        print(f"\nLoading 3D data for points A, B, C, D from: {xlsx_3d_paths}")
        data_3d_list = []
        for idx, path in enumerate(xlsx_3d_paths):
            try:
                point = ['A', 'B', 'C', 'D'][idx]  # Assign point by order
                df = pd.read_excel(path)
                df['point'] = point
                data_3d_list.append(df)
            except Exception as e:
                print(f"❌ ERROR loading 3D data from {path}: {e}")
        self.data_3d = pd.concat(data_3d_list, ignore_index=True)
        print(f"✓ 3D data loaded and combined! Shape: {self.data_3d.shape}")
        print(f"  - Columns: {self.data_3d.columns.tolist()}")

        # --- Standardize column names ---
        print("\n📝 Standardizing column names...")

        # For 0D data: keep only relevant columns and ensure correct names
        # If your 0D data already has columns named 'time', 'temperature_0d', and 'point', just keep those
        self.data_0d = self.data_0d[['time', 'temperature_0d', 'point']]

        # For 3D data: rename columns explicitly based on your actual file structure
        # Adjust these names if your 3D Excel files have different column names
        self.data_3d = self.data_3d.rename(columns={
            'Convergence history of Static Temperature on battery (in SI units)': 'temperature_3d',
            'Unnamed: 1': 'flow_time',
            'Unnamed: 2': 'time_step'
        })

        # Clean up non-numeric rows
        self.data_3d['temperature_3d'] = pd.to_numeric(self.data_3d['temperature_3d'], errors='coerce')
        self.data_3d['flow_time'] = pd.to_numeric(self.data_3d['flow_time'], errors='coerce')
        self.data_3d['time_step'] = pd.to_numeric(self.data_3d['time_step'], errors='coerce')
        self.data_3d = self.data_3d.dropna(subset=['temperature_3d', 'flow_time', 'time_step'])
        self.data_3d = self.data_3d.reset_index(drop=True)

        print("✓ Column names standardized")

        # Fix 0D data (Kelvin to Celsius if needed)
        if self.data_0d['temperature_0d'].mean() > 200:
            self.data_0d['temperature_0d'] = self.data_0d['temperature_0d'] - 273.15

        # Fix 3D data (scale and convert if needed)
        if self.data_3d['temperature_3d'].max() > 1000:
            self.data_3d['temperature_3d'] = self.data_3d['temperature_3d'] / 1000
        if self.data_3d['temperature_3d'].mean() > 200:
            self.data_3d['temperature_3d'] = self.data_3d['temperature_3d'] - 273.15


        # --- Display data summary ---
        print("\n📊 DATA SUMMARY:")
        print("0D Data Preview:")
        print(self.data_0d.head())
        print(f"0D Temperature range: {self.data_0d['temperature_0d'].min():.2f} to {self.data_0d['temperature_0d'].max():.2f}")

        print("\n3D Data Preview:")
        print(self.data_3d.head())
        print(f"3D Temperature range: {self.data_3d['temperature_3d'].min():.2f} to {self.data_3d['temperature_3d'].max():.2f}")

        return self.data_0d, self.data_3d
    
    def visualize_raw_data(self):
        """
        VISUALIZATION: UNDERSTAND YOUR DATA PATTERNS
        ==========================================
        
        Creates 4 plots to help you understand:
        1. 0D temperature evolution over time
        2. 3D temperature evolution over time  
        3. Temperature distribution comparison
        4. Direct correlation (if data sizes match)
        """
        
        print("\n" + "="*50)
        print("VISUALIZING RAW DATA PATTERNS")
        print("="*50)
        
        # Create figure with 4 subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Raw Simulation Data Analysis', fontsize=16, fontweight='bold')
        
        # PLOT 1: 0D Temperature vs Time
        axes[0,0].plot(self.data_0d['time'], self.data_0d['temperature_0d'], 
                       'b-', linewidth=2, label='0D Model')
        axes[0,0].set_title('0D Model: Temperature Evolution', fontweight='bold')
        axes[0,0].set_xlabel('Time')
        axes[0,0].set_ylabel('Temperature (0D)')
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].legend()
        
        # Add thermal runaway threshold lines (based on Li-ion battery research)
        axes[0,0].axhline(y=60, color='orange', linestyle='--', alpha=0.7, label='Warning (60°C)')
        axes[0,0].axhline(y=120, color='red', linestyle='--', alpha=0.7, label='Critical (120°C)')
        axes[0,0].legend()
        
        # PLOT 2: 3D Temperature vs Flow Time
        axes[0,1].plot(self.data_3d['flow_time'], self.data_3d['temperature_3d'], 
                       'r-', linewidth=2, label='3D Model')
        axes[0,1].set_title('3D Model: Temperature Evolution', fontweight='bold')
        axes[0,1].set_xlabel('Flow Time')
        axes[0,1].set_ylabel('Temperature (3D)')
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].legend()
        
        # Add thermal runaway threshold lines
        axes[0,1].axhline(y=60, color='orange', linestyle='--', alpha=0.7, label='Warning (60°C)')
        axes[0,1].axhline(y=120, color='red', linestyle='--', alpha=0.7, label='Critical (120°C)')
        axes[0,1].legend()
        
        # PLOT 3: Temperature Distribution Comparison (FIXED)
        try:
            # Clean and filter data first
            temp_0d_clean = self.data_0d['temperature_0d'].dropna()
            temp_3d_clean = self.data_3d['temperature_3d'].dropna()
    
            # Convert to numeric and remove any invalid values
            temp_0d_clean = pd.to_numeric(temp_0d_clean, errors='coerce').dropna()
            temp_3d_clean = pd.to_numeric(temp_3d_clean, errors='coerce').dropna()
    
            # Check if we have valid data
            if len(temp_0d_clean) == 0 or len(temp_3d_clean) == 0:
                axes[1,0].text(0.5, 0.5, 'No valid temperature data available', 
                      ha='center', va='center', transform=axes[1,0].transAxes,
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7))
                axes[1,0].set_title('Temperature Distribution - Data Error')
                return
    
            # Handle temperature unit conversion if needed (Kelvin to Celsius)
            if temp_0d_clean.max() > 500:  # Likely in Kelvin
                temp_0d_clean = temp_0d_clean - 273.15
            if temp_3d_clean.max() > 500:  # Likely in Kelvin
                temp_3d_clean = temp_3d_clean - 273.15
    
            # Filter realistic temperature range (0-300°C for battery applications)
            temp_0d_clean = temp_0d_clean[(temp_0d_clean >= 0) & (temp_0d_clean <= 300)]
            temp_3d_clean = temp_3d_clean[(temp_3d_clean >= 0) & (temp_3d_clean <= 300)]
    
            # Calculate optimal number of bins
            n_bins_0d = min(30, max(10, len(temp_0d_clean) // 20))
            n_bins_3d = min(30, max(10, len(temp_3d_clean) // 20))
            n_bins = min(n_bins_0d, n_bins_3d)
    
            # Create histograms with proper scaling
            axes[1,0].hist(temp_0d_clean, bins=n_bins, alpha=0.6, 
                   label=f'0D Model (n={len(temp_0d_clean)})', 
                   color='blue', density=True, edgecolor='darkblue')
            axes[1,0].hist(temp_3d_clean, bins=n_bins, alpha=0.6, 
                   label=f'3D Model (n={len(temp_3d_clean)})', 
                   color='red', density=True, edgecolor='darkred')
    
            # Add statistical information
            mean_0d = temp_0d_clean.mean()
            mean_3d = temp_3d_clean.mean()
            std_0d = temp_0d_clean.std()
            std_3d = temp_3d_clean.std()
    
            # Add mean lines
            axes[1,0].axvline(mean_0d, color='blue', linestyle='--', linewidth=2, 
                     label=f'0D Mean: {mean_0d:.1f}°C')
            axes[1,0].axvline(mean_3d, color='red', linestyle='--', linewidth=2, 
                     label=f'3D Mean: {mean_3d:.1f}°C')
    
            # Add thermal runaway thresholds
            axes[1,0].axvline(60, color='orange', linestyle=':', alpha=0.8, 
                     label='Warning (60°C)')
            axes[1,0].axvline(120, color='darkred', linestyle=':', alpha=0.8, 
                label='Critical (120°C)')
    
        # Format plot
            axes[1,0].set_title('Temperature Distribution Comparison', fontweight='bold')
            axes[1,0].set_xlabel('Temperature (°C)')
            axes[1,0].set_ylabel('Probability Density')
            axes[1,0].legend(loc='upper right', fontsize=8)
            axes[1,0].grid(True, alpha=0.3)

    # Add statistics text box
            stats_text = (f'Statistics:\n'
                 f'0D: μ={mean_0d:.1f}°C, σ={std_0d:.1f}°C\n'
                 f'3D: μ={mean_3d:.1f}°C, σ={std_3d:.1f}°C\n'
                 f'Difference: {abs(mean_3d-mean_0d):.1f}°C')

            axes[1,0].text(0.02, 0.98, stats_text, transform=axes[1,0].transAxes,
                  verticalalignment='top', bbox=dict(boxstyle='round', 
                  facecolor='lightblue', alpha=0.8), fontsize=9)

        except Exception as e:
            # Error handling
            axes[1,0].text(0.5, 0.5, f'Error creating distribution plot:\n{str(e)}', 
                  ha='center', va='center', transform=axes[1,0].transAxes,
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
            axes[1,0].set_title('Temperature Distribution - Error')
            print(f"Error in temperature distribution plot: {e}")


        # PLOT 4: Correlation Analysis (if possible)
        from scipy.interpolate import interp1d
        # PLOT 4: 0D vs 3D Temperature Correlation (with interpolation)
        from scipy.interpolate import interp1d

        # Find overlapping time range
        min_time = max(self.data_0d['time'].min(), self.data_3d['flow_time'].min())
        max_time = min(self.data_0d['time'].max(), self.data_3d['flow_time'].max())
        n_points = min(len(self.data_0d), len(self.data_3d))
        common_time = np.linspace(min_time, max_time, n_points)

        # Interpolate both datasets onto the common grid
        interp_0d = interp1d(self.data_0d['time'], self.data_0d['temperature_0d'], kind='linear', fill_value='extrapolate')
        interp_3d = interp1d(self.data_3d['flow_time'], self.data_3d['temperature_3d'], kind='linear', fill_value='extrapolate')
        temp_0d_aligned = interp_0d(common_time)
        temp_3d_aligned = interp_3d(common_time)

        # Now plot the correlation
        axes[1,1].scatter(temp_0d_aligned, temp_3d_aligned, alpha=0.6, color='purple')
        axes[1,1].plot(
            [min(temp_0d_aligned.min(), temp_3d_aligned.min()), max(temp_0d_aligned.max(), temp_3d_aligned.max())],
            [min(temp_0d_aligned.min(), temp_3d_aligned.min()), max(temp_0d_aligned.max(), temp_3d_aligned.max())],
            'k--', alpha=0.5, label='Perfect Correlation'
        )
        axes[1,1].set_title('0D vs 3D Temperature Correlation', fontweight='bold')
        axes[1,1].set_xlabel('0D Temperature')
        axes[1,1].set_ylabel('3D Temperature')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)

                
        plt.tight_layout()
        plt.show()
        
        print("✓ Raw data visualization completed")
    
    def align_data_by_interpolation(self):
        """
        CRITICAL STEP: DATA ALIGNMENT
        ============================
        
        Since 0D and 3D simulations may have different time steps,
        we need to align them using interpolation for fair comparison.
        
        This function:
        1. Finds common time range between both datasets
        2. Creates uniform time grid
        3. Interpolates both temperature profiles to common grid
        
        Returns:
            DataFrame: Aligned data with common time steps
        """
        
        print("\n" + "="*50)
        print("ALIGNING DATA USING INTERPOLATION")
        print("="*50)
        
        # Find overlapping time range
        min_time_0d = self.data_0d['time'].min()
        max_time_0d = self.data_0d['time'].max()
        min_time_3d = self.data_3d['flow_time'].min()
        max_time_3d = self.data_3d['flow_time'].max()
        
        print(f"0D time range: {min_time_0d:.3f} to {max_time_0d:.3f}")
        print(f"3D time range: {min_time_3d:.3f} to {max_time_3d:.3f}")
        
        # Calculate common time range (intersection)
        min_time = max(min_time_0d, min_time_3d)
        max_time = min(max_time_0d, max_time_3d)
        
        print(f"Common time range: {min_time:.3f} to {max_time:.3f}")
        
        # Create uniform time grid
        # Use minimum of both dataset lengths to avoid extrapolation issues
        n_points = min(len(self.data_0d), len(self.data_3d))
        common_time = np.linspace(min_time, max_time, n_points)
        
        print(f"Creating {n_points} interpolated points")
        
        # Interpolate 0D temperature to common time grid
        temp_0d_interp = np.interp(common_time, 
                                   self.data_0d['time'], 
                                   self.data_0d['temperature_0d'])
        
        # Interpolate 3D temperature to common time grid
        temp_3d_interp = np.interp(common_time, 
                                   self.data_3d['flow_time'], 
                                   self.data_3d['temperature_3d'])
        
        # Create aligned dataset
        self.aligned_data = pd.DataFrame({
            'time': common_time,
            'temperature_0d': temp_0d_interp,
            'temperature_3d': temp_3d_interp
        })
        
        print(f"✓ Data alignment completed")
        print(f"  - Aligned dataset shape: {self.aligned_data.shape}")
        print(f"  - Time step: {(max_time - min_time) / (n_points - 1):.6f}")
        
        return self.aligned_data
    
    def calculate_correction_factor(self):
        """
        CORE CALCULATION: CORRECTION FACTOR COMPUTATION
        ==============================================
        
        The correction factor is the ratio: 3D_temperature / 0D_temperature
        This tells us how much the 0D model under/over-predicts compared to 3D.
        
        Also calculates initial error percentage between 0D and 3D models.
        
        Returns:
            DataFrame: Data with correction factors added
        """
        
        print("\n" + "="*50)
        print("CALCULATING CORRECTION FACTORS")
        print("="*50)
        
        # Remove points where 0D temperature is zero (avoid division by zero)
        initial_points = len(self.aligned_data)
        mask = self.aligned_data['temperature_0d'] != 0
        self.aligned_data = self.aligned_data[mask].copy()
        removed_points = initial_points - len(self.aligned_data)
        
        if removed_points > 0:
            print(f"⚠️  Removed {removed_points} points with zero 0D temperature")
        
        # CALCULATE CORRECTION FACTOR
        # correction_factor = T_3D / T_0D
        self.aligned_data['correction_factor'] = (
            self.aligned_data['temperature_3d'] / self.aligned_data['temperature_0d']
        )
        
        # CALCULATE INITIAL ERROR PERCENTAGE (0D vs 3D)
        # Error = |T_3D - T_0D| / T_3D * 100
        absolute_error = np.abs(self.aligned_data['temperature_3d'] - self.aligned_data['temperature_0d'])
        self.initial_error_percentage = np.mean(absolute_error / self.aligned_data['temperature_3d'] * 100)
        
        print(f"📊 INITIAL MODEL COMPARISON:")
        print(f"  - Average error between 0D and 3D: {self.initial_error_percentage:.2f}%")
        
        # REMOVE OUTLIERS IN CORRECTION FACTOR
        # Use IQR method to remove extreme outliers
        q1 = self.aligned_data['correction_factor'].quantile(0.25)
        q3 = self.aligned_data['correction_factor'].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        initial_size = len(self.aligned_data)
        self.aligned_data = self.aligned_data[
            (self.aligned_data['correction_factor'] >= lower_bound) &
            (self.aligned_data['correction_factor'] <= upper_bound)
        ].copy()
        outliers_removed = initial_size - len(self.aligned_data)
        
        if outliers_removed > 0:
            print(f"⚠️  Removed {outliers_removed} outlier points")
        
        # DISPLAY CORRECTION FACTOR STATISTICS
        print(f"\n📈 CORRECTION FACTOR STATISTICS:")
        print(f"  - Mean: {self.aligned_data['correction_factor'].mean():.4f}")
        print(f"  - Std:  {self.aligned_data['correction_factor'].std():.4f}")
        print(f"  - Min:  {self.aligned_data['correction_factor'].min():.4f}")
        print(f"  - Max:  {self.aligned_data['correction_factor'].max():.4f}")
        print(f"  - Final dataset size: {len(self.aligned_data)} points")
        
        return self.aligned_data
    
    def engineer_features(self):
        """
        FEATURE ENGINEERING: CREATE ML INPUT FEATURES
        =============================================
        
        Creates additional features based on thermal physics and time-series analysis:
        1. Normalized features (time, temperature)
        2. Derivatives (rate of temperature change)
        3. Rolling statistics (thermal inertia effects)
        4. Temperature range indicators (thermal runaway zones)
        5. Interaction features
        
        Returns:
            DataFrame: Enhanced dataset with engineered features
        """
        
        print("\n" + "="*50)
        print("ENGINEERING FEATURES FOR ML MODELS")
        print("="*50)
        
        # 1. TIME-BASED FEATURES
        print("🔧 Creating time-based features...")
        # Normalize time to [0, 1] range
        self.aligned_data['time_normalized'] = (
            (self.aligned_data['time'] - self.aligned_data['time'].min()) /
            (self.aligned_data['time'].max() - self.aligned_data['time'].min())
        )
        
        # 2. TEMPERATURE-BASED FEATURES
        print("🔧 Creating temperature-based features...")
        # Normalize 0D temperature to [0, 1] range
        self.aligned_data['temp_0d_normalized'] = (
            (self.aligned_data['temperature_0d'] - self.aligned_data['temperature_0d'].min()) /
            (self.aligned_data['temperature_0d'].max() - self.aligned_data['temperature_0d'].min())
        )
        
        # 3. DERIVATIVE FEATURES (Rate of Change)
        print("🔧 Creating derivative features...")
        # Temperature rate of change (important for thermal runaway detection)
        self.aligned_data['temp_0d_derivative'] = np.gradient(self.aligned_data['temperature_0d'])
        self.aligned_data['temp_3d_derivative'] = np.gradient(self.aligned_data['temperature_3d'])
        
        # 4. ROLLING STATISTICS (Thermal Inertia Effects)
        print("🔧 Creating rolling statistics...")
        # Window size: 5 points or 10% of data, whichever is smaller
        window_size = max(2, min(5, len(self.aligned_data) // 10))
        
        if window_size >= 2:
            # Rolling mean (smoothed temperature trend)
            self.aligned_data['temp_0d_rolling_mean'] = (
                self.aligned_data['temperature_0d'].rolling(window=window_size, center=True).mean()
            )
            # Rolling standard deviation (temperature variability)
            self.aligned_data['temp_0d_rolling_std'] = (
                self.aligned_data['temperature_0d'].rolling(window=window_size, center=True).std()
            )
        
        # Fill NaN values from rolling operations
        self.aligned_data = self.aligned_data.fillna(method='bfill').fillna(method='ffill')
        
        # 5. THERMAL RUNAWAY ZONE INDICATORS
        print("🔧 Creating thermal runaway zone indicators...")
        # Based on Li-ion battery thermal runaway research thresholds
        self.aligned_data['temp_range_low'] = (self.aligned_data['temperature_0d'] < 60).astype(int)      # Normal operation
        self.aligned_data['temp_range_medium'] = ((self.aligned_data['temperature_0d'] >= 60) & 
                                                 (self.aligned_data['temperature_0d'] < 120)).astype(int)  # Warning zone
        self.aligned_data['temp_range_high'] = (self.aligned_data['temperature_0d'] >= 120).astype(int)   # Critical zone
        
        # 6. INTERACTION FEATURES
        print("🔧 Creating interaction features...")
        # Temperature-time interaction (captures thermal evolution patterns)
        self.aligned_data['temp_time_interaction'] = (
            self.aligned_data['temperature_0d'] * self.aligned_data['time_normalized']
        )
        
        # Temperature-derivative interaction (captures acceleration effects)
        self.aligned_data['temp_derivative_interaction'] = (
            self.aligned_data['temperature_0d'] * self.aligned_data['temp_0d_derivative']
        )
        
        print(f"✓ Feature engineering completed")
        print(f"  - Final dataset shape: {self.aligned_data.shape}")
        print(f"  - Total features created: {self.aligned_data.shape[1] - 3}")  # Subtract time, temp_0d, temp_3d
        print(f"  - Feature columns: {[col for col in self.aligned_data.columns if col not in ['time', 'temperature_0d', 'temperature_3d', 'correction_factor']]}")
        
        return self.aligned_data

# ============================================================================
# MACHINE LEARNING PIPELINE CLASS
# ============================================================================

class ThermalRunawayMLPipeline:
    """
    COMPREHENSIVE ML PIPELINE CLASS
    ==============================
    
    This class implements:
    1. Multiple ML algorithms (ensemble approach)
    2. Model training and evaluation
    3. Performance comparison
    4. Feature importance analysis
    5. Ensemble predictions
    6. Error percentage calculations
    """
    
    def __init__(self):
        """Initialize ML pipeline with empty containers"""
        self.models = {}                    # Store trained models
        self.model_performance = {}         # Store performance metrics
        self.best_model = None             # Best performing model
        self.feature_importance = {}       # Feature importance scores
        self.test_error_percentages = {}   # Error percentages for each model
        
        print("✓ ML Pipeline initialized successfully")
        
    def prepare_features_target(self, data):
        """
        PREPARE DATA FOR MACHINE LEARNING
        ================================
        
        Separates features (X) from target variable (y) for ML training.
        
        Args:
            data (DataFrame): Processed data with features and target
            
        Returns:
            tuple: (X, y, feature_names)
        """
        
        print("\n" + "="*50)
        print("PREPARING FEATURES AND TARGET FOR ML")
        print("="*50)
        
        # Define feature columns (exclude target and identifier columns)
        exclude_cols = ['correction_factor', 'temperature_3d', 'time']
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        
        # Create feature matrix X
        X = data[feature_cols].copy()
        
        # Create target vector y (what we want to predict)
        y = data['correction_factor'].copy()
        
        print(f"📊 DATA PREPARATION SUMMARY:")
        print(f"  - Feature matrix shape: {X.shape}")
        print(f"  - Target vector shape: {y.shape}")
        print(f"  - Number of features: {len(feature_cols)}")
        print(f"  - Features used: {feature_cols}")
        
        # Check for any missing values
        missing_features = X.isnull().sum().sum()
        missing_target = y.isnull().sum()
        
        if missing_features > 0 or missing_target > 0:
            print(f"⚠️  Missing values detected: Features={missing_features}, Target={missing_target}")
            # Fill missing values if any
            X = X.fillna(X.mean())
            y = y.fillna(y.mean())
            print("✓ Missing values filled with mean values")
        
        return X, y, feature_cols
    def _train_point_models(self, point, features):
        """Train models with proper error handling"""
        point_results = {}
    
        for model_name, model in self.models.items():
            try:
            # Training logic here
                point_results[model_name] = { ... }
            except Exception as e:
                print(f"❌ {model_name} failed for {point}: {str(e)}")
                point_results[model_name] = {'error': str(e)}
    
        return point_results

    def _initialize_models(self):
        """Initialize ML models with optimized parameters"""
        return {
            'RandomForest': RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.config.random_state,
            n_jobs=-1
            ),
            'XGBoost': xgb.XGBRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.config.random_state,
            n_jobs=-1
            ),
        # Fixed line below
            'GradientBoosting': GradientBoostingRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            random_state=self.config.random_state
            ),
            'Ridge': Ridge(alpha=1.0, random_state=self.config.random_state),
            'Lasso': Lasso(alpha=0.1, random_state=self.config.random_state),
            'SVR': SVR(C=1.0, epsilon=0.1, kernel='rbf'),
            'NeuralNetwork': MLPRegressor(
                hidden_layer_sizes=(100, 50, 25),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=1000,
                random_state=self.config.random_state
            )
        }

    def initialize_models(self):
        """
        INITIALIZE MULTIPLE ML ALGORITHMS
        ================================
        
        Creates ensemble of different ML algorithms, each capturing different aspects:
        - Linear models: Linear relationships
        - Tree models: Non-linear patterns and interactions
        - SVM: Complex decision boundaries
        - Neural networks: Complex non-linear relationships
        """
        
        print("\n" + "="*50)
        print("INITIALIZING ML MODELS")
        print("="*50)
        
        self.models = {
            # LINEAR MODELS - Good for understanding linear relationships
            'linear_regression': LinearRegression(),
            'ridge_regression': Ridge(alpha=1.0),  # L2 regularization
            'lasso_regression': Lasso(alpha=0.1),  # L1 regularization (feature selection)
            
            # TREE-BASED MODELS - Excellent for non-linear patterns
            'random_forest': RandomForestRegressor(
                n_estimators=100,      # Number of trees
                max_depth=10,          # Maximum tree depth
                min_samples_split=5,   # Minimum samples to split
                min_samples_leaf=2,    # Minimum samples in leaf
                random_state=42,       # For reproducibility
                n_jobs=-1             # Use all CPU cores
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,      # Number of boosting stages
                learning_rate=0.1,     # Learning rate
                max_depth=6,           # Maximum tree depth
                min_samples_split=5,   # Minimum samples to split
                random_state=42        # For reproducibility
            ),
            'xgboost': xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                min_child_weight=1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ),
            
            # SUPPORT VECTOR MACHINE - Good for complex boundaries
            'svr': SVR(
                kernel='rbf',          # Radial basis function kernel
                C=1.0,                 # Regularization parameter
                gamma='scale',         # Kernel coefficient
                epsilon=0.1            # Epsilon-tube for SVR
            ),
            
            # NEURAL NETWORK - For complex non-linear relationships
            'neural_network': MLPRegressor(
                hidden_layer_sizes=(100, 50, 25),  # 3 hidden layers
                activation='relu',                  # ReLU activation
                solver='adam',                      # Adam optimizer
                alpha=0.001,                        # L2 regularization
                learning_rate='adaptive',           # Adaptive learning rate
                max_iter=1000,                      # Maximum iterations
                early_stopping=True,                # Stop early if no improvement
                validation_fraction=0.1,            # Validation set size
                random_state=42                     # For reproducibility
            )
    }
        
        print(f"✓ Initialized {len(self.models)} ML models:")
        for name in self.models.keys():
            print(f"  - {name}")
        
    def train_and_evaluate_models(self, X_train, X_test, y_train, y_test):
        """
        TRAIN ALL MODELS AND EVALUATE PERFORMANCE
        ========================================
        
        For each model:
        1. Train on training data
        2. Make predictions on both train and test sets
        3. Calculate multiple performance metrics
        4. Calculate error percentages
        5. Store all results for comparison
        
        Args:
            X_train, X_test: Feature matrices for train/test
            y_train, y_test: Target vectors for train/test
        """
        
        print("\n" + "="*50)
        print("TRAINING AND EVALUATING ALL MODELS")
        print("="*50)
        
        for name, model in self.models.items():
            print(f"\n🔄 Training {name.upper()}...")
            
            try:
                # TRAIN THE MODEL
                model.fit(X_train, y_train)
                print(f"  ✓ Training completed")
                
                # MAKE PREDICTIONS
                y_pred_train = model.predict(X_train)  # Training predictions
                y_pred_test = model.predict(X_test)    # Test predictions
                
                # CALCULATE PERFORMANCE METRICS
                # Mean Absolute Error
                train_mae = mean_absolute_error(y_train, y_pred_train)
                test_mae = mean_absolute_error(y_test, y_pred_test)
                
                # Root Mean Square Error
                train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                
                # R² Score (coefficient of determination)
                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)
                
                # CALCULATE ERROR PERCENTAGES
                # Average percentage error on test set
                test_error_pct = np.mean(np.abs((y_test - y_pred_test) / y_test) * 100)
                train_error_pct = np.mean(np.abs((y_train - y_pred_train) / y_train) * 100)
                
                # STORE ALL PERFORMANCE METRICS
                self.model_performance[name] = {
                    'train_mae': train_mae,
                    'test_mae': test_mae,
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'train_error_pct': train_error_pct,
                    'test_error_pct': test_error_pct,
                    'predictions_train': y_pred_train,
                    'predictions_test': y_pred_test
                }
                
                # Store error percentage separately for easy access
                self.test_error_percentages[name] = test_error_pct
                
                # DISPLAY PERFORMANCE SUMMARY
                print(f"  📊 Performance Metrics:")
                print(f"    - Test MAE:     {test_mae:.4f}")
                print(f"    - Test RMSE:    {test_rmse:.4f}")
                print(f"    - Test R²:      {test_r2:.4f}")
                print(f"    - Test Error %: {test_error_pct:.2f}%")
                
                # Check for overfitting
                r2_diff = train_r2 - test_r2
                if r2_diff > 0.1:
                    print(f"    ⚠️  Potential overfitting detected (R² diff: {r2_diff:.3f})")
                
            except Exception as e:
                print(f"  ❌ Error training {name}: {e}")
                # Store failed model info
                self.model_performance[name] = {
                    'error': str(e),
                    'test_r2': -999,  # Very low score to exclude from best model selection
                    'test_error_pct': 999
                }
        
        print(f"\n✓ Model training completed for {len(self.models)} models")
    
    def select_best_model(self):
        """
        SELECT BEST PERFORMING MODEL
        ===========================
        
        Selects the model with highest test R² score (best predictive performance).
        
        Returns:
            tuple: (best_model_name, best_model_object)
        """
        
        print("\n" + "="*50)
        print("SELECTING BEST MODEL")
        print("="*50)
        
        best_score = -np.inf
        best_name = None
        
        # Find model with highest test R² score
        for name, performance in self.model_performance.items():
            if 'test_r2' in performance and performance['test_r2'] > best_score:
                best_score = performance['test_r2']
                best_name = name
        
        if best_name is None:
            print("❌ No valid models found!")
            return None, None
        
        self.best_model = self.models[best_name]
        
        print(f"🏆 BEST MODEL SELECTED: {best_name.upper()}")
        print(f"  - Test R² Score: {best_score:.4f}")
        print(f"  - Test Error %:  {self.test_error_percentages[best_name]:.2f}%")
        
        # Display top 3 models for comparison
        sorted_models = sorted(
            [(name, perf['test_r2']) for name, perf in self.model_performance.items() 
             if 'test_r2' in perf and perf['test_r2'] > -999],
            key=lambda x: x[1], reverse=True
        )
        
        print(f"\n📊 TOP 3 MODELS:")
        for i, (name, score) in enumerate(sorted_models[:3], 1):
            error_pct = self.test_error_percentages.get(name, 'N/A')
            print(f"  {i}. {name}: R²={score:.4f}, Error={error_pct:.2f}%")
        
        return best_name, self.best_model
    
    def extract_feature_importance(self, feature_names):
        """
        EXTRACT FEATURE IMPORTANCE FROM TREE-BASED MODELS
        ================================================
        
        Analyzes which features are most important for predictions.
        Only works with tree-based models that have feature_importances_ attribute.
        
        Args:
            feature_names (list): Names of features used in training
        """
        
        print("\n" + "="*50)
        print("EXTRACTING FEATURE IMPORTANCE")
        print("="*50)
        
        tree_models = ['random_forest', 'gradient_boosting', 'xgboost']
        
        for name in tree_models:
            if name in self.models and name in self.model_performance:
                if hasattr(self.models[name], 'feature_importances_'):
                    importance = self.models[name].feature_importances_
                    self.feature_importance[name] = dict(zip(feature_names, importance))
                    
                    # Display top 5 most important features
                    sorted_features = sorted(
                        self.feature_importance[name].items(),
                        key=lambda x: x[1], reverse=True
                    )
                    
                    print(f"\n🔍 {name.upper()} - Top 5 Important Features:")
                    for i, (feature, importance) in enumerate(sorted_features[:5], 1):
                        print(f"  {i}. {feature}: {importance:.4f}")
        
        if not self.feature_importance:
            print("ℹ️  No tree-based models available for feature importance analysis")
    
    def create_ensemble_prediction(self, X):
        """
        CREATE ENSEMBLE PREDICTION
        =========================
        
        Combines predictions from top-performing models using weighted averaging.
        Weights are based on model performance (R² scores).
        
        Args:
            X: Feature matrix for prediction
            
        Returns:
            array: Ensemble predictions
        """
        
        print("\n" + "="*50)
        print("CREATING ENSEMBLE PREDICTION")
        print("="*50)
        
        # Select top 3 models based on test R² score
        valid_models = [(name, perf) for name, perf in self.model_performance.items() 
                       if 'test_r2' in perf and perf['test_r2'] > -999]
        
        if len(valid_models) == 0:
            print("❌ No valid models for ensemble!")
            return None
        
        sorted_models = sorted(valid_models, key=lambda x: x[1]['test_r2'], reverse=True)
        top_models = sorted_models[:min(3, len(sorted_models))]  # Top 3 or all available
        
        predictions = []
        weights = []
        model_names = []
        
        # Collect predictions and weights from top models
        for name, performance in top_models:
            try:
                pred = self.models[name].predict(X)
                predictions.append(pred)
                weights.append(max(0, performance['test_r2']))  # Ensure non-negative weights
                model_names.append(name)
            except Exception as e:
                print(f"⚠️  Skipping {name} in ensemble due to error: {e}")
        
        if len(predictions) == 0:
            print("❌ No valid predictions for ensemble!")
            return None
        
        # Normalize weights
        weights = np.array(weights)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(len(weights)) / len(weights)  # Equal weights if all R² are 0 or negative
        
        # Create weighted average ensemble
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
        
        print(f"🎯 ENSEMBLE CREATED:")
        print(f"  - Models used: {model_names}")
        print(f"  - Weights: {[f'{w:.3f}' for w in weights]}")
        print(f"  - Prediction shape: {ensemble_pred.shape}")
        
        return ensemble_pred
    
    def calculate_corrected_error_percentage(self, y_true, y_pred_correction, original_0d_temps, original_3d_temps):
        """
        CALCULATE ERROR PERCENTAGE AFTER CORRECTION
        ==========================================
        
        Compares the corrected 0D temperatures with actual 3D temperatures
        to show improvement from ML correction.
        
        Args:
            y_true: True correction factors
            y_pred_correction: Predicted correction factors
            original_0d_temps: Original 0D temperatures
            original_3d_temps: Original 3D temperatures
            
        Returns:
            float: Error percentage after correction
        """
        
        # Calculate corrected 0D temperatures using predicted correction factors
        corrected_0d_temps = original_0d_temps * y_pred_correction
        
        # Calculate error percentage between corrected 0D and actual 3D
        absolute_error = np.abs(original_3d_temps - corrected_0d_temps)
        error_percentage = np.mean(absolute_error / original_3d_temps * 100)
        
        return error_percentage

# ============================================================================
# VISUALIZATION CLASS
# ============================================================================

class ThermalRunawayVisualizer:
    """
    COMPREHENSIVE VISUALIZATION SUITE
    ================================
    
    Creates detailed visualizations for:
    1. Model performance comparison
    2. Prediction accuracy analysis
    3. Feature importance plots
    4. Correction factor analysis
    5. Error percentage comparisons
    """
    
    def __init__(self, ml_pipeline, data_processor):
        """
        Initialize visualizer with ML pipeline and data processor
        
        Args:
            ml_pipeline: Trained ML pipeline object
            data_processor: Data processor object with processed data
        """
        self.ml_pipeline = ml_pipeline
        self.data_processor = data_processor
        
        print("✓ Visualizer initialized successfully")
    
    def plot_model_performance_comparison(self):
        """
        COMPREHENSIVE MODEL PERFORMANCE COMPARISON
        ========================================
        
        Creates 4 subplots comparing all models across different metrics:
        1. Mean Absolute Error (MAE)
        2. Root Mean Square Error (RMSE)  
        3. R² Score
        4. Error Percentages
        """
        
        print("\n" + "="*50)
        print("CREATING MODEL PERFORMANCE COMPARISON")
        print("="*50)
        
        # Filter out failed models
        valid_models = {name: perf for name, perf in self.ml_pipeline.model_performance.items() 
                       if 'test_r2' in perf and perf['test_r2'] > -999}
        
        if not valid_models:
            print("❌ No valid models to visualize!")
            return
        
        models = list(valid_models.keys())
        
        # Create figure with 4 subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Performance Comparison - Thermal Runaway ML Pipeline', 
                     fontsize=16, fontweight='bold')
        
        # PLOT 1: Test MAE Comparison
        test_mae = [valid_models[model]['test_mae'] for model in models]
        bars1 = axes[0,0].bar(models, test_mae, color='skyblue', alpha=0.8, edgecolor='navy')
        axes[0,0].set_title('Test Mean Absolute Error (MAE)', fontweight='bold')
        axes[0,0].set_ylabel('MAE')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars1, test_mae):
            axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(test_mae)*0.01,
                          f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # PLOT 2: Test RMSE Comparison
        test_rmse = [valid_models[model]['test_rmse'] for model in models]
        bars2 = axes[0,1].bar(models, test_rmse, color='lightcoral', alpha=0.8, edgecolor='darkred')
        axes[0,1].set_title('Test Root Mean Square Error (RMSE)', fontweight='bold')
        axes[0,1].set_ylabel('RMSE')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars2, test_rmse):
            axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(test_rmse)*0.01,
                          f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # PLOT 3: Test R² Comparison
        test_r2 = [valid_models[model]['test_r2'] for model in models]
        bars3 = axes[1,0].bar(models, test_r2, color='lightgreen', alpha=0.8, edgecolor='darkgreen')
        axes[1,0].set_title('Test R² Score (Higher is Better)', fontweight='bold')
        axes[1,0].set_ylabel('R² Score')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].set_ylim(0, 1)  # R² typically ranges from 0 to 1
        
        # Add value labels on bars
        for bar, value in zip(bars3, test_r2):
            axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                          f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # PLOT 4: Error Percentage Comparison
        test_error_pct = [valid_models[model]['test_error_pct'] for model in models]
        bars4 = axes[1,1].bar(models, test_error_pct, color='gold', alpha=0.8, edgecolor='orange')
        axes[1,1].set_title('Test Error Percentage (Lower is Better)', fontweight='bold')
        axes[1,1].set_ylabel('Error Percentage (%)')
        axes[1,1].tick_params(axis='x', rotation=45)
        axes[1,1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars4, test_error_pct):
            axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(test_error_pct)*0.01,
                          f'{value:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        print("✓ Model performance comparison visualization completed")
    
    def plot_prediction_accuracy(self, X_test, y_test, best_model_name):
        """
        DETAILED PREDICTION ACCURACY ANALYSIS
        ====================================
        
        Creates 3 plots for the best model:
        1. Actual vs Predicted scatter plot
        2. Residuals plot (error analysis)
        3. Residuals distribution histogram
        
        Args:
            X_test: Test feature matrix
            y_test: Test target values
            best_model_name: Name of the best performing model
        """
        
        print(f"\n" + "="*50)
        print(f"ANALYZING PREDICTION ACCURACY - {best_model_name.upper()}")
        print("="*50)
        
        if best_model_name not in self.ml_pipeline.model_performance:
            print(f"❌ Model {best_model_name} not found in performance data!")
            return
        
        # Get predictions from best model
        y_pred = self.ml_pipeline.model_performance[best_model_name]['predictions_test']
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'Prediction Accuracy Analysis - {best_model_name.upper()}', 
                     fontsize=16, fontweight='bold')
        
        # PLOT 1: Actual vs Predicted Scatter Plot
        axes[0].scatter(y_test, y_pred, alpha=0.6, color='blue', s=50)
        
        # Add perfect prediction line (y = x)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        axes[0].set_xlabel('Actual Correction Factor')
        axes[0].set_ylabel('Predicted Correction Factor')
        axes[0].set_title('Actual vs Predicted Values')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Add correlation coefficient
        correlation = np.corrcoef(y_test, y_pred)[0, 1]
        axes[0].text(0.05, 0.95, f'Correlation: {correlation:.4f}', 
                    transform=axes[0].transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow"))
        
        # PLOT 2: Residuals Plot
        residuals = y_test - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.6, color='green', s=50)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2, label='Zero Error')
        axes[1].set_xlabel('Predicted Correction Factor')
        axes[1].set_ylabel('Residuals (Actual - Predicted)')
        axes[1].set_title('Residuals Plot')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        # Add residual statistics
        mean_residual = residuals.mean()
        std_residual = residuals.std()
        axes[1].text(0.05, 0.95, f'Mean: {mean_residual:.4f}\nStd: {std_residual:.4f}', 
                    transform=axes[1].transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        # PLOT 3: Residuals Distribution Histogram
        axes[2].hist(residuals, bins=20, alpha=0.7, color='orange', edgecolor='black')
        axes[2].axvline(x=0, color='r', linestyle='--', lw=2, label='Zero Error')
        axes[2].axvline(x=mean_residual, color='blue', linestyle='-', lw=2, label=f'Mean: {mean_residual:.4f}')
        axes[2].set_xlabel('Residuals')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('Residuals Distribution')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        
        plt.tight_layout()
        plt.show()
        
        # PRINT DETAILED PERFORMANCE STATISTICS
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        error_pct = np.mean(np.abs((y_test - y_pred) / y_test) * 100)
        
        print(f"\n📊 DETAILED PERFORMANCE SUMMARY:")
        print(f"  - Mean Absolute Error (MAE):     {mae:.6f}")
        print(f"  - Root Mean Square Error (RMSE): {rmse:.6f}")
        print(f"  - R² Score:                      {r2:.6f}")
        print(f"  - Average Error Percentage:      {error_pct:.2f}%")
        print(f"  - Mean Residual:                 {mean_residual:.6f}")
        print(f"  - Std Residual:                  {std_residual:.6f}")
        print(f"  - Correlation Coefficient:       {correlation:.6f}")
        
        print("✓ Prediction accuracy analysis completed")
    
    def plot_feature_importance(self):
        """
        FEATURE IMPORTANCE VISUALIZATION
        ===============================
        
        Creates horizontal bar plots showing which features are most important
        for each tree-based model's predictions.
        """
        
        print("\n" + "="*50)
        print("CREATING FEATURE IMPORTANCE PLOTS")
        print("="*50)
        
        if not self.ml_pipeline.feature_importance:
            print("ℹ️  No feature importance data available (no tree-based models)")
            return
        
        n_models = len(self.ml_pipeline.feature_importance)
        fig, axes = plt.subplots(1, n_models, figsize=(8*n_models, 8))
        
        if n_models == 1:
            axes = [axes]
        
        fig.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold')
        
        for idx, (model_name, importance_dict) in enumerate(self.ml_pipeline.feature_importance.items()):
            features = list(importance_dict.keys())
            importance = list(importance_dict.values())
            
            # Sort by importance (descending)
            sorted_idx = np.argsort(importance)[::-1]
            features = [features[i] for i in sorted_idx]
            importance = [importance[i] for i in sorted_idx]
            
            # Create horizontal bar plot
            y_pos = np.arange(len(features))
            bars = axes[idx].barh(y_pos, importance, alpha=0.8, 
                                 color=plt.cm.viridis(np.linspace(0, 1, len(features))))
            
            axes[idx].set_yticks(y_pos)
            axes[idx].set_yticklabels(features)
            axes[idx].set_xlabel('Importance Score')
            axes[idx].set_title(f'{model_name.upper()}\nFeature Importance', fontweight='bold')
            axes[idx].grid(True, alpha=0.3, axis='x')
            
            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars, importance)):
                axes[idx].text(bar.get_width() + max(importance)*0.01, bar.get_y() + bar.get_height()/2,
                              f'{value:.4f}', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        print("✓ Feature importance visualization completed")
    
    def plot_correction_factor_analysis(self):
        """
        COMPREHENSIVE CORRECTION FACTOR ANALYSIS
        =======================================
        
        Creates 4 detailed plots analyzing correction factors:
        1. Correction factor evolution over time
        2. Correction factor vs 0D temperature (colored by time)
        3. 0D vs 3D temperature comparison
        4. Correction factor distribution with statistics
        """
        
        print("\n" + "="*50)
        print("CREATING CORRECTION FACTOR ANALYSIS")
        print("="*50)
        
        data = self.data_processor.aligned_data
        
        # Create figure with 4 subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Correction Factor Analysis - Thermal Runaway Study', 
                     fontsize=16, fontweight='bold')
        
        # PLOT 1: Correction Factor vs Time
        axes[0,0].plot(data['time'], data['correction_factor'], 'b-', linewidth=2, alpha=0.8)
        axes[0,0].set_xlabel('Time')
        axes[0,0].set_ylabel('Correction Factor (3D/0D)')
        axes[0,0].set_title('Correction Factor Evolution Over Time')
        axes[0,0].grid(True, alpha=0.3)
        
        # Add horizontal line at correction factor = 1 (perfect match)
        axes[0,0].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Perfect Match (CF=1)')
        axes[0,0].legend()
        
        # Add statistics text box
        cf_mean = data['correction_factor'].mean()
        cf_std = data['correction_factor'].std()
        axes[0,0].text(0.02, 0.98, f'Mean: {cf_mean:.3f}\nStd: {cf_std:.3f}', 
                      transform=axes[0,0].transAxes, va='top',
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        # PLOT 2: Correction Factor vs 0D Temperature (colored by time)
        scatter = axes[0,1].scatter(data['temperature_0d'], data['correction_factor'], 
                                   c=data['time'], cmap='viridis', alpha=0.7, s=50)
        axes[0,1].set_xlabel('0D Temperature')
        axes[0,1].set_ylabel('Correction Factor (3D/0D)')
        axes[0,1].set_title('Correction Factor vs 0D Temperature')
        axes[0,1].grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=axes[0,1])
        cbar.set_label('Time', rotation=270, labelpad=20)
        
        # Add thermal runaway zone indicators
        axes[0,1].axvline(x=60, color='orange', linestyle='--', alpha=0.7, label='Warning (60°C)')
        axes[0,1].axvline(x=120, color='red', linestyle='--', alpha=0.7, label='Critical (120°C)')
        axes[0,1].legend()
        
        # PLOT 3: Temperature Comparison (0D vs 3D)
        axes[1,0].plot(data['time'], data['temperature_0d'], 'b-', label='0D Model', linewidth=2)
        axes[1,0].plot(data['time'], data['temperature_3d'], 'r-', label='3D Model', linewidth=2)
        axes[1,0].set_xlabel('Time')
        axes[1,0].set_ylabel('Temperature')
        axes[1,0].set_title('0D vs 3D Temperature Comparison')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Add thermal runaway zones
        axes[1,0].axhline(y=60, color='orange', linestyle='--', alpha=0.5, label='Warning Zone')
        axes[1,0].axhline(y=120, color='red', linestyle='--', alpha=0.5, label='Critical Zone')
        
        # Calculate and display initial error percentage
        initial_error = self.data_processor.initial_error_percentage
        axes[1,0].text(0.02, 0.98, f'Initial Error: {initial_error:.2f}%', 
                      transform=axes[1,0].transAxes, va='top',
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow"))
        
        # PLOT 4: Correction Factor Distribution
        n, bins, patches = axes[1,1].hist(data['correction_factor'], bins=30, alpha=0.7, 
                                         color='purple', edgecolor='black', density=True)
        
        # Add statistical lines
        axes[1,1].axvline(cf_mean, color='red', linestyle='-', linewidth=2, 
                         label=f'Mean: {cf_mean:.3f}')
        axes[1,1].axvline(cf_mean + cf_std, color='orange', linestyle='--', 
                         label=f'+1 Std: {cf_mean + cf_std:.3f}')
        axes[1,1].axvline(cf_mean - cf_std, color='orange', linestyle='--', 
                         label=f'-1 Std: {cf_mean - cf_std:.3f}')
        axes[1,1].axvline(1.0, color='green', linestyle=':', linewidth=2, 
                         label='Perfect Match (1.0)')
        
        axes[1,1].set_xlabel('Correction Factor')
        axes[1,1].set_ylabel('Density')
        axes[1,1].set_title('Correction Factor Distribution')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        # Add distribution statistics
        cf_median = data['correction_factor'].median()
        cf_min = data['correction_factor'].min()
        cf_max = data['correction_factor'].max()
        
        stats_text = f'Statistics:\nMean: {cf_mean:.3f}\nMedian: {cf_median:.3f}\nStd: {cf_std:.3f}\nMin: {cf_min:.3f}\nMax: {cf_max:.3f}'
        axes[1,1].text(0.98, 0.98, stats_text, transform=axes[1,1].transAxes, 
                      va='top', ha='right', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        
        plt.tight_layout()
        plt.show()
        
        print("✓ Correction factor analysis visualization completed")
    
    def plot_error_percentage_comparison(self, initial_error, corrected_error_dict):
        """
        ERROR PERCENTAGE COMPARISON VISUALIZATION
        ========================================
        
        Compares initial error (0D vs 3D) with corrected errors from different models.
        
        Args:
            initial_error (float): Initial error percentage between 0D and 3D
            corrected_error_dict (dict): Dictionary of model names and their corrected error percentages
        """
        
        print("\n" + "="*50)
        print("CREATING ERROR PERCENTAGE COMPARISON")
        print("="*50)
        
        # Prepare data for plotting
        models = ['Initial (0D vs 3D)'] + list(corrected_error_dict.keys())
        errors = [initial_error] + list(corrected_error_dict.values())
        colors = ['red'] + ['green' if err < initial_error else 'orange' for err in corrected_error_dict.values()]
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.bar(models, errors, color=colors, alpha=0.7, edgecolor='black')

        # Customize plot
        ax.set_title('Error Percentage Comparison: Before vs After ML Correction', 
                     fontsize=16, fontweight='bold')
        ax.set_ylabel('Error Percentage (%)', fontsize=12)
        ax.set_xlabel('Models', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, errors):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(errors)*0.01,
                   f'{value:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        # Add improvement annotations
        for i, (model, error) in enumerate(zip(list(corrected_error_dict.keys()), list(corrected_error_dict.values()))):
            improvement = initial_error - error
            improvement_pct = (improvement / initial_error) * 100 if initial_error > 0 else 0
            if improvement > 0:
                ax.text(i+1, error/2, f'↓{improvement_pct:.1f}%\nimprovement', 
                       ha='center', va='center', fontweight='bold', color='white',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="darkgreen", alpha=0.8))
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        
        print("✓ Error percentage comparison visualization completed")

def print_correction_factors_for_all_points(xlsx_0d_path, xlsx_3d_paths):
    """
    PRINTS THE CORRECTION FACTOR STATISTICS FOR EACH POINT (A, B, C, D)
    ===================================================================
    For each point, loads the data, aligns, and prints mean, min, max correction factor.
    Args:
        xlsx_0d_path (str): Path to the 0D Excel file.
        xlsx_3d_paths (list): List of paths to the 3D Excel files for points A, B, C, D.
    """
    points = ['A', 'B', 'C', 'D']
    print("="*60)
    print("CORRECTION FACTOR STATISTICS FOR EACH POINT")
    print("="*60)
    for idx, point in enumerate(points):
        print(f"\n--- Point {point} ---")
        try:
            processor = ThermalRunawayDataProcessor()
            # Load only the relevant 0D and 3D data for this point
            data_0d, data_3d = processor.load_data(xlsx_0d_path, [xlsx_3d_paths[idx]])
            # Filter for this point
            data_0d = data_0d[data_0d['point'] == point].reset_index(drop=True)
            data_3d = data_3d[data_3d['point'] == point].reset_index(drop=True)
            processor.data_0d = data_0d
            processor.data_3d = data_3d
            processor.align_data_by_interpolation()
            processor.calculate_correction_factor()
            cf = processor.aligned_data['correction_factor']
            print(f"Correction Factor Stats for Point {point}:")
            print(f"  Mean: {cf.mean():.4f}")
            print(f"  Min:  {cf.min():.4f}")
            print(f"  Max:  {cf.max():.4f}")
        except Exception as e:
            print(f"  Error processing point {point}: {e}")

# ============================================================================
# MAIN EXECUTION PIPELINE
# ============================================================================

def main_thermal_runaway_pipeline(xlsx_0d_path, xlsx_3d_paths):
    """
    MAIN EXECUTION FUNCTION - COMPLETE THERMAL RUNAWAY ML PIPELINE
    ==============================================================
    
    This function orchestrates the entire machine learning pipeline:
    1. Data loading and preprocessing
    2. Feature engineering
    3. Model training and evaluation
    4. Ensemble creation
    5. Comprehensive visualization
    6. Results saving
    
    Args:
        xlsx_0d_path (str): Path to your 0D Excel file (all points)
        xlsx_3d_paths (list): List of paths to your  3D Excel files (A, B, C, D)
    
    Returns:
        dict: Complete results including models, performance metrics, and predictions
    """
    
    print("="*80)
    print("🚀 THERMAL RUNAWAY ML CORRECTION PIPELINE STARTING")
    print("="*80)
    print("This pipeline will:")
    print("1. Load and process your CSV data")
    print("2. Engineer features for ML models")
    print("3. Train multiple ML algorithms")
    print("4. Create ensemble predictions")
    print("5. Generate comprehensive visualizations")
    print("6. Calculate error improvements")
    print("="*80)
    
    # ========================================================================
    # STEP 1: INITIALIZE DATA PROCESSOR
    # ========================================================================
    print("\n🔧 STEP 1: INITIALIZING DATA PROCESSOR")
    data_processor = ThermalRunawayDataProcessor()
    
    # ========================================================================
    # STEP 2: LOAD AND VISUALIZE RAW DATA
    # ========================================================================
    print("\n📂 STEP 2: LOADING AND VISUALIZING RAW DATA")
    try:
        data_0d, data_3d = data_processor.load_data(xlsx_0d_path, xlsx_3d_paths)
        data_processor.visualize_raw_data()
    except Exception as e:
        print(f"❌ Error in data loading: {e}")
        print("Please check your data file paths and format!")
        return None
    
    # ========================================================================
    # STEP 3: ALIGN DATA AND CALCULATE CORRECTION FACTORS
    # ========================================================================
    print("\n🔄 STEP 3: ALIGNING DATA AND CALCULATING CORRECTION FACTORS")
    try:
        aligned_data = data_processor.align_data_by_interpolation()
        corrected_data = data_processor.calculate_correction_factor()
    except Exception as e:
        print(f"❌ Error in data alignment: {e}")
        return None
    
    # ========================================================================
    # STEP 4: ENGINEER FEATURES FOR ML
    # ========================================================================
    print("\n⚙️ STEP 4: ENGINEERING FEATURES FOR MACHINE LEARNING")
    try:
        final_data = data_processor.engineer_features()
    except Exception as e:
        print(f"❌ Error in feature engineering: {e}")
        return None
    
    # ========================================================================
    # STEP 5: PREPARE ML PIPELINE
    # ========================================================================
    print("\n🤖 STEP 5: PREPARING MACHINE LEARNING PIPELINE")
    ml_pipeline = ThermalRunawayMLPipeline()
    
    # Prepare features and target variables
    try:
        X, y, feature_names = ml_pipeline.prepare_features_target(final_data)
    except Exception as e:
        print(f"❌ Error in feature preparation: {e}")
        return None
    
    # Split data for training and testing (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    # Add these lines:
    test_indices = X_test.index
    original_0d_temps = aligned_data.loc[test_indices, 'temperature_0d']
    original_3d_temps = aligned_data.loc[test_indices, 'temperature_3d']
    
    # Scale features for better ML performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame for easier handling
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)
    
    print(f"✅ Data preparation completed:")
    print(f"  - Training set size: {X_train_scaled.shape}")
    print(f"  - Test set size: {X_test_scaled.shape}")
    print(f"  - Number of features: {len(feature_names)}")
    
    # ========================================================================
    # STEP 6: TRAIN MACHINE LEARNING MODELS
    # ========================================================================
    print("\n🏋️ STEP 6: TRAINING MACHINE LEARNING MODELS")
    try:
        ml_pipeline.initialize_models()
        ml_pipeline.train_and_evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test)
    except Exception as e:
        print(f"❌ Error in model training: {e}")
        return None
    
    # ========================================================================
    # STEP 7: SELECT BEST MODEL AND EXTRACT INSIGHTS
    # ========================================================================
    print("\n🏆 STEP 7: MODEL SELECTION AND ANALYSIS")
    try:
        best_model_name, best_model = ml_pipeline.select_best_model()
        ml_pipeline.extract_feature_importance(feature_names)
    except Exception as e:
        print(f"❌ Error in model selection: {e}")
        return None
    
    # ========================================================================
    # STEP 8: CREATE ENSEMBLE PREDICTION
    # ========================================================================
    print("\n🎯 STEP 8: CREATING ENSEMBLE PREDICTION")
    try:
        ensemble_pred = ml_pipeline.create_ensemble_prediction(X_test_scaled)
        
        if ensemble_pred is not None:
            ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
            ensemble_r2 = r2_score(y_test, ensemble_pred)
            ensemble_error_pct = np.mean(np.abs((y_test - ensemble_pred) / y_test) * 100)
            
            print(f"📊 ENSEMBLE PERFORMANCE:")
            print(f"  - Ensemble MAE: {ensemble_mae:.4f}")
            print(f"  - Ensemble R²: {ensemble_r2:.4f}")
            print(f"  - Ensemble Error %: {ensemble_error_pct:.2f}%")
    except Exception as e:
        print(f"❌ Error in ensemble creation: {e}")
        ensemble_pred = None
        ensemble_mae = None
        ensemble_r2 = None
        ensemble_error_pct = None
    
    # ========================================================================
    # STEP 9: CALCULATE ERROR IMPROVEMENTS
    # ========================================================================
    print("\n📈 STEP 9: CALCULATING ERROR IMPROVEMENTS")
    
    # Get test data indices for error calculation
    test_indices = X_test.index
    original_0d_temps = final_data.loc[test_indices, 'temperature_0d']
    original_3d_temps = final_data.loc[test_indices, 'temperature_3d']
    
    # Calculate corrected error percentages for each model
    corrected_errors = {}
    
    for model_name, performance in ml_pipeline.model_performance.items():
        if 'predictions_test' in performance:
            try:
                y_pred = performance['predictions_test']
                corrected_error = ml_pipeline.calculate_corrected_error_percentage(
                    y_test, y_pred, original_0d_temps, original_3d_temps
                )
                corrected_errors[model_name] = corrected_error
            except:
                continue
    
    # Add ensemble corrected error if available
    if ensemble_pred is not None:
        ensemble_corrected_error = ml_pipeline.calculate_corrected_error_percentage(
            y_test, ensemble_pred, original_0d_temps, original_3d_temps
        )
        corrected_errors['Ensemble'] = ensemble_corrected_error
    
    # Display error improvement summary
    initial_error = data_processor.initial_error_percentage
    print(f"\n🎉 ERROR IMPROVEMENT SUMMARY:")
    print(f"  - Initial 0D vs 3D Error: {initial_error:.2f}%")
    
    for model_name, corrected_error in corrected_errors.items():
        improvement = initial_error - corrected_error
        improvement_pct = (improvement / initial_error) * 100 if initial_error > 0 else 0
        print(f"  - {model_name} Corrected Error: {corrected_error:.2f}% (↓{improvement_pct:.1f}% improvement)")
    
    # ========================================================================
    # STEP 10: COMPREHENSIVE VISUALIZATION
    # ========================================================================
    print("\n📊 STEP 10: GENERATING COMPREHENSIVE VISUALIZATIONS")
    
    try:
        visualizer = ThermalRunawayVisualizer(ml_pipeline, data_processor)
        
        # Plot model performance comparison
        print("  Creating model performance comparison...")
        visualizer.plot_model_performance_comparison()
        
        # Plot prediction accuracy for best model
        if best_model_name:
            print(f"  Creating prediction accuracy analysis for {best_model_name}...")
            visualizer.plot_prediction_accuracy(X_test_scaled, y_test, best_model_name)
        
        # Plot feature importance
        print("  Creating feature importance plots...")
        visualizer.plot_feature_importance()
        
        # Plot correction factor analysis
        print("  Creating correction factor analysis...")
        visualizer.plot_correction_factor_analysis()
        
        # Plot error percentage comparison
        print("  Creating error percentage comparison...")
        visualizer.plot_error_percentage_comparison(initial_error, corrected_errors)
        
    except Exception as e:
        print(f"⚠️ Warning: Some visualizations failed: {e}")
    
    # ========================================================================
    # STEP 11: SAVE RESULTS AND MODELS
    # ========================================================================
    print("\n💾 STEP 11: SAVING RESULTS AND MODELS")
    
    try:
        # Save the best model and scaler
        if best_model:
            joblib.dump(best_model, f'best_thermal_model_{best_model_name}.pkl')
            print(f"✅ Best model saved: best_thermal_model_{best_model_name}.pkl")
        
        joblib.dump(scaler, 'feature_scaler.pkl')
        print("✅ Feature scaler saved: feature_scaler.pkl")
        
        # Save performance summary
        performance_summary = pd.DataFrame(ml_pipeline.model_performance).T
        performance_summary.to_csv('model_performance_summary.csv')
        print("✅ Performance summary saved: model_performance_summary.csv")
        
        # Save predictions and results
        results_df = pd.DataFrame({
            'actual_correction_factor': y_test,
            'predicted_best_model': ml_pipeline.model_performance[best_model_name]['predictions_test'] if best_model_name else None,
            'predicted_ensemble': ensemble_pred if ensemble_pred is not None else None,
        })
        
        # Add original temperatures for reference
        original_0d_temps = final_data['temperature_0d']
        original_3d_temps = final_data['temperature_3d']
        results_df['original_0d_temp'] = original_0d_temps.values
        results_df['original_3d_temp'] = original_3d_temps.values
        
        # Calculate corrected temperatures
        if best_model_name:
            results_df['corrected_0d_temp_best'] = original_0d_temps.values * results_df['predicted_best_model']
        
        if ensemble_pred is not None:
            results_df['corrected_0d_temp_ensemble'] = original_0d_temps.values * results_df['predicted_ensemble']
        
        results_df.to_csv('prediction_results.csv', index=False)
        print("✅ Prediction results saved: prediction_results.csv")
        
        # Save error improvement summary
        error_summary = pd.DataFrame({
            'Model': ['Initial'] + list(corrected_errors.keys()),
            'Error_Percentage': [initial_error] + list(corrected_errors.values())
        })
        error_summary['Improvement_Percentage'] = error_summary['Error_Percentage'].apply(
            lambda x: 0 if x == initial_error else ((initial_error - x) / initial_error * 100)
        )
        error_summary.to_csv('error_improvement_summary.csv', index=False)
        print("✅ Error improvement summary saved: error_improvement_summary.csv")
        
    except Exception as e:
        print(f"⚠️ Warning: Some files could not be saved: {e}")
    
    # ========================================================================
    # STEP 12: FINAL SUMMARY AND RECOMMENDATIONS
    # ========================================================================
    print("\n" + "="*80)
    print("🎉 THERMAL RUNAWAY ML PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    print(f"\n📋 FINAL SUMMARY:")
    print(f"  🏆 Best Model: {best_model_name}")
    if best_model_name:
        best_performance = ml_pipeline.model_performance[best_model_name]
        print(f"  📊 Best Model Performance:")
        print(f"    - R² Score: {best_performance['test_r2']:.4f}")
        print(f"    - MAE: {best_performance['test_mae']:.4f}")
        print(f"    - Error %: {best_performance['test_error_pct']:.2f}%")
    
    if ensemble_pred is not None:
        print(f"  🎯 Ensemble Performance:")
        print(f"    - R² Score: {ensemble_r2:.4f}")
        print(f"    - MAE: {ensemble_mae:.4f}")
        print(f"    - Error %: {ensemble_error_pct:.2f}%")
    
    print(f"\n📈 ERROR REDUCTION ACHIEVED:")
    print(f"  - Initial Error: {initial_error:.2f}%")
    if corrected_errors:
        best_corrected_error = min(corrected_errors.values())
        best_corrected_model = min(corrected_errors, key=corrected_errors.get)
        improvement = ((initial_error - best_corrected_error) / initial_error * 100)
        print(f"  - Best Corrected Error: {best_corrected_error:.2f}% ({best_corrected_model})")
        print(f"  - Total Improvement: {improvement:.1f}%")
    
    print(f"\n📁 FILES SAVED:")
    print(f"  - best_thermal_model_{best_model_name}.pkl")
    print(f"  - feature_scaler.pkl")
    print(f"  - model_performance_summary.csv")
    print(f"  - prediction_results.csv")
    print(f"  - error_improvement_summary.csv")
    
    print(f"\n🚀 NEXT STEPS:")
    print(f"  1. Review the generated visualizations")
    print(f"  2. Analyze feature importance results")
    print(f"  3. Use the saved model for new predictions")
    print(f"  4. Consider deploying in your thermal management system")
    
    # Return comprehensive results
    return {
        'data_processor': data_processor,
        'ml_pipeline': ml_pipeline,
        'best_model': best_model,
        'best_model_name': best_model_name,
        'scaler': scaler,
        'feature_names': feature_names,
        'performance_summary': performance_summary,
        'initial_error_percentage': initial_error,
        'corrected_errors': corrected_errors,
        'ensemble_prediction': ensemble_pred,
        'test_results': results_df
    }

def main():
    # Define your file paths here
    xlsx_0d_path = r"C:\Users\KIIT0001\Desktop\JU\Thermal_Runaway\Arush\MATLAB DATA.xlsx"
    xlsx_3d_paths = [
        r"C:\Users\KIIT0001\Desktop\JU\Thermal_Runaway\Arush\Point A\pointA.xlsx.txt.xlsx",
        r"C:\Users\KIIT0001\Desktop\JU\Thermal_Runaway\Arush\Point B\vol-mon-1.out.xlsx",
        r"C:\Users\KIIT0001\Desktop\JU\Thermal_Runaway\Arush\Point C\vol-mon-1.out.xlsx",
        r"C:\Users\KIIT0001\Desktop\JU\Thermal_Runaway\Arush\Point D\vol-mon-1.out.xlsx"
    ]
    # Call your main pipeline function
    main_thermal_runaway_pipeline(xlsx_0d_path, xlsx_3d_paths)

if __name__ == "__main__":
    main()
