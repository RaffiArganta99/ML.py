import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import LabelEncoder
import glob

warnings.filterwarnings('ignore')

class CryptoTargetEngineer:
    """
    Advanced target engineering system for cryptocurrency prediction models.
    Supports multiple prediction types: binary classification, regression, and multi-class.
    """
    
    def __init__(self, data_dir=None, output_dir=None):
        """
        Initialize target engineer with input and output directories
        
        Args:
            data_dir (str): Directory containing processed data files
            output_dir (str): Directory to save labeled data
        """
        # Get current script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Set default paths
        if data_dir is None:
            # Look for processed data in parent directory
            self.data_dir = os.path.join(os.path.dirname(script_dir), "processed")
        else:
            self.data_dir = data_dir
            
        if output_dir is None:
            # Save labeled data in 'labeled' directory
            self.output_dir = os.path.join(os.path.dirname(script_dir), "labeled")
        else:
            self.output_dir = output_dir
            
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Storage for labeled datasets
        self.labeled_data = {}
        
        print(f"📂 Input directory: {self.data_dir}")
        print(f"📂 Output directory: {self.output_dir}")
    
    def find_processed_files(self):
        """Find all processed data files"""
        patterns = [
            os.path.join(self.data_dir, "*.feather"),
            os.path.join(self.data_dir, "*.pkl"),
            os.path.join(self.data_dir, "*.csv")
        ]
        
        files = []
        for pattern in patterns:
            files.extend(glob.glob(pattern))
        
        if not files:
            print(f"❌ No processed files found in '{self.data_dir}'")
            return []
        
        print(f"📂 Found {len(files)} processed files:")
        for file in files:
            print(f"   📄 {os.path.basename(file)}")
        
        return files
    
    def load_processed_data(self, filepath):
        """Load processed data from file"""
        try:
            filename = os.path.basename(filepath)
            print(f"\n📖 Loading: {filename}")
            
            if filepath.endswith('.feather'):
                df = pd.read_feather(filepath)
                if 'timestamp' in df.columns:
                    df.set_index('timestamp', inplace=True)
            elif filepath.endswith('.pkl'):
                df = pd.read_pickle(filepath)
            elif filepath.endswith('.csv'):
                df = pd.read_csv(filepath)
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
            
            print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
            return df
            
        except Exception as e:
            print(f"❌ Error loading {filepath}: {e}")
            return None
    
    def create_binary_targets(self, df, horizons=[1, 4, 12, 24], threshold=0.02):
        """
        Create binary classification targets (Up/Down)
        
        Args:
            df (pd.DataFrame): Input data with price columns
            horizons (list): Prediction horizons in hours
            threshold (float): Minimum price change threshold (2% default)
            
        Returns:
            pd.DataFrame: DataFrame with binary target columns
        """
        print(f"🎯 Creating binary targets for horizons: {horizons}")
        
        labeled_df = df.copy()
        
        for h in horizons:
            # Calculate future returns
            future_return = df['close'].shift(-h) / df['close'] - 1
            
            # Simple binary: 1 if price goes up, 0 if down
            labeled_df[f'target_binary_{h}h'] = (future_return > 0).astype(int)
            
            # Threshold-based binary: 1 if significant up move, 0 otherwise
            labeled_df[f'target_binary_thresh_{h}h'] = (future_return > threshold).astype(int)
            
            # Strong movement binary: 1 if up > threshold, -1 if down < -threshold, 0 otherwise
            strong_target = np.where(future_return > threshold, 1,
                                   np.where(future_return < -threshold, -1, 0))
            labeled_df[f'target_strong_{h}h'] = strong_target
            
            # Store the raw future return for analysis
            labeled_df[f'future_return_{h}h'] = future_return
        
        return labeled_df
    
    def create_regression_targets(self, df, horizons=[1, 4, 12, 24]):
        """
        Create regression targets (continuous returns)
        
        Args:
            df (pd.DataFrame): Input data
            horizons (list): Prediction horizons in hours
            
        Returns:
            pd.DataFrame: DataFrame with regression target columns
        """
        print(f"📈 Creating regression targets for horizons: {horizons}")
        
        labeled_df = df.copy()
        
        for h in horizons:
            # Raw return
            labeled_df[f'target_return_{h}h'] = df['close'].shift(-h) / df['close'] - 1
            
            # Log return
            labeled_df[f'target_log_return_{h}h'] = np.log(df['close'].shift(-h) / df['close'])
            
            # Price ratio
            labeled_df[f'target_price_ratio_{h}h'] = df['close'].shift(-h) / df['close']
            
            # Absolute price change
            labeled_df[f'target_price_change_{h}h'] = df['close'].shift(-h) - df['close']
            
            # Volatility-adjusted return
            if 'volatility_7d' in df.columns:
                vol_adj_return = labeled_df[f'target_return_{h}h'] / (df['volatility_7d'] + 1e-8)
                labeled_df[f'target_vol_adj_return_{h}h'] = vol_adj_return
        
        return labeled_df
    
    def create_multiclass_targets(self, df, horizons=[1, 4, 12, 24], 
                                thresholds=[0.01, 0.03, 0.05]):
        """
        Create multi-class targets (Strong Down, Down, Neutral, Up, Strong Up)
        
        Args:
            df (pd.DataFrame): Input data
            horizons (list): Prediction horizons in hours
            thresholds (list): Thresholds for class boundaries
            
        Returns:
            pd.DataFrame: DataFrame with multi-class target columns
        """
        print(f"🎨 Creating multi-class targets for horizons: {horizons}")
        
        labeled_df = df.copy()
        
        for h in horizons:
            future_return = df['close'].shift(-h) / df['close'] - 1
            
            for i, thresh in enumerate(thresholds):
                classes = np.where(future_return > thresh, 2,      # Strong Up
                          np.where(future_return > thresh/2, 1,    # Up
                          np.where(future_return < -thresh, -2,    # Strong Down
                          np.where(future_return < -thresh/2, -1,  # Down
                                   0))))                           # Neutral
                
                labeled_df[f'target_multiclass_{h}h_t{i+1}'] = classes
        
        return labeled_df
    
    def create_breakout_targets(self, df, horizons=[4, 12, 24], 
                              bb_multiplier=2.0, volume_multiplier=1.5):
        """
        Create breakout detection targets
        
        Args:
            df (pd.DataFrame): Input data
            horizons (list): Prediction horizons in hours
            bb_multiplier (float): Bollinger Band multiplier
            volume_multiplier (float): Volume spike multiplier
            
        Returns:
            pd.DataFrame: DataFrame with breakout target columns
        """
        print(f"💥 Creating breakout targets for horizons: {horizons}")
        
        labeled_df = df.copy()
        
        # Calculate additional indicators if not present
        if 'bb_upper' not in df.columns:
            sma_20 = df['close'].rolling(20).mean()
            std_20 = df['close'].rolling(20).std()
            labeled_df['bb_upper'] = sma_20 + (std_20 * bb_multiplier)
            labeled_df['bb_lower'] = sma_20 - (std_20 * bb_multiplier)
        
        if 'volume_sma' not in df.columns:
            labeled_df['volume_sma'] = df['volume'].rolling(20).mean()
        
        for h in horizons:
            # Future price movement
            future_high = df['high'].rolling(h).max().shift(-h)
            future_low = df['low'].rolling(h).min().shift(-h)
            
            # Bollinger Band breakouts
            bb_upward_breakout = (future_high > labeled_df['bb_upper']).astype(int)
            bb_downward_breakout = (future_low < labeled_df['bb_lower']).astype(int)
            
            # Volume-confirmed breakouts
            future_volume_spike = (df['volume'].rolling(h).max().shift(-h) > 
                                 labeled_df['volume_sma'] * volume_multiplier)
            
            # Combined breakout signals
            labeled_df[f'target_breakout_up_{h}h'] = (bb_upward_breakout & future_volume_spike).astype(int)
            labeled_df[f'target_breakout_down_{h}h'] = (bb_downward_breakout & future_volume_spike).astype(int)
            
            # General breakout (either direction)
            labeled_df[f'target_breakout_any_{h}h'] = (
                labeled_df[f'target_breakout_up_{h}h'] | 
                labeled_df[f'target_breakout_down_{h}h']
            ).astype(int)
            
            # Breakout strength (how far beyond the bands)
            upward_strength = np.maximum(0, (future_high - labeled_df['bb_upper']) / labeled_df['bb_upper'])
            downward_strength = np.maximum(0, (labeled_df['bb_lower'] - future_low) / labeled_df['bb_lower'])
            
            labeled_df[f'target_breakout_strength_up_{h}h'] = upward_strength
            labeled_df[f'target_breakout_strength_down_{h}h'] = downward_strength
        
        return labeled_df
    
    def create_volatility_targets(self, df, horizons=[4, 12, 24]):
        """
        Create volatility prediction targets
        
        Args:
            df (pd.DataFrame): Input data
            horizons (list): Prediction horizons in hours
            
        Returns:
            pd.DataFrame: DataFrame with volatility target columns
        """
        print(f"📊 Creating volatility targets for horizons: {horizons}")
        
        labeled_df = df.copy()
        
        for h in horizons:
            # Future realized volatility
            future_returns = df['close'].pct_change().shift(-h).rolling(h).std()
            labeled_df[f'target_volatility_{h}h'] = future_returns
            
            # Future high-low range
            future_hl_range = (df['high'].rolling(h).max().shift(-h) - 
                             df['low'].rolling(h).min().shift(-h)) / df['close']
            labeled_df[f'target_hl_range_{h}h'] = future_hl_range
            
            # Volatility regime (high/low volatility periods)
            current_vol = df['volatility_7d'] if 'volatility_7d' in df.columns else future_returns.rolling(168).mean()
            vol_percentile = current_vol.rolling(168*4).rank(pct=True)  # 4 weeks
            
            labeled_df[f'target_vol_regime_{h}h'] = np.where(vol_percentile > 0.8, 2,  # High vol
                                                           np.where(vol_percentile < 0.2, 0, 1))  # Low vol, Medium vol
        
        return labeled_df
    
    def create_support_resistance_targets(self, df, horizons=[4, 12, 24], 
                                        lookback_window=48):
        """
        Create support/resistance level targets
        
        Args:
            df (pd.DataFrame): Input data
            horizons (list): Prediction horizons in hours
            lookback_window (int): Hours to look back for S/R levels
            
        Returns:
            pd.DataFrame: DataFrame with S/R target columns
        """
        print(f"📏 Creating support/resistance targets for horizons: {horizons}")
        
        labeled_df = df.copy()
        
        # Calculate support and resistance levels
        labeled_df['resistance'] = df['high'].rolling(lookback_window).max()
        labeled_df['support'] = df['low'].rolling(lookback_window).min()
        
        for h in horizons:
            future_high = df['high'].rolling(h).max().shift(-h)
            future_low = df['low'].rolling(h).min().shift(-h)
            
            # Resistance break targets
            resistance_break = (future_high > labeled_df['resistance'] * 1.001).astype(int)  # 0.1% buffer
            labeled_df[f'target_resistance_break_{h}h'] = resistance_break
            
            # Support break targets
            support_break = (future_low < labeled_df['support'] * 0.999).astype(int)  # 0.1% buffer
            labeled_df[f'target_support_break_{h}h'] = support_break
            
            # Distance to levels
            labeled_df[f'target_resistance_distance_{h}h'] = (labeled_df['resistance'] - df['close']) / df['close']
            labeled_df[f'target_support_distance_{h}h'] = (df['close'] - labeled_df['support']) / df['close']
        
        return labeled_df
    
    def clean_targets(self, df, min_target_coverage=0.95):
        """
        Clean target columns by removing incomplete data
        
        Args:
            df (pd.DataFrame): DataFrame with targets
            min_target_coverage (float): Minimum data coverage required
            
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        print("🧹 Cleaning target data...")
        
        # Find target columns
        target_cols = [col for col in df.columns if col.startswith('target_')]
        
        print(f"Found {len(target_cols)} target columns")
        
        # Remove rows where most targets are NaN
        target_data = df[target_cols]
        coverage = target_data.notna().sum(axis=1) / len(target_cols)
        
        valid_rows = coverage >= min_target_coverage
        cleaned_df = df[valid_rows].copy()
        
        removed_rows = len(df) - len(cleaned_df)
        if removed_rows > 0:
            print(f"🗑️  Removed {removed_rows} rows with insufficient target coverage")
        
        # Fill remaining NaN values in targets with appropriate defaults
        for col in target_cols:
            if cleaned_df[col].isna().sum() > 0:
                if 'binary' in col or 'breakout' in col:
                    cleaned_df[col] = cleaned_df[col].fillna(0)
                elif 'return' in col or 'change' in col:
                    cleaned_df[col] = cleaned_df[col].fillna(0.0)
                elif 'multiclass' in col:
                    cleaned_df[col] = cleaned_df[col].fillna(0)
        
        return cleaned_df
    
    def generate_target_summary(self, df, symbol):
        """Generate summary statistics for targets"""
        print(f"\n🎯 TARGET SUMMARY for {symbol}")
        print("=" * 60)
        
        target_cols = [col for col in df.columns if col.startswith('target_')]
        
        for col in target_cols:
            if df[col].dtype in ['int64', 'float64']:
                print(f"\n📊 {col}:")
                print(f"   Count: {df[col].count():,}")
                print(f"   Mean: {df[col].mean():.4f}")
                print(f"   Std: {df[col].std():.4f}")
                print(f"   Min: {df[col].min():.4f}")
                print(f"   Max: {df[col].max():.4f}")
                
                # For binary/multiclass targets, show class distribution
                if 'binary' in col or 'multiclass' in col or 'breakout' in col:
                    value_counts = df[col].value_counts().sort_index()
                    print(f"   Distribution: {dict(value_counts)}")
    
    def save_labeled_data(self, df, symbol, target_types, format='feather'):
        """Save labeled data to file"""
        try:
            clean_symbol = symbol.replace('/', '_').replace(':', '_')
            target_suffix = '_'.join(target_types)
            
            if format == 'feather':
                df_to_save = df.reset_index()
                filepath = os.path.join(self.output_dir, f"{clean_symbol}_labeled_{target_suffix}.feather")
                df_to_save.to_feather(filepath)
            elif format == 'pkl':
                filepath = os.path.join(self.output_dir, f"{clean_symbol}_labeled_{target_suffix}.pkl")
                df.to_pickle(filepath)
            elif format == 'csv':
                filepath = os.path.join(self.output_dir, f"{clean_symbol}_labeled_{target_suffix}.csv")
                df.to_csv(filepath)
            
            file_size = os.path.getsize(filepath)
            print(f"💾 Saved labeled data: {os.path.basename(filepath)} ({file_size:,} bytes)")
            
            return filepath
            
        except Exception as e:
            print(f"❌ Error saving labeled data for {symbol}: {e}")
            return None
    
    def process_all_files(self, target_types=['binary', 'regression', 'multiclass'], 
                         horizons=[1, 4, 12, 24], output_format='feather'):
        """
        Process all files and create specified target types
        
        Args:
            target_types (list): Types of targets to create
            horizons (list): Prediction horizons
            output_format (str): Output file format
        """
        print("🚀 CRYPTO TARGET ENGINEERING v1.0")
        print("=" * 60)
        
        # Find processed files
        files = self.find_processed_files()
        
        if not files:
            return
        
        successful_processing = 0
        
        for filepath in files:
            # Extract symbol from filename
            filename = os.path.basename(filepath)
            symbol = filename.split('_')[0].replace('_', '/')
            
            # Load data
            df = self.load_processed_data(filepath)
            
            if df is not None:
                labeled_df = df.copy()
                
                # Create requested target types
                if 'binary' in target_types:
                    labeled_df = self.create_binary_targets(labeled_df, horizons)
                
                if 'regression' in target_types:
                    labeled_df = self.create_regression_targets(labeled_df, horizons)
                
                if 'multiclass' in target_types:
                    labeled_df = self.create_multiclass_targets(labeled_df, horizons)
                
                if 'breakout' in target_types:
                    labeled_df = self.create_breakout_targets(labeled_df, horizons)
                
                if 'volatility' in target_types:
                    labeled_df = self.create_volatility_targets(labeled_df, horizons)
                
                if 'support_resistance' in target_types:
                    labeled_df = self.create_support_resistance_targets(labeled_df, horizons)
                
                # Clean targets
                labeled_df = self.clean_targets(labeled_df)
                
                # Save labeled data
                saved_path = self.save_labeled_data(labeled_df, symbol, target_types, output_format)
                
                if saved_path:
                    # Store in memory
                    self.labeled_data[symbol] = labeled_df
                    
                    # Generate summary
                    self.generate_target_summary(labeled_df, symbol)
                    
                    successful_processing += 1
                
                print("-" * 50)
        
        # Final summary
        print("=" * 60)
        print("📊 TARGET ENGINEERING SUMMARY:")
        print(f"✅ Successfully processed: {successful_processing}/{len(files)} files")
        print(f"❌ Failed processing: {len(files) - successful_processing}/{len(files)} files")
        print(f"🎯 Target types created: {', '.join(target_types)}")
        print(f"⏰ Prediction horizons: {horizons} hours")
        
        if successful_processing > 0:
            print(f"\n💾 Labeled files saved in '{self.output_dir}/' directory")
        
        print("\n🎉 Target engineering completed!")
        return self.labeled_data


def main():
    """Main function to run target engineering"""
    
    # Initialize target engineer
    engineer = CryptoTargetEngineer()
    
    # Check if processed data directory exists
    if not os.path.exists(engineer.data_dir):
        print(f"❌ Processed data directory not found: {engineer.data_dir}")
        print("🔧 Please run the data preprocessor first to create processed data files")
        return
    
    # Configuration
    target_types = ['binary', 'regression', 'multiclass', 'breakout']
    horizons = [1, 4, 12, 24]  # 1h, 4h, 12h, 24h predictions
    
    print("🎯 Target Engineering Configuration:")
    print(f"   Target types: {target_types}")
    print(f"   Prediction horizons: {horizons} hours")
    print()
    
    # Process all files
    labeled_data = engineer.process_all_files(
        target_types=target_types,
        horizons=horizons,
        output_format='feather'
    )
    
    # Optional: Show available labeled datasets
    if labeled_data:
        print(f"\n🔄 Labeled datasets available in memory:")
        for symbol, df in labeled_data.items():
            target_cols = [col for col in df.columns if col.startswith('target_')]
            print(f"   📈 {symbol}: {len(df)} records, {len(target_cols)} targets")


if __name__ == "__main__":
    main()