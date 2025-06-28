import os
import pandas as pd
import numpy as np
from datetime import datetime
import glob
import warnings

warnings.filterwarnings('ignore')

class CryptoDataPreprocessor:
    """
    Advanced cryptocurrency data preprocessor for cleaning, normalizing, 
    and preparing data for statistical analysis and modeling.
    """
    
    def __init__(self, data_dir=None, output_dir=None):
        """
        Initialize preprocessor with input and output directories
        
        Args:
            data_dir (str): Directory containing raw CSV files
            output_dir (str): Directory to save processed data
        """
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Set default paths relative to project root
        if data_dir is None:
            # Go up one level from models/ to project root, then to data/
            self.data_dir = os.path.join(os.path.dirname(script_dir), "data")
        else:
            self.data_dir = data_dir
            
        if output_dir is None:
            # Use current directory (models/) as output
            self.output_dir = os.path.join(os.path.dirname(script_dir), "processed")
        else:
            self.output_dir = output_dir
            
        self.processed_data = {}
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Print paths for debugging
        print(f"📂 Data directory: {self.data_dir}")
        print(f"📂 Output directory: {self.output_dir}")
        
    def find_csv_files(self):
        """Find all CSV files in the data directory"""
        pattern = os.path.join(self.data_dir, "*.csv")
        csv_files = glob.glob(pattern)
        
        if not csv_files:
            print(f"❌ No CSV files found in '{self.data_dir}' directory")
            return []
        
        print(f"📂 Found {len(csv_files)} CSV files:")
        for file in csv_files:
            print(f"   📄 {os.path.basename(file)}")
        
        return csv_files
    
    def load_and_validate_data(self, filepath):
        """
        Load CSV file and perform basic validation
        
        Args:
            filepath (str): Path to CSV file
            
        Returns:
            pd.DataFrame or None: Loaded and validated DataFrame
        """
        try:
            print(f"\n📖 Loading: {os.path.basename(filepath)}")
            
            # Load CSV
            df = pd.read_csv(filepath)
            
            # Basic validation
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"❌ Missing columns: {missing_cols}")
                return None
            
            print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
            print(f"📅 Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading {filepath}: {e}")
            return None
    
    def clean_and_process_data(self, df, symbol):
        """
        Clean and process the raw data
        
        Args:
            df (pd.DataFrame): Raw data
            symbol (str): Symbol name for identification
            
        Returns:
            pd.DataFrame: Processed data
        """
        print(f"🧹 Processing data for {symbol}...")
        
        # Make a copy to avoid modifying original
        processed_df = df.copy()
        
        # 1. Parse timestamp and set as index
        if processed_df['timestamp'].dtype == 'object':
            processed_df['timestamp'] = pd.to_datetime(processed_df['timestamp'])
        
        processed_df.set_index('timestamp', inplace=True)
        processed_df.sort_index(inplace=True)
        
        # 2. Ensure numeric columns are float
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
        
        # 3. Drop rows with NaN values
        initial_rows = len(processed_df)
        processed_df.dropna(inplace=True)
        dropped_rows = initial_rows - len(processed_df)
        
        if dropped_rows > 0:
            print(f"🗑️  Dropped {dropped_rows} rows with NaN values")
        
        # 4. Calculate returns and statistical features
        processed_df = self.calculate_returns(processed_df)
        
        # 5. Calculate technical indicators
        processed_df = self.calculate_technical_indicators(processed_df)
        
        # 6. Add time-based features
        processed_df = self.add_time_features(processed_df)
        
        print(f"✅ Processing complete: {len(processed_df)} rows, {len(processed_df.columns)} columns")
        
        return processed_df
    
    def calculate_returns(self, df):
        """
        Calculate various types of returns for statistical analysis
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with returns columns added
        """
        print("📊 Calculating returns...")
        
        # Simple returns (percentage change)
        df['returns'] = df['close'].pct_change()
        
        # Log returns (natural logarithm of price ratios)
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Price differences
        df['price_diff'] = df['close'].diff()
        
        # High-Low spread
        df['hl_spread'] = df['high'] - df['low']
        df['hl_spread_pct'] = (df['hl_spread'] / df['close']) * 100
        
        # Open-Close difference
        df['oc_diff'] = df['close'] - df['open']
        df['oc_diff_pct'] = (df['oc_diff'] / df['open']) * 100
        
        # Volatility (rolling standard deviation of returns)
        df['volatility_7d'] = df['returns'].rolling(window=7*24).std()  # 7 days for hourly data
        df['volatility_30d'] = df['returns'].rolling(window=30*24).std()  # 30 days for hourly data
        
        return df
    
    def calculate_technical_indicators(self, df):
        """
        Calculate basic technical indicators
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with technical indicators added
        """
        print("📈 Calculating technical indicators...")
        
        # Moving averages
        df['sma_7'] = df['close'].rolling(window=7).mean()
        df['sma_25'] = df['close'].rolling(window=25).mean()
        df['sma_99'] = df['close'].rolling(window=99).mean()
        
        # Exponential moving averages
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        sma_20 = df['close'].rolling(window=20).mean()
        std_20 = df['close'].rolling(window=20).std()
        df['bb_upper'] = sma_20 + (std_20 * 2)
        df['bb_lower'] = sma_20 - (std_20 * 2)
        df['bb_middle'] = sma_20
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (df['close'] - df['bb_lower']) / df['bb_width']
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        return df
    
    def add_time_features(self, df):
        """
        Add time-based features for analysis
        
        Args:
            df (pd.DataFrame): DataFrame with datetime index
            
        Returns:
            pd.DataFrame: DataFrame with time features added
        """
        print("🕐 Adding time features...")
        
        # Extract time components
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['year'] = df.index.year
        
        # Binary features
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_month_end'] = (df.index.day >= 28).astype(int)
        df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
        
        return df
    
    def save_processed_data(self, df, symbol, format='feather'):
        """
        Save processed data to file
        
        Args:
            df (pd.DataFrame): Processed data
            symbol (str): Symbol name
            format (str): Output format ('feather', 'pkl', 'csv')
        """
        try:
            # Clean symbol name for filename
            clean_symbol = symbol.replace('/', '_').replace(':', '_')
            
            if format == 'feather':
                # Reset index to save timestamp as column for feather format
                df_to_save = df.reset_index()
                filepath = os.path.join(self.output_dir, f"{clean_symbol}_processed.feather")
                df_to_save.to_feather(filepath)
                
            elif format == 'pkl':
                filepath = os.path.join(self.output_dir, f"{clean_symbol}_processed.pkl")
                df.to_pickle(filepath)
                
            elif format == 'csv':
                filepath = os.path.join(self.output_dir, f"{clean_symbol}_processed.csv")
                df.to_csv(filepath)
            
            file_size = os.path.getsize(filepath)
            print(f"💾 Saved processed data: {os.path.basename(filepath)} ({file_size:,} bytes)")
            
            return filepath
            
        except Exception as e:
            print(f"❌ Error saving processed data for {symbol}: {e}")
            return None
    
    def generate_summary_stats(self, df, symbol):
        """
        Generate summary statistics for the processed data
        
        Args:
            df (pd.DataFrame): Processed data
            symbol (str): Symbol name
        """
        print(f"\n📊 SUMMARY STATISTICS for {symbol}")
        print("=" * 50)
        
        # Basic info
        print(f"Data points: {len(df):,}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"Total days: {(df.index.max() - df.index.min()).days}")
        
        # Price statistics
        price_stats = df['close'].describe()
        print(f"\n💰 PRICE STATISTICS:")
        print(f"   Current: ${df['close'].iloc[-1]:.4f}")
        print(f"   Min: ${price_stats['min']:.4f}")
        print(f"   Max: ${price_stats['max']:.4f}")
        print(f"   Mean: ${price_stats['mean']:.4f}")
        print(f"   Std: ${price_stats['std']:.4f}")
        
        # Returns statistics
        if 'returns' in df.columns:
            returns_stats = df['returns'].describe()
            print(f"\n📈 RETURNS STATISTICS:")
            print(f"   Mean return: {returns_stats['mean']*100:.4f}%")
            print(f"   Std return: {returns_stats['std']*100:.4f}%")
            print(f"   Min return: {returns_stats['min']*100:.2f}%")
            print(f"   Max return: {returns_stats['max']*100:.2f}%")
            print(f"   Skewness: {df['returns'].skew():.4f}")
            print(f"   Kurtosis: {df['returns'].kurtosis():.4f}")
        
        # Volume statistics
        if 'volume' in df.columns:
            volume_stats = df['volume'].describe()
            print(f"\n📊 VOLUME STATISTICS:")
            print(f"   Mean volume: {volume_stats['mean']:,.0f}")
            print(f"   Max volume: {volume_stats['max']:,.0f}")
            print(f"   Min volume: {volume_stats['min']:,.0f}")
        
        # Data quality
        total_cells = len(df) * len(df.columns)
        nan_cells = df.isnull().sum().sum()
        print(f"\n🔍 DATA QUALITY:")
        print(f"   Total cells: {total_cells:,}")
        print(f"   NaN cells: {nan_cells:,}")
        print(f"   Completeness: {((total_cells-nan_cells)/total_cells)*100:.2f}%")
        
    def process_all_files(self, output_format='feather'):
        """
        Process all CSV files in the data directory
        
        Args:
            output_format (str): Output format for processed files
        """
        print("🚀 CRYPTO DATA PREPROCESSOR v1.0")
        print("=" * 60)
        
        # Find CSV files
        csv_files = self.find_csv_files()
        
        if not csv_files:
            return
        
        successful_processing = 0
        
        for filepath in csv_files:
            # Extract symbol from filename
            filename = os.path.basename(filepath)
            symbol = filename.replace('.csv', '').replace('_binance_direct', '').replace('_binance', '').replace('_', '/')
            
            # Load data
            df = self.load_and_validate_data(filepath)
            
            if df is not None:
                # Process data
                processed_df = self.clean_and_process_data(df, symbol)
                
                # Save processed data
                saved_path = self.save_processed_data(processed_df, symbol, output_format)
                
                if saved_path:
                    # Store in memory for potential further use
                    self.processed_data[symbol] = processed_df
                    
                    # Generate summary
                    self.generate_summary_stats(processed_df, symbol)
                    
                    successful_processing += 1
                
                print("-" * 50)
        
        # Final summary
        print("=" * 60)
        print("📊 PREPROCESSING SUMMARY:")
        print(f"✅ Successfully processed: {successful_processing}/{len(csv_files)} files")
        print(f"❌ Failed processing: {len(csv_files) - successful_processing}/{len(csv_files)} files")
        
        if successful_processing > 0:
            print(f"\n💾 Processed files saved in '{self.output_dir}/' directory:")
            try:
                for file in os.listdir(self.output_dir):
                    if file.endswith(('.feather', '.pkl', '.csv')):
                        size = os.path.getsize(os.path.join(self.output_dir, file))
                        print(f"   📄 {file} ({size:,} bytes)")
            except:
                pass
        
        print("\n🎉 Preprocessing completed!")
        return self.processed_data


def main():
    """Main function to run the preprocessing"""
    
    # Initialize preprocessor with auto-detected paths
    preprocessor = CryptoDataPreprocessor()
    
    # Check if data directory exists
    if not os.path.exists(preprocessor.data_dir):
        print(f"❌ Data directory not found: {preprocessor.data_dir}")
        print("🔧 Please check your folder structure:")
        print("   altcoin-ai-env/")
        print("   ├── data/           # CSV files should be here")
        print("   ├── models/         # This script is here")
        print("   └── ...")
        return
    
    # Process all files
    processed_data = preprocessor.process_all_files(output_format='feather')
    
    # Optional: Access processed data programmatically
    if processed_data:
        print(f"\n🔄 Processed data available in memory for {len(processed_data)} symbols:")
        for symbol in processed_data.keys():
            print(f"   📈 {symbol}: {len(processed_data[symbol])} records")


if __name__ == "__main__":
    main()