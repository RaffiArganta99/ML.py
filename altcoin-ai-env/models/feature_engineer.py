import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
import glob
import ta

warnings.filterwarnings('ignore')

class CryptoFeatureEngineer:
    """
    Advanced feature engineering system for cryptocurrency prediction models.
    Creates technical indicators, patterns, and domain-specific features.
    """
    
    def __init__(self, data_dir=None, output_dir=None):
        """
        Initialize feature engineer with input and output directories
        
        Args:
            data_dir (str): Directory containing labeled data files
            output_dir (str): Directory to save feature-engineered data
        """
        # Get current script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Set default paths
        if data_dir is None:
            # Look for labeled data in parent directory
            self.data_dir = os.path.join(os.path.dirname(script_dir), "labeled")
        else:
            self.data_dir = data_dir
            
        if output_dir is None:
            # Save feature data in 'features' directory
            self.output_dir = os.path.join(os.path.dirname(script_dir), "features")
        else:
            self.output_dir = output_dir
            
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Storage for feature datasets
        self.feature_data = {}
        self.feature_importance = {}
        
        print(f"📂 Input directory: {self.data_dir}")
        print(f"📂 Output directory: {self.output_dir}")
    
    def find_labeled_files(self):
        """Find all labeled data files"""
        patterns = [
            os.path.join(self.data_dir, "*labeled*.feather"),
            os.path.join(self.data_dir, "*labeled*.pkl"),
            os.path.join(self.data_dir, "*labeled*.csv")
        ]
        
        files = []
        for pattern in patterns:
            files.extend(glob.glob(pattern))
        
        if not files:
            print(f"❌ No labeled files found in '{self.data_dir}'")
            return []
        
        print(f"📂 Found {len(files)} labeled files:")
        for file in files:
            print(f"   📄 {os.path.basename(file)}")
        
        return files
    
    def load_labeled_data(self, filepath):
        """Load labeled data from file"""
        try:
            filename = os.path.basename(filepath)
            print(f"\n📖 Loading: {filename}")
            
            if filepath.endswith('.feather'):
                df = pd.read_feather(filepath)
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
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
    
    def create_price_features(self, df):
        """Create price-based technical features"""
        print("📈 Creating price-based features...")
        
        feature_df = df.copy()
        
        # Moving Averages
        periods = [5, 10, 13, 21, 34, 50, 100, 200]
        for period in periods:
            if len(df) >= period:
                # Simple Moving Average
                feature_df[f'sma_{period}'] = df['close'].rolling(period).mean()
                
                # Exponential Moving Average
                feature_df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
                
                # Price relative to MA
                feature_df[f'price_vs_sma_{period}'] = df['close'] / feature_df[f'sma_{period}'] - 1
                feature_df[f'price_vs_ema_{period}'] = df['close'] / feature_df[f'ema_{period}'] - 1
        
        # Moving Average Crossovers
        if len(df) >= 50:
            feature_df['ma_cross_13_21'] = np.where(
                feature_df['ema_13'] > feature_df['ema_21'], 1, 0
            )
            feature_df['ma_cross_21_50'] = np.where(
                feature_df['ema_21'] > feature_df['ema_50'], 1, 0
            )
        
        # Bollinger Bands
        if len(df) >= 20:
            bb_period = 20
            bb_std = 2
            sma_bb = df['close'].rolling(bb_period).mean()
            std_bb = df['close'].rolling(bb_period).std()
            
            feature_df['bb_upper'] = sma_bb + (std_bb * bb_std)
            feature_df['bb_lower'] = sma_bb - (std_bb * bb_std)
            feature_df['bb_middle'] = sma_bb
            feature_df['bb_width'] = (feature_df['bb_upper'] - feature_df['bb_lower']) / feature_df['bb_middle']
            feature_df['bb_position'] = (df['close'] - feature_df['bb_lower']) / (feature_df['bb_upper'] - feature_df['bb_lower'])
        
        # Price channels
        if len(df) >= 14:
            feature_df['highest_14'] = df['high'].rolling(14).max()
            feature_df['lowest_14'] = df['low'].rolling(14).min()
            feature_df['channel_position'] = (df['close'] - feature_df['lowest_14']) / (feature_df['highest_14'] - feature_df['lowest_14'])
        
        return feature_df
    
    def create_momentum_features(self, df):
        """Create momentum-based technical features"""
        print("⚡ Creating momentum features...")
        
        feature_df = df.copy()
        
        # RSI
        periods = [7, 14, 21]
        for period in periods:
            if len(df) >= period * 2:
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0).rolling(period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
                rs = gain / loss
                feature_df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # MACD
        if len(df) >= 26:
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            feature_df['macd'] = ema_12 - ema_26
            feature_df['macd_signal'] = feature_df['macd'].ewm(span=9).mean()
            feature_df['macd_histogram'] = feature_df['macd'] - feature_df['macd_signal']
            
            # MACD crossover signals
            feature_df['macd_bullish'] = np.where(feature_df['macd'] > feature_df['macd_signal'], 1, 0)
        
        # Stochastic Oscillator
        if len(df) >= 14:
            k_period = 14
            d_period = 3
            lowest_low = df['low'].rolling(k_period).min()
            highest_high = df['high'].rolling(k_period).max()
            
            feature_df['stoch_k'] = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low)
            feature_df['stoch_d'] = feature_df['stoch_k'].rolling(d_period).mean()
            
            # Stochastic signals
            feature_df['stoch_oversold'] = np.where(feature_df['stoch_k'] < 20, 1, 0)
            feature_df['stoch_overbought'] = np.where(feature_df['stoch_k'] > 80, 1, 0)
        
        # Rate of Change (ROC)
        periods = [1, 4, 12, 24]
        for period in periods:
            if len(df) >= period:
                feature_df[f'roc_{period}'] = df['close'].pct_change(period)
        
        # Momentum
        if len(df) >= 10:
            feature_df['momentum_10'] = df['close'] - df['close'].shift(10)
        
        # Williams %R
        if len(df) >= 14:
            period = 14
            highest_high = df['high'].rolling(period).max()
            lowest_low = df['low'].rolling(period).min()
            feature_df['williams_r'] = -100 * (highest_high - df['close']) / (highest_high - lowest_low)
        
        return feature_df
    
    def create_volume_features(self, df):
        """Create volume-based features"""
        print("📊 Creating volume features...")
        
        feature_df = df.copy()
        
        # Volume Moving Averages
        periods = [5, 10, 20, 50]
        for period in periods:
            if len(df) >= period:
                feature_df[f'volume_sma_{period}'] = df['volume'].rolling(period).mean()
                feature_df[f'volume_ratio_{period}'] = df['volume'] / feature_df[f'volume_sma_{period}']
        
        # VWAP (Volume Weighted Average Price)
        if len(df) >= 20:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            feature_df['vwap'] = (typical_price * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
            feature_df['price_vs_vwap'] = df['close'] / feature_df['vwap'] - 1
        
        # On-Balance Volume (OBV)
        obv = [0]
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.append(obv[-1] + df['volume'].iloc[i])
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.append(obv[-1] - df['volume'].iloc[i])
            else:
                obv.append(obv[-1])
        
        feature_df['obv'] = obv
        
        if len(df) >= 10:
            feature_df['obv_sma_10'] = feature_df['obv'].rolling(10).mean()
            feature_df['obv_momentum'] = feature_df['obv'] - feature_df['obv_sma_10']
        
        # Volume Price Trend (VPT)
        if len(df) >= 2:
            price_change = df['close'].pct_change()
            vpt = (price_change * df['volume']).cumsum()
            feature_df['vpt'] = vpt
        
        # Volume spikes
        if len(df) >= 20:
            vol_mean = df['volume'].rolling(20).mean()
            vol_std = df['volume'].rolling(20).std()
            feature_df['volume_spike'] = np.where(
                df['volume'] > vol_mean + 2 * vol_std, 1, 0
            )
        
        return feature_df
    
    def create_volatility_features(self, df):
        """Create volatility-based features"""
        print("🌪️ Creating volatility features...")
        
        feature_df = df.copy()
        
        # Average True Range (ATR)
        periods = [7, 14, 21]
        for period in periods:
            if len(df) >= period:
                high_low = df['high'] - df['low']
                high_close = abs(df['high'] - df['close'].shift())
                low_close = abs(df['low'] - df['close'].shift())
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                feature_df[f'atr_{period}'] = true_range.rolling(period).mean()
                
                # ATR percentage
                feature_df[f'atr_pct_{period}'] = feature_df[f'atr_{period}'] / df['close']
        
        # Historical Volatility
        periods = [7, 14, 30]
        for period in periods:
            if len(df) >= period:
                returns = df['close'].pct_change()
                feature_df[f'volatility_{period}'] = returns.rolling(period).std() * np.sqrt(24)  # Annualized
        
        # High-Low Range
        feature_df['hl_range'] = (df['high'] - df['low']) / df['close']
        feature_df['hl_range_sma_10'] = feature_df['hl_range'].rolling(10).mean()
        
        # Intraday price movements
        feature_df['body_size'] = abs(df['close'] - df['open']) / df['open']
        feature_df['upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['close']
        feature_df['lower_shadow'] = (np.minimum(df['open'], df['close']) - df['low']) / df['close']
        
        return feature_df
    
    def create_pattern_features(self, df):
        """Create candlestick pattern features"""
        print("🕯️ Creating pattern features...")
        
        feature_df = df.copy()
        
        # Basic candle types
        feature_df['is_green'] = np.where(df['close'] > df['open'], 1, 0)
        feature_df['is_red'] = np.where(df['close'] < df['open'], 1, 0)
        feature_df['is_doji'] = np.where(abs(df['close'] - df['open']) / df['open'] < 0.001, 1, 0)
        
        # Body and shadow ratios
        body_size = abs(df['close'] - df['open'])
        high_low_range = df['high'] - df['low']
        
        feature_df['body_ratio'] = body_size / high_low_range
        feature_df['upper_shadow_ratio'] = (df['high'] - np.maximum(df['open'], df['close'])) / high_low_range
        feature_df['lower_shadow_ratio'] = (np.minimum(df['open'], df['close']) - df['low']) / high_low_range
        
        # Hammer and shooting star patterns
        feature_df['hammer'] = np.where(
            (feature_df['lower_shadow_ratio'] > 0.6) & 
            (feature_df['upper_shadow_ratio'] < 0.1) &
            (feature_df['body_ratio'] < 0.3), 1, 0
        )
        
        feature_df['shooting_star'] = np.where(
            (feature_df['upper_shadow_ratio'] > 0.6) & 
            (feature_df['lower_shadow_ratio'] < 0.1) &
            (feature_df['body_ratio'] < 0.3), 1, 0
        )
        
        # Engulfing patterns
        if len(df) >= 2:
            prev_body = abs(df['close'].shift(1) - df['open'].shift(1))
            curr_body = abs(df['close'] - df['open'])
            
            feature_df['bullish_engulfing'] = np.where(
                (df['close'].shift(1) < df['open'].shift(1)) &  # Previous red
                (df['close'] > df['open']) &  # Current green
                (df['open'] < df['close'].shift(1)) &  # Gap down open
                (df['close'] > df['open'].shift(1)) &  # Close above prev open
                (curr_body > prev_body), 1, 0  # Larger body
            )
            
            feature_df['bearish_engulfing'] = np.where(
                (df['close'].shift(1) > df['open'].shift(1)) &  # Previous green
                (df['close'] < df['open']) &  # Current red
                (df['open'] > df['close'].shift(1)) &  # Gap up open
                (df['close'] < df['open'].shift(1)) &  # Close below prev open
                (curr_body > prev_body), 1, 0  # Larger body
            )
        
        return feature_df
    
    def create_lag_features(self, df, lags=[1, 2, 3, 4, 6, 12, 24]):
        """Create lagged features for time series context"""
        print("⏰ Creating lag features...")
        
        feature_df = df.copy()
        
        # Price lags
        for lag in lags:
            if len(df) > lag:
                feature_df[f'close_lag_{lag}'] = df['close'].shift(lag)
                feature_df[f'return_lag_{lag}'] = df['close'].pct_change().shift(lag)
                feature_df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        
        # Rolling statistics
        windows = [3, 6, 12, 24]
        for window in windows:
            if len(df) >= window:
                feature_df[f'close_mean_{window}'] = df['close'].rolling(window).mean()
                feature_df[f'close_std_{window}'] = df['close'].rolling(window).std()
                feature_df[f'return_mean_{window}'] = df['close'].pct_change().rolling(window).mean()
                feature_df[f'return_std_{window}'] = df['close'].pct_change().rolling(window).std()
        
        return feature_df
    
    def create_market_structure_features(self, df):
        """Create market structure features"""
        print("🏗️ Creating market structure features...")
        
        feature_df = df.copy()
        
        # Support and resistance levels
        windows = [20, 50, 100]
        for window in windows:
            if len(df) >= window:
                feature_df[f'resistance_{window}'] = df['high'].rolling(window).max()
                feature_df[f'support_{window}'] = df['low'].rolling(window).min()
                
                # Distance to levels
                feature_df[f'dist_to_resistance_{window}'] = (feature_df[f'resistance_{window}'] - df['close']) / df['close']
                feature_df[f'dist_to_support_{window}'] = (df['close'] - feature_df[f'support_{window}']) / df['close']
        
        # Higher highs and lower lows
        if len(df) >= 10:
            feature_df['higher_high'] = np.where(
                df['high'] > df['high'].rolling(10).max().shift(1), 1, 0
            )
            feature_df['lower_low'] = np.where(
                df['low'] < df['low'].rolling(10).min().shift(1), 1, 0
            )
        
        # Trend strength
        if len(df) >= 20:
            slope_periods = [10, 20]
            for period in slope_periods:
                x = np.arange(period)
                slopes = []
                for i in range(period, len(df)):
                    y = df['close'].iloc[i-period:i].values
                    slope = np.polyfit(x, y, 1)[0]
                    slopes.append(slope / df['close'].iloc[i])  # Normalized slope
                
                feature_df[f'trend_slope_{period}'] = [np.nan] * period + slopes
        
        return feature_df
    
    def calculate_feature_importance(self, X, y, method='random_forest'):
        """Calculate feature importance using different methods"""
        print(f"🔍 Calculating feature importance using {method}...")
        
        # Remove non-numeric columns and handle NaN
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_cols].fillna(0)
        y_clean = y.fillna(0)
        
        # Ensure we have valid data
        valid_mask = ~(np.isnan(X_numeric).any(axis=1) | np.isnan(y_clean))
        X_clean = X_numeric[valid_mask]
        y_clean = y_clean[valid_mask]
        
        if len(X_clean) < 10:
            print("❌ Insufficient data for feature importance calculation")
            return pd.Series(index=numeric_cols, data=0)
        
        try:
            if method == 'random_forest':
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
                rf.fit(X_clean, y_clean)
                importance = pd.Series(rf.feature_importances_, index=X_clean.columns)
            
            elif method == 'mutual_info':
                importance_values = mutual_info_regression(X_clean, y_clean, random_state=42)
                importance = pd.Series(importance_values, index=X_clean.columns)
            
            elif method == 'correlation':
                importance = abs(X_clean.corrwith(pd.Series(y_clean, index=X_clean.index)))
                importance = importance.fillna(0)
            
            return importance.sort_values(ascending=False)
            
        except Exception as e:
            print(f"❌ Error calculating feature importance: {e}")
            return pd.Series(index=numeric_cols, data=0)
    
    def select_features(self, df, target_cols, top_k=50, correlation_threshold=0.95):
        """Select best features based on importance and correlation"""
        print(f"🎯 Selecting top {top_k} features...")
        
        # Get feature columns (exclude targets and basic OHLCV)
        exclude_cols = ['open', 'high', 'low', 'close', 'volume'] + \
                      [col for col in df.columns if col.startswith('target_')]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Get numeric feature columns only
        numeric_feature_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_feature_cols) == 0:
            print("❌ No numeric features found")
            return []
        
        print(f"📊 Total features available: {len(numeric_feature_cols)}")
        
        # Calculate feature importance for main target
        main_target = None
        for target in target_cols:
            if target in df.columns and not df[target].isna().all():
                main_target = target
                break
        
        if main_target is None:
            print("❌ No valid target found for feature selection")
            return numeric_feature_cols[:top_k]
        
        # Calculate importance
        feature_importance = self.calculate_feature_importance(
            df[numeric_feature_cols], df[main_target]
        )
        
        # Remove highly correlated features
        selected_features = []
        correlation_matrix = df[numeric_feature_cols].corr().abs()
        
        for feature in feature_importance.index:
            if len(selected_features) >= top_k:
                break
                
            # Check correlation with already selected features
            if not selected_features:
                selected_features.append(feature)
            else:
                max_corr = correlation_matrix.loc[feature, selected_features].max()
                if max_corr < correlation_threshold:
                    selected_features.append(feature)
        
        print(f"✅ Selected {len(selected_features)} features after correlation filter")
        
        # Store feature importance
        self.feature_importance[main_target] = feature_importance
        
        return selected_features
    
    def scale_features(self, df, feature_cols, method='standard'):
        """Scale features using specified method"""
        print(f"⚖️ Scaling features using {method} scaler...")
        
        scaled_df = df.copy()
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            print(f"❌ Unknown scaling method: {method}")
            return scaled_df
        
        # Scale only numeric features
        numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns
        
        if len(numeric_features) > 0:
            scaled_values = scaler.fit_transform(df[numeric_features].fillna(0))
            scaled_df[numeric_features] = scaled_values
        
        return scaled_df
    
    def generate_feature_summary(self, df, symbol, selected_features):
        """Generate summary of created features"""
        print(f"\n🔧 FEATURE SUMMARY for {symbol}")
        print("=" * 60)
        
        # Count features by type
        feature_types = {
            'Price-based': [f for f in selected_features if any(x in f for x in ['sma', 'ema', 'bb', 'price'])],
            'Momentum': [f for f in selected_features if any(x in f for x in ['rsi', 'macd', 'stoch', 'roc', 'momentum'])],
            'Volume': [f for f in selected_features if any(x in f for x in ['volume', 'vwap', 'obv', 'vpt'])],
            'Volatility': [f for f in selected_features if any(x in f for x in ['atr', 'volatility', 'range'])],
            'Pattern': [f for f in selected_features if any(x in f for x in ['hammer', 'engulfing', 'doji', 'body'])],
            'Lag': [f for f in selected_features if any(x in f for x in ['lag', 'mean', 'std'])],
            'Structure': [f for f in selected_features if any(x in f for x in ['resistance', 'support', 'trend'])]
        }
        
        for feature_type, features in feature_types.items():
            if features:
                print(f"\n📊 {feature_type}: {len(features)} features")
                for feature in features[:5]:  # Show first 5
                    print(f"   • {feature}")
                if len(features) > 5:
                    print(f"   ... and {len(features) - 5} more")
        
        print(f"\n✅ Total selected features: {len(selected_features)}")
        print(f"📈 Data shape: {df.shape}")
        
        # Feature importance top 10
        target_cols = [col for col in df.columns if col.startswith('target_')]
        if target_cols and target_cols[0] in self.feature_importance:
            print(f"\n🔝 Top 10 Most Important Features:")
            top_features = self.feature_importance[target_cols[0]].head(10)
            for i, (feature, importance) in enumerate(top_features.items(), 1):
                print(f"   {i:2d}. {feature}: {importance:.4f}")
    
    def save_feature_data(self, df, symbol, format='feather'):
        """Save feature-engineered data to file"""
        try:
            clean_symbol = symbol.replace('/', '_').replace(':', '_')
            
            if format == 'feather':
                df_to_save = df.reset_index()
                filepath = os.path.join(self.output_dir, f"{clean_symbol}_features.feather")
                df_to_save.to_feather(filepath)
            elif format == 'pkl':
                filepath = os.path.join(self.output_dir, f"{clean_symbol}_features.pkl")
                df.to_pickle(filepath)
            elif format == 'csv':
                filepath = os.path.join(self.output_dir, f"{clean_symbol}_features.csv")
                df.to_csv(filepath)
            
            file_size = os.path.getsize(filepath)
            print(f"💾 Saved feature data: {os.path.basename(filepath)} ({file_size:,} bytes)")
            
            return filepath
            
        except Exception as e:
            print(f"❌ Error saving feature data for {symbol}: {e}")
            return None
    
    def process_all_files(self, top_k_features=50, scaling_method=None, output_format='feather'):
        """
        Process all labeled files and create features
        
        Args:
            top_k_features (int): Number of top features to select
            scaling_method (str): Scaling method ('standard', 'minmax', 'robust', None)
            output_format (str): Output file format
        """
        print("🚀 CRYPTO FEATURE ENGINEERING v1.0")
        print("=" * 60)
        
        # Find labeled files
        files = self.find_labeled_files()
        
        if not files:
            return
        
        successful_processing = 0
        
        for filepath in files:
            # Extract symbol from filename
            filename = os.path.basename(filepath)
            symbol = filename.split('_')[0].replace('_', '/')
            
            # Load labeled data
            df = self.load_labeled_data(filepath)
            
            if df is not None:
                # Create all features
                feature_df = df.copy()
                
                # Price-based features
                feature_df = self.create_price_features(feature_df)
                
                # Momentum features
                feature_df = self.create_momentum_features(feature_df)
                
                # Volume features
                feature_df = self.create_volume_features(feature_df)
                
                # Volatility features
                feature_df = self.create_volatility_features(feature_df)
                
                # Pattern features
                feature_df = self.create_pattern_features(feature_df)
                
                # Lag features
                feature_df = self.create_lag_features(feature_df)

                # Market structure features
                feature_df = self.create_market_structure_features(feature_df)

                # Drop rows with excessive NaNs after feature generation
                feature_df = feature_df.dropna(thresh=int(0.9 * feature_df.shape[1]))

                # Select features
                target_cols = [col for col in df.columns if col.startswith('target_')]
                selected_features = self.select_features(feature_df, target_cols, top_k=top_k_features)

                # Optionally scale features
                if scaling_method:
                    feature_df = self.scale_features(feature_df, selected_features, method=scaling_method)

                # Generate summary
                self.generate_feature_summary(feature_df, symbol, selected_features)

                # Save feature data
                self.save_feature_data(feature_df, symbol, format=output_format)

                # Store in memory
                self.feature_data[symbol] = feature_df
                successful_processing += 1

        print(f"\n✅ Done. {successful_processing}/{len(files)} files processed.")

if __name__ == "__main__":
    engine = CryptoFeatureEngineer()
    engine.process_all_files()
