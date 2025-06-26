import os
import ssl
import certifi
import time
import pandas as pd
import ccxt
import requests
import json
from datetime import datetime, timedelta

# 🔐 Set environment SSL certificate
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

def test_internet_connection():
    """Test basic internet connectivity"""
    try:
        response = requests.get('https://httpbin.org/ip', timeout=10)
        print(f"✅ Internet OK - Your IP: {response.json().get('origin', 'Unknown')}")
        return True
    except Exception as e:
        print(f"❌ Internet connection failed: {e}")
        return False

def test_binance_api_direct():
    """Test Binance API directly with requests"""
    try:
        print("🔍 Testing Binance API directly...")
        
        # Test ping endpoint
        ping_url = "https://api.binance.com/api/v3/ping"
        response = requests.get(ping_url, timeout=15)
        
        if response.status_code == 200:
            print("✅ Binance API ping successful")
            
            # Test server time
            time_url = "https://api.binance.com/api/v3/time"
            time_response = requests.get(time_url, timeout=15)
            
            if time_response.status_code == 200:
                server_time = time_response.json()
                print(f"✅ Binance server time: {datetime.fromtimestamp(server_time['serverTime']/1000)}")
                return True
            else:
                print(f"❌ Server time failed: {time_response.status_code}")
                return False
        else:
            print(f"❌ Binance ping failed: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectTimeout:
        print("❌ Connection timeout - Check firewall/proxy settings")
        return False
    except requests.exceptions.SSLError as e:
        print(f"❌ SSL Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Direct API test failed: {e}")
        return False

def fetch_binance_data_direct(symbol, interval='1h', limit=200):
    """Fetch data directly from Binance API using requests"""
    try:
        # Convert symbol format (OP/USDT -> OPUSDT)
        binance_symbol = symbol.replace('/', '')
        
        url = "https://api.binance.com/api/v3/klines"
        params = {
            'symbol': binance_symbol,
            'interval': interval,
            'limit': limit
        }
        
        print(f"📡 Fetching {symbol} directly from Binance...")
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'count', 'taker_buy_volume', 
                'taker_buy_quote_volume', 'ignore'
            ])
            
            # Keep only needed columns and convert types
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
            df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
            df['symbol'] = symbol
            
            print(f"✅ Successfully fetched {len(df)} records for {symbol}")
            return df
            
        else:
            print(f"❌ API request failed for {symbol}: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error fetching {symbol}: {e}")
        return None

def create_ccxt_exchange_alternative():
    """Try alternative exchanges if Binance fails"""
    exchanges_to_try = [
        ('binance', ccxt.binance),
        ('bybit', ccxt.bybit),
        ('okx', ccxt.okx),
        ('kucoin', ccxt.kucoin)
    ]
    
    for name, exchange_class in exchanges_to_try:
        try:
            print(f"🔄 Trying {name.upper()} exchange...")
            
            if name == 'binance':
                exchange = exchange_class({
                    'timeout': 30000,
                    'enableRateLimit': True,
                    'options': {'defaultType': 'spot'},
                    'headers': {'User-Agent': 'Mozilla/5.0'}
                })
            else:
                exchange = exchange_class({
                    'timeout': 30000,
                    'enableRateLimit': True,
                    'options': {'defaultType': 'spot'}
                })
            
            # Test connection
            markets = exchange.load_markets()
            print(f"✅ {name.upper()} connection successful! {len(markets)} markets available")
            return exchange, name
            
        except Exception as e:
            print(f"❌ {name.upper()} failed: {str(e)[:100]}...")
            continue
    
    return None, None

def save_data(df, symbol, source="binance"):
    """Save DataFrame to CSV file"""
    try:
        # Get current script directory (engine folder)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Go up one level to parent directory (where data folder should be)
        parent_dir = os.path.dirname(script_dir)
        
        # Create data directory in parent folder
        data_dir = os.path.join(parent_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

        filename = symbol.replace('/', '_').replace(':', '_')
        filepath = os.path.join(data_dir, f"{filename}_{source}.csv")

        df.to_csv(filepath, index=False)
        print(f"💾 Saved {symbol} data to {filepath}")
        
        # Show stats
        latest_price = df['close'].iloc[-1]
        price_change = ((latest_price - df['close'].iloc[0]) / df['close'].iloc[0]) * 100
        
        print(f"📊 Latest price: ${latest_price:.4f}")
        print(f"📊 Price change: {price_change:+.2f}%")
        print(f"📊 Volume avg: {df['volume'].mean():.0f}")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving {symbol}: {e}")
        return False


def main():
    """Main function with multiple fallback strategies"""
    print("🚀 Advanced Altcoin Data Fetcher v2.0")
    print("=" * 60)
    
    # Step 1: Test internet connection
    if not test_internet_connection():
        print("❌ No internet connection. Please check your network.")
        return
    
    # Step 2: Test Binance API directly
    binance_direct_works = test_binance_api_direct()
    
    # Altcoins to fetch
    coins = ['OP/USDT', 'AR/USDT', 'SUI/USDT']
    successful_downloads = 0
    
    print(f"\n📈 Target coins: {', '.join(coins)}")
    print("=" * 60)
    
    # Strategy 1: Direct API calls if available
    if binance_direct_works:
        print("🎯 Strategy 1: Using direct Binance API calls")
        print("-" * 40)
        
        for i, symbol in enumerate(coins, 1):
            print(f"[{i}/{len(coins)}] Processing {symbol}...")
            
            df = fetch_binance_data_direct(symbol)
            
            if df is not None:
                if save_data(df, symbol, "binance_direct"):
                    successful_downloads += 1
            
            if i < len(coins):
                print("⏳ Waiting 2 seconds...")
                time.sleep(2)
            print("-" * 30)
    
    # Strategy 2: Try CCXT with alternative exchanges
    if successful_downloads == 0:
        print("🎯 Strategy 2: Trying CCXT with alternative exchanges")
        print("-" * 40)
        
        exchange, exchange_name = create_ccxt_exchange_alternative()
        
        if exchange:
            for i, symbol in enumerate(coins, 1):
                try:
                    print(f"[{i}/{len(coins)}] Processing {symbol} on {exchange_name.upper()}...")
                    
                    if symbol in exchange.markets:
                        ohlcv = exchange.fetch_ohlcv(symbol, '1h', limit=200)
                        
                        if ohlcv:
                            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                            df['symbol'] = symbol
                            
                            if save_data(df, symbol, exchange_name):
                                successful_downloads += 1
                        else:
                            print(f"❌ No data returned for {symbol}")
                    else:
                        print(f"❌ {symbol} not available on {exchange_name.upper()}")
                        
                except Exception as e:
                    print(f"❌ Error with {symbol}: {e}")
                
                if i < len(coins):
                    print("⏳ Waiting 2 seconds...")
                    time.sleep(2)
                print("-" * 30)
    
    # Final summary
    print("=" * 60)
    print("📊 FINAL SUMMARY:")
    print(f"✅ Successfully downloaded: {successful_downloads}/{len(coins)} coins")
    print(f"❌ Failed downloads: {len(coins) - successful_downloads}/{len(coins)} coins")
    
    if successful_downloads > 0:
        print(f"💾 Data files saved in 'data/' directory")
        print("📁 Files created:")
        
        try:
            # Update path for checking files
            script_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(script_dir)
            data_dir = os.path.join(parent_dir, "data")
            
            if os.path.exists(data_dir):
                for file in os.listdir(data_dir):
                    if file.endswith('.csv'):
                        size = os.path.getsize(os.path.join(data_dir, file))
                        print(f"   📄 {file} ({size:,} bytes)")
        except Exception as e:
            print(f"   ❌ Error listing files: {e}")
    else:
        print("\n🔧 TROUBLESHOOTING SUGGESTIONS:")
        print("1. Check your internet connection")
        print("2. Disable VPN if using one")
        print("3. Check Windows Firewall/Antivirus settings")
        print("4. Try running as Administrator")
        print("5. Check if your ISP blocks cryptocurrency APIs")
    
    print("\n🎉 Process completed!")

if __name__ == "__main__":
    main()