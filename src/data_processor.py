import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class GraphDataEngine:
    """
    Phase 1 Engine: Fetches, processes, and now VISUALIZES data.
    """
    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = None          
        self.macro_data = None    
        self.feature_tensor = None 
        self.asset_names = None
        self.dates = None
        
    def fetch_market_data(self):
        print(f"--- [1/5] Fetching data for {len(self.tickers)} assets ---")
        raw_data = yf.download(self.tickers, start=self.start_date, end=self.end_date, auto_adjust=True)
        if 'Close' in raw_data.columns:
            self.data = raw_data['Close']
        else:
            self.data = raw_data
        self.data = self.data.dropna(axis=1, thresh=int(len(self.data)*0.97)).ffill().bfill()
        self.asset_names = self.data.columns.tolist()
        print(f"✔ Assets Loaded: {len(self.asset_names)} | Shape: {self.data.shape}")
        
    def fetch_macro_features(self):
        print("--- [2/5] Fetching Macro Context ---")
        macro_tickers = ['^VIX', '^TNX'] 
        macro = yf.download(macro_tickers, start=self.start_date, end=self.end_date, auto_adjust=True)
        if isinstance(macro.columns, pd.MultiIndex):
            try: macro = macro['Close'] 
            except: pass 
        self.macro_data = macro.ffill().bfill()
        print(f"✔ Macro Data Loaded | Shape: {self.macro_data.shape}")
        
    def generate_technical_features(self, window=20):
        print("--- [3/5] Generating Technical Indicators ---")
        feature_map = {}
        feature_map['log_ret'] = np.log(self.data / self.data.shift(1)).fillna(0)
        feature_map['volatility'] = feature_map['log_ret'].rolling(window=window).std().fillna(0)
        
        rsi_df = pd.DataFrame(index=self.data.index, columns=self.data.columns)
        macd_df = pd.DataFrame(index=self.data.index, columns=self.data.columns)
        for ticker in self.data.columns:
            try:
                rsi_df[ticker] = RSIIndicator(close=self.data[ticker], window=14).rsi()
                macd_df[ticker] = MACD(close=self.data[ticker]).macd_diff()
            except:
                rsi_df[ticker] = 50
                macd_df[ticker] = 0
        feature_map['rsi'] = rsi_df.fillna(50)
        feature_map['macd'] = macd_df.fillna(0)
        print("✔ Indicators Generated")
        return feature_map

    def build_graph_tensor(self, feature_map):
        print("--- [4/5] Building Graph Tensor ---")
        common_index = self.data.index.intersection(self.macro_data.index)
        self.macro_data = self.macro_data.loc[common_index]
        self.data = self.data.loc[common_index] # Align raw data too
        for key in feature_map:
            feature_map[key] = feature_map[key].loc[common_index]
            
        tech_list = [feature_map['log_ret'], feature_map['volatility'], 
                     feature_map['rsi'], feature_map['macd']]
        tensor = np.stack([df.values for df in tech_list], axis=-1)
        
        T, N, F = tensor.shape
        macro_vals = self.macro_data.values
        macro_expanded = np.tile(macro_vals[:, np.newaxis, :], (1, N, 1))
        
        final_tensor = np.concatenate([tensor, macro_expanded], axis=-1)
        
        warmup = 30
        self.feature_tensor = final_tensor[warmup:, :, :]
        self.dates = common_index[warmup:]
        self.data = self.data.iloc[warmup:] # Keep raw prices aligned
        print(f"✔ Tensor Built. Raw Shape: {self.feature_tensor.shape}")
        return self.feature_tensor

    def normalize_data(self):
        print("--- [5/5] Normalizing Data ---")
        T, N, F = self.feature_tensor.shape
        reshaped = self.feature_tensor.reshape(-1, F)
        scaler = StandardScaler()
        normalized = scaler.fit_transform(reshaped)
        self.feature_tensor = normalized.reshape(T, N, F)
        print(f"✔ Normalization Complete. Final Tensor Shape: {self.feature_tensor.shape}")
        return self.feature_tensor

    # --- NEW VISUALIZATION METHODS ---
    def plot_static_correlation(self):
        """Visualizes the average historical correlation."""
        print("Creating Static Correlation Heatmap...")
        log_rets = np.log(self.data / self.data.shift(1)).dropna()
        corr_matrix = log_rets.corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
        plt.title("Historical Static Correlation (Log Returns)")
        plt.tight_layout()
        plt.show()

    def plot_normalized_prices(self):
        """Visualizes the normalized returns to show volatility clusters."""
        print("Creating Normalized Returns Plot...")
        norm_returns = self.feature_tensor[:, :, 0] # Index 0 is log returns
        
        plt.figure(figsize=(12, 6))
        for i in range(len(self.asset_names)):
            plt.plot(self.dates, norm_returns[:, i], label=self.asset_names[i], alpha=0.5, linewidth=1)
        
        plt.title("Normalized Log Returns (Z-Scores) Over Time")
        plt.xlabel("Date")
        plt.ylabel("Normalized Return")
        plt.legend(loc='upper right', fontsize='x-small', ncol=3)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()