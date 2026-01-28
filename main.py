import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# Import our custom modules
# Run this script with: python -m main
from src.data_processor import GraphDataEngine
from src.model import PortfolioGNN

# --- CONFIGURATION ---
TICKERS = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA', # Tech
           'JPM', 'BAC',                           # Finance
           'JNJ', 'PG',                            # Defensive
           'XOM', 'CVX']                           # Energy
START_DATE = '2019-01-01' # Includes Covid Crash
END_DATE = '2022-01-01'
WINDOW_SIZE = 30
BATCH_SIZE = 32
EPOCHS = 30 # Reduced for quicker demo
LEARNING_RATE = 0.001

# Dates to visualize the learned graph structure
PLOT_DATES = ['2020-03-20', # Peak Covid Crash
              '2021-06-01'] # Calm Bull Market

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on Device: {device}")

# --- VISUALIZATION HELPER FUNCTIONS ---
def plot_adjacency_heatmap(adj_matrix, asset_names, date_str):
    """Plots the learned dynamic graph connection strengths."""
    adj_np = adj_matrix.detach().cpu().numpy()
    plt.figure(figsize=(8, 6))
    sns.heatmap(adj_np, xticklabels=asset_names, yticklabels=asset_names, 
                cmap='viridis', vmin=0, vmax=1, annot=False)
    plt.title(f"Dynamic Graph Connections on {date_str}\n(Brighter = Stronger Link)")
    plt.tight_layout()
    plt.show()

def plot_portfolio_weights(weights_list, asset_names, dates):
    """Plots how allocation changes over time."""
    weights_np = np.concatenate(weights_list, axis=0)
    df = pd.DataFrame(weights_np, index=dates, columns=asset_names)
    
    plt.figure(figsize=(12, 6))
    df.plot.area(stacked=True, alpha=0.85, ax=plt.gca(), cmap='tab20')
    plt.title("Portfolio Allocation Weights Over Time (Stack = 100%)")
    plt.xlabel("Date")
    plt.ylabel("Weight")
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.margins(0, 0)
    plt.tight_layout()
    plt.show()

def plot_backtest(weights_list, raw_returns, dates):
    """Compares GNN Portfolio vs Equal Weight Benchmark."""
    weights_np = np.concatenate(weights_list, axis=0)
    
    # GNN Returns
    port_daily = np.sum(weights_np * raw_returns, axis=1)
    port_cum = np.cumprod(1 + port_daily) - 1
    
    # Benchmark Returns (Equal Weight)
    num_assets = raw_returns.shape[1]
    ew_weights = np.ones(num_assets) / num_assets
    ew_daily = np.sum(ew_weights * raw_returns, axis=1)
    ew_cum = np.cumprod(1 + ew_daily) - 1
    
    plt.figure(figsize=(12, 6))
    plt.plot(dates, port_cum * 100, label='GNN Regime-Adaptive Portfolio', color='blue', linewidth=2)
    plt.plot(dates, ew_cum * 100, label='Equal Weight Benchmark', color='grey', linestyle='--', linewidth=1.5)
    plt.title("Backtest Performance: Cumulative Returns")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# --- TRAINING & MAIN LOGIC ---
def sharpe_loss(weights, returns):
    # Minimize Negative Sharpe Ratio
    portfolio_ret = torch.sum(weights * returns, dim=1)
    mean_ret = torch.mean(portfolio_ret)
    volatility = torch.std(portfolio_ret) + 1e-6
    return -(mean_ret / volatility)

def run_project():
    # 1. Data Ingestion & Initial Visuals
    print("\n--- Phase 1: Data & Initial Visuals ---")
    engine = GraphDataEngine(TICKERS, START_DATE, END_DATE)
    engine.fetch_market_data()
    engine.fetch_macro_features()
    feats = engine.generate_technical_features()
    engine.build_graph_tensor(feats)
    final_data = engine.normalize_data()
    
    # SHOW INITIAL GRAPHS
    engine.plot_static_correlation()
    engine.plot_normalized_prices()
    
    # Prepare tensors
    data_tensor = torch.FloatTensor(final_data).to(device)
    # Get raw log returns for backtesting (aligned with tensor dates)
    raw_log_rets = np.log(engine.data / engine.data.shift(1)).fillna(0)
    raw_log_rets = raw_log_rets.loc[engine.dates].values

    # 2. Model Setup
    num_nodes, num_features = data_tensor.shape[1], data_tensor.shape[2]
    model = PortfolioGNN(num_nodes=num_nodes, num_features=num_features).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. Training Loop
    print("\n--- Phase 2: Training Portfolio GNN ---")
    loss_history = []
    model.train()
    num_days = data_tensor.shape[0]
    
    for epoch in range(EPOCHS):
        epoch_loss = 0
        count = 0
        for t in range(WINDOW_SIZE, num_days - 1, BATCH_SIZE):
            if t + BATCH_SIZE >= num_days: break
            
            # Create batch
            x_batch, y_batch = [], []
            for i in range(BATCH_SIZE):
                curr = t + i
                x_batch.append(data_tensor[curr-WINDOW_SIZE : curr])
                # Target is normalized return at t (which is next day relative to window end)
                y_batch.append(data_tensor[curr, :, 0]) 
                
            x_b = torch.stack(x_batch).to(device)
            y_b = torch.stack(y_batch).to(device)
            
            optimizer.zero_grad()
            weights, _ = model(x_b)
            loss = sharpe_loss(weights, y_b)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            count += 1
            
        avg_loss = epoch_loss / count
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Loss (Neg Sharpe): {avg_loss:.4f}")

    # Plot Training Loss
    plt.figure(figsize=(8, 4))
    plt.plot(loss_history, label='Training Loss')
    plt.title("Optimization Convergence (Negative Sharpe Ratio)")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(True); plt.show()

    # 4. Post-Training Analysis & Visualization
    print("\n--- Phase 3: Results & Visualizations ---")
    model.eval()
    all_weights = []
    
    # Identify indices for graph plotting dates
    plot_indices = {}
    for pd_str in PLOT_DATES:
        try:
            idx = engine.dates.get_loc(pd_str)
            if idx >= WINDOW_SIZE: plot_indices[idx] = pd_str
        except: pass

    with torch.no_grad():
        for t in range(WINDOW_SIZE, num_days, BATCH_SIZE):
            if t + BATCH_SIZE > num_days: break # Drop last incomplete batch for simplicity
            
            x_batch = []
            batch_indices = []
            for i in range(BATCH_SIZE):
                curr = t + i
                x_batch.append(data_tensor[curr-WINDOW_SIZE : curr])
                batch_indices.append(curr)

            weights, adj_matrices = model(torch.stack(x_batch).to(device))
            all_weights.append(weights.cpu().numpy())
            
            # Check if any date in this batch needs a graph plot
            for i, global_idx in enumerate(batch_indices):
                if global_idx in plot_indices:
                    plot_adjacency_heatmap(adj_matrices[i], engine.asset_names, plot_indices[global_idx])
    
    # Align dates and returns for backtest
    # Predictions start at index WINDOW_SIZE
    num_preds = len(all_weights) * BATCH_SIZE
    pred_dates = engine.dates[WINDOW_SIZE : WINDOW_SIZE + num_preds]
    pred_returns = raw_log_rets[WINDOW_SIZE : WINDOW_SIZE + num_preds]
    
    # FINAL PLOTS
    plot_portfolio_weights(all_weights, engine.asset_names, pred_dates)
    plot_backtest(all_weights, pred_returns, pred_dates)

if __name__ == "__main__":
    run_project()