# monte_carlo_portfolio.py
# Requirements: pip install numpy pandas yfinance scipy openpyxl
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

def load_portfolio_from_excel(excel_file, sheet_name=None):
    """
    Load portfolio holdings from Excel file.
    Expected format:
    - Column 1: 'Ticker' or 'Symbol' (e.g., AAPL, MSFT)
    - Column 2: 'Weight' or 'Allocation' (e.g., 0.35 or 35%)
    
    Alternative formats supported:
    - 'Ticker', 'Shares', 'Price' (will calculate weights from market values)
    - 'Symbol', 'Market_Value' (will calculate relative weights)
    """
    try:
        # Read Excel file
        if sheet_name:
            df = pd.read_excel(excel_file, sheet_name=sheet_name, engine='openpyxl')
        else:
            df = pd.read_excel(excel_file, engine='openpyxl')
        
        # Clean column names (remove spaces, convert to lowercase)
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Try to identify ticker column
        ticker_cols = ['ticker', 'symbol', 'stock', 'asset']
        ticker_col = None
        for col in ticker_cols:
            if col in df.columns:
                ticker_col = col
                break
        
        if ticker_col is None:
            # Assume first column is ticker
            ticker_col = df.columns[0]
            print(f"Warning: Using first column '{ticker_col}' as ticker column")
        
        # Extract tickers
        tickers = df[ticker_col].dropna().astype(str).str.strip().str.upper().tolist()
        
        # Try to calculate weights
        weights = None
        
        # Method 1: Direct weight/allocation column
        weight_cols = ['weight', 'allocation', 'percent', 'percentage', 'allocation_%']
        for col in weight_cols:
            if col in df.columns:
                weights = df[col].dropna().values
                # Convert percentages to decimals if needed
                if np.max(weights) > 1:
                    weights = weights / 100.0
                break
        
        # Method 2: Calculate from shares and price
        if weights is None and 'shares' in df.columns and 'price' in df.columns:
            market_values = df['shares'] * df['price']
            weights = market_values / market_values.sum()
            weights = weights.values
        
        # Method 3: Use market values directly
        if weights is None:
            value_cols = ['market_value', 'value', 'amount', 'market_val']
            for col in value_cols:
                if col in df.columns:
                    market_values = df[col].dropna().values
                    weights = market_values / market_values.sum()
                    break
        
        # Method 4: Equal weights if no weight info found
        if weights is None:
            print("Warning: No weight information found. Using equal weights.")
            weights = np.ones(len(tickers)) / len(tickers)
        
        # Ensure same length
        min_len = min(len(tickers), len(weights))
        tickers = tickers[:min_len]
        weights = weights[:min_len]
        
        # Normalize weights to sum to 1
        weights = np.array(weights, dtype=float)
        weights = weights / weights.sum()
        
        print(f"Loaded {len(tickers)} holdings from Excel file:")
        for i, (ticker, weight) in enumerate(zip(tickers, weights)):
            print(f"  {ticker}: {weight:.4f} ({weight*100:.2f}%)")
        
        return tickers, weights
        
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        print("Please ensure your Excel file has the following format:")
        print("Column 1: Ticker/Symbol (AAPL, MSFT, etc.)")
        print("Column 2: Weight/Allocation (0.35 or 35%)")
        sys.exit(1)

def download_prices(tickers, start="2015-01-01", end=None):
    """Download historical prices for given tickers"""
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")
    
    print(f"Downloading price data for {len(tickers)} assets from {start} to {end}...")
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    
    if isinstance(df, pd.Series):
        df = df.to_frame()
    
    # Check for missing data
    initial_assets = len(tickers)
    df = df.dropna(how="all").dropna(axis=1, how="any")
    final_assets = len(df.columns)
    
    if final_assets < initial_assets:
        missing = set(tickers) - set(df.columns)
        print(f"Warning: {initial_assets - final_assets} assets dropped due to insufficient data: {missing}")
    
    return df

def compute_log_returns(price_df):
    """Compute daily log returns"""
    return np.log(price_df / price_df.shift(1)).dropna()

def annualize_mean_cov(mu_daily, cov_daily, trading_days=252):
    """Annualize daily statistics"""
    mu_ann = mu_daily * trading_days
    cov_ann = cov_daily * trading_days
    return mu_ann, cov_ann

def simulate_paths(mu_daily, cov_daily, weights, init_value=100000.0, days=252, n_sims=10000, seed=42):
    """Run Monte Carlo simulation"""
    np.random.seed(seed)
    k = len(weights)
    L = np.linalg.cholesky(cov_daily)  # Cholesky decomposition for correlation
    
    # Generate correlated random returns
    Z = np.random.normal(size=(n_sims, days, k))
    mu_vec = mu_daily.values
    correlated = mu_vec + np.einsum("ij,sdj->sdi", L, Z)
    
    # Portfolio daily log-return per scenario
    w = np.asarray(weights)
    port_logret = np.einsum("k,sdk->sd", w, correlated)
    
    # Convert to terminal values
    gross = np.exp(port_logret)
    terminal_gross = gross.prod(axis=1)
    terminal_values = init_value * terminal_gross
    
    return terminal_values, port_logret, gross

def risk_metrics(terminal_values, init_value, horizon_days):
    """Calculate risk metrics"""
    returns = terminal_values / init_value - 1.0
    exp_ret = returns.mean()
    vol = returns.std(ddof=1)
    var_5 = np.quantile(returns, 0.05)
    cvar_5 = returns[returns <= var_5].mean() if (returns <= var_5).any() else var_5
    prob_loss = (returns < 0).mean()
    
    return {
        "horizon_days": horizon_days,
        "expected_return": float(exp_ret),
        "volatility": float(vol),
        "VaR_5pct": float(var_5),
        "CVaR_5pct": float(cvar_5),
        "probability_loss": float(prob_loss),
        "terminal_value_mean": float(terminal_values.mean()),
        "terminal_value_median": float(np.median(terminal_values)),
        "terminal_value_max": float(np.max(terminal_values)),
        "terminal_value_min": float(np.min(terminal_values)),
        "std_terminal_value": float(terminal_values.std())
    }

def create_sample_excel():
    """Create a sample Excel file for reference"""
    sample_data = {
        'Ticker': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'TLT', 'GLD'],
        'Weight': [0.25, 0.25, 0.20, 0.15, 0.10, 0.05]
    }
    
    df = pd.DataFrame(sample_data)
    filename = 'sample_portfolio.xlsx'
    df.to_excel(filename, index=False, engine='openpyxl')
    print(f"Created sample Excel file: {filename}")
    return filename

def plot_histogram(terminal_values, init_value, mean_value, median_value, var_5pct_value, output="histogram.png"):
    plt.figure(figsize=(9,6))
    sns.histplot(terminal_values, bins=50, color='navy', alpha=0.7)
    plt.axvline(init_value, color='black', linestyle='--', lw=1.2, label='Initial Value')
    plt.axvline(mean_value, color='blue', linestyle='-', lw=2, label='Mean')
    plt.axvline(median_value, color='purple', linestyle='-', lw=2, label='Median')
    plt.axvline(var_5pct_value, color='red', linestyle='-', lw=2, label='5% VaR')
    plt.xlabel('Terminal Portfolio Value ($)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Monte Carlo Simulated Portfolio Values')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output)
    plt.close()

def plot_paths_confidence_bands(port_logret_paths, init_value, output="portfolio_evolution.png"):
    n_paths, n_days = min(20, len(port_logret_paths)), port_logret_paths.shape[1]
    gross_paths = np.exp(port_logret_paths[:n_paths])
    path_values = np.column_stack([np.full(n_paths, init_value), init_value * gross_paths.cumprod(axis=1)])
    days = np.arange(n_days+1)
    for i in range(n_paths):
        plt.plot(days, path_values[i], color='steelblue', lw=0.8, alpha=0.8)
    # Percentile bands
    all_paths = np.exp(port_logret_paths)
    all_values = np.column_stack([np.full(all_paths.shape[0], init_value), init_value * all_paths.cumprod(axis=1)])
    percentiles = np.percentile(all_values, [5,25,50,75,95], axis=0)
    plt.fill_between(days, percentiles[0], percentiles[4], color='navy', alpha=0.12, label='5-95% band')
    plt.fill_between(days, percentiles[1], percentiles[3], color='navy', alpha=0.22, label='25-75% band')
    plt.plot(days, percentiles[2], color='red', lw=2, label='Median')
    plt.axhline(init_value, color='black', linestyle='--', lw=1.2, label='Initial Value')
    plt.xlabel('Trading Days')
    plt.ylabel('Portfolio Value ($)')
    plt.title('Simulated Portfolio Value Evolution with Confidence Bands')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output)
    plt.close()

def plot_risk_bar(metrics_dict, output="risk_metrics.png"):
    risk_data = [
        ("Expected Return", metrics_dict["expected_return"], "Return"),
        ("Volatility (Std Dev)", metrics_dict["volatility"], "Risk"),
        ("Value at Risk (5%)", metrics_dict["VaR_5pct"], "Risk"),
        ("Conditional VaR (5%)", metrics_dict["CVaR_5pct"], "Risk"),
        ("Probability of Loss", metrics_dict["probability_loss"], "Risk"),
        ("Sharpe Ratio (approx)", metrics_dict["expected_return"]/metrics_dict["volatility"] if metrics_dict["volatility"]>0 else 0, "Risk-Adjusted"),
        ("Maximum Gain", metrics_dict["terminal_value_max"] / 100000.0 - 1, "Return"),
        ("Maximum Loss", metrics_dict["terminal_value_min"] / 100000.0 - 1, "Risk"),
    ]
    df = pd.DataFrame(risk_data, columns=['Metric', 'Value', 'Category'])
    palette = {'Return':'royalblue', 'Risk':'crimson', 'Risk-Adjusted':'forestgreen'}
    plt.figure(figsize=(9,6))
    metric_fmt = []
    for m in df['Metric']:
        if "Ratio" in m: metric_fmt.append("{:.2f}")
        else: metric_fmt.append("{:.2%}")
    bars = sns.barplot(data=df, y='Metric', x='Value', hue='Category', palette=palette, dodge=False, orient='h')
    for i, (val, fmt) in enumerate(zip(df['Value'], metric_fmt)):
        plt.text(val if val>0 else 0, i, fmt.format(val), va='center', ha='left' if val>0 else 'right',
                 fontsize=10, color='black', weight='bold')
    plt.title('Monte Carlo Portfolio Risk Analysis')
    plt.xlabel('Value')
    plt.legend(title='Category')
    plt.tight_layout()
    plt.savefig(output)
    plt.close()

def main():
    # Configuration
    excel_file = "portfolio_holdings.xlsx"  # Change this to your Excel file path
    sheet_name = None  # Use first sheet by default, or specify sheet name
    start_date = "2015-01-01"
    end_date = None
    init_value = 100000.0
    horizon_days = 252
    n_sims = 20000
    seed = 123
    
    print("=== Monte Carlo Portfolio Simulation ===\n")
    
    # Check if Excel file exists
    if not os.path.exists(excel_file):
        print(f"Excel file '{excel_file}' not found.")
        create_sample = input("Create a sample Excel file? (y/n): ").lower().strip()
        if create_sample == 'y':
            excel_file = create_sample_excel()
        else:
            print("Please create an Excel file with your portfolio holdings and try again.")
            sys.exit(1)
    
    # Load portfolio from Excel
    tickers, weights = load_portfolio_from_excel(excel_file, sheet_name)
    
    # Download price data
    prices = download_prices(tickers, start=start_date, end=end_date)
    
    # Align weights with available data
    available_tickers = list(prices.columns)
    if len(available_tickers) != len(tickers):
        print("Adjusting weights for available data...")
        # Map weights to available tickers
        ticker_weight_map = dict(zip(tickers, weights))
        aligned_weights = []
        for ticker in available_tickers:
            if ticker in ticker_weight_map:
                aligned_weights.append(ticker_weight_map[ticker])
            else:
                aligned_weights.append(0.0)
        
        # Renormalize
        aligned_weights = np.array(aligned_weights)
        if aligned_weights.sum() > 0:
            aligned_weights = aligned_weights / aligned_weights.sum()
            weights = aligned_weights
        else:
            print("Error: No valid tickers found in price data")
            sys.exit(1)
    
    # Calculate returns and statistics
    log_rets = compute_log_returns(prices)
    mu_daily = log_rets.mean()
    cov_daily = log_rets.cov()
    mu_ann, cov_ann = annualize_mean_cov(mu_daily, cov_daily)
    
    # Run Monte Carlo simulation
    print(f"\nRunning Monte Carlo simulation...")
    print(f"Simulations: {n_sims:,}")
    print(f"Horizon: {horizon_days} days")
    print(f"Initial value: ${init_value:,.2f}")
    
    terminal_values, port_logret_paths, gross_paths = simulate_paths(
        mu_daily, cov_daily, weights, 
        init_value=init_value, days=horizon_days, n_sims=n_sims, seed=seed
    )
    
    # Calculate and display results
    metrics = risk_metrics(terminal_values, init_value, horizon_days)
    
    print(f"\n=== SIMULATION RESULTS ===")
    print(f"Portfolio Assets: {available_tickers}")
    print(f"Final Weights: {[f'{w:.4f}' for w in weights]}")
    print(f"\nDaily Log-Return Statistics:")
    for asset, ret in mu_daily.items():
        print(f"  {asset}: {ret:.6f} ({ret*252:.4f} annualized)")
    
    print(f"\nHorizon Risk Metrics ({horizon_days} days):")
    for key, value in metrics.items():
        if isinstance(value, float):
            if 'return' in key or 'var' in key.lower():
                print(f"  {key}: {value:.4f} ({value*100:.2f}%)")
            elif 'value' in key:
                print(f"  {key}: ${value:,.2f}")
            else:
                print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    
    # Save results
    results_df = pd.DataFrame({
        'terminal_values': terminal_values,
        'simple_returns': terminal_values / init_value - 1.0
    })
    results_df.to_csv('monte_carlo_results.csv', index=False)
    
    # Save portfolio summary
    portfolio_summary = pd.DataFrame({
        'Ticker': available_tickers,
        'Weight': weights,
        'Daily_Mean_Return': mu_daily.values,
        'Annualized_Mean_Return': mu_ann.values
    })
    portfolio_summary.to_csv('portfolio_summary.csv', index=False)

    # Generate charts
    pd.DataFrame({'terminal_values': terminal_values, 'simple_returns': terminal_values/init_value-1.0}).to_csv('monte_carlo_results.csv', index=False)
    
    plot_histogram(
        terminal_values, init_value,
        metrics["terminal_value_mean"], metrics["terminal_value_median"],
        metrics["VaR_5pct"]*init_value + init_value,
        output="histogram.png"
    )
    plot_paths_confidence_bands(port_logret_paths, init_value, output="portfolio_evolution.png")
    plot_risk_bar(metrics, output="risk_metrics.png")
    print("Charts saved: histogram.png, portfolio_evolution.png, risk_metrics.png")
    
    print(f"\nResults saved to:")
    print(f"  - monte_carlo_results.csv (simulation outcomes)")
    print(f"  - portfolio_summary.csv (portfolio details)")

if __name__ == "__main__":
    main()

