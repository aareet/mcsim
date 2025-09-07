# monte_carlo_portfolio.py
# Requirements: pip install numpy pandas yfinance scipy
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from scipy.stats import norm

def download_prices(tickers, start="2015-01-01", end=None):
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")
    df = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return df.dropna(how="all").dropna(axis=1, how="any")

def compute_log_returns(price_df):
    return np.log(price_df / price_df.shift(1)).dropna()

def annualize_mean_cov(mu_daily, cov_daily, trading_days=252):
    mu_ann = mu_daily * trading_days
    cov_ann = cov_daily * trading_days
    return mu_ann, cov_ann

def simulate_paths(mu_daily, cov_daily, weights, init_value=100000.0, days=252, n_sims=10000, seed=42):
    np.random.seed(seed)
    k = len(weights)
    L = np.linalg.cholesky(cov_daily)  # correlation structure
    # Pre-generate standard normals: shape (n_sims, days, k)
    Z = np.random.normal(size=(n_sims, days, k))
    # Create correlated daily returns: r_t = mu + L z
    # Broadcast mu to (days, k)
    mu_vec = mu_daily.values  # shape (k,)
    correlated = mu_vec + np.einsum("ij,sdj->sdi", L, Z)  # (n_sims, days, k)
    # Portfolio daily log-return per scenario per day
    w = np.asarray(weights)
    port_logret = np.einsum("k,sdk->sd", w, correlated)  # (n_sims, days)
    # Convert log-returns to gross simple returns, compound to terminal wealth
    gross = np.exp(port_logret)  # per day gross return
    terminal_gross = gross.prod(axis=1)  # (n_sims,)
    terminal_values = init_value * terminal_gross
    return terminal_values, port_logret, gross

def risk_metrics(terminal_values, init_value, horizon_days):
    # Simple horizon return
    returns = terminal_values / init_value - 1.0
    exp_ret = returns.mean()
    vol = returns.std(ddof=1)
    var_5 = np.quantile(returns, 0.05)  # 5% quantile
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
        "terminal_value_median": float(np.median(terminal_values))
    }

def main():
    # User parameters
    tickers = ["AAPL", "MSFT", "TLT", "GLD"]  # example set
    weights = [0.35, 0.35, 0.20, 0.10]        # must sum to 1
    start = "2015-01-01"
    end = None
    init_value = 100000.0
    horizon_days = 252           # e.g., 1 year
    n_sims = 20000
    seed = 123

    # Checks
    w = np.array(weights, dtype=float)
    w = w / w.sum()

    # Data and estimates
    prices = download_prices(tickers, start=start, end=end)
    log_rets = compute_log_returns(prices)

    # Align weights to downloaded tickers order
    cols = list(prices.columns)
    if len(cols) != len(w):
        raise ValueError(f"Number of tickers ({len(cols)}) != number of weights ({len(w)}).")
    mu_daily = log_rets.mean()                  # vector of daily log-return means
    cov_daily = log_rets.cov()                  # daily log-return covariance matrix

    # Optional: annualized figures for reference
    mu_ann, cov_ann = annualize_mean_cov(mu_daily, cov_daily, trading_days=252)

    # Monte Carlo simulation
    terminal_values, port_logret_paths, gross_paths = simulate_paths(
        mu_daily, cov_daily, w, init_value=init_value, days=horizon_days, n_sims=n_sims, seed=seed
    )

    # Risk metrics
    metrics = risk_metrics(terminal_values, init_value, horizon_days)

    # Output
    print("Input tickers:", cols)
    print("Weights (sum=1):", w.tolist())
    print("Daily log-return mean (per asset):")
    print(mu_daily.to_string())
    print("\nAnnualized log-return mean (approx):")
    print(mu_ann.to_string())
    print("\nHorizon metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")

    # Save distributions for further analysis (optional)
    pd.DataFrame({"terminal_values": terminal_values}).to_csv("mc_terminal_values.csv", index=False)
    # Save per-scenario returns summary if desired
    # Example: final simple returns
    final_simple_returns = terminal_values / init_value - 1.0
    pd.DataFrame({"final_simple_return": final_simple_returns}).to_csv("mc_final_returns.csv", index=False)

if __name__ == "__main__":
    main()

