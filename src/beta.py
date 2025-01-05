import numpy as np
import pandas as pd

from numba import njit
from tqdm import tqdm

@njit
def serial_1d_kalman_filter(
    H: np.ndarray,      # Benchmark returns (acts as measurement "matrix")
    Z: np.ndarray,      # Asset returns (the measurements)
    x0: float = 0.5,    # Initial state (beta)
    q: float = 1e-2,    # Process noise variance Q
    r: float = 1e-4     # Measurement noise variance R
) -> np.ndarray:
    """Perform a 1D Kalman filter to estimate a time-varying state (beta).

    ## Parameters:
    
    H : np.ndarray
        1D array representing the measurement coefficient at each time step
        (e.g., benchmark returns).
    Z : np.ndarray
        1D array of observed measurements (e.g., asset returns).
    x0 : float, optional
        Initial state estimate (beta) at time 0. Default is 1.0.
    q : float, optional
        Process noise variance Q, controlling how much the state is allowed
        to change from step to step. Default is 1e-3.
    r : float, optional
        Measurement noise variance R, describing the uncertainty in each
        measurement. Default is 1e-2.

    ## Returns:
    np.ndarray: 1D array of the filtered state estimates (beta) at each time step.
"""
    T = len(H)
    x = x0
    P = 0.0
    out = np.zeros(T)
    for k in range(T):
        x_ = x
        P_ = P + q
        y = Z[k] - H[k]*x_
        S = H[k]*H[k]*P_ + r
        K = (P_*H[k])/S if S != 0 else 0.0
        x = x_ + K*y
        P = (1 - K*H[k])*P_
        out[k] = x
    return out


def kalman_beta(
    benchmark: pd.Series, 
    assets: pd.DataFrame | pd.Series,
    x0: float = 0.5,
    q: float = 1e-3,
    r: float = 1e-4,
    convergence_period: int = 0
) -> pd.DataFrame:
    """Apply the 1D Kalman filter to estimate a time-varying beta between
    a benchmark and one or more asset return series, handling each column's
    missing data individually.

    ## Parameters:
    benchmark : pd.Series
        Benchmark return series. Index is time, values are benchmark returns.
    assets : pd.Series or pd.DataFrame
        Asset return series (or multiple series). Index is time, values are returns.
        If a DataFrame, each column is treated as a separate asset.
    x0 : float, optional
        Initial beta guess. Default is 1.0.
    q : float, optional
        Process noise variance. Default is 1e-3.
    r : float, optional
        Measurement noise variance. Default is 1e-2.
    convergence_period : int, optional
        Number of initial estimates to discard (set to NaN) to allow for filter convergence.
        Default is 0 (no discarding).

    ## Returns:
    pd.DataFrame
        A DataFrame of filtered beta estimates, with the same columns as the input `assets`.
        Each column's beta is aligned to that column’s non-missing times (and any
        intersecting times in `benchmark`).
    """
    s_check = False
    if isinstance(assets, pd.Series):
        assets = assets.to_frame(name=assets.name or "asset")
        s_check = True

    df_beta = pd.DataFrame(index=assets.index, columns=assets.columns, dtype=float)
    bench_nonan = benchmark.dropna()
    
    for col in tqdm(assets.columns):
        col_series = assets[col].dropna()
        col_index = col_series.index.intersection(bench_nonan.index)

        if col_index.empty: continue
        bench_aligned = bench_nonan.loc[col_index]
        asset_aligned = col_series.loc[col_index]

        H_arr = bench_aligned.values
        Z_arr = asset_aligned.values

        betas_array = serial_1d_kalman_filter(H_arr, Z_arr, x0, q, r)
        if convergence_period > 0: betas_array[:convergence_period] = np.nan
        df_beta.loc[col_index, col] = betas_array

    if s_check: df_beta = df_beta.squeeze(axis=1)
    
    return df_beta

@njit
def compute_ols_rolling_beta(
    returns1:np.ndarray,
    returns2:np.ndarray,
    window:int=252
    ):
    """Compute rolling beta coefficients between two return series using OLS regression.
    
    ## Parameters:
    returns1 : np.ndarray
        Array of benchmark returns.
    returns2 : np.ndarray
        Array of asset returns.
    window : int, optional
        Rolling window size for calculation. Default is 126.
        
    ## Returns:
    np.ndarray
        Array of rolling beta coefficients aligned to input arrays. Values before 
        window length are set to NaN.
    """
    n = len(returns1)
    betas = np.full(n, np.nan)
    
    for i in range(window-1, n):
        x = returns1[i-window+1:i+1]
        y = returns2[i-window+1:i+1]
        
        cov = np.cov(x, y)[0,1]
        var_x = np.var(x)
        
        betas[i] = cov/var_x if var_x != 0 else np.nan
        
    return betas

def rolling_ols_beta(benchmark, assets, window=252):
    """Calculate rolling OLS betas between a benchmark and multiple asset return series.
    
    ## Parameters:
    benchmark : pd.Series
        Benchmark return series. Index is time, values are returns.
    assets : pd.Series or pd.DataFrame
        Asset return series (or multiple series). Index is time, values are returns.
        If DataFrame, each column is processed separately.
    window : int, optional
        Rolling window size for beta calculation. Default is 252.
        
    ## Returns:
    pd.DataFrame
        DataFrame of rolling beta estimates with same columns as input assets.
        Each column's beta is aligned to non-missing times in both that asset
        and benchmark series.
    """
    if isinstance(assets, pd.Series):
        assets = assets.to_frame(name=assets.name or "asset")
        
    df_beta = pd.DataFrame(index=assets.index, columns=assets.columns, dtype=float)
    
    for col in tqdm(assets.columns):
        asset_col = assets[col].dropna()
        common_idx = asset_col.index.intersection(benchmark.dropna().index)
        
        if len(common_idx) < window: 
            continue
            
        betas = compute_ols_rolling_beta(
            benchmark.loc[common_idx].values, 
            asset_col.loc[common_idx].values, 
            window
        )
        df_beta.loc[common_idx, col] = betas
        
    return df_beta.dropna(how="all")