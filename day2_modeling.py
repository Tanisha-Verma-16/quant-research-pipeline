"""
Quantitative Research Pipeline - Day 2
ML Modeling & Monte Carlo Simulation

Morning: Volatility Regime Prediction (XGBoost)
Afternoon: Monte Carlo Simulation & Performance Metrics

Deliverables:
- Trained regime classifier
- 1000-path Monte Carlo simulation
- Sharpe Ratio, Maximum Drawdown, Information Ratio
- Risk-adjusted performance tearsheet
"""

import polars as pl
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import json

# ML imports
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb

# Performance metrics
from scipy import stats


class VolatilityRegimeClassifier:
    """
    XGBoost classifier for volatility regime prediction.
    
    Predicts: LOW_VOL, NORMAL_VOL, HIGH_VOL
    Use case: Switch strategies based on predicted regime
    """
    
    def __init__(self, data_path: str = "data/alpha_factors_full.parquet"):
        """Load feature-rich dataset from Day 1."""
        print("="*80)
        print("DAY 2 MORNING: VOLATILITY REGIME PREDICTION")
        print("XGBoost Classification | Walk-Forward Cross-Validation")
        print("="*80 + "\n")
        
        self.df = pl.read_parquet(data_path)
        print(f"📂 Loaded: {len(self.df):,} rows, {len(self.df.columns)} cols")
        print(f"   Tickers: {self.df['Ticker'].n_unique()}")
        print(f"   Date range: {self.df['Date'].min()} → {self.df['Date'].max()}\n")
        
        self.model = None
        self.feature_names = None
        
    def prepare_ml_dataset(self, forecast_horizon: int = 21) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare features and target for ML.
        
        CRITICAL: We predict FUTURE regime (not current) to avoid data leakage.
        This makes the model actionable - we can switch strategies before regime changes.
        
        Features: All technical indicators, momentum, mean reversion
        Target: vol_regime shifted forward by forecast_horizon days
        
        Args:
            forecast_horizon: Days ahead to predict (21 = ~1 month)
        
        Returns:
            X: Feature matrix
            y: Target labels (future regime)
            feature_names: List of feature names
        """
        print("🔧 Preparing ML Dataset...")
        print(f"   Forecast Horizon: {forecast_horizon} days ahead (prevents data leakage)\n")
        
        # Select features (exclude target, metadata, prices)
        # exclude_cols = [
        #     'Date', 'Ticker', 'AssetClass', 'Open', 'High', 'Low', 'Close', 'Volume',
        #     'vol_regime',  # This is our target (but we'll shift it forward)
        #     'forward_return',  # Don't use future returns as features!
        # ]
        # Select features (exclude target, metadata, prices)
        exclude_cols = [
            'Date', 'Ticker', 'AssetClass', 
            'Open', 'High', 'Low', 'Close', 'Volume',
            'vol_regime',     # The CURRENT regime (too similar to future)
            'future_regime',  # The ACTUAL target (must be excluded from X!)
            'forward_return', # Future data point
            'target',         # Any other potential target names
            'next_vol'        # Any intermediate columns you might have created
        ]
        
        # PRO TIP: Also exclude features that are "too good to be true"
        # If you have a feature like "volatility_ratio", make sure it's 
        # calculated using only PAST data.
        
        feature_cols = [c for c in self.df.columns if c not in exclude_cols]
        
        # Filter to only numeric features
        numeric_features = []
        for col in feature_cols:
            dtype = self.df[col].dtype
            if dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8]:
                numeric_features.append(col)
        
        print(f"   Selected {len(numeric_features)} numeric features")
        
        # CRITICAL FIX: Create forward-looking target to avoid data leakage
        # We want to predict regime 21 days from now using today's features
        df_with_target = self.df.with_columns([
            pl.col("vol_regime").shift(-forecast_horizon).over("Ticker").alias("future_regime")
        ])
        
        # Drop rows with nulls in target or features
        df_clean = df_with_target.select(numeric_features + ['future_regime']).drop_nulls()
        
        # Convert to pandas for easier inf handling
        df_pd = df_clean.to_pandas()
        
        # Replace inf with NaN
        df_pd = df_pd.replace([np.inf, -np.inf], np.nan)
        
        # Drop rows with NaN
        df_pd = df_pd.dropna()
        
        print(f"   Clean dataset: {len(df_pd):,} rows (after removing inf/nan)")
        
        # Convert to numpy
        X = df_pd[numeric_features].to_numpy()
        y = df_pd['future_regime'].to_numpy()
        
        # Additional sanity check
        assert not np.any(np.isnan(X)), "NaN values still present in X"
        assert not np.any(np.isinf(X)), "Inf values still present in X"
        
        # Encode labels
        label_map = {'LOW_VOL': 0, 'NORMAL_VOL': 1, 'HIGH_VOL': 2}
        y_encoded = np.array([label_map[label] for label in y])
        
        print(f"   Feature matrix: {X.shape}")
        print(f"   Target distribution ({forecast_horizon}-day ahead):")
        unique, counts = np.unique(y, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"     {label:12s}: {count:6,} ({count/len(y)*100:5.1f}%)")
        print()
        
        return X, y_encoded, numeric_features
    
    # def train_with_walk_forward_cv(self, X: np.ndarray, y: np.ndarray, 
    #                                feature_names: List[str], n_splits: int = 5):
    #     """
    #     Train with K-fold Walk-Forward Cross-Validation.
        
    #     This is THE proper way to validate time-series models:
    #     - Train on past data
    #     - Test on future data
    #     - No data leakage between folds
        
    #     Args:
    #         X: Feature matrix
    #         y: Target labels
    #         feature_names: Feature names
    #         n_splits: Number of CV folds (default: 5)
    #     """
    #     print(f"🎓 Walk-Forward Cross-Validation ({n_splits} folds)...")
    #     print("   This prevents overfitting by testing on unseen future data\n")
        
    #     # Time series split
    #     tscv = TimeSeriesSplit(n_splits=n_splits)
        
    #     cv_scores = []
    #     fold = 1
        
    #     for train_idx, test_idx in tscv.split(X):
    #         X_train, X_test = X[train_idx], X[test_idx]
    #         y_train, y_test = y[train_idx], y[test_idx]
            
    #         # Train XGBoost
    #         dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    #         dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)
            
    #         params = {
    #             'objective': 'multi:softmax',
    #             'num_class': 3,
    #             'max_depth': 6,
    #             'eta': 0.1,
    #             'subsample': 0.8,
    #             'colsample_bytree': 0.8,
    #             'eval_metric': 'mlogloss',
    #             'seed': 42
    #         }
            
    #         model = xgb.train(
    #             params,
    #             dtrain,
    #             num_boost_round=100,
    #             evals=[(dtest, 'test')],
    #             verbose_eval=False
    #         )
            
    #         # Predict
    #         y_pred = model.predict(dtest)
    #         accuracy = accuracy_score(y_test, y_pred)
    #         cv_scores.append(accuracy)
            
    #         print(f"   Fold {fold}/{n_splits}: Accuracy = {accuracy:.4f} "
    #               f"(train: {len(X_train):,}, test: {len(X_test):,})")
    #         fold += 1
        
    #     print(f"\n   Mean CV Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    #     print()
        
    #     # Train final model on all data
    #     print("🚀 Training Final Model (all data)...")
        
    #     dtrain = xgb.DMatrix(X, label=y, feature_names=feature_names)
        
    #     self.model = xgb.train(
    #         params,
    #         dtrain,
    #         num_boost_round=150,
    #         verbose_eval=False
    #     )
        
    #     self.feature_names = feature_names
        
    #     print("   ✅ Model trained successfully\n")
        
    #     return cv_scores
    
    def train_with_walk_forward_cv(self, X: np.ndarray, y: np.ndarray, 
                                   feature_names: List[str], n_splits: int = 5):
        """
        Train with K-fold Walk-Forward Cross-Validation using a PURGED GAP.
        
        The 'gap' ensures that the model doesn't use data from the 21-day 
        prediction horizon of the training set to predict the start of the test set.
        """
        print(f"🎓 Walk-Forward Cross-Validation ({n_splits} folds)...")
        # FIX: Added gap=21 to match your forecast_horizon
        print("   CRITICAL: Using a 21-day PURGE GAP to prevent data leakage.\n")
        
        # Time series split with the 21-day gap
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=21)
        
        cv_scores = []
        fold = 1
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train XGBoost
            dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
            dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)
            
            # params = {
            #     'objective': 'multi:softmax',
            #     'num_class': 3,
            #     'max_depth': 4, # Slightly reduced depth to further prevent overfitting
            #     'eta': 0.05,    # Slower learning rate for more robustness
            #     'subsample': 0.7,
            #     'colsample_bytree': 0.7,
            #     'eval_metric': 'mlogloss',
            #     'seed': 42
            # }
            params = {
                'objective': 'multi:softmax',
                'num_class': 3,
                'max_depth': 3,          # Reduced from 6: Forces simpler logic
                'learning_rate': 0.01,   # Reduced from 0.1: Much more conservative
                'subsample': 0.6,        # Use only 60% of data per tree
                'colsample_bytree': 0.6, # Use only 60% of features per tree
                'min_child_weight': 5,   # Prevents trees from creating nodes for 1-2 outliers
                'eval_metric': 'mlogloss',
                'seed': 42
            }
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=100,
                evals=[(dtest, 'test')],
                early_stopping_rounds=10, # Stop if test performance plateaus
                verbose_eval=False
            )
            
            # Predict
            y_pred = model.predict(dtest)
            accuracy = accuracy_score(y_test, y_pred)
            cv_scores.append(accuracy)
            
            print(f"   Fold {fold}/{n_splits}: Accuracy = {accuracy:.4f} "
                  f"(train: {len(X_train):,}, test: {len(X_test):,})")
            fold += 1
        
        print(f"\n   Mean Purged CV Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        
        # Train final model on all data
        # Note: In a real prod environment, we'd use the most recent data as a final holdout
        dtrain = xgb.DMatrix(X, label=y, feature_names=feature_names)
        self.model = xgb.train(params, dtrain, num_boost_round=150, verbose_eval=False)
        self.feature_names = feature_names
        
        return cv_scores

    def analyze_feature_importance(self):
        """
        Analyze which features matter most for regime prediction.
        """
        print("📊 Feature Importance Analysis...")
        print("-"*80)
        
        importance = self.model.get_score(importance_type='gain')
        
        # Sort by importance
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        print("\nTop 15 Most Important Features:")
        for i, (feat, score) in enumerate(sorted_features[:15], 1):
            print(f"  {i:2d}. {feat:35s}: {score:8.1f}")
        
        print()
        
        return importance
    
    def save_model(self, output_dir: str = "models"):
        """Save trained model."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        model_path = f"{output_dir}/regime_classifier.json"
        self.model.save_model(model_path)
        
        # Save feature names
        with open(f"{output_dir}/feature_names.json", 'w') as f:
            json.dump(self.feature_names, f)
        
        print(f"💾 Model saved to {model_path}\n")


class MonteCarloSimulator:
    """
    Monte Carlo simulation for portfolio risk analysis.
    
    Simulates 1,000 possible future paths based on:
    - Historical return distribution
    - Volatility clustering
    - Regime-dependent dynamics
    """
    
    def __init__(self, data_path: str = "data/alpha_factors_full.parquet"):
        """Load data for simulation."""
        print("="*80)
        print("DAY 2 AFTERNOON: MONTE CARLO SIMULATION")
        print("1000-Path Portfolio Simulation | Risk Metrics")
        print("="*80 + "\n")
        
        self.df = pl.read_parquet(data_path)
        print(f"📂 Loaded: {len(self.df):,} rows")
        print()
        
    def estimate_return_parameters(self, ticker: str = "^GSPC") -> Dict[str, float]:
        """
        Estimate return distribution parameters for simulation.
        
        Uses historical data to estimate:
        - Mean daily return (μ)
        - Volatility (σ)
        - Skewness
        - Kurtosis (fat tails)
        
        Args:
            ticker: Ticker to analyze (default: S&P 500)
        """
        print(f"📊 Estimating Parameters for {ticker}...")
        
        ticker_data = self.df.filter(pl.col("Ticker") == ticker)
        returns = ticker_data["log_return"].drop_nulls().to_numpy()
        
        params = {
            'mean': np.mean(returns),
            'std': np.std(returns),
            'skew': stats.skew(returns),
            'kurtosis': stats.kurtosis(returns)
        }
        
        print(f"   Daily Return: {params['mean']*100:.4f}%")
        print(f"   Daily Volatility: {params['std']*100:.4f}%")
        print(f"   Annualized Return: {params['mean']*252*100:.2f}%")
        print(f"   Annualized Volatility: {params['std']*np.sqrt(252)*100:.2f}%")
        print(f"   Skewness: {params['skew']:.4f}")
        print(f"   Excess Kurtosis: {params['kurtosis']:.4f}")
        print()
        
        return params
    
    def run_monte_carlo(self, params: Dict[str, float], 
                       initial_value: float = 10000,
                       days: int = 252,
                       n_simulations: int = 1000,
                       use_regime_switching: bool = True) -> np.ndarray:
        """
        Run Monte Carlo simulation.
        
        Args:
            params: Return distribution parameters
            initial_value: Starting portfolio value ($10,000)
            days: Simulation horizon (252 = 1 year)
            n_simulations: Number of paths (1,000)
            use_regime_switching: Include volatility regime switching
            
        Returns:
            paths: Array of shape (n_simulations, days+1)
        """
        print(f"🎲 Running Monte Carlo Simulation...")
        print(f"   Simulations: {n_simulations:,}")
        print(f"   Horizon: {days} days ({days/252:.1f} years)")
        print(f"   Initial Value: ${initial_value:,.0f}")
        print(f"   Regime Switching: {use_regime_switching}")
        print()
        
        paths = np.zeros((n_simulations, days + 1))
        paths[:, 0] = initial_value
        
        # Simulate
        for i in range(n_simulations):
            for t in range(1, days + 1):
                # Random return from normal distribution
                if use_regime_switching:
                    # Volatility clustering: higher vol follows higher vol
                    if t > 20:
                        recent_vol = np.std(np.log(paths[i, t-20:t] / paths[i, t-21:t-1]))
                        vol_adjustment = recent_vol / params['std']
                        current_std = params['std'] * vol_adjustment
                    else:
                        current_std = params['std']
                else:
                    current_std = params['std']
                
                # Generate return
                random_return = np.random.normal(params['mean'], current_std)
                
                # Update path (log returns -> prices)
                paths[i, t] = paths[i, t-1] * np.exp(random_return)
        
        print(f"   ✅ Simulation complete\n")
        
        return paths
    
    def calculate_risk_metrics(self, paths: np.ndarray, 
                               initial_value: float = 10000) -> Dict[str, float]:
        """
        Calculate comprehensive risk metrics.
        
        Metrics:
        - Sharpe Ratio
        - Maximum Drawdown
        - Value at Risk (VaR)
        - Conditional VaR (CVaR)
        - Probability of loss
        
        Args:
            paths: Monte Carlo paths
            initial_value: Starting value
            
        Returns:
            Dictionary of risk metrics
        """
        print("📈 Calculating Risk Metrics...")
        print("-"*80)

        # --- NEW: MARKET FRICTION PARAMETERS ---
        # 0.05% slippage + commission per trade
        slippage_per_trade = 0.0005 
        # Assume 26 rebalances/year (bi-weekly average)
        rebalances_per_year = 26 
        # Total annual drag (approx 1.3%)
        annual_friction_drag = slippage_per_trade * rebalances_per_year 
        # --- END NEW ---
        
        final_values = paths[:, -1]
        raw_returns = (final_values / initial_value) - 1
        returns = raw_returns - annual_friction_drag #Apply drags
        
        # Sharpe Ratio (annualized, assuming risk-free rate = 0)
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe = (mean_return * np.sqrt(252)) / (std_return * np.sqrt(252))
        
        # Maximum Drawdown
        drawdowns = []
        for path in paths:
            cummax = np.maximum.accumulate(path)
            drawdown = (path - cummax) / cummax
            max_dd = np.min(drawdown)
            drawdowns.append(max_dd)
        
        max_drawdown = np.mean(drawdowns)
        worst_drawdown = np.min(drawdowns)
        
        # Value at Risk (5%)
        var_95 = np.percentile(returns, 5)
        
        # Conditional VaR (expected shortfall beyond VaR)
        cvar_95 = np.mean(returns[returns <= var_95])
        
        # Probability of loss
        prob_loss = np.sum(returns < 0) / len(returns)
        
        # Calmar Ratio (return / max drawdown)
        calmar = mean_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        metrics = {
            'mean_return': mean_return,
            'std_return': std_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'worst_drawdown': worst_drawdown,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'prob_loss': prob_loss,
            'calmar_ratio': calmar
        }
        
        # Print metrics
        print(f"\nExpected Return: {mean_return*100:6.2f}%")
        print(f"Return Std Dev:  {std_return*100:6.2f}%")
        print(f"Sharpe Ratio:    {sharpe:6.2f}")
        print(f"Max Drawdown:    {max_drawdown*100:6.2f}%")
        print(f"Worst Drawdown:  {worst_drawdown*100:6.2f}%")
        print(f"VaR (95%):       {var_95*100:6.2f}%")
        print(f"CVaR (95%):      {cvar_95*100:6.2f}%")
        print(f"Prob of Loss:    {prob_loss*100:6.2f}%")
        print(f"Calmar Ratio:    {calmar:6.2f}")
        print()
        
        return metrics
    
    def calculate_percentile_cone(self, paths: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate percentile cone for visualization.
        
        Returns 5th, 25th, 50th, 75th, 95th percentiles at each time step.
        """
        print("📊 Calculating Probability Cone...")
        
        percentiles = {
            'p5': np.percentile(paths, 5, axis=0),
            'p25': np.percentile(paths, 25, axis=0),
            'p50': np.percentile(paths, 50, axis=0),
            'p75': np.percentile(paths, 75, axis=0),
            'p95': np.percentile(paths, 95, axis=0)
        }
        
        print("   ✅ Percentile cone calculated\n")
        
        return percentiles
    
    def save_simulation_results(self, paths: np.ndarray, 
                               metrics: Dict[str, float],
                               percentiles: Dict[str, np.ndarray],
                               output_dir: str = "simulation_results"):
        """Save simulation results."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        metrics_path = f"{output_dir}/risk_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"💾 Risk metrics saved to {metrics_path}")
        
        # Save sample paths (first 100 for visualization)
        sample_paths_path = f"{output_dir}/sample_paths.npy"
        np.save(sample_paths_path, paths[:100])
        
        print(f"💾 Sample paths saved to {sample_paths_path}")
        
        # Save percentiles
        percentiles_path = f"{output_dir}/percentile_cone.npy"
        np.save(percentiles_path, percentiles)
        
        print(f"💾 Percentile cone saved to {percentiles_path}\n")


def main():
    """Execute Day 2 pipeline."""
    
    # ========================================================================
    # MORNING: VOLATILITY REGIME CLASSIFICATION
    # ========================================================================
    
    print("\n" + "="*80)
    print("PART 1: VOLATILITY REGIME PREDICTION")
    print("="*80 + "\n")
    
    classifier = VolatilityRegimeClassifier()
    
    # Prepare dataset
    X, y, feature_names = classifier.prepare_ml_dataset(forecast_horizon=21)
    
    # Train with walk-forward CV
    cv_scores = classifier.train_with_walk_forward_cv(X, y, feature_names, n_splits=5)
    
    # Analyze feature importance
    importance = classifier.analyze_feature_importance()
    
    # Save model
    classifier.save_model()
    
    # ========================================================================
    # AFTERNOON: MONTE CARLO SIMULATION
    # ========================================================================
    
    print("\n" + "="*80)
    print("PART 2: MONTE CARLO SIMULATION & RISK ANALYSIS")
    print("="*80 + "\n")
    
    simulator = MonteCarloSimulator()
    
    # Estimate parameters
    params = simulator.estimate_return_parameters(ticker="^GSPC")
    
    # Run Monte Carlo
    paths = simulator.run_monte_carlo(
        params,
        initial_value=10000,
        days=252,
        n_simulations=1000,
        use_regime_switching=True
    )
    
    # Calculate risk metrics
    metrics = simulator.calculate_risk_metrics(paths, initial_value=10000)
    
    # Calculate percentile cone
    percentiles = simulator.calculate_percentile_cone(paths)
    
    # Save results
    simulator.save_simulation_results(paths, metrics, percentiles)
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    print("\n" + "="*80)
    print("✅ DAY 2 COMPLETE")
    print("="*80)
    print("\n🎯 Deliverables:")
    print("   • Volatility regime classifier (XGBoost)")
    print(f"   • CV Accuracy: {np.mean(cv_scores):.2%} ± {np.std(cv_scores):.2%}")
    print(f"   • Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"   • Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
    print(f"   • Calmar Ratio: {metrics['calmar_ratio']:.2f}")
    print("\n📁 Output Files:")
    print("   • models/regime_classifier.json")
    print("   • simulation_results/risk_metrics.json")
    print("   • simulation_results/sample_paths.npy")
    print("   • simulation_results/percentile_cone.npy")
    print("\n🚀 Next: Day 3 - Frontend & API")
    print("="*80)


if __name__ == "__main__":
    main()