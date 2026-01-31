"""
Quantitative Research Pipeline - Day 1 Morning (PRODUCTION SCALE)
Multi-Asset Factor Research: Momentum, Volatility, Mean Reversion

Optimized for 100+ tickers with rate-limit handling and factor engineering.
"""

import yfinance as yf
import polars as pl
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

class ScalableQuantPipeline:
    """
    Production-grade pipeline for cross-asset factor research.
    Handles 100+ tickers with rate limiting and parallel downloads.
    """

    # Asset class definitions for cross-sectional analysis
    ASSET_CLASSES = {
        "US_INDICES": ['^GSPC', '^NDX', '^DJI', '^RUT', '^VIX', '^IXIC'],
        "INTL_INDICES": ['^FTSE', '^N225', '^HSI', '^BSESN', '^NSEI'],
        "US_MEGACAP": ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA', 'JPM', 'BRK-B', 'JNJ'],
        "US_ETFS": ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'GLD', 'TLT', 'HYG', 'XLF', 'XLK'],
        "INDIA_LARGECAP": ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS',
                           'SBIN.NS', 'ITC.NS', 'LT.NS'],
        "COMMODITIES_FX": ['GC=F', 'CL=F', 'SI=F', 'NG=F', 'DX-Y.NYB', 'USDINR=X']
    }

    def __init__(self, lookback_days: int = 1095, use_all_assets: bool = True):
        """
        Initialize scalable pipeline.

        Args:
            lookback_days: Historical window (default: 3 years for factor research)
            use_all_assets: If True, use all asset classes. If False, select specific ones.
        """
        self.lookback_days = lookback_days
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=lookback_days)

        if use_all_assets:
            self.tickers = self._get_all_tickers()
        else:
            self.tickers = []

        self.ticker_metadata = self._create_metadata()

    def _get_all_tickers(self) -> List[str]:
        """Flatten all asset classes into single list."""
        all_tickers = []
        for tickers in self.ASSET_CLASSES.values():
            all_tickers.extend(tickers)
        return list(set(all_tickers))  # Remove duplicates

    def _create_metadata(self) -> pl.DataFrame:
        """Create metadata mapping tickers to asset classes."""
        metadata = []
        for asset_class, tickers in self.ASSET_CLASSES.items():
            for ticker in tickers:
                metadata.append({"Ticker": ticker, "AssetClass": asset_class})
        return pl.DataFrame(metadata)

    def add_custom_tickers(self, tickers: List[str], asset_class: str = "CUSTOM"):
        """Add custom tickers to the pipeline."""
        self.tickers.extend(tickers)
        for ticker in tickers:
            self.ticker_metadata = pl.concat([
                self.ticker_metadata,
                pl.DataFrame([{"Ticker": ticker, "AssetClass": asset_class}])
            ])

    def fetch_ohlcv_parallel(self, max_workers: int = 5, retry_limit: int = 3) -> pl.DataFrame:
        """
        Parallel download with rate limiting and retry logic.
        Yahoo Finance limits: ~2000 requests/hour, ~48 requests/second.

        Args:
            max_workers: Concurrent downloads (5 is safe for Yahoo)
            retry_limit: Retries per ticker on failure

        Returns:
            Polars DataFrame with all OHLCV data
        """
        print(f"📊 Fetching {len(self.tickers)} tickers in parallel (workers={max_workers})...")
        print(f"   Date range: {self.start_date.date()} → {self.end_date.date()}\n")

        all_data = []
        failed_tickers = []

        def download_ticker(ticker: str) -> tuple:
            """Download single ticker with retry logic."""
            for attempt in range(retry_limit):
                try:
                    time.sleep(0.5)  # Rate limiting: 2 requests/second

                    df = yf.download(
                        ticker,
                        start=self.start_date,
                        end=self.end_date,
                        progress=False,
                        auto_adjust=True  # Use adjusted prices
                    )

                    if df.empty:
                        return (ticker, None, "No data returned")

                    # Flatten multi-index columns if they exist
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)

                    # Reset index and standardize column names
                    df_reset = df.reset_index()

                    # Ensure standard OHLCV column names
                    expected_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                    available_cols = ['Date'] + [col for col in expected_cols[1:] if col in df_reset.columns]
                    df_reset = df_reset[available_cols]

                    # Convert to Polars
                    df_pl = pl.from_pandas(df_reset)
                    df_pl = df_pl.with_columns(pl.lit(ticker).alias("Ticker"))

                    return (ticker, df_pl, None)

                except Exception as e:
                    if attempt == retry_limit - 1:
                        return (ticker, None, str(e))
                    time.sleep(2 ** attempt)  # Exponential backoff

            return (ticker, None, "Max retries exceeded")

        # Parallel download
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(download_ticker, ticker): ticker
                      for ticker in self.tickers}

            for future in as_completed(futures):
                ticker, df_pl, error = future.result()

                if df_pl is not None:
                    all_data.append(df_pl)
                    print(f"  ✅ {ticker:12s} → {len(df_pl):,} rows")
                else:
                    failed_tickers.append((ticker, error))
                    print(f"  ❌ {ticker:12s} → FAILED ({error})")

        if failed_tickers:
            print(f"\n⚠️  {len(failed_tickers)} ticker(s) failed. Consider retry or removal.")

        if not all_data:
            raise ValueError("No data downloaded successfully!")

        combined = pl.concat(all_data)
        print(f"\n✅ Total rows downloaded: {len(combined):,}")

        return combined

    def engineer_factor_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Engineer alpha factors for momentum, volatility, and mean reversion research.

        Factors created:
        - Log returns (basis for all factors)
        - Rolling volatility (regime detection)
        - Z-scored volatility (normalized regime)
        - Momentum signals (12M, 6M, 3M, 1M)
        - Mean reversion signals (RSI-like normalized returns)
        - Volume features (liquidity proxy)
        """
        print("\n🔬 Engineering Factor Features...")

        df = df.sort(["Ticker", "Date"])

        # === RETURNS ===
        print("  → Log returns")
        df = df.with_columns([
            (pl.col("Close").log() - pl.col("Close").shift(1).log())
            .over("Ticker")
            .alias("log_return")
        ])

        # === VOLATILITY FACTORS ===
        print("  → Volatility factors (20D, 60D)")
        df = df.with_columns([
            # 20-day volatility (1 month)
            pl.col("log_return").rolling_std(window_size=20).over("Ticker").alias("vol_20d"),
            # 60-day volatility (3 months)
            pl.col("log_return").rolling_std(window_size=60).over("Ticker").alias("vol_60d"),
        ])

        # Z-scored volatility (252-day lookback = 1 year)
        print("  → Z-scored volatility")
        df = df.with_columns([
            ((pl.col("vol_20d") - pl.col("vol_20d").rolling_mean(252).over("Ticker"))
             / pl.col("vol_20d").rolling_std(252).over("Ticker"))
            .alias("z_vol")
        ])

        # === MOMENTUM FACTORS ===
        print("  → Momentum factors (1M, 3M, 6M, 12M)")
        df = df.with_columns([
            # 1-month momentum (21 trading days)
            (pl.col("Close") / pl.col("Close").shift(21) - 1).over("Ticker").alias("mom_1m"),
            # 3-month momentum
            (pl.col("Close") / pl.col("Close").shift(63) - 1).over("Ticker").alias("mom_3m"),
            # 6-month momentum
            (pl.col("Close") / pl.col("Close").shift(126) - 1).over("Ticker").alias("mom_6m"),
            # 12-month momentum (excluding last month per Jegadeesh & Titman)
            (pl.col("Close").shift(21) / pl.col("Close").shift(252) - 1).over("Ticker").alias("mom_12m"),
        ])

        # === MEAN REVERSION FACTORS ===
        print("  → Mean reversion factors (normalized returns)")
        df = df.with_columns([
            # 20-day normalized returns (RSI-like)
            ((pl.col("Close") - pl.col("Close").rolling_mean(20).over("Ticker"))
             / pl.col("Close").rolling_std(20).over("Ticker"))
            .alias("mean_rev_20d"),
            # 60-day normalized returns
            ((pl.col("Close") - pl.col("Close").rolling_mean(60).over("Ticker"))
             / pl.col("Close").rolling_std(60).over("Ticker"))
            .alias("mean_rev_60d"),
        ])

        # === VOLUME FACTORS (Liquidity proxy) ===
        if "Volume" in df.columns:
            print("  → Volume factors")
            df = df.with_columns([
                # Volume Z-score (liquidity spikes)
                ((pl.col("Volume") - pl.col("Volume").rolling_mean(20).over("Ticker"))
                 / pl.col("Volume").rolling_std(20).over("Ticker"))
                .alias("volume_z")
            ])

        print("✅ Factor engineering complete\n")
        return df

    def create_cross_sectional_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Create cross-sectional (relative) features within asset classes.
        Essential for factor investing (ranking stocks vs peers).
        """
        print("🌐 Creating cross-sectional features...")

        # Join asset class metadata
        df = df.join(self.ticker_metadata, on="Ticker", how="left")

        # Rank momentum within asset class (0-1 scale)
        df = df.with_columns([
            pl.col("mom_1m").rank().over(["Date", "AssetClass"]).alias("mom_1m_rank_pct"),
            pl.col("mom_3m").rank().over(["Date", "AssetClass"]).alias("mom_3m_rank_pct"),
        ])

        # Normalize ranks to 0-1
        df = df.with_columns([
            (pl.col("mom_1m_rank_pct") / pl.col("mom_1m_rank_pct").max().over(["Date", "AssetClass"]))
            .alias("mom_1m_rank_pct"),
            (pl.col("mom_3m_rank_pct") / pl.col("mom_3m_rank_pct").max().over(["Date", "AssetClass"]))
            .alias("mom_3m_rank_pct"),
        ])

        print("✅ Cross-sectional features added\n")
        return df

    def data_leakage_check(self, df: pl.DataFrame):
        """Comprehensive data leakage validation."""
        print("🔍 Running Data Leakage Assertions...\n")

        # Check 1: Chronological ordering
        for ticker in df["Ticker"].unique():
            ticker_data = df.filter(pl.col("Ticker") == ticker)
            assert ticker_data["Date"].is_sorted(), \
                f"❌ LEAK: {ticker} dates not sorted chronologically"

        # Check 2: First return is null (no previous price)
        first_returns = df.group_by("Ticker").agg(pl.col("log_return").first())
        assert first_returns["log_return"].null_count() == len(df["Ticker"].unique()), \
            "❌ LEAK: First log return must be null"

        # Check 3: Momentum features have nulls in lookback period
        assert df["mom_12m"].null_count() > 0, \
            "❌ LEAK: 12M momentum should have nulls for first 252 days"

        # Check 4: No future data in rolling windows
        sample_ticker = df["Ticker"].unique()[0]
        sample_data = df.filter(pl.col("Ticker") == sample_ticker).head(30)
        assert sample_data["vol_20d"].null_count() >= 19, \
            "❌ LEAK: 20D volatility should be null for first 19 rows"

        print("✅ All data leakage checks passed!\n")

    def save_research_dataset(self, df: pl.DataFrame, output_dir: str = "data"):
        """Save cleaned dataset optimized for research."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Drop nulls from initial warm-up period
        df_clean = df.drop_nulls(subset=["log_return", "vol_20d", "mom_3m"])

        # Save full dataset
        full_path = f"{output_dir}/factor_dataset_full.parquet"
        df_clean.write_parquet(full_path)
        print(f"💾 Saved full dataset: {full_path}")
        print(f"   Rows: {len(df_clean):,} | Columns: {len(df_clean.columns)}")
        print(f"   Size: {Path(full_path).stat().st_size / (1024*1024):.2f} MB\n")

        # Save by asset class for stratified analysis
        for asset_class in df_clean["AssetClass"].unique():
            df_subset = df_clean.filter(pl.col("AssetClass") == asset_class)
            subset_path = f"{output_dir}/factor_{asset_class.lower()}.parquet"
            df_subset.write_parquet(subset_path)
            print(f"   → {asset_class}: {len(df_subset):,} rows ({subset_path})")

    def generate_factor_summary(self, df: pl.DataFrame):
        """Generate summary statistics for research documentation."""
        print("\n" + "="*80)
        print("📊 FACTOR RESEARCH SUMMARY STATISTICS")
        print("="*80 + "\n")

        df_clean = df.drop_nulls(subset=["log_return"])

        print(f"Total Observations: {len(df_clean):,}")
        print(f"Unique Tickers: {df_clean['Ticker'].n_unique()}")
        print(f"Date Range: {df_clean['Date'].min()} → {df_clean['Date'].max()}")
        print(f"Trading Days: {df_clean.group_by('Ticker').agg(pl.col('Date').n_unique()).select(pl.col('Date').mean())[0,0]:.0f} avg\n")

        # Factor correlations (aggregate across all assets)
        print("📈 FACTOR CORRELATIONS (Aggregate)")
        print("-" * 80)
        factor_cols = ["log_return", "vol_20d", "mom_1m", "mom_3m", "mom_6m", "mean_rev_20d"]
        corr = df_clean.select(factor_cols).drop_nulls().to_pandas().corr()
        print(corr.round(3))
        print()

        # Asset class performance
        print("💼 ASSET CLASS PERFORMANCE")
        print("-" * 80)
        for asset_class in sorted(df_clean["AssetClass"].unique()):
            ac_data = df_clean.filter(pl.col("AssetClass") == asset_class)
            mean_ret = ac_data["log_return"].mean()
            vol = ac_data["log_return"].std()
            sharpe = mean_ret / vol * np.sqrt(252) if vol > 0 else 0

            print(f"{asset_class:20s} | Return: {mean_ret*252:7.2%} | Vol: {vol*np.sqrt(252):6.2%} | Sharpe: {sharpe:5.2f}")


def main():
    """
    Execute production-scale Day 1 Morning pipeline.
    """
    print("="*80)
    print("QUANTITATIVE RESEARCH PIPELINE - DAY 1 MORNING (PRODUCTION)")
    print("Multi-Asset Factor Engineering: Momentum, Volatility, Mean Reversion")
    print("="*80 + "\n")

    # Initialize pipeline with all asset classes
    pipeline = ScalableQuantPipeline(lookback_days=1095, use_all_assets=True)

    print(f"📋 Pipeline configured:")
    print(f"   Total tickers: {len(pipeline.tickers)}")
    for ac, tickers in pipeline.ASSET_CLASSES.items():
        print(f"   {ac:20s}: {len(tickers)} tickers")
    print()

    # Step 1: Parallel download with rate limiting
    df = pipeline.fetch_ohlcv_parallel(max_workers=5, retry_limit=3)

    # Step 2: Engineer factor features
    df = pipeline.engineer_factor_features(df)

    # Step 3: Create cross-sectional features
    df = pipeline.create_cross_sectional_features(df)

    # Step 4: Validate no data leakage
    pipeline.data_leakage_check(df)

    # Step 5: Generate research summary
    pipeline.generate_factor_summary(df)

    # Step 6: Save research dataset
    pipeline.save_research_dataset(df)

    print("\n" + "="*80)
    print("✅ DAY 1 MORNING COMPLETE")
    print("="*80)
    print("\n📁 Output Files:")
    print("   • data/factor_dataset_full.parquet    (Full dataset)")
    print("   • data/factor_*.parquet                (By asset class)")
    print("\n🚀 Next Steps:")
    print("   → Day 1 Afternoon: Alpha Factor Engineering (RSI, MACD, custom signals)")
    print("   → Day 2: ML Modeling & Monte Carlo Simulation")
    print("="*80)


if __name__ == "__main__":
    main()