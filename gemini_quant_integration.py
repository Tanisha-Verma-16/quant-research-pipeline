"""
Gemini Flash 2.5 Integration for Quantitative Research
Novel Features: Sentiment Analysis, Regime Explanations, NL Querying

Setup:
pip install google-generativeai

"""

import google.generativeai as genai
import os
import json
from typing import Dict, List, Optional
import polars as pl
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()


# Configure Gemini
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.5-flash')


class GeminiQuantAnalyst:
    """
    AI-powered quantitative analyst using Gemini Flash 2.5.
    
    Features:
    1. News sentiment scoring
    2. Regime change explanations
    3. Portfolio research reports
    4. Natural language queries
    """
    
    def __init__(self):
        self.model = model
        
    # ========================================================================
    # 1. SENTIMENT-AUGMENTED ALPHA FACTORS
    # ========================================================================
    
    def analyze_news_sentiment(self, ticker: str, headlines: List[str]) -> Dict:
        """
        Analyze news sentiment for a ticker using Gemini.
        
        Returns sentiment score (-1 to +1) and reasoning.
        """
        
        prompt = f"""You are a quantitative analyst specializing in sentiment analysis.

Analyze the sentiment of these recent news headlines for {ticker}:

{chr(10).join(f"- {h}" for h in headlines)}

Provide your analysis in JSON format:
{{
    "sentiment_score": <float between -1 (very bearish) and +1 (very bullish)>,
    "confidence": <float between 0 and 1>,
    "key_themes": [<list of 2-3 main themes>],
    "risk_factors": [<list of 2-3 specific risks mentioned>],
    "summary": "<2 sentence summary>"
}}

Be objective and focus on market-moving information."""

        try:
            response = self.model.generate_content(prompt)
            
            # Extract JSON from response
            text = response.text
            if "```json" in text:
                json_str = text.split("```json")[1].split("```")[0].strip()
            else:
                json_str = text.strip()
            
            result = json.loads(json_str)
            
            return {
                "ticker": ticker,
                "sentiment_score": result["sentiment_score"],
                "confidence": result["confidence"],
                "themes": result["key_themes"],
                "risks": result["risk_factors"],
                "summary": result["summary"],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
            return {
                "ticker": ticker,
                "sentiment_score": 0.0,
                "confidence": 0.0,
                "error": str(e)
            }
    
    def batch_sentiment_analysis(self, ticker_headlines: Dict[str, List[str]]) -> pl.DataFrame:
        """
        Analyze sentiment for multiple tickers.
        
        Returns Polars DataFrame with sentiment scores as new features.
        """
        print(f"🧠 Analyzing sentiment for {len(ticker_headlines)} tickers using Gemini...")
        
        results = []
        for ticker, headlines in ticker_headlines.items():
            sentiment = self.analyze_news_sentiment(ticker, headlines)
            results.append(sentiment)
            print(f"  ✅ {ticker}: Sentiment = {sentiment['sentiment_score']:.2f}")
        
        df = pl.DataFrame(results)
        return df
    
    # ========================================================================
    # 2. REGIME CHANGE EXPLANATIONS
    # ========================================================================
    
    def explain_regime_prediction(
        self, 
        ticker: str,
        current_regime: str,
        predicted_regime: str,
        features: Dict[str, float],
        recent_news: Optional[List[str]] = None
    ) -> str:
        """
        Generate human-readable explanation for regime prediction.
        """
        
        # Format features for readability
        feature_text = "\n".join([
            f"- {k}: {v:.4f}" for k, v in sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        ])
        
        news_context = ""
        if recent_news:
            news_context = f"\n\nRecent News Headlines:\n" + "\n".join(f"- {h}" for h in recent_news)
        
        prompt = f"""You are a senior quantitative analyst explaining a volatility regime prediction.

Ticker: {ticker}
Current Regime: {current_regime}
Predicted Regime (21 days ahead): {predicted_regime}

Top Technical Indicators:
{feature_text}
{news_context}

Explain in 2-3 sentences:
1. WHY this regime change is predicted (cite specific technical factors)
2. What this means for traders (risk implications)
3. Historical context (if regime is changing)

Write in professional but accessible language suitable for institutional investors."""

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Unable to generate explanation: {e}"
    
    # ========================================================================
    # 3. AUTOMATED RESEARCH REPORTS
    # ========================================================================
    
    def generate_daily_report(
        self,
        top_momentum: List[Dict],
        risk_metrics: Dict,
        regime_predictions: List[Dict]
    ) -> str:
        """
        Generate institutional-grade daily research report.
        """
        
        # Format data for Gemini
        momentum_text = "\n".join([
            f"- {stock['Ticker']}: Alpha={stock.get('alpha_momentum', 0):.2f}, Decile={stock.get('mom_decile', 0)}"
            for stock in top_momentum[:10]
        ])
        
        regime_text = "\n".join([
            f"- {pred['ticker']}: {pred['current_regime']} → {pred['predicted_regime']} (confidence: {pred['confidence']:.0%})"
            for pred in regime_predictions[:5]
        ])
        
        prompt = f"""You are a hedge fund quantitative analyst writing the daily research briefing.

=== TOP MOMENTUM SIGNALS ===
{momentum_text}

=== PORTFOLIO RISK METRICS ===
- Sharpe Ratio: {risk_metrics.get('sharpe_ratio', 0):.2f}
- Max Drawdown: {risk_metrics.get('max_drawdown', 0):.2%}
- VaR (95%): {risk_metrics.get('var_95', 0):.2%}

=== REGIME PREDICTIONS (21-day ahead) ===
{regime_text}

Write a professional 1-page investment memo covering:
1. **Executive Summary** (2 sentences - key takeaway)
2. **Top Opportunities** (3 momentum stocks worth watching, with rationale)
3. **Risk Assessment** (current portfolio risk posture)
4. **Regime Outlook** (which sectors to favor based on predicted regimes)
5. **Recommended Actions** (2-3 concrete trading ideas)

Format professionally with markdown headers."""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Unable to generate report: {e}"
    
    # ========================================================================
    # 4. NATURAL LANGUAGE QUERYING
    # ========================================================================
    
    def parse_natural_query(self, user_query: str) -> Dict:
        """
        Convert natural language query to API parameters.
        
        Example: "Which tech stocks have low correlation with bonds?"
        → {sector: "tech", comparison: "TLT", metric: "correlation", threshold: "low"}
        """
        
        prompt = f"""You are an API query translator for a quantitative finance system.

User query: "{user_query}"

Extract the following parameters and return as JSON:
{{
    "intent": "<query_type: correlation|momentum|risk|sentiment|prediction>",
    "tickers": [<list of ticker symbols mentioned, or empty if sector-based>],
    "sector": "<sector if mentioned: tech|finance|energy|india|commodities|null>",
    "metric": "<metric: correlation|sharpe|alpha|volatility|drawdown>",
    "threshold": "<if filtering: high|low|top|bottom|null>",
    "time_horizon": "<if mentioned: 1m|3m|6m|1y|null>",
    "comparison_asset": "<if comparing to another asset, ticker symbol>"
}}

Only extract parameters explicitly mentioned or clearly implied. Use null for unclear parameters."""

        try:
            response = self.model.generate_content(prompt)
            
            # Extract JSON
            text = response.text
            if "```json" in text:
                json_str = text.split("```json")[1].split("```")[0].strip()
            else:
                json_str = text.strip()
            
            params = json.loads(json_str)
            return params
            
        except Exception as e:
            print(f"Error parsing query: {e}")
            return {"intent": "unknown", "error": str(e)}
    
    def answer_natural_query(
        self,
        user_query: str,
        api_response: Dict
    ) -> str:
        """
        Convert API response to natural language answer.
        """
        
        prompt = f"""You are a quantitative analyst assistant.

User asked: "{user_query}"

API returned this data:
{json.dumps(api_response, indent=2)}

Provide a clear, concise answer in 2-3 sentences:
1. Directly answer their question
2. Highlight the most important finding
3. Suggest a follow-up action if relevant

Write professionally but conversationally."""

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Unable to generate answer: {e}"
    
    # ========================================================================
    # 5. FEATURE ENGINEERING ASSISTANT
    # ========================================================================
    
    def suggest_custom_factors(
        self,
        ticker: str,
        current_features: List[str],
        historical_data: pl.DataFrame
    ) -> List[str]:
        """
        Use Gemini to suggest novel alpha factors based on data patterns.
        """
        
        # Sample some statistics
        stats = historical_data.select([
            pl.col("log_return").mean().alias("avg_return"),
            pl.col("log_return").std().alias("volatility"),
            pl.col("vol_20d").mean().alias("avg_vol_20d")
        ]).to_dicts()[0]
        
        prompt = f"""You are a quantitative researcher specializing in factor discovery.

For ticker {ticker}, we currently use these {len(current_features)} factors:
{', '.join(current_features[:20])}...

Historical statistics:
- Average daily return: {stats['avg_return']:.4%}
- Volatility: {stats['volatility']:.4%}
- Average 20-day vol: {stats.get('avg_vol_20d', 0):.4%}

Suggest 3 NOVEL alpha factors we could engineer that might have predictive power.
Focus on:
1. Cross-asset relationships (e.g., correlation with VIX)
2. Temporal patterns (e.g., day-of-week effects)
3. Microstructure (e.g., bid-ask spread proxies)

Return as JSON array:
[
  {{"name": "factor_name", "formula": "technical description", "rationale": "why it might work"}},
  ...
]"""

        try:
            response = self.model.generate_content(prompt)
            
            # Extract JSON
            text = response.text
            if "```json" in text:
                json_str = text.split("```json")[1].split("```")[0].strip()
            else:
                json_str = text.strip()
            
            suggestions = json.loads(json_str)
            return suggestions
            
        except Exception as e:
            print(f"Error getting suggestions: {e}")
            return []


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_usage():
    """Demonstrate Gemini integration."""
    
    analyst = GeminiQuantAnalyst()
    
    # 1. Sentiment Analysis
    print("\n" + "="*80)
    print("1. NEWS SENTIMENT ANALYSIS")
    print("="*80)
    
    headlines = [
        "Apple announces record iPhone sales in Q4",
        "AAPL faces regulatory scrutiny in EU markets",
        "Morgan Stanley upgrades Apple to overweight"
    ]
    
    sentiment = analyst.analyze_news_sentiment("AAPL", headlines)
    print(f"\nSentiment Score: {sentiment['sentiment_score']:.2f}")
    print(f"Summary: {sentiment['summary']}")
    
    # 2. Regime Explanation
    print("\n" + "="*80)
    print("2. REGIME CHANGE EXPLANATION")
    print("="*80)
    
    explanation = analyst.explain_regime_prediction(
        ticker="AAPL",
        current_regime="NORMAL_VOL",
        predicted_regime="HIGH_VOL",
        features={
            "z_vol": 2.3,
            "rsi_14": 72,
            "vol_trend": 0.15,
            "atr_pct": 0.025
        },
        recent_news=headlines
    )
    
    print(f"\n{explanation}")
    
    # 3. Natural Language Query
    print("\n" + "="*80)
    print("3. NATURAL LANGUAGE QUERY")
    print("="*80)
    
    query = "Which tech stocks have negative correlation with bonds?"
    params = analyst.parse_natural_query(query)
    print(f"\nQuery: {query}")
    print(f"Extracted Parameters: {json.dumps(params, indent=2)}")


if __name__ == "__main__":
    # Check for API key
    if not os.environ.get("GEMINI_API_KEY"):
        print("⚠️  Set GEMINI_API_KEY environment variable first!")
        print("   export GEMINI_API_KEY='your_key_here'")
    else:
        example_usage()