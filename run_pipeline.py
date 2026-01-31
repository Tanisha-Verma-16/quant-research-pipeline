"""
Master Execution Pipeline - Quantitative Research System
=========================================================

Orchestrates entire pipeline from data ingestion to API deployment:
1. Data ingestion & factor engineering (Day 1)
2. ML modeling & Monte Carlo simulation (Day 2)
3. SHAP interpretability analysis (Day 3)
4. Portfolio strategy testing (Day 3)
5. Financial reasoning agent demo (Day 3)
6. API server launch (Production)

Author: Senior Quant ML Engineer
For: Hackathon Demo & Production Deployment
"""

import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
import json


class QuantPipelineOrchestrator:
    """
    Master orchestrator for the entire quantitative research pipeline.
    
    Ensures proper execution order and validates outputs at each stage.
    """
    
    def __init__(self):
        self.start_time = datetime.now()
        self.execution_log = []
        self.failed_steps = []
        self.log_file = f"pipeline_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
    def log_step(self, step_name: str, status: str, duration: float = 0):
        """Log pipeline step execution."""
        self.execution_log.append({
            "step": step_name,
            "status": status,
            "duration_seconds": duration,
            "timestamp": datetime.now().isoformat()
        })
        
    def print_header(self, title: str):
        """Print formatted section header."""
        print("\n" + "="*80)
        print(f"  {title}")
        print("="*80 + "\n")
        
    def run_script(self, script_name: str, description: str, required: bool = True) -> bool:
        """
        Run a Python script and track results.
        
        Args:
            script_name: Script filename
            description: Human-readable description
            required: If True, pipeline stops on failure
            
        Returns:
            True if successful, False otherwise
        """
        self.print_header(f"STEP: {description}")
        print(f"📄 Executing: {script_name}")
        print(f"⏰ Started: {datetime.now().strftime('%H:%M:%S')}\n")
        
        step_start = time.time()
        
        try:
            result = subprocess.run(
                [sys.executable, script_name],
                capture_output=False,
                text=True,
                check=True
            )
            
            duration = time.time() - step_start
            
            print(f"\n✅ {description} completed successfully")
            print(f"⏱️  Duration: {duration:.1f} seconds")
            
            self.log_step(script_name, "SUCCESS", duration)
            return True
            
        except subprocess.CalledProcessError as e:
            duration = time.time() - step_start
            
            print(f"\n❌ {description} FAILED")
            print(f"   Error code: {e.returncode}")
            print(f"⏱️  Duration: {duration:.1f} seconds")
            
            self.log_step(script_name, "FAILED", duration)
            self.failed_steps.append(script_name)
            
            if required:
                print(f"\n🛑 Critical step failed. Stopping pipeline.")
                return False
            else:
                print(f"\n⚠️  Non-critical step failed. Continuing...")
                return True
                
        except FileNotFoundError:
            print(f"\n❌ Script not found: {script_name}")
            self.log_step(script_name, "NOT_FOUND", 0)
            self.failed_steps.append(script_name)
            
            if required:
                print(f"\n🛑 Required script missing. Stopping pipeline.")
                return False
            return True
    
    def validate_outputs(self, stage: str) -> bool:
        """
        Validate that expected outputs exist after each stage.
        
        Args:
            stage: Pipeline stage name
            
        Returns:
            True if all expected files exist
        """
        validations = {
            "day1_morning": [
                "data/factor_dataset_full.parquet",
                "data/factor_us_megacap.parquet"
            ],
            "day1_afternoon": [
                "data/alpha_factors_full.parquet",
                "data/alpha_factors_sample.csv"
            ],
            "day2": [
                "models/regime_classifier.json",
                "models/feature_names.json",
                "simulation_results/risk_metrics.json",
                "simulation_results/sample_paths.npy"
            ],
            "shap": [
                "shap_results/global_feature_importance.csv",
                "shap_results/HIGH_VOL_importance.csv"
            ]
        }
        
        if stage not in validations:
            return True
        
        print(f"\n🔍 Validating {stage} outputs...")
        
        all_exist = True
        for file_path in validations[stage]:
            if Path(file_path).exists():
                size_kb = Path(file_path).stat().st_size / 1024
                print(f"  ✅ {file_path} ({size_kb:.1f} KB)")
            else:
                print(f"  ❌ MISSING: {file_path}")
                all_exist = False
        
        return all_exist
    
    def run_full_pipeline(self, skip_data: bool = False):
        """
        Execute full pipeline from start to finish.
        
        Args:
            skip_data: If True, skip Day 1 (use existing data)
        """
        
        print("\n" + "="*80)
        print("  QUANTITATIVE RESEARCH PIPELINE - FULL EXECUTION")
        print("  From Data Ingestion to Production Deployment")
        print("="*80)
        print(f"\n⏰ Pipeline started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🐍 Python: {sys.version.split()[0]}")
        print(f"📁 Working directory: {Path.cwd()}\n")
        
        # ====================================================================
        # STAGE 1: DATA INGESTION & FACTOR ENGINEERING
        # ====================================================================
        
        if not skip_data:
            if not self.run_script(
                "day1_morning.py",
                "Day 1 Morning - Data Ingestion & Basic Factors",
                required=True
            ):
                return self.finish_pipeline()
            
            if not self.validate_outputs("day1_morning"):
                print("\n⚠️  Day 1 Morning validation failed, but continuing...")
            
            time.sleep(2)  # Brief pause between stages
            
            if not self.run_script(
                "day1_afternoon.py",
                "Day 1 Afternoon - Advanced Alpha Factors (Pure Polars)",
                required=True
            ):
                return self.finish_pipeline()
            
            if not self.validate_outputs("day1_afternoon"):
                print("\n⚠️  Day 1 Afternoon validation failed, but continuing...")
            
            time.sleep(2)
        else:
            print("\n⏩ Skipping Day 1 (using existing data)")
            self.log_step("day1_morning.py", "SKIPPED", 0)
            self.log_step("day1_afternoon.py", "SKIPPED", 0)
        
        # ====================================================================
        # STAGE 2: ML MODELING & MONTE CARLO
        # ====================================================================
        
        if not self.run_script(
            "day2_modeling.py",
            "Day 2 - ML Regime Classifier & Monte Carlo Simulation",
            required=True
        ):
            return self.finish_pipeline()
        
        if not self.validate_outputs("day2"):
            print("\n⚠️  Day 2 validation failed. Some features may not work.")
        
        time.sleep(2)
        
        # ====================================================================
        # STAGE 3: INTERPRETABILITY & STRATEGY
        # ====================================================================
        
        # SHAP Analysis (optional but recommended)
        if Path("shap_analysis.py").exists():
            if not self.run_script(
                "shap_analysis.py",
                "Day 3 - SHAP Interpretability Analysis",
                required=False
            ):
                print("\n⚠️  SHAP analysis failed. Model still works, but less explainable.")
            
            if not self.validate_outputs("shap"):
                print("\n⚠️  SHAP outputs incomplete.")
            
            time.sleep(2)
        else:
            print("\n⏩ Skipping SHAP analysis (shap_analysis.py not found)")
            self.log_step("shap_analysis.py", "SKIPPED", 0)
        
        # Portfolio Strategy Testing (optional)
        if Path("portfolio_strategy_decision_engine.py").exists():
            if not self.run_script(
                "portfolio_strategy_decision_engine.py",
                "Day 3 - Portfolio Strategy Decision Engine Demo",
                required=False
            ):
                print("\n⚠️  Strategy demo failed, but API will still work.")
            
            time.sleep(2)
        else:
            print("\n⏩ Skipping strategy demo (file not found)")
            self.log_step("portfolio_strategy_decision_engine.py", "SKIPPED", 0)
        
        # Gemini Reasoning Agent Demo (optional)
        if Path("financial_reasoning_agent.py").exists():
            if not self.run_script(
                "financial_reasoning_agent.py",
                "Day 3 - Gemini Financial Reasoning Agent Demo",
                required=False
            ):
                print("\n⚠️  Gemini demo failed. API will use template responses.")
            
            time.sleep(2)
        else:
            print("\n⏩ Skipping Gemini demo (file not found)")
            self.log_step("financial_reasoning_agent.py", "SKIPPED", 0)
        
        # ====================================================================
        # STAGE 4: PIPELINE COMPLETION
        # ====================================================================
        
        self.finish_pipeline()
    
    def finish_pipeline(self):
        """Print final summary and save execution log."""
        
        total_duration = (datetime.now() - self.start_time).total_seconds()
        
        self.print_header("PIPELINE EXECUTION COMPLETE")
        
        # Summary statistics
        successful = len([s for s in self.execution_log if s["status"] == "SUCCESS"])
        failed = len([s for s in self.execution_log if s["status"] == "FAILED"])
        skipped = len([s for s in self.execution_log if s["status"] == "SKIPPED"])
        
        print(f"⏱️  Total Duration: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
        print(f"✅ Successful Steps: {successful}")
        print(f"❌ Failed Steps: {failed}")
        print(f"⏩ Skipped Steps: {skipped}")
        
        if failed > 0:
            print(f"\n⚠️  Failed steps:")
            for step in self.failed_steps:
                print(f"   • {step}")
        
        # Save execution log
        log_path = "pipeline_execution_log.json"
        with open(log_path, 'w') as f:
            json.dump({
                "start_time": self.start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "total_duration_seconds": total_duration,
                "summary": {
                    "successful": successful,
                    "failed": failed,
                    "skipped": skipped
                },
                "steps": self.execution_log
            }, f, indent=2)
        
        print(f"\n📄 Execution log saved: {log_path}")
        
        # Next steps guidance
        print("\n" + "="*80)
        print("🚀 NEXT STEPS")
        print("="*80)
        
        if failed == 0:
            print("\n✅ All critical steps completed successfully!")
            print("\n📁 Output Files Generated:")
            print("   • data/alpha_factors_full.parquet (factor dataset)")
            print("   • models/regime_classifier.json (trained XGBoost)")
            print("   • simulation_results/risk_metrics.json (Monte Carlo)")
            print("   • shap_results/*.csv (interpretability)")
            
            print("\n🌐 To start the API server:")
            print("   uvicorn main:app --reload")
            print("\n📖 API documentation will be at:")
            print("   http://localhost:8000/docs")
            
            print("\n🎯 Key API Endpoints:")
            print("   • /api/predict-regime/{ticker}")
            print("   • /api/strategy-recommendation/{ticker}")
            print("   • /api/executive-summary/{ticker}")
            print("   • /api/portfolio-analysis/{ticker}")
            
        else:
            print("\n⚠️  Some steps failed. Review errors above.")
            print("   Pipeline may still be usable with reduced functionality.")
        
        print("\n" + "="*80)


def main():
    """
    Main execution function.
    
    Usage:
        python run_pipeline.py              # Full pipeline
        python run_pipeline.py --skip-data  # Skip Day 1 (use existing data)
    """
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Execute quantitative research pipeline"
    )
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Skip Day 1 data ingestion (use existing data)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: Skip optional steps (SHAP, demos)"
    )
    
    args = parser.parse_args()
    
    orchestrator = QuantPipelineOrchestrator()
    
    try:
        orchestrator.run_full_pipeline(skip_data=args.skip_data)
    except KeyboardInterrupt:
        print("\n\n🛑 Pipeline interrupted by user")
        orchestrator.finish_pipeline()
    except Exception as e:
        print(f"\n\n❌ Pipeline failed with error: {e}")
        orchestrator.finish_pipeline()


if __name__ == "__main__":
    main()