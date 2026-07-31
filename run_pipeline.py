"""
run_pipeline.py
---------------
Master script: runs data preparation → BYOL pre-training → analysis head training
in sequence. Run this from the project root.

Usage:
    python run_pipeline.py
"""
import subprocess
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def run(script: str):
    print(f"\n{'='*60}")
    print(f"  RUNNING: {script}")
    print(f"{'='*60}")
    result = subprocess.run(
        [sys.executable, script],
        cwd=PROJECT_ROOT,
        env={**os.environ, 'PYTHONPATH': PROJECT_ROOT}
    )
    if result.returncode != 0:
        print(f"\n[ERROR] {script} exited with code {result.returncode}. Stopping.")
        sys.exit(result.returncode)
    print(f"\n[✓] {script} completed successfully.")

if __name__ == "__main__":
    run("data/prepare_dataset.py")
    run("ssl_training/train_byol.py")
    run("ssl_training/train_analysis_heads.py")
    run("feature_extraction/extract_embeddings.py")
    run("clustering/discover_sites.py")
    run("classification/train_classifier.py")
    print("\n" + "="*60)
    print("  ALL 6 PIPELINE STAGES COMPLETE!")
    print("  Launch dashboard with:")
    print("  streamlit run dashboard/streamlit_app.py")
    print("="*60)
