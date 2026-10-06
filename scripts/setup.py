#!/usr/bin/env python3
"""One-command automated environment setup, dataset synthesis, model training, and report generation."""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_python_version():
    print("=" * 60)
    print(" STEP 1: Verifying Python Runtime")
    print("=" * 60)
    version = sys.version_info
    print(f"Detected Python version: {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("[WARNING] Python 3.10+ is recommended for this project.")
    else:
        print("[OK] Python version compatible.\n")


def create_directories():
    print("=" * 60)
    print(" STEP 2: Creating Project Directory Structure")
    print("=" * 60)
    dirs = [
        "data/raw", "data/processed", "data/synthetic",
        "models", "notebooks", "src", "api", "demo", "scripts", "tests", "reports",
    ]
    for d in dirs:
        p = PROJECT_ROOT / d
        p.mkdir(parents=True, exist_ok=True)
        print(f" [CREATED/EXISTS] {d}")
    print("[OK] All directories prepared.\n")


def check_dependencies():
    print("=" * 60)
    print(" STEP 3: Checking Python Dependencies")
    print("=" * 60)
    modules = ["numpy", "pandas", "scipy", "yaml", "fastapi", "uvicorn", "pydantic"]
    missing = []
    for mod in modules:
        try:
            __import__(mod)
            print(f" [OK] {mod} is available.")
        except ImportError:
            print(f" [MISSING] {mod}")
            missing.append(mod)

    if missing:
        print("\nNotice: Some packages are missing. To install full dependencies in your environment, run:")
        print("  pip install -r requirements.txt\n")
    else:
        print("[OK] Core dependencies verified.\n")


def run_pipeline():
    print("=" * 60)
    print(" STEP 4: Executing Data Generation, Training & Evaluation")
    print("=" * 60)
    from scripts.train_model import run_pipeline as train_main
    train_main()
    print("[OK] End-to-end pipeline finished successfully.\n")


def print_instructions():
    print("=" * 60)
    print(" SETUP COMPLETE! NEXT STEPS:")
    print("=" * 60)
    print("1. To launch the interactive demo and API:")
    print("   python -m uvicorn api.server:app --reload")
    print("\n2. Then open your browser at:")
    print("   http://localhost:8000")
    print("\n3. To run automated tests:")
    print("   python -m unittest discover tests")
    print("=" * 60)


if __name__ == "__main__":
    check_python_version()
    create_directories()
    check_dependencies()
    run_pipeline()
    print_instructions()
