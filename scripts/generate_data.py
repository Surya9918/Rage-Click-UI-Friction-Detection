#!/usr/bin/env python3
"""Script to generate synthetic user interaction telemetry."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_generation import SyntheticTelemetryGenerator
from src.utils import get_logger

logger = get_logger("generate_data")

def main():
    logger.info("Initializing synthetic data generation...")
    generator = SyntheticTelemetryGenerator()

    output_dir = PROJECT_ROOT / "data" / "synthetic"
    features_csv = generator.save_synthetic_dataset(output_dir)

    print(f"\n[SUCCESS] Synthetic interaction features saved: {features_csv}")

if __name__ == "__main__":
    main()
