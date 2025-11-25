#!/usr/bin/env python3
"""
Quick validation script to run before training.
Exits with code 1 if critical errors are found.

Usage:
    python tests/validate_before_train.py --config config/config.yaml
    
Or from train.py:
    from tests.test_dataset import DatasetValidator
    validator = DatasetValidator(train, val, test, class_names)
    if not validator.validate_all():
        raise ValueError("Dataset validation failed!")
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from test_dataset import validate_dataset_from_config


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate dataset before training')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--strict', action='store_true',
                       help='Fail on warnings too')
    
    args = parser.parse_args()
    
    print("\n" + "🔍 " * 20)
    print("PRE-TRAINING DATASET VALIDATION")
    print("🔍 " * 20)
    
    success = validate_dataset_from_config(args.config)
    
    if success:
        print("\n✅ Dataset is ready for training!")
        sys.exit(0)
    else:
        print("\n🚫 Dataset has critical errors. Fix before training!")
        sys.exit(1)


if __name__ == '__main__':
    main()

