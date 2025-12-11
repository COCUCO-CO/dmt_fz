#!/usr/bin/env python3
"""
Quick test runner for EEG Cleaning pipeline tests.

Usage:
    python tests/run_tests.py           # Run all tests
    python tests/run_tests.py --fast    # Run fast tests only (no real EEG data)
    python tests/run_tests.py --module state  # Run specific module tests
"""

import sys
import subprocess
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Run EEG Cleaning tests')
    parser.add_argument('--fast', action='store_true', 
                       help='Run fast tests only (synthetic data)')
    parser.add_argument('--module', type=str, 
                       help='Run tests for specific module (state, filters, bad_channels, etc.)')
    parser.add_argument('--integration', action='store_true',
                       help='Run integration tests only')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output')
    parser.add_argument('-x', '--exitfirst', action='store_true',
                       help='Exit on first failure')
    
    args = parser.parse_args()
    
    # Build pytest command
    cmd = ['python', '-m', 'pytest']
    
    # Test directory
    test_dir = Path(__file__).parent
    
    # Select tests
    if args.module:
        test_file = test_dir / f'test_{args.module}.py'
        if not test_file.exists():
            print(f"Error: Test file not found: {test_file}")
            sys.exit(1)
        cmd.append(str(test_file))
    elif args.integration:
        cmd.append(str(test_dir / 'test_integration.py'))
    else:
        cmd.append(str(test_dir))
    
    # Options
    if args.verbose:
        cmd.append('-v')
    else:
        cmd.append('-v')  # Always verbose for now
    
    if args.fast:
        cmd.extend(['-m', 'not slow'])
    
    if args.exitfirst:
        cmd.append('-x')
    
    # Add timing
    cmd.append('--durations=10')
    
    # Print command
    print(f"Running: {' '.join(cmd)}")
    print("-" * 60)
    
    # Run
    result = subprocess.run(cmd, cwd=test_dir.parent)
    
    sys.exit(result.returncode)


if __name__ == '__main__':
    main()



