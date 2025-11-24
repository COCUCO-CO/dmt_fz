#!/usr/bin/env python3
"""
Train separate models for each frequency band.

This script trains one GAT model per frequency band (Delta, Theta, Alpha, Beta, Gamma)
to avoid mixing graphs from different bands in the same dataset.

This is the methodologically correct approach because:
1. Each band represents different neural processes
2. Avoids confounding band-specific patterns with condition-specific patterns
3. Allows comparison of which bands are most informative
4. Results can be combined via ensemble methods
"""

import argparse
import sys
import yaml
import shutil
from pathlib import Path
from typing import Dict, Any
import logging
import json
import subprocess

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

logger = logging.getLogger(__name__)


def train_single_band(config: Dict[str, Any], band: str, force_rebuild: bool = False) -> Dict[str, Any]:
    """
    Train a model for a single frequency band.
    
    Args:
        config: Configuration dictionary
        band: Frequency band name (Delta, Theta, Alpha, Beta, Gamma)
        force_rebuild: Whether to force dataset rebuild
        
    Returns:
        Dictionary with training results
    """
    print(f"\n{'='*80}")
    print(f"Training model for band: {band}")
    print(f"{'='*80}\n")
    
    # Modify config to use only this band
    band_config = config.copy()
    band_config['data']['bands'] = [band]
    
    # Update output paths to include band name
    band_suffix = f"_{band.lower()}"
    band_config['paths']['output_dir'] = str(Path(config['paths']['output_dir']).parent / f"output{band_suffix}")
    band_config['paths']['checkpoints'] = str(Path(config['paths']['checkpoints']).parent / f"checkpoints{band_suffix}")
    band_config['paths']['tensorboard'] = str(Path(config['paths']['tensorboard']).parent / f"runs{band_suffix}")
    
    # Save temporary config
    temp_config_path = Path(f"config/config_temp_{band.lower()}.yaml")
    with open(temp_config_path, 'w') as f:
        yaml.dump(band_config, f, default_flow_style=False)
    
    # Run training
    cmd = ["python", "train.py", "--config", str(temp_config_path)]
    if force_rebuild:
        cmd.append("--force-rebuild")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        
        # Load results
        results_file = Path(band_config['paths']['output_dir']) / "test_results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                results = json.load(f)
        else:
            results = {"status": "completed", "results_file_not_found": True}
        
        # Clean up temp config
        if temp_config_path.exists():
            temp_config_path.unlink()
        
        return {
            "band": band,
            "status": "success",
            "results": results
        }
        
    except subprocess.CalledProcessError as e:
        print(f"Error training {band}: {e}")
        
        # Clean up temp config
        if temp_config_path.exists():
            temp_config_path.unlink()
        
        return {
            "band": band,
            "status": "failed",
            "error": str(e)
        }


def compare_band_results(base_output_dir: Path, bands: list):
    """
    Compare results across all trained bands.
    
    Args:
        base_output_dir: Base output directory
        bands: List of band names
    """
    print(f"\n{'='*80}")
    print("Comparing Results Across Frequency Bands")
    print(f"{'='*80}\n")
    
    results_summary = []
    
    for band in bands:
        results_file = base_output_dir.parent / f"output_{band.lower()}" / "test_results.json"
        
        if results_file.exists():
            with open(results_file, 'r') as f:
                data = json.load(f)
                
            results_summary.append({
                "Band": band,
                "Test Accuracy": f"{data.get('test_accuracy', 0)*100:.2f}%",
                "Test F1": f"{data.get('test_f1', 0):.4f}",
                "Test Precision": f"{data.get('test_precision', 0):.4f}",
                "Test Recall": f"{data.get('test_recall', 0):.4f}",
            })
        else:
            results_summary.append({
                "Band": band,
                "Test Accuracy": "N/A",
                "Test F1": "N/A",
                "Test Precision": "N/A",
                "Test Recall": "N/A",
            })
    
    # Print table
    print(f"{'Band':<10} {'Accuracy':<12} {'F1 Score':<12} {'Precision':<12} {'Recall':<12}")
    print("-" * 68)
    
    for result in results_summary:
        print(f"{result['Band']:<10} {result['Test Accuracy']:<12} {result['Test F1']:<12} "
              f"{result['Test Precision']:<12} {result['Test Recall']:<12}")
    
    # Save comparison to file
    comparison_file = base_output_dir.parent / "band_comparison_results.json"
    with open(comparison_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n✓ Comparison saved to: {comparison_file}")
    
    # Find best band
    accuracies = []
    for i, result in enumerate(results_summary):
        try:
            acc = float(result['Test Accuracy'].rstrip('%'))
            accuracies.append((bands[i], acc))
        except:
            pass
    
    if accuracies:
        best_band, best_acc = max(accuracies, key=lambda x: x[1])
        print(f"\n🏆 Best performing band: {best_band} ({best_acc:.2f}% accuracy)")


def main():
    parser = argparse.ArgumentParser(description='Train GAT models per frequency band')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--force-rebuild', action='store_true',
                       help='Force rebuild of datasets')
    parser.add_argument('--bands', type=str, nargs='+', 
                       default=None,
                       help='Specific bands to train (default: all bands from config)')
    parser.add_argument('--skip-comparison', action='store_true',
                       help='Skip final comparison of results')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get bands to train
    if args.bands:
        bands = args.bands
    else:
        bands = config['data']['bands']
    
    print(f"\n{'='*80}")
    print("Training Separate Models for Each Frequency Band")
    print(f"{'='*80}")
    print(f"\nBands to train: {', '.join(bands)}")
    print(f"Total models: {len(bands)}")
    print(f"Force rebuild: {args.force_rebuild}")
    
    # Train each band
    all_results = []
    
    for band in bands:
        result = train_single_band(config, band, args.force_rebuild)
        all_results.append(result)
    
    # Print summary
    print(f"\n{'='*80}")
    print("Training Summary")
    print(f"{'='*80}\n")
    
    for result in all_results:
        status_symbol = "✓" if result['status'] == 'success' else "✗"
        print(f"{status_symbol} {result['band']:<10} - {result['status']}")
    
    # Compare results
    if not args.skip_comparison:
        base_output_dir = Path(config['paths']['output_dir'])
        compare_band_results(base_output_dir, bands)
    
    print(f"\n{'='*80}")
    print("All Done!")
    print(f"{'='*80}\n")
    
    print("Next steps:")
    print("1. View results for each band:")
    for band in bands:
        print(f"   - output_{band.lower()}/test_results.json")
    
    print("\n2. View TensorBoard logs:")
    for band in bands:
        print(f"   - tensorboard --logdir=runs_{band.lower()}")
    
    print("\n3. Compare band results:")
    print(f"   - cat band_comparison_results.json")
    
    print("\n4. Optional: Create ensemble from all bands")


if __name__ == '__main__':
    main()

