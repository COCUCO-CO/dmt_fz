# Graph Attention Networks for EEG Brain State Classification

## 📋 Overview

This module implements an end-to-end pipeline for classifying brain states (DMT vs EC vs EO) using **Graph Attention Networks (GAT)** trained on EEG synchronization graphs.

### Why Graph Neural Networks?

Unlike standard deep learning approaches (e.g., EEGNet), this method:
- ✅ Uses **pre-computed synchronization matrices** from your pipeline
- ✅ Respects the **natural graph structure** of brain connectivity
- ✅ Incorporates **Kuramoto order parameters** and **metastability** as features
- ✅ Provides **interpretable attention weights** showing which connections matter
- ✅ Is **coherent with your neuroscience analysis** (phase synchronization, networks)

---

## 🏗️ Architecture

```
EEG Epoch (phases-*.pkl)
    ↓
Synchronization Matrix (NxN)  →  Graph with weighted edges
    +
Node Features (phase stats, amplitude, complexity)
    +
Graph Features (Kuramoto coherence, metastability)
    ↓
Graph Attention Network (4 layers, 8 heads)
    ↓
Attention Pooling
    ↓
MLP Classifier
    ↓
Output: [DMT, EC, EO]
```

### Key Components

1. **Graph Construction** (`data/dataset_builder.py`)
   - Nodes: EEG channels or brain parcels (from source localization)
   - Edges: Synchronization strength > threshold
   - Node features: Phase/amplitude statistics, temporal complexity
   - Edge features: Synchronization value
   - Graph features: Kuramoto mean/std, topology metrics

2. **GAT Model** (`models/gat_model.py`)
   - Multi-head attention with edge attributes
   - Residual connections for deep networks
   - Flexible pooling (attention, mean+max, Set2Set)
   - MLP classifier with batch normalization

3. **Training** (`train.py`)
   - TensorBoard logging
   - Early stopping
   - Learning rate scheduling
   - Gradient clipping
   - Checkpointing

---

## 📂 Directory Structure

```
machine_learning/clf/
├── config/
│   └── config.yaml              # Configuration file (EDIT THIS)
├── data/
│   ├── __init__.py
│   ├── dataset_builder.py       # Graph construction from phases-*.pkl
│   └── cache/                   # Dataset cache (auto-generated)
├── models/
│   ├── __init__.py
│   └── gat_model.py             # GAT architecture
├── utils/
│   ├── __init__.py
│   ├── logger.py                # Logging utilities
│   └── visualization.py         # Plotting functions
├── analysis/
│   └── analyze_graphs.py        # Statistical analysis DMT vs EC
├── checkpoints/                 # Model checkpoints (auto-generated)
├── output/                      # Results and logs (auto-generated)
├── runs/                        # TensorBoard logs (auto-generated)
├── train.py                     # Main training script
├── run_pipeline.sh              # End-to-end execution script
└── README.md                    # This file
```

---

## 🚀 Quick Start

### ⚠️ IMPORTANT: Per-Band Training Strategy

**Recommended approach:** Train separate models for each frequency band to avoid confounding band-specific patterns with condition-specific patterns.

**Quick test with Alpha band (30-60 min):**
```bash
cd /media/storage_hdd/dmt_fz/machine_learning/clf
./quick_train_alpha.sh
```

**Train all bands (2-5 hours):**
```bash
python train_per_band.py
```

📖 **See `PER_BAND_GUIDE.md` for detailed explanation of why this is methodologically correct.**

### Alternative: Mixed Multi-Band Training (Not Recommended)

If you want to quickly test the pipeline with all bands mixed (exploratory only):

```bash
cd /media/storage_hdd/dmt_fz/machine_learning/clf
chmod +x run_pipeline.sh
./run_pipeline.sh
```

⚠️ **Note:** This mixes graphs from different bands in the same dataset, which confounds band-specific with condition-specific patterns. Use only for initial exploration.

### Step 3: View Results

**Per-band training (recommended):**
```bash
# View results for each band
cat output_alpha/test_results.json
cat output_theta/test_results.json
# ... etc

# View comparison across all bands
cat band_comparison_results.json

# TensorBoard for specific band
tensorboard --logdir=runs_alpha

# Or view all bands together
tensorboard --logdir_spec=Delta:runs_delta,Theta:runs_theta,Alpha:runs_alpha,Beta:runs_beta,Gamma:runs_gamma
```

**Mixed training (if used):**
```bash
tensorboard --logdir=runs
cat output/test_results.json
```

**Output files (per-band):**
- `output_alpha/test_results.json` - Alpha band metrics
- `output_alpha/confusion_matrix.png` - Alpha confusion matrix
- `checkpoints_alpha/best_model.pt` - Alpha trained model
- `band_comparison_results.json` - Summary across all bands

---

## 📊 Configuration Guide

### Key Parameters

#### Graph Construction (`data.graph`)

```yaml
edge_threshold: 0.3  # Minimum synchronization value to create edge
# Lower = more edges (denser graph)
# Higher = fewer edges (sparser graph)
# Typical range: 0.2 - 0.5
```

**How to choose:**
- Start with 0.3 (moderate density)
- Check `output/visualizations/graph_statistics.png`
- If graphs too sparse → lower threshold
- If graphs too dense → higher threshold

#### Node Features (`data.node_features`)

```yaml
use_phase_stats: true         # Mean/std of instantaneous phase
use_amplitude_stats: true     # Mean/std of amplitude envelope
use_temporal_complexity: true # Entropy, kurtosis, etc.
use_network_label: false      # One-hot encoding of brain network
```

**Recommended:** Keep all `true` for maximum information

#### Model Architecture (`model.architecture`)

```yaml
hidden_dim: 128              # Hidden layer dimension (64-256)
num_gat_layers: 4            # Number of GAT layers (2-6)
num_attention_heads: 8       # Attention heads (4-16)
dropout: 0.4                 # Dropout rate (0.3-0.6)
```

**For small datasets:** Reduce `hidden_dim` and `num_gat_layers`  
**For large datasets:** Increase for more capacity

#### Training (`training`)

```yaml
num_epochs: 300              # Maximum epochs
batch_size: 32               # Batch size (adjust for GPU memory)
learning_rate: 0.001         # Initial learning rate
early_stopping:
  patience: 50               # Stop if no improvement for N epochs
```

---

## 🔧 Advanced Usage

### Per-Band Training (Recommended)

Train separate models for each frequency band:

```bash
# Train all bands
python train_per_band.py

# Train specific bands only
python train_per_band.py --bands Alpha Theta

# Force rebuild datasets
python train_per_band.py --force-rebuild
```

### Ensemble Predictions

Combine predictions from multiple band-specific models:

```bash
# Average probabilities (default)
python ensemble_bands.py --method average

# Weighted average (by validation accuracy)
python ensemble_bands.py --method weighted

# Majority voting
python ensemble_bands.py --method voting
```

### Train with Custom Config

```bash
python train.py --config my_config.yaml
```

### Force Rebuild Dataset

If you modified `phases-*.pkl` files or changed graph construction parameters:

```bash
python train.py --force-rebuild
```

### Analyze Graphs Only

To run statistical analysis without training:

```bash
python analysis/analyze_graphs.py --config config/config.yaml
```

### Use Pre-trained Model

```python
import torch
from models import BrainStateGAT

# Load checkpoint
checkpoint = torch.load('checkpoints/best_model.pt')
config = checkpoint['config']

# Create model
model = create_model_from_config(config, ...)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Predict
with torch.no_grad():
    output = model(graph_data)
    prediction = output.argmax(dim=1)
```

---

## 📈 Interpreting Results

### Training Curves

Monitor `output/training_curves.png`:
- **Overfitting:** Val loss increases while train loss decreases
  - Solution: Increase dropout, reduce model size, or add more data
- **Underfitting:** Both losses plateau at high values
  - Solution: Increase model capacity or train longer

### Confusion Matrix

`output/confusion_matrix.png` shows per-class performance:
- Diagonal = correct predictions
- Off-diagonal = confusions between classes
- Look for systematic confusions (e.g., DMT often classified as EC)

### Graph Statistics

`output/analysis/` contains:
- `graph_features_all.csv` - All extracted features
- `comparison_DMT_vs_EC.csv` - Statistical tests
- `feature_distributions.png` - Histograms by condition
- `band_comparison_*.png` - Differences across frequency bands

**Key features to check:**
- `kuramoto_mean`: Higher = more coherent (synchronized)
- `kuramoto_std`: Higher = more metastable (fluctuating)
- `density`: Graph connectivity
- `mean_sync`: Average synchronization strength

---

## 🐛 Troubleshooting

### Issue: "No graphs were created"

**Causes:**
- Incorrect `phases_dir` path
- Missing `phases-*.pkl` files
- Wrong `use_stc` setting (should match your data)

**Fix:**
```bash
# Check if files exist
ls /media/storage_hdd/dmt_fz/fwd-inv-stc/DMT/phases-*.pkl

# Verify config
cat config/config.yaml | grep phases_dir
```

### Issue: "CUDA out of memory"

**Fix:** Reduce batch size in `config.yaml`:
```yaml
training:
  batch_size: 16  # or 8
```

### Issue: "All graphs have no edges"

**Fix:** Lower edge threshold:
```yaml
data:
  graph:
    edge_threshold: 0.1  # was 0.3
```

### Issue: Model not learning (accuracy ~33%)

**Possible causes:**
- Learning rate too high → reduce to 0.0001
- Model too small → increase hidden_dim
- Data imbalance → check class distribution
- Features not informative → try different node/graph features

---

## 🔬 Understanding Your Features

### From `phases-*.pkl` Files

Your pipeline generates these files with structure:
```python
{
    'syncros_stc': {band: [NxN matrices]},    # → Graph edges
    'phases_stc': {band: [NxT arrays]},       # → Node features (phase)
    'amplitudes_stc': {band: [NxT arrays]},   # → Node features (amplitude)
    'kuramoto_stc': {band: [T arrays]}        # → Graph features (coherence)
}
```

### What the Model Learns

1. **Local patterns:** Which brain regions synchronize together
2. **Global patterns:** Overall coherence and metastability
3. **Frequency-specific:** Different patterns in different bands
4. **Temporal:** Statistics over time within each epoch

---

## 📊 Comparison with Other Methods

| Method | Input | Strengths | Weaknesses |
|--------|-------|-----------|------------|
| **EEGNet (current)** | Raw EEG | End-to-end learning | Black box, ignores sync |
| **Random Forest (current)** | PSD features | Interpretable | Loses spatial structure |
| **GAT (this module)** | Sync graphs | Uses your pipeline, interpretable attention | Requires graph construction |

**When to use GAT:**
- You want to leverage your synchronization analysis
- You need interpretability (which connections matter)
- You want to compare with your Pearson correlations

**When to use EEGNet:**
- You want maximum predictive power
- You don't care about mechanistic interpretation

---

## 🎯 Expected Performance

Baseline accuracies (3-class: DMT/EC/EO):
- **Random:** 33.3%
- **Good model:** 70-85%
- **Excellent model:** >85%

If you get <60%, check:
1. Class balance (are classes roughly equal?)
2. Data quality (are epochs correctly labeled?)
3. Hyperparameters (especially learning rate and dropout)

---

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{veličković2018graph,
  title={Graph attention networks},
  author={Veli{\v{c}}kovi{\'c}, Petar and Cucurull, Guillem and Casanova, Arantxa and Romero, Adriana and Li{\`o}, Pietro and Bengio, Yoshua},
  journal={ICLR},
  year={2018}
}
```

---

## 💡 Tips for Best Results

1. **Start simple:** Train on one frequency band (Alpha) first
2. **Check data:** Run `analyze_graphs.py` before training
3. **Monitor TensorBoard:** Watch for overfitting early
4. **Tune threshold:** Graph density strongly affects performance
5. **Try ensembles:** Train separate models per band, then combine
6. **Compare with baselines:** Your Pearson correlation results

---

## 🤝 Contributing

To add new features:
1. Add configuration in `config/config.yaml`
2. Modify `dataset_builder.py` for new graph features
3. Adjust `gat_model.py` if architecture changes needed
4. Update this README

---

## 📧 Questions?

For issues specific to this module:
- Check TensorBoard logs: `tensorboard --logdir=runs`
- Check training log: `output/logs/train_*.log`
- Verify configuration: `cat config/config.yaml`

For questions about the overall pipeline:
- See main documentation: `/media/storage_hdd/dmt_fz/docs/README.md`

---

**Last updated:** 2025-01-13  
**Compatible with:** PyTorch 2.0+, PyTorch Geometric 2.3+

