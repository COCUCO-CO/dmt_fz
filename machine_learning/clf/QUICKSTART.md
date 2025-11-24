# Quick Start Guide - 5 Minutes to Results

## Prerequisites Check

```bash
# Ensure you're in the right directory
cd /media/storage_hdd/dmt_fz/machine_learning/clf

# Verify Python environment
python --version  # Should be 3.8+

# Check if packages installed
python -c "import torch, torch_geometric; print('✓ Ready')"
```

If packages not found, run:
```bash
pip install -r requirements.txt
```

---

## Step 1: Verify Data (30 seconds)

Check that your `phases-*.pkl` files exist:

```bash
ls /media/storage_hdd/dmt_fz/fwd-inv-stc/DMT/phases-*.pkl | wc -l
ls /media/storage_hdd/dmt_fz/fwd-inv-stc/EC/phases-*.pkl | wc -l
ls /media/storage_hdd/dmt_fz/fwd-inv-stc/EO/phases-*.pkl | wc -l
```

You should see ~29 files for each condition.

---

## Step 2: Configure (1 minute)

Edit `config/config.yaml` if needed:

```bash
nano config/config.yaml  # or your favorite editor
```

**Key settings to check:**
- `paths.phases_dir` points to your data
- `data.use_stc: true` for source space (or `false` for electrode space)
- `training.num_epochs: 300` (reduce to 50 for quick test)

---

## Step 3: Run Pipeline (2 minutes)

Execute the full pipeline:

```bash
./run_pipeline.sh
```

This will:
1. Build graph dataset from your phases files
2. Analyze graph statistics
3. Train GAT model
4. Generate visualizations

**Expected output:**
```
================================================================================
          Graph Attention Networks for EEG Brain State Classification
================================================================================

[INFO] Using configuration: config/config.yaml
[INFO] Python version: Python 3.10.x
...
```

---

## Step 4: Monitor Training (Optional)

In another terminal, start TensorBoard:

```bash
cd /media/storage_hdd/dmt_fz/machine_learning/clf
tensorboard --logdir=runs
```

Open browser: http://localhost:6006

---

## Step 5: Check Results (1 minute)

After training completes:

```bash
# View test accuracy
cat output/test_results.json

# See confusion matrix
eog output/confusion_matrix.png  # or your image viewer

# Check training curves
eog output/training_curves.png
```

Expected test accuracy: **70-85%** (random baseline is 33%)

---

## What's Happening?

1. **Graph Construction**: Each EEG epoch → graph
   - Nodes = brain regions/channels
   - Edges = synchronization strength
   - Features = phase/amplitude statistics, Kuramoto parameters

2. **Model Training**: Graph Attention Network learns which connections discriminate between DMT/EC/EO

3. **Evaluation**: Test set performance + visualizations

---

## Quick Test (Faster)

To test on a subset without full training:

```bash
# Edit config to use only Alpha band
nano config/config.yaml
# Change: bands: ["Alpha"]

# Train with fewer epochs
python train.py --config config/config.yaml
```

---

## Troubleshooting

### "No graphs were created"
→ Check `phases_dir` path in config

### "CUDA out of memory"
→ Reduce `batch_size` to 16 or 8

### "Module not found"
→ Install requirements: `pip install -r requirements.txt`

### Model accuracy stuck at ~33%
→ Check if data is loading correctly:
```bash
python analysis/analyze_graphs.py
# Should show differences between DMT/EC/EO
```

---

## Next Steps

After successful run:

1. **Analyze Results**:
   ```bash
   # Statistical comparison
   ls output/analysis/comparison_*.csv
   
   # Attention weights
   python analysis/visualize_attention.py --checkpoint checkpoints/best_model.pt
   ```

2. **Tune Hyperparameters**: Edit `config/config.yaml`
   - Try different `edge_threshold` values
   - Adjust `hidden_dim` and `num_gat_layers`
   - Experiment with different frequency bands

3. **Compare with Other Methods**: Check how GAT compares with your:
   - Pearson correlation results (in `pearson_results/`)
   - EEGNet models (in `EEGNet/`)
   - Random Forest predictions (in `spectral_sources/`)

---

## Typical Workflow

```bash
# 1. Quick analysis
python analysis/analyze_graphs.py

# 2. Train model
./run_pipeline.sh

# 3. View results
tensorboard --logdir=runs
cat output/test_results.json

# 4. Visualize attention
python analysis/visualize_attention.py --checkpoint checkpoints/best_model.pt

# 5. Detailed evaluation
python evaluate.py --checkpoint checkpoints/best_model.pt
```

---

## Full Documentation

See `README.md` for complete documentation including:
- Detailed configuration guide
- Architecture explanation
- Advanced usage
- Interpretation tips

See `SETUP.md` for installation details.

---

**Time to first results: ~5-10 minutes**  
**Full training (29 subjects, 5 bands): ~30-60 minutes on GPU**

