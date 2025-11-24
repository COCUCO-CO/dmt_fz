# Per-Band Training Strategy Guide

## 🎯 Why Train Per Band?

### The Problem with Multi-Band Training

When mixing graphs from all frequency bands in a single dataset:

```
Dataset contains:
- Graph 1: DMT, Alpha band, high sync
- Graph 2: DMT, Delta band, low sync  
- Graph 3: EC, Alpha band, high sync
```

**Problem:** The model doesn't know which band each graph comes from! 

Graph 1 and Graph 3 (both Alpha) might look more similar to each other than Graph 1 and Graph 2 (both DMT but different bands). This confounds **band-specific patterns** with **condition-specific patterns**.

### The Solution: Separate Models Per Band

Train 5 independent models:
- **Model_Delta**: Only Delta graphs (DMT vs EC vs EO)
- **Model_Theta**: Only Theta graphs (DMT vs EC vs EO)
- **Model_Alpha**: Only Alpha graphs (DMT vs EC vs EO)
- **Model_Beta**: Only Beta graphs (DMT vs EC vs EO)
- **Model_Gamma**: Only Gamma graphs (DMT vs EC vs EO)

**Benefits:**
1. ✅ **Methodologically correct**: No confounding
2. ✅ **Interpretable**: "Alpha band is 85% accurate, Delta is 70%"
3. ✅ **Neuroscience insight**: Which bands carry DMT signature?
4. ✅ **Ensemble-ready**: Can combine predictions later
5. ✅ **Comparable**: See which bands are most informative

---

## 🚀 Quick Start

### Option A: Train All Bands Sequentially

```bash
cd /media/storage_hdd/dmt_fz/machine_learning/clf

# Train all 5 bands (will take ~2-5 hours total)
python train_per_band.py
```

**Output structure:**
```
machine_learning/clf/
├── checkpoints_delta/
│   └── best_model.pt
├── checkpoints_theta/
│   └── best_model.pt
├── checkpoints_alpha/
│   └── best_model.pt
├── checkpoints_beta/
│   └── best_model.pt
├── checkpoints_gamma/
│   └── best_model.pt
├── output_delta/
│   └── test_results.json
├── output_theta/
│   └── test_results.json
├── output_alpha/
│   └── test_results.json
├── output_beta/
│   └── test_results.json
├── output_gamma/
│   └── test_results.json
├── runs_delta/      # TensorBoard logs
├── runs_theta/
├── runs_alpha/
├── runs_beta/
├── runs_gamma/
└── band_comparison_results.json  # Summary comparison
```

### Option B: Train Specific Bands Only

```bash
# Train only Alpha and Beta
python train_per_band.py --bands Alpha Beta
```

### Option C: Train One Band at a Time

```bash
# Start with Alpha (most studied in neuroscience)
python train_per_band.py --bands Alpha

# Check results
cat output_alpha/test_results.json

# If good, continue with others
python train_per_band.py --bands Theta Beta Gamma Delta
```

---

## 📊 Viewing Results

### 1. View Individual Band Results

```bash
# Alpha band results
cat output_alpha/test_results.json
```

**Example output:**
```json
{
  "test_accuracy": 0.82,
  "test_f1": 0.81,
  "test_precision": 0.83,
  "test_recall": 0.80
}
```

### 2. View Comparison Across All Bands

```bash
cat band_comparison_results.json
```

**Example output:**
```
Band       Accuracy     F1 Score     Precision    Recall      
--------------------------------------------------------------------
Delta      70.45%       0.6892       0.7021       0.6798
Theta      75.32%       0.7456       0.7589       0.7312
Alpha      82.15%       0.8123       0.8287       0.7989      ← Best!
Beta       77.89%       0.7701       0.7834       0.7598
Gamma      68.23%       0.6745       0.6891       0.6612
```

### 3. View TensorBoard Logs

```bash
# View all bands together (in different tabs)
tensorboard --logdir_spec=\
Delta:runs_delta,\
Theta:runs_theta,\
Alpha:runs_alpha,\
Beta:runs_beta,\
Gamma:runs_gamma

# Or view individually
tensorboard --logdir=runs_alpha
```

---

## 🔬 Ensemble Methods

After training individual bands, combine their predictions:

### Method 1: Average Probabilities (Default)

```bash
python ensemble_bands.py --method average
```

### Method 2: Weighted Average (Use Best Bands More)

```bash
# Weight by validation accuracy
python ensemble_bands.py --method weighted
```

### Method 3: Majority Voting

```bash
# Each band votes for one class
python ensemble_bands.py --method voting
```

### Method 4: Max Probability

```bash
# Take maximum confidence across bands
python ensemble_bands.py --method max
```

---

## 📈 Expected Results

### Individual Band Performance (DMT vs EC vs EO)

| Band | Expected Accuracy | Neuroscience Rationale |
|------|------------------|------------------------|
| **Alpha** | 75-85% | Dominant resting state rhythm, strongly modulated by psychedelics |
| **Theta** | 70-80% | Memory, navigation, altered states |
| **Beta** | 70-80% | Attention, active processing |
| **Delta** | 65-75% | Deep states, unconscious processing |
| **Gamma** | 60-70% | Binding, consciousness (more variable) |

### Ensemble Performance

- **Expected:** 80-90% (2-5% improvement over best single band)
- **Minimum:** Should match or exceed best individual band
- **If worse:** Check that test sets are identical across bands

---

## 🧪 Scientific Interpretation

### Questions You Can Answer:

1. **Which band is most informative for DMT classification?**
   - Look at individual accuracies in `band_comparison_results.json`

2. **Does DMT affect all bands equally?**
   - Compare accuracies: high accuracy = strong band-specific signature

3. **Which bands are most similar/different?**
   - Look at correlation of errors between bands

4. **Can we predict DMT from a single band?**
   - If one band has >80% accuracy, yes!

5. **Do bands provide complementary information?**
   - If ensemble > best single band → yes
   - If ensemble ≈ best single band → redundant

---

## 🔧 Configuration

### Modify Training Parameters Per Band

If you want different hyperparameters for each band:

```python
# Edit train_per_band.py, add band-specific configs:

if band == "Gamma":
    # Gamma needs more regularization (higher variability)
    band_config['training']['dropout'] = 0.6
    
elif band == "Delta":
    # Delta might need more epochs
    band_config['training']['num_epochs'] = 500
```

### Computational Considerations

**Time estimates (on GPU):**
- Per band: 30-60 minutes
- All 5 bands: 2.5-5 hours
- Ensemble evaluation: 5-10 minutes

**Storage:**
- Per band checkpoint: ~100 MB
- Total (5 bands): ~500 MB checkpoints + ~1 GB outputs

**Memory:**
- Same as single-band training (~4-8 GB GPU)
- Bands are trained sequentially (not parallel)

---

## 🎓 Advanced: Custom Band Combinations

### Train on Band Combinations

```python
# Edit config to combine bands
data:
  bands: ["Alpha", "Beta"]  # Only Alpha + Beta
```

This creates graphs with features from multiple bands **combined** rather than separate.

**When to use:**
- Exploring specific band interactions
- Limited computational resources
- Targeting specific neural processes

---

## 📚 Comparison with Other Approaches

### Approach 1: Mixed Multi-Band Dataset (Original)
❌ **Problem:** Confounds band and condition  
✅ **Pro:** Single model, faster training  
👉 **Use when:** Exploratory phase only

### Approach 2: Per-Band Models (This Guide)
✅ **Pro:** Methodologically correct, interpretable  
❌ **Con:** More training time, more storage  
👉 **Use when:** Final analysis, publication

### Approach 3: Band as One-Hot Feature
✅ **Pro:** Single model learns band-specific patterns  
❌ **Con:** Complex, harder to interpret  
👉 **Use when:** Want single model but correct approach

### Approach 4: Hierarchical Multi-Band Fusion
✅ **Pro:** End-to-end learning across bands  
❌ **Con:** Most complex, requires careful design  
👉 **Use when:** Maximum performance needed

---

## 🐛 Troubleshooting

### Issue: "Dataset cache exists for different bands"

```bash
# Force rebuild
python train_per_band.py --force-rebuild
```

### Issue: One band has very low accuracy (<50%)

**Possible causes:**
1. Bad data quality in that band
2. Lower signal-to-noise ratio (especially Gamma)
3. Hyperparameters not optimal for that band

**Solutions:**
- Check raw data quality
- Adjust band-specific hyperparameters
- Try different preprocessing

### Issue: Ensemble worse than best single band

**Possible causes:**
1. Test sets not identical (different random seeds)
2. Weak bands dragging down ensemble
3. Overfitting in individual models

**Solutions:**
```bash
# Use only top 3 bands
python ensemble_bands.py --bands Alpha Theta Beta
```

---

## 💡 Tips for Best Results

1. **Start with Alpha**: Most reliable, establish baseline
2. **Monitor TensorBoard**: Check for overfitting per band
3. **Compare with statistics**: Your `analyze_graphs.py` showed which bands differ most
4. **Match with neuroscience**: Do results align with DMT phenomenology?
5. **Report all bands**: Even "negative results" (low accuracy) are informative

---

## 📖 Next Steps

After training per-band models:

1. ✅ Compare results with your Pearson correlation analysis
2. ✅ Match with your spectral power analysis  
3. ✅ Validate against EEGNet results
4. ✅ Interpret: which brain rhythms encode DMT states?
5. ✅ Publish: "Alpha band synchronization as biomarker for DMT"

---

## 📧 Questions?

- Check individual band logs: `output_{band}/logs/train_*.log`
- Compare with mixed training: Does per-band improve results?
- View attention weights per band: `python analysis/visualize_attention.py --checkpoint checkpoints_alpha/best_model.pt`

---

**Last updated:** November 2025  
**Status:** ✅ Production Ready - Methodologically Correct Approach

