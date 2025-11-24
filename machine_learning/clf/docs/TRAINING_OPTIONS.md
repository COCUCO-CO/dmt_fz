# Training Options Summary

## 🎯 What to Run?

You have **3 main options** for training GAT models:

---

## ✅ **Option 1: Quick Test with Alpha Band (RECOMMENDED START)**

**Time:** 30-60 minutes  
**Purpose:** Validate pipeline works correctly  
**Methodologically:** ✅ Correct

```bash
cd /media/storage_hdd/dmt_fz/machine_learning/clf
./quick_train_alpha.sh
```

**What it does:**
- Trains GAT model on Alpha band (8-13 Hz) only
- Alpha is the most studied band for psychedelics
- Fastest way to see if everything works

**Output:**
- `output_alpha/test_results.json` - Metrics
- `checkpoints_alpha/best_model.pt` - Trained model
- `runs_alpha/` - TensorBoard logs

**Next step:** If accuracy is good (>70%), proceed to Option 2

---

## ✅ **Option 2: Train All Bands Separately (RECOMMENDED FINAL)**

**Time:** 2-5 hours total  
**Purpose:** Full analysis, publication-ready  
**Methodologically:** ✅ Correct

```bash
cd /media/storage_hdd/dmt_fz/machine_learning/clf
python train_per_band.py
```

**What it does:**
- Trains 5 separate models (Delta, Theta, Alpha, Beta, Gamma)
- Each model only sees graphs from one band
- Avoids confounding band-specific vs condition-specific patterns

**Output:**
```
checkpoints_delta/best_model.pt
checkpoints_theta/best_model.pt
checkpoints_alpha/best_model.pt
checkpoints_beta/best_model.pt
checkpoints_gamma/best_model.pt

output_delta/test_results.json
output_theta/test_results.json
output_alpha/test_results.json
output_beta/test_results.json
output_gamma/test_results.json

band_comparison_results.json  ← Summary
```

**Scientific questions answered:**
1. Which band is most informative for DMT classification?
2. Does DMT affect all bands equally?
3. Can we predict DMT state from a single band?

**Next step:** Create ensemble (Option 3)

---

## ✅ **Option 3: Ensemble All Bands**

**Time:** 5-10 minutes  
**Purpose:** Combine predictions for best accuracy  
**Prerequisites:** Must run Option 2 first

```bash
cd /media/storage_hdd/dmt_fz/machine_learning/clf
python ensemble_bands.py --method average
```

**Ensemble methods:**
- `average`: Average probabilities (default)
- `weighted`: Weight by validation accuracy
- `voting`: Majority vote
- `max`: Maximum confidence

**Expected improvement:** +2-5% over best single band

**Output:**
- `output/ensemble_results.json`
- `output/ensemble_confusion_matrix.png`

---

## ⚠️ **Option 4: Mixed Multi-Band (NOT RECOMMENDED)**

**Time:** 30-60 minutes  
**Purpose:** Exploratory only  
**Methodologically:** ❌ Problematic (confounds band with condition)

```bash
cd /media/storage_hdd/dmt_fz/machine_learning/clf
./run_pipeline.sh
```

**Problem:**
- Mixes graphs from all bands in same dataset
- Model doesn't know which band each graph is from
- Alpha-DMT might look more like Alpha-EC than Delta-DMT

**When to use:**
- Quick exploratory analysis only
- DO NOT use for final results or publication

---

## 📊 Comparison Table

| Option | Time | Methodological | Interpretable | Publication-Ready | Computational |
|--------|------|----------------|---------------|-------------------|---------------|
| **1. Alpha Only** | 30-60m | ✅ | ✅ | ⚠️ Partial | Low |
| **2. Per-Band** | 2-5h | ✅ | ✅ | ✅ | Medium |
| **3. Ensemble** | +5-10m | ✅ | ✅ | ✅ | Low |
| **4. Mixed** | 30-60m | ❌ | ❌ | ❌ | Low |

---

## 🚀 Recommended Workflow

### Day 1: Quick Validation
```bash
# 1. Quick test with Alpha
./quick_train_alpha.sh

# 2. Check results
cat output_alpha/test_results.json

# Expected: 70-85% accuracy
# If < 60%: check data quality, hyperparameters
# If > 70%: proceed to full training
```

### Day 2: Full Training
```bash
# 3. Train all bands (can run overnight)
python train_per_band.py

# 4. Compare results
cat band_comparison_results.json

# 5. View TensorBoard
tensorboard --logdir_spec=Delta:runs_delta,Theta:runs_theta,Alpha:runs_alpha,Beta:runs_beta,Gamma:runs_gamma
```

### Day 3: Ensemble & Analysis
```bash
# 6. Create ensemble
python ensemble_bands.py --method average

# 7. Compare with individual bands
cat output/ensemble_results.json

# 8. Analyze attention weights per band
python analysis/visualize_attention.py --checkpoint checkpoints_alpha/best_model.pt
python analysis/visualize_attention.py --checkpoint checkpoints_theta/best_model.pt
# ... etc
```

---

## 💡 Decision Tree

```
Start Here
    ↓
Do you have 30 minutes?
    YES → Run ./quick_train_alpha.sh
    |     ↓
    |     Accuracy > 70%?
    |         YES → Continue with full training
    |         NO → Check data/hyperparameters
    |
    NO → Come back when you have time

Full Training Phase
    ↓
Do you have 2-5 hours?
    YES → Run python train_per_band.py
    |     ↓
    |     All bands trained?
    |         YES → Create ensemble
    |         NO → Debug failed bands
    |
    NO → Train bands one at a time

Ensemble Phase
    ↓
All individual models trained?
    YES → Run python ensemble_bands.py
    |     ↓
    |     Ensemble better than best single?
    |         YES → Use ensemble for final results
    |         NO → Use best single band
    |
    NO → Finish training all bands first
```

---

## 🔬 Expected Results

### Individual Band Accuracies (DMT vs EC vs EO)

| Band | Frequency | Expected Acc | Why? |
|------|-----------|--------------|------|
| **Alpha** | 8-13 Hz | **75-85%** | Dominant resting rhythm, strong psychedelic modulation |
| **Theta** | 4-8 Hz | 70-80% | Memory, navigation, altered states |
| **Beta** | 13-30 Hz | 70-80% | Attention, active processing |
| **Delta** | 1-4 Hz | 65-75% | Deep states, less specific |
| **Gamma** | 30-100 Hz | 60-70% | High variability, lower SNR |

### Ensemble
- **Expected:** 80-90%
- **Minimum:** Should match best single band
- **Improvement:** +2-5% over best single band

---

## 🐛 Troubleshooting

### "No module named torch_geometric"
```bash
pip install -r requirements.txt
```

### "CUDA out of memory"
Edit `config/config.yaml`:
```yaml
training:
  batch_size: 16  # Reduce from 32
```

### "Dataset cache exists for different bands"
```bash
python train_per_band.py --force-rebuild
```

### One band has very low accuracy (<50%)
- Check data quality for that band
- May need band-specific hyperparameters
- Gamma often has lower accuracy (normal)

### Ensemble worse than best single band
- Ensure test sets are identical (same random seed)
- Try using only top 3 bands:
  ```bash
  python ensemble_bands.py --bands Alpha Theta Beta
  ```

---

## 📖 Documentation

- **Quick Start:** `QUICKSTART.md`
- **Full Documentation:** `README.md`
- **Per-Band Strategy:** `PER_BAND_GUIDE.md` ← Read this for scientific rationale
- **Implementation Details:** `IMPLEMENTATION_SUMMARY.md`
- **Conv Types:** `CONV_TYPES_GUIDE.md`
- **Setup:** `SETUP.md`

---

## ✅ Checklist

Before starting:
- [ ] Check data exists: `ls /media/storage_hdd/dmt_fz/fwd-inv-stc/DMT/phases-*.pkl`
- [ ] Verify packages: `python -c "import torch, torch_geometric; print('OK')"`
- [ ] Have GPU available (optional but faster): `nvidia-smi`
- [ ] Have 2-5 hours available (for full training)

After training:
- [ ] Check all bands completed successfully
- [ ] View results: `cat band_comparison_results.json`
- [ ] Compare with your Pearson correlation analysis
- [ ] Identify which bands are most informative
- [ ] Optional: Create ensemble

For publication:
- [ ] Use per-band training (not mixed)
- [ ] Report individual band accuracies
- [ ] Discuss neuroscience interpretation
- [ ] Compare with other methods (EEGNet, Random Forest)

---

**Last Updated:** November 2025  
**Status:** ✅ Ready to Use

