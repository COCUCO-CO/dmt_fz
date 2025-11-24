# 🚀 START HERE - Per-Band Training Implementation

## ✅ What Was Implemented

I've implemented the **methodologically correct per-band training strategy** to avoid confounding band-specific patterns with condition-specific patterns.

---

## 🎯 The Problem (That We Fixed)

**Before:** Mixed all bands in one dataset
- Graph 1: DMT, Alpha band
- Graph 2: DMT, Delta band  
- Graph 3: EC, Alpha band

**Problem:** Model doesn't know which band! Alpha-DMT might look more like Alpha-EC than Delta-DMT.

**Now:** Separate models per band ✅
- Model_Alpha: Only Alpha graphs (DMT vs EC vs EO)
- Model_Theta: Only Theta graphs (DMT vs EC vs EO)
- ... etc

---

## 📁 New Files Created

### Main Scripts
1. **`train_per_band.py`** - Train all bands separately
2. **`ensemble_bands.py`** - Combine predictions from multiple bands
3. **`quick_train_alpha.sh`** - Quick test with Alpha only (30-60 min)

### Documentation
4. **`PER_BAND_GUIDE.md`** - Complete guide with scientific rationale
5. **`TRAINING_OPTIONS.md`** - Summary of all training options
6. **`START_HERE.md`** - This file!

### Updated
7. **`README.md`** - Updated with per-band recommendations

---

## 🏃 How to Start NOW

### Option A: Quick Test (Recommended First)

```bash
cd /media/storage_hdd/dmt_fz/machine_learning/clf
./quick_train_alpha.sh
```

**What it does:**
- Trains GAT model on Alpha band (8-13 Hz) only
- Takes 30-60 minutes
- Expected accuracy: 75-85%

**Output:**
- `output_alpha/test_results.json`
- `checkpoints_alpha/best_model.pt`
- `runs_alpha/` (TensorBoard logs)

**Check results:**
```bash
cat output_alpha/test_results.json
tensorboard --logdir=runs_alpha
```

---

### Option B: Train All Bands (After Alpha Test)

```bash
cd /media/storage_hdd/dmt_fz/machine_learning/clf
python train_per_band.py
```

**What it does:**
- Trains 5 separate models (Delta, Theta, Alpha, Beta, Gamma)
- Takes 2-5 hours total
- Saves results per band

**Output structure:**
```
checkpoints_alpha/best_model.pt
checkpoints_theta/best_model.pt
checkpoints_beta/best_model.pt
checkpoints_delta/best_model.pt
checkpoints_gamma/best_model.pt

output_alpha/test_results.json
output_theta/test_results.json
output_beta/test_results.json
output_delta/test_results.json
output_gamma/test_results.json

band_comparison_results.json  ← Summary across all bands
```

**Check results:**
```bash
# View comparison
cat band_comparison_results.json

# View individual bands
cat output_alpha/test_results.json
cat output_theta/test_results.json

# TensorBoard for all bands
tensorboard --logdir_spec=Delta:runs_delta,Theta:runs_theta,Alpha:runs_alpha,Beta:runs_beta,Gamma:runs_gamma
```

---

### Option C: Create Ensemble (After Training All)

```bash
cd /media/storage_hdd/dmt_fz/machine_learning/clf
python ensemble_bands.py --method average
```

**What it does:**
- Combines predictions from all band-specific models
- Expected improvement: +2-5% over best single band

**Output:**
- `output/ensemble_results.json`
- `output/ensemble_confusion_matrix.png`

---

## 📊 What to Expect

### Individual Band Performance

| Band | Expected Accuracy | Significance |
|------|------------------|--------------|
| **Alpha** | **75-85%** ⭐ | Most reliable, strong psychedelic modulation |
| **Theta** | 70-80% | Memory, navigation, altered states |
| **Beta** | 70-80% | Attention, active processing |
| **Delta** | 65-75% | Deep states |
| **Gamma** | 60-70% | High variability |

### Ensemble
- **Expected:** 80-90%
- **Should be:** ≥ best single band

---

## 🔬 Scientific Value

This approach allows you to answer:

1. ✅ **Which band is most informative for DMT?**
   → Compare individual accuracies

2. ✅ **Does DMT affect all bands equally?**
   → Large accuracy differences = band-specific effects

3. ✅ **Can we predict DMT from a single band?**
   → If Alpha > 80%, yes!

4. ✅ **Do bands provide complementary info?**
   → If ensemble > best single band, yes!

---

## 📖 Documentation

**Quick reference:** `TRAINING_OPTIONS.md`

**Detailed guides:**
- `PER_BAND_GUIDE.md` - Why per-band is correct
- `README.md` - Full documentation
- `QUICKSTART.md` - 5-minute guide
- `IMPLEMENTATION_SUMMARY.md` - Technical details

---

## 🎓 Recommended Workflow

### Day 1: Validation
```bash
# Test with Alpha band (fastest)
./quick_train_alpha.sh

# Check if accuracy > 70%
cat output_alpha/test_results.json
```

### Day 2: Full Training
```bash
# Train all bands (can run overnight)
python train_per_band.py

# Check results
cat band_comparison_results.json
```

### Day 3: Analysis
```bash
# Create ensemble
python ensemble_bands.py

# Visualize attention
python analysis/visualize_attention.py --checkpoint checkpoints_alpha/best_model.pt

# Compare with your other results
# - Pearson correlations
# - EEGNet performance
# - Spectral analysis
```

---

## ⚡ Quick Commands

### Check if ready to start
```bash
# Check data
ls /media/storage_hdd/dmt_fz/fwd-inv-stc/DMT/phases-*.pkl | wc -l
# Should show ~29 files

# Check packages
python -c "import torch, torch_geometric; print('✓ Ready')"

# Check GPU (optional)
nvidia-smi
```

### Start training
```bash
cd /media/storage_hdd/dmt_fz/machine_learning/clf

# Quick test (30-60 min)
./quick_train_alpha.sh

# Or full training (2-5 hours)
python train_per_band.py
```

### Monitor training
```bash
# Watch progress
tail -f output_alpha/logs/train_*.log

# TensorBoard
tensorboard --logdir=runs_alpha
```

### Check results
```bash
# Individual band
cat output_alpha/test_results.json

# All bands comparison
cat band_comparison_results.json

# Ensemble
cat output/ensemble_results.json
```

---

## 🐛 Troubleshooting

### "ModuleNotFoundError"
```bash
pip install -r requirements.txt
```

### "CUDA out of memory"
Edit `config/config.yaml`:
```yaml
training:
  batch_size: 16  # Reduce from 32
```

### "Dataset cache exists"
```bash
python train_per_band.py --force-rebuild
```

---

## ✅ Current Status

- ✅ Dataset generated (`data/cache/dataset_*.pkl`)
- ✅ Statistical analysis completed (`output/analysis/`)
- ✅ Visualizations created (graphs, boxplots, etc.)
- ✅ Per-band training implemented
- ✅ Ensemble method implemented
- ⏳ **Next:** Train models!

---

## 🎯 Your Next Command

**Just run this:**

```bash
cd /media/storage_hdd/dmt_fz/machine_learning/clf
./quick_train_alpha.sh
```

Then wait 30-60 minutes and check results!

---

**Questions?** Read `PER_BAND_GUIDE.md` for scientific rationale or `TRAINING_OPTIONS.md` for all options.

**Ready to publish?** Use per-band training (not mixed), report individual band accuracies.

---

## 📞 Support

If something doesn't work:
1. Check `output_alpha/logs/train_*.log` for errors
2. Verify data exists: `ls /media/storage_hdd/dmt_fz/fwd-inv-stc/DMT/phases-*.pkl`
3. Check GPU: `nvidia-smi`
4. Review config: `cat config/config.yaml`

---

**Let's go! Start with `./quick_train_alpha.sh` 🚀**

