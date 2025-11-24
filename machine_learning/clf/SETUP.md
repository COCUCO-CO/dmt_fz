# Setup Guide

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- Conda or virtualenv

## Installation Steps

### 1. Create Virtual Environment

```bash
conda create -n gat_clf python=3.10
conda activate gat_clf
```

Or with virtualenv:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### 2. Install PyTorch

**For CUDA 11.8:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CPU only:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 3. Install PyTorch Geometric

```bash
pip install torch-geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

**Note:** Replace `cu118` with your CUDA version (e.g., `cu121` or `cpu`)

### 4. Install Other Dependencies

```bash
pip install -r requirements.txt
```

### 5. Verify Installation

```bash
python -c "import torch; import torch_geometric; print('PyTorch:', torch.__version__); print('PyG:', torch_geometric.__version__); print('CUDA available:', torch.cuda.is_available())"
```

Expected output:
```
PyTorch: 2.0.0+cu118
PyG: 2.3.0
CUDA available: True
```

## Configuration

1. Copy example config and edit paths:

```bash
cd /media/storage_hdd/dmt_fz/machine_learning/clf
cat config/config.yaml
```

2. Update these paths in `config/config.yaml`:

```yaml
paths:
  phases_dir: "/media/storage_hdd/dmt_fz/fwd-inv-stc"  # Where your phases-*.pkl are
  output_dir: "/media/storage_hdd/dmt_fz/machine_learning/clf/output"
```

3. Verify your data exists:

```bash
ls /media/storage_hdd/dmt_fz/fwd-inv-stc/DMT/phases-*.pkl | head -5
ls /media/storage_hdd/dmt_fz/fwd-inv-stc/EC/phases-*.pkl | head -5
ls /media/storage_hdd/dmt_fz/fwd-inv-stc/EO/phases-*.pkl | head -5
```

## Quick Test

Test on a small subset:

```bash
# Edit config to use fewer files (for testing)
python train.py --config config/config.yaml
```

## Troubleshooting

### Issue: PyTorch Geometric installation fails

Try installing from conda:
```bash
conda install pyg -c pyg
```

### Issue: CUDA version mismatch

Check your CUDA version:
```bash
nvcc --version
# or
nvidia-smi
```

Then install matching PyTorch version from: https://pytorch.org/get-started/locally/

### Issue: Out of memory

Reduce batch size in config:
```yaml
training:
  batch_size: 16  # or 8
```

### Issue: Import errors

Make sure you're in the right directory and environment:
```bash
cd /media/storage_hdd/dmt_fz/machine_learning/clf
conda activate gat_clf
python -c "import sys; print(sys.path)"
```

## Directory Structure After Setup

```
machine_learning/clf/
├── config/
│   └── config.yaml           ✓ Configured
├── data/
│   ├── cache/                ✓ Created (empty)
│   └── ...
├── checkpoints/              ✓ Created (empty)
├── output/                   ✓ Created (empty)
├── runs/                     ✓ Created (empty)
└── ...
```

## Next Steps

Once setup is complete:

1. Run full pipeline:
```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

2. Or run steps individually:
```bash
# Analyze graphs
python analysis/analyze_graphs.py

# Train model
python train.py

# Evaluate
python evaluate.py --checkpoint checkpoints/best_model.pt
```

3. Monitor training:
```bash
tensorboard --logdir=runs
```

## Hardware Recommendations

- **Minimum:** 8GB RAM, CPU only
- **Recommended:** 16GB RAM, NVIDIA GPU with 8GB+ VRAM
- **Optimal:** 32GB RAM, NVIDIA GPU with 16GB+ VRAM

Training time (29 subjects, all bands):
- CPU: ~4-6 hours
- GPU: ~30-60 minutes

