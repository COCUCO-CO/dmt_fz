# Phase Analysis Pipeline

Direct clustering of synchronization matrices + Kuramoto analysis.

## Purpose

Analyze brain connectivity patterns **directly from phase data** (without neural network models) to:
1. Find natural clusters in synchronization patterns
2. Compare clusters with Kuramoto order parameter
3. Determine if DMT/EC/EO conditions separate naturally
4. Establish a "ground truth" baseline for later model comparison

## Directory Structure

```
phase_analysis/
├── config/
│   └── config.yaml           # Pipeline configuration
├── scripts/
│   ├── 01_compute_sync_and_kuramoto.py   # Step 1: Process raw phases
│   ├── 02_cluster_sync_matrices.py       # Step 2: UMAP + Clustering
│   └── 03_analyze_results.py             # Step 3: Statistical analysis
├── output/
│   ├── sync_matrices/        # Computed sync matrices + Kuramoto
│   │   ├── sync_data.pkl
│   │   └── kuramoto_data.pkl
│   ├── clustering/           # Clustering results
│   │   ├── clustering_results.pkl
│   │   ├── embedding_2d.png
│   │   ├── cluster_composition.png
│   │   ├── kuramoto_analysis.png
│   │   └── cluster_metrics.json
│   └── visualizations/       # Detailed analysis plots
│       ├── condition_separation.png
│       ├── kuramoto_analysis_detailed.png
│       └── summary_stats.json
└── README.md
```

## Quick Start

```bash
cd /media/storage_hdd/dmt_fz/machine_learning/phase_analysis

# Step 1: Compute sync matrices and Kuramoto (takes ~5-10 min)
python scripts/01_compute_sync_and_kuramoto.py

# Step 2: Cluster and visualize (takes ~2-5 min)
python scripts/02_cluster_sync_matrices.py

# Step 3: Statistical analysis (takes ~1 min)
python scripts/03_analyze_results.py
```

## Configuration

Edit `config/config.yaml` to:
- Change data paths
- Select conditions/bands
- Adjust clustering parameters (UMAP, HDBSCAN)
- Use EEG (24 channels) or source space (102 parcels)

## Key Outputs

### 1. Sync Matrices (`output/sync_matrices/`)
- `sync_data.pkl`: Vectorized sync matrices ready for clustering
- `kuramoto_data.pkl`: Kuramoto order parameter per epoch

### 2. Clustering (`output/clustering/`)
- `embedding_2d.png`: 2D UMAP visualization colored by:
  - Cluster
  - Condition (DMT/EC/EO)
  - Kuramoto R
- `cluster_composition.png`: How clusters map to conditions/bands
- `cluster_metrics.json`: Silhouette score, etc.

### 3. Analysis (`output/visualizations/`)
- `condition_separation.png`: Detailed condition-cluster analysis
- `kuramoto_analysis_detailed.png`: Kuramoto correlation analysis
- `summary_stats.json`: Statistical test results including:
  - Adjusted Rand Index (cluster ↔ condition)
  - Normalized Mutual Information
  - Kuramoto ANOVA across clusters
  - Cohen's d effect size

## Interpretation

### Key Metrics

| Metric | Range | Interpretation |
|--------|-------|----------------|
| Adjusted Rand Index | [-0.5, 1] | 0 = random, 1 = perfect match clusters↔conditions |
| Normalized Mutual Info | [0, 1] | Higher = clusters predict conditions better |
| Silhouette Score | [-1, 1] | Higher = clusters are well-separated |
| Cohen's d | [0, ∞) | >0.8 = large effect size for Kuramoto difference |

### What to Look For

1. **Do conditions separate?** High ARI/NMI means clusters correspond to DMT/EC/EO
2. **Kuramoto gradient?** If clusters have different Kuramoto = sync level matters
3. **Band specificity?** Some bands may show clearer separation
4. **Temporal stability?** High stability = brain stays in same state

## Dependencies

```bash
pip install numpy scipy scikit-learn matplotlib seaborn pyyaml tqdm
pip install umap-learn hdbscan  # For UMAP and HDBSCAN clustering
pip install pandas  # For some visualizations
```

## Next Steps

After Phase 1 (direct analysis):
- **Phase 2**: Train VAE on same data
- **Phase 3**: Compare VAE clusters with these direct clusters
- **Phase 4**: Analyze VAE attention weights










