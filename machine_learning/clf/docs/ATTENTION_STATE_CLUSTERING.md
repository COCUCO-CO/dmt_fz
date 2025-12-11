# Attention State Clustering Analysis

This guide explains how to extract GAT attention matrices and cluster them to find recurring "brain states" in the EEG data.

## Concept

The Graph Attention Network (GAT) learns attention weights that represent the importance of connections between brain regions. These attention patterns may reveal:

1. **Recurring states**: Similar connectivity patterns that appear at different times
2. **Condition-specific states**: Patterns that are more common during DMT vs baseline
3. **Temporal dynamics**: How the brain transitions between states over time

This is conceptually similar to "EEG microstates" but using the learned representations from the classifier.

## Workflow

### Step 1: Train the model (if not done)

```bash
cd /media/storage_hdd/dmt_fz/machine_learning/clf
python train.py --config config/config_dmt_ec_stc.yaml
```

### Step 2: Extract attention matrices

The extraction script runs each epoch from the test set through the model (in temporal order) and saves the attention matrices.

```bash
# List available test subjects
python analysis/extract_attention_states.py --config config/config_dmt_ec_stc.yaml --list-subjects

# Extract all test subjects
python analysis/extract_attention_states.py --config config/config_dmt_ec_stc.yaml

# Extract specific subject only
python analysis/extract_attention_states.py --config config/config_dmt_ec_stc.yaml --subject S01

# Extract with more sample images
python analysis/extract_attention_states.py --config config/config_dmt_ec_stc.yaml --num-images 100

# Save per-head attention (not just averaged)
python analysis/extract_attention_states.py --config config/config_dmt_ec_stc.yaml --save-per-head
```

**Output structure:**
```
output_stc/attention_states/
├── S01/
│   ├── DMT/
│   │   ├── tensors/
│   │   │   ├── attention_layer0_mean.npy    # [n_epochs, 102, 102]
│   │   │   └── attention_layer1_mean.npy
│   │   ├── images/
│   │   │   ├── epoch0000_layer0.png         # First 50 epochs only
│   │   │   └── ...
│   │   └── metadata.pkl
│   └── EC/
│       └── ...
├── S02/
│   └── ...
└── extraction_summary.pkl
```

### Step 3: Cluster the attention states

```bash
# Auto-optimize number of clusters
python analysis/attention_clustering.py --input-dir output_stc/attention_states

# Use specific number of clusters
python analysis/attention_clustering.py --input-dir output_stc/attention_states --n-clusters 5

# Use layer 1 instead of layer 0
python analysis/attention_clustering.py --input-dir output_stc/attention_states --layer layer1

# Use concatenated features from all layers
python analysis/attention_clustering.py --input-dir output_stc/attention_states --layer all

# Specific subject only
python analysis/attention_clustering.py --input-dir output_stc/attention_states --subject S01

# Different clustering method
python analysis/attention_clustering.py --input-dir output_stc/attention_states --method gmm
```

**Output:**
```
output_stc/attention_states/clustering/
├── tsne_clusters.png           # t-SNE visualization of states
├── state_distribution.png      # State frequency per condition
├── transition_matrices.png     # State transition probabilities
├── pca_variance.png            # PCA explained variance
├── elbow_curves.png            # Cluster optimization metrics
└── clustering_results.pkl      # Full results for further analysis
```

## Interpreting Results

### t-SNE Visualization
- Each point is one epoch
- Colors show cluster assignment or condition
- Clusters that separate by condition suggest condition-specific states

### State Distribution
- Shows how often each state occurs in DMT vs EC
- States that are much more common in one condition are interesting

### Transition Matrices
- Show probability of transitioning from one state to another
- Different transition patterns between DMT/EC suggest different dynamics
- Diagonal-dominant = stable states; off-diagonal = rapid switching

### Key Questions to Answer

1. **Are there DMT-specific states?** Look for clusters that are predominantly DMT epochs
2. **Are there baseline-specific states?** Same for EC
3. **What's the transition pattern?** Does DMT show more rapid state switching?
4. **What do the centroids look like?** Which brain connections are characteristic of each state?

## Advanced Analysis

### Load results for custom analysis

```python
import pickle
import numpy as np

# Load clustering results
with open('output_stc/attention_states/clustering/clustering_results.pkl', 'rb') as f:
    results = pickle.load(f)

labels = results['labels']
metadata = results['metadata']
temporal = results['temporal_analysis']

# Get DMT vs EC indices
dmt_idx = [i for i, m in enumerate(metadata) if m['condition'] == 'DMT']
ec_idx = [i for i, m in enumerate(metadata) if m['condition'] == 'EC']

# Compare state distributions
dmt_states = labels[dmt_idx]
ec_states = labels[ec_idx]

# Statistical test
from scipy.stats import chi2_contingency
contingency = np.zeros((2, results['n_clusters']))
for i in range(results['n_clusters']):
    contingency[0, i] = (dmt_states == i).sum()
    contingency[1, i] = (ec_states == i).sum()

chi2, p, dof, expected = chi2_contingency(contingency)
print(f"Chi-square test: χ²={chi2:.2f}, p={p:.4f}")
```

### Load attention matrices for specific analysis

```python
import numpy as np

# Load attention matrices for one subject/condition
att = np.load('output_stc/attention_states/S01/DMT/tensors/attention_layer0_mean.npy')
print(f"Shape: {att.shape}")  # [n_epochs, 102, 102]

# Get mean attention pattern
mean_att = att.mean(axis=0)

# Find most connected regions
node_importance = mean_att.sum(axis=0) + mean_att.sum(axis=1)
top_nodes = np.argsort(node_importance)[::-1][:10]
print(f"Top connected nodes: {top_nodes}")
```

## Model Details

- **Architecture**: GATv2 with 2 layers, 8 attention heads
- **Nodes**: 102 Schaefer parcels (source space)
- **Edges**: Fully connected (all pairs)
- **Attention matrix**: 102 × 102, averaged over 8 heads

## Troubleshooting

### "No attention data found"
- Check that the input directory contains the extracted states
- Verify the subject name matches exactly

### "CUDA out of memory"
- Use `--subject` to process one subject at a time
- Reduce batch size in extraction script

### Poor clustering (low silhouette)
- Try different `--n-components` values (5, 15, 20)
- Try `--method gmm` or `--method hierarchical`
- Consider using `--layer all` to combine layers







