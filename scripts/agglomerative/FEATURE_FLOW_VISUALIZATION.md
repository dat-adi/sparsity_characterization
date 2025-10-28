# Feature Flow Visualization for Agglomerative Clustering

## Overview

The agglomerative clustering script now generates **two complementary visualizations** to help you understand the hierarchical structure:

1. **Dendrogram** (`*_dendrogram.png`) - Shows cluster sizes and statistics at each level
2. **Feature Flow Diagram** (`*_feature_flow.png`) - **NEW!** Shows how individual features contribute to the next level

## What the Feature Flow Visualization Shows

The feature flow visualization is a **Sankey-style diagram** that tracks:
- **Individual feature positions** at each hierarchical level
- **Which features become representatives** (selected to continue to next level)
- **Cluster membership** through color-coding
- **Feature lineage** from bottom to top of the hierarchy

### Visual Elements

#### Markers
- **Large circles (○)** = Selected representative features
- **Small dots (·)** = Cluster members (not selected)
- **Colors** = Cluster membership (each cluster has a unique color)

#### Lines
- **Thick dark lines** = Representative feature's path through hierarchy
- **Thin light lines** = Non-selected features connecting to their representative

### Reading the Diagram

1. **Bottom (Level 0)**: All initial features organized into groups
2. **Vertical axis**: Hierarchical levels (bottom to top)
3. **Horizontal axis**: Feature positions within each level
4. **Lines**: Show which features from level N feed into which representative at level N+1

## Example Interpretation

For a 4-level hierarchy (128 seeds × 8 = 1024 features):

```
Level 0: 1024 features (128 groups of 8)
         ↓ Select 1 rep per group
Level 1: 128 features (16 groups of 8)
         ↓ Select 1 rep per group
Level 2: 16 features (2 groups of 8)
         ↓ Select 1 rep per group
Level 3: 2 features (1 group of 2)
         ↓ Select 1 rep
Final:   1 feature (root cluster)
```

In the visualization, you can **trace any representative** from Level 0 all the way to the top by following the thick colored lines. This shows exactly which initial features contribute to the final hierarchical structure.

## Key Insights

### What You Can Learn

1. **Feature Stability**: Representatives that persist through multiple levels indicate stable/central features
2. **Cluster Coherence**: Tight groupings (minimal line crossing) suggest well-formed clusters
3. **Merging Patterns**: How clusters combine at higher levels
4. **Selection Bias**: Whether certain regions are over/under-represented in higher levels

### Comparison with Dendrogram

| Aspect | Dendrogram | Feature Flow |
|--------|-----------|--------------|
| Shows cluster sizes | ✓ | ✗ |
| Shows statistics | ✓ | ✗ |
| Tracks individual features | ✗ | ✓ |
| Shows feature lineage | ✗ | ✓ |
| Shows selection process | ✗ | ✓ |

**Use both together** for complete understanding:
- Dendrogram → quantitative cluster properties
- Feature flow → qualitative feature contributions

## Configuration

Both visualizations are generated automatically when running:

```bash
python scripts/agglomerative/agglomerative_clustering.py
```

### Parameters (in script)

```python
N_SEED_FEATURES = 1024  # Number of seed features
GROUP_SIZE = 8          # Features per group at each level
```

Visualization files are saved to:
```
results/visualizations/agglomerative_clustering/
  ├── {matrix_name}_dendrogram.png
  └── {matrix_name}_feature_flow.png  ← NEW!
```

## Technical Details

### Implementation

The feature flow visualization (`visualize_feature_flow()` at line 532) works by:

1. **Building a feature lineage map**: Tracks each feature through all levels
2. **Assigning x-positions**: Places features horizontally based on cluster membership
3. **Drawing connections**: Links members to their representatives
4. **Highlighting representatives**: Makes selected features prominent

### Performance

For large hierarchies (1024+ features), the visualization:
- Uses transparent lines to show density without obscuring structure
- Colors clusters consistently across levels
- Highlights only representatives with thick lines to reduce visual clutter

## Limitations

1. **Visual complexity**: Very large hierarchies (8000+ features) may be hard to read
2. **Static view**: Cannot interactively explore individual feature paths
3. **No feature labels**: Individual features are not labeled (would be too cluttered)

## Future Enhancements

Potential improvements:
- Interactive version (D3.js/Plotly) for zooming and filtering
- Heatmap overlay showing feature similarity
- Animation showing clustering process step-by-step
- Feature importance overlays (size/color by importance)

## Questions?

See the main agglomerative clustering script documentation in:
- `scripts/agglomerative/agglomerative_clustering.py` (docstrings)
- Project README at `README.md`
