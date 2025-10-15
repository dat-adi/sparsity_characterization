import marimo

__generated_with = "0.16.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import torch
    import sklearn
    import matplotlib.pyplot as plt
    import seaborn as sns
    import random
    return plt, random, sklearn, torch


@app.cell
def _(torch):
    down_proj = torch.load("../../data/clustering/wanda/layer1-mlp.down_proj.pt")
    up_proj = torch.load("../../data/clustering/wanda/layer1-mlp.up_proj.pt")
    gate_proj = torch.load("../../data/clustering/wanda/layer1-mlp.gate_proj.pt")
    return (down_proj,)


@app.cell
def _(down_proj, random, torch):
    random.seed(42)
    def create_feature_subset(main_feature_idx: int, n_comparitive_features: int, matrix):
        """Create a subset from a weights matrix with randomly selected features"""
        n_features = matrix.shape[1]
        feature_range = list(range(0, n_features))
        random_feature_range = feature_range[:main_feature_idx] + feature_range[main_feature_idx+1:]
        random_feature_indices = random.sample(random_feature_range, min(n_comparitive_features, len(random_feature_range)))

        # In our setup, we always have the feature that we're comparing every other feature with
        # at the start of the subset.
        return torch.cat([
            matrix[:, main_feature_idx].unsqueeze(1),
            matrix[:, random_feature_indices]
        ], dim=1)

    feature_subset = create_feature_subset(0, 127, down_proj)
    feature_subset = (feature_subset.abs() > 0).int()
    feature_subset.shape
    return create_feature_subset, feature_subset


@app.cell
def _(feature_subset, plt, torch):
    def get_coactivation_gradient(matrix):
        """Arranges the features for the main feature (located at idx 0) from most relevant to least relevant

        Utilizes the hamming distance to make this measurement. 
        """
        hamming_distances = torch.tensor([(matrix[:, 0] != matrix[:, i]).sum() for i in range(1, matrix.shape[1])])
        print(hamming_distances)
        sorted_indices = torch.argsort(hamming_distances, descending=False)

        # adding in the 0th index to pad the coactivation gradient with the main feature
        sorted_indices = torch.cat([torch.tensor([0]), sorted_indices])
        return matrix[:, sorted_indices], sorted_indices

    coactivation_gradient, sorted_indices = get_coactivation_gradient(feature_subset)
    plt.spy(coactivation_gradient.cpu()), coactivation_gradient.shape
    return coactivation_gradient, get_coactivation_gradient, sorted_indices


@app.cell
def _(coactivation_gradient):
    data = coactivation_gradient.cpu()
    data.shape
    return (data,)


@app.cell
def _(data, plt, sklearn, sorted_indices):
    import numpy as np
    from sklearn.manifold import TSNE

    def visualize_clusters(kmeans, data, sorted_indices):
        """
        Visualize k-means clustering results for feature co-activation patterns

        Args:
            kmeans: Fitted KMeans object
            data: Coactivation gradient matrix (samples x features), shape (n_samples, 128)
            sorted_indices: Indices showing the original feature order
        """
        labels = kmeans.labels_

        # Convert to numpy if needed
        if hasattr(data, 'numpy'):
            data_np = data.cpu().numpy()
        else:
            data_np = data

        # Sort features by cluster assignment for visualization
        cluster_sort_idx = np.argsort(labels)
        sorted_labels = labels[cluster_sort_idx]

        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))

        # 1. Reordered Heatmap (Primary Visualization)
        ax1 = plt.subplot(2, 2, 1)
        # data is (n_samples, n_features), we want (n_features, n_samples) for heatmap
        reordered_data = data_np[:, cluster_sort_idx].T
        im = ax1.imshow(
            reordered_data,
            aspect='auto',
            cmap='binary',
            interpolation='nearest'
        )

        # Add cluster boundaries
        cluster_boundaries = np.where(np.diff(sorted_labels))[0] + 1
        for boundary in cluster_boundaries:
            ax1.axhline(y=boundary, color='red', linewidth=2, alpha=0.7)

        ax1.set_xlabel('Samples', fontsize=12)
        ax1.set_ylabel('Features (sorted by cluster)', fontsize=12)
        ax1.set_title(f'Activation Patterns (k={kmeans.n_clusters})', fontsize=14)
        plt.colorbar(im, ax=ax1, label='Activation')

        # 2. t-SNE 2D Projection
        ax2 = plt.subplot(2, 2, 2)
        # kmeans was fit on data directly, so data is already (n_samples, n_features)
        # But for t-SNE we want each FEATURE as a point, so we need (n_features, n_samples)
        # However, kmeans.labels_ has length n_samples, which suggests you fit on samples
        # Let me check: you did kmeans.fit_predict(data) where data is (n_samples, 128)
        # So labels correspond to samples, not features

        # Actually, for your use case, you want to cluster FEATURES not SAMPLES
        # So we need to transpose before fitting kmeans
        # But since you already fit it, let's work with what we have

        # If labels has length matching n_samples, then you clustered samples
        # For feature clustering, we need to refit
        print(f"Data shape: {data_np.shape}")
        print(f"Labels shape: {labels.shape}")

        # Assuming you want to cluster features (columns), transpose data
        features_as_points = data_np.T  # Shape: (128, 4096)

        # Refit kmeans on features (or use existing if it was already correct)
        if len(labels) != features_as_points.shape[0]:
            print("Refitting k-means on features (transposed data)...")
            kmeans_corrected = sklearn.cluster.KMeans(n_clusters=kmeans.n_clusters, random_state=42)
            labels = kmeans_corrected.fit_predict(features_as_points)
            cluster_sort_idx = np.argsort(labels)
            sorted_labels = labels[cluster_sort_idx]
            kmeans = kmeans_corrected

        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features_as_points)-1))
        embedding = tsne.fit_transform(features_as_points)

        scatter = ax2.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=labels,
            cmap='tab10',
            s=100,
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5
        )
        ax2.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax2.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax2.set_title(f'Feature Clusters (k={kmeans.n_clusters})', fontsize=14)
        plt.colorbar(scatter, ax=ax2, label='Cluster ID')

        # 3. Cluster Size Distribution
        ax3 = plt.subplot(2, 2, 3)
        unique, counts = np.unique(labels, return_counts=True)
        bars = ax3.bar(unique, counts, color='steelblue', edgecolor='black', alpha=0.7)
        ax3.set_xlabel('Cluster ID', fontsize=12)
        ax3.set_ylabel('Number of Features', fontsize=12)
        ax3.set_title('Cluster Size Distribution', fontsize=14)
        ax3.set_xticks(unique)

        # Add count labels on bars
        for i, (cluster_id, count) in enumerate(zip(unique, counts)):
            ax3.text(cluster_id, count + 0.5, str(count), ha='center', fontsize=10)

        # 4. Statistics and Block Coherence
        ax4 = plt.subplot(2, 2, 4)

        # Calculate block coherence for MMM analysis
        block_size = 8
        sorted_data = data_np[:, cluster_sort_idx]  # Shape: (4096, 128)
        n_blocks = sorted_data.shape[1] // block_size
        coherence_scores = []

        for block_idx in range(n_blocks):
            start = block_idx * block_size
            end = start + block_size
            block_data = sorted_data[:, start:end]  # Shape: (4096, 8)

            # Count samples where all 8 features are zero (exploitable by MMM)
            all_zero = (block_data.sum(axis=1) == 0).sum()
            coherence = all_zero / block_data.shape[0]
            coherence_scores.append(coherence)

        mean_coherence = np.mean(coherence_scores) if coherence_scores else 0

        # Calculate within-cluster variances
        cluster_variances = []
        for cluster_id in range(kmeans.n_clusters):
            cluster_mask = (labels == cluster_id)
            cluster_points = features_as_points[cluster_mask]
            cluster_center = kmeans.cluster_centers_[cluster_id]
            variance = np.mean(np.sum((cluster_points - cluster_center)**2, axis=1))
            cluster_variances.append(variance)

        # Display statistics
        ax4.axis('off')
        stats_text = f"""
        Clustering Statistics:
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━
        k = {kmeans.n_clusters}
        Total Inertia = {kmeans.inertia_:.2f}
        Iterations = {kmeans.n_iter_}

        Cluster Analysis:
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━
        Mean cluster size = {np.mean(counts):.1f}
        Std cluster size = {np.std(counts):.1f}

        Mean within-cluster variance = {np.mean(cluster_variances):.2f}

        MMM Block Analysis:
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━
        Mean 8-vector coherence = {mean_coherence:.3f}
        Blocks analyzed = {n_blocks}

        Interpretation:
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━
        {'✓ Good clustering' if kmeans.inertia_ / len(features_as_points) < 100 else '✗ Weak clustering'}
        {'✓ Block structure exists' if mean_coherence > 0.3 else '✗ Limited block structure'}
        """

        ax4.text(0.05, 0.95, stats_text, fontsize=11, family='monospace',
                 verticalalignment='top', transform=ax4.transAxes)

        plt.tight_layout()
        plt.show()

        # Additional detailed block coherence plot
        if n_blocks > 0:
            fig2, ax5 = plt.subplots(figsize=(14, 5))
            bars = ax5.bar(range(n_blocks), coherence_scores, color='seagreen', 
                           edgecolor='black', alpha=0.7)
            ax5.axhline(y=0.5, color='red', linestyle='--', linewidth=2, 
                        label='50% threshold', alpha=0.7)
            ax5.axhline(y=mean_coherence, color='blue', linestyle=':', linewidth=2,
                        label=f'Mean: {mean_coherence:.3f}', alpha=0.7)
            ax5.set_xlabel('Block Index (8-vector groups)', fontsize=12)
            ax5.set_ylabel('Coherence Score (fraction all-zero)', fontsize=12)
            ax5.set_title('Block Coherence Analysis for MMM Optimization', fontsize=14)
            ax5.legend(fontsize=11)
            ax5.set_ylim([0, 1])
            ax5.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.show()

        # Print summary
        print(f"\n{'='*50}")
        print(f"CLUSTERING SUMMARY")
        print(f"{'='*50}")
        print(f"Total inertia: {kmeans.inertia_:.2f}")
        print(f"Cluster sizes: {counts}")
        print(f"Mean block coherence: {mean_coherence:.3f}")
        print(f"\nMMM Applicability: {'HIGH' if mean_coherence > 0.4 else 'MEDIUM' if mean_coherence > 0.2 else 'LOW'}")
        print(f"{'='*50}\n")

        return {
            'inertia': kmeans.inertia_,
            'cluster_sizes': counts,
            'cluster_variances': cluster_variances,
            'mean_coherence': mean_coherence,
            'coherence_scores': coherence_scores,
            'corrected_labels': labels
        }


    # Transpose data so features are data points
    features_as_points = data.T.cpu().numpy()  # Shape: (128, 4096) where the features are now represented as rows
    kmeans_correct = sklearn.cluster.KMeans(n_clusters=8, random_state=42)
    labels_correct = kmeans_correct.fit_predict(features_as_points)

    results = visualize_clusters(kmeans_correct, data, sorted_indices)
    print(results)
    return (visualize_clusters,)


@app.cell
def _(
    create_feature_subset,
    down_proj,
    get_coactivation_gradient,
    random,
    sklearn,
    visualize_clusters,
):
    def run_analysis(matrix, seed):
        """Over 10 random seeds, pick a random column, pick a random subset, investigate"""
        random.seed(seed)
        main_ft_col_idx = random.randint(0, matrix.shape[1])
        feature_subset = create_feature_subset(main_ft_col_idx, 127, matrix)
        feature_subset = (feature_subset.abs() > 0).int()
        coactivation_gradient, sorted_indices = get_coactivation_gradient(feature_subset)
        data = coactivation_gradient.cpu()
        features_as_points = data.T.cpu().numpy()  # Shape: (128, 4096) where the features are now represented as rows
        kmeans_correct = sklearn.cluster.KMeans(n_clusters=8, random_state=seed)
        labels_correct = kmeans_correct.fit_predict(features_as_points)

        results = visualize_clusters(kmeans_correct, data, sorted_indices)

    for seed in range(1):
        run_analysis(down_proj, seed)
    return (run_analysis,)


@app.cell
def _(torch):
    sparsegpt_down_proj = torch.load("../../data/clustering/sparsegpt/layer1-mlp.down_proj.pt")
    sparsegpt_up_proj = torch.load("../../data/clustering/sparsegpt/layer1-mlp.up_proj.pt")
    sparsegpt_gate_proj = torch.load("../../data/clustering/sparsegpt/layer1-mlp.gate_proj.pt")
    return (sparsegpt_up_proj,)


@app.cell
def _(run_analysis, sparsegpt_up_proj):
    for s in range(1):
        run_analysis(sparsegpt_up_proj, s)
    return


if __name__ == "__main__":
    app.run()
