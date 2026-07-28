import numpy as np
from sklearn.neighbors import NearestNeighbors

def get_graph_summary_features(graph):
    """
    "The Kitchen Sink": Combines Spatial Precision + Histogram Robustness.
    """
    
    # --- PART 1: Spatial Statistics (Precision) ---
    # indices: 0(X), 1(Y), 4(Sin), 5(Cos), 6(DistToCore)
    spatial_indices = [0, 1, 4, 5, 6]
    
    if graph.num_nodes > 0:
        spatial_mean = graph.x[:, spatial_indices].mean(axis=0).numpy()
        spatial_std = graph.x[:, spatial_indices].std(axis=0).numpy()
    else:
        spatial_mean = np.zeros(5)
        spatial_std = np.zeros(5)

    # --- PART 2: Robust Histograms (Distributions) ---
    # We L1 normalize these so density (dropout) doesn't affect them.
    
    # 2a. Edge Lengths
    if graph.num_edges > 0:
        edge_distances = graph.edge_attr[:, 0].numpy()
        bins = [0, 20, 40, 60, 80, 100, 150, 300]
        dist_hist, _ = np.histogram(edge_distances, bins=bins)
        # NORMALIZE to probability (Sum=1)
        if dist_hist.sum() > 0: dist_hist = dist_hist / dist_hist.sum()
        
        dist_median = np.median(edge_distances)
    else:
        dist_hist = np.zeros(7)
        dist_median = 0

    # 2b. Relative Angles
    if graph.num_edges > 0:
        edge_angles = graph.edge_attr[:, 1].numpy()
        angle_bins = np.linspace(0, 3.14, 8)
        angle_hist, _ = np.histogram(edge_angles, bins=angle_bins)
        # NORMALIZE
        if angle_hist.sum() > 0: angle_hist = angle_hist / angle_hist.sum()
    else:
        angle_hist = np.zeros(7)

    # 2c. Minutiae Type Ratio
    if graph.num_nodes > 0:
        types = graph.x[:, -1].numpy()
        bif_ratio = np.sum(types == 1) / graph.num_nodes
    else:
        bif_ratio = 0.0

    # 2d. Local Topology (Density)
    local_feats = []
    if graph.num_nodes > 4:
        coords = graph.x[:, :2].numpy()
        nbrs = NearestNeighbors(n_neighbors=5, algorithm='brute').fit(coords)
        distances, indices = nbrs.kneighbors(coords)
        neighbor_dists = distances[:, 1:].flatten()
        local_hist, _ = np.histogram(neighbor_dists, bins=[0, 15, 30, 45, 100])
        # NORMALIZE
        if local_hist.sum() > 0: local_hist = local_hist / local_hist.sum()
        local_feats.extend(local_hist)
    else:
        local_feats.extend(np.zeros(4))

    return np.concatenate([
        spatial_mean,       # 5 features
        spatial_std,        # 5 features
        dist_hist,          # 7 features
        [dist_median],      # 1 feature
        angle_hist,         # 7 features
        [bif_ratio],        # 1 feature
        local_feats         # 4 features
    ])

def create_feature_vector_for_pair(graph1, graph2):
    f1 = get_graph_summary_features(graph1)
    f2 = get_graph_summary_features(graph2)
    
    # 1. Standard Differences
    diff = np.abs(f1 - f2)
    
    # 2. Chi-Squared for Histograms (approximation for whole vector)
    chi_sq = (f1 - f2)**2 / (f1 + f2 + 1e-6)
    
    # 3. Product (Correlation)
    prod = f1 * f2
    
    return np.concatenate([diff, chi_sq, prod])