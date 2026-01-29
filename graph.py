from models import CascadeFM, LinearModel, SimpleMLP, ResidualMLP
import numpy as np
from trainer import clone_model, gd_train_logistic, train_linear_head_sgd, train_model_sgd, CostTracker
from typing import Dict, List, Tuple, Union, Optional, Any
from collections import defaultdict, deque
from scipy.stats import entropy, wasserstein_distance
from scipy.spatial.distance import jensenshannon
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

def select_seed_node(
    X_train: Dict[Any, np.ndarray],
    y_train: Dict[Any, np.ndarray],
) -> Any:
    """
    Select seed node as medoid based on mean feature vectors.
    
    Returns node with minimum sum of distances to all other nodes.
    """
    nodes = list(X_train.keys())
    n = len(nodes)
    
    means = np.stack([X_train[v].mean(axis=0) for v in nodes], axis=0)  # [n, d]
    dists = np.linalg.norm(means[:, None, :] - means[None, :, :], axis=-1)  # [n, n]
    
    sum_dists = dists.sum(axis=1)
    medoid_idx = np.argmin(sum_dists)
    return nodes[medoid_idx]
    
def random_tree_edges(nodes: List[Any], rng: np.random.Generator) -> List[Tuple[Any, Any]]:
    """
    Generate random tree edges using random parent assignment.
    
    Returns list of undirected edges.
    """
    nodes = list(nodes)
    n = len(nodes)
    
    if n <= 1:
        return []
    
    shuffled = rng.permutation(nodes).tolist()
    edges = []
    
    for i in range(1, n):
        parent_idx = rng.integers(0, i)
        edges.append((shuffled[parent_idx], shuffled[i]))
    
    return edges

def compute_feature_distance(
    nodes: List[Any],
    X_dict: Dict[Any, np.ndarray],
    y_dict: Dict[int, np.ndarray],  # kept for API compatibility
) -> np.ndarray:
    """
    Compute a robust, normalized pairwise distance matrix between node features.

    Key properties:
    - Missing nodes → NaN distances
    - Same-shape matrices → flattened representation
    - Different-shape matrices → [mean | variance] representation
    - Distances are normalized by number of overlapping dimensions
    - Vectorized computation when possible
    """

    T = len(nodes)

    # ---------- Guard: no usable features ----------
    if X_dict is None or not hasattr(X_dict, "get"):
        return np.full((T, T), np.nan, dtype=float)

    # ---------- Collect feature matrices ----------
    mats = []
    for v in nodes:
        m = X_dict.get(v, None)
        if m is None:
            mats.append(None)
            continue
        arr = np.asarray(m, dtype=float)
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            arr = arr.reshape(1, -1)
        mats.append(arr)

    available = [m for m in mats if m is not None and m.size > 0]
    if len(available) == 0:
        return np.full((T, T), np.nan, dtype=float)

    shapes = {m.shape for m in available}
    same_shape = len(shapes) == 1

    # ---------- Build feature vectors ----------
    if same_shape:
        # Flatten matrices
        vec_len = available[0].size
        feats = np.full((T, vec_len), np.nan, dtype=float)
        for i, m in enumerate(mats):
            if m is None or m.size != vec_len:
                continue
            feats[i] = m.reshape(-1)

    else:
        # Mean + variance embedding (more informative than mean alone)
        max_dim = max(m.shape[1] for m in available)
        feats = np.full((T, 2 * max_dim), np.nan, dtype=float)

        for i, m in enumerate(mats):
            if m is None or m.size == 0:
                continue
            mean = np.nanmean(m, axis=0)
            var = np.nanvar(m, axis=0)

            d = min(len(mean), max_dim)
            feats[i, :d] = mean[:d]
            feats[i, max_dim : max_dim + d] = var[:d]

    # ---------- Pairwise normalized Euclidean distance ----------
    dist = np.full((T, T), np.nan, dtype=float)

    for i in range(T):
        xi = feats[i]
        if not np.any(np.isfinite(xi)):
            continue
        for j in range(i, T):
            xj = feats[j]
            mask = np.isfinite(xi) & np.isfinite(xj)
            if not np.any(mask):
                continue

            diff = xi[mask] - xj[mask]
            # normalization removes bias from varying overlap size
            d = np.sqrt(np.mean(diff * diff))

            dist[i, j] = d
            dist[j, i] = d

    return dist


def compute_target_distance(
    nodes: List[Any],
    y_dict: Dict[Any, np.ndarray],
) -> np.ndarray:
    """
    Pairwise Euclidean distance between flattened target vectors.
    """
    features = np.stack([y_dict[v].reshape(-1) for v in nodes])
    return np.linalg.norm(
        features[:, None, :] - features[None, :, :],
        axis=-1,
    )

def compute_gradient_distance(
    nodes: List[Any],
    X_dict: Dict[Any, np.ndarray],
    y_dict: Dict[Any, np.ndarray],
    normalize: bool = True,
) -> np.ndarray:
    """
    Feature-space proxy distance using gradients at theta=0.
    g_v = X^T y approximates (X^T X) (theta_v^*).
    """
    grads = []
    for v in nodes:
        X = X_dict[v]
        y = y_dict[v].reshape(-1)
        g = X.T @ y
        grads.append(g)

    grads = np.stack(grads)

    if normalize:
        norms = np.linalg.norm(grads, axis=1, keepdims=True) + 1e-8
        grads = grads / norms

    return np.linalg.norm(
        grads[:, None, :] - grads[None, :, :],
        axis=-1,
    )


def compute_model_distance(
    nodes: List[Any],
    X_dict: Dict[Any, np.ndarray],
    y_dict: Dict[Any, np.ndarray],
    regularization: float = 1e-2,
) -> np.ndarray:
    """
    Oracle distance: ||theta_i - theta_j||.
    Computationally expensive: O(T d^3).
    """
    thetas = []
    d = X_dict[nodes[0]].shape[1]

    for v in nodes:
        X = X_dict[v]
        y = y_dict[v].reshape(-1)
        A = X.T @ X + regularization * np.eye(d)
        b = X.T @ y
        thetas.append(np.linalg.solve(A, b))

    thetas = np.stack(thetas)

    return np.linalg.norm(
        thetas[:, None, :] - thetas[None, :, :],
        axis=-1,
    )

def compute_kl_distance(
    nodes: List[Any],
    y_dict: Dict[Any, np.ndarray],
    n_bins: int = 20,
) -> np.ndarray:
    all_y = np.concatenate([y_dict[v].reshape(-1) for v in nodes])
    bins = np.linspace(all_y.min(), all_y.max(), n_bins)

    hists = []
    for v in nodes:
        hist, _ = np.histogram(
            y_dict[v].reshape(-1),
            bins=bins,
            density=True,
        )
        hist = hist + 1e-8
        hist /= hist.sum()
        hists.append(hist)

    n = len(nodes)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = 0.5 * (
                entropy(hists[i], hists[j]) +
                entropy(hists[j], hists[i])
            )
            dist[i, j] = dist[j, i] = d

    return dist

def compute_wasserstein_distance(
    nodes: List[Any],
    y_dict: Dict[Any, np.ndarray],
) -> np.ndarray:

    n = len(nodes)
    dist = np.zeros((n, n))

    for i in range(n):
        yi = y_dict[nodes[i]].reshape(-1)
        for j in range(i + 1, n):
            yj = y_dict[nodes[j]].reshape(-1)
            d = wasserstein_distance(yi, yj)
            dist[i, j] = dist[j, i] = d

    return dist

def compute_mmd_feature_distance(
        nodes: List[Any],
        X_dict: Dict[Any, np.ndarray],
        y_dict: Dict[int, np.ndarray]
) -> np.ndarray:
    """
    Compute Maximum Mean Discrepancy (MMD) distance between task feature sets.
    Uses RBF kernel with median heuristic for bandwidth.
    Features are standardized per task before computing MMD.
    
    Returns:
        Symmetric distance matrix (T x T) with non-negative values.
        NaN if either task has no valid features.
    """
    T = len(nodes)
    if X_dict is None or not hasattr(X_dict, "get"):
        return np.full((T, T), np.nan, dtype=float)
    
    # Collect and standardize features per task
    feat_sets = []
    for v in nodes:
        X = X_dict.get(v, None)
        if X is None or not isinstance(X, np.ndarray) or X.size == 0:
            feat_sets.append(None)
            continue
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        # Standardize (zero mean, unit std per feature)
        X_mean = np.nanmean(X_arr, axis=0, keepdims=True)
        X_std = np.nanstd(X_arr, axis=0, keepdims=True) + 1e-8
        X_std = np.where(X_std == 0, 1.0, X_std)
        X_arr = (X_arr - X_mean) / X_std
        # Remove NaN rows
        valid_mask = np.all(np.isfinite(X_arr), axis=1)
        if valid_mask.sum() == 0:
            feat_sets.append(None)
        else:
            feat_sets.append(X_arr[valid_mask])
    
    dist = np.full((T, T), np.nan, dtype=float)
    
    for i in range(T):
        Xi = feat_sets[i]
        if Xi is None:
            continue
        for j in range(i, T):
            Xj = feat_sets[j]
            if Xj is None:
                continue
            
            # Compute median bandwidth heuristic
            # Concatenate samples and compute pairwise distances
            all_samples = np.vstack([Xi, Xj])
            if all_samples.shape[0] < 2:
                dist[i, j] = dist[j, i] = 0.0
                continue
            
            # Sample subset for median heuristic if data is large
            max_samples = 1000
            if all_samples.shape[0] > max_samples:
                idx = np.random.choice(all_samples.shape[0], max_samples, replace=False)
                sample_subset = all_samples[idx]
            else:
                sample_subset = all_samples
            
            pairwise_sq = np.sum((sample_subset[:, None, :] - sample_subset[None, :, :])**2, axis=2)
            median_sq = np.median(pairwise_sq[pairwise_sq > 0])
            sigma = np.sqrt(median_sq / 2.0) if median_sq > 0 else 1.0
            gamma = 1.0 / (2.0 * sigma**2)
            
            # Compute MMD using RBF kernel
            def rbf_kernel(X, Y, gamma):
                sq_dist = np.sum((X[:, None, :] - Y[None, :, :])**2, axis=2)
                return np.exp(-gamma * sq_dist)
            
            Kxx = rbf_kernel(Xi, Xi, gamma)
            Kyy = rbf_kernel(Xj, Xj, gamma)
            Kxy = rbf_kernel(Xi, Xj, gamma)
            
            ni, nj = Xi.shape[0], Xj.shape[0]
            mmd_sq = (Kxx.sum() - np.trace(Kxx)) / (ni * (ni - 1)) if ni > 1 else 0.0
            mmd_sq += (Kyy.sum() - np.trace(Kyy)) / (nj * (nj - 1)) if nj > 1 else 0.0
            mmd_sq -= 2.0 * Kxy.mean()
            
            mmd = float(np.sqrt(max(0.0, mmd_sq)))
            dist[i, j] = dist[j, i] = mmd
    
    return dist


def compute_mean_cov_feature_distance(
        nodes: List[Any],
        X_dict: Dict[Any, np.ndarray],
        y_dict: Dict[int, np.ndarray]
) -> np.ndarray:
    """
    Compute distributional feature distance using mean and covariance.
    Assumes Gaussian approximation: d(i,j) = ||μ_i - μ_j||₂ + ||Σ_i - Σ_j||_F
    
    Covariances are regularized with εI for numerical stability.
    
    Returns:
        Symmetric distance matrix (T x T).
        NaN if task has insufficient samples or invalid features.
    """
    T = len(nodes)
    if X_dict is None or not hasattr(X_dict, "get"):
        return np.full((T, T), np.nan, dtype=float)
    
    eps = 1e-6
    means = []
    covs = []
    
    for v in nodes:
        X = X_dict.get(v, None)
        if X is None or not isinstance(X, np.ndarray) or X.size == 0:
            means.append(None)
            covs.append(None)
            continue
        
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        
        # Remove NaN rows
        valid_mask = np.all(np.isfinite(X_arr), axis=1)
        X_clean = X_arr[valid_mask]
        
        if X_clean.shape[0] < 2:
            means.append(None)
            covs.append(None)
            continue
        
        mu = np.mean(X_clean, axis=0)
        cov = np.cov(X_clean, rowvar=False, ddof=1)
        if cov.ndim == 0:
            cov = cov.reshape(1, 1)
        # Regularize
        cov += eps * np.eye(cov.shape[0])
        
        means.append(mu)
        covs.append(cov)
    
    dist = np.full((T, T), np.nan, dtype=float)
    
    for i in range(T):
        if means[i] is None or covs[i] is None:
            continue
        for j in range(i, T):
            if means[j] is None or covs[j] is None:
                continue
            
            mu_i, mu_j = means[i], means[j]
            cov_i, cov_j = covs[i], covs[j]
            
            # Ensure compatible dimensions (pad if needed)
            d_i, d_j = len(mu_i), len(mu_j)
            if d_i != d_j:
                # Use smaller dimension
                d = min(d_i, d_j)
                mu_i = mu_i[:d]
                mu_j = mu_j[:d]
                cov_i = cov_i[:d, :d]
                cov_j = cov_j[:d, :d]
            
            mean_dist = float(np.linalg.norm(mu_i - mu_j))
            cov_dist = float(np.linalg.norm(cov_i - cov_j, ord='fro'))
            
            d_val = mean_dist + cov_dist
            dist[i, j] = dist[j, i] = d_val
    
    return dist


def compute_js_target_distance(
        nodes: List[Any],
        X_dict: Dict[Any, np.ndarray],
        y_dict: Dict[int, np.ndarray]
) -> np.ndarray:
    """
    Compute Jensen-Shannon distance between target distributions.
    Targets are converted to empirical distributions via histograms.
    Small smoothing is added to avoid zero probabilities.
    
    Returns:
        Symmetric distance matrix (T x T) in [0, sqrt(2)].
        NaN if either task has no valid targets.
    """
    
    T = len(nodes)
    if y_dict is None or not hasattr(y_dict, "get"):
        return np.full((T, T), np.nan, dtype=float)
    
    # Determine global bin edges from all targets
    all_targets = []
    for v in nodes:
        y = y_dict.get(v, None)
        if y is not None and isinstance(y, np.ndarray) and y.size > 0:
            y_arr = np.asarray(y, dtype=float).flatten()
            y_arr = y_arr[np.isfinite(y_arr)]
            all_targets.append(y_arr)
    
    if len(all_targets) == 0:
        return np.full((T, T), np.nan, dtype=float)
    
    all_concat = np.concatenate(all_targets)
    n_bins = min(50, max(10, int(np.sqrt(len(all_concat)))))
    bin_edges = np.histogram_bin_edges(all_concat, bins=n_bins)
    
    # Compute histograms with smoothing
    smoothing = 1e-8
    histograms = []
    for v in nodes:
        y = y_dict.get(v, None)
        if y is None or not isinstance(y, np.ndarray) or y.size == 0:
            histograms.append(None)
            continue
        y_arr = np.asarray(y, dtype=float).flatten()
        y_arr = y_arr[np.isfinite(y_arr)]
        if y_arr.size == 0:
            histograms.append(None)
            continue
        
        hist, _ = np.histogram(y_arr, bins=bin_edges)
        hist = hist.astype(float) + smoothing
        hist /= hist.sum()
        histograms.append(hist)
    
    dist = np.full((T, T), np.nan, dtype=float)
    
    for i in range(T):
        if histograms[i] is None:
            continue
        for j in range(i, T):
            if histograms[j] is None:
                continue
            
            # Jensen-Shannon distance (sqrt of divergence)
            js_dist = float(jensenshannon(histograms[i], histograms[j]))
            dist[i, j] = dist[j, i] = js_dist
    
    return dist


def compute_cka_distance(
        nodes: List[Any],
        X_dict: Dict[Any, np.ndarray],
        y_dict: Dict[int, np.ndarray]
) -> np.ndarray:
    """
    Compute Centered Kernel Alignment (CKA) distance between representations.
    Uses linear CKA (inner product kernel).
    Distance = 1 - CKA.
    
    If X_dict contains learned embeddings (representations), computes CKA.
    Otherwise returns NaN.
    
    Returns:
        Symmetric distance matrix (T x T) in [0, 2].
        NaN if either task has no valid representations.
    """
    T = len(nodes)
    if X_dict is None or not hasattr(X_dict, "get"):
        return np.full((T, T), np.nan, dtype=float)
    
    # Collect representations and center them
    reps = []
    for v in nodes:
        Z = X_dict.get(v, None)
        if Z is None or not isinstance(Z, np.ndarray) or Z.size == 0:
            reps.append(None)
            continue
        Z_arr = np.asarray(Z, dtype=float)
        if Z_arr.ndim == 1:
            Z_arr = Z_arr.reshape(-1, 1)
        
        # Remove NaN rows
        valid_mask = np.all(np.isfinite(Z_arr), axis=1)
        Z_clean = Z_arr[valid_mask]
        
        if Z_clean.shape[0] < 2:
            reps.append(None)
            continue
        
        # Center
        Z_centered = Z_clean - Z_clean.mean(axis=0, keepdims=True)
        reps.append(Z_centered)
    
    dist = np.full((T, T), np.nan, dtype=float)
    
    for i in range(T):
        if reps[i] is None:
            continue
        for j in range(i, T):
            if reps[j] is None:
                continue
            
            Zi = reps[i]
            Zj = reps[j]
            
            # Compute linear kernels (Gram matrices)
            Ki = Zi @ Zi.T
            Kj = Zj @ Zj.T
            
            # Center kernels (doubly centered)
            def center_kernel(K):
                n = K.shape[0]
                H = np.eye(n) - np.ones((n, n)) / n
                return H @ K @ H
            
            Ki_c = center_kernel(Ki)
            Kj_c = center_kernel(Kj)
            
            # CKA = HSIC(Ki, Kj) / sqrt(HSIC(Ki, Ki) * HSIC(Kj, Kj))
            hsic_ij = np.trace(Ki_c @ Kj_c)
            hsic_ii = np.trace(Ki_c @ Ki_c)
            hsic_jj = np.trace(Kj_c @ Kj_c)
            
            if hsic_ii <= 0 or hsic_jj <= 0:
                cka = 0.0
            else:
                cka = hsic_ij / np.sqrt(hsic_ii * hsic_jj)
            
            # Distance = 1 - CKA (bounded in [0, 2] if CKA in [-1, 1])
            d_val = float(1.0 - cka)
            dist[i, j] = dist[j, i] = d_val
    
    return dist

def mst_from_distance_matrix(nodes: List[Any], dist_matrix: np.ndarray) -> List[Tuple[Any, Any]]:
    """
    Compute minimum spanning tree from a distance matrix using Prim's algorithm.
    
    Args:
        nodes: List of node identifiers
        dist_matrix: n x n symmetric distance matrix
    
    Returns:
        List of undirected edges (u, v) forming the MST
    """
    n = len(nodes)
    if n == 0:
        return []
    
    visited = [False] * n
    visited[0] = True
    edges = []
    
    # Priority queue: (distance, from_idx, to_idx)
    import heapq
    pq = []
    
    # Add edges from first node
    for j in range(1, n):
        heapq.heappush(pq, (dist_matrix[0, j], 0, j))
    
    while len(edges) < n - 1 and pq:
        dist, u_idx, v_idx = heapq.heappop(pq)
        
        if visited[v_idx]:
            continue
        
        visited[v_idx] = True
        edges.append((nodes[u_idx], nodes[v_idx]))
        
        # Add new edges from v_idx
        for j in range(n):
            if not visited[j]:
                heapq.heappush(pq, (dist_matrix[v_idx, j], v_idx, j))
    
    return edges

def build_mst_tree_with_distance(
    V: List[Any],
    y: Dict[Any, np.ndarray],
    X: Optional[Dict[Any, np.ndarray]] = None,
    distance_type: str = "feature",
    seed: Optional[int] = None,
    **kwargs
) -> Tuple[Any, Dict[Any, Any], List[Tuple[Any, Any]], Dict[str, Any]]:
    """
    Build MST-based tree with different distance metrics.
    
    Args:
        V: list of node identifiers
        y: dict mapping node -> target vector
        X: dict mapping node -> feature matrix (needed for gradient/model distances)
        distance_type: one of ["feature", "target", "gradient", "model", "kl", "wasserstein"]
        seed: optional root node (if None, chooses node with minimal total distance)
    
    Returns:
        root, parent_dict, directed_edges, builder_meta
    """
    nodes = sorted(V)
    n = len(nodes)
    
    if n == 0:
        raise ValueError("Empty node list")
    if n == 1:
        return nodes[0], {nodes[0]: None}, [], {"method": f"mst_{distance_type}"}
    
    # Compute distance matrix based on type
    # print(f"  Building MST with distance_type={distance_type}")
    
    if distance_type == "feature":
        dist = compute_feature_distance(nodes, X, y)
    elif distance_type == "target":
        dist = compute_target_distance(nodes, y)
    elif distance_type == "gradient":
        if X is None:
            raise ValueError("X required for gradient distance")
        dist = compute_gradient_distance(nodes, X, y)
    elif distance_type == "model":
        if X is None:
            raise ValueError("X required for model distance")
        regularization = kwargs.get("regularization", 1e-2)
        dist = compute_model_distance(nodes, X, y, regularization)
    elif distance_type == "kl":
        dist = compute_kl_distance(nodes, y)
    elif distance_type == "wasserstein":
        dist = compute_wasserstein_distance(nodes, y)
    elif distance_type == "mmd_feature":
        dist = compute_mmd_feature_distance(nodes, X, y)
    elif distance_type == "mean_cov_feature":
        dist = compute_mean_cov_feature_distance(nodes, X, y)
    elif distance_type == "js_target":
        dist = compute_js_target_distance(nodes, X, y)
    elif distance_type == "cka":
        dist = compute_cka_distance(nodes, X, y)
    else:
        raise ValueError(f"Unknown distance_type: {distance_type}")
        
    # Compute MST
    undirected_edges = mst_from_distance_matrix(nodes, dist)
    
    # Choose seed if not provided
    if seed is None:
        sum_d = dist.sum(axis=1)
        seed = nodes[int(np.argmin(sum_d))]
    
    # Root the tree
    root, directed_edges, parent = root_tree(undirected_edges, seed)
    
    builder_meta = {
        "method": f"mst_{distance_type}",
        "distance_type": distance_type,
        "nodes": nodes,
        "distance_matrix": dist,
        "num_edges": len(directed_edges),
    }
    
    return root, parent, directed_edges, builder_meta

def build_mst_tree(
    V: List[Any],
    y: Dict[Any, np.ndarray],
    seed: Optional[int] = None,
    **kwargs
) -> Tuple[Any, Dict[Any, Any], List[Tuple[Any, Any]], Dict[str, Any]]:
    """
    Build MST using feature-based distance (original implementation).
    This is equivalent to build_mst_tree_with_distance(..., distance_type="feature")
    """
    return build_mst_tree_with_distance(
        V=V, y=y, X=None, distance_type="cka", seed=seed
    )

def root_tree(undirected_edges: List[Tuple[Any, Any]], root: Any) -> Tuple[Any, List[Tuple[Any, Any]], Dict[Any, Any]]:
    """
    Convert undirected tree to directed tree rooted at given node.
    
    Returns:
        root, directed_edges, parent_dict
    """
    from collections import defaultdict, deque
    
    # Build adjacency list
    adj = defaultdict(list)
    for u, v in undirected_edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # BFS from root to create directed edges
    parent = {root: None}
    directed_edges = []
    q = deque([root])
    
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in parent:
                parent[v] = u
                directed_edges.append((u, v))
                q.append(v)
    
    return root, directed_edges, parent

def compute_node_depths(
    root: Any,
    edges: List[Tuple[Any, Any]],
) -> Dict[Any, int]:
    """Compute depth of each node from root."""
    depths = {root: 0}
    
    # Build children map
    children = defaultdict(list)
    for u, v in edges:
        children[u].append(v)
    
    queue = deque([root])
    while queue:
        u = queue.popleft()
        for v in children[u]:
            depths[v] = depths[u] + 1
            queue.append(v)
    
    return depths

def cascade_train_tree(
    tree_edges: List[Tuple[Any, Any]],
    root: Any,
    X_train_n: Dict[Any, np.ndarray],
    y_train_n: Dict[Any, np.ndarray],
    B: int,
    b_seed: int,
    eta: float,
    device: torch.device,
    model_class: str = 'linear',
    task: str = "regression",
    initial_model: Optional[nn.Module] = None,
    reg: float = 1e-4,
    **kwargs,
) -> Tuple[Dict[Any, Union[nn.Module, np.ndarray]], Any]:
    """
    Unified cascade training for regression and binary classification.

    Args:
        tree_edges: directed edges (parent, child)
        root: root node
        X_train_n: dict node -> [n, d]
        y_train_n: dict node -> targets
        B: total budget
        b_seed: steps for root
        eta: learning rate
        device: torch device
        model_class: linear | mlp | residual | fm | logistic
        task: regression | classification
        initial_model: optional pretrained model for regression
        reg: L2 regularization (logistic only)

    Returns:
        models: dict node -> trained model (nn.Module for regression, np.ndarray for classification)
        tracker: CostTracker
    """
    from collections import defaultdict, deque
    from copy import deepcopy

    assert task in {"regression", "classification"}

    # ------------------------
    # Build node set and children map
    # ------------------------
    nodes = set([root])
    for u, v in tree_edges:
        nodes.add(u)
        nodes.add(v)
    nodes = list(nodes)
    
    children = defaultdict(list)
    for u, v in tree_edges:
        children[u].append(v)

    # ------------------------
    # Budget allocation
    # ------------------------
    n_children = max(0, len(nodes) - 1)
    b_remaining = max(0, B - b_seed)
    b_per_node = max(1, b_remaining // n_children) if n_children > 0 else 0

    d = next(iter(X_train_n.values())).shape[1]

    tracker = CostTracker()
    models = {}

    # ============================================================
    # CLASSIFICATION: logistic regression cascade
    # ============================================================
    if task == "classification":
        assert model_class == "logistic", \
            "For classification, model_class must be 'logistic'"

        theta = {}

        # ---- root ----
        theta[root] = gd_train_logistic(
            np.zeros(d),
            X_train_n[root],
            y_train_n[root],
            steps=b_seed,
            lr=eta,
            reg=reg,
            device=device,
        )

        # ---- cascade via BFS ----
        q = deque([root])
        visited = {root}

        while q:
            u = q.popleft()
            for v in children[u]:
                if v not in visited:
                    theta[v] = gd_train_logistic(
                        theta[u].copy(),
                        X_train_n[v],
                        y_train_n[v],
                        steps=b_per_node,
                        lr=eta,
                        reg=reg,
                        device=device,
                    )
                    visited.add(v)
                    q.append(v)

        # ---- handle disconnected nodes (if any) ----
        for v in nodes:
            if v not in theta:
                theta[v] = gd_train_logistic(
                    np.zeros(d),
                    X_train_n[v],
                    y_train_n[v],
                    steps=b_per_node,
                    lr=eta,
                    reg=reg,
                    device=device,
                )

        return theta, tracker

    # ============================================================
    # REGRESSION: PyTorch models
    # ============================================================
    
    # ---- Initialize root model ----
    if initial_model is not None:
        root_model = deepcopy(initial_model)
    else:
        if model_class == 'linear':
            root_model = LinearModel(d)
        elif model_class == 'mlp':
            root_model = SimpleMLP(d, hidden=kwargs.get('hidden', 32))
        elif model_class == 'residual':
            root_model = ResidualMLP(
                d,
                hidden=kwargs.get('hidden', 32),
                eps=kwargs.get('eps', 0.1),
            )
        elif model_class == 'fm':
            root_model = CascadeFM(device=device)
        else:
            raise ValueError(f"Unknown model_class: {model_class}")

    # ============================================================
    # FAST PATH: CascadeFM (cached features)
    # ============================================================
    if isinstance(root_model, CascadeFM):
        # Cache features once
        features = {}
        for v in nodes:
            Xv = torch.from_numpy(X_train_n[v]).float().to(device)
            features[v] = root_model.extract_features(Xv)

        # ---- Train root ----
        y0 = torch.from_numpy(y_train_n[root]).float().to(device)
        train_linear_head_sgd(
            root_model.head,
            features[root],
            y0,
            steps=b_seed,
            eta=eta,
        )
        models[root] = root_model

        # ---- Cascade via BFS ----
        q = deque([root])
        while q:
            u = q.popleft()
            for v in children[u]:
                child = deepcopy(models[u])
                yv = torch.from_numpy(y_train_n[v]).float().to(device)
                train_linear_head_sgd(
                    child.head,
                    features[v],
                    yv,
                    steps=b_per_node,
                    eta=eta,
                )
                models[v] = child
                q.append(v)

        return models, tracker

    # ============================================================
    # STANDARD PATH: linear / mlp / residual
    # ============================================================
    
    # ---- Train root ----
    root_model = train_model_sgd(
        root_model,
        X_train_n[root],
        y_train_n[root],
        steps=b_seed,
        eta=eta,
        device=device,
        cost_tracker=tracker,
    )
    models[root] = root_model

    # ---- Cascade via BFS ----
    q = deque([root])
    while q:
        u = q.popleft()
        for v in children[u]:
            child = deepcopy(models[u])
            child = train_model_sgd(
                child,
                X_train_n[v],
                y_train_n[v],
                steps=b_per_node,
                eta=eta,
                device=device,
                cost_tracker=tracker,
            )
            models[v] = child
            q.append(v)

    return models, tracker

def cascade_train_tree_with_allocation(
    tree_edges: List[Tuple[Any, Any]],
    root: Any,
    X_train_n: Dict[Any, np.ndarray],
    y_train_n: Dict[Any, np.ndarray],
    budget_allocation: Dict[Any, int],
    eta: float,
    device: torch.device,
    model_class: str = 'linear',
    **kwargs,
) -> Tuple[Dict[Any, nn.Module], Any]:
    """Train models on tree using cascade transfer learning with specified budget allocation."""
    nodes = set([root])
    for u, v in tree_edges:
        nodes.add(u)
        nodes.add(v)
    nodes = list(nodes)
    
    d = next(iter(X_train_n.values())).shape[1]
    
    children = defaultdict(list)
    for u, v in tree_edges:
        children[u].append(v)
    
    if model_class == 'linear':
        root_model = LinearModel(d)
    elif model_class == 'mlp':
        root_model = SimpleMLP(d, hidden=kwargs.get('hidden', 32))
    elif model_class == 'residual':
        root_model = ResidualMLP(d, hidden=kwargs.get('hidden', 32), eps=kwargs.get('eps', 0.1))
    else:
        raise ValueError(f"Unknown model_class: {model_class}")
    
    tracker = CostTracker()
    
    root_steps = budget_allocation.get(root, 0)
    root_model = train_model_sgd(
        root_model, X_train_n[root], y_train_n[root],
        steps=root_steps, eta=eta, device=device,
        cost_tracker=tracker
    )
    
    models = {root: root_model}
    
    queue = deque([root])
    
    while queue:
        u = queue.popleft()
        
        for v in children[u]:
            child_model = clone_model(models[u])
            v_steps = budget_allocation.get(v, 0)
            
            if v_steps > 0:
                child_model = train_model_sgd(
                    child_model, X_train_n[v], y_train_n[v],
                    steps=v_steps, eta=eta, device=device,
                    cost_tracker=tracker
                )
            
            models[v] = child_model
            queue.append(v)
    
    return models, tracker

def random_tree_recursive(T: int, rng: Optional[np.random.Generator] = None) -> List[Tuple[int, int]]:
    """Generate a random tree using random parent assignment."""
    if rng is None:
        rng = np.random.default_rng()
    
    edges = []
    for v in range(1, T):
        parent = rng.integers(0, v)
        edges.append((parent, v))
    return edges

