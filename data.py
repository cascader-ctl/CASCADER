import gzip
from itertools import combinations
import os
import pickle
import matplotlib.pyplot as plt
import urllib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple, Optional, Any

############
# UK dataset
############

def load_weave_data(
    csv_path: str,
    node_col: str = "id_unique",  # Explicitly use id_unique
    y_col: str = "consumption",
    feature_prefix: str = "consumption_l",
    drop_cols: List[str] = None,
) -> Tuple[Dict[Any, np.ndarray], Dict[Any, np.ndarray]]:
    """
    Load WEAVE-style CSV into per-node dicts:
        X_dict[v] -> (T_v, d), y_dict[v] -> (T_v,)
    
    Parameters
    ----------
    csv_path : str
        Path to CSV file.
    node_col : str
        Column identifying unique nodes/sites (default: 'id_unique').
    y_col : str
        Target column name.
    feature_prefix : str
        Prefix for feature columns (e.g., 'consumption_l' for lag features).
    drop_cols : List[str]
        Columns to explicitly drop (e.g., ['Unnamed: 0']).
    """
    df = pd.read_csv(csv_path)
    
    # Drop unwanted columns
    if drop_cols is None:
        drop_cols = ['Unnamed: 0']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    
    # Verify node column exists
    if node_col not in df.columns:
        raise ValueError(f"Node column '{node_col}' not found. Available: {df.columns.tolist()}")
    
    print(f"Using '{node_col}' as node identifier")
    print(f"Unique nodes: {df[node_col].nunique()}")
    
    # Sort by node, then by time if available
    time_col = None
    for cand in ["date", "timestamp", "time"]:
        if cand in df.columns:
            time_col = cand
            break
    
    if time_col is not None:
        df = df.sort_values([node_col, time_col])
        print(f"Sorted by '{node_col}' and '{time_col}'")
    else:
        df = df.sort_values([node_col])
    
    # Get feature columns
    feature_cols = [c for c in df.columns if c.startswith(feature_prefix)]
    if len(feature_cols) == 0:
        raise ValueError(f"No feature columns starting with '{feature_prefix}'")
    
    # add instant + sin/cos temporal features if time_col exists
    if time_col is not None:
        df[time_col] = pd.to_datetime(df[time_col])
        df['hour'] = df[time_col].dt.hour
        df['dayofweek'] = df[time_col].dt.dayofweek
        
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        
        feature_cols += ['hour', 'dayofweek', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']

    print(f"Using {len(feature_cols)} feature columns")
    print(f"Target column: '{y_col}'")
    
    X_dict: Dict[Any, np.ndarray] = {}
    y_dict: Dict[Any, np.ndarray] = {}
    
    for v, g in df.groupby(node_col):
        X = g[feature_cols].to_numpy(dtype=np.float32)
        y = g[y_col].to_numpy(dtype=np.float32)
        
        # Drop rows with NaNs
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X = X[mask]
        y = y[mask]
        
        if len(X) == 0:
            print(f"  Warning: Node {v} has no valid samples, skipping.")
            continue
            
        X_dict[v] = X
        y_dict[v] = y
    
    print(f"Loaded {len(X_dict)} nodes")
    
    return X_dict, y_dict

def train_test_split_per_node(
    X_dict: Dict[Any, np.ndarray],
    y_dict: Dict[Any, np.ndarray],
    test_frac: float = 0.3,
    seed: int = 0,
) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Split data into train/test for each node.
    
    Only used if train/test not already split in separate files.
    """
    rng = np.random.default_rng(seed)
    
    X_train, y_train = {}, {}
    X_test, y_test = {}, {}
    
    for v in X_dict.keys():
        n = X_dict[v].shape[0]
        n_test = max(1, int(n * test_frac))
        n_train = n - n_test
        
        indices = rng.permutation(n)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]
        
        X_train[v] = X_dict[v][train_idx]
        y_train[v] = y_dict[v][train_idx]
        X_test[v] = X_dict[v][test_idx]
        y_test[v] = y_dict[v][test_idx]
    
    return X_train, y_train, X_test, y_test

def normalize_xy(
    X: np.ndarray, 
    y: np.ndarray,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Normalize features and targets to zero mean, unit variance."""
    X_mean = X.mean(axis=0, keepdims=True)
    X_std = X.std(axis=0, keepdims=True) + eps
    
    y_mean = y.mean()
    y_std = y.std() + eps
    
    X_norm = (X - X_mean) / X_std
    y_norm = (y - y_mean) / y_std
    
    stats = {
        "X_mean": X_mean,
        "X_std": X_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }
    
    return X_norm, y_norm, stats


def denormalize_y(y_norm: np.ndarray, stats: Dict) -> np.ndarray:
    """Reverse normalization on predictions."""
    return y_norm * stats["y_std"] + stats["y_mean"]


def compute_metrics_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """Compute MSE and MAPE."""
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    mse = np.mean((y_true - y_pred) ** 2)
    
    mask = np.abs(y_true) > 1e-8
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan
    
    return float(mse), float(mape)

###################
# Synthetic dataset
###################

def generate_tasks(T=256, d=20, n_train=64, n_test=256,
                   tau=0.5, sigma=0.1, rng=None):
    rng = np.random.default_rng(rng)
    theta0 = rng.normal(size=(d,))
    thetas = theta0 + rng.normal(scale=tau, size=(T, d))

    def make_split(n):
        X = rng.normal(size=(T, n, d))
        eps = rng.normal(scale=sigma, size=(T, n))
        y = np.einsum('tnd,td->tn', X, thetas) + eps
        return X, y

    X_tr, y_tr = make_split(n_train)
    X_te, y_te = make_split(n_test)
    return thetas, theta0, (X_tr, y_tr), (X_te, y_te)

def generate_clustered_tasks(T=200, d=10, n_train=32, n_test=128,
                             n_clusters=4, tau_within=0.2, tau_between=1.0,
                             sigma=0.1, rng=None):
    rng = np.random.default_rng(rng)
    theta_global = rng.normal(size=(d,))
    cluster_shifts = rng.normal(scale=tau_between, size=(n_clusters, d))
    cluster_centers = theta_global + cluster_shifts
    thetas = np.zeros((T, d))
    cluster_ids = rng.integers(0, n_clusters, size=T)
    for t in range(T):
        c = cluster_ids[t]
        theta_c = cluster_centers[c]
        thetas[t] = theta_c + rng.normal(scale=tau_within, size=(d,))

    def make_split(n):
        X = rng.normal(size=(T, n, d))
        eps = rng.normal(scale=sigma, size=(T, n))
        y = np.einsum('tnd,td->tn', X, thetas) + eps
        return X, y

    X_tr, y_tr = make_split(n_train)
    X_te, y_te = make_split(n_test)
    return thetas, theta_global, cluster_centers, (X_tr, y_tr), (X_te, y_te), cluster_ids

def plot_tasks_2d(
    thetas,
    theta0=None,
    cluster_ids=None,
    cluster_centers=None,
    title=None,
    figsize=(6, 6),
    ax=None,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    T, d = thetas.shape
    
    # Project to 2D if necessary
    if d > 2:
        # Combine all points for consistent PCA
        all_points = [thetas]
        if theta0 is not None:
            all_points.append(theta0.reshape(1, -1))
        if cluster_centers is not None:
            all_points.append(cluster_centers)
        
        combined = np.vstack(all_points)
        pca = PCA(n_components=2)
        combined_2d = pca.fit_transform(combined)
        
        # Split back
        thetas_2d = combined_2d[:T]
        idx = T
        if theta0 is not None:
            theta0_2d = combined_2d[idx]
            idx += 1
        else:
            theta0_2d = None
        if cluster_centers is not None:
            cluster_centers_2d = combined_2d[idx:idx + len(cluster_centers)]
        else:
            cluster_centers_2d = None
        
        xlabel = f"PC1"
        ylabel = f"PC2"
    else:
        thetas_2d = thetas[:, :2]
        theta0_2d = theta0[:2] if theta0 is not None else None
        cluster_centers_2d = cluster_centers[:, :2] if cluster_centers is not None else None
        xlabel = r"$\theta_1$"
        ylabel = r"$\theta_2$"
        
    # Plot task parameters
    if cluster_ids is not None:
        cluster_ids_arr = np.asarray(cluster_ids)
        unique_clusters = np.unique(cluster_ids_arr)
        n_clusters = len(unique_clusters)
        cmap = plt.cm.get_cmap('tab10', max(1, n_clusters))
        for idx_c, c in enumerate(unique_clusters):
            mask = cluster_ids_arr == c
            ax.scatter(thetas_2d[mask, 0], thetas_2d[mask, 1],
                       color=cmap(idx_c), alpha=0.6, s=50, label=f"Cluster {int(c)+1}")
        
        # Plot cluster centers
        if cluster_centers_2d is not None:
            for idx_c, c in enumerate(unique_clusters):
                if idx_c < len(cluster_centers_2d):
                    ax.scatter(cluster_centers_2d[idx_c, 0], cluster_centers_2d[idx_c, 1],
                               color=cmap(idx_c), marker='X', s=140, edgecolors='black', linewidths=1.2)
    else:
        ax.scatter(thetas_2d[:, 0], thetas_2d[:, 1],
                   c='steelblue', alpha=0.6, s=50, label="Tasks")
    
    # Plot global mean
    if theta0_2d is not None:
        ax.scatter(theta0_2d[0], theta0_2d[1],
                   c='red', marker='*', s=200, edgecolors='black',
                   linewidths=1.2, label=r"$\theta_0$ (global mean)", zorder=10)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    # reduce legend clutter if many clusters
    try:
        if len(ax.legend_.texts) if getattr(ax, "legend_", None) is not None else True:
            ax.legend(loc='best', fontsize=8)
    except Exception:
        ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    return fig, ax

def plot_clustered_tasks_2d(
    T=200,
    d=10,
    n_train=32,
    n_test=128,
    n_clusters=4,
    tau_within_values=(2, 5, 10),
    tau_between=1.5,
    sigma=0.1,
    seed=42,
    figsize=(18, 6),
):
    """
    Plot clustered task parameters in 2D for multiple tau_within values,
    fused into a single figure with subfigures in a row.
    """
    rng = np.random.default_rng(seed)

    fig, axs = plt.subplots(
        1, len(tau_within_values),
        figsize=figsize,
        # sharex=True,
        # sharey=True,
    )

    if len(tau_within_values) == 1:
        axs = [axs]

    for ax, tau_within in zip(axs, tau_within_values):
        thetas, theta_global, cluster_centers, _, _, cluster_ids = (
            generate_clustered_tasks(
                T=T,
                d=d,
                n_train=n_train,
                n_test=n_test,
                n_clusters=n_clusters,
                tau_within=tau_within,
                tau_between=tau_between,
                sigma=sigma,
                rng=rng,
            )
        )

        plot_tasks_2d(
            thetas,
            theta0=theta_global,
            cluster_ids=cluster_ids,
            cluster_centers=cluster_centers,
            title=rf"$\tau_{{\mathrm{{within}}}} = {tau_within}$",
            ax=ax,
        )

    fig.suptitle(
        rf"Synthetic Tasks (T={T}, K={n_clusters}, $\tau_{{\mathrm{{between}}}}={tau_between}$, $\sigma={sigma}$)",
        fontsize=14,
        y=1.02,
    )

    plt.tight_layout()
    return fig, axs

############
# Image dataset
############

def download_file(url: str, filepath: str):
    """Download file if it doesn't exist."""
    if not os.path.exists(filepath):
        print(f"Downloading {url}...")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        urllib.request.urlretrieve(url, filepath)
        print(f"Saved to {filepath}")


def load_fashion_mnist(data_dir: str = './data/fashion_mnist') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load Fashion-MNIST dataset using pure NumPy."""
    base_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
    files = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz'
    }
    
    os.makedirs(data_dir, exist_ok=True)
    
    # Download files
    for key, filename in files.items():
        filepath = os.path.join(data_dir, filename)
        download_file(base_url + filename, filepath)
    
    # Load images
    def load_images(filepath):
        with gzip.open(filepath, 'rb') as f:
            # Skip magic number and dimensions
            f.read(16)
            data = np.frombuffer(f.read(), dtype=np.uint8)
            return data.reshape(-1, 28*28).astype(np.float32) / 255.0
    
    def load_labels(filepath):
        with gzip.open(filepath, 'rb') as f:
            # Skip magic number and count
            f.read(8)
            return np.frombuffer(f.read(), dtype=np.uint8)
    
    X_train = load_images(os.path.join(data_dir, files['train_images']))
    y_train = load_labels(os.path.join(data_dir, files['train_labels']))
    X_test = load_images(os.path.join(data_dir, files['test_images']))
    y_test = load_labels(os.path.join(data_dir, files['test_labels']))
    
    return X_train, y_train, X_test, y_test


def load_cifar10(data_dir: str = './data/cifar10') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load CIFAR-10 dataset using pure NumPy."""
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filepath = os.path.join(data_dir, "cifar-10-python.tar.gz")
    extracted_dir = os.path.join(data_dir, "cifar-10-batches-py")
    
    os.makedirs(data_dir, exist_ok=True)
    
    if not os.path.exists(extracted_dir):
        download_file(url, filepath)
        
        import tarfile
        print("Extracting CIFAR-10...")
        with tarfile.open(filepath, 'r:gz') as tar:
            tar.extractall(data_dir)
    
    def load_batch(filepath):
        with open(filepath, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
            data = batch[b'data'].astype(np.float32) / 255.0
            labels = np.array(batch[b'labels'])
            return data, labels
    
    # Load training batches
    X_train_list, y_train_list = [], []
    for i in range(1, 6):
        batch_file = os.path.join(extracted_dir, f'data_batch_{i}')
        X_batch, y_batch = load_batch(batch_file)
        X_train_list.append(X_batch)
        y_train_list.append(y_batch)
    
    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)
    
    X_test, y_test = load_batch(os.path.join(extracted_dir, 'test_batch'))
    return X_train, y_train, X_test, y_test

def pca_fit_transform(X: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit PCA and transform data (pure NumPy implementation)."""
    mean = X.mean(axis=0)
    X_centered = X - mean
    
    if X.shape[0] < X.shape[1]:
        cov = X_centered @ X_centered.T / (X.shape[0] - 1)
        U, S, _ = np.linalg.svd(cov)
        components = X_centered.T @ U[:, :n_components]
        components = components / np.linalg.norm(components, axis=0)
        components = components.T
    else:
        cov = X_centered.T @ X_centered / (X.shape[0] - 1)
        _, S, Vt = np.linalg.svd(cov)
        components = Vt[:n_components]
    
    X_transformed = X_centered @ components.T
    
    explained_var = S[:n_components].sum() / S.sum()
    print(f"PCA: {n_components} components, explained variance: {explained_var:.3f}")
    return X_transformed.astype(np.float32), mean, components


def pca_transform(X: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    """Transform data using fitted PCA."""
    X_centered = X - mean
    return (X_centered @ components.T).astype(np.float32)


def standardize(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Standardize features to zero mean and unit variance."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std

def generate_binary_tasks(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    n_tasks: int,
    n_train_per_task: int = 100,
    n_test_per_task: int = 200,
    task_type: str = "random_pairs",
    rng: Optional[np.random.Generator] = None,
    SEED: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    """
    Generate binary classification tasks from multi-class dataset.
    
    Parameters
    ----------
    X_train, y_train : training data and labels
    X_test, y_test : test data and labels
    n_tasks : number of tasks to generate
    n_train_per_task : training samples per task
    n_test_per_task : test samples per task
    task_type : how to generate class pairs
        - "random_pairs": random pairs of classes
        - "one_vs_all": class 0 vs each other class
        - "sequential": (0,1), (1,2), (2,3), etc.
    
    Returns
    -------
    X_tr : (n_tasks, n_train_per_task, d)
    y_tr : (n_tasks, n_train_per_task)
    X_te : (n_tasks, n_test_per_task, d)
    y_te : (n_tasks, n_test_per_task)
    class_pairs : list of (class_a, class_b)
    """
    if rng is None:
        rng = np.random.default_rng(SEED)
    
    n_classes = len(np.unique(y_train))
    d = X_train.shape[1]
    
    all_pairs = list(combinations(range(n_classes), 2))
    
    if task_type == "random_pairs":
        pair_indices = rng.choice(len(all_pairs), size=n_tasks, replace=True)
        class_pairs = [all_pairs[i] for i in pair_indices]
    elif task_type == "one_vs_all":
        class_pairs = [(0, i) for i in range(1, n_classes)]
        while len(class_pairs) < n_tasks:
            class_pairs = class_pairs + class_pairs
        class_pairs = class_pairs[:n_tasks]
    elif task_type == "sequential":
        class_pairs = [(i, (i+1) % n_classes) for i in range(n_classes)]
        while len(class_pairs) < n_tasks:
            class_pairs = class_pairs + class_pairs
        class_pairs = class_pairs[:n_tasks]
    else:
        raise ValueError(f"Unknown task_type: {task_type}")
    
    X_tr = np.zeros((n_tasks, n_train_per_task, d), dtype=np.float32)
    y_tr = np.zeros((n_tasks, n_train_per_task), dtype=np.float32)
    X_te = np.zeros((n_tasks, n_test_per_task, d), dtype=np.float32)
    y_te = np.zeros((n_tasks, n_test_per_task), dtype=np.float32)
    
    for t, (class_a, class_b) in enumerate(class_pairs):
        train_idx_a = np.where(y_train == class_a)[0]
        train_idx_b = np.where(y_train == class_b)[0]
        test_idx_a = np.where(y_test == class_a)[0]
        test_idx_b = np.where(y_test == class_b)[0]
        
        n_per_class_train = n_train_per_task // 2
        sampled_train_a = rng.choice(train_idx_a, size=n_per_class_train, replace=False)
        sampled_train_b = rng.choice(train_idx_b, size=n_per_class_train, replace=False)
        
        X_tr[t, :n_per_class_train] = X_train[sampled_train_a]
        X_tr[t, n_per_class_train:] = X_train[sampled_train_b]
        y_tr[t, :n_per_class_train] = 0
        y_tr[t, n_per_class_train:] = 1
        
        perm = rng.permutation(n_train_per_task)
        X_tr[t] = X_tr[t, perm]
        y_tr[t] = y_tr[t, perm]
        
        n_per_class_test = n_test_per_task // 2
        sampled_test_a = rng.choice(test_idx_a, size=n_per_class_test, replace=False)
        sampled_test_b = rng.choice(test_idx_b, size=n_per_class_test, replace=False)
        
        X_te[t, :n_per_class_test] = X_test[sampled_test_a]
        X_te[t, n_per_class_test:] = X_test[sampled_test_b]
        y_te[t, :n_per_class_test] = 0
        y_te[t, n_per_class_test:] = 1
        
        perm = rng.permutation(n_test_per_task)
        X_te[t] = X_te[t, perm]
        y_te[t] = y_te[t, perm]
    
    return X_tr, y_tr, X_te, y_te, class_pairs


def generate_clustered_binary_tasks(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    n_tasks: int,
    n_clusters: int = 4,
    n_train_per_task: int = 100,
    n_test_per_task: int = 200,
    rng: Optional[np.random.Generator] = None,
    SEED: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Tuple[int, int]], np.ndarray]:
    """Generate clustered binary tasks where tasks in same cluster share class pairs."""
    if rng is None:
        rng = np.random.default_rng(SEED)
    
    n_classes = len(np.unique(y_train))
    d = X_train.shape[1]
    all_pairs = list(combinations(range(n_classes), 2))
    
    # Assign cluster pairs
    cluster_pair_idx = rng.choice(len(all_pairs), size=n_clusters, replace=False)
    cluster_pairs = [all_pairs[i] for i in cluster_pair_idx]
    
    # Assign tasks to clusters
    tasks_per_cluster = n_tasks // n_clusters
    cluster_ids = np.repeat(np.arange(n_clusters), tasks_per_cluster)
    if len(cluster_ids) < n_tasks:
        cluster_ids = np.concatenate([cluster_ids, rng.choice(n_clusters, size=n_tasks - len(cluster_ids))])
    rng.shuffle(cluster_ids)
    
    class_pairs = [cluster_pairs[cluster_ids[t]] for t in range(n_tasks)]
    
    X_tr = np.zeros((n_tasks, n_train_per_task, d), dtype=np.float32)
    y_tr = np.zeros((n_tasks, n_train_per_task), dtype=np.float32)
    X_te = np.zeros((n_tasks, n_test_per_task, d), dtype=np.float32)
    y_te = np.zeros((n_tasks, n_test_per_task), dtype=np.float32)
    
    for t, (class_a, class_b) in enumerate(class_pairs):
        train_idx_a = np.where(y_train == class_a)[0]
        train_idx_b = np.where(y_train == class_b)[0]
        test_idx_a = np.where(y_test == class_a)[0]
        test_idx_b = np.where(y_test == class_b)[0]
        
        n_per_class_train = n_train_per_task // 2
        sampled_train_a = rng.choice(train_idx_a, size=n_per_class_train, replace=False)
        sampled_train_b = rng.choice(train_idx_b, size=n_per_class_train, replace=False)
        
        X_tr[t, :n_per_class_train] = X_train[sampled_train_a]
        X_tr[t, n_per_class_train:] = X_train[sampled_train_b]
        y_tr[t, :n_per_class_train] = 0
        y_tr[t, n_per_class_train:] = 1
        
        perm = rng.permutation(n_train_per_task)
        X_tr[t] = X_tr[t, perm]
        y_tr[t] = y_tr[t, perm]
        
        n_per_class_test = n_test_per_task // 2
        sampled_test_a = rng.choice(test_idx_a, size=n_per_class_test, replace=False)
        sampled_test_b = rng.choice(test_idx_b, size=n_per_class_test, replace=False)
        
        X_te[t, :n_per_class_test] = X_test[sampled_test_a]
        X_te[t, n_per_class_test:] = X_test[sampled_test_b]
        y_te[t, :n_per_class_test] = 0
        y_te[t, n_per_class_test:] = 1
        
        perm = rng.permutation(n_test_per_task)
        X_te[t] = X_te[t, perm]
        y_te[t] = y_te[t, perm]
    
    return X_tr, y_tr, X_te, y_te, class_pairs, cluster_ids
