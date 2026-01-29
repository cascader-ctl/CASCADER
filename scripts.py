from collections import defaultdict, deque
from dataclasses import dataclass
from data import generate_binary_tasks, generate_clustered_binary_tasks, generate_clustered_tasks, generate_tasks, normalize_xy, denormalize_y
from graph import build_mst_tree_with_distance, cascade_train_tree, cascade_train_tree_with_allocation, random_tree_recursive, select_seed_node, random_tree_edges, root_tree, mst_from_distance_matrix
from models import CascadeFM, LinearModel, SimpleMLP, ResidualMLP
import torch
from trainer import compute_accuracy, compute_metrics_arrays, CostTracker, DEVICE, gd_train_logistic, train_linear_head_sgd, train_model_sgd, predict_node
from typing import Dict, Any, Optional
import numpy as np
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

def compute_budget_allocation(B: int, T: int, seed_fraction: float = 0.1) -> Tuple[int, int]:
    """
    Return (seed_steps, child_steps) such that:
      seed_steps + (T-1)*child_steps <= B
    """
    seed_steps = max(1, int(seed_fraction * B))
    remaining = max(0, B - seed_steps)
    n_children = max(1, T - 1)
    child_steps = remaining // n_children
    
    if child_steps == 0 and remaining > 0:
        child_steps = 1
    
    # Safety check
    total_used = seed_steps + n_children * child_steps
    while total_used > B and child_steps > 0:
        child_steps -= 1
        total_used = seed_steps + n_children * child_steps
    
    assert seed_steps + n_children * child_steps <= B
    return seed_steps, child_steps


def compute_indiv_steps(B: int, T: int) -> int:
    """Compute steps per task for independent training."""
    return max(1, B // T)

def run_indiv(
    X_train_n: Dict,
    y_train_n: Dict,
    X_test_n: Dict,
    y_test: Dict,
    node_stats: Dict,
    B: int,
    eta: float,
    model_class: str = 'linear',
    **kwargs
) -> Tuple[Dict, Dict, Dict, Dict]:
    """Run individual (no-transfer) training baseline."""
    
    nodes = sorted(X_train_n.keys())
    n = len(nodes)
    d = next(iter(X_train_n.values())).shape[1]

    # Budget per node
    b_per_node = max(1, B // n)

    mses, mapes, preds = {}, {}, {}
    tracker = CostTracker()

    if model_class == 'fm':
        # Instantiate ONE FM (shared frozen backbone)
        fm = CascadeFM(device=DEVICE)

        # ---- cache features once ----
        features_train = {}
        features_test = {}
        for v in nodes:
            Xtr = torch.from_numpy(X_train_n[v]).float().to(DEVICE)
            Xte = torch.from_numpy(X_test_n[v]).float().to(DEVICE)
            features_train[v] = fm.extract_features(Xtr)
            features_test[v] = fm.extract_features(Xte)

        # ---- train independent heads ----
        for v in nodes:
            # fresh head per task
            model = CascadeFM(device=DEVICE)
            ytr = torch.from_numpy(y_train_n[v]).float().to(DEVICE)

            train_linear_head_sgd(
                model.head,
                features_train[v],
                ytr,
                steps=b_per_node,
                eta=eta,
            )

            # prediction
            with torch.no_grad():
                yhat_te_n = model.forward_from_features(features_test[v]).cpu().numpy()

            yhat_te = denormalize_y(yhat_te_n, node_stats[v])
            mse_v, mape_v = compute_metrics_arrays(y_test[v], yhat_te)

            mses[v] = mse_v
            mapes[v] = mape_v
            preds[v] = {"yhat_test": yhat_te}

        info = {"tracker": tracker}
        return mses, mapes, preds, info

    for v in nodes:
        if model_class == 'linear':
            model = LinearModel(d)
        elif model_class == 'mlp':
            model = SimpleMLP(d)
        elif model_class == 'residual':
            model = ResidualMLP(
                d,
                hidden=kwargs.get('hidden', 32),
                eps=kwargs.get('eps', 0.1),
            )
        else:
            raise ValueError(
                f"Unknown model_class: {model_class}. "
                "Choose 'linear', 'mlp', 'residual' or 'fm'."
            )

        model = train_model_sgd(
            model,
            X_train_n[v],
            y_train_n[v],
            steps=b_per_node,
            eta=eta,
            device=DEVICE,
            cost_tracker=tracker,
        )

        yhat_te_n = predict_node(model, X_test_n[v])
        yhat_te = denormalize_y(yhat_te_n, node_stats[v])

        mse_v, mape_v = compute_metrics_arrays(y_test[v], yhat_te)
        mses[v] = mse_v
        mapes[v] = mape_v
        preds[v] = {"yhat_test": yhat_te}

    info = {"tracker": tracker}
    return mses, mapes, preds, info


def run_star(
    X_train_n: Dict,
    y_train_n: Dict,
    X_test_n: Dict,
    y_test: Dict,
    node_stats: Dict,
    B: int,
    b_seed: int,
    eta: float,
    global_root: Any,
    model_class: str = 'linear',
    **kwargs
) -> Tuple[Dict, Dict, Dict]:
    """Run star topology CTL."""
    nodes = sorted(X_train_n.keys())
    
    # Star edges: root -> all others
    tree_edges = [(global_root, v) for v in nodes if v != global_root]
    
    models, tracker = cascade_train_tree(
        tree_edges=tree_edges,
        root=global_root,
        X_train_n=X_train_n,
        y_train_n=y_train_n,
        B=B,
        b_seed=b_seed,
        eta=eta,
        device=DEVICE,
        model_class=model_class,
        **kwargs
    )
    
    mses, mapes, preds = {}, {}, {}
    for v in nodes:
        yhat_te_n = predict_node(models[v], X_test_n[v])
        yhat_te = denormalize_y(yhat_te_n, node_stats[v])
        
        mse_v, mape_v = compute_metrics_arrays(y_test[v], yhat_te)
        mses[v] = mse_v
        mapes[v] = mape_v
        preds[v] = {"yhat_test": yhat_te}
    
    info = {"tracker": tracker}
    return mses, mapes, preds, info


def run_random_tree(
    X_train_n: Dict,
    y_train_n: Dict,
    X_test_n: Dict,
    y_test: Dict,
    node_stats: Dict,
    B: int,
    b_seed: int,
    eta: float,
    global_root: Any,
    seed: int = 0,
    model_class: str = 'linear',
    **kwargs
) -> Tuple[Dict, Dict, Dict, Dict]:
    """Run random tree CTL."""
    nodes = sorted(X_train_n.keys())
    rng = np.random.default_rng(seed)
    
    # Generate random tree
    edges_undirected = random_tree_edges(nodes, rng)
    _, tree_edges, parent = root_tree(edges_undirected, global_root)
    models, tracker = cascade_train_tree(
        tree_edges=tree_edges,
        root=global_root,
        X_train_n=X_train_n,
        y_train_n=y_train_n,
        B=B,
        b_seed=b_seed,
        eta=eta,
        device=DEVICE,
        model_class=model_class,
        **kwargs
    )
    
    mses, mapes, preds = {}, {}, {}
    for v in nodes:
        yhat_te_n = predict_node(models[v], X_test_n[v])
        yhat_te = denormalize_y(yhat_te_n, node_stats[v])
        
        mse_v, mape_v = compute_metrics_arrays(y_test[v], yhat_te)
        mses[v] = mse_v
        mapes[v] = mape_v
        preds[v] = {"yhat_test": yhat_te}
    
    info = {"root": global_root, "edges": tree_edges, "parent": parent, "tracker": tracker}
    return mses, mapes, preds, info

def run_mst_tree(
    X_train_n: Dict,
    y_train_n: Dict,
    X_test_n: Dict,
    y_test: Dict,
    node_stats: Dict,
    B: int,
    b_seed: int,
    eta: float,
    global_root: Any,
    model_class: str = 'linear',
    observed_labels: bool = True,
    **kwargs
) -> Tuple[Dict, Dict, Dict, Dict]:
    """Run MST-based CTL."""
    nodes = sorted(X_train_n.keys())
    n = len(nodes)
    
    if not observed_labels:
        # Compute distance matrix from mean features
        means = np.stack([X_train_n[v].mean(axis=0) for v in nodes], axis=0)
        dist_matrix = np.linalg.norm(means[:, None, :] - means[None, :, :], axis=-1)
    else:
        # Compute distance matrix from observed labels
        w_obs = np.einsum('tnd,tn->td', np.stack([X_train_n[v] for v in nodes]), np.stack([y_train_n[v].flatten() for v in nodes]))
        dist_matrix = np.linalg.norm(w_obs[:, None, :] - w_obs[None, :, :], axis=-1)

    # Build MST
    edges_undirected = mst_from_distance_matrix(nodes, dist_matrix)
    _, tree_edges, parent = root_tree(edges_undirected, global_root)
    
    models, tracker = cascade_train_tree(
        tree_edges=tree_edges,
        root=global_root,
        X_train_n=X_train_n,
        y_train_n=y_train_n,
        B=B,
        b_seed=b_seed,
        eta=eta,
        device=DEVICE,
        model_class=model_class,
        **kwargs
    )
    
    mses, mapes, preds = {}, {}, {}
    for v in nodes:
        yhat_te_n = predict_node(models[v], X_test_n[v])
        yhat_te = denormalize_y(yhat_te_n, node_stats[v])
        
        mse_v, mape_v = compute_metrics_arrays(y_test[v], yhat_te)
        mses[v] = mse_v
        mapes[v] = mape_v
        preds[v] = {"yhat_test": yhat_te}
    
    info = {"root": global_root, "edges": tree_edges, "parent": parent, "tracker": tracker}
    return mses, mapes, preds, info

def run_sweep_experiment_with_mst_variants(
    Ts: Tuple[int, ...] = (100, 200),
    Bs: Tuple[int, ...] = (500, 1000, 2000),
    d: int = 5,
    n_train: int = 64,
    n_test: int = 128,
    tau: float = 10.0,
    sigma: float = 5.0,
    lr: float = 0.01,
    k_neighbors: int = 10,
    reps: int = 10,
    clustered: bool = False,
    n_clusters: int = 5,
    tau_between: float = 5.0,
    model_class: str = "linear",
    seed_fraction: float = 0.1,
    device: Optional[torch.device] = None,
    base_seed: int = 42,
    mst_distance_types: Tuple[str, ...] = ("feature", "target", "gradient", "model"),
    source_mode: str = "medoid", 
    **align_kwargs,
) -> Tuple[List[str], Dict[str, Dict[Tuple[int, int], List[float]]]]:
    """
    Run sweep experiment with multiple MST variants based on different distance metrics.
    Data is generated once per configuration before running repetitions.
    
    Args:
        source_mode: How to initialize root for tree methods
            - "medoid": Select medoid task as root, train on its data
            - "pooled": Pool all data, train shared source, then cascade
    
    Includes Task Clustering + Shared Initialization baseline.
    """    
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    if source_mode not in ("medoid", "pooled"):
        raise ValueError(f"source_mode must be 'medoid' or 'pooled', got {source_mode}")
    
    # Build method list including all MST variants
    methods = ["Indiv", "Star", "CTL-RandTree"]

    for dist_type in mst_distance_types:
        methods.append(f"CTL-MST-{dist_type}")
        
    # Store per-repetition mean MSEs
    results = {m: {(T, B): [] for T in Ts for B in Bs} for m in methods}
    
    base_rng = np.random.default_rng(base_seed)
    
    for T in Ts:
        for B in Bs:
            print(f"\n{'='*60}")
            print(f"Running T={T}, B={B}, source_mode={source_mode}")
            print(f"{'='*60}")
            
            # Compute budget allocation
            indiv_steps = compute_indiv_steps(B, T)
            seed_steps, child_steps = compute_budget_allocation(B, T, seed_fraction)
            k_nn = min(k_neighbors, T - 1)
            
            # Adjust n_clusters for this T
            n_clusters_adj = min(n_clusters, max(2, T // 3))
            
            # Generate all data for this configuration upfront
            print(f"Generating {reps} datasets for T={T}, B={B}...")
            datasets = []
            for rep in range(reps):
                rep_seed = base_rng.integers(1, 1_000_000)
                rng = np.random.default_rng(rep_seed)
                
                # Generate data
                if clustered:
                    thetas, theta_global, cluster_centers, (X_tr, y_tr), (X_te, y_te), cluster_ids = (
                        generate_clustered_tasks(
                            T=T, d=d, n_train=n_train, n_test=n_test,
                            n_clusters=n_clusters, tau_within=tau, tau_between=tau_between,
                            sigma=sigma, rng=rng
                        )
                    )
                else:
                    thetas, theta0, (X_tr, y_tr), (X_te, y_te) = generate_tasks(
                        T=T, d=d, n_train=n_train, n_test=n_test,
                        tau=tau, sigma=sigma, rng=rng
                    )
                
                # Convert to dicts
                X_train = {i: X_tr[i] for i in range(T)}
                y_train = {i: y_tr[i] for i in range(T)}
                X_test = {i: X_te[i] for i in range(T)}
                y_test = {i: y_te[i] for i in range(T)}
                
                datasets.append({
                    'X_train': X_train,
                    'y_train': y_train,
                    'X_test': X_test,
                    'y_test': y_test,
                    'rep_seed': rep_seed,
                })
            
            print(f"Data generation complete. Running experiments...")
            
            # Now run experiments on pre-generated data
            for rep, data in enumerate(datasets):
                rep_seed = data['rep_seed']
                rng = np.random.default_rng(rep_seed)
                
                X_train = data['X_train']
                y_train = data['y_train']
                X_test = data['X_test']
                y_test = data['y_test']
                nodes = list(range(T))
                
                # Normalize
                X_train_n, y_train_n, node_stats = {}, {}, {}
                X_test_n = {}
                for v in nodes:
                    Xn, yn, stats = normalize_xy(X_train[v], y_train[v])
                    X_train_n[v] = Xn
                    y_train_n[v] = yn
                    node_stats[v] = stats
                    X_test_n[v] = (X_test[v] - stats["X_mean"]) / stats["X_std"]
                
                # Select seed node (medoid) - used for tree structure even in pooled mode
                seed = select_seed_node(X_train, y_train)
                
                def evaluate_models(models):
                    mses = {}
                    for v, model in models.items():
                        if v == "source":  # Skip virtual source node
                            continue
                        yhat_n = predict_node(model, X_test_n[v])
                        yhat = denormalize_y(yhat_n, node_stats[v])
                        mse, _ = compute_metrics_arrays(y_test[v], yhat)
                        mses[v] = mse
                    return mses
                
                def create_model():
                    """Helper to create a fresh model."""
                    if model_class == "linear":
                        return LinearModel(d)
                    elif model_class == "mlp":
                        return SimpleMLP(d, hidden=align_kwargs.get("hidden", 32))
                    else:
                        return LinearModel(d)
                
                # =====================================================
                # PREPARE POOLED SOURCE MODEL (if source_mode == "pooled")
                # =====================================================
                source_model = None
                if source_mode == "pooled":
                    X_pooled_all = np.vstack([X_train_n[v] for v in nodes])
                    y_pooled_all = np.vstack([y_train_n[v].reshape(-1, 1) for v in nodes])
                    
                    source_model = create_model()
                    source_model = train_model_sgd(
                        source_model, X_pooled_all, y_pooled_all,
                        steps=seed_steps, eta=lr, device=device
                    )
                
                # =====================================================
                # 1. INDIVIDUAL BASELINE
                # =====================================================
                models_indiv = {}
                for v in nodes:
                    model = create_model()
                    model = train_model_sgd(
                        model, X_train_n[v], y_train_n[v],
                        steps=indiv_steps, eta=lr, device=device
                    )
                    models_indiv[v] = model
                
                mses_indiv = evaluate_models(models_indiv)
                results["Indiv"][(T, B)].append(np.mean(list(mses_indiv.values())))
                
                # =====================================================
                # 2. STAR TOPOLOGY
                # =====================================================
                edges_star = [(seed, v) for v in nodes if v != seed]
                models_star, _ = cascade_train_tree(
                    tree_edges=edges_star, root=seed,
                    X_train_n=X_train_n, y_train_n=y_train_n,
                    B=B, b_seed=seed_steps, eta=lr, device=device,
                    model_class=model_class,
                    initial_model=source_model,  # None if medoid, pre-trained if pooled
                    **{k: v for k, v in align_kwargs.items() if k in ["hidden", "eps"]}
                )
                
                mses_star = evaluate_models(models_star)
                results["Star"][(T, B)].append(np.mean(list(mses_star.values())))
                
                # =====================================================
                # 3. RANDOM TREE
                # =====================================================
                edges_rand = []
                for v in range(1, T):
                    parent = rng.integers(0, v)
                    edges_rand.append((parent, v))
                
                adj = defaultdict(list)
                for u, v in edges_rand:
                    adj[u].append(v)
                    adj[v].append(u)
                
                parent_map = {seed: None}
                edges_rand_directed = []
                q = deque([seed])
                while q:
                    u = q.popleft()
                    for v in adj[u]:
                        if v not in parent_map:
                            parent_map[v] = u
                            edges_rand_directed.append((u, v))
                            q.append(v)
                
                models_rand, _ = cascade_train_tree(
                    tree_edges=edges_rand_directed, root=seed,
                    X_train_n=X_train_n, y_train_n=y_train_n,
                    B=B, b_seed=seed_steps, eta=lr, device=device,
                    model_class=model_class,
                    initial_model=source_model,
                    **{k: v for k, v in align_kwargs.items() if k in ["hidden", "eps"]}
                )
                
                mses_rand = evaluate_models(models_rand)
                results["CTL-RandTree"][(T, B)].append(np.mean(list(mses_rand.values())))
                
                # =====================================================
                # 4. MST VARIANTS
                # =====================================================
                y_means = {v: y_train_n[v].flatten() for v in nodes}
                mst_mses = {}
                
                for dist_type in mst_distance_types:
                    method_name = f"CTL-MST-{dist_type}"
                    
                    try:
                        root_mst, parent_mst, edges_mst, builder_mst = build_mst_tree_with_distance(
                            V=nodes,
                            y=y_means,
                            X=X_train_n if dist_type in ["gradient", "model"] else None,
                            distance_type=dist_type,
                            seed=seed,
                        )
                        
                        models_mst, _ = cascade_train_tree(
                            tree_edges=edges_mst, root=root_mst,
                            X_train_n=X_train_n, y_train_n=y_train_n,
                            B=B, b_seed=seed_steps, eta=lr, device=device,
                            model_class=model_class,
                            initial_model=source_model,
                            **{k: v for k, v in align_kwargs.items() if k in ["hidden", "eps"]}
                        )
                        
                        mses_mst = evaluate_models(models_mst)
                        mst_mean = np.mean(list(mses_mst.values()))
                        mst_mses[dist_type] = mst_mean
                        results[method_name][(T, B)].append(mst_mean)
                    
                    except Exception as e:
                        print(f"Warning: {method_name} failed: {e}")
                        import traceback
                        traceback.print_exc()
                        results[method_name][(T, B)].append(np.nan)
                        mst_mses[dist_type] = np.nan
                                
                # Print progress
                mst_results_str = ", ".join([
                    f"{dt}={mst_mses.get(dt, np.nan):.1f}"
                    for dt in mst_distance_types
                ])
                
                print(
                    f"  rep {rep+1}/{reps} | "
                    f"Indiv={np.mean(list(mses_indiv.values())):.1f}, "
                    f"Star={np.mean(list(mses_star.values())):.1f}, "
                    f"Rand={np.mean(list(mses_rand.values())):.1f}, "
                    f"MST[{mst_results_str}]"
                )
    
    return methods, results

def compute_node_depths_from_edges(
    root: Any, 
    edges: List[Tuple[Any, Any]]
) -> Dict[Any, int]:
    """Compute depth of each node from root using BFS."""
    children = defaultdict(list)
    for parent, child in edges:
        children[parent].append(child)
    
    depths = {root: 0}
    queue = deque([root])
    
    while queue:
        node = queue.popleft()
        for child in children[node]:
            depths[child] = depths[node] + 1
            queue.append(child)
    
    return depths


def compute_parent_map(
    root: Any,
    edges: List[Tuple[Any, Any]]
) -> Dict[Any, Any]:
    """Build parent map from directed edges."""
    parent_map = {root: None}
    for parent, child in edges:
        parent_map[child] = parent
    return parent_map


@dataclass
class AllocationConfig:
    """Configuration for budget allocation."""
    method: str
    alpha: float = 1.0
    beta: float = 1.0
    eps: float = 1e-8


def allocate_budget_from_weights(
    total_budget: int,
    weights: Dict[Any, float],
    min_budget: int = 0
) -> Dict[Any, int]:
    """Allocate integer budgets proportional to weights using largest-remainder method."""
    nodes = list(weights.keys())
    n = len(nodes)
    
    if n == 0:
        return {}
    
    min_total = min_budget * n
    if total_budget < min_total:
        raise ValueError(f"Total budget {total_budget} < minimum required {min_total}")
    
    distributable = total_budget - min_total
    
    # Normalize weights
    total_weight = sum(weights.values())
    if total_weight <= 0:
        total_weight = n
        weights = {node: 1.0 for node in nodes}
    
    # Compute real allocations
    real_alloc = {
        node: distributable * (weights[node] / total_weight)
        for node in nodes
    }
    
    # Floor allocations
    floor_alloc = {node: int(np.floor(real_alloc[node])) for node in nodes}
    
    # Compute remainders
    remainders = {node: real_alloc[node] - floor_alloc[node] for node in nodes}
    
    # Distribute remaining budget
    allocated = sum(floor_alloc.values())
    remaining = distributable - allocated
    
    sorted_nodes = sorted(nodes, key=lambda n: remainders[n], reverse=True)
    
    for i in range(int(remaining)):
        floor_alloc[sorted_nodes[i]] += 1
    
    # Add minimum budget
    final_alloc = {node: floor_alloc[node] + min_budget for node in nodes}
    
    assert sum(final_alloc.values()) == total_budget
    
    return final_alloc


def compute_allocation_weights(
    nodes: List[Any],
    root: Any,
    edges: List[Tuple[Any, Any]],
    distance_matrix: Optional[Dict[Tuple[Any, Any], float]],
    config: AllocationConfig
) -> Dict[Any, float]:
    """Compute weights for budget allocation based on the specified method."""
    depths = compute_node_depths_from_edges(root, edges)
    parent_map = compute_parent_map(root, edges)
    
    weights = {}
    
    if config.method == 'uniform':
        weights = {node: 1.0 for node in nodes}
    
    elif config.method == 'depth_increasing':
        for node in nodes:
            weights[node] = (depths[node] + 1) ** config.alpha
    
    elif config.method == 'depth_decreasing':
        for node in nodes:
            weights[node] = 1.0 / ((depths[node] + 1) ** config.alpha)
    
    elif config.method == 'edge_length':
        if distance_matrix is None:
            raise ValueError("edge_length method requires distance_matrix")
        
        for node in nodes:
            parent = parent_map[node]
            if parent is None:
                weights[node] = 1.0
            else:
                dist = distance_matrix.get((parent, node), 
                       distance_matrix.get((node, parent), config.eps))
                weights[node] = (dist + config.eps) ** config.beta
    
    elif config.method == 'hybrid':
        if distance_matrix is None:
            raise ValueError("hybrid method requires distance_matrix")
        
        for node in nodes:
            depth_factor = (depths[node] + 1) ** config.alpha
            
            parent = parent_map[node]
            if parent is None:
                edge_factor = 1.0
            else:
                dist = distance_matrix.get((parent, node),
                       distance_matrix.get((node, parent), config.eps))
                edge_factor = (dist + config.eps) ** config.beta
            
            weights[node] = depth_factor * edge_factor
    
    else:
        raise ValueError(f"Unknown allocation method: {config.method}")
    
    return weights

def compute_budget_allocation_scheme(
    nodes: List[Any],
    root: Any,
    edges: List[Tuple[Any, Any]],
    total_budget: int,
    seed_budget: int,
    distance_matrix: Optional[Dict[Tuple[Any, Any], float]] = None,
    config: Optional[AllocationConfig] = None
) -> Dict[Any, int]:
    """Compute per-node budget allocation for CTL."""
    if config is None:
        config = AllocationConfig(method='uniform')
    
    allocation = {root: seed_budget}
    
    child_budget = total_budget - seed_budget
    child_nodes = [n for n in nodes if n != root]
    
    if len(child_nodes) == 0:
        return allocation
    
    if child_budget <= 0:
        for node in child_nodes:
            allocation[node] = 0
        return allocation
    
    all_weights = compute_allocation_weights(
        nodes, root, edges, distance_matrix, config
    )
    child_weights = {n: all_weights[n] for n in child_nodes}
    
    child_allocation = allocate_budget_from_weights(
        child_budget, child_weights, min_budget=0
    )
    
    allocation.update(child_allocation)
    
    return allocation

def compute_edge_distances(
    edges: List[Tuple[Any, Any]],
    distance_builder: Any
) -> Dict[Tuple[Any, Any], float]:
    """Extract distances for edges from the distance builder."""
    distances = {}
    for u, v in edges:
        try:
            dist = distance_builder.get_distance(u, v)
        except:
            dist = 1.0
        distances[(u, v)] = dist
    return distances

def run_budget_allocation_ablation_sweep(
    T: int = 200,
    d: int = 5,
    n_train: int = 64,
    n_test: int = 128,
    Bs: Tuple[int, ...] = (500, 1000, 2000),
    tau_withins: Tuple[float, ...] = (2.0, 5.0, 10.0), 
    n_clusters: int = 5,
    tau_between: float = 5.0,
    sigma: float = 5.0,
    lr: float = 0.01,
    seed_fraction: float = 0.1,
    reps: int = 50,
    base_seed: int = 42,
    device: Optional[torch.device] = None,
    allocation_methods: Optional[List[str]] = None,
    alpha_values: Tuple[float, ...] = (1.0,),
    beta_values: Tuple[float, ...] = (1.0,),
    report_metric: str = 'mse',
) -> Dict[str, Dict[Tuple[float, int], List[float]]]:
    """
    Run ablation study with sweep over tau_within and budget.
    
    Args:
        tau_withins: Tuple of within-cluster dispersion values to test
        ...other args same as before...
    
    Returns:
        Dict mapping method_name -> {(tau_within, B): [metric_per_rep]}
    """
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    if allocation_methods is None:
        allocation_methods = [
            'uniform',
            'depth_increasing',
            'depth_decreasing', 
            'edge_length',
        ]
    
    # Build all allocation configs to test
    configs_to_test = []
    
    for method in allocation_methods:
        if method == 'uniform':
            configs_to_test.append(('Uniform', AllocationConfig(method='uniform')))
        
        elif method == 'depth_increasing':
            for alpha in alpha_values:
                name = f'Depth inc.(α={alpha})'
                configs_to_test.append((name, AllocationConfig(
                    method='depth_increasing', alpha=alpha
                )))
        
        elif method == 'depth_decreasing':
            for alpha in alpha_values:
                name = f'Depth dec.(α={alpha})'
                configs_to_test.append((name, AllocationConfig(
                    method='depth_decreasing', alpha=alpha
                )))
        
        elif method == 'edge_length':
            for beta in beta_values:
                name = f'EdgeLen(β={beta})'
                configs_to_test.append((name, AllocationConfig(
                    method='edge_length', beta=beta
                )))
        
        elif method == 'hybrid':
            for alpha in alpha_values:
                for beta in beta_values:
                    name = f'Hybrid(α={alpha},β={beta})'
                    configs_to_test.append((name, AllocationConfig(
                        method='hybrid', alpha=alpha, beta=beta
                    )))
    
    # Initialize results storage with (tau_within, B) keys
    results = {
        name: {(tau, B): [] for tau in tau_withins for B in Bs}
        for name, _ in configs_to_test
    }
    
    base_rng = np.random.default_rng(base_seed)
    nodes = list(range(T))
    
    print("=" * 80)
    print("ABLATION STUDY: Budget Allocation Schemes for CTL-MST-gradient")
    print("WITH TAU_WITHIN SWEEP")
    print("=" * 80)
    print(f"Tasks T={T}, dim d={d}, n_train={n_train}, n_test={n_test}")
    print(f"Clusters={n_clusters}, τ_between={tau_between}, σ={sigma}")
    print(f"τ_within values: {tau_withins}")
    print(f"Budgets: {Bs}")
    print(f"Repetitions: {reps}")
    print(f"Allocation methods: {[name for name, _ in configs_to_test]}")
    print("=" * 80)
    
    # Nested loops: tau_within -> B -> reps
    for tau_within in tau_withins:
        print(f"\n{'='*70}")
        print(f"TAU_WITHIN = {tau_within}")
        print(f"{'='*70}")
        
        for B in Bs:
            print(f"\n  Budget B = {B}")
            print(f"  {'-'*60}")
            
            seed_budget = int(B * seed_fraction)
            
            for rep in range(reps):
                rep_seed = base_rng.integers(1, 1_000_000)
                rng = np.random.default_rng(rep_seed)
                
                # Generate clustered data with current tau_within
                thetas, theta_global, cluster_centers, (X_tr, y_tr), (X_te, y_te), cluster_ids = (
                    generate_clustered_tasks(
                        T=T, d=d, n_train=n_train, n_test=n_test,
                        n_clusters=n_clusters, tau_within=tau_within, 
                        tau_between=tau_between, sigma=sigma, rng=rng
                    )
                )
                
                # Convert to dicts
                X_train = {i: X_tr[i] for i in range(T)}
                y_train = {i: y_tr[i] for i in range(T)}
                X_test = {i: X_te[i] for i in range(T)}
                y_test = {i: y_te[i] for i in range(T)}
                
                # Normalize data
                X_train_n, y_train_n, node_stats = {}, {}, {}
                X_test_n = {}
                for v in nodes:
                    Xn, yn, stats = normalize_xy(X_train[v], y_train[v])
                    X_train_n[v] = Xn
                    y_train_n[v] = yn
                    node_stats[v] = stats
                    X_test_n[v] = (X_test[v] - stats["X_mean"]) / stats["X_std"]
                
                # Select seed node
                seed = select_seed_node(X_train, y_train)
                
                # Build MST with gradient distance
                y_means = {v: y_train_n[v].flatten() for v in nodes}
                
                root_mst, parent_mst, edges_mst, builder_mst = build_mst_tree_with_distance(
                    V=nodes,
                    y=y_means,
                    X=X_train_n,
                    distance_type="gradient",
                    seed=seed,
                )
                
                # Extract edge distances
                edge_distances = compute_edge_distances(edges_mst, builder_mst)
                
                # Evaluation function
                def evaluate_models(models):
                    mses = []
                    for v, model in models.items():
                        yhat_n = predict_node(model, X_test_n[v])
                        yhat = denormalize_y(yhat_n, node_stats[v])
                        mse, _ = compute_metrics_arrays(y_test[v], yhat)
                        mses.append(mse)
                    
                    mean_mse = np.mean(mses)
                    
                    if report_metric == 'rmse':
                        return np.sqrt(mean_mse)
                    else:
                        return mean_mse
                
                # Test each allocation scheme
                rep_results = {}
                
                for config_name, config in configs_to_test:
                    try:
                        # Compute budget allocation
                        allocation = compute_budget_allocation_scheme(
                            nodes=nodes,
                            root=root_mst,
                            edges=edges_mst,
                            total_budget=B,
                            seed_budget=seed_budget,
                            distance_matrix=edge_distances,
                            config=config
                        )
                        
                        # Train with this allocation
                        models, _ = cascade_train_tree_with_allocation(
                            tree_edges=edges_mst,
                            root=root_mst,
                            X_train_n=X_train_n,
                            y_train_n=y_train_n,
                            budget_allocation=allocation,
                            eta=lr,
                            device=device,
                            model_class='linear'
                        )
                        
                        # Evaluate
                        mean_metric = evaluate_models(models)
                        results[config_name][(tau_within, B)].append(mean_metric)
                        rep_results[config_name] = mean_metric
                        
                    except Exception as e:
                        print(f"    Warning: {config_name} failed on rep {rep}: {e}")
                        results[config_name][(tau_within, B)].append(np.nan)
                        rep_results[config_name] = np.nan
                
                # Progress output
                if (rep + 1) % 10 == 0 or rep == 0:
                    results_str = " | ".join([
                        f"{name}={rep_results.get(name, np.nan):.1f}"
                        for name, _ in configs_to_test
                    ])
                    print(f"    Rep {rep+1:3d}/{reps} | {results_str}")
    
    return results

def summarize_ablation_results_sweep(
    results: Dict[str, Dict[Tuple[float, int], List[float]]],
    tau_withins: Tuple[float, ...] = (2.0, 5.0, 10.0),
    Bs: Tuple[int, ...] = (500, 1000, 2000),
    metric_name: str = 'MSE'
) -> None:
    """
    Print summary table of ablation results with tau_within sweep.
    
    Args:
        results: Dict from run_budget_allocation_ablation_sweep
        tau_withins: tau_within values tested
        Bs: Budgets tested
        metric_name: Name of metric for display
    """
    methods = list(results.keys())
    
    print("\n" + "=" * 100)
    print(f"ABLATION STUDY RESULTS: Mean {metric_name} ± Std")
    print("=" * 100)
    
    for tau in tau_withins:
        print(f"\nτ_within = {tau}")
        print("-" * 100)
        
        # Header
        header = f"{'Method':<25}"
        for B in Bs:
            header += f" | B={B:<15}"
        print(header)
        print("-" * 100)
        
        # Results for each method
        for method in methods:
            row = f"{method:<25}"
            for B in Bs:
                vals = results[method][(tau, B)]
                vals = [v for v in vals if not np.isnan(v)]
                if vals:
                    mean = np.mean(vals)
                    std = np.std(vals)
                    row += f" | {mean:.2f} ± {std:.2f}  "
                else:
                    row += f" | {'N/A':<15}"
            print(row)
    
    print("=" * 100)
    
    # Find best method for each (tau, B) combination
    print(f"\nBest method per (τ_within, B) combination (lowest {metric_name}):")
    for tau in tau_withins:
        print(f"\n  τ_within = {tau}:")
        for B in Bs:
            best_method = None
            best_mean = float('inf')
            for method in methods:
                vals = [v for v in results[method][(tau, B)] if not np.isnan(v)]
                if vals:
                    mean = np.mean(vals)
                    if mean < best_mean:
                        best_mean = mean
                        best_method = method
            print(f"    B={B}: {best_method} ({metric_name}={best_mean:.2f})")

def run_experiment_grid(
    X_train: Dict[Any, np.ndarray],
    y_train: Dict[Any, np.ndarray],
    X_test: Dict[Any, np.ndarray],
    y_test: Dict[Any, np.ndarray],
    Bs: List[int],
    etas: List[float],
    b_proto_fracs: List[float],
    n_repeats: int = 10,
    model_class: str = "linear",
    mst_distance_types: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
    seed: int = 42,
    source_mode: str = "medoid",
    **kwargs
) -> List[Dict]:
    """
    Run comprehensive experiments with MST variants on WEAVE data.
    
    Args:
        X_train, y_train: Training data dictionaries (node -> array)
        X_test, y_test: Test data dictionaries
        Bs: List of total budgets
        etas: List of learning rates
        b_proto_fracs: List of seed budget fractions
        n_repeats: Number of repetitions per configuration
        model_class: Model architecture ("linear", "mlp", "residual")
        mst_distance_types: List of MST distance metrics to evaluate
        device: Torch device
        seed: Random seed
        source_mode: How to initialize root for tree methods
            - "medoid": Select medoid task as root, train on its data
            - "pooled": Pool all data, train shared source, then cascade
        **kwargs
    
    Returns:
        List of result dictionaries with comprehensive statistics
    """
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    if mst_distance_types is None:
        mst_distance_types = ["feature", "target", "gradient", "model"]
    
    if source_mode not in ("medoid", "pooled"):
        raise ValueError(f"source_mode must be 'medoid' or 'pooled', got {source_mode}")
    
    nodes = sorted(set(X_train.keys()) & set(X_test.keys()))
    T = len(nodes)
    
    print(f"\n{'='*60}")
    print(f"Experiment Configuration")
    print(f"{'='*60}")
    print(f"Nodes: {T}")
    print(f"Budgets: {Bs}")
    print(f"Learning rates: {etas}")
    print(f"Seed fractions: {b_proto_fracs}")
    print(f"Repeats per config: {n_repeats}")
    print(f"Model: {model_class}")
    print(f"MST variants: {mst_distance_types}")
    print(f"Source mode: {source_mode}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    all_results = []
    total_configs = len(Bs) * len(etas) * len(b_proto_fracs)
    config_idx = 0
    
    for B in Bs:
        for eta in etas:
            for b_proto_frac in b_proto_fracs:
                config_idx += 1
                print(f"\n{'='*60}")
                print(f"Config {config_idx}/{total_configs}")
                print(f"B={B}, eta={eta:.4f}, b_proto_frac={b_proto_frac:.3f}")
                print(f"{'='*60}")
                
                # Storage for this configuration
                indiv_rmses, star_rmses, random_rmses = [], [], []
                global_mtl_rmses = []
                mst_rmses = {dist_type: [] for dist_type in mst_distance_types}
                
                full_results_all = []
                
                # Compute budget allocation
                b_seed = max(1, int(b_proto_frac * B))
                
                for rep in range(n_repeats):
                    print(f"\nRepeat {rep+1}/{n_repeats}")
                    
                    # Normalize data
                    X_train_n, y_train_n, node_stats = {}, {}, {}
                    X_test_n = {}
                    
                    for v in nodes:
                        Xn, yn, stats = normalize_xy(X_train[v], y_train[v])
                        X_train_n[v] = Xn
                        y_train_n[v] = yn
                        node_stats[v] = stats
                        X_test_n[v] = (X_test[v] - stats["X_mean"]) / stats["X_std"]
                    
                    # Select seed node (used for tree structure)
                    global_root = select_seed_node(X_train, y_train)
                    
                    # Get input dimension
                    input_dim = next(iter(X_train_n.values())).shape[1]
                    
                    # Helper to create model
                    def create_model():
                        if model_class == "linear":
                            return LinearModel(input_dim)
                        elif model_class == "mlp":
                            return SimpleMLP(input_dim, hidden=kwargs.get("hidden", 32))
                        elif model_class == "residual":
                            return ResidualMLP(input_dim, 
                                             hidden=kwargs.get("hidden", 50),
                                             eps=kwargs.get("eps", 0.03))
                        else:
                            return LinearModel(input_dim)
                    
                    # Evaluation helper
                    def evaluate_models(models):
                        mses = {}
                        for v, model in models.items():
                            if v == "source":
                                continue
                            yhat_n = predict_node(model, X_test_n[v])
                            yhat = denormalize_y(yhat_n, node_stats[v])
                            mse, _ = compute_metrics_arrays(y_test[v], yhat)
                            mses[v] = mse
                        return mses
                    
                    # =====================================================
                    # PREPARE POOLED SOURCE MODEL (if source_mode == "pooled")
                    # =====================================================
                    source_model = None
                    X_pooled_all = np.vstack([X_train_n[v] for v in nodes])
                    y_pooled_all = np.vstack([y_train_n[v].reshape(-1, 1) for v in nodes])
                    
                    if source_mode == "pooled":
                        source_model = create_model()
                        source_model = train_model_sgd(
                            source_model, X_pooled_all, y_pooled_all,
                            steps=b_seed, eta=eta, device=device
                        )
                    
                    # =====================================================
                    # 0. GLOBAL MTL BASELINE
                    # =====================================================
                    print("  Running Global MTL...")
                    try:
                        global_model = create_model()
                        global_model = train_model_sgd(
                            global_model, X_pooled_all, y_pooled_all,
                            steps=B, eta=eta, device=device
                        )
                        
                        models_global = {v: global_model for v in nodes}
                        mses_global = evaluate_models(models_global)
                        global_mtl_rmses.append(np.mean([np.sqrt(mse) for mse in mses_global.values()]))
                        
                    except Exception as e:
                        print(f"    Warning: Global MTL failed: {e}")
                        global_mtl_rmses.append(np.nan)
                        mses_global = {}
                    
                    # =====================================================
                    # 1. INDIVIDUAL BASELINE
                    # =====================================================
                    print("  Running Individual...")
                    models_indiv = {}
                    b_indiv = max(1, B // T)
                    
                    for v in nodes:
                        model = create_model()
                        model = train_model_sgd(
                            model, X_train_n[v], y_train_n[v],
                            steps=b_indiv, eta=eta, device=device
                        )
                        models_indiv[v] = model
                    
                    mses_indiv = evaluate_models(models_indiv)
                    indiv_rmses.append(np.mean([np.sqrt(mse) for mse in mses_indiv.values()]))
                    
                    # =====================================================
                    # 2. STAR TOPOLOGY
                    # =====================================================
                    print("  Running Star...")
                    edges_star = [(global_root, v) for v in nodes if v != global_root]
                    models_star, _ = cascade_train_tree(
                        tree_edges=edges_star,
                        root=global_root,
                        X_train_n=X_train_n,
                        y_train_n=y_train_n,
                        B=B,
                        b_seed=b_seed,
                        eta=eta,
                        device=device,
                        model_class=model_class,
                        initial_model=source_model,
                        **{k: v for k, v in kwargs.items() if k in ["hidden", "eps"]}
                    )
                    
                    mses_star = evaluate_models(models_star)
                    star_rmses.append(np.mean([np.sqrt(mse) for mse in mses_star.values()]))
                    
                    # =====================================================
                    # 3. RANDOM TREE
                    # =====================================================
                    print("  Running Random Tree...")
                    rng = np.random.default_rng(seed + rep)
                    edges_rand = []
                    node_list = list(range(T))
                    for i in range(1, T):
                        parent = rng.integers(0, i)
                        edges_rand.append((node_list[parent], node_list[i]))
                    
                    adj = defaultdict(list)
                    for u, v in edges_rand:
                        adj[u].append(v)
                        adj[v].append(u)
                    
                    parent_map = {global_root: None}
                    edges_rand_directed = []
                    q = deque([global_root])
                    visited = {global_root}
                    
                    adj_keys = defaultdict(list)
                    for u, v in edges_rand:
                        adj_keys[nodes[u]].append(nodes[v])
                        adj_keys[nodes[v]].append(nodes[u])
                    
                    while q:
                        u = q.popleft()
                        for v in adj_keys[u]:
                            if v not in visited:
                                visited.add(v)
                                parent_map[v] = u
                                edges_rand_directed.append((u, v))
                                q.append(v)
                    
                    models_rand, _ = cascade_train_tree(
                        tree_edges=edges_rand_directed,
                        root=global_root,
                        X_train_n=X_train_n,
                        y_train_n=y_train_n,
                        B=B,
                        b_seed=b_seed,
                        eta=eta,
                        device=device,
                        model_class=model_class,
                        initial_model=source_model,
                        **{k: v for k, v in kwargs.items() if k in ["hidden", "eps"]}
                    )
                    
                    mses_rand = evaluate_models(models_rand)
                    random_rmses.append(np.mean([np.sqrt(mse) for mse in mses_rand.values()]))
                    
                    # =====================================================
                    # 4. MST VARIANTS (ALL DISTANCE TYPES)
                    # =====================================================
                    print("  Running MST variants...")
                    y_means = {v: y_train_n[v].flatten() for v in nodes}
                    
                    for dist_type in mst_distance_types:
                        try:
                            _, mst_parent, mst_edges, _ = build_mst_tree_with_distance(
                                V=nodes,
                                y=y_means,
                                X=X_train_n if dist_type in ["gradient", "model"] else None,
                                distance_type=dist_type,
                                seed=global_root,
                            )
                            
                            models_mst, _ = cascade_train_tree(
                                tree_edges=mst_edges,
                                root=global_root,
                                X_train_n=X_train_n,
                                y_train_n=y_train_n,
                                B=B,
                                b_seed=b_seed,
                                eta=eta,
                                device=device,
                                model_class=model_class,
                                initial_model=source_model,
                                **{k: v for k, v in kwargs.items() if k in ["hidden", "eps"]}
                            )
                            
                            mses_mst = evaluate_models(models_mst)
                            mst_rmses[dist_type].append(np.mean([np.sqrt(mse) for mse in mses_mst.values()]))
                        except Exception as e:
                            print(f"    Warning: MST-{dist_type} failed: {e}")
                            mst_rmses[dist_type].append(np.nan)
                                                        
                    # Store detailed results
                    rep_results = {
                        "global_mtl_mse": mses_global if mses_global else {},
                        "indiv_mse": mses_indiv,
                        "star_mse": mses_star,
                        "random_mse": mses_rand,
                    }
                    for dt in mst_distance_types:
                        if mst_rmses[dt] and not np.isnan(mst_rmses[dt][-1]):
                            rep_results[f"mst_{dt}_mse"] = mses_mst
                    
                    full_results_all.append(rep_results)
                    
                    print(" done")
                
                # =====================================================
                # AGGREGATE RESULTS
                # =====================================================
                result = {
                    "B": B,
                    "eta": eta,
                    "b_proto_frac": b_proto_frac,
                    "b_seed": b_seed,
                    "source_mode": source_mode,
                    
                    # Global MTL
                    "global_mtl_rmse_mean": np.nanmean(global_mtl_rmses),
                    "global_mtl_rmse_se": np.nanstd(global_mtl_rmses) / np.sqrt(len([r for r in global_mtl_rmses if not np.isnan(r)])) if any(not np.isnan(r) for r in global_mtl_rmses) else np.nan,
                    
                    # Individual
                    "indiv_rmse_mean": np.mean(indiv_rmses),
                    "indiv_rmse_se": np.std(indiv_rmses) / np.sqrt(n_repeats),
                    
                    # Star
                    "star_rmse_mean": np.mean(star_rmses),
                    "star_rmse_se": np.std(star_rmses) / np.sqrt(n_repeats),
                    
                    # Random
                    "random_rmse_mean": np.mean(random_rmses),
                    "random_rmse_se": np.std(random_rmses) / np.sqrt(n_repeats),
                    
                    # Full results
                    "full_results_all": full_results_all,
                }
                
                # Add MST variants
                for dist_type in mst_distance_types:
                    valid_rmses = [r for r in mst_rmses[dist_type] if not np.isnan(r)]
                    if valid_rmses:
                        result[f"mst_{dist_type}_rmse_mean"] = np.mean(valid_rmses)
                        result[f"mst_{dist_type}_rmse_se"] = np.std(valid_rmses) / np.sqrt(len(valid_rmses))
                    else:
                        result[f"mst_{dist_type}_rmse_mean"] = np.nan
                        result[f"mst_{dist_type}_rmse_se"] = np.nan
                
                all_results.append(result)
                
                # Print summary
                print(f"\n  Global MTL: {result['global_mtl_rmse_mean']:.4f} ± {result['global_mtl_rmse_se']:.4f}")
                print(f"  Indiv: {result['indiv_rmse_mean']:.4f} ± {result['indiv_rmse_se']:.4f} | "
                      f"Star: {result['star_rmse_mean']:.4f} ± {result['star_rmse_se']:.4f} | "
                      f"Random: {result['random_rmse_mean']:.4f} ± {result['random_rmse_se']:.4f}")
                
                for dist_type in mst_distance_types:
                    mean_key = f"mst_{dist_type}_rmse_mean"
                    se_key = f"mst_{dist_type}_rmse_se"
                    if not np.isnan(result[mean_key]):
                        print(f"  MST-{dist_type}: {result[mean_key]:.4f} ± {result[se_key]:.4f}")
    
    return all_results

def run_image_classification_experiment(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    Ts: Tuple[int, ...] = (50, 100),
    Bs: Tuple[int, ...] = (50, 100, 200),
    n_train_per_task: int = 100,
    n_test_per_task: int = 200,
    lr: float = 0.1,
    reg: float = 1e-4,
    reps: int = 5,
    clustered: bool = False,
    n_clusters: int = 4,
    task_type: str = "random_pairs",
    seed_fraction: float = 0.1,
    device: Optional[torch.device] = None,
    SEED: int = 42,
    mst_distance_types: Tuple[str, ...] = ("feature", "target", "gradient", "model"),
    **kwargs
):
    """
    Run CTL experiment on image classification tasks with MST variants.
    
    Unified interface matching run_experiment_grid for regression.
    """
    from collections import defaultdict

    device = device or torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    # Build method list
    methods = ["Indiv", "Star", "CTL-RandTree"]
    for dist_type in mst_distance_types:
        methods.append(f"CTL-MST-{dist_type}")

    # Store results
    results = {
        m: {(T, B): [] for T in Ts for B in Bs}
        for m in methods
    }
    
    results_per_task = {
        m: {(T, B): [] for T in Ts for B in Bs}
        for m in methods
    }

    base_rng = np.random.default_rng(SEED if not clustered else SEED + 1000)
    d = X_train.shape[1]

    print(f"\n{'='*60}")
    print(f"Image Classification Experiment")
    print(f"{'='*60}")
    print(f"Tasks (T): {Ts}")
    print(f"Budgets (B): {Bs}")
    print(f"Samples per task: train={n_train_per_task}, test={n_test_per_task}")
    print(f"Learning rate: {lr}")
    print(f"Regularization: {reg}")
    print(f"Repeats: {reps}")
    print(f"Seed fraction: {seed_fraction}")
    print(f"Clustered: {clustered}")
    print(f"Task type: {task_type}")
    print(f"MST variants: {mst_distance_types}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    for T in Ts:
        for B in Bs:
            print(f"\n{'='*60}")
            print(f"Configuration: T={T}, B={B}")
            print(f"{'='*60}")
            
            # Compute budget allocation
            indiv_steps = max(1, B // T)
            seed_steps = max(1, int(seed_fraction * B))
            n_children = max(0, T - 1)
            child_steps = max(1, (B - seed_steps) // n_children) if n_children > 0 else 0

            for rep in range(reps):
                print(f"\nRepeat {rep+1}/{reps}")
                
                rng = np.random.default_rng(base_rng.integers(1, 1_000_000))

                # Generate tasks
                if clustered:
                    X_tr, y_tr, X_te, y_te, class_pairs, cluster_ids = (
                        generate_clustered_binary_tasks(
                            X_train, y_train, X_test, y_test,
                            n_tasks=T, n_clusters=n_clusters,
                            n_train_per_task=n_train_per_task,
                            n_test_per_task=n_test_per_task,
                            rng=rng,
                        )
                    )
                else:
                    X_tr, y_tr, X_te, y_te, class_pairs = (
                        generate_binary_tasks(
                            X_train, y_train, X_test, y_test,
                            n_tasks=T,
                            n_train_per_task=n_train_per_task,
                            n_test_per_task=n_test_per_task,
                            task_type=task_type,
                            rng=rng,
                        )
                    )

                # Convert to dictionaries with INTEGER keys
                X_train_dict = {int(v): X_tr[v] for v in range(T)}
                y_train_dict = {int(v): y_tr[v] for v in range(T)}
                nodes = list(range(T))

                # Select seed node (medoid by feature distance)
                means = X_tr.mean(axis=1)
                D_feat = np.linalg.norm(
                    means[:, None, :] - means[None, :, :], axis=-1
                )
                seed = int(np.argmin(D_feat.sum(axis=1)))

                # Evaluation helper
                def evaluate_theta(theta_dict):
                    """Evaluate logistic models and return accuracies."""
                    accs = {}
                    for v in range(T):
                        acc = compute_accuracy(theta_dict[v], X_te[v], y_te[v])
                        accs[v] = acc
                    return accs

                # =====================================================
                # 1. INDIVIDUAL BASELINE
                # =====================================================
                print("  Running Individual...")
                theta_indiv = {}
                for v in range(T):
                    theta_indiv[v] = gd_train_logistic(
                        np.zeros(d),
                        X_tr[v], y_tr[v],
                        steps=indiv_steps,
                        lr=lr,
                        reg=reg,
                        device=device,
                    )

                accs_indiv = evaluate_theta(theta_indiv)
                acc_indiv_mean = np.mean(list(accs_indiv.values()))
                results["Indiv"][(T, B)].append(acc_indiv_mean)
                results_per_task["Indiv"][(T, B)].append(np.array(list(accs_indiv.values())))

                # =====================================================
                # 2. STAR TOPOLOGY
                # =====================================================
                print("  Running Star...")
                edges_star = [(seed, v) for v in range(T) if v != seed]
                
                theta_star, _ = cascade_train_tree(
                    tree_edges=edges_star,
                    root=seed,
                    X_train_n=X_train_dict,
                    y_train_n=y_train_dict,
                    B=B,
                    b_seed=seed_steps,
                    eta=lr,
                    device=device,
                    model_class="logistic",
                    task="classification",
                    reg=reg,
                )

                accs_star = evaluate_theta(theta_star)
                acc_star_mean = np.mean(list(accs_star.values()))
                results["Star"][(T, B)].append(acc_star_mean)
                results_per_task["Star"][(T, B)].append(np.array(list(accs_star.values())))

                # =====================================================
                # 3. RANDOM TREE
                # =====================================================
                print("  Running Random Tree...")
                edges_rand = random_tree_recursive(T, rng=rng)
                _, edges_rand_dir, _ = root_tree(edges_rand, seed)

                theta_rand, _ = cascade_train_tree(
                    tree_edges=edges_rand_dir,
                    root=seed,
                    X_train_n=X_train_dict,
                    y_train_n=y_train_dict,
                    B=B,
                    b_seed=seed_steps,
                    eta=lr,
                    device=device,
                    model_class="logistic",
                    task="classification",
                    reg=reg,
                )

                accs_rand = evaluate_theta(theta_rand)
                acc_rand_mean = np.mean(list(accs_rand.values()))
                results["CTL-RandTree"][(T, B)].append(acc_rand_mean)
                results_per_task["CTL-RandTree"][(T, B)].append(np.array(list(accs_rand.values())))

                # =====================================================
                # 4. MST VARIANTS (ALL DISTANCE TYPES)
                # =====================================================
                print("  Running MST variants...")
                y_dict_for_mst = {int(v): y_train_dict[v].flatten() for v in nodes}
                mst_accs = {}
                
                for dist_type in mst_distance_types:
                    method_name = f"CTL-MST-{dist_type}"
                    
                    try:
                        # Build MST with specified distance type
                        root_mst, parent_mst, edges_mst, builder_mst = build_mst_tree_with_distance(
                            V=nodes,
                            y=y_dict_for_mst,
                            X=X_train_dict if dist_type in ["gradient", "model", "feature", "mmd_feature", "mean_cov_feature", "cka"] else None,
                            distance_type=dist_type,
                            seed=seed,
                            regularization=reg,
                        )
                        
                        # Train with MST structure
                        theta_mst, _ = cascade_train_tree(
                            tree_edges=edges_mst,
                            root=root_mst,
                            X_train_n=X_train_dict,
                            y_train_n=y_train_dict,
                            B=B,
                            b_seed=seed_steps,
                            eta=lr,
                            device=device,
                            model_class="logistic",
                            task="classification",
                            reg=reg,
                        )
                        
                        accs_mst = evaluate_theta(theta_mst)
                        acc_mst_mean = np.mean(list(accs_mst.values()))
                        
                        mst_accs[dist_type] = acc_mst_mean
                        results[method_name][(T, B)].append(acc_mst_mean)
                        results_per_task[method_name][(T, B)].append(np.array(list(accs_mst.values())))
                        
                    except Exception as e:
                        print(f"    Warning: MST-{dist_type} failed: {e}")
                        mst_accs[dist_type] = np.nan
                        results[method_name][(T, B)].append(np.nan)
                        results_per_task[method_name][(T, B)].append(np.full(T, np.nan))

                # Print progress
                mst_str = ", ".join([
                    f"{dt[:3]}={mst_accs.get(dt, np.nan):.3f}"
                    for dt in mst_distance_types
                ])
                
                print(
                    f"  Indiv={acc_indiv_mean:.3f}, "
                    f"Star={acc_star_mean:.3f}, "
                    f"Rand={acc_rand_mean:.3f}, "
                    f"MST[{mst_str}]"
                )

    return methods, results, results_per_task