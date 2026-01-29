import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from typing import Dict, List, Optional, Tuple

def plot_results_comparison(
    results_list: List[Dict], 
    metric: str = "mse"
) -> plt.Figure:
    """Plot boxplot comparison of methods."""
    methods = ["indiv", "star", "random", "mst", "align"]
    colors = {
        "indiv": "gray",
        "star": "deepskyblue",
        "random": "violet",
        "mst": "orange",
        "align": "lightgreen",
    }
    
    Bs = sorted(set(r["B"] for r in results_list))
    
    fig, axes = plt.subplots(1, len(Bs), figsize=(5 * len(Bs), 5), sharey=True)
    if len(Bs) == 1:
        axes = [axes]
    
    for ax, B in zip(axes, Bs):
        B_results = [r for r in results_list if r["B"] == B]
        
        data = []
        labels = []
        for method in methods:
            vals = [r[f"{method}_{metric}"] for r in B_results 
                    if np.isfinite(r[f"{method}_{metric}"])]
            if vals:
                data.append(vals)
                labels.append(method)
        
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        for patch, label in zip(bp['boxes'], labels):
            patch.set_facecolor(colors.get(label, "gray"))
            patch.set_alpha(0.7)
        
        ax.set_title(f"B = {B}")
        ax.set_ylabel(f"Mean {metric.upper()}" if ax == axes[0] else "")
        ax.tick_params(axis='x', rotation=30)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f"WEAVE - {metric.upper()} Comparison", fontsize=14)
    plt.tight_layout()
    return fig

def plot_results_comparison_grid(
    results_list: List[Dict], 
    metric: str = "rmse"
) -> plt.Figure:
    """Plot boxplot comparison of methods for each (B, (k, eta, b_proto_frac)) combination using all repetitions."""
    methods = ["indiv", "star", "random", "mst", "align"]
    map_method = {
        "indiv": "Indiv",
        "star": "Star",
        "random": "CTL-RandTree",
        "mst": "CTL-MST",
        "align": "CTL-AlignTree",
    }
    colors = {
        "Indiv": "gray",
        "Star": "deepskyblue",
        "CTL-RandTree": "violet",
        "CTL-MST": "orange",
        "CTL-AlignTree": "lightgreen",
    }
    
    Bs = sorted(set(r["B"] for r in results_list))
    # unique combos of (k, eta, b_proto_frac)
    combos = sorted({(r["k_neighbors"], r.get("eta", np.nan), r.get("b_proto_frac", np.nan)) for r in results_list})
    
    n_rows = len(Bs)
    n_cols = len(combos)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), sharey='row')
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]
    
    for i, B in enumerate(Bs):
        for j, (k, eta, b_frac) in enumerate(combos):
            ax = axes[i, j]
            subset = [r for r in results_list if r["B"] == B and r["k_neighbors"] == k 
                      and np.isclose(r.get("eta", np.nan), eta, equal_nan=True)
                      and np.isclose(r.get("b_proto_frac", np.nan), b_frac, equal_nan=True)]
            data = []
            labels = []
            for method in methods:
                vals = []
                for r in subset:
                    arr = r.get(f"{method}_{metric}_all", [])
                    if isinstance(arr, (list, np.ndarray)):
                        vals.extend(list(arr))
                    else:
                        vals.append(arr)
                vals = [v for v in vals if np.isfinite(v)]
                if vals:
                    data.append(vals)
                    labels.append(map_method[method])
            if data:
                bp = ax.boxplot(data, labels=labels, patch_artist=True)
                for patch, label in zip(bp['boxes'], labels):
                    patch.set_facecolor(colors.get(label, "gray"))
                    patch.set_alpha(0.7)
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
            title = f"k={k}\nη={eta}\nb={b_frac}"
            ax.set_title(title, fontsize=9)
            if j == 0:
                ax.set_ylabel(metric.upper())
            ax.tick_params(axis='x', rotation=30)
            ax.grid(True, alpha=0.3, axis='y')
    
    # build nice column labels (top)
    col_labels = [f"k={k}, η={eta}, b={b_frac}" for (k, eta, b_frac) in combos]
    plt.suptitle(f"WEAVE - {metric.upper()} Comparison by B and (k, η, b_frac)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

def plot_ctl_comparison(
    results: Dict[str, Dict[Tuple[int, int], List[float]]],
    methods: List[str],
    Ts: Tuple[int, ...],
    Bs: Tuple[int, ...],
    colors: Dict[str, str],
    title: str = "CTL Methods vs Baselines",
    figsize: Tuple[int, int] = (14, 5),
    legend_ncol: int = 2,
):
    """
    Plot comparison of CTL methods against baselines across T and B.
    Handles all MST variants with proper color mapping.
    """
    fig, axes = plt.subplots(1, len(Ts), figsize=figsize, sharey=True)
    
    if len(Ts) == 1:
        axes = [axes]
    
    for ax_idx, T in enumerate(Ts):
        ax = axes[ax_idx]
        
        for method in methods:
            means = []
            stds = []
            
            for B in Bs:
                key = (T, B)
                if key in results[method] and len(results[method][key]) > 0:
                    vals = np.array(results[method][key])
                    # Filter NaN values
                    vals = vals[~np.isnan(vals)]
                    if len(vals) > 0:
                        means.append(vals.mean())
                        stds.append(vals.std())
                    else:
                        means.append(np.nan)
                        stds.append(np.nan)
                else:
                    means.append(np.nan)
                    stds.append(np.nan)
            
            means = np.array(means)
            stds = np.array(stds)
            
            # Skip if all NaN
            if np.all(np.isnan(means)):
                continue
            
            # Shorten label for MST methods
            label = method.replace('CTL-MST-', 'MST-')
            
            ax.errorbar(
                Bs, means, yerr=stds,
                marker='o', capsize=4, label=label,
                color=colors.get(method, None),
                linewidth=2, markersize=6,
                alpha=0.8,
            )
        
        ax.set_xlabel("Budget B", fontsize=11)
        if ax_idx == 0:
            ax.set_ylabel("Mean Test MSE", fontsize=11)
        ax.set_title(f"T = {T}", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=legend_ncol, loc='best')
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig, axes

def plot_significance_matrix(
    results: Dict[str, Dict[Tuple[int, int], List[float]]],
    methods: List[str],
    Ts: Tuple[int, ...],
    Bs: Tuple[int, ...],
    tau_focus: float,
    baseline: str = "Indiv",
    alpha: float = 0.05,
    figsize: Tuple[int, int] = (14, 10),
):
    """
    Plot a heatmap showing statistical significance of method comparisons.
    
    Args:
        results: Dict[method -> Dict[(T, B) -> List[mse_values]]]
        methods: List of method names
        Ts: Tuple of T values
        Bs: Tuple of B values
        tau_focus: tau value being visualized
        baseline: Baseline method for comparison
        alpha: Significance level
        figsize: Figure size
    
    Returns:
        fig, ax: matplotlib figure and axis objects
    """
    # Compute statistical tests
    test_results = compute_pairwise_statistical_tests(results, baseline=baseline, alpha=alpha)
    
    # Methods to compare (exclude baseline)
    methods_to_test = [m for m in methods if m != baseline]
    
    # Create matrix for heatmap
    n_conditions = len(Ts) * len(Bs)
    n_methods = len(methods_to_test)
    
    sig_matrix = np.zeros((n_methods, n_conditions))
    effect_matrix = np.zeros((n_methods, n_conditions))
    
    condition_labels = []
    
    cond_idx = 0
    for T in Ts:
        for B in Bs:
            key = (T, B)
            condition_labels.append(f"T={T}\nB={B}")
            
            if key not in test_results:
                cond_idx += 1
                continue
            
            for m_idx, method in enumerate(methods_to_test):
                if method in test_results[key]:
                    result = test_results[key][method]
                    sig_matrix[m_idx, cond_idx] = 1 if result['significant'] else 0
                    effect_matrix[m_idx, cond_idx] = result['cohens_d']
                else:
                    sig_matrix[m_idx, cond_idx] = np.nan
                    effect_matrix[m_idx, cond_idx] = np.nan
            
            cond_idx += 1
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot effect sizes with significance overlay
    mask = np.isnan(effect_matrix)
    im = ax.imshow(effect_matrix, cmap='RdBu_r', aspect='auto', vmin=-2, vmax=2)
    
    # Add significance markers
    for i in range(n_methods):
        for j in range(n_conditions):
            if not mask[i, j]:
                if sig_matrix[i, j] == 1:
                    ax.text(j, i, '***', ha='center', va='center', 
                           color='black', fontsize=16, fontweight='bold')
    
    # Set labels
    ax.set_xticks(range(n_conditions))
    ax.set_xticklabels(condition_labels, rotation=0, ha='center')
    ax.set_yticks(range(n_methods))
    ax.set_yticklabels(methods_to_test)
    
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Method')
    ax.set_title(f'Statistical Significance vs {baseline} (τ={tau_focus})\n'
                 f'Effect Size (Cohen\'s d) with significance markers (***)')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Cohen\'s d (positive = method better than baseline)')
    
    plt.tight_layout()
    
    return fig, ax

def results_to_dataframe(results_list: List[Dict]) -> pd.DataFrame:
    """Convert results to DataFrame."""
    rows = []
    for r in results_list:
        row = {
            "B": r["B"],
            "k": r["k_neighbors"],
        }
        for method in ["indiv", "star", "random", "mst", "align"]:
            if f"{method}_mse" in r:
                row[f"{method}_mse"] = r[f"{method}_mse"]
            elif f"{method}_rmse" in r:
                row[f"{method}_rmse"] = r[f"{method}_rmse"]
            
            if f"{method}_mape" in r:
                row[f"{method}_mape"] = r[f"{method}_mape"]
        
        if "seed" in r:
            row["seed"] = r["seed"]
        rows.append(row)
    return pd.DataFrame(rows)

def plot_results_comparison_se(results_list, metric="rmse"):
    methods = ["indiv", "star", "random", "mst", "align"]
    map_method = {
        "indiv": "Indiv",
        "star": "Star",
        "random": "CTL-RandTree",
        "mst": "CTL-MST",
        "align": "CTL-AlignTree",
    }
    colors = {
        "Indiv": "gray",
        "Star": "deepskyblue",
        "CTL-RandTree": "violet",
        "CTL-MST": "orange",
        "CTL-AlignTree": "lightgreen",
    }
    Bs = sorted(set(r["B"] for r in results_list))
    fig, axes = plt.subplots(1, len(Bs), figsize=(5 * len(Bs), 5), sharey=True)
    if len(Bs) == 1:
        axes = [axes]
    for ax, B in zip(axes, Bs):
        B_results = [r for r in results_list if r["B"] == B]
        if len(B_results) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            continue
        labels = [map_method[method] for method in methods]
        data = []
        errors = []
        for method in methods:
            # collect means & ses across all (k, eta, b_proto_frac) for this B
            means = [r.get(f"{method}_{metric}_mean", np.nan) for r in B_results]
            ses = [r.get(f"{method}_{metric}_se", np.nan) for r in B_results]
            mean = np.nanmean(means) if len(means) > 0 else np.nan
            se = np.nanmean([s for s in ses if not np.isnan(s)]) if any(~np.isnan(ses)) else np.nan
            data.append(mean)
            errors.append(se)
        positions = np.arange(1, len(methods) + 1)
        # Represent means as points with error bars
        bp = ax.boxplot([[v] if not np.isnan(v) else [] for v in data], labels=labels, patch_artist=True)
        for patch, label in zip(bp['boxes'], labels):
            patch.set_facecolor(colors.get(label, "gray"))
            patch.set_alpha(0.7)
        # Add error bars (one per method)
        ax.errorbar(positions, data, yerr=errors, fmt='o', color='black', alpha=0.8, capsize=5)
        ax.set_title(f"B = {B}")
        ax.set_ylabel(f"Mean {metric.upper()}" if ax == axes[0] else "")
        ax.tick_params(axis='x', rotation=30)
        ax.grid(True, alpha=0.3, axis='y')
    plt.suptitle(f"WEAVE - {metric.upper()} Comparison (Mean ± SE aggregated over k, eta, b_proto_frac)", fontsize=14)
    plt.tight_layout()
    return fig

def compute_ranks_per_node(results: Dict, metric: str = "mse") -> pd.DataFrame:
    """
    Compute ranks of methods for each node for a single repetition.
    Args:
        results: Single experiment result dict (from one repetition)
        metric: "mse" or "mape" (lower is better)
    Returns:
        DataFrame with columns [node, method, value, rank]
    """
    methods = ["indiv", "star", "random", "mst", "align"]
    metric_key = "mses" if metric == "mse" else "mapes"
    nodes = list(results["indiv"][metric_key].keys())
    rows = []
    for node in nodes:
        node_values = {method: results[method][metric_key].get(node, np.nan) for method in methods}
        valid_methods = [(m, v) for m, v in node_values.items() if np.isfinite(v)]
        if len(valid_methods) == 0:
            continue
        sorted_methods = sorted(valid_methods, key=lambda x: x[1])
        ranks = {m: rank + 1 for rank, (m, _) in enumerate(sorted_methods)}
        for method in methods:
            rows.append({
                "node": node,
                "method": method,
                "value": node_values[method],
                "rank": ranks.get(method, np.nan),
            })
    return pd.DataFrame(rows)

def compute_mean_rank(results: Dict, metric: str = "mse") -> pd.DataFrame:
    """
    Compute mean rank of each method across all nodes for a single repetition.
    """
    df_ranks = compute_ranks_per_node(results, metric=metric)
    summary = df_ranks.groupby("method").agg({
        "rank": ["mean", "std"],
        "value": "mean",
    }).reset_index()
    summary.columns = ["method", "mean_rank", "std_rank", "mean_value"]
    wins = df_ranks[df_ranks["rank"] == 1].groupby("method").size()
    summary["wins"] = summary["method"].map(wins).fillna(0).astype(int)
    n_nodes = df_ranks["node"].nunique()
    summary["win_pct"] = (summary["wins"] / n_nodes * 100).round(1)
    summary = summary.sort_values("mean_rank").reset_index(drop=True)
    return summary

def compute_mean_rank_grid(
    results_list: List[Dict],
    metric: str = "mse",
    rep: int = 0,
) -> pd.DataFrame:
    """
    Compute mean rank across all configurations in grid for a given repetition.
    Args:
        results_list: List of result dicts from run_experiment_grid
        metric: "mse" or "mape"
        rep: which repetition to use (default: 0)
    Returns:
        DataFrame with mean ranks per (B, k) configuration
    """
    rows = []
    for r in results_list:
        B = r["B"]
        k = r["k_neighbors"]
        full_results = r["full_results_all"][rep]
        rank_summary = compute_mean_rank(full_results, metric=metric)
        row = {"B": B, "k": k}
        for _, method_row in rank_summary.iterrows():
            method = method_row["method"]
            row[f"{method}_mean_rank"] = method_row["mean_rank"]
            row[f"{method}_wins"] = method_row["wins"]
            row[f"{method}_win_pct"] = method_row["win_pct"]
        rows.append(row)
    return pd.DataFrame(rows)

def compute_mean_rank_across_reps(results_list, metric="mse"):
    """
    Compute mean rank and win rate across all repetitions for each (B, k) configuration.
    Returns a DataFrame with mean/std of ranks and win rates.
    """
    methods = ["indiv", "star", "random", "mst", "align"]
    rows = []
    for r in results_list:
        B = r["B"]
        k = r["k_neighbors"]
        eta = r["eta"]
        b_proto_frac = r["b_proto_frac"]
        n_reps = len(r["full_results_all"])
        # Collect per-rep summaries
        rep_summaries = []
        for rep in range(n_reps):
            full_results = r["full_results_all"][rep]
            summary = compute_mean_rank(full_results, metric=metric)
            rep_summaries.append(summary.set_index("method"))
        # Aggregate across reps
        agg = {}
        for method in methods:
            ranks = [s.loc[method, "mean_rank"] for s in rep_summaries]
            wins = [s.loc[method, "wins"] for s in rep_summaries]
            win_pct = [s.loc[method, "win_pct"] for s in rep_summaries]
            agg[f"{method}_mean_rank"] = np.mean(ranks)
            agg[f"{method}_std_rank"] = np.std(ranks, ddof=1)
            agg[f"{method}_mean_wins"] = np.mean(wins)
            agg[f"{method}_mean_win_pct"] = np.mean(win_pct)
        row = {"B": B, "k": k, "eta": eta, "b_proto_frac": b_proto_frac}
        row.update(agg)
        rows.append(row)
    return pd.DataFrame(rows)

def print_grid_rank_summary_avg(results_list, metric="mse"):
    df_ranks = compute_mean_rank_across_reps(results_list, metric=metric)
    methods = ["indiv", "star", "random", "mst", "align"]
    print(f"\nMean Rank Summary ({metric.upper()}) - Averaged over repetitions")
    print("=" * 80)
    header = f"{'B':>6} {'k':>4} {'eta':>6} {'b_frac':>8}"
    for m in methods:
        header += f" {m:>10}"
    print(header)
    print("-" * 80)
    for _, row in df_ranks.iterrows():
        line = f"{int(row['B']):>6} {int(row['k']):>4} {row['eta']:>6.3f} {row['b_proto_frac']:>8.2f}"
        ranks = {m: row[f"{m}_mean_rank"] for m in methods}
        best_method = min(ranks, key=ranks.get)
        for m in methods:
            rank_val = row[f"{m}_mean_rank"]
            if m == best_method:
                line += f" {rank_val:>9.2f}*"
            else:
                line += f" {rank_val:>10.2f}"
        print(line)
    print("=" * 80)
    print("\nOverall Mean Rank:")
    for m in methods:
        col = f"{m}_mean_rank"
        overall = df_ranks[col].mean()
        print(f"  {m:<10}: {overall:.2f}")

def plot_rank_comparison_avg(
    results_list,
    metric="mse",
    figsize=(10, 6),
):
    methods = ["indiv", "star", "random", "mst", "align"]
    colors = {
        "indiv": "gray",
        "star": "deepskyblue",
        "random": "violet",
        "mst": "orange",
        "align": "lightgreen",
    }
    df_ranks = compute_mean_rank_across_reps(results_list, metric=metric)
    overall_ranks = {m: df_ranks[f"{m}_mean_rank"].mean() for m in methods}
    overall_stds = {m: df_ranks[f"{m}_mean_rank"].std() for m in methods}
    sorted_methods = sorted(methods, key=lambda m: overall_ranks[m])
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(sorted_methods))
    bars = ax.bar(
        x,
        [overall_ranks[m] for m in sorted_methods],
        yerr=[overall_stds[m] for m in sorted_methods],
        color=[colors[m] for m in sorted_methods],
        capsize=5,
        alpha=0.8,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_methods)
    ax.set_ylabel("Mean Rank (lower is better)")
    ax.set_title(f"Method Comparison - Mean Rank ({metric.upper()})")
    ax.axhline(y=3, color='gray', linestyle='--', alpha=0.5, label='Neutral rank')
    for bar, m in zip(bars, sorted_methods):
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    ax.set_ylim(0, 5.5)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    return fig

def print_rank_summary(results: Dict, metric: str = "mse", title: str = ""):
    summary = compute_mean_rank(results, metric=metric)
    if title:
        print(f"\n{title}")
    print("=" * 70)
    print(f"{'Method':<10} {'Mean Rank':>10} {'Std Rank':>10} {'Mean Value':>12} {'Wins':>6} {'Win %':>8}")
    print("-" * 70)
    for _, row in summary.iterrows():
        print(f"{row['method']:<10} {row['mean_rank']:>10.2f} {row['std_rank']:>10.2f} "
              f"{row['mean_value']:>12.4f} {row['wins']:>6d} {row['win_pct']:>7.1f}%")
    print("=" * 70)

def print_grid_rank_summary(results_list: List[Dict], metric: str = "mse", rep: int = 0):
    df_ranks = compute_mean_rank_grid(results_list, metric=metric, rep=rep)
    methods = ["indiv", "star", "random", "mst", "align"]
    print(f"\nMean Rank Summary ({metric.upper()})")
    print("=" * 80)
    header = f"{'B':>6} {'k':>4}"
    for m in methods:
        header += f" {m:>10}"
    print(header)
    print("-" * 80)
    for _, row in df_ranks.iterrows():
        line = f"{int(row['B']):>6} {int(row['k']):>4}"
        ranks = {m: row[f"{m}_mean_rank"] for m in methods}
        best_method = min(ranks, key=ranks.get)
        for m in methods:
            rank_val = row[f"{m}_mean_rank"]
            if m == best_method:
                line += f" {rank_val:>9.2f}*"
            else:
                line += f" {rank_val:>10.2f}"
        print(line)
    print("=" * 80)
    print("\nOverall Mean Rank:")
    for m in methods:
        col = f"{m}_mean_rank"
        overall = df_ranks[col].mean()
        print(f"  {m:<10}: {overall:.2f}")

def plot_rank_comparison(
    results_list: List[Dict],
    metric: str = "mse",
    rep: int = 0,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    methods = ["indiv", "star", "random", "mst", "align"]
    colors = {
        "indiv": "gray",
        "star": "deepskyblue",
        "random": "violet",
        "mst": "orange",
        "align": "lightgreen",
    }
    df_ranks = compute_mean_rank_grid(results_list, metric=metric, rep=rep)
    overall_ranks = {m: df_ranks[f"{m}_mean_rank"].mean() for m in methods}
    overall_stds = {m: df_ranks[f"{m}_mean_rank"].std() for m in methods}
    sorted_methods = sorted(methods, key=lambda m: overall_ranks[m])
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(sorted_methods))
    bars = ax.bar(
        x,
        [overall_ranks[m] for m in sorted_methods],
        yerr=[overall_stds[m] for m in sorted_methods],
        color=[colors[m] for m in sorted_methods],
        capsize=5,
        alpha=0.8,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_methods)
    ax.set_ylabel("Mean Rank (lower is better)")
    ax.set_title(f"Method Comparison - Mean Rank ({metric.upper()})")
    ax.axhline(y=3, color='gray', linestyle='--', alpha=0.5, label='Neutral rank')
    for bar, m in zip(bars, sorted_methods):
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    ax.set_ylim(0, 5.5)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    return fig

def plot_wins_comparison(
    results_list: List[Dict],
    metric: str = "mse",
    rep: int = 0,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    methods = ["indiv", "star", "random", "mst", "align"]
    colors = {
        "indiv": "gray",
        "star": "deepskyblue",
        "random": "violet",
        "mst": "orange",
        "align": "lightgreen",
    }
    df_ranks = compute_mean_rank_grid(results_list, metric=metric, rep=rep)
    total_wins = {m: df_ranks[f"{m}_wins"].sum() for m in methods}
    total_nodes = sum(total_wins.values())
    win_pct = {m: total_wins[m] / total_nodes * 100 if total_nodes > 0 else 0 for m in methods}
    sorted_methods = sorted(methods, key=lambda m: win_pct[m], reverse=True)
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(sorted_methods))
    bars = ax.bar(
        x,
        [win_pct[m] for m in sorted_methods],
        color=[colors[m] for m in sorted_methods],
        alpha=0.8,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_methods)
    ax.set_ylabel("Win Percentage (%)")
    ax.set_title(f"Method Comparison - Win Rate ({metric.upper()})")
    for bar, m in zip(bars, sorted_methods):
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    return fig

def plot_wins_comparison_avg(
    results_list: List[Dict],
    metric: str = "mse",
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot win percentage comparison as bar chart, averaged over all repetitions.
    """
    methods = ["indiv", "star", "random", "mst", "align"]
    colors = {
        "indiv": "gray",
        "star": "deepskyblue",
        "random": "violet",
        "mst": "orange",
        "align": "lightgreen",
    }
    df_ranks = compute_mean_rank_across_reps(results_list, metric=metric)
    # Average win percentage for each method across grid
    win_pct = {m: df_ranks[f"{m}_mean_win_pct"].mean() for m in methods}
    sorted_methods = sorted(methods, key=lambda m: win_pct[m], reverse=True)
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(sorted_methods))
    bars = ax.bar(
        x,
        [win_pct[m] for m in sorted_methods],
        color=[colors[m] for m in sorted_methods],
        alpha=0.8,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_methods)
    ax.set_ylabel("Win Percentage (%)")
    ax.set_title(f"Method Comparison - Win Rate ({metric.upper()})")
    # Add value labels on bars
    for bar, m in zip(bars, sorted_methods):
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    return fig

def compute_win_pct(results_list, metric="mse"):
    """
    Compute win percentage for each method per (B, k) configuration, averaged over repetitions.
    Returns a DataFrame with columns: B, k, method, win_pct
    """
    methods = ["indiv", "star", "random", "mst", "align"]
    rows = []
    for r in results_list:
        B = r["B"]
        eta = r["eta"]
        b_proto_frac = r["b_proto_frac"]
        k = r["k_neighbors"]
        n_reps = len(r["full_results_all"])
        # Collect per-rep summaries
        rep_summaries = []
        for rep in range(n_reps):
            full_results = r["full_results_all"][rep]
            summary = compute_mean_rank(full_results, metric=metric)
            rep_summaries.append(summary.set_index("method"))
        for method in methods:
            win_pct = [s.loc[method, "win_pct"] for s in rep_summaries]
            rows.append({
                "B": B,
                "k": k,
                "eta": eta,
                "b_proto_frac": b_proto_frac,
                "method": method,
                "win_pct_mean": np.mean(win_pct),
                "win_pct_std": np.std(win_pct, ddof=1),
            })
    return pd.DataFrame(rows)

def plot_wins_comparison(
    results_list: List[Dict],
    metric: str = "mse",
    figsize: Tuple[int, int] = (14, 6),
) -> plt.Figure:
    """
    Plot win percentage comparison as grouped bar chart per (B, k, eta, b_proto_frac) configuration.
    Aggregates win% across repetitions for each combo.
    """
    methods = ["indiv", "star", "random", "mst", "align"]
    colors = {
        "indiv": "gray",
        "star": "deepskyblue",
        "random": "violet",
        "mst": "orange",
        "align": "lightgreen",
    }

    df = compute_win_pct(results_list, metric=metric)
    if df.empty:
        raise ValueError("No data returned by compute_win_pct")

    # unique combos in deterministic order
    combo_cols = ["B", "k", "eta", "b_proto_frac"]
    combos_df = df[combo_cols].drop_duplicates().sort_values(by=["B", "k", "eta", "b_proto_frac"]).reset_index(drop=True)
    combos = [tuple(row) for row in combos_df.values]
    n_groups = len(combos)
    x = np.arange(n_groups)
    width = 0.15

    fig, ax = plt.subplots(figsize=figsize)

    # ensure ordering per combo for each method
    combo_index = pd.MultiIndex.from_tuples(combos, names=combo_cols)
    for i, method in enumerate(methods):
        method_df = df[df["method"] == method].set_index(combo_cols).reindex(combo_index)
        win_means = method_df["win_pct_mean"].values
        win_stds = method_df["win_pct_std"].values
        # replace nan means with zeros (or leave NaN to skip plotting) - keep NaN so missing bars don't show
        ax.bar(x + i * width, win_means, width=width, color=colors[method], label=method.capitalize(),
               yerr=win_stds, capsize=4, alpha=0.9)

    # X tick labels per combo
    def fmt_val(v):
        if pd.isna(v):
            return "NA"
        if isinstance(v, float):
            return f"{v:.3g}"
        return str(v)

    xtick_labels = [f"B={int(B)}, k={int(k)}\nη={fmt_val(eta)}, b={fmt_val(b)}" for (B, k, eta, b) in combos]
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(xtick_labels, rotation=45, ha='right')
    ax.set_ylabel("Win Percentage (%)")
    ax.set_title(f"Method Comparison - Win Rate per (B, k, η, b) ({metric.upper()})")
    ax.legend(ncol=1, bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    return fig

def weave_results_to_latex(
    results_list: list,
    metric: str = "rmse",
    dataset_name: str = "WEAVE",
    caption: str = "Test RMSE",
    label: str = "tab:weave_results",
    show_std: bool = True,
) -> Tuple[str, pd.DataFrame]:
    """
    Generate LaTeX table for WEAVE results (mean ± std error) per (B, eta, b_proto_frac).
    Dynamically detects available methods (baselines + MST variants) and excludes 'align'.
    """
    # Detect available methods from results_list
    methods_set = set()
    for r in results_list:
        for k in r.keys():
            if k.endswith(f"_{metric}_mean"):
                method = k[:-len(f"_{metric}_mean")]
                if "align" not in method.lower():
                    methods_set.add(method)
    
    # Order methods: baselines first, then MST variants sorted
    methods = []
    for base in ["indiv", "star", "random"]:
        if base in methods_set:
            methods.append(base)
            methods_set.discard(base)
    methods.extend(sorted(methods_set))  # remaining are MST variants
    
    if not methods:
        raise ValueError("No valid methods found in results_list")
    
    # Build display mapping
    map_method = {
        "indiv": "Indiv",
        "star": "Star",
        "random": "CTL-RandTree",
    }
    for m in methods:
        if m.startswith("mst_"):
            variant = m[len("mst_"):]
            map_method[m] = f"CTL-MST-{variant}"
        elif m not in map_method:
            map_method[m] = m

    # Collect unique combos (B, eta, b_proto_frac)
    combos = sorted({
        (r.get("B", np.nan), r.get("eta", np.nan), r.get("b_proto_frac", np.nan))
        for r in results_list
    })

    rows = []
    for (B, eta, b_frac) in combos:
        matches = [
            r for r in results_list
            if (r.get("B", np.nan) == B or (np.isnan(B) and np.isnan(r.get("B", np.nan))))
            and (np.isnan(eta) and np.isnan(r.get("eta", np.nan)) or 
                 (not np.isnan(eta) and not np.isnan(r.get("eta", np.nan)) and np.isclose(r.get("eta", np.nan), eta)))
            and (np.isnan(b_frac) and np.isnan(r.get("b_proto_frac", np.nan)) or 
                 (not np.isnan(b_frac) and not np.isnan(r.get("b_proto_frac", np.nan)) and np.isclose(r.get("b_proto_frac", np.nan), b_frac)))
        ]
        if len(matches) == 0:
            continue
        
        row = {"B": B, "eta": eta, "b_proto_frac": b_frac}
        for m in methods:
            means = [r.get(f"{m}_{metric}_mean", np.nan) for r in matches]
            ses = [r.get(f"{m}_{metric}_se", np.nan) for r in matches]
            mean_val = float(np.nanmean(means)) if any(~np.isnan(means)) else np.nan
            se_val = float(np.nanmean([s for s in ses if not np.isnan(s)])) if any(~np.isnan(ses)) else np.nan
            row[f"{m}_mean"] = mean_val
            row[f"{m}_se"] = se_val
        rows.append(row)

    df = pd.DataFrame(rows)

    n_methods = len(methods)
    col_spec = "c|" + "c" * n_methods

    latex_lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
    ]

    header1 = [r"$B$"] + [
        map_method.get(m, m).replace("_", r"\_") for m in methods
    ]
    latex_lines.append(" & ".join(header1) + r" \\")
    latex_lines.append(r"\midrule")
    
    prev_B = None
    for _, row in df.iterrows():
        if prev_B is not None and row["B"] != prev_B:
            latex_lines.append(r"\midrule")
        prev_B = row["B"]
        
        vals = [(m, row[f"{m}_mean"]) for m in methods if not np.isnan(row[f"{m}_mean"])]
        sorted_methods = sorted(vals, key=lambda x: x[1])  # lower is better
        best = sorted_methods[0][0] if len(sorted_methods) > 0 else None
        second = sorted_methods[1][0] if len(sorted_methods) > 1 else None

        cells = [
            str(int(row["B"])) if not np.isnan(row["B"]) else "NA",
        ]
        
        for m in methods:
            mean_val = row[f"{m}_mean"]
            se_val = row[f"{m}_se"]
            if np.isnan(mean_val):
                cells.append("—")
            else:
                if show_std and not np.isnan(se_val):
                    val_str = f"{mean_val:.3f} \\pm {se_val:.3f}"
                else:
                    val_str = f"{mean_val:.3f}"
                if m == best:
                    cells.append(f"$\\mathbf{{{val_str}}}$")
                elif m == second:
                    cells.append(f"$\\underline{{{val_str}}}$")
                else:
                    cells.append(f"${val_str}$")
        latex_lines.append(" & ".join(cells) + r" \\")
    
    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table}",
    ])
    return "\n".join(latex_lines), df

def visualize_weave_results_table(
    df: pd.DataFrame,
    methods: List[str],
    etas: Tuple[float, ...],
    b_proto_fracs: Tuple[float, ...],
    Bs: Tuple[int, ...],
    colors: Dict[str, str],
    figsize: Tuple[int, int] = (16, 10),
    show_std: bool = False,
    normalize_per_condition: bool = False,
    metric: str = "rmse",
):
    """
    Visualize WEAVE results as heatmaps showing mean metric across conditions.
    
    Args:
        df: DataFrame with columns: B, eta, b_proto_frac, {method}_{metric}_mean, {method}_{metric}_se
        methods: List of method names
        etas: Tuple of eta (learning rate) values
        b_proto_fracs: Tuple of seed budget fraction values
        Bs: Tuple of B (total budget) values
        colors: Dictionary mapping methods to colors
        figsize: Figure size
        show_std: If True, show standard error in annotations
        normalize_per_condition: If True, normalize by best method per condition
        metric: Metric name ("rmse" or "mse")
    
    Returns:
        fig, axes: Matplotlib figure and axes
    """    
    n_etas = len(etas)
    fig, axes = plt.subplots(n_etas, 1, figsize=figsize, sharex=False)
    
    if n_etas == 1:
        axes = [axes]
    
    for eta_idx, eta in enumerate(etas):
        ax = axes[eta_idx]
        
        # Filter data for this eta
        eta_df = df[np.isclose(df['eta'], eta)]
        
        # Create matrix: rows = methods, cols = (b_proto_frac, B) conditions
        n_conditions = len(b_proto_fracs) * len(Bs)
        matrix = np.zeros((len(methods), n_conditions))
        std_matrix = np.zeros((len(methods), n_conditions))
        condition_labels = []
        
        for cond_idx, (b_frac, B) in enumerate([(b, B) for b in b_proto_fracs for B in Bs]):
            condition_labels.append(f"b={b_frac:.2f}\nB={B}")
            
            # Get row for this condition
            cond_df = eta_df[
                np.isclose(eta_df['b_proto_frac'], b_frac) & (eta_df['B'] == B)
            ]
            
            if len(cond_df) == 0:
                for m_idx in range(len(methods)):
                    matrix[m_idx, cond_idx] = np.nan
                    std_matrix[m_idx, cond_idx] = np.nan
                continue
            
            row = cond_df.iloc[0]
            
            for m_idx, method in enumerate(methods):
                mean_val = row.get(f'{method}_{metric}_mean', np.nan)
                std_val = row.get(f'{method}_{metric}_se', np.nan)
                matrix[m_idx, cond_idx] = mean_val
                std_matrix[m_idx, cond_idx] = std_val
        
        # Normalize if requested
        if normalize_per_condition:
            for cond_idx in range(n_conditions):
                col = matrix[:, cond_idx]
                valid_vals = col[~np.isnan(col)]
                if len(valid_vals) > 0:
                    best_val = valid_vals.min()
                    matrix[:, cond_idx] = (col / best_val - 1) * 100
        
        # Create heatmap
        if normalize_per_condition:
            cmap = 'RdYlGn_r'
            vmin, vmax = 0, 50
        else:
            cmap = 'RdYlGn_r'
            valid_vals = matrix[~np.isnan(matrix)]
            if len(valid_vals) > 0:
                vmin, vmax = valid_vals.min(), valid_vals.max()
            else:
                vmin, vmax = 0, 1
        
        im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
        
        # Shorten method labels
        display_labels = [
            m.replace('mst_', '(CTL) MST-').replace('indiv', 'Indiv').replace('star', '(TL) StarTree').replace('random', '(CTL) RandTree')
            for m in methods
        ]
        
        # Set ticks
        ax.set_xticks(np.arange(n_conditions))
        ax.set_yticks(np.arange(len(methods)))
        ax.set_xticklabels(condition_labels, fontsize=9)
        ax.set_yticklabels(display_labels, fontsize=10)
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        for i in range(len(methods)):
            for j in range(n_conditions):
                val = matrix[i, j]
                std = std_matrix[i, j]
                
                if np.isnan(val):
                    text_str = "-"
                else:
                    if normalize_per_condition:
                        text_str = f"{val:+.1f}%"
                    else:
                        if show_std and not np.isnan(std):
                            text_str = f"{val:.3f}\n±{std:.3f}"
                        else:
                            text_str = f"{val:.3f}"
                
                if np.isnan(val):
                    color = "gray"
                else:
                    normalized_val = (val - vmin) / (vmax - vmin + 1e-8)
                    color = "white" if normalized_val > 0.6 else "black"
                
                ax.text(
                    j, i, text_str,
                    ha="center", va="center",
                    color=color,
                    fontsize=8, fontweight='bold'
                )
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if normalize_per_condition:
            cbar.set_label("% above best", rotation=270, labelpad=15, fontsize=9)
        else:
            cbar.set_label(f"Test {metric.upper()}", rotation=270, labelpad=15, fontsize=9)
        
        ax.set_title(rf"$\eta$ = {eta:.4f}", fontsize=12, fontweight='bold')
        ax.set_ylabel("Method", fontsize=10)
        if eta_idx == n_etas - 1:
            ax.set_xlabel("Experimental Conditions", fontsize=10)
    
    fig.suptitle(
        f"WEAVE Results: Test {metric.upper()} across Methods and Conditions",
        fontsize=14, fontweight='bold', y=0.995
    )
    plt.tight_layout()
    
    return fig, axes

def visualize_weave_results_bars(
    df: pd.DataFrame,
    methods: List[str],
    etas: Tuple[float, ...],
    b_proto_fracs: Tuple[float, ...],
    Bs: Tuple[int, ...],
    colors: Dict[str, str],
    figsize: Tuple[int, int] = (16, 5),
    baseline: str = "indiv",
    metric: str = "rmse",
):
    """
    Visualize WEAVE results as grouped bar charts showing relative performance.
    
    Args:
        df: DataFrame with WEAVE results
        methods: List of method names
        etas: Tuple of eta values
        b_proto_fracs: Tuple of seed budget fractions
        Bs: Tuple of budget values
        colors: Dictionary mapping methods to colors
        figsize: Figure size
        baseline: Method to use as baseline for comparison
        metric: Metric name
    
    Returns:
        fig, axes: Matplotlib figure and axes
    """    
    n_etas = len(etas)
    fig, axes = plt.subplots(1, n_etas, figsize=figsize, sharey=True)
    
    if n_etas == 1:
        axes = [axes]
    
    for eta_idx, eta in enumerate(etas):
        ax = axes[eta_idx]
        
        eta_df = df[np.isclose(df['eta'], eta)]
        
        method_means = []
        method_stds = []
        method_labels = []
        
        for method in methods:
            if method == baseline:
                continue
            
            vals = []
            for _, row in eta_df.iterrows():
                mean_val = row.get(f'{method}_{metric}_mean', np.nan)
                baseline_val = row.get(f'{baseline}_{metric}_mean', np.nan)
                
                if not np.isnan(mean_val) and not np.isnan(baseline_val):
                    improvement = (baseline_val - mean_val) / baseline_val * 100
                    vals.append(improvement)
            
            if len(vals) > 0:
                method_means.append(np.mean(vals))
                method_stds.append(np.std(vals))
                label = method.replace('mst_', '(CTL) MST-').replace('indiv', 'Indiv').replace('star', '(TL) StarTree').replace('random', '(CTL) RandTree')
                method_labels.append(label)
            else:
                method_means.append(0)
                method_stds.append(0)
                label = method.replace('mst_', '(CTL) MST-').replace('indiv', 'Indiv').replace('star', '(CTL) StarTree').replace('random', '(CTL) RandTree')
                method_labels.append(label)
        
        x = np.arange(len(method_labels))
        bars = ax.bar(
            x, method_means, yerr=method_stds,
            capsize=5, alpha=0.8,
            color=[colors.get(m, 'gray') for m in methods if m != baseline]
        )
        
        norm = plt.Normalize(vmin=min(method_means) if method_means else 0, vmax=max(method_means) if method_means else 1)
        cmap = plt.cm.RdYlGn

        for bar, mean in zip(bars, method_means):
            bar_color = cmap(norm(mean))
            bar.set_color(bar_color)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label("% Improvement", rotation=270, labelpad=15, fontsize=9)
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        
        ax.set_xticks(x)
        ax.set_xticklabels(method_labels, rotation=45, ha='right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        if eta_idx == 0:
            ax.set_ylabel(f"% Improvement vs {baseline.capitalize()}", fontsize=10)
        
        for i, (mean, std) in enumerate(zip(method_means, method_stds)):
            if abs(mean) > 0.5:
                ax.text(
                    i, mean + (std if mean > 0 else -std),
                    f"{mean:+.1f}%",
                    ha='center', va='bottom' if mean > 0 else 'top',
                    fontsize=8, fontweight='bold'
                )
    
    fig.suptitle(
        f"Average Performance vs {baseline.capitalize()} Baseline",
        fontsize=14, fontweight='bold', y=0.98
    )
    plt.tight_layout()
    
    return fig, axes


def visualize_weave_results_comparison(
    df: pd.DataFrame,
    methods: List[str],
    etas: Tuple[float, ...],
    b_proto_fracs: Tuple[float, ...],
    Bs: Tuple[int, ...],
    colors: Dict[str, str],
    figsize: Tuple[int, int] = (18, 6),
    methods_to_highlight: Optional[List[str]] = None,
    metric: str = "rmse",
):
    """
    Visualize WEAVE results as line plots comparing methods across budgets.
    
    Args:
        df: DataFrame with WEAVE results
        methods: List of method names
        etas: Tuple of eta values
        b_proto_fracs: Tuple of seed budget fractions
        Bs: Tuple of budget values
        colors: Dictionary mapping methods to colors
        figsize: Figure size
        methods_to_highlight: If provided, only plot these methods
        metric: Metric name
    
    Returns:
        fig, axes: Matplotlib figure and axes
    """    
    if methods_to_highlight is None:
        methods_to_plot = methods
    else:
        methods_to_plot = methods_to_highlight
    
    n_etas = len(etas)
    n_fracs = len(b_proto_fracs)
    
    fig, axes = plt.subplots(n_etas, n_fracs, figsize=figsize, sharex=True, sharey=True)
    
    if n_etas == 1 and n_fracs == 1:
        axes = np.array([[axes]])
    elif n_etas == 1:
        axes = axes.reshape(1, -1)
    elif n_fracs == 1:
        axes = axes.reshape(-1, 1)
    
    for eta_idx, eta in enumerate(etas):
        for frac_idx, b_frac in enumerate(b_proto_fracs):
            ax = axes[eta_idx, frac_idx]
            
            subset = df[np.isclose(df['eta'], eta) & np.isclose(df['b_proto_frac'], b_frac)]
            
            for method in methods_to_plot:
                means = []
                stds = []
                
                for B in Bs:
                    row = subset[subset['B'] == B]
                    if len(row) > 0:
                        mean_val = row.iloc[0].get(f'{method}_{metric}_mean', np.nan)
                        std_val = row.iloc[0].get(f'{method}_{metric}_se', np.nan)
                        means.append(mean_val)
                        stds.append(std_val)
                    else:
                        means.append(np.nan)
                        stds.append(np.nan)
                
                means = np.array(means)
                stds = np.array(stds)
                
                if np.all(np.isnan(means)):
                    continue
                
                label = method.replace('mst_', '(CTL) MST-').replace('indiv', 'Indiv').replace('star', '(TL) StarTree').replace('random', '(CTL) RandTree')
                
                ax.errorbar(
                    Bs, means, yerr=stds,
                    marker='o', capsize=4, label=label,
                    color=colors.get(method, None),
                    linewidth=2, markersize=6,
                    alpha=0.8,
                )
            
            ax.set_title(rf"$\eta$={eta:.4f}, b_seed={b_frac:.2f}", fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            if eta_idx == n_etas - 1:
                ax.set_xlabel("Budget B", fontsize=10)
            if frac_idx == 0:
                ax.set_ylabel(f"Test {metric.upper()}", fontsize=10)
            
            if eta_idx == 0 and frac_idx == n_fracs - 1:
                ax.legend(fontsize=8, loc='best', ncol=1)
    
    fig.suptitle(
        "Method Comparison Across Conditions",
        fontsize=14, fontweight='bold', y=0.995
    )
    plt.tight_layout()
    
    return fig, axes


def sweep_results_to_latex(results_iid, results_clustered, methods, Ts, Bs,
                           caption="Method comparison across $T$ and $B$",
                           label="tab:sweep_results"):
    """
    Generate a LaTeX table from sweep experiment results.
    
    Parameters
    ----------
    results_iid : dict
        Results from IID tasks: {method: {(T, B): [arrays]}}
    results_clustered : dict
        Results from clustered tasks: {method: {(T, B): [arrays]}}
    methods : list of str
        Method names.
    Ts : tuple/list
        Number of tasks values.
    Bs : tuple/list
        Budget values.
    
    Returns
    -------
    latex_str : str
        LaTeX table code.
    df : pd.DataFrame
        DataFrame with the results.
    """
    
    rows = []
    
    for method in methods:
        for T in Ts:
            for B in Bs:
                key = (T, B)
                # IID results
                if method in results_iid and key in results_iid[method]:
                    arr_iid = np.concatenate(results_iid[method][key])
                    mean_iid = np.mean(arr_iid)
                    std_iid = np.std(arr_iid)
                else:
                    mean_iid, std_iid = np.nan, np.nan
                
                # Clustered results
                if method in results_clustered and key in results_clustered[method]:
                    arr_clust = np.concatenate(results_clustered[method][key])
                    mean_clust = np.mean(arr_clust)
                    std_clust = np.std(arr_clust)
                else:
                    mean_clust, std_clust = np.nan, np.nan
                
                rows.append({
                    'Method': method,
                    'T': T,
                    'B': B,
                    'IID (mean)': mean_iid,
                    'IID (std)': std_iid,
                    'Clustered (mean)': mean_clust,
                    'Clustered (std)': std_clust,
                })
    
    df = pd.DataFrame(rows)
    
    # Build LaTeX table: columns are (T, B) combinations
    TB_pairs = [(T, B) for T in Ts for B in Bs]
    n_cols = len(TB_pairs)
    
    latex_lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l" + "c" * (n_cols * 2) + "}",
        r"\toprule",
    ]
    
    # Header row 1: Data type
    header1 = r"& \multicolumn{" + str(n_cols) + r"}{c}{IID Tasks} & \multicolumn{" + str(n_cols) + r"}{c}{Clustered Tasks} \\"
    latex_lines.append(header1)
    latex_lines.append(r"\cmidrule(lr){2-" + str(1 + n_cols) + r"} \cmidrule(lr){" + str(2 + n_cols) + "-" + str(1 + 2*n_cols) + "}")
    
    # Header row 2: (T, B) values
    tb_headers = " & ".join([f"$T\\!={T}, B\\!={B}$" for T, B in TB_pairs])
    header2 = r"Method & " + tb_headers + " & " + tb_headers + r" \\"
    latex_lines.append(header2)
    latex_lines.append(r"\midrule")
    
    # Data rows
    for method in methods:
        row_data = [method.replace("_", r"\_")]
        
        # IID columns
        for T, B in TB_pairs:
            df_row = df[(df['Method'] == method) & (df['T'] == T) & (df['B'] == B)]
            if len(df_row) > 0 and not np.isnan(df_row['IID (mean)'].values[0]):
                mean_val = df_row['IID (mean)'].values[0]
                std_val = df_row['IID (std)'].values[0]
                row_data.append(f"${mean_val:.2f} \\pm {std_val:.2f}$")
            else:
                row_data.append("--")
        
        # Clustered columns
        for T, B in TB_pairs:
            df_row = df[(df['Method'] == method) & (df['T'] == T) & (df['B'] == B)]
            if len(df_row) > 0 and not np.isnan(df_row['Clustered (mean)'].values[0]):
                mean_val = df_row['Clustered (mean)'].values[0]
                std_val = df_row['Clustered (std)'].values[0]
                row_data.append(f"${mean_val:.2f} \\pm {std_val:.2f}$")
            else:
                row_data.append("--")
        
        latex_lines.append(" & ".join(row_data) + r" \\")
    
    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ])
    
    latex_str = "\n".join(latex_lines)
    
    return latex_str, df


def sweep_results_to_latex_with_best(results_iid, results_clustered, methods, Ts, Bs,
                                      caption="Method comparison (best in bold, second underlined)",
                                      label="tab:sweep_best"):
    """
    Generate a LaTeX table with sweep results, best in bold, second best underlined.
    
    Returns
    -------
    latex_str : str
        LaTeX table code.
    df : pd.DataFrame
        DataFrame with the results.
    """    
    TB_pairs = [(T, B) for T in Ts for B in Bs]
    
    # Collect all means
    means_iid = {tb: {} for tb in TB_pairs}
    means_clust = {tb: {} for tb in TB_pairs}
    stds_iid = {tb: {} for tb in TB_pairs}
    stds_clust = {tb: {} for tb in TB_pairs}
    
    for method in methods:
        for tb in TB_pairs:
            if method in results_iid and tb in results_iid[method]:
                arr = np.concatenate(results_iid[method][tb])
                means_iid[tb][method] = np.mean(arr)
                stds_iid[tb][method] = np.std(arr)
            
            if method in results_clustered and tb in results_clustered[method]:
                arr = np.concatenate(results_clustered[method][tb])
                means_clust[tb][method] = np.mean(arr)
                stds_clust[tb][method] = np.std(arr)
    
    # Find best and second best
    def get_top_two(means_dict):
        if len(means_dict) < 2:
            items = list(means_dict.keys())
            return (items[0] if items else None, None)
        sorted_methods = sorted(means_dict.keys(), key=lambda m: means_dict[m])
        return sorted_methods[0], sorted_methods[1]
    
    best_iid, second_iid = {}, {}
    best_clust, second_clust = {}, {}
    for tb in TB_pairs:
        if means_iid[tb]:
            best_iid[tb], second_iid[tb] = get_top_two(means_iid[tb])
        if means_clust[tb]:
            best_clust[tb], second_clust[tb] = get_top_two(means_clust[tb])
    
    n_cols = len(TB_pairs)
    
    # Build LaTeX table
    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l" + "c" * (n_cols * 2) + "}",
        r"\toprule",
    ]
    
    # Header row 1
    header1 = r"& \multicolumn{" + str(n_cols) + r"}{c}{Homogeneous Tasks} & \multicolumn{" + str(n_cols) + r"}{c}{Clustered Tasks} \\"
    latex_lines.append(header1)
    latex_lines.append(r"\cmidrule(lr){2-" + str(1 + n_cols) + r"} \cmidrule(lr){" + str(2 + n_cols) + "-" + str(1 + 2*n_cols) + "}")
    
    # Header row 2
    tb_headers = " & ".join([f"$T\\!={T}, B\\!={B}$" for T, B in TB_pairs])
    header2 = r"Method & " + tb_headers + " & " + tb_headers + r" \\"
    latex_lines.append(header2)
    latex_lines.append(r"\midrule")
    
    # Data rows
    for method in methods:
        method_display = method.replace("_", r"\_")
        row_data = [method_display]
        
        # IID columns
        for tb in TB_pairs:
            if tb in means_iid and method in means_iid[tb]:
                mean_val = means_iid[tb][method]
                std_val = stds_iid[tb][method]
                val_str = f"{mean_val:.2f} \\pm {std_val:.2f}"
                if method == best_iid.get(tb):
                    row_data.append(f"$\\mathbf{{{val_str}}}$")
                elif method == second_iid.get(tb):
                    row_data.append(f"$\\underline{{{val_str}}}$")
                else:
                    row_data.append(f"${val_str}$")
            else:
                row_data.append("--")
        
        # Clustered columns
        for tb in TB_pairs:
            if tb in means_clust and method in means_clust[tb]:
                mean_val = means_clust[tb][method]
                std_val = stds_clust[tb][method]
                val_str = f"{mean_val:.2f} \\pm {std_val:.2f}"
                if method == best_clust.get(tb):
                    row_data.append(f"$\\mathbf{{{val_str}}}$")
                elif method == second_clust.get(tb):
                    row_data.append(f"$\\underline{{{val_str}}}$")
                else:
                    row_data.append(f"${val_str}$")
            else:
                row_data.append("--")
        
        latex_lines.append(" & ".join(row_data) + r" \\")
    
    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ])
    
    latex_str = "\n".join(latex_lines)
    
    # Build DataFrame
    rows = []
    for method in methods:
        for tb in TB_pairs:
            T, B = tb
            rows.append({
                'Method': method,
                'T': T,
                'B': B,
                'IID (mean ± std)': f"{means_iid[tb].get(method, np.nan):.2f} ± {stds_iid[tb].get(method, np.nan):.2f}",
                'Clustered (mean ± std)': f"{means_clust[tb].get(method, np.nan):.2f} ± {stds_clust[tb].get(method, np.nan):.2f}",
            })
    df = pd.DataFrame(rows)
    
    return latex_str, df

def sweep_results_to_latex_by_T(results_iid, results_clustered, methods, Ts, Bs,
                                 caption="Method comparison by T (best in bold, second underlined)",
                                 label="tab:sweep_by_T"):
    """
    Generate separate LaTeX sub-tables for each T value.
    
    Returns
    -------
    latex_str : str
        LaTeX table code.
    df : pd.DataFrame
        DataFrame with the results.
    """    
    all_rows = []
    
    # Collect all means
    means_iid = {}
    means_clust = {}
    stds_iid = {}
    stds_clust = {}
    
    for T in Ts:
        for B in Bs:
            tb = (T, B)
            means_iid[tb] = {}
            means_clust[tb] = {}
            stds_iid[tb] = {}
            stds_clust[tb] = {}
            
            for method in methods:
                if method in results_iid and tb in results_iid[method]:
                    arr = np.concatenate(results_iid[method][tb])
                    means_iid[tb][method] = np.mean(arr)
                    stds_iid[tb][method] = np.std(arr)
                
                if method in results_clustered and tb in results_clustered[method]:
                    arr = np.concatenate(results_clustered[method][tb])
                    means_clust[tb][method] = np.mean(arr)
                    stds_clust[tb][method] = np.std(arr)
    
    # Find best and second best
    def get_top_two(means_dict):
        if len(means_dict) < 2:
            items = list(means_dict.keys())
            return (items[0] if items else None, None)
        sorted_methods = sorted(means_dict.keys(), key=lambda m: means_dict[m])
        return sorted_methods[0], sorted_methods[1]
    
    best_iid, second_iid = {}, {}
    best_clust, second_clust = {}, {}
    for T in Ts:
        for B in Bs:
            tb = (T, B)
            if means_iid[tb]:
                best_iid[tb], second_iid[tb] = get_top_two(means_iid[tb])
            if means_clust[tb]:
                best_clust[tb], second_clust[tb] = get_top_two(means_clust[tb])
    
    n_Bs = len(Bs)
    
    # Build LaTeX table
    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{ll" + "c" * (n_Bs * 2) + "}",
        r"\toprule",
    ]
    
    # Header row 1
    header1 = r"& & \multicolumn{" + str(n_Bs) + r"}{c}{IID Tasks} & \multicolumn{" + str(n_Bs) + r"}{c}{Clustered Tasks} \\"
    latex_lines.append(header1)
    latex_lines.append(r"\cmidrule(lr){3-" + str(2 + n_Bs) + r"} \cmidrule(lr){" + str(3 + n_Bs) + "-" + str(2 + 2*n_Bs) + "}")
    
    # Header row 2
    b_headers = " & ".join([f"$B={B}$" for B in Bs])
    header2 = r"$T$ & Method & " + b_headers + " & " + b_headers + r" \\"
    latex_lines.append(header2)
    latex_lines.append(r"\midrule")
    
    # Data rows grouped by T
    for i, T in enumerate(Ts):
        for j, method in enumerate(methods):
            method_display = method.replace("_", r"\_")
            
            # First column: T value (only for first method in group)
            if j == 0:
                row_data = [f"${T}$", method_display]
            else:
                row_data = ["", method_display]
            
            # IID columns
            for B in Bs:
                tb = (T, B)
                if tb in means_iid and method in means_iid[tb]:
                    mean_val = means_iid[tb][method]
                    std_val = stds_iid[tb][method]
                    val_str = f"{mean_val:.2f} \\pm {std_val:.2f}"
                    if method == best_iid.get(tb):
                        row_data.append(f"$\\mathbf{{{val_str}}}$")
                    elif method == second_iid.get(tb):
                        row_data.append(f"$\\underline{{{val_str}}}$")
                    else:
                        row_data.append(f"${val_str}$")
                else:
                    row_data.append("--")
            
            # Clustered columns
            for B in Bs:
                tb = (T, B)
                if tb in means_clust and method in means_clust[tb]:
                    mean_val = means_clust[tb][method]
                    std_val = stds_clust[tb][method]
                    val_str = f"{mean_val:.2f} \\pm {std_val:.2f}"
                    if method == best_clust.get(tb):
                        row_data.append(f"$\\mathbf{{{val_str}}}$")
                    elif method == second_clust.get(tb):
                        row_data.append(f"$\\underline{{{val_str}}}$")
                    else:
                        row_data.append(f"${val_str}$")
                else:
                    row_data.append("--")
            
            latex_lines.append(" & ".join(row_data) + r" \\")
            
            all_rows.append({
                'T': T, 'Method': method,
                **{f'IID B={B}': f"{means_iid.get((T,B), {}).get(method, np.nan):.2f}" for B in Bs},
                **{f'Clust B={B}': f"{means_clust.get((T,B), {}).get(method, np.nan):.2f}" for B in Bs},
            })
        
        # Add midrule between T groups (except last)
        if i < len(Ts) - 1:
            latex_lines.append(r"\midrule")
    
    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ])
    
    latex_str = "\n".join(latex_lines)
    df = pd.DataFrame(all_rows)
    
    return latex_str, df

def sweep_results_to_latex_unified(
    all_results: Dict[float, Dict],
    methods: List[str],
    Ts: Tuple[int, ...],
    Bs: Tuple[int, ...],
    taus: Tuple[float, ...],
    caption: str = "Test MSE sweep results",
    label: str = "tab:sweep_unified",
    show_std: bool = True,
    group_mst: bool = True,
) -> Tuple[str, pd.DataFrame]:
    """
    Generate unified LaTeX table for regression sweep results.
    Handles all MST variants.
    
    Args:
        all_results: Dict[tau -> Dict[method -> Dict[(T, B) -> List[values]]]]
        group_mst: if True, group MST methods in a multicolumn header
    
    Returns:
        latex_str: LaTeX table string
        df: DataFrame with all results
    """
    
    # Method display name mapping
    DISPLAY_NAMES = {
        "Indiv": "Indiv",
        "Star": "Star",
        "CTL-RandTree": "RandTree",
        "CTL-MST-feature": "feature",
        "CTL-MST-target": "target",
        "CTL-MST-gradient": "gradient",
        "CTL-MST-model": "model",
        "CTL-MST-kl": "kl",
        "CTL-MST-wasserstein": "wass",
    }
    
    def get_display_name(m: str) -> str:
        if m in DISPLAY_NAMES:
            return DISPLAY_NAMES[m]
        # Fallback: strip common prefixes
        return m.replace('CTL-MST-', '').replace('CTL-', '')
    
    def format_val(mean, std, is_best, is_second):
        """Format value with bold/underline."""
        if np.isnan(mean):
            return "$-$"
        
        if show_std:
            val_str = f"{mean:.1f} \\pm {std:.1f}"
        else:
            val_str = f"{mean:.1f}"
        
        if is_best:
            return f"$\\mathbf{{{val_str}}}$"
        elif is_second:
            return f"$\\underline{{{val_str}}}$"
        else:
            return f"${val_str}$"
    
    rows = []
    
    for tau in taus:
        for T in Ts:
            for B in Bs:
                key = (T, B)
                results = all_results[tau]
                
                means = []
                stds = []
                for m in methods:
                    if m in results and key in results[m] and len(results[m][key]) > 0:
                        vals = np.array(results[m][key])
                        vals = vals[~np.isnan(vals)]
                        if len(vals) > 0:
                            means.append(vals.mean())
                            stds.append(vals.std())
                        else:
                            means.append(np.nan)
                            stds.append(np.nan)
                    else:
                        means.append(np.nan)
                        stds.append(np.nan)
                
                valid_means = [(i, m) for i, m in enumerate(means) if not np.isnan(m)]
                valid_means.sort(key=lambda x: x[1])
                
                best_idx = valid_means[0][0] if len(valid_means) > 0 else -1
                second_idx = valid_means[1][0] if len(valid_means) > 1 else -1
                
                row = {
                    'tau': tau,
                    'T': T,
                    'B': B,
                }
                
                for i, m in enumerate(methods):
                    row[f'{m}_formatted'] = format_val(
                        means[i], stds[i],
                        i == best_idx, i == second_idx
                    )
                                
                rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Categorize methods for grouping
    mst_methods = [m for m in methods if 'CTL-MST-' in m]
    base_methods = [m for m in methods if m not in mst_methods]
    
    # Build LaTeX table
    latex_lines = []
    latex_lines.append("\\begin{table}[ht]")
    latex_lines.append("\\centering")
    latex_lines.append("\\small")
    latex_lines.append("\\resizebox{\\textwidth}{!}{")
    
    n_methods = len(methods)
    col_spec = "ccc|" + 'c' * n_methods
    latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex_lines.append("\\toprule")
    
    # Build multicolumn header row if grouping is enabled
    if group_mst and len(mst_methods) > 0:
        header1_parts = ["", "", ""]  # tau, T, B
        
        # Add base methods (no grouping)
        for m in base_methods:
            header1_parts.append("")
        
        # Add MST methods group
        header1_parts.append(f"\\multicolumn{{{len(mst_methods)}}}{{c}}{{MST Variants}}")
        
        # Build header1 string
        header1 = " & ".join(header1_parts) + " \\\\"
        latex_lines.append(header1)
        latex_lines.append("\\cmidrule(lr){" + str(4 + len(base_methods)) + "-" + str(3 + len(methods)) + "}")
    
    # Column headers row
    header2 = "$\\tau$ & $T$ & $B$"
    
    # Maintain method order: base -> MST
    ordered_methods = base_methods + mst_methods
    
    for m in ordered_methods:
        display_name = get_display_name(m)
        header2 += f" & \\rotatebox{{45}}{{{display_name}}}"
    header2 += " \\\\"
    latex_lines.append(header2)
    latex_lines.append("\\midrule")
    
    # Data rows
    prev_tau = None
    for _, row in df.iterrows():
        if prev_tau is not None and row['tau'] != prev_tau:
            latex_lines.append("\\midrule")
        
        line = f"{row['tau']:.0f} & {row['T']} & {row['B']}"
        
        for m in ordered_methods:
            col_key = f'{m}_formatted'
            if col_key in row:
                line += f" & {row[col_key]}"
            else:
                line += " & $-$"
        
        line += " \\\\"
        latex_lines.append(line)
        prev_tau = row['tau']
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("}")
    latex_lines.append(f"\\caption{{{caption}}}")
    latex_lines.append(f"\\label{{{label}}}")
    latex_lines.append("\\end{table}")
    
    return "\n".join(latex_lines), df

def extract_mean(value):
    """Extract the mean value from a string in the format '{mean} ± {std}'."""
    import re
    match = re.match(r"([0-9.]+)\s*±\s*[0-9.]+", value)
    if match:
        return float(match.group(1))  # Extract the mean value
    else:
        raise ValueError(f"Invalid format: {value}")
    
def compute_pairwise_statistical_tests(results, baseline="Indiv", alpha=0.05):
    """
    Compute pairwise statistical tests comparing each method to baseline.
    
    Args:
        results: Dict[method -> Dict[(T, B) -> List[mse_values]]]
        baseline: baseline method name
        alpha: significance level
    
    Returns:
        test_results: Dict[(T, B) -> Dict[method -> test_result]]
    """
    test_results = {}
    
    for key in results[baseline].keys():
        test_results[key] = {}
        
        baseline_mses = np.array(results[baseline][key])
        baseline_mses = baseline_mses[~np.isnan(baseline_mses)]
        
        if len(baseline_mses) < 2:
            continue
            
        for method in results.keys():
            if method == baseline:
                continue
                
            if key not in results[method]:
                continue
                
            method_mses = np.array(results[method][key])
            method_mses = method_mses[~np.isnan(method_mses)]
            
            # Check both arrays have sufficient valid data
            if len(method_mses) < 2:
                continue
            
            # Ensure equal lengths by taking minimum
            min_len = min(len(baseline_mses), len(method_mses))
            if min_len < 2:
                continue
                
            baseline_subset = baseline_mses[:min_len]
            method_subset = method_mses[:min_len]
            
            # Perform paired t-test
            t_stat, p_value = stats.ttest_rel(baseline_subset, method_subset)
            
            # Calculate effect size (Cohen's d)
            diff = baseline_subset - method_subset
            cohens_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0
            
            test_results[key][method] = {
                'p_value': p_value,
                'significant': p_value < alpha,
                'cohens_d': cohens_d,
                't_stat': t_stat,
                'baseline_mean': baseline_subset.mean(),
                'method_mean': method_subset.mean(),
            }
    
    return test_results

def print_statistical_comparison(results, methods, Ts, Bs):
    """Print statistical comparison results."""
    print("\nSTATISTICAL COMPARISON (vs Indiv)")
    print("=" * 80)
    
    test_results = compute_pairwise_statistical_tests(results)
    
    for T in Ts:
        for B in Bs:
            key = (T, B)
            
            if key not in test_results:
                continue
                
            print(f"\nT={T}, B={B}")
            print("-" * 60)
            
            for method in methods:
                if method == "Indiv":
                    continue
                    
                if method not in test_results[key]:
                    print(f"{method:20s}: No data")
                    continue
                    
                result = test_results[key][method]
                sig_marker = "***" if result['significant'] else "   "
                
                print(f"{method:20s}: p={result['p_value']:.4f} {sig_marker} "
                      f"d={result['cohens_d']:+.3f} "
                      f"(base={result['baseline_mean']:.1f}, "
                      f"method={result['method_mean']:.1f})")
                
def compute_statistical_tests(
    results: Dict[str, Dict[Tuple[int, int], List[float]]],
    baseline: str = "Indiv",
    methods: Optional[List[str]] = None,
    alpha: float = 0.05,
) -> "pd.DataFrame":
    """
    Compute paired statistical tests of each method vs baseline for each (T,B).
    Returns DataFrame with columns:
      ['T','B','Method','Mean_Diff','Std_Diff','p_value','wilcoxon_p','cohens_d','significant']
    Mean_Diff and Std_Diff are in percentage points.
    """
    all_methods = list(results.keys())
    if methods is None:
        methods = [m for m in all_methods if m != baseline]
    rows = []

    # collect all (T,B) keys from baseline (fallback to union if missing)
    keys = set()
    for m in results:
        keys.update(results[m].keys())
    keys = sorted(keys)

    for key in keys:
        T, B = key
        base_vals = np.array(results.get(baseline, {}).get(key, []), dtype=float)
        base_vals = base_vals[~np.isnan(base_vals)]

        for m in methods:
            if m == baseline:
                continue
            meth_vals = np.array(results.get(m, {}).get(key, []), dtype=float)
            meth_vals = meth_vals[~np.isnan(meth_vals)]

            # require at least 2 paired samples
            n = min(len(base_vals), len(meth_vals))
            if n < 2:
                continue

            base_p = base_vals[:n]
            meth_p = meth_vals[:n]

            diffs = (meth_p - base_p)  # in [0,1] accuracy units
            mean_diff = float(np.mean(diffs) * 100.0)  # percentage points
            std_diff = float(np.std(diffs, ddof=0) * 100.0)

            # Paired t-test
            try:
                t_stat, p_value = stats.ttest_rel(meth_p, base_p, nan_policy='omit')
            except Exception:
                t_stat, p_value = np.nan, np.nan

            # Wilcoxon signed-rank (non-parametric)
            try:
                wil_res = stats.wilcoxon(meth_p, base_p)
                wil_p = wil_res.pvalue
            except Exception:
                wil_p = np.nan

            # Cohen's d for paired samples: mean(diffs) / std(diffs, ddof=1)
            denom = np.std(diffs, ddof=1) if np.std(diffs, ddof=1) > 0 else np.nan
            cohens_d = (np.mean(diffs) / denom) if not np.isnan(denom) else np.nan

            significant = False
            if not np.isnan(p_value) and p_value < alpha:
                significant = True

            rows.append({
                "T": T,
                "B": B,
                "Method": m,
                "Mean_Diff": mean_diff,
                "Std_Diff": std_diff,
                "p_value": float(p_value) if not np.isnan(p_value) else np.nan,
                "wilcoxon_p": float(wil_p) if not np.isnan(wil_p) else np.nan,
                "cohens_d": float(cohens_d) if not np.isnan(cohens_d) else np.nan,
                "significant": bool(significant),
                "n_pairs": int(n),
            })

    df = pd.DataFrame(rows)
    # order columns
    cols = ["T","B","Method","n_pairs","Mean_Diff","Std_Diff","p_value","wilcoxon_p","cohens_d","significant"]
    return df[cols]

def print_statistical_summary(
    results: Dict[str, Dict[Tuple[int, int], List[float]]],
    methods: List[str],
    Ts: Tuple[int, ...],
    Bs: Tuple[int, ...],
):
    """Print summary with proper statistics across repetitions."""
    print("\n" + "=" * 70)
    print("STATISTICAL SUMMARY (Mean ± Std across repetitions)")
    print("=" * 70)
    
    for T in Ts:
        for B in Bs:
            key = (T, B)
            print(f"\nT={T}, B={B}:")
            print("-" * 50)
            
            for m in methods:
                if key in results[m] and len(results[m][key]) > 0:
                    accs = np.array(results[m][key]) * 100  # Convert to %
                    print(f"  {m:20s}: {accs.mean():.2f} ± {accs.std():.2f}%")
    
    # Statistical tests
    print("\n" + "=" * 70)
    print("STATISTICAL TESTS (vs Indiv)")
    print("=" * 70)
    
    test_results = compute_statistical_tests(results)
    
    for _, row in test_results.iterrows():
        sig_marker = "*" if row['significant'] else ""
        print(
            f"T={row['T']}, B={row['B']}, {row['Method']:15s}: "
            f"Δ={row['Mean_Diff']:+.2f}% ± {row['Std_Diff']:.2f}%, "
            f"p={row['p_value']:.4f}{sig_marker}, "
            f"d={row['cohens_d']:.3f}"
        )

def visualize_sweep_results_table(
    df: pd.DataFrame,
    methods: List[str],
    taus: Tuple[float, ...],
    Ts: Tuple[int, ...],
    Bs: Tuple[int, ...],
    colors: Dict[str, str],
    figsize: Tuple[int, int] = (16, 10),
    show_std: bool = False,
    normalize_per_condition: bool = False,
):
    """
    Visualize sweep results as heatmaps showing mean MSE across conditions.
    
    Args:
        df: DataFrame from sweep_results_to_latex_unified
        methods: List of method names
        taus: Tuple of tau values
        Ts: Tuple of T values
        Bs: Tuple of B values
        colors: Dictionary mapping methods to colors
        figsize: Figure size
        show_std: If True, add standard deviation as error bars
        normalize_per_condition: If True, normalize by best method per condition
    
    Returns:
        fig, axes: Matplotlib figure and axes
    """    
    n_taus = len(taus)
    fig, axes = plt.subplots(n_taus, 1, figsize=figsize, sharex=False)
    
    if n_taus == 1:
        axes = [axes]
    
    # Prepare data for each tau
    for tau_idx, tau in enumerate(taus):
        ax = axes[tau_idx]
        
        # Filter data for this tau
        tau_df = df[df['tau'] == tau]
        
        # Create matrix: rows = methods, cols = (T, B) conditions
        n_conditions = len(Ts) * len(Bs)
        matrix = np.zeros((len(methods), n_conditions))
        std_matrix = np.zeros((len(methods), n_conditions))
        condition_labels = []
        
        for cond_idx, (T, B) in enumerate([(T, B) for T in Ts for B in Bs]):
            condition_labels.append(f"T={T}\nB={B}")
            
            # Get row for this condition
            cond_df = tau_df[(tau_df['T'] == T) & (tau_df['B'] == B)]
            
            if len(cond_df) == 0:
                for m_idx in range(len(methods)):
                    matrix[m_idx, cond_idx] = np.nan
                    std_matrix[m_idx, cond_idx] = np.nan
                continue
            
            row = cond_df.iloc[0]
            
            for m_idx, method in enumerate(methods):
                mean_val = row.get(f'Clust_{method}_mean', np.nan)
                std_val = row.get(f'Clust_{method}_std', np.nan)
                matrix[m_idx, cond_idx] = mean_val
                std_matrix[m_idx, cond_idx] = std_val
        
        # Normalize if requested
        if normalize_per_condition:
            for cond_idx in range(n_conditions):
                col = matrix[:, cond_idx]
                valid_vals = col[~np.isnan(col)]
                if len(valid_vals) > 0:
                    best_val = valid_vals.min()
                    matrix[:, cond_idx] = (col / best_val - 1) * 100  # Percentage above best
        
        # Create heatmap
        if normalize_per_condition:
            # Diverging colormap for normalized values
            cmap = 'RdYlGn_r'
            vmin, vmax = 0, 50  # 0% to 50% above best
        else:
            # Sequential colormap for raw MSE
            cmap = 'RdYlGn_r'
            valid_vals = matrix[~np.isnan(matrix)]
            if len(valid_vals) > 0:
                vmin, vmax = valid_vals.min(), valid_vals.max()
            else:
                vmin, vmax = 0, 1
        
        im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
        
        # Shorten method labels
        display_labels = [m.replace('CTL-MST-', 'MST-').replace('CTL-', '') for m in methods]
        
        # Set ticks
        ax.set_xticks(np.arange(n_conditions))
        ax.set_yticks(np.arange(len(methods)))
        ax.set_xticklabels(condition_labels, fontsize=9)
        ax.set_yticklabels(display_labels, fontsize=10)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        for i in range(len(methods)):
            for j in range(n_conditions):
                val = matrix[i, j]
                std = std_matrix[i, j]
                
                if np.isnan(val):
                    text_str = "-"
                else:
                    if normalize_per_condition:
                        text_str = f"{val:+.1f}%"
                    else:
                        if show_std and not np.isnan(std):
                            text_str = f"{val:.0f}\n±{std:.0f}"
                        else:
                            text_str = f"{val:.0f}"
                
                # Determine text color based on background
                if np.isnan(val):
                    color = "gray"
                else:
                    # Use white text for dark backgrounds
                    normalized_val = (val - vmin) / (vmax - vmin + 1e-8)
                    color = "white" if normalized_val > 0.6 else "black"
                
                ax.text(
                    j, i, text_str,
                    ha="center", va="center",
                    color=color,
                    fontsize=8, fontweight='bold'
                )
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if normalize_per_condition:
            cbar.set_label("% above best", rotation=270, labelpad=15, fontsize=9)
        else:
            cbar.set_label("Test MSE", rotation=270, labelpad=15, fontsize=9)
        
        # Title
        ax.set_title(rf"$\tau$ = {tau}", fontsize=12, fontweight='bold')
        ax.set_ylabel("Method", fontsize=10)
        if tau_idx == n_taus - 1:
            ax.set_xlabel("Experimental Conditions", fontsize=10)
    
    fig.suptitle(
        "Sweep Results: Test MSE across Methods and Conditions",
        fontsize=14, fontweight='bold', y=0.995
    )
    plt.tight_layout()
    
    return fig, axes


def visualize_sweep_results_bars(
    df: pd.DataFrame,
    methods: List[str],
    taus: Tuple[float, ...],
    Ts: Tuple[int, ...],
    Bs: Tuple[int, ...],
    colors: Dict[str, str],
    figsize: Tuple[int, int] = (16, 5),
    baseline: str = "Indiv",
):
    """
    Visualize sweep results as grouped bar charts showing relative performance.
    
    Args:
        df: DataFrame from sweep_results_to_latex_unified
        methods: List of method names
        taus: Tuple of tau values
        Ts: Tuple of T values
        Bs: Tuple of B values
        colors: Dictionary mapping methods to colors
        figsize: Figure size
        baseline: Method to use as baseline for comparison
    
    Returns:
        fig, axes: Matplotlib figure and axes
    """    
    n_taus = len(taus)
    fig, axes = plt.subplots(1, n_taus, figsize=figsize, sharey=True)
    
    if n_taus == 1:
        axes = [axes]
    
    for tau_idx, tau in enumerate(taus):
        ax = axes[tau_idx]
        
        # Filter data for this tau
        tau_df = df[df['tau'] == tau]
        
        # Aggregate across conditions (T, B)
        method_means = []
        method_stds = []
        method_labels = []
        
        for method in methods:
            if method == baseline:
                continue
            
            # Get all mean values for this method
            vals = []
            for _, row in tau_df.iterrows():
                mean_val = row.get(f'Clust_{method}_mean', np.nan)
                baseline_val = row.get(f'Clust_{baseline}_mean', np.nan)
                
                if not np.isnan(mean_val) and not np.isnan(baseline_val):
                    # Compute percentage improvement (negative = better)
                    improvement = (baseline_val - mean_val) / baseline_val * 100
                    vals.append(improvement)
            
            if len(vals) > 0:
                method_means.append(np.mean(vals))
                method_stds.append(np.std(vals))
                method_labels.append(method.replace('CTL-MST-', 'MST-').replace('CTL-', ''))
            else:
                method_means.append(0)
                method_stds.append(0)
                method_labels.append(method.replace('CTL-MST-', 'MST-').replace('CTL-', ''))
        
        # Create bar chart
        x = np.arange(len(method_labels))
        bars = ax.bar(
            x, method_means, yerr=method_stds,
            capsize=5, alpha=0.8,
            color=[colors.get(m, 'gray') for m in methods if m != baseline]
        )
        
        # Color bars based on positive/negative
        # Normalize the color range to the data range
        norm = plt.Normalize(vmin=min(method_means), vmax=max(method_means))
        cmap = plt.cm.RdYlGn

        for i, (bar, mean) in enumerate(zip(bars, method_means)):
            # Map the mean value to a color in the colormap
            bar_color = cmap(norm(mean))
            bar.set_color(bar_color)

        # Add a colorbar to the plot
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label("% Improvement", rotation=270, labelpad=15, fontsize=9)
        
        # Add horizontal line at 0
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        
        # Formatting
        ax.set_xticks(x)
        ax.set_xticklabels(method_labels, rotation=45, ha='right', fontsize=9)
        ax.set_title(rf"$\tau$ = {tau}", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        if tau_idx == 0:
            ax.set_ylabel(f"% Improvement vs {baseline}", fontsize=10)
        
        # Add value labels on bars
        for i, (mean, std) in enumerate(zip(method_means, method_stds)):
            if abs(mean) > 0.5:  # Only label if substantial
                ax.text(
                    i, mean + (std if mean > 0 else -std),
                    f"{mean:+.1f}%",
                    ha='center', va='bottom' if mean > 0 else 'top',
                    fontsize=8, fontweight='bold'
                )
    
    fig.suptitle(
        f"Average Performance vs {baseline} Baseline",
        fontsize=14, fontweight='bold', y=0.98
    )
    plt.tight_layout()
    
    return fig, axes


def visualize_sweep_results_comparison(
    df: pd.DataFrame,
    methods: List[str],
    taus: Tuple[float, ...],
    Ts: Tuple[int, ...],
    Bs: Tuple[int, ...],
    colors: Dict[str, str],
    figsize: Tuple[int, int] = (18, 6),
    methods_to_highlight: Optional[List[str]] = None,
):
    """
    Visualize sweep results as line plots comparing methods across budgets.
    
    Args:
        df: DataFrame from sweep_results_to_latex_unified
        methods: List of method names
        taus: Tuple of tau values
        Ts: Tuple of T values
        Bs: Tuple of B values
        colors: Dictionary mapping methods to colors
        figsize: Figure size
        methods_to_highlight: If provided, only plot these methods
    
    Returns:
        fig, axes: Matplotlib figure and axes
    """    
    if methods_to_highlight is None:
        methods_to_plot = methods
    else:
        methods_to_plot = methods_to_highlight
    
    n_taus = len(taus)
    n_ts = len(Ts)
    
    fig, axes = plt.subplots(n_taus, n_ts, figsize=figsize, sharex=True, sharey=True)
    
    if n_taus == 1 and n_ts == 1:
        axes = np.array([[axes]])
    elif n_taus == 1:
        axes = axes.reshape(1, -1)
    elif n_ts == 1:
        axes = axes.reshape(-1, 1)
    
    for tau_idx, tau in enumerate(taus):
        for t_idx, T in enumerate(Ts):
            ax = axes[tau_idx, t_idx]
            
            # Filter data for this tau and T
            subset = df[(df['tau'] == tau) & (df['T'] == T)]
            
            for method in methods_to_plot:
                means = []
                stds = []
                
                for B in Bs:
                    row = subset[subset['B'] == B]
                    if len(row) > 0:
                        mean_val = row.iloc[0].get(f'Clust_{method}_mean', np.nan)
                        std_val = row.iloc[0].get(f'Clust_{method}_std', np.nan)
                        means.append(mean_val)
                        stds.append(std_val)
                    else:
                        means.append(np.nan)
                        stds.append(np.nan)
                
                means = np.array(means)
                stds = np.array(stds)
                
                # Skip if all NaN
                if np.all(np.isnan(means)):
                    continue
                
                label = method.replace('CTL-MST-', 'MST-').replace('CTL-', '')
                
                ax.errorbar(
                    Bs, means, yerr=stds,
                    marker='o', capsize=4, label=label,
                    color=colors.get(method, None),
                    linewidth=2, markersize=6,
                    alpha=0.8,
                )
            
            # Formatting
            ax.set_title(rf"$\tau$={tau}, T={T}", fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            if tau_idx == n_taus - 1:
                ax.set_xlabel("Budget B", fontsize=10)
            if t_idx == 0:
                ax.set_ylabel("Test MSE", fontsize=10)
            
            if tau_idx == 0 and t_idx == n_ts - 1:
                ax.legend(fontsize=8, loc='best', ncol=1)
    
    fig.suptitle(
        "Method Comparison Across Conditions",
        fontsize=14, fontweight='bold', y=0.995
    )
    plt.tight_layout()
    
    return fig, axes

FAMILY_MAP = {
    "Feature": ["feature", "mmd", "meancov", "mean_cov", "cka"],
    "Target": ["target", "kl", "js", "wasserstein"],
    "Optimization": ["gradient", "model"],
}

VALUE_RE = re.compile(r"\$?\s*([\d.]+)\s*\\pm\s*([\d.]+)\s*\$?")


def parse_value(cell):
    """Parse 'mean ± std' from a LaTeX cell."""
    m = VALUE_RE.search(cell)
    if m is None:
        return None
    return float(m.group(1)), float(m.group(2))


def family_of_column(col_name):
    """Map column name to family."""
    col = col_name.lower().replace("-", "").replace("_", "").replace(" ", "")
    for fam, keys in FAMILY_MAP.items():
        for k in keys:
            k_clean = k.replace("-", "").replace("_", "")
            if k_clean in col:
                return fam
    return None

def simplify_table(tex_in: str, tex_out: str):
    """
    Reads a 17-column LaTeX table and outputs a 10-column version
    by averaging metric families.
    """
    from pathlib import Path
    import re

    content = Path(tex_in).read_text()
    lines = content.split("\n")

    # Column indices (0-based) in the 17-column table
    PARAM_COLS = [0, 1, 2, 3]  # Dataset, tau, T, B
    BASELINE_COLS = [4, 5, 6]  # Indiv, Star, CTL-RandTree
    FEATURE_COLS = [7, 13, 14, 16]  # Feature, MMD, MeanCov, CKA
    TARGET_COLS = [8, 11, 12, 15]   # Target, KL, Wasserstein, JS
    OPTIM_COLS = [9, 10]            # Gradient, Model

    def parse_value(cell: str):
        """Extract mean and std from '$mean \\pm std$' format."""
        cell = cell.strip()
        # Remove \mathbf{}, \underline{}, $
        cell = re.sub(r'\\mathbf\{([^}]*)\}', r'\1', cell)
        cell = re.sub(r'\\underline\{([^}]*)\}', r'\1', cell)
        cell = cell.replace('$', '').strip()
        
        match = re.search(r'([\d.]+)\s*\\pm\s*([\d.]+)', cell)
        if match:
            return float(match.group(1)), float(match.group(2))
        return None, None

    def average_cells(cells: list, indices: list):
        """Average values from specified column indices."""
        means, stds = [], []
        for i in indices:
            if i < len(cells):
                m, s = parse_value(cells[i])
                if m is not None:
                    means.append(m)
                    stds.append(s)
        if means:
            return sum(means) / len(means), sum(stds) / len(stds)
        return None, None

    # Build output
    out_lines = [
        "\\begin{table*}[t]",
        "    \\centering",
        "    \\caption{Simplified results with metric families averaged.}",
        "    \\label{tab:simplified}",
        "    \\resizebox{\\textwidth}{!}{%",
        "    \\begin{tabular}{cccc|ccc|ccc}",
        "    \\toprule",
        "    \\multicolumn{4}{c|}{Params}",
        "    & \\multicolumn{3}{c|}{Baselines}",
        "    & \\multicolumn{3}{c}{CTL-MST (averaged)} \\\\",
        "    Dataset & $\\tau$ & $T$ & $B$",
        "    & Indiv & Star & RandTree",
        "    & Feature & Target & Optim \\\\",
        "    \\midrule",
    ]

    # Join multi-line rows and process
    rows_processed = 0
    current_row = ""
    in_tabular = False
    
    for line in lines:
        # Track when we're inside tabular
        if "\\begin{tabular}" in line:
            in_tabular = True
            continue
        if "\\end{tabular}" in line:
            in_tabular = False
            continue
        if not in_tabular:
            continue
            
        # Skip header/structure lines
        if any(x in line for x in ["\\toprule", "\\midrule", "\\bottomrule", 
                                    "\\multicolumn{4}", "\\multicolumn{3}",
                                    "Dataset & $\\tau$"]):
            continue
        
        # Handle section headers (multicolumn{17})
        if "\\multicolumn{17}" in line:
            # Extract section title
            match = re.search(r'\\textbf\{([^}]+)\}', line)
            if match:
                title = match.group(1)
                out_lines.append(f"    \\multicolumn{{10}}{{c}}{{\\textbf{{{title}}}}} \\\\")
                out_lines.append("    \\midrule")
            continue
        
        # Accumulate row content
        current_row += " " + line.strip()
        
        # Check if row is complete (ends with \\)
        if "\\\\" not in current_row:
            continue
        
        # Process complete row
        row_content = current_row.replace("\\\\", "").strip()
        current_row = ""
        
        if not row_content:
            continue
            
        # Split by &
        cells = [c.strip() for c in row_content.split("&")]
        
        if len(cells) != 17:
            continue
        
        # Build new row
        new_cells = []
        
        # Parameters (keep as-is)
        for i in PARAM_COLS:
            new_cells.append(cells[i])
        
        # Baselines (keep as-is, but clean formatting)
        for i in BASELINE_COLS:
            cell = cells[i].strip()
            # Keep the value but remove bold/underline for consistency
            clean = re.sub(r'\\mathbf\{([^}]*)\}', r'\1', cell)
            clean = re.sub(r'\\underline\{([^}]*)\}', r'\1', clean)
            new_cells.append(clean)
        
        # Averaged families
        for family_indices in [FEATURE_COLS, TARGET_COLS, OPTIM_COLS]:
            mean_val, mean_std = average_cells(cells, family_indices)
            if mean_val is not None:
                if mean_val < 100:  # Likely percentage or small RMSE
                    new_cells.append(f"${mean_val:.1f} \\pm {mean_std:.1f}$")
                else:
                    new_cells.append(f"${mean_val:.0f} \\pm {mean_std:.0f}$")
            else:
                new_cells.append("--")
        
        out_lines.append("    " + " & ".join(new_cells) + " \\\\")
        rows_processed += 1

    # Footer
    out_lines.extend([
        "    \\bottomrule",
        "    \\end{tabular}",
        "    }",
        "\\end{table*}",
    ])

    Path(tex_out).write_text("\n".join(out_lines))
    print(f" Written {tex_out} with {rows_processed} data rows")


def plot_ablation_results_sweep(
    results: Dict[str, Dict[Tuple[float, int], List[float]]],
    tau_withins: Tuple[float, ...] = (2.0, 5.0, 10.0),
    Bs: Tuple[int, ...] = (500, 1000, 2000),
    metric_name: str = 'RMSE',
    save_path: Optional[str] = None
) -> None:
    """
    Create grid visualization of ablation results.
    Rows = tau_within values, Columns = Budget values
    
    Args:
        results: Dict from run_budget_allocation_ablation_sweep
        tau_withins: tau_within values tested
        Bs: Budgets tested
        metric_name: Name of metric for y-axis label
        save_path: Optional path to save figure
    """
    methods = list(results.keys())
    n_methods = len(methods)
    n_taus = len(tau_withins)
    n_budgets = len(Bs)
    
    # Create grid: rows=tau_within, cols=budget
    fig, axes = plt.subplots(n_taus, n_budgets, 
                             figsize=(5 * n_budgets, 4 * n_taus),
                             squeeze=False)
    
    colors = plt.cm.Set2(np.linspace(0, 1, n_methods))
    
    for i, tau in enumerate(tau_withins):
        for j, B in enumerate(Bs):
            ax = axes[i, j]
            
            means = []
            stds = []
            
            for method in methods:
                vals = [v for v in results[method][(tau, B)] if not np.isnan(v)]
                if vals:
                    means.append(np.mean(vals))
                    stds.append(np.std(vals))
                else:
                    means.append(0)
                    stds.append(0)
            
            x = np.arange(n_methods)
            bars = ax.bar(x, means, yerr=stds, capsize=3, color=colors, 
                         edgecolor='black', alpha=0.8)
            
            # Set title for each subplot
            ax.set_title(f'τ_within={tau}, B={B}', fontsize=12, fontweight='bold')
            
            # Labels
            if i == n_taus - 1:  # Bottom row
                ax.set_xlabel('Allocation Method', fontsize=10)
                ax.set_xticks(x)
                ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
            else:
                ax.set_xticks(x)
                ax.set_xticklabels([])
            
            if j == 0:  # Left column
                ax.set_ylabel(f'{metric_name}', fontsize=10)
            
            ax.grid(axis='y', alpha=0.3)
            
            # Mark best method with red border
            if means:
                best_idx = np.argmin(means)
                bars[best_idx].set_edgecolor('red')
                bars[best_idx].set_linewidth(2.5)
            
            # Add value labels on bars for best method
            if means:
                best_val = means[best_idx]
                ax.text(best_idx, best_val, f'{best_val:.1f}', 
                       ha='center', va='bottom', fontweight='bold', color='red')
    
    # Overall title
    fig.suptitle(f'Budget Allocation Ablation Study: {metric_name} across τ_within and Budget',
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()


def plot_ablation_heatmaps(
    results: Dict[str, Dict[Tuple[float, int], List[float]]],
    tau_withins: Tuple[float, ...] = (2.0, 5.0, 10.0),
    Bs: Tuple[int, ...] = (500, 1000, 2000),
    metric_name: str = 'MSE',
    save_path: Optional[str] = None
) -> None:
    """
    Create heatmap visualization showing performance of each method across (tau, B).
    One heatmap per allocation method.
    
    Args:
        results: Dict from run_budget_allocation_ablation_sweep
        tau_withins: tau_within values tested
        Bs: Budgets tested
        metric_name: Name of metric
        save_path: Optional path to save figure
    """
    methods = list(results.keys())
    n_methods = len(methods)
    
    # Calculate grid dimensions
    n_cols = min(3, n_methods)
    n_rows = (n_methods + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, 
                             figsize=(5 * n_cols, 4 * n_rows),
                             squeeze=False)
    
    for idx, method in enumerate(methods):
        i = idx // n_cols
        j = idx % n_cols
        ax = axes[i, j]
        
        # Build matrix: rows=tau, cols=B
        matrix = np.zeros((len(tau_withins), len(Bs)))
        
        for ti, tau in enumerate(tau_withins):
            for bi, B in enumerate(Bs):
                vals = [v for v in results[method][(tau, B)] if not np.isnan(v)]
                if vals:
                    matrix[ti, bi] = np.mean(vals)
                else:
                    matrix[ti, bi] = np.nan
        
        # Create heatmap
        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(Bs)))
        ax.set_yticks(np.arange(len(tau_withins)))
        ax.set_xticklabels(Bs)
        ax.set_yticklabels(tau_withins)
        
        ax.set_xlabel('Budget B', fontsize=10)
        ax.set_ylabel('τ_within', fontsize=10)
        ax.set_title(method, fontsize=11, fontweight='bold')
        
        # Add text annotations
        for ti in range(len(tau_withins)):
            for bi in range(len(Bs)):
                val = matrix[ti, bi]
                if not np.isnan(val):
                    text = ax.text(bi, ti, f'{val:.1f}',
                                  ha="center", va="center", color="black", fontsize=9)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label=metric_name)
    
    # Hide unused subplots
    for idx in range(n_methods, n_rows * n_cols):
        i = idx // n_cols
        j = idx % n_cols
        axes[i, j].axis('off')
    
    fig.suptitle(f'Budget Allocation Ablation: {metric_name} Heatmaps',
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved heatmap figure to {save_path}")
    
    plt.show()

def image_results_to_latex_unified(
    results: Dict[str, Dict[Tuple[int, int], List[float]]],
    methods: List[str], 
    Ts: Tuple[int, ...], 
    Bs: Tuple[int, ...],
    dataset_name: str = "Dataset",
    caption: str = "Classification Accuracy",
    label: str = "tab:image_results",
    show_std: bool = True,
) -> Tuple[str, pd.DataFrame]:
    """
    Generate unified LaTeX table for image classification results with MST variants.
    """
    rows = []
    for T in Ts:
        for B in Bs:
            key = (T, B)
            row = {"T": T, "B": B}
            
            for m in methods:
                if m in results and key in results[m] and len(results[m][key]) > 0:
                    arr = np.array(results[m][key])
                    valid_arr = arr[~np.isnan(arr)]
                    if len(valid_arr) > 0:
                        row[f"{m}_mean"] = np.mean(valid_arr) * 100
                        row[f"{m}_std"] = np.std(valid_arr) * 100
                    else:
                        row[f"{m}_mean"] = np.nan
                        row[f"{m}_std"] = np.nan
                else:
                    row[f"{m}_mean"] = np.nan
                    row[f"{m}_std"] = np.nan
            
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    n_methods = len(methods)
    col_spec = "cc|" + "c" * n_methods
    
    latex_lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{" + caption + f" ({dataset_name})" + "}",
        r"\label{" + label + "}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
    ]
    
    # Header
    header1 = [
        r"\multicolumn{2}{c|}{Params}",
        r"\multicolumn{" + str(n_methods) + r"}{c}{Methods}"
    ]
    latex_lines.append(" & ".join(header1) + r" \\")
    
    method_headers = [m.replace("_", r"\_").replace("-", r"-") for m in methods]
    header2 = ["$T$", "$B$"] + method_headers
    latex_lines.append(" & ".join(header2) + r" \\")
    latex_lines.append(r"\midrule")
    
    prev_T = None
    for row in rows:
        if prev_T is not None and row["T"] != prev_T:
            latex_lines.append(r"\midrule")
        prev_T = row["T"]
        
        # Find best and second best
        means = [(m, row[f"{m}_mean"]) for m in methods if not np.isnan(row[f"{m}_mean"])]
        sorted_means = sorted(means, key=lambda x: -x[1])  # Higher is better for accuracy
        best = sorted_means[0][0] if len(sorted_means) > 0 else None
        second = sorted_means[1][0] if len(sorted_means) > 1 else None
        
        cells = [str(row["T"]), str(row["B"])]
        
        for m in methods:
            mean_val = row[f"{m}_mean"]
            std_val = row[f"{m}_std"]
            if np.isnan(mean_val):
                cells.append("—")
            else:
                if show_std:
                    val_str = f"{mean_val:.1f} \\pm {std_val:.1f}"
                else:
                    val_str = f"{mean_val:.1f}"
                
                if m == best:
                    cells.append(f"$\\mathbf{{{val_str}}}$")
                elif m == second:
                    cells.append(f"$\\underline{{{val_str}}}$")
                else:
                    cells.append(f"${val_str}$")
        
        latex_lines.append(" & ".join(cells) + r" \\")
    
    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table}",
    ])
    
    return "\n".join(latex_lines), df


def visualize_image_results_table(
    results: Dict[str, Dict[Tuple[int, int], List[float]]],
    methods: List[str],
    Ts: Tuple[int, ...],
    Bs: Tuple[int, ...],
    metric_name: str = "Accuracy",
) -> pd.DataFrame:
    """
    Create summary table for image classification results.
    """
    rows = []
    
    for T in Ts:
        for B in Bs:
            key = (T, B)
            row = {'T': T, 'B': B}
            
            for method in methods:
                if key in results[method] and len(results[method][key]) > 0:
                    vals = np.array(results[method][key])
                    valid_vals = vals[~np.isnan(vals)]
                    
                    if len(valid_vals) > 0:
                        mean_val = np.mean(valid_vals) * 100
                        std_val = np.std(valid_vals) * 100
                        row[method] = f"{mean_val:.2f} ± {std_val:.2f}"
                    else:
                        row[method] = "N/A"
                else:
                    row[method] = "N/A"
            
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Reorder columns
    cols = ['T', 'B'] + methods
    df = df[cols]
    
    print(f"\n{'='*80}")
    print(f"SUMMARY TABLE: {metric_name} (%)")
    print(f"{'='*80}")
    print(df.to_string(index=False))
    print()
    
    return df

def visualize_image_results_bars(
    results: Dict[str, Dict[Tuple[int, int], List[float]]],
    methods: List[str],
    Ts: Tuple[int, ...],
    Bs: Tuple[int, ...],
    colors: Optional[Dict[str, str]] = None,
    figsize: Tuple[int, int] = (16, 6),
    baseline: str = "Indiv",
    dataset_name: str = "Dataset",
):
    """
    Visualize image classification results as grouped bar charts showing relative performance.
    
    Args:
        results: Dict[method, Dict[(T, B), List[float]]] - accuracy results
        methods: List of method names
        Ts: Tuple of T values (number of tasks)
        Bs: Tuple of B values (budget)
        colors: Dictionary mapping methods to colors (optional)
        figsize: Figure size
        baseline: Method to use as baseline for comparison
        dataset_name: Name of dataset for title
    
    Returns:
        fig, axes: Matplotlib figure and axes
    """
    if colors is None:
        colors = {
            "Star": "#3498db",
            "CTL-RandTree": "#9b59b6",
            "CTL-MST-feature": "#e74c3c",
            "CTL-MST-target": "#e67e22",
            "CTL-MST-gradient": "#f39c12",
            "CTL-MST-model": "#16a085",
            "CTL-MST-kl": "#27ae60",
            "CTL-MST-wasserstein": "#2980b9",
            "CTL-MST-mmd_feature": "#8e44ad",
            "CTL-MST-mean_cov_feature": "#c0392b",
            "CTL-MST-js_target": "#d35400",
            "CTL-MST-cka": "#2c3e50",
        }
    
    n_configs = len(Ts)
    fig, axes = plt.subplots(1, n_configs, figsize=figsize, sharey=True)
    
    if n_configs == 1:
        axes = [axes]
    
    for t_idx, T in enumerate(Ts):
        ax = axes[t_idx]
        
        # Aggregate across budgets for this T
        method_improvements = []
        method_stds = []
        method_labels = []
        
        for method in methods:
            if method == baseline:
                continue
            
            # Collect improvements across all B values
            improvements = []
            for B in Bs:
                key = (T, B)
                
                baseline_vals = np.array(results[baseline].get(key, []))
                method_vals = np.array(results[method].get(key, []))
                
                # Filter out NaNs
                baseline_vals = baseline_vals[~np.isnan(baseline_vals)]
                method_vals = method_vals[~np.isnan(method_vals)]
                
                if len(baseline_vals) > 0 and len(method_vals) > 0:
                    # Compute percentage point improvement (higher accuracy = better)
                    # Convert to percentage points
                    baseline_mean = baseline_vals.mean() * 100
                    method_mean = method_vals.mean() * 100
                    improvement = method_mean - baseline_mean
                    improvements.append(improvement)
            
            if len(improvements) > 0:
                method_improvements.append(np.mean(improvements))
                method_stds.append(np.std(improvements))
                label = method.replace('CTL-MST-', 'MST-').replace('CTL-', '')
                method_labels.append(label)
            else:
                method_improvements.append(0)
                method_stds.append(0)
                label = method.replace('CTL-MST-', 'MST-').replace('CTL-', '')
                method_labels.append(label)
        
        # Create bar chart
        x = np.arange(len(method_labels))
        
        # Determine color for each bar
        bar_colors = []
        for method in methods:
            if method == baseline:
                continue
            bar_colors.append(colors.get(method, 'gray'))
        
        bars = ax.bar(
            x, method_improvements, yerr=method_stds,
            capsize=5, alpha=0.8, color=bar_colors
        )
        
        # Color bars based on positive/negative with gradient
        if len(method_improvements) > 0 and max(method_improvements) > min(method_improvements):
            norm = plt.Normalize(
                vmin=min(method_improvements) - 0.5, 
                vmax=max(method_improvements) + 0.5
            )
            cmap = plt.cm.RdYlGn
            
            for bar, mean in zip(bars, method_improvements):
                bar_color = cmap(norm(mean))
                bar.set_color(bar_color)
            
            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, pad=0.02, aspect=30)
            cbar.set_label("Accuracy Gain (pp)", rotation=270, labelpad=15, fontsize=9)
        
        # Add horizontal line at 0
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
        
        # Formatting
        ax.set_xticks(x)
        ax.set_xticklabels(method_labels, rotation=45, ha='right', fontsize=9)
        ax.set_title(f"T = {T} tasks", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        if t_idx == 0:
            ax.set_ylabel(f"Accuracy Gain vs {baseline} (pp)", fontsize=10)
        
        # Add value labels on bars (only for significant improvements)
        for i, (mean, std) in enumerate(zip(method_improvements, method_stds)):
            if abs(mean) > 0.3:  # Only label if improvement > 0.3pp
                y_offset = std + 0.2 if mean > 0 else -std - 0.2
                ax.text(
                    i, mean + y_offset,
                    f"{mean:+.1f}",
                    ha='center', 
                    va='bottom' if mean > 0 else 'top',
                    fontsize=8, 
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none')
                )
    
    fig.suptitle(
        f"{dataset_name}: Average Performance vs {baseline}",
        fontsize=14, fontweight='bold', y=0.98
    )
    plt.tight_layout()
    
    return fig, axes

def visualize_image_results_heatmap(
    results: Dict[str, Dict[Tuple[int, int], List[float]]],
    methods: List[str],
    Ts: Tuple[int, ...],
    Bs: Tuple[int, ...],
    baseline: str = "Indiv",
    dataset_name: str = "Dataset",
    figsize: Tuple[int, int] = (12, 8),
):
    """
    Visualize results as a heatmap of improvements over baseline.
    """
    method_list = [m for m in methods if m != baseline]
    configs = [(T, B) for T in Ts for B in Bs]
    
    improvement_matrix = np.zeros((len(method_list), len(configs)))
    
    for i, method in enumerate(method_list):
        for j, (T, B) in enumerate(configs):
            baseline_vals = np.array(results[baseline].get((T, B), []))
            method_vals = np.array(results[method].get((T, B), []))
            
            baseline_vals = baseline_vals[~np.isnan(baseline_vals)]
            method_vals = method_vals[~np.isnan(method_vals)]
            
            if len(baseline_vals) > 0 and len(method_vals) > 0:
                improvement = (method_vals.mean() - baseline_vals.mean()) * 100
                improvement_matrix[i, j] = improvement
            else:
                improvement_matrix[i, j] = np.nan
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)
    
    method_labels = [m.replace('CTL-MST-', 'MST-').replace('CTL-', '') for m in method_list]
    config_labels = [f"T={T}, B={B}" for T, B in configs]
    
    sns.heatmap(
        improvement_matrix,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn',
        center=0,
        xticklabels=config_labels,
        yticklabels=method_labels,
        cbar_kws={'label': 'Accuracy Gain (pp)'},
        ax=ax,
        linewidths=0.5,
        linecolor='gray'
    )
    
    ax.set_title(f"{dataset_name}: Performance vs {baseline} Baseline", fontsize=14, fontweight='bold')
    ax.set_xlabel("Configuration", fontsize=11)
    ax.set_ylabel("Method", fontsize=11)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return fig, ax