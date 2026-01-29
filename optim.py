from data import *
from graph import build_mst_tree, cascade_train_tree, select_seed_node
import optuna
import numpy as np
from typing import Callable, Dict, Any
from trainer import predict_node

class OptunaCascadeOptimizer:
    """
    Hyperparameter optimizer for alignment-based cascade learning.

    - Supports model-specific parameters (e.g., hidden, eps).
    - Tree rebuilt per trial.
    - No shared-state mutation.
    """

    def __init__(
        self,
        eval_fn: Callable[[Dict[Any, float]], float],
        optim_space: Dict[str, tuple],
        n_trials: int = 40,
        seed: int = 0,
        verbose: bool = True,
    ):
        self.eval_fn = eval_fn
        self.optim_space = optim_space
        self.n_trials = n_trials
        self.seed = seed
        self.verbose = verbose

    def _suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters for the current trial.
        """
        params = {}
        for name, spec in self.optim_space.items():
            kind = spec[0]
            if kind == "loguniform":
                _, low, high = spec
                params[name] = trial.suggest_float(name, low, high, log=True)
            elif kind == "uniform":
                _, low, high = spec
                params[name] = trial.suggest_float(name, low, high)
            elif kind == "int":
                _, low, high = spec
                params[name] = trial.suggest_int(name, low, high)
            else:
                raise ValueError(f"Unknown search type: {kind}")
        return params

    def _objective(
        self,
        trial,
        *,
        build_tree_fn,
        cascade_train_fn,
        evaluate_models_fn,
        base_tree_kwargs,
        base_cascade_kwargs,
        B,
        model_class: str,
    ):
        """
        Objective function for Optuna to minimize.
        """
        opt_params = self._suggest_params(trial)

        eta = opt_params["eta"]
        b_proto_frac = opt_params["b_proto_frac"]
        ball_top_q = opt_params.get("ball_top_q", None)

        b_seed = max(1, int(b_proto_frac * B))

        tree_kwargs = dict(base_tree_kwargs)
        tree_kwargs["eta"] = eta
        if ball_top_q is not None:
            tree_kwargs["ball_top_q"] = ball_top_q

        cascade_kwargs = dict(base_cascade_kwargs)
        cascade_kwargs["eta"] = eta
        cascade_kwargs["b_seed"] = b_seed
        cascade_kwargs["model_class"] = model_class

        if model_class in ["mlp", "residual"]:
            cascade_kwargs["hidden"] = opt_params.get("hidden", 32)
        if model_class == "residual":
            cascade_kwargs["eps"] = opt_params.get("eps", 0.1)

        # run pipeline
        root, parent, edges, _ = build_tree_fn(**tree_kwargs)
        models, _ = cascade_train_fn(tree_edges=edges, root=root, **cascade_kwargs)

        # evaluate models
        mses = evaluate_models_fn(models)
        loss = self.eval_fn(mses)

        return loss

    def run(
        self,
        *,
        build_tree_fn: Callable[..., Any],
        cascade_train_fn: Callable[..., Dict],
        evaluate_models_fn: Callable[[Dict[Any, Any]], Dict[Any, float]],
        base_tree_kwargs: Dict[str, Any],
        base_cascade_kwargs: Dict[str, Any],
        B: int,
        model_class: str,
    ) -> Dict[str, Any]:
        """
        Run optimization loop using Optuna.

        Args:
            build_tree_fn: function to build the alignment tree
            cascade_train_fn: function to train the cascade
            evaluate_models_fn: function to evaluate models
            base_tree_kwargs: arguments for the tree-building function
            base_cascade_kwargs: arguments for the cascade training function
            B: total budget
            model_class: the model class to use (e.g., "linear", "mlp", "residual")
        """
        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)

        study.optimize(
            lambda trial: self._objective(
                trial,
                build_tree_fn=build_tree_fn,
                cascade_train_fn=cascade_train_fn,
                evaluate_models_fn=evaluate_models_fn,
                base_tree_kwargs=base_tree_kwargs,
                base_cascade_kwargs=base_cascade_kwargs,
                B=B,
                model_class=model_class,
            ),
            n_trials=self.n_trials,
        )

        self.best_params = study.best_params
        self.best_value = study.best_value

        if self.verbose:
            print(
                f"[Optuna] Best params: {self.best_params} "
                f"(val loss={self.best_value:.4f})"
            )

        return self.best_params
    
def optimize_synthetic_dataset(X_train, y_train, X_test, y_test, nodes, VAL_SIZE, DEVICE, model_class, B, optim_space, n_trials=40, k_neighbors=10, SEED=0):
    """
    Optimize the cascade learning procedure for synthetic datasets (homogeneous and clustered).

    Args:
        X_train: Dictionary of training input data for each node.
        y_train: Dictionary of training target data for each node.
        X_test: Dictionary of test input data for each node.
        y_test: Dictionary of test target data for each node.
        nodes: List of node identifiers.
        VAL_SIZE: Number of validation samples per node.
        DEVICE: Torch device (e.g., "cpu" or "cuda").
        model_class: Model class to use (e.g., "linear", "mlp", "residual").
        B: Total budget for training.
        optim_space: Hyperparameter search space for Optuna.
        n_trials: Number of optimization trials.

    Returns:
        best_params: Best hyperparameters found during optimization.
    """
    # Split training data into training and validation sets
    X_train_n, y_train_n = {}, {}
    X_val_n, y_val_n = {}, {}
    X_test_n = {}
    node_stats = {}

    for v in nodes:
        Xn, yn, stats = normalize_xy(X_train[v], y_train[v])
        node_stats[v] = stats

        X_val_n[v], y_val_n[v] = Xn[:VAL_SIZE], yn[:VAL_SIZE]
        X_train_n[v], y_train_n[v] = Xn[VAL_SIZE:], yn[VAL_SIZE:]

        X_test_n[v] = (X_test[v] - stats["X_mean"]) / stats["X_std"]

    # Select global root
    global_root = select_seed_node(X_train, y_train)

    # Tree configuration
    tree_config = {
        "pruning": "ball_soft",
        "ball_slack_mode": "additive",
        "k_neighbors": k_neighbors,
        "seed": global_root,
    }

    # Cascade configuration
    cascade_config = {
        "B": B,
        "device": DEVICE,
        "model_class": model_class,
    }

    # Define the evaluation function
    def evaluate_on_validation(models):
        rmses = {}
        for v, model in models.items():
            yhat = predict_node(model, X_val_n[v])
            yhat = denormalize_y(yhat, node_stats[v])
            mse, _ = compute_metrics_arrays(y_val_n[v], yhat)
            rmses[v] = np.sqrt(mse)
        return rmses

    # Define the global MSE evaluation metric
    global_mse = lambda mses: float(np.mean(list(mses.values())))

    # Initialize the optimizer
    optimizer = OptunaCascadeOptimizer(
        eval_fn=global_mse,
        optim_space=optim_space,
        n_trials=n_trials,
        seed=SEED,
    )

    # Run the optimization
    print("\n" + "=" * 60)
    print("Running Optimization")
    print("=" * 60)
    best_params = optimizer.run(
        build_tree_fn=build_mst_tree,
        cascade_train_fn=cascade_train_tree,
        evaluate_models_fn=lambda models: evaluate_on_validation(models),
        base_tree_kwargs={
            "V": nodes,
            "y": {v: y_train_n[v].flatten() for v in nodes},
            **tree_config,
        },
        base_cascade_kwargs={
            "X_train_n": X_train_n,
            "y_train_n": y_train_n,
            **cascade_config,
        },
        B=B,
        model_class=model_class,
    )

    return best_params

def evaluate_on_validation(models, X_val_n, y_val_n, node_stats):
    """
    Evaluate models on the validation set and compute RMSE for each node.

    Args:
        models: Dictionary of trained models for each node.
        X_val_n: Dictionary of validation input data for each node.
        y_val_n: Dictionary of validation target data for each node.
        node_stats: Dictionary of normalization statistics for each node.

    Returns:
        rmses: Dictionary of RMSE values for each node.
    """
    rmses = {}
    for v, model in models.items():
        yhat = predict_node(model, X_val_n[v])  
        yhat = denormalize_y(yhat, node_stats[v])  
        mse, _ = compute_metrics_arrays(y_val_n[v], yhat)  
        rmses[v] = np.sqrt(mse)  
    return rmses

def predict_logistic(theta: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Predict probabilities for logistic regression.
    """
    logits = X @ theta
    return 1.0 / (1.0 + np.exp(-logits))

def evaluate_on_validation_logistic(
    thetas: Dict[Any, np.ndarray],
    X_val_n: Dict[Any, np.ndarray],
    y_val_n: Dict[Any, np.ndarray],
) -> Dict[Any, float]:
    """
    Evaluate logistic cascade on validation data.
    Returns per-node error (1 - accuracy).
    """
    errors = {}
    for v, theta in thetas.items():
        probs = predict_logistic(theta, X_val_n[v])
        preds = (probs >= 0.5).astype(int)
        acc = (preds == y_val_n[v]).mean()
        errors[v] = 1.0 - acc
    return errors
