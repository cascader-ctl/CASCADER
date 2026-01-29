# CTL: Cascaded Transfer Learning

CTL (**Cascaded Transfer Learning**) is a lightweight research framework for **many-task learning under a global training budget**.  
Instead of training tasks independently or relying on expensive joint multi-task learning, CTL organizes tasks into a **directed tree** and performs **cascade transfer**: models are trained at a root task and progressively fine-tuned along the tree.

---

## Key ideas

- **Tree-structured transfer learning**  
  Tasks are connected by directed edges; each task inherits parameters from its parent and is fine-tuned locally.

- **Global budget constraint**  
  A fixed budget \(B\) of gradient steps is shared across all tasks, making transfer decisions meaningful.


---

## Features

### Models
- `LinearModel` — exact match with theory.
- `SimpleMLP` — minimal non-linear extension.
- `ResidualMLP` — near-linear model preserving alignment structure.

### Training
- Full-batch SGD training.
- Model cloning for cascade transfer.
- Cost tracking (gradient steps, wall-clock time).

### Hyperparameter Optimization
- Optuna-based optimizer for cascade-level parameters:
  - alignment learning rate `eta`
  - seed budget fraction
- Clean train/validation/test protocol (no leakage).

---

## Quickstart

The easiest way to run experiments is via the provided notebook: `expe-alignment-weave.ipynb`.
It walks through:

1. data loading and normalization ;
2. tree constructions ; 
3. cascade training and baselines ; 
4. hyperparameter optimization ;
5. final test evaluation.

## Citation

If you use the CTL in your work, please cite the corresponding [paper]():

```bibtex
@article{anonymous_2026_CTL,
author    = {Anonymous authors},
title     = {Cascaded Transfer: Learning Many Tasks under Budget Constraints},
year      = {2026}
}
```

## License

This project is licensed under the GPL License - see the LICENSE file for details.
