---
layout: default
---

---
</div>
<div class="hero-right">
<img src="assets/logo.png" alt="CASCADER logo" class="logo" />
</div>
</div>


---


## Abstract


Transfer learning under distribution shift remains a central challenge in modern machine learning, particularly when labeled data are scarce or unavailable in the target domain. **CASCADER** introduces a cascade transfer learning framework that progressively adapts deep energy-based representations across domains. By structuring adaptation as a sequence of intermediate transformations, CASCADER enables stable, interpretable, and data-efficient transfer, outperforming direct domain adaptation strategies across a wide range of benchmarks.


---


## Method Overview


<div class="figure-block">
<img src="assets/overview.png" alt="CASCADER method overview" />
<p class="figure-caption">
Overview of the CASCADER framework. Domain adaptation is performed through a cascade of intermediate representations, progressively reducing domain mismatch.
</p>
</div>


---


## Key Contributions


<ul class="contributions">
<li>A novel cascade-based formulation for transfer learning under domain shift.</li>
<li>Energy-based representations enabling stable intermediate adaptation.</li>
<li>Theoretical insights into error propagation across cascade stages.</li>
<li>Strong empirical performance across vision and time-series benchmarks.</li>
</ul>


---


## Experimental Results


<div class="figure-grid">
<div class="figure-block">
<img src="assets/results_1.png" alt="Benchmark results" />
<p class="figure-caption">Performance comparison across benchmark datasets.</p>
</div>
<div class="figure-block">
<img src="assets/results_2.png" alt="Ablation study" />
<p class="figure-caption">Ablation study highlighting the role of cascade depth.</p>
</div>
</div>


---


## Reproducibility


The full implementation of CASCADER, including training scripts, configuration files, and evaluation protocols, is available in this repository. All experiments are fully reproducible using the provided configuration files.


---


## Citation


If you use CASCADER in your research, please cite:


```bibtex
@article{cascader2026,
title={CASCADER: Cascade Semi-supervised Cross-domain Adaptation for Deep Energy-based Representations},
author={Anonymous},
journal={ICML},
year={2026}
}
