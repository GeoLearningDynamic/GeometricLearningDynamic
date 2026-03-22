# Geometric Foundations of Racing Dynamics – Experimental Validation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)


This repository contains the code and data for the experimental validation of the four theorems presented in the paper *Geometric Foundations of Racing Dynamics: How Gradient Descent Adapts Network Capacity on Data Manifolds with Application to Bayesian R‑LayerNorm*.

Code for **"A Mathematical Framework for Geometric Learning Dynamics in Neural Networks"** (Mostafa, 2026).

## 📋 Overview

This repository implements the complete mathematical framework proving four fundamental theorems about geometric learning in neural networks:

- **Theorem 1 (Manifold Alignment):** Neuron weights converge exponentially to the data manifold's tangent space
- **Theorem 2 (Angular Momentum Conservation):** For rotation-equivariant tasks, angular momentum tensor is conserved
- **Theorem 3 (Winning Ticket Concentration):** High-norm neurons follow von Mises-Fisher distribution
- **Theorem 4 (Stable Racing Dynamics):** Bayesian R‑LayerNorm provides noise-adaptive growth suppression

The code implements:

- Synthetic sphere dataset (exact tangent space) to test manifold alignment, angular momentum conservation, and winning ticket concentration.
- Real‑world MNIST and Fashion‑MNIST datasets with PCA approximation to test winning ticket concentration and noise gating with Bayesian R‑LayerNorm.
- Full‑batch gradient flow for synthetic experiments and SGD with momentum for real data.

All experiments were run under severe computational constraints (free Kaggle P100 GPU, 16 GB VRAM). The results are proof‑of‑concept and support the theoretical claims.

## 🚀 Quick Start

### Installation

```python
git clone https://github.com/GeoLearningDynamic/GeometricLearningDynamic.git
cd geometric-learning-dynamics
pip install -r requirements.txt# GeometricLearningDynamic
```
Run Main Experiment
```python
# Run with 5 seeds, 3000 steps
python experiments/run_main_experiment.py --seeds 5 --steps 3000

# Results saved to results/main_experiment/
```
📊 Results

After running the main experiment, you'll get:
Theorem	Finding	Figure
T1	Exponential decay of |w⊥|	theorem1_alignment.png
T2	Angular momentum conservation	theorem2_angular_momentum.png
T3	vMF concentration of winners	theorem3_concentration.png
T4	Noise gating effect	theorem4_noise_gating.png

📝 Citation
```python
@article{Mostafa2026Geometric,
  author  = {Mohsen Mostafa},
  title   = {Geometric Foundations of Racing Dynamics: How Gradient Descent Adapts Network Capacity on Data Manifolds with Application to Bayesian R‑LayerNorm},
  journal = {Under Review},
  year    = {2026}
}
```

Contact

For questions or issues, please open an issue on GitHub or contact mohsen.mostafa.ai@outlook.com.


