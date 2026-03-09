"""
Geometric Learning Dynamics Framework
=====================================
A mathematical framework for understanding geometric learning in neural networks.

Modules:
- data: Manifold dataset generators
- models: Neural network components (gated networks, Bayesian normalization)
- simulator: Gradient flow simulation
- utils: Metrics, plotting, and statistics
"""

from .data import ManifoldDataset
from .models import GatedNetwork, BayesianRLayerNorm, psi
from .simulator import GradientFlowSimulator
from .utils import metrics, plotting, stats

__version__ = "1.0.0"
__author__ = "Mohsen Mostafa"
