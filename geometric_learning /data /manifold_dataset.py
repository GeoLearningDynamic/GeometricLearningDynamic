"""
Base class for manifold datasets.
"""

import torch
import numpy as np
from abc import ABC, abstractmethod

class ManifoldDataset(ABC):
    """Abstract base class for datasets on Riemannian manifolds."""
    
    def __init__(self, n_samples, dim, noise=0.0, seed=42):
        self.n_samples = n_samples
        self.dim = dim
        self.noise = noise
        self.seed = seed
        self.X = None
        self.y = None
        self.y_onehot = None
        self.generate()
    
    @abstractmethod
    def generate(self):
        """Generate points on the manifold."""
        pass
    
    @abstractmethod
    def tangent_project(self, x, v):
        """Project vector v onto tangent space at x."""
        pass
    
    def centroid(self, weights):
        """Weighted centroid of all points."""
        weighted_sum = (weights.unsqueeze(1) * self.X).sum(dim=0)
        norm = torch.norm(weighted_sum)
        if norm > 1e-12:
            return weighted_sum / norm
        return torch.randn(self.dim) * 0.1
