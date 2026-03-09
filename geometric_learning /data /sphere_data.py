"""
Sphere manifold dataset.
"""

import torch
import torch.nn.functional as F
from .manifold_dataset import ManifoldDataset

class SphereDataset(ManifoldDataset):
    """Uniform points on sphere S^{dim-1}."""
    
    def generate(self):
        torch.manual_seed(self.seed)
        z = torch.randn(self.n_samples, self.dim)
        self.X = z / torch.norm(z, dim=1, keepdim=True)
        self.y = (self.X[:, 0] > 0).long()
        self.y_onehot = F.one_hot(self.y, num_classes=2).float()
    
    def tangent_project(self, x, v):
        """Project onto tangent space: orthogonal to x."""
        return v - (v @ x) * x
