"""
Gated network with fixed prototypes.
"""

import torch

class GatedNetwork:
    """Single hidden layer with gating based on distance to prototypes."""
    
    def __init__(self, M, dim, sigma_g=0.5, seed=42):
        torch.manual_seed(seed)
        self.M = M
        self.dim = dim
        self.sigma_g = sigma_g
        
        # Incoming weights (initialized on sphere)
        self.w1 = torch.randn(M, dim)
        self.w1 /= torch.norm(self.w1, dim=1, keepdim=True)
        
        # Outgoing weights (2D output)
        self.w2 = torch.randn(2, M) * 0.1
        
        # Fixed prototypes for gating
        self.prototypes = torch.randn(M, dim)
        self.prototypes /= torch.norm(self.prototypes, dim=1, keepdim=True)
    
    def gating(self, x):
        """Compute g_α(x) = exp(-||x - c_α||²/(2σ²))"""
        diff = x.unsqueeze(0) - self.prototypes
        sq_dist = (diff ** 2).sum(dim=1)
        return torch.exp(-sq_dist / (2 * self.sigma_g ** 2))
    
    def forward(self, x):
        """Compute hidden activations h = g_α * (w1_α·x)."""
        g = self.gating(x)
        h = g * (x @ self.w1.T)
        return h, g
