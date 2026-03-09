"""
Bayesian R‑LayerNorm with ψ-function.
"""

import torch

def psi(t):
    """Stable ψ-function: ψ(t) = log(1+t) - t/(1+t)"""
    return torch.log1p(t) - t / (1 + t)

class BayesianRLayerNorm:
    """Bayesian R‑LayerNorm with uncertainty quantification."""
    
    def __init__(self, dim, alpha=0.1, lam=1.0, momentum=0.9):
        self.dim = dim
        self.alpha = alpha
        self.lam = lam
        self.momentum = momentum
        self.running_E = torch.zeros(1)
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
    
    def __call__(self, x, update=True):
        mu = x.mean()
        sigma = x.std(unbiased=False) + 1e-5
        E = x.var(unbiased=False)
        
        if update:
            with torch.no_grad():
                self.running_E = self.momentum * self.running_E + (1 - self.momentum) * E
        
        E_used = self.running_E if not update else E
        sigma_eff = sigma * torch.exp(self.alpha * psi(self.lam * E_used))
        x_norm = (x - mu) / sigma_eff
        
        return self.gamma * x_norm + self.beta, sigma_eff
