"""
Gradient flow simulator for geometric learning dynamics.
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import trange

class GradientFlowSimulator:
    """Simulates gradient flow dynamics with diagnostics tracking."""
    
    def __init__(self, dataset, network, lr=0.1, use_bayesian=False, 
                 alpha=0.1, lam=1.0):
        self.dataset = dataset
        self.net = network
        self.lr = lr
        self.use_bayesian = use_bayesian
        
        # Import here to avoid circular imports
        from ..models import BayesianRLayerNorm
        
        if use_bayesian:
            self.bn = BayesianRLayerNorm(dataset.dim, alpha, lam)
        else:
            self.bn = None
        
        self.history = {
            'step': [],
            'norm_perp': [],
            'cos_delta': [],
            'a_alpha': [],
            'L12': [],
        }
    
    def compute_gradient(self, X, y_onehot):
        """Compute average gradient signals."""
        M, dim = self.net.M, self.net.dim
        device = X.device
        
        gamma_sum = torch.zeros(M, dim, device=device)
        grad_w2_sum = torch.zeros(2, M, device=device)
        w1, w2 = self.net.w1, self.net.w2
        
        for i in range(len(X)):
            x = X[i]
            yh = y_onehot[i]
            h, g = self.net.forward(x)
            logits = w2 @ h
            probs = F.softmax(logits, dim=0)
            delta = probs - yh
            dot = delta @ w2
            gamma_sum += g.unsqueeze(1) * dot.unsqueeze(1) * x.unsqueeze(0)
            grad_w2_sum += delta.unsqueeze(1) * h.unsqueeze(0)
        
        N = len(X)
        gamma_avg = gamma_sum / N
        grad_w2_avg = grad_w2_sum / N
        
        # Compute diagnostics
        with torch.no_grad():
            x_center = X.mean(dim=0)
            if self.dataset.manifold_type == 'sphere':
                x_center = x_center / torch.norm(x_center)
            
            proj = (w1 @ x_center).unsqueeze(1) * x_center.unsqueeze(0)
            w_perp = w1 - proj
            norm_perp = torch.norm(w_perp, dim=1)
            
            gamma_norm = torch.norm(gamma_avg, dim=1) + 1e-8
            cos_delta = (w1 * gamma_avg).sum(dim=1) / (torch.norm(w1, dim=1) * gamma_norm)
            
            w1_norm = torch.norm(w1, dim=1)
            w2_norm = torch.norm(w2, dim=0)
            a_alpha = w1_norm * w2_norm
            
            if M >= 2:
                L12 = w1[0,0]*w1[1,1] - w1[0,1]*w1[1,0]
            else:
                L12 = torch.tensor(0.0)
        
        diag = {
            'norm_perp': norm_perp.cpu().numpy(),
            'cos_delta': cos_delta.cpu().numpy(),
            'a_alpha': a_alpha.cpu().numpy(),
            'L12': L12.cpu().numpy(),
        }
        
        return gamma_avg, grad_w2_avg, diag
    
    def step(self, X, y_onehot):
        """Perform one gradient step."""
        gamma_avg, grad_w2_avg, diag = self.compute_gradient(X, y_onehot)
        
        w2_norms = torch.norm(self.net.w2, dim=0)
        update_w1 = self.lr * w2_norms.unsqueeze(1) * gamma_avg
        self.net.w1 += update_w1
        self.net.w2 -= self.lr * grad_w2_avg
        
        return diag
    
    def run(self, steps, record_every=10):
        """Run simulation for given number of steps."""
        X = self.dataset.X
        y_onehot = self.dataset.y_onehot
        
        for step in trange(steps, desc='Simulating'):
            diag = self.step(X, y_onehot)
            
            if step % record_every == 0:
                self.history['step'].append(step)
                self.history['norm_perp'].append(diag['norm_perp'])
                self.history['cos_delta'].append(diag['cos_delta'])
                self.history['a_alpha'].append(diag['a_alpha'])
                self.history['L12'].append(diag['L12'])
        
        # Convert lists to arrays
        for k in self.history:
            if k != 'step':
                self.history[k] = np.array(self.history[k])
