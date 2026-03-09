"""
Publication-quality plotting functions.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

def plot_theorem1(steps, avg_perp, std_perp, out_dir):
    """Plot Theorem 1: Manifold alignment (exponential decay)."""
    plt.figure(figsize=(10, 6))
    plt.semilogy(steps, avg_perp, 'b-', linewidth=2.5, label='Mean')
    plt.fill_between(steps, avg_perp-std_perp, avg_perp+std_perp, 
                     alpha=0.2, color='b')
    plt.xlabel('Gradient Step', fontsize=12)
    plt.ylabel('$\\|w_\\perp\\|$ (log scale)', fontsize=12)
    plt.title('Theorem 1: Manifold Alignment', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'theorem1_alignment.png'), dpi=300)
    plt.close()

def plot_theorem2(steps, L12_mean, L12_std, out_dir):
    """Plot Theorem 2: Angular momentum conservation."""
    plt.figure(figsize=(10, 6))
    plt.plot(steps, L12_mean, 'g-', linewidth=2.5, label='Mean')
    plt.fill_between(steps, L12_mean-L12_std, L12_mean+L12_std, 
                     alpha=0.2, color='g')
    plt.xlabel('Gradient Step', fontsize=12)
    plt.ylabel('Angular Momentum $L_{12}$', fontsize=12)
    plt.title('Theorem 2: Angular Momentum Conservation', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'theorem2_angular_momentum.png'), dpi=300)
    plt.close()

def plot_theorem3(dirs, out_dir):
    """Plot Theorem 3: Winning ticket concentration (heatmap)."""
    if len(dirs) == 0:
        return
    
    plt.figure(figsize=(10, 8))
    plt.hist2d(dirs[:,0], dirs[:,1], bins=40, cmap='hot', density=True)
    plt.colorbar(label='Density')
    plt.xlabel('$u_1$', fontsize=12)
    plt.ylabel('$u_2$', fontsize=12)
    plt.title(f'Theorem 3: Winning Ticket Concentration (n={len(dirs)} neurons)', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'theorem3_concentration.png'), dpi=300)
    plt.close()

def plot_theorem4(steps, nobayes_mean, nobayes_std, bayes_mean, bayes_std, out_dir):
    """Plot Theorem 4: Noise gating effect."""
    plt.figure(figsize=(10, 6))
    plt.plot(steps, nobayes_mean, 'r-', linewidth=2.5, label='Without Bayesian')
    plt.fill_between(steps, nobayes_mean-nobayes_std, nobayes_mean+nobayes_std, 
                     alpha=0.2, color='r')
    plt.plot(steps, bayes_mean, 'b-', linewidth=2.5, label='With Bayesian')
    plt.fill_between(steps, bayes_mean-bayes_std, bayes_mean+bayes_std, 
                     alpha=0.2, color='b')
    plt.xlabel('Gradient Step', fontsize=12)
    plt.ylabel('Average Norm Product $a_\\alpha$', fontsize=12)
    plt.title('Theorem 4: Noise Gating with Bayesian R‑LayerNorm', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'theorem4_noise_gating.png'), dpi=300)
    plt.close()
