"""
Metrics computation and summary functions.
"""

import numpy as np
import os

def compute_decay_rate(perp_values, steps):
    """Compute exponential decay rate η from ||w⊥|| values."""
    y0 = perp_values[:,0].mean()
    yT = perp_values[:,-1].mean()
    eta = -np.log(yT/y0) / steps[-1]
    return eta, y0, yT

def compute_drift(L12_values):
    """Compute drift in angular momentum."""
    initial = L12_values[:,0]
    final = L12_values[:,-1]
    drift = final - initial
    return drift.mean(), drift.std()

def save_summary(results, config, out_dir):
    """Save summary statistics to text file."""
    with open(os.path.join(out_dir, 'results_summary.txt'), 'w') as f:
        f.write("="*60 + "\n")
        f.write("GEOMETRIC LEARNING DYNAMICS - EXPERIMENTAL RESULTS\n")
        f.write("="*60 + "\n\n")
        
        f.write("Configuration:\n")
        for k, v in config.items():
            f.write(f"  {k}: {v}\n")
        f.write("\n")
        
        # Theorem 1
        f.write("Theorem 1: Manifold Alignment\n")
        f.write(f"  Initial ||w⊥||: {results['th1_initial']:.4f} ± {results['th1_initial_std']:.4f}\n")
        f.write(f"  Final ||w⊥||:   {results['th1_final']:.4f} ± {results['th1_final_std']:.4f}\n")
        f.write(f"  Decay rate η:   {results['th1_eta']:.6f}\n\n")
        
        # Theorem 2
        f.write("Theorem 2: Angular Momentum Conservation\n")
        f.write(f"  Mean drift: {results['th2_drift_mean']:.8f} ± {results['th2_drift_std']:.8f}\n\n")
        
        # Theorem 3
        f.write("Theorem 3: Winning Ticket Concentration\n")
        f.write(f"  Mean resultant length R: {results['th3_R_mean']:.4f} ± {results['th3_R_std']:.4f}\n")
        f.write(f"  Per seed: {results['th3_R_seeds']}\n\n")
        
        # Theorem 4
        f.write("Theorem 4: Noise Gating\n")
        f.write(f"  Without Bayesian: {results['th4_nobayes']:.4f} ± {results['th4_nobayes_std']:.4f}\n")
        f.write(f"  With Bayesian:    {results['th4_bayes']:.4f} ± {results['th4_bayes_std']:.4f}\n")
        f.write(f"  Reduction:        {results['th4_reduction']:.2f}%\n")
