#!/usr/bin/env python3
"""
Quick test for Theorem 4 with extreme parameters.
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geometric_learning.data import SphereDataset
from geometric_learning.models import GatedNetwork
from geometric_learning.simulator import GradientFlowSimulator

def main():
    print("="*60)
    print("THEOREM 4 TEST - EXTREME PARAMETERS")
    print("="*60)
    
    # Extreme parameters
    config = {
        'M': 50,
        'dim': 10,
        'N': 500,
        'steps': 1000,
        'lr': 0.2,
        'alpha': 1.0,
        'lam': 10.0,
        'noise': 5.0,
        'seed': 42
    }
    
    # Create output directory
    out_dir = 'results/test_experiment'
    os.makedirs(out_dir, exist_ok=True)
    
    # Create mixed dataset (clean + noisy)
    clean = SphereDataset(config['N']//2, config['dim'], noise=0.0, seed=config['seed'])
    noisy = SphereDataset(config['N']//2, config['dim'], noise=config['noise'], 
                          seed=config['seed']+1000)
    
    X = torch.cat([clean.X, noisy.X], dim=0)
    y = torch.cat([clean.y, noisy.y], dim=0)
    y_onehot = F.one_hot(y, num_classes=2).float()
    
    class Mixed: pass
    mixed = Mixed()
    mixed.X = X; mixed.y = y; mixed.y_onehot = y_onehot
    mixed.dim = config['dim']; mixed.manifold_type = 'sphere'
    
    # Run without Bayesian
    print("\nRunning without Bayesian...")
    net1 = GatedNetwork(config['M'], config['dim'], seed=config['seed'])
    sim1 = GradientFlowSimulator(mixed, net1, lr=config['lr'], use_bayesian=False)
    sim1.run(steps=config['steps'], record_every=20)
    
    # Run with Bayesian
    print("Running with Bayesian...")
    net2 = GatedNetwork(config['M'], config['dim'], seed=config['seed'])
    sim2 = GradientFlowSimulator(mixed, net2, lr=config['lr'], use_bayesian=True,
                                 alpha=config['alpha'], lam=config['lam'])
    sim2.run(steps=config['steps'], record_every=20)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(sim1.history['step'], sim1.history['a_alpha'].mean(axis=1), 
             'r-', linewidth=2.5, label='Without Bayesian')
    plt.plot(sim2.history['step'], sim2.history['a_alpha'].mean(axis=1), 
             'b-', linewidth=2.5, label='With Bayesian')
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Average Norm Product', fontsize=12)
    plt.title('Theorem 4: Noise Gating - Extreme Parameters\n' +
              f'(α={config["alpha"]}, λ={config["lam"]}, noise={config["noise"]})',
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'theorem4_test.png'), dpi=300)
    plt.show()
    
    # Print results
    final_nobayes = sim1.history['a_alpha'].mean(axis=1)[-1]
    final_bayes = sim2.history['a_alpha'].mean(axis=1)[-1]
    reduction = (final_nobayes - final_bayes)/final_nobayes*100
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Without Bayesian final: {final_nobayes:.4f}")
    print(f"With Bayesian final:    {final_bayes:.4f}")
    print(f"Growth reduction:       {reduction:.2f}%")
    
    with open(os.path.join(out_dir, 'test_summary.txt'), 'w') as f:
        f.write("THEOREM 4 TEST RESULTS\n")
        f.write("="*40 + "\n\n")
        f.write(f"Parameters: {config}\n\n")
        f.write(f"Without Bayesian: {final_nobayes:.4f}\n")
        f.write(f"With Bayesian:    {final_bayes:.4f}\n")
        f.write(f"Reduction:        {reduction:.2f}%\n")
    
    print(f"\nResults saved to {out_dir}/")

if __name__ == '__main__':
    main()
