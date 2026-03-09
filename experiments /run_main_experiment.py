#!/usr/bin/env python3
"""
Main experiment runner for Theorems 1-4.
"""

import argparse
import os
import sys
import yaml
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geometric_learning.data import SphereDataset
from geometric_learning.models import GatedNetwork
from geometric_learning.simulator import GradientFlowSimulator
from geometric_learning.utils import plotting, metrics

def run_theorem1(seed, args):
    """Run Theorem 1 experiment."""
    dataset = SphereDataset(args['N'], args['dim'], noise=0.0, seed=seed)
    net = GatedNetwork(args['M'], args['dim'], seed=seed)
    sim = GradientFlowSimulator(dataset, net, lr=args['lr'], use_bayesian=False)
    sim.run(steps=args['steps'], record_every=20)
    avg_perp = sim.history['norm_perp'].mean(axis=1)
    return {'steps': sim.history['step'], 'avg_perp': avg_perp}

def run_theorem2(seed, args):
    """Run Theorem 2 experiment."""
    dataset = SphereDataset(args['N'], args['dim'], seed=seed)
    net = GatedNetwork(args['M'], args['dim'], seed=seed)
    sim = GradientFlowSimulator(dataset, net, lr=args['lr'], use_bayesian=False)
    sim.run(steps=args['steps'], record_every=20)
    return {'steps': sim.history['step'], 'L12': sim.history['L12']}

def run_theorem3(seed, args):
    """Run Theorem 3 experiment."""
    dataset = SphereDataset(args['N'], args['dim'], seed=seed)
    net = GatedNetwork(args['M'], args['dim'], seed=seed)
    sim = GradientFlowSimulator(dataset, net, lr=args['lr'], use_bayesian=False)
    sim.run(steps=args['steps'], record_every=20)
    
    final_a = sim.history['a_alpha'][-1, :]
    threshold = np.percentile(final_a, 80)
    winners = final_a > threshold
    winner_indices = np.where(winners)[0]
    
    if len(winner_indices) > 0:
        final_w1 = net.w1.detach().cpu().numpy()
        dirs = final_w1[winner_indices]
        dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
        mean_vec = dirs.mean(axis=0)
        R = np.linalg.norm(mean_vec)
    else:
        R = 0.0
        dirs = np.array([])
    
    return {'R': R, 'dirs': dirs}

def run_theorem4(seed, args):
    """Run Theorem 4 experiment."""
    from geometric_learning.data import SphereDataset
    from geometric_learning.models import GatedNetwork
    from geometric_learning.simulator import GradientFlowSimulator
    
    clean = SphereDataset(args['N']//2, args['dim'], noise=0.0, seed=seed)
    noisy = SphereDataset(args['N']//2, args['dim'], noise=args['noise'], seed=seed+1000)
    
    X = torch.cat([clean.X, noisy.X], dim=0)
    y = torch.cat([clean.y, noisy.y], dim=0)
    y_onehot = torch.nn.functional.one_hot(y, num_classes=2).float()
    
    class Mixed: pass
    mixed = Mixed()
    mixed.X = X; mixed.y = y; mixed.y_onehot = y_onehot
    mixed.dim = args['dim']; mixed.manifold_type = 'sphere'
    
    net1 = GatedNetwork(args['M'], args['dim'], seed=seed)
    sim1 = GradientFlowSimulator(mixed, net1, lr=args['lr'], use_bayesian=False)
    sim1.run(steps=args['steps'], record_every=20)
    
    net2 = GatedNetwork(args['M'], args['dim'], seed=seed)
    sim2 = GradientFlowSimulator(mixed, net2, lr=args['lr'], use_bayesian=True,
                                 alpha=args['alpha'], lam=args['lam'])
    sim2.run(steps=args['steps'], record_every=20)
    
    a1 = sim1.history['a_alpha'].mean(axis=1)
    a2 = sim2.history['a_alpha'].mean(axis=1)
    
    return {'steps': sim1.history['step'], 'a_nobayes': a1, 'a_bayes': a2}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='experiments/config/main_config.yaml')
    parser.add_argument('--seeds', type=int, default=None)
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--out_dir', type=str, default=None)
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override with command line args
    if args.seeds:
        config['seeds'] = args.seeds
    if args.steps:
        config['steps'] = args.steps
    if args.out_dir:
        config['out_dir'] = args.out_dir
    
    print("="*60)
    print("GEOMETRIC LEARNING DYNAMICS - MAIN EXPERIMENT")
    print("="*60)
    print(f"Running with config: {config}")
    
    os.makedirs(config['out_dir'], exist_ok=True)
    
    # Storage
    th1_avg_perp = []
    th2_L12 = []
    th3_R = []
    th4_a_nobayes = []
    th4_a_bayes = []
    
    for seed in range(config['seeds']):
        print(f"\n>>> Seed {seed+1}/{config['seeds']}")
        
        # Theorem 1
        res1 = run_theorem1(seed, config)
        th1_avg_perp.append(res1['avg_perp'])
        steps_vals = res1['steps']
        
        # Theorem 2
        res2 = run_theorem2(seed, config)
        th2_L12.append(res2['L12'])
        
        # Theorem 3
        res3 = run_theorem3(seed, config)
        th3_R.append(res3['R'])
        
        # Theorem 4
        res4 = run_theorem4(seed, config)
        th4_a_nobayes.append(res4['a_nobayes'])
        th4_a_bayes.append(res4['a_bayes'])
    
    # Convert to arrays
    th1_avg_perp = np.array(th1_avg_perp)
    th2_L12 = np.array(th2_L12)
    th3_R = np.array(th3_R)
    th4_a_nobayes = np.array(th4_a_nobayes)
    th4_a_bayes = np.array(th4_a_bayes)
    
    # Compute metrics
    eta, y0, yT = metrics.compute_decay_rate(th1_avg_perp, steps_vals)
    drift_mean, drift_std = metrics.compute_drift(th2_L12)
    
    results_dict = {
        'th1_initial': y0,
        'th1_initial_std': th1_avg_perp[:,0].std(),
        'th1_final': yT,
        'th1_final_std': th1_avg_perp[:,-1].std(),
        'th1_eta': eta,
        'th2_drift_mean': drift_mean,
        'th2_drift_std': drift_std,
        'th3_R_mean': th3_R.mean(),
        'th3_R_std': th3_R.std(),
        'th3_R_seeds': ', '.join([f'{x:.4f}' for x in th3_R]),
        'th4_nobayes': th4_a_nobayes[:,-1].mean(),
        'th4_nobayes_std': th4_a_nobayes[:,-1].std(),
        'th4_bayes': th4_a_bayes[:,-1].mean(),
        'th4_bayes_std': th4_a_bayes[:,-1].std(),
        'th4_reduction': (th4_a_nobayes[:,-1].mean() - th4_a_bayes[:,-1].mean()) / th4_a_nobayes[:,-1].mean() * 100
    }
    
    # Generate plots
    plotting.plot_theorem1(steps_vals, 
                          th1_avg_perp.mean(axis=0), 
                          th1_avg_perp.std(axis=0), 
                          config['out_dir'])
    
    plotting.plot_theorem2(steps_vals,
                          th2_L12.mean(axis=0),
                          th2_L12.std(axis=0),
                          config['out_dir'])
    
    # Collect all dirs for Theorem 3
    all_dirs = []
    for seed in range(config['seeds']):
        res3 = run_theorem3(seed, config)
        if res3['dirs'].size > 0:
            all_dirs.append(res3['dirs'])
    if all_dirs:
        all_dirs = np.concatenate(all_dirs, axis=0)
        plotting.plot_theorem3(all_dirs, config['out_dir'])
    
    plotting.plot_theorem4(steps_vals,
                          th4_a_nobayes.mean(axis=0),
                          th4_a_nobayes.std(axis=0),
                          th4_a_bayes.mean(axis=0),
                          th4_a_bayes.std(axis=0),
                          config['out_dir'])
    
    # Save summary
    metrics.save_summary(results_dict, config, config['out_dir'])
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    print(f"Results saved to: {config['out_dir']}/")

if __name__ == '__main__':
    main()
