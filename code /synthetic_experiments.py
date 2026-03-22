"""
Full synthetic experiments: sphere and Swiss roll.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from utils import SphereDataset, SwissRollDataset, GatedNetwork, GradientFlowSimulator
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def synth_theorem1(seed, args, manifold='sphere'):
    if manifold == 'sphere':
        dataset = SphereDataset(args.N, args.dim, rotation_invariant=False, seed=seed)
    else:
        dataset = SwissRollDataset(args.N, noise=0.1, seed=seed)
    net = GatedNetwork(args.M, dataset.X.shape[1], use_bayesian=False, seed=seed).to(device)
    sim = GradientFlowSimulator(dataset, net, lr=args.lr)
    sim.run(steps=args.steps, record_every=args.record_every, name=f"Th1_{manifold}_seed{seed}")
    return {'steps': sim.history['step'], 'norm_perp': sim.history['norm_perp'],
            'cos_align': sim.history['cos_align']}

def synth_theorem2(seed, args):
    dataset = SphereDataset(args.N, args.dim, rotation_invariant=True, seed=seed)
    net = GatedNetwork(args.M, args.dim, use_bayesian=False, constant_gate=True, seed=seed).to(device)
    with torch.no_grad():
        u = net.w1[0].clone()
        v = net.w1[1].clone()
        v = v - (v @ u) * u
        v = v / torch.norm(v) * torch.norm(net.w1[1])
        net.w1[1] = v
    sim = GradientFlowSimulator(dataset, net, lr=args.lr)
    sim.run(steps=args.steps, record_every=args.record_every, name=f"Th2_seed{seed}")
    return {'steps': sim.history['step'], 'L12': sim.history['L12']}

def synth_theorem3(seed, args):
    dataset = SphereDataset(args.N, args.dim, rotation_invariant=False, seed=seed)
    net = GatedNetwork(args.M, args.dim, use_bayesian=False, seed=seed).to(device)
    sim = GradientFlowSimulator(dataset, net, lr=args.lr)
    sim.run(steps=args.steps, record_every=args.record_every, name=f"Th3_seed{seed}")
    w1 = net.w1.detach().cpu().numpy()
    w2 = net.w2.detach().cpu().numpy()
    w1_norm = np.linalg.norm(w1, axis=1)
    w2_norm = np.linalg.norm(w2, axis=0)
    a_per_neuron = w1_norm * w2_norm
    threshold = np.percentile(a_per_neuron, 80)
    winners = a_per_neuron > threshold
    if winners.sum() > 0:
        dirs = w1[winners]
        dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
        mean_vec = dirs.mean(axis=0)
        R = np.linalg.norm(mean_vec)
    else:
        R = 0.0
    return {'R': R}

def synth_theorem4(seed, args):
    # Only run if stable (noise capped)
    if args.noise > 12.0:
        return None
    clean = SphereDataset(args.N, args.dim, rotation_invariant=False, seed=seed)
    X_noisy = clean.X + args.noise * torch.randn_like(clean.X)
    X = torch.cat([clean.X, X_noisy], dim=0)
    y = torch.cat([clean.y, clean.y], dim=0)
    y_onehot = torch.nn.functional.one_hot(y, num_classes=2).float()
    class Mixed:
        def __init__(self, X, y_onehot):
            self.X = X
            self.y_onehot = y_onehot
        def tangent_project(self, x_center, v):
            return v - (v @ x_center) * x_center
    mixed = Mixed(X, y_onehot)

    net1 = GatedNetwork(args.M, args.dim, use_bayesian=False, seed=seed).to(device)
    sim1 = GradientFlowSimulator(mixed, net1, lr=args.lr)
    sim1.run(steps=args.steps, record_every=args.record_every, name=f"Th4_nobayes_seed{seed}")

    net2 = GatedNetwork(args.M, args.dim, use_bayesian=True, alpha=args.alpha, lam=args.lam, seed=seed).to(device)
    sim2 = GradientFlowSimulator(mixed, net2, lr=args.lr)
    sim2.run(steps=args.steps, record_every=args.record_every, name=f"Th4_bayes_seed{seed}")

    a1 = sim1.history['a_alpha']
    a2 = sim2.history['a_alpha']
    return {'steps': sim1.history['step'], 'a_nobayes': a1, 'a_bayes': a2}

if __name__ == "__main__":
    # Parameters
    args = type('Args', (), {
        'M': 100,
        'dim': 30,
        'N': 1000,
        'steps': 20000,
        'record_every': 200,
        'lr': 0.1,
        'alpha': 1.0,
        'lam': 10.0,
        'noise': 12.0,
        'seeds': 10,
        'out_dir': '../results'
    })()
    os.makedirs(args.out_dir, exist_ok=True)

    # Run sphere experiments
    th1_norm = []
    th2_L12 = []
    th3_R = []
    for seed in range(args.seeds):
        print(f"Seed {seed+1}/{args.seeds}")
        r1 = synth_theorem1(seed, args, 'sphere')
        th1_norm.append(r1['norm_perp'])
        r2 = synth_theorem2(seed, args)
        th2_L12.append(r2['L12'])
        r3 = synth_theorem3(seed, args)
        th3_R.append(r3['R'])

    # Aggregate results and plot
    steps = r1['steps']
    th1_norm = np.array(th1_norm)
    th2_L12 = np.array(th2_L12)
    th3_R = np.array(th3_R)

    # Plot Figure 1a
    plt.figure(figsize=(10,6))
    mean = th1_norm.mean(axis=0)
    std = th1_norm.std(axis=0)
    plt.semilogy(steps, mean, linewidth=2)
    plt.fill_between(steps, mean-std, mean+std, alpha=0.2)
    plt.xlabel('Step')
    plt.ylabel('$\\|w_\\perp\\|$')
    plt.title('Theorem 1: Manifold Alignment on Sphere')
    plt.grid(True)
    plt.savefig(os.path.join(args.out_dir, 'synth_theorem1_compare.png'), dpi=150)
    plt.show()

    # Print summary
    print(f"Sphere Theorem 1 reduction: {(1 - mean[-1]/mean[0])*100:.2f}%")
    drift = th2_L12[:,-1] - th2_L12[:,0]
    print(f"Theorem 2 absolute drift: {drift.mean():.4f} ± {drift.std():.4f}")
    print(f"Theorem 3 R: {th3_R.mean():.4f} ± {th3_R.std():.4f}")
