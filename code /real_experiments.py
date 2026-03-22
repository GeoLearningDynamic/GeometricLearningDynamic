"""
Real‑data experiments: MNIST and Fashion‑MNIST.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from utils import RealDataset, MLP
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def real_theorem3(seed, args, dataset_name='mnist'):
    dataset = RealDataset(name=dataset_name, n_components=args.pca_dim, seed=seed, train=True)
    X = dataset.X.to(device)
    y = dataset.y_bin.to(device)
    model = MLP(input_dim=784, hidden_dims=[256,128], output_dim=2,
                use_bayesian=False, seed=seed).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(X, y), batch_size=128, shuffle=True)

    for epoch in range(args.epochs):
        for data, target in loader:
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

    w1 = model.net[0].weight.detach().cpu().numpy()
    w1_norm = np.linalg.norm(w1, axis=1)
    threshold = np.percentile(w1_norm, 80)
    winners = w1_norm > threshold
    if winners.sum() > 0:
        dirs = w1[winners]
        dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
        mean_vec = dirs.mean(axis=0)
        R = np.linalg.norm(mean_vec)
    else:
        R = 0.0
    return R

def real_theorem4(seed, args, dataset_name='mnist'):
    dataset = RealDataset(name=dataset_name, n_components=args.pca_dim, seed=seed, train=True)
    noise = torch.randn_like(dataset.X) * args.noise
    X_noisy = dataset.X + noise
    X_noisy = X_noisy.to(device)
    y = dataset.y_bin.to(device)

    # Without Bayesian
    model_no = MLP(input_dim=784, hidden_dims=[256,128], output_dim=2,
                   use_bayesian=False, seed=seed).to(device)
    optimizer_no = optim.SGD(model_no.parameters(), lr=args.lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    loader_no = DataLoader(TensorDataset(X_noisy, y), batch_size=128, shuffle=True)
    for epoch in range(args.epochs):
        for data, target in loader_no:
            optimizer_no.zero_grad()
            out = model_no(data)
            loss = criterion(out, target)
            loss.backward()
            optimizer_no.step()

    # With Bayesian
    model_bayes = MLP(input_dim=784, hidden_dims=[256,128], output_dim=2,
                      use_bayesian=True, alpha=args.alpha, lam=args.lam, seed=seed).to(device)
    optimizer_bayes = optim.SGD(model_bayes.parameters(), lr=args.lr, momentum=0.9)
    loader_bayes = DataLoader(TensorDataset(X_noisy, y), batch_size=128, shuffle=True)
    for epoch in range(args.epochs):
        for data, target in loader_bayes:
            optimizer_bayes.zero_grad()
            out = model_bayes(data)
            loss = criterion(out, target)
            loss.backward()
            optimizer_bayes.step()

    # Evaluate on clean test set
    test_dataset = RealDataset(name=dataset_name, n_components=args.pca_dim, seed=seed+1000, train=False)
    X_test = test_dataset.X.to(device)
    y_test = test_dataset.y_bin.to(device)
    model_no.eval()
    model_bayes.eval()
    with torch.no_grad():
        acc_no = (model_no(X_test).argmax(dim=1) == y_test).float().mean().item()
        acc_bayes = (model_bayes(X_test).argmax(dim=1) == y_test).float().mean().item()
    return acc_no, acc_bayes

if __name__ == "__main__":
    args = type('Args', (), {
        'pca_dim': 50,
        'lr': 0.01,
        'epochs': 30,
        'noise': 2.0,
        'alpha': 1.0,
        'lam': 10.0,
        'seeds': 5,
        'out_dir': '../results'
    })()

    os.makedirs(args.out_dir, exist_ok=True)

    # MNIST
    R_mnist = []
    acc_no_mnist = []
    acc_bayes_mnist = []
    for seed in range(args.seeds):
        R_mnist.append(real_theorem3(seed, args, 'mnist'))
        acc_no, acc_bayes = real_theorem4(seed, args, 'mnist')
        acc_no_mnist.append(acc_no)
        acc_bayes_mnist.append(acc_bayes)

    # Fashion-MNIST
    R_fashion = []
    acc_no_fashion = []
    acc_bayes_fashion = []
    for seed in range(args.seeds):
        R_fashion.append(real_theorem3(seed, args, 'fashion_mnist'))
        acc_no, acc_bayes = real_theorem4(seed, args, 'fashion_mnist')
        acc_no_fashion.append(acc_no)
        acc_bayes_fashion.append(acc_bayes)

    # Print summary
    print(f"MNIST Theorem 3 R: {np.mean(R_mnist):.4f} ± {np.std(R_mnist):.4f}")
    print(f"MNIST Theorem 4 accuracy gain: {(np.mean(acc_bayes_mnist)-np.mean(acc_no_mnist))*100:.2f}%")
    print(f"Fashion-MNIST Theorem 3 R: {np.mean(R_fashion):.4f} ± {np.std(R_fashion):.4f}")
    print(f"Fashion-MNIST Theorem 4 accuracy gain: {(np.mean(acc_bayes_fashion)-np.mean(acc_no_fashion))*100:.2f}%")

    # Plot Figure 2a
    plt.figure(figsize=(8,5))
    x = np.arange(args.seeds)
    width = 0.35
    plt.bar(x - width/2, acc_no_mnist, width, label='Without Bayesian', color='red')
    plt.bar(x + width/2, acc_bayes_mnist, width, label='With Bayesian', color='blue')
    plt.xlabel('Seed')
    plt.ylabel('Test Accuracy')
    plt.title('Theorem 4: Noise Gating on Noisy MNIST')
    plt.legend()
    plt.savefig(os.path.join(args.out_dir, 'real_theorem4_mnist.png'), dpi=150)
    plt.show()

    # Figure 2b
    plt.figure(figsize=(8,5))
    plt.bar(x - width/2, acc_no_fashion, width, label='Without Bayesian', color='red')
    plt.bar(x + width/2, acc_bayes_fashion, width, label='With Bayesian', color='blue')
    plt.xlabel('Seed')
    plt.ylabel('Test Accuracy')
    plt.title('Theorem 4: Noise Gating on Noisy Fashion-MNIST')
    plt.legend()
    plt.savefig(os.path.join(args.out_dir, 'real_theorem4_fashion.png'), dpi=150)
    plt.show()
