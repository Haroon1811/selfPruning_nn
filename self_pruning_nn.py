"""
Self-Pruning Neural Network for CIFAR-10
This script implements a feed-forward neural network that learns to prune itself
during training using learnable gating parameters. The key idea:
  - Each weight has a corresponding "gate_score" parameter.
  - Gates = sigmoid(gate_scores) ∈ (0, 1) multiply the weights element-wise.
  - An L1 sparsity penalty on the gates pushes many of them toward 0,
    effectively pruning those weights from the network.
"""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")          # Non-interactive backend (safe for all environments)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


# PART 1 — PrunableLinear Layer

class PrunableLinear(nn.Module):
    """
    A custom linear layer augmented with learnable gate parameters.

    For every weight w_ij there is a corresponding gate_score g_ij.
    During the forward pass:
        gates        = sigmoid(gate_scores)        ∈ (0, 1)
        pruned_weight = weight * gates              (element-wise multiplication)
        output        = x @ pruned_weight.T + bias   (linear layer operation)

    Because sigmoid and element-wise multiplication are both differentiable,
    gradients flow back to both `weight` and `gate_scores` through the standard
    autograd mechanism — no custom backward needed.

    During inference / evaluation you can call hard_prune() to zero out weights
    whose gate falls below a threshold, producing a genuinely sparse network.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # Standard weight & bias (same as nn.Linear)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))

        # Learnable gate scores — same shape as weight.
        # sigmoid(0.5) ≈ 0.62
        self.gate_scores = nn.Parameter(torch.full((out_features, in_features), 0.5))

        # Weight initialization (Kaiming uniform — same as nn.Linear default)
        nn.init.kaiming_uniform_(self.weight, a=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: Map gate_scores, gates ∈ (0, 1)
        gates = torch.sigmoid(self.gate_scores)

        # Step 2: Element-wise multiply weights by gates (soft pruning)
        pruned_weights = self.weight * gates

        # Step 3:gradients propagate to both `self.weight` and `self.gate_scores` automatically.
        return F.linear(x, pruned_weights, self.bias)

    def get_gates(self) -> torch.Tensor:
        """Return the current gate values (detached from graph)."""
        return torch.sigmoid(self.gate_scores).detach()

    def sparsity_loss(self) -> torch.Tensor:
        """L1 norm of gates for this layer (used in the total loss)."""
        return torch.sigmoid(self.gate_scores).abs().sum()

    def hard_prune(self, threshold: float = 1e-2) -> int:
        """
        Post-training: zero out weights whose gate < threshold.
        Returns the number of connections pruned.
        """
        with torch.no_grad():
            mask = self.get_gates() < threshold
            self.weight[mask] = 0.0
            return mask.sum().item()

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}"


# Network Structure Definition

class SelfPruningNet(nn.Module):
    """
    A feed-forward network for CIFAR-10 (32*32*3 = 3072 inputs, 10 classes).

    Architecture:
        Input (3072)
            → PrunableLinear(3072 → 1024) → BatchNorm → ReLU → Dropout
            → PrunableLinear(1024 → 512)  → BatchNorm → ReLU → Dropout
            → PrunableLinear(512  → 256)  → BatchNorm → ReLU → Dropout
            → PrunableLinear(256  → 10)   (logits)

    BatchNorm is applied AFTER gating so the network can still learn a clean
    representation even as many weights are zeroed.
    """

    def __init__(self, dropout_rate: float = 0.3):
        super().__init__()

        self.prunable_layers = nn.ModuleList([
            PrunableLinear(3072, 1024),
            PrunableLinear(1024, 512),
            PrunableLinear(512,  256),
            PrunableLinear(256,  10),
        ])

        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)   # Flatten CIFAR image to (B, 3072)

        x = self.prunable_layers[0](x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.prunable_layers[1](x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.prunable_layers[2](x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.prunable_layers[3](x)   # Logits — no activation
        return x

    def total_sparsity_loss(self) -> torch.Tensor:
        """Sum of L1 gate norms across ALL PrunableLinear layers."""
        return sum(layer.sparsity_loss() for layer in self.prunable_layers)

    def compute_sparsity_level(self, threshold: float = 1e-2) -> float:
        """
        Fraction of weights whose gate value is below `threshold`.
        A value of 0.80 means 80 % of weights are effectively pruned.
        """
        total_weights = 0
        pruned_weights = 0
        for layer in self.prunable_layers:
            gates = layer.get_gates()
            total_weights  += gates.numel()
            pruned_weights += (gates < threshold).sum().item()
        return pruned_weights / total_weights

    def all_gate_values(self) -> np.ndarray:
        """Collect every gate value across the whole network into one array."""
        all_gates = []
        for layer in self.prunable_layers:
            all_gates.append(layer.get_gates().cpu().numpy().ravel())
        return np.concatenate(all_gates)

# PART 2 — Total Loss

def compute_total_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    model: SelfPruningNet,
    lambda_sparse: float,
                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Total Loss = CrossEntropyLoss + λ × SparsityLoss

    Args:
        logits        : Raw model outputs  (B, 10)
        targets       : Ground-truth labels (B,)
        model         : The SelfPruningNet instance
        lambda_sparse : Sparsity regularisation strength

    Returns:
        total_loss, classification_loss, sparsity_loss  (all scalar tensors)
    """
    classification_loss = F.cross_entropy(logits, targets)

    # L1 norm of all sigmoid-gated values — encourages exact zeros
    sparsity_loss = model.total_sparsity_loss()

    total_loss = classification_loss + lambda_sparse * sparsity_loss
    return total_loss, classification_loss, sparsity_loss


# PART 3 — Data Loading

def get_cifar10_loaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader]:
    """
    Download (if needed) and return train / test DataLoaders for CIFAR-10.

    Augmentation (train only):
        - Random horizontal flip
        - Random crop with padding
        - Colour jitter (slight)

    Normalisation uses CIFAR-10 channel statistics.
    """
    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std  = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True,  download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=256, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, test_loader

# PART 3 — Training Loop

def train_one_epoch(
    model: SelfPruningNet,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    lambda_sparse: float,
    device: torch.device,
    scheduler=None,
                  ) -> dict:
    """Run one full training epoch. Returns a dict of average metrics."""
    model.train()
    total_loss_sum  = 0.0
    cls_loss_sum    = 0.0
    sparse_loss_sum = 0.0
    correct = 0
    n_samples = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        total_loss, cls_loss, sparse_loss = compute_total_loss(
            logits, labels, model, lambda_sparse
                                                             )

        total_loss.backward()

        # Gradient clipping — stabilises training when λ is large
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # ── Bookkeeping ──
        batch_size = images.size(0)
        total_loss_sum  += total_loss.item()  * batch_size
        cls_loss_sum    += cls_loss.item()    * batch_size
        sparse_loss_sum += sparse_loss.item() * batch_size
        correct  += (logits.argmax(1) == labels).sum().item()
        n_samples += batch_size

    return {
        "total_loss"   : total_loss_sum  / n_samples,
        "cls_loss"     : cls_loss_sum    / n_samples,
        "sparse_loss"  : sparse_loss_sum / n_samples,
        "train_acc"    : correct         / n_samples,
           }


@torch.no_grad()
def evaluate(
    model: SelfPruningNet,
    loader: DataLoader,
    device: torch.device,
      ) -> float:
    """Return test accuracy (0-1)."""
    model.eval()
    correct   = 0
    n_samples = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(1)
        correct   += (preds == labels).sum().item()
        n_samples += images.size(0)
    return correct / n_samples


def train_model(
    lambda_sparse: float,
    device: torch.device,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 30,
    lr: float = 3e-3,
    weight_decay: float = 1e-4,
    verbose: bool = True,
              ) -> dict:
    """
    Full training run for a single lambda value.

    Returns a dict with:
        model            :- the trained SelfPruningNet
        history          :-  list of per-epoch metric dicts
        test_accuracy    :-  final test accuracy
        sparsity_level   :-  fraction of gates < 1e-2 after training
        gate_values      :- numpy array of all gate values
    """
    model = SelfPruningNet(dropout_rate=0.3).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    # Cosine annealing — gently reduces lr so gates settle near 0 or 1
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = []

    for epoch in range(1, epochs + 1):
        metrics = train_one_epoch(
            model, train_loader, optimizer, lambda_sparse, device
        )
        scheduler.step()

        sparsity = model.compute_sparsity_level()
        metrics["sparsity"] = sparsity

        history.append(metrics)

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(
                f"  Epoch {epoch:3d}/{epochs} | "
                f"TotalLoss={metrics['total_loss']:.4f} | "
                f"ClsLoss={metrics['cls_loss']:.4f} | "
                f"SparseLoss={metrics['sparse_loss']:.6f} | "
                f"TrainAcc={metrics['train_acc']*100:.1f}% | "
                f"Sparsity={sparsity*100:.1f}%"
            )

    test_acc      = evaluate(model, test_loader, device)
    sparsity_lvl  = model.compute_sparsity_level()
    gate_values   = model.all_gate_values()

    return {
        "model"          : model,
        "history"        : history,
        "test_accuracy"  : test_acc,
        "sparsity_level" : sparsity_lvl,
        "gate_values"    : gate_values,
        "lambda"         : lambda_sparse,
    }


# ─────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────

def plot_gate_distribution(results: list[dict], save_path: str = "gate_distribution.png"):
    """
    For each λ, plot the histogram of all final gate values.
    A successful pruning shows a strong spike near 0 and a separate
    cluster of active gates away from 0.
    """
    n = len(results)
    fig = plt.figure(figsize=(6 * n, 5))
    fig.suptitle("Gate Value Distributions by λ", fontsize=16, fontweight="bold")

    colours = ["#2196F3", "#FF5722", "#4CAF50"]

    for i, res in enumerate(results):
        ax = fig.add_subplot(1, n, i + 1)
        gates = res["gate_values"]
        lam   = res["lambda"]
        acc   = res["test_accuracy"] * 100
        sp    = res["sparsity_level"] * 100

        ax.hist(gates, bins=80, color=colours[i % len(colours)],
                edgecolor="white", linewidth=0.3, alpha=0.85)
        ax.set_title(
            f"lambda = {lam}\nTest Acc = {acc:.1f}%  |  Sparsity = {sp:.1f}%",
            fontsize=11
        )
        ax.set_xlabel("Gate Value", fontsize=10)
        ax.set_ylabel("Count",      fontsize=10)
        ax.set_xlim(0, 1)
        ax.axvline(x=0.01, color="red", linestyle="--", linewidth=1.2,
                   label="Prune threshold (0.01)")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[Plot saved] → {save_path}")


def plot_training_curves(results: list[dict], save_path: str = "training_curves.png"):
    """Plot train accuracy and sparsity over epochs for each λ."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colours  = ["#2196F3", "#FF5722", "#4CAF50"]
    linestyles = ["-", "--", "-."]

    for i, res in enumerate(results):
        history = res["history"]
        epochs  = range(1, len(history) + 1)
        lam     = res["lambda"]
        c, ls   = colours[i], linestyles[i]

        train_acc = [h["train_acc"] * 100 for h in history]
        sparsity  = [h["sparsity"]  * 100 for h in history]

        axes[0].plot(epochs, train_acc, color=c, linestyle=ls, label=f"λ={lam}")
        axes[1].plot(epochs, sparsity,  color=c, linestyle=ls, label=f"λ={lam}")

    axes[0].set_title("Training Accuracy over Epochs",  fontsize=13)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend(); axes[0].grid(alpha=0.35)

    axes[1].set_title("Network Sparsity Level over Epochs", fontsize=13)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Sparsity (%)")
    axes[1].legend(); axes[1].grid(alpha=0.35)

    plt.suptitle("Training Dynamics — Self-Pruning Network", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot saved] → {save_path}")

# Results Summary

def print_results_table(results: list[dict]):
    """Print a formatted results table to stdout."""
    print("\n" + "=" * 60)
    print(f"{'Lambda':>12} | {'Test Accuracy':>14} | {'Sparsity Level':>15}")
    print("-" * 60)
    for res in results:
        lam  = res["lambda"]
        acc  = res["test_accuracy"] * 100
        sp   = res["sparsity_level"] * 100
        print(f"{lam:>12.4f} | {acc:>13.2f}% | {sp:>14.2f}%")
    print("=" * 60 + "\n")


# Main Entry Point


def main():
    parser = argparse.ArgumentParser(description="Self-Pruning Neural Network CIFAR-10")
    parser.add_argument("--epochs",     type=int,   default=30,
                        help="Training epochs per λ (default: 30)")
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--lr",         type=float, default=3e-3)
    parser.add_argument("--lambdas",    type=float, nargs="+",
                        default=[1e-5, 1e-4, 1e-3],
                        help="Sparsity λ values to sweep (default: 1e-5 1e-4 1e-3)")
    parser.add_argument("--data_dir",   type=str,   default="./data")
    parser.add_argument("--output_dir", type=str,   default="./outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Self-Pruning Neural Network  |  Device: {device}")
    print(f"  Epochs: {args.epochs}  |  Batch: {args.batch_size}  |  LR: {args.lr}")
    print(f"  Lambda sweep: {args.lambdas}")
    print(f"{'='*60}\n")

    # ── Data ──
    train_loader, test_loader = get_cifar10_loaders(
        data_dir=args.data_dir, batch_size=args.batch_size
    )

    # ── Sweep lambda values ──
    all_results = []
    for lam in args.lambdas:
        print(f"\n{'─'*50}")
        print(f"  Training with λ = {lam}")
        print(f"{'─'*50}")

        result = train_model(
            lambda_sparse=lam,
            device=device,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=args.epochs,
            lr=args.lr,
        )
        all_results.append(result)

        print(
            f"\n  lambda={lam} | Test Acc={result['test_accuracy']*100:.2f}%"
            f" | Sparsity={result['sparsity_level']*100:.2f}%"
        )

    # ── Report ──
    print_results_table(all_results)

    # ── Plots ──
    gate_plot_path   = os.path.join(args.output_dir, "gate_distribution.png")
    curves_plot_path = os.path.join(args.output_dir, "training_curves.png")
    plot_gate_distribution(all_results, save_path=gate_plot_path)
    plot_training_curves(all_results,   save_path=curves_plot_path)

    # ── Save numeric results as JSON ──
    summary = [
        {
            "lambda"         : r["lambda"],
            "test_accuracy"  : round(r["test_accuracy"]  * 100, 2),
            "sparsity_level" : round(r["sparsity_level"] * 100, 2),
        }
        for r in all_results
    ]
    summary_path = os.path.join(args.output_dir, "results_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Results saved] → {summary_path}")

    # ── Save best model ──
    best = max(all_results, key=lambda r: r["test_accuracy"])
    best_path = os.path.join(args.output_dir, "best_model.pt")
    torch.save(best["model"].state_dict(), best_path)
    print(f"[Best model saved] → {best_path}  (λ={best['lambda']}, acc={best['test_accuracy']*100:.2f}%)")


if __name__ == "__main__":
    main()
