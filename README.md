# Self-Pruning Neural Network — Tredence Case Study

> **AI Engineering Intern Case Study | 2025 Cohort**

A neural network that **learns to prune itself during training** using learnable gate parameters and L1 sparsity regularisation. Built from scratch on PyTorch for CIFAR-10 image classification.

---

## Core Idea

Each weight `w_ij` is multiplied by a learnable gate:

```
gate_ij        = sigmoid(gate_score_ij)    ∈ (0, 1)
effective_w_ij = w_ij × gate_ij
```

Training minimises:

```
L_total = L_cross_entropy + λ × Σ gate_ij
```

The L1 penalty drives unimportant gates to **exactly 0**, pruning those connections without any post-training step.

---

## Quick Start

```bash
# Clone and install
git clone <your-repo-url>
cd self-pruning-nn
pip install -r requirements.txt

# Train with default λ sweep (1e-5, 1e-4, 1e-3)
python self_pruning_nn.py

# Custom λ sweep and epoch count
python self_pruning_nn.py --epochs 50 --lambdas 1e-6 1e-5 1e-4 1e-3
```

Outputs are saved to `./outputs/`:
- `gate_distribution.png` — histogram of all gate values per λ
- `training_curves.png`   — accuracy & sparsity over epochs
- `results_summary.json`  — numeric summary table
- `best_model.pt`         — saved weights of the best model

---

## File Structure

```
├── self_pruning_nn.py   # All code: layer, network, loss, training, evaluation
├── REPORT.md            # Written report with theory, results, and analysis
├── requirements.txt     # Dependencies
└── outputs/             # Auto-created at runtime
```

---

## Results Summary (example)

| Lambda (λ) | Test Accuracy | Sparsity Level |
|:----------:|:-------------:|:--------------:|
| `1e-5`     | ~53 %         | ~20 %          |
| `1e-4`     | ~48 %         | ~62 %          |
| `1e-3`     | ~34 %         | ~90 %          |

See `REPORT.md` for full analysis.

---

## Key Design Choices

| Component | Choice | Why |
|:----------|:-------|:----|
| Gate activation | Sigmoid | Smooth, bounded [0,1], fully differentiable |
| Sparsity loss | L1 norm | Constant gradient drives gates to exactly 0 |
| Optimiser | AdamW + Cosine LR | Stable convergence; gates settle cleanly |
| Normalisation | BatchNorm after gate | Stabilises even with many zeroed weights |

---

*Submitted for the Tredence Studio — AI Agents Engineering Team.*
