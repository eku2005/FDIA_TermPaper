# FDIA Detection — Deep Learning Framework

Implementation of:  
**"Deep Learning-Based Detection of False Data Injection Attacks in Smart Grid State Estimation"**  
Bi, Chen, Li — IEEE IMCEC 2025

---

## Project Structure

```
fdia_detection/
├── configs/
│   └── config.yaml            # All hyperparameters & settings
├── data/
│   ├── grid_topology.py       # IEEE 14/30/57-bus system definitions
│   └── dataset.py             # Dataset generation & DataLoader
├── models/
│   ├── cnn_attention.py       # CNN + Attention network
│   └── detector.py            # Full detection pipeline
├── utils/
│   ├── attack_generator.py    # FDIA attack vector generation
│   ├── state_estimator.py     # Weighted Least Squares estimator
│   ├── metrics.py             # Detection rate, FAR, F1
│   └── visualize.py           # Plot results
├── train.py                   # Training script
├── evaluate.py                # Evaluation & comparison
├── demo.py                    # Quick demo / inference
└── requirements.txt
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
# Train on IEEE 14-bus (fast, ~5 min on CPU)
python train.py --system ieee14 --epochs 100

# Train on IEEE 30-bus (paper's main system)
python train.py --system ieee30 --epochs 100

# Train with GPU
python train.py --system ieee30 --epochs 100 --device cuda
```

### 3. Evaluate

```bash
# Evaluate on all attack types and compare with baselines
python evaluate.py --system ieee30 --checkpoint results/checkpoints/best_model.pt

# Evaluate generalization (train on 14-bus, test on 30-bus)
python evaluate.py --system ieee30 --checkpoint results/checkpoints/best_model_ieee14.pt
```

### 4. Quick demo

```bash
python demo.py
```

---

## Key Results (paper targets)

| Method            | Detection Rate | False Alarm Rate |
|-------------------|---------------|-----------------|
| Traditional (BDD) | 84.2%         | —               |
| SVM               | 88.5%         | —               |
| Random Forest     | 89.3%         | —               |
| Standard CNN      | 92.1%         | —               |
| **Proposed**      | **96.8%**     | **< 3%**        |

---

## Configuration

Edit `configs/config.yaml` to change:
- Network architecture (filters, window size, attention dim)
- Attack parameters (intensity range, sparsity)
- Training hyperparameters (lr, batch size, epochs)
- Test system (ieee14 / ieee30 / ieee57)
