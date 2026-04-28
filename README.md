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

## Configuration

Edit `configs/config.yaml` to change:
- Network architecture (filters, window size, attention dim)
- Attack parameters (intensity range, sparsity)
- Training hyperparameters (lr, batch size, epochs)
- Test system (ieee14 / ieee30 / ieee57)

---

## Generalizability and Extensibility

The core deep learning architecture in this repository is fully generic, while the data generation pipeline is modularly tied to specific power grid topologies.

### 1. Deep Learning Model (Fully Generic)

The underlying neural network (`CNNAttentionDetector`) is **dataset-agnostic**.

- The model dynamically infers:
  - Number of measurement channels (`m`)
  - Temporal sliding window size (`w`)

These are automatically extracted from the initialized `DataLoader` during training.

If time-series data of different dimensions is provided:

- The **1D Convolutional layers** automatically adapt to the new input shape.
- The **Attention mechanism** scales its input weights accordingly.
- No hard-coded architectural changes are required.

This makes the model reusable across different grid sizes and measurement configurations without modifying the core deep learning code.

---

### 2. Data Generation Pipeline (Topology-Specific)

Because this project simulates physically realistic and stealthy **False Data Injection Attacks (FDIA)** using the Jacobian formulation:

\[
a = Hc
\]

the dataset generation process depends on the physical topology of the power grid.

Currently, the code is tested and validated for:

- IEEE 14-bus system  
- IEEE 30-bus system  

---

## Extending to a New Power System

To apply the framework to a new grid (for example, IEEE 118-bus), **no changes to the machine learning model are needed.**

You only need to:

### Step 1 — Define a New Grid Topology

Add the physical parameters of the new system inside:

```python
data/grid_topology.py
```

Create a new `GridSystem` class containing:

- Bus active/reactive loads  
- Branch resistance/reactance data  
- Any topology-specific parameters  

---

### Step 2 — Register the New System

Add the new system to the `get_system()` factory function.

---

## Summary

- **Model architecture:** Fully generic and topology-independent  
- **Data generation:** Modular and extensible to new grids  
- **Current support:** IEEE 14-bus, IEEE 30-bus  
- **Extending to new systems:** Add topology definition only — training pipeline remains unchanged

