# Reproduction Report: Federated Class-Continual Learning (FCIL)

This repository contains the reproduction and extension of the **FedCBDR (Federated Class-Incremental Learning with Global-Perspective Data Replay)** paper. We stress-tested the original paper's State-of-the-Art (SOTA) configurations under highly constrained edge-computing scenarios (fewer communication rounds, limited memory buffers, and fewer clients) to uncover the physical boundaries and algorithmic behaviors of the architecture.

---

## Features Implemented

* **Federated Class-Incremental Learning Pipeline**: Standard FCIL framework with FedAvg aggregation.
* **Non-IID Data Partitioning**: Dirichlet distribution-based data splitting across clients ($\beta$ or $\alpha$).
* **Task-aware Temperature Scaling (TTS)**: Custom loss function to prevent catastrophic forgetting.
* **Advanced Replay Buffer Strategies**:
    * `none`: Finetuning baseline (no buffer).
    * `random`: Random selection baseline.
    * `gdr`: Global-perspective Data Replay (exact paper implementation using Pseudo Features, Server-side SVD, and Leverage Scores).
    * `kmeans`: Local feature K-Means selection.
    * `svd_kmeans`: Local SVD compression + K-Means (extension for better Tabular performance).
* **Dataset Support**: Vision datasets (CIFAR-10, CIFAR-100, TinyImageNet) and Tabular Data (Letter/Digit Recognition).

---

## How to Run

### 1. Prerequisites
Ensure you have PyTorch and other required libraries installed. You can install them by running:
```bash
pip install -r requirements.txt
```
### 2. Running Vision Experiments
To run experiments on image datasets (CIFAR-10, CIFAR-100, or TinyImageNet), execute:

```bash
python main_tts.py
```

### 3. Running Tabular Experiments
To run experiments on the Letter/Digit Recognition tabular dataset, execute:

```bash
python main_tabular.py
```

---

## Configuration

The configurations are hardcoded at the top of the main execution files (`main_tts.py` and `main_tabular.py`). You can adjust the experimental setup by editing these global variables before running the scripts:

### Core Settings
* `DATASET`: Choose between `'cifar10'`, `'cifar100'`, or `'tinyimagenet'` (in `main_tts.py`).
* `CONFIG_BUFFER_TYPE`: Select the memory replay strategy. Options include `'none'`, `'random'`, `'gdr'`, `'kmeans'`, and `'svd_kmeans'`.
* `CONFIG_USE_TTS`: Set to `True` to enable Task-aware Temperature Scaling Loss, or `False` to use standard Cross Entropy Loss.

### Federated Learning Settings
* `NUM_CLIENTS`: Total number of simulated clients (e.g., 5 or 10).
* `NUM_ROUNDS_PER_TASK`: Number of federated communication rounds per incremental task.
* `DIRICHLET_ALPHA`: Data heterogeneity parameter ($\beta$ or $\alpha$). Common values are 0.1 (high non-IID), 0.5, or 1.0 (more uniform).

---

## Viewing Results
After an experiment finishes, a CSV file log will be automatically generated in the root directory (e.g., `report_metrics_gdr_tts.csv`). You can use the provided Jupyter Notebook (`visualize_metrics.ipynb`) to read these CSV files and automatically generate comparison accuracy charts.
