# OpenPolymer2025 - Kaggle Competition Solution

Structured repository for polymer property prediction using XGBoost and GNN ensemble models.


## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/ArpBansal/OpenPolymer-2025
cd OpenPolymer-2025
```

### 2. Activate the virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

These two below can be executed parallely:
```bash
python gnn_model/infer.py
```

```bash
python gnn_model/infer.py
```

Then at final for ensemble:
```bash
python main.py
```

## Usage
This code dir paths assumes as being in kaggle notebook


# The Approach

## Overview

This solution uses an **ensemble approach** combining two complementary machine learning methods:
1. **XGBoost** - Feature-rich gradient boosting on molecular descriptors and fingerprints
2. **Graph Neural Networks (GNN)** - Structure-aware deep learning on molecular graphs

The final predictions are obtained by **averaging** the outputs from both models, leveraging the strengths of traditional feature engineering and modern graph-based deep learning.

## Target Properties

The models predict 5 polymer properties:
1. **Tg** - Glass transition temperature
2. **FFV** - Fractional free volume
3. **Tc** - Critical temperature
4. **Density** - Material density
5. **Rg** - Radius of gyration

## Architecture

### 1. XGBoost Models (`ml_based/`)

**Model Architecture:**
- Separate XGBoost regressor for each of the 5 target properties
- Task-specific hyperparameters adapted to data availability:
  - Low data (Rg, Tc): Shallower trees (depth 4-5), higher regularization
  - Medium data (Tg, Density): Balanced parameters (depth 6)
  - High data (FFV): Deeper trees (depth 7), lower regularization

**Feature Engineering:**
- **Molecular Descriptors** (primary):
  - **Mordred descriptors** (~1600 features): Comprehensive 2D topological and physicochemical descriptors
  - **RDKit descriptors** (200+ features): Standard molecular properties
  - **Custom physicochemical features**:
    - Rigidity metrics: Aromatic rings, sp³ carbon fraction, rotatable bonds
    - Bulkiness indicators: Molar refractivity, accessible surface area
    - Shape indices: Balaban J, Kappa2
    - Intermolecular force proxies: H-bond donors/acceptors
- **Molecular Fingerprints** (optional):
  - MACCS keys (167 bits) - structural key-based fingerprint
  - Morgan fingerprints (128 bits, radius=2) - circular fingerprints
  - Atom pair fingerprints (128 bits)
  - Topological torsion fingerprints (128 bits)
- **Graph-based features**:
  - Graph diameter, average shortest path length, cycle count
- **Label-specific feature selection**: Each target uses 4 most relevant RDKit descriptors

**Training Details:**
- Loss function: Mean Absolute Error (MAE)
- Validation split: 80-20 train-val split
- Early stopping: 50 rounds patience on validation MAE
- 3000 iterations with learning rate 0.05

### 2. Graph Neural Network Models (`gnn_model/`)

**Model Architecture:**
- **Backbone**: Graph Attention Networks (GAT) with 2 convolutional layers
  - Layer 1: Multi-head attention (8 heads) with concatenation
  - Layer 2: Multi-head attention (8 heads) with averaging
  - Hidden dimensions: 128 channels per layer
  - Edge-conditioned attention using bond features
- **Pooling**: Global mean pooling to aggregate node representations
- **Task-specific heads**: Dynamic MLP with configurable neurons and dropout per target
  - Combines graph embeddings with global molecular features
  - ELU activations with dropout regularization

**Graph Representation:**
- **Node features** (25 dimensions):
  - Atom type one-hot encoding (20 atom types: C, N, O, F, P, S, Cl, Br, I, H, Si, Na, *, B, Ge, Sn, Se, Te, Ca, Cd)
  - Atom degree, formal charge, total hydrogen count
  - Hybridization type, aromaticity
- **Edge features** (1 dimension):
  - Bond type as double (single=1, double=2, triple=3, aromatic=1.5)
- **Global features** (4 dimensions per target):
  - Task-specific RDKit descriptors selected for each target:
    - Tg: HallKierAlpha, MolLogP, NumRotatableBonds, TPSA
    - FFV: NHOHCount, NumRotatableBonds, MolWt, TPSA
    - Tc: MolLogP, NumValenceElectrons, SPS, MolWt
    - Density: MolWt, MolMR, FractionCSP3, NumHeteroatoms
    - Rg: HallKierAlpha, MolWt, NumValenceElectrons, qed

**Training Details:**
- Optimizer: Adam with learning rate 0.001
- Loss function: L1Loss (MAE)
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=10 epochs)
- Early stopping: Patience of 30 epochs on validation loss
- Batch size: 32
- Maximum epochs: 300

### 3. Ensemble Strategy

**Method**: Simple arithmetic mean of predictions
```python
final_prediction = (xgb_prediction + gnn_prediction) / 2
```

**Rationale**:
- XGBoost captures complex non-linear relationships through extensive feature engineering
- GNN leverages molecular graph structure and spatial relationships
- Averaging reduces model-specific biases and improves generalization

## Data Processing Pipeline

### Pre-processing

**1. SMILES Cleaning and Validation:**
- Remove problematic patterns: R-groups (`[R]`, `[R1]`, etc.), invalid fragments
- Validate all SMILES with RDKit
- Canonicalize SMILES strings for consistency
- Filter out unparseable molecules

**2. External Data Integration:**
- Incorporate multiple external polymer property databases
- SMILES-based data matching and merging
- Fill missing values from external sources
- Aggregate duplicate SMILES by averaging target values
- Typical gain: 1000-5000 additional samples per target

**3. Feature Scaling:**
- StandardScaler normalization for XGBoost features
- Min-max scaling for GNN node/edge features (implicitly handled)

**4. Feature Selection (XGBoost):**
- Remove highly correlated features (correlation > 0.96)
- Task-specific feature importance ranking
- Remove 15-25 least important features per target (model-dependent)
- Methods: Permutation importance, SHAP values

**5. Data Augmentation (optional):**
- SMILES augmentation: Randomized SMILES generation from same molecule
- Gaussian Mixture Models for synthetic sample generation
- Helps with low-data targets (Tc, Rg)

### Post-processing

**For XGBoost:**
- Direct predictions (no transformation needed)
- Predictions are already in target scale

**For GNN:**
- Direct predictions (no transformation needed)
- Model outputs are trained on original scale

**Ensemble:**
- Average predictions from both models
- Final output: `submission.csv` with columns `[id, Tg, FFV, Tc, Density, Rg]`

## Key Design Choices

1. **Task-Specific Models**: Separate models for each target allow specialized hyperparameter tuning and feature selection
2. **Hybrid Feature Representation**: Combines hand-crafted molecular descriptors (XGBoost) with learned graph representations (GNN)
3. **Extensive Data Cleaning**: Aggressive SMILES validation prevents training on invalid molecules
4. **External Data Utilization**: Significantly expands training data beyond competition dataset
5. **Simple Ensemble**: Arithmetic mean is robust and avoids overfitting to validation weights


Model weights and data at
https://www.kaggle.com/models/arpit1bansal/mix-models-neurips/

Citation
Gang Liu, Jiaxin Xu, Eric Inae, Yihan Zhu, Ying Li, Tengfei Luo, Meng Jiang, Yao Yan, Walter Reade, Sohier Dane, Addison Howard, and María Cruz. NeurIPS - Open Polymer Prediction 2025. https://kaggle.com/competitions/neurips-open-polymer-prediction-2025, 2025. Kaggle.