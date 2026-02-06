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
pip install requirements.txt
```

## Usage
This code dir paths assumes as being in kaggle notebook


## Target Properties

The models predict 5 polymer properties:
1. **Tg** - Glass transition temperature
2. **FFV** - Fractional free volume
3. **Tc** - Critical temperature
4. **Density** - Material density
5. **Rg** - Radius of gyration


Model weights and data at
https://www.kaggle.com/models/arpit1bansal/mix-models-neurips/

Citation
Gang Liu, Jiaxin Xu, Eric Inae, Yihan Zhu, Ying Li, Tengfei Luo, Meng Jiang, Yao Yan, Walter Reade, Sohier Dane, Addison Howard, and María Cruz. NeurIPS - Open Polymer Prediction 2025. https://kaggle.com/competitions/neurips-open-polymer-prediction-2025, 2025. Kaggle.