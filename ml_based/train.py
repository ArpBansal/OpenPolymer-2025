# --- XGB Hyperparameter Grid Search with DB Caching ---
from xgboost import XGBRegressor
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
from utils import *
from config import *
from rdkit import Chem
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, StackingRegressor
from lightgbm import LGBMRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor
from data_processing import combine_train_val, prepare_label_data, smiles_to_combined_fingerprints_with_descriptors
import numpy as np
import time
import torch
import torch.nn as nn
import pandas as pd
import os
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
from data_processing import get_canonical_smiles, drop_correlated_features


def xgb_grid_search_with_db(X, y, param_grid, db_path="xgb_tuning.db"):
    """
    For each param set in grid, check DB. If not present, train and save result.
    param_grid: dict of param lists, e.g. {'max_depth':[3,5], 'learning_rate':[0.01,0.1]}
    """
    tried = 0
    best_score = None
    best_params = None
    # Split X, y into train/val for early stopping
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    for params in ParameterGrid(param_grid):
        param_hash = get_param_hash(params)
        if check_db_for_params(db_path, param_hash):
            print(f"Skipping already tried params: {params}")
            continue
        # print(f"Trying params: {json.dumps(params, sort_keys=True)}")
        model = XGBRegressor(**params)
        # Provide eval_set for early stopping
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        y_pred = model.predict(X_val)
        from sklearn.metrics import mean_absolute_error
        score = mean_absolute_error(y_val, y_pred)
        print(f"Result: MAE={score:.6f} for params: {json.dumps(params, sort_keys=True)}")
        # For MAE, lower is better
        if (best_score is None) or (score < best_score):
            best_score = score
            best_params = params.copy()
            print(f"New best MAE: {best_score:.6f} with params: {json.dumps(best_params, sort_keys=True)}")
        save_result_to_db(db_path, param_hash, params, score)
        tried += 1
    print(f"Tried {tried} new parameter sets.")
    if best_score is not None:
        print(f"Best score overall: {best_score:.6f} with params: {json.dumps(best_params, sort_keys=True)}")

from sklearn.linear_model import RidgeCV, ElasticNetCV


def train_with_other_models(model_name, label, X_train, y_train, X_val, y_val):
    """
    Train a regression model using the specified model_name, with hyperparameters
    adapted to the data size of the target label.
    """
    print(f"Training {model_name} model for label: {label}")
    if model_name == 'tabnet':
        try:
            from pytorch_tabnet.tab_model import TabNetRegressor
        except ImportError:
            raise ImportError("pytorch-tabnet is not installed. Please install it with 'pip install pytorch-tabnet'.")
        
        # --- Define TabNet parameters based on label ---
        if label in ['Rg', 'Tc']: # Low Data
            params = {'n_d': 8, 'n_a': 8, 'n_steps': 3, 'gamma': 1.3, 'lambda_sparse': 1e-4}
        elif label == 'FFV': # High Data
            params = {'n_d': 24, 'n_a': 24, 'n_steps': 5, 'gamma': 1.5, 'lambda_sparse': 1e-5}
        else: # Medium Data
            params = {'n_d': 16, 'n_a': 16, 'n_steps': 5, 'gamma': 1.5, 'lambda_sparse': 1e-5}

        Model = TabNetRegressor(**params, seed=42, verbose=0)
        Model.fit(
            X_train.values, y_train.values.reshape(-1, 1),
            eval_set=[(X_val.values, y_val.values.reshape(-1, 1))],
            eval_metric=['mae'], # ACTION: Changed from 'rmse' to 'mae'
            max_epochs=200, patience=20, batch_size=1024, virtual_batch_size=128
        )

    elif model_name == 'catboost':
        # --- Define CatBoost parameters based on label ---
        params = {'iterations': 3000, 'learning_rate': 0.05, 'loss_function': 'MAE', 'eval_metric': 'MAE', 'random_seed': Config.random_state, 'verbose': False}
        if label in ['Rg', 'Tc']: # Low Data
            params.update({'depth': 5, 'l2_leaf_reg': 7})
        elif label == 'FFV': # High Data
            params.update({'depth': 7, 'l2_leaf_reg': 2})
        else: # Medium Data
            params.update({'depth': 6, 'l2_leaf_reg': 3})

        Model = CatBoostRegressor(**params)
        Model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, use_best_model=True)

    elif model_name == 'lgbm':
        # --- Define LightGBM parameters based on label ---
        params = {'n_estimators': 3000, 'learning_rate': 0.05, 'objective': 'mae', 'random_state': Config.random_state, 'verbose': -1, 'verbosity': -1}
        if label in ['Rg', 'Tc']: # Low Data
            params.update({'max_depth': 4, 'num_leaves': 20, 'reg_lambda': 5.0})
        elif label == 'FFV': # High Data
            params.update({'max_depth': 7, 'num_leaves': 40, 'reg_lambda': 1.0})
        else: # Medium Data
            params.update({'max_depth': 6, 'num_leaves': 31, 'reg_lambda': 1.0})

        Model = LGBMRegressor(**params)
        Model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='mae', callbacks=[lgb.early_stopping(stopping_rounds=50)])

    elif model_name == 'extratrees':
        # --- Define ExtraTrees parameters based on label ---
        params = {'n_estimators': 300, 'criterion': 'absolute_error', 'random_state': Config.random_state, 'n_jobs': -1}
        if label in ['Rg', 'Tc']: # Low Data: Prevent overfitting by requiring more samples per leaf
            params.update({'min_samples_leaf': 3, 'max_features': 0.8})
        else: # High/Medium Data
            params.update({'min_samples_leaf': 1, 'max_features': 1.0})
            
        Model = ExtraTreesRegressor(**params)
        X_all, y_all = combine_train_val(X_train, X_val, y_train, y_val)
        Model.fit(X_all, y_all)

    elif model_name == 'randomforest':
        # --- Define RandomForest parameters based on label ---
        params = {'n_estimators': 1000, 'criterion': 'absolute_error', 'random_state': Config.random_state, 'n_jobs': -1}
        if label in ['Rg', 'Tc']: # Low Data
            params.update({'min_samples_leaf': 3, 'max_features': 0.8, 'max_depth': 15})
        else: # High/Medium Data
            params.update({'min_samples_leaf': 1, 'max_features': 1.0, 'max_depth': None})

        Model = RandomForestRegressor(**params)
        X_all, y_all = combine_train_val(X_train, X_val, y_train, y_val)
        Model.fit(X_all, y_all)

    elif model_name == 'hgbm':
        from sklearn.ensemble import HistGradientBoostingRegressor
        # --- Define HGBM parameters based on label ---
        params = {'max_iter': 1000, 'learning_rate': 0.05, 'loss': 'absolute_error', 'early_stopping': True, 'random_state': 42} # ACTION: Changed loss to 'absolute_error'
        if label in ['Rg', 'Tc']: # Low Data
            params.update({'max_depth': 4, 'l2_regularization': 1.0})
        elif label == 'FFV': # High Data
            params.update({'max_depth': 7, 'l2_regularization': 0.1})
        else: # Medium Data
            params.update({'max_depth': 6, 'l2_regularization': 0.5})

        Model = HistGradientBoostingRegressor(**params)
        X_all, y_all = combine_train_val(X_train, X_val, y_train, y_val)
        Model.fit(X_all, y_all)

    elif model_name == 'nn':
        # The 'train_with_nn' function already uses different configs per label, which is excellent!
        # Just ensure the loss function inside it is nn.L1Loss()
        Model = train_with_nn(label, X_train, X_val, y_train, y_val)
    else:
        raise ValueError(f"Unknown or unavailable model: {model_name}")
    
    return Model

def train_with_autogluon(label, X_train, y_train):
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError:
        raise ImportError("AutoGluon is not installed. Please install it with 'pip install autogluon'.")
    import pandas as pd
    import uuid
    # Prepare data for AutoGluon (must be DataFrame with column names)
    X_train_df = pd.DataFrame(X_train)
    y_train_series = pd.Series(y_train, name=label)
    train_data = X_train_df.copy()
    train_data[label] = y_train_series.values

    unique_path = f"autogluon_{label}_{int(time.time())}_{uuid.uuid4().hex}"

    hyperparameters = {
        "GBM": {},
        "CAT": {},
        "XGB": {},
        "NN_TORCH": {},
        "RF": {},
        "XT": {}
    }

    hyperparameter_tune_kwargs = {
        "num_trials": 50,
        "scheduler": "local",
        "searcher": "auto"
    }

    time_limit = 300 if getattr(Config, 'debug', False) else 3600

    predictor = TabularPredictor(
        label=label,
        eval_metric="mae",  # Use 'mae' for regression
        path=unique_path
    ).fit(
        train_data,
        presets="best_quality",
        hyperparameters=hyperparameters,
        hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
        num_bag_folds=5,
        num_stack_levels=2,
        time_limit=time_limit
    )

    print("\n[AutoGluon] Leaderboard:")
    leaderboard = predictor.leaderboard(silent=False)

    print("\n[AutoGluon] Model Info:")
    print(predictor.info())

    print("\n[AutoGluon] Model Names:")
    model_names = leaderboard["model"].tolist()   # <- FIXED here
    print(model_names)

    # Save feature importance to CSV
    fi_df = predictor.feature_importance(train_data)
    fi_path = f"NeurIPS/autogluon_feature_importance_{label}.csv"  
    fi_df.to_csv(fi_path)
    print(f"[AutoGluon] Feature importance saved to {fi_path}")

    return predictor

def train_model(
    model,
    X_train, X_val, y_train, y_val,
    epochs=3000, batch_size=32, lr=1e-3, weight_decay=1e-4, patience=30, verbose=True
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_val = np.asarray(X_val)
    y_val = np.asarray(y_val)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    use_early_stopping = X_val is not None and y_val is not None
    loss = torch.tensor(0.0)
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
        if verbose and (epoch+1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

        if use_early_stopping:
            model.eval()
            with torch.no_grad():
                val_preds = model(X_val_tensor)
                val_loss = criterion(val_preds, y_val_tensor).item()
            scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}, best val loss: {best_val_loss:.4f}")
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                break

    return model


def train_with_stacking(label, X_train, y_train):
    """
    Trains XGBoost, ExtraTrees, and CatBoost using sklearn's StackingRegressor.
    Returns the fitted stacking model and base models.
    """
    estimators = [
    ('xgb', XGBRegressor(n_estimators=10000, learning_rate=0.01, max_depth=5, subsample=0.8, colsample_bytree=0.8, gamma=0.1, reg_lambda=1.0, objective="reg:absoluteerror", random_state=Config.random_state, n_jobs=-1)),
    ('lgbm', LGBMRegressor(n_estimators=10000, learning_rate=0.01, num_leaves=64, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5, reg_lambda=1.0, max_depth=-1, objective="mae", random_state=Config.random_state, n_jobs=-1, verbose=-1)),
        ('cb', CatBoostRegressor(iterations=10000, learning_rate=0.01, depth=7, l2_leaf_reg=5, bagging_temperature=0.8, random_seed=Config.random_state, verbose=0)),
    ('rf', RandomForestRegressor(n_estimators=500, max_depth=12, min_samples_leaf=3, random_state=Config.random_state, n_jobs=-1, criterion='absolute_error'))
    ]

    # Candidate final estimators
    final_estimators = {
        "ElasticNet": ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], n_alphas=100, max_iter=50000, tol=1e-3, cv=5, n_jobs=-1),
        "CatBoost": CatBoostRegressor(iterations=3000, learning_rate=0.03, depth=6, l2_leaf_reg=3, random_seed=Config.random_state, verbose=0),
    "LightGBM": LGBMRegressor(n_estimators=3000, learning_rate=0.03, num_leaves=64, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5, reg_lambda=1.0, max_depth=-1, objective="mae", random_state=Config.random_state, n_jobs=-1, verbose=-1),
    "XGBoost": XGBRegressor(n_estimators=3000, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, gamma=0.1, objective="reg:absoluteerror", random_state=Config.random_state, n_jobs=-1)
    }

    final_estimator = final_estimators['ElasticNet']
    
    stacker = StackingRegressor(estimators=estimators, final_estimator=final_estimator, passthrough=True, cv=5, n_jobs=-1)

    stacker.fit(X_train, y_train)
    return stacker

class FeedforwardNet(nn.Module):
    def __init__(self, input_dim, neurons, dropouts):
        super().__init__()
        layers = []
        for i, n in enumerate(neurons):
            layers.append(nn.Linear(input_dim, n))
            # layers.append(nn.BatchNorm1d(n))
            layers.append(nn.ReLU())
            if i < len(dropouts) and dropouts[i] > 0:
                layers.append(nn.Dropout(dropouts[i]))
            input_dim = n
        layers.append(nn.Linear(input_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def train_with_nn(label, X_train, X_val, y_train, y_val):

    input_dim = X_train.shape[1]
    
    if getattr(Config, "search_nn", False):
        print(f"--- Starting targeted NN architecture search for label: {label} ---")
        
        search_configs_by_label = {
            'Low': [ # For Rg, Tc. Simple models with high regularization.
                # --- Single Layer Focus ---
                {"neurons": [32], "dropouts": [0.3]},
                {"neurons": [64], "dropouts": [0.4]},
                {"neurons": [128], "dropouts": [0.5]},
                {"neurons": [256], "dropouts": [0.5]},

                # --- Two Layer Rectangular Focus (based on Rg's winner) ---
                {"neurons": [64, 64], "dropouts": [0.4, 0.4]},
                {"neurons": [96, 96], "dropouts": [0.5, 0.5]},
                {"neurons": [128, 128], "dropouts": [0.5, 0.5]}, # Previous winner
                {"neurons": [192, 192], "dropouts": [0.5, 0.5]},
                {"neurons": [256, 256], "dropouts": [0.5, 0.5]},

                # --- Two Layer Tapering Focus ---
                {"neurons": [128, 32], "dropouts": [0.5, 0.3]},
                {"neurons": [128, 64], "dropouts": [0.5, 0.4]},
                {"neurons": [256, 64], "dropouts": [0.5, 0.4]},
                {"neurons": [256, 128], "dropouts": [0.5, 0.4]},
                {"neurons": [512, 128], "dropouts": [0.5, 0.4]},

                # --- Three Layer Focus ---
                {"neurons": [64, 64, 64], "dropouts": [0.4, 0.4, 0.4]},
                {"neurons": [128, 128, 128], "dropouts": [0.5, 0.5, 0.5]},
                {"neurons": [128, 64, 32], "dropouts": [0.5, 0.4, 0.3]},
                {"neurons": [256, 128, 64], "dropouts": [0.5, 0.4, 0.3]},
                {"neurons": [256, 64, 32], "dropouts": [0.5, 0.4, 0.3]},
                {"neurons": [512, 128, 32], "dropouts": [0.5, 0.4, 0.3]},
            ],

            'Medium': [ # For Tg, Density. Balanced complexity.
                # --- Two Layer Focus ---
                {"neurons": [256, 64], "dropouts": [0.4, 0.3]},
                {"neurons": [256, 128], "dropouts": [0.4, 0.3]},
                {"neurons": [512, 64], "dropouts": [0.5, 0.3]},
                {"neurons": [512, 128], "dropouts": [0.5, 0.4]}, # Previous winner
                {"neurons": [512, 256], "dropouts": [0.5, 0.4]},
                {"neurons": [1024, 128], "dropouts": [0.5, 0.4]},
                {"neurons": [1024, 256], "dropouts": [0.5, 0.4]},
                {"neurons": [256, 256], "dropouts": [0.4, 0.4]},
                {"neurons": [512, 512], "dropouts": [0.5, 0.5]},

                # --- Three Layer Focus ---
                {"neurons": [256, 128, 64], "dropouts": [0.4, 0.3, 0.2]},
                {"neurons": [512, 128, 64], "dropouts": [0.5, 0.4, 0.3]},
                {"neurons": [512, 256, 64], "dropouts": [0.5, 0.4, 0.2]},
                {"neurons": [512, 256, 128], "dropouts": [0.5, 0.4, 0.3]}, # Previous winner
                {"neurons": [1024, 256, 64], "dropouts": [0.5, 0.4, 0.3]},
                {"neurons": [1024, 512, 128], "dropouts": [0.5, 0.4, 0.3]},
                {"neurons": [1024, 512, 256], "dropouts": [0.5, 0.4, 0.3]},
                {"neurons": [256, 256, 256], "dropouts": [0.4, 0.4, 0.4]},
                {"neurons": [512, 512, 512], "dropouts": [0.5, 0.5, 0.5]},

                # --- Four Layer Focus ---
                {"neurons": [512, 256, 128, 64], "dropouts": [0.5, 0.4, 0.3, 0.2]},
                {"neurons": [1024, 512, 256, 128], "dropouts": [0.5, 0.4, 0.3, 0.2]},
            ],

            'High': [ # For FFV. Exploring width and depth.
                # --- Refining Around Winner ([512, 256]) ---
                {"neurons": [512, 128], "dropouts": [0.3, 0.2]},
                {"neurons": [512, 256], "dropouts": [0.3, 0.2]}, # Previous winner
                {"neurons": [512, 512], "dropouts": [0.3, 0.3]},
                {"neurons": [1024, 256], "dropouts": [0.4, 0.3]},
                {"neurons": [1024, 512], "dropouts": [0.4, 0.3]},
                {"neurons": [1024, 1024], "dropouts": [0.4, 0.4]},
                {"neurons": [2048, 512], "dropouts": [0.5, 0.4]},
                {"neurons": [2048, 1024], "dropouts": [0.5, 0.4]},

                # --- Three Layer Focus ---
                {"neurons": [512, 256, 128], "dropouts": [0.3, 0.2, 0.2]},
                {"neurons": [1024, 256, 64], "dropouts": [0.4, 0.3, 0.2]},
                {"neurons": [1024, 512, 128], "dropouts": [0.4, 0.3, 0.2]},
                {"neurons": [1024, 512, 256], "dropouts": [0.4, 0.3, 0.2]},
                {"neurons": [2048, 512, 128], "dropouts": [0.5, 0.4, 0.3]},
                {"neurons": [2048, 1024, 512], "dropouts": [0.5, 0.4, 0.3]},
                {"neurons": [512, 512, 512], "dropouts": [0.3, 0.3, 0.3]},
                {"neurons": [1024, 1024, 1024], "dropouts": [0.4, 0.4, 0.4]},

                # --- Four+ Layer Focus ---
                {"neurons": [512, 256, 256, 128], "dropouts": [0.3, 0.2, 0.2, 0.1]},
                {"neurons": [1024, 512, 256, 128], "dropouts": [0.4, 0.3, 0.2, 0.2]},
                {"neurons": [1024, 512, 512, 256], "dropouts": [0.4, 0.3, 0.3, 0.2]},
                {"neurons": [512, 512, 512, 512], "dropouts": [0.3, 0.3, 0.3, 0.3]},
            ]
        }
        
        # Determine which set of configs to use
        if label in ['Rg', 'Tc']:
            configs_to_search = search_configs_by_label['Low']
            print("Using search space for LOW data targets.")
        elif label == 'FFV':
            configs_to_search = search_configs_by_label['High']
            print("Using search space for HIGH data targets.")
        else: # Tg, Density
            configs_to_search = search_configs_by_label['Medium']
            print("Using search space for MEDIUM data targets.")

        results = []
        for i, cfg in enumerate(configs_to_search):
            print(f"\n---> Searching config {i+1}/{len(configs_to_search)}: Neurons={cfg['neurons']}, Dropouts={cfg['dropouts']}")
            model = FeedforwardNet(input_dim, cfg["neurons"], cfg["dropouts"])
            model = train_model(model, X_train, X_val, y_train, y_val, verbose=False) # Turn off verbose for cleaner search logs
            
            model.eval()
            with torch.no_grad():
                X_val_np = np.asarray(X_val) if isinstance(X_val, pd.DataFrame) else X_val
                device = next(model.parameters()).device
                X_val_tensor = torch.tensor(X_val_np, dtype=torch.float32).to(device)
                y_pred = model(X_val_tensor).cpu().numpy().flatten()
            
            val_mae = mean_absolute_error(y_val, y_pred)
            print(f"     Resulting Val MAE: {val_mae:.6f}")
            results.append({"neurons": cfg["neurons"], "dropouts": cfg["dropouts"], "val_mae": val_mae})
        
        df = pd.DataFrame(results)
        print("\n--- Neural Network Search Results ---")
        print(df.sort_values(by='val_mae').to_string(index=False))
        
        df.to_csv(f"nn_config_validation_mae_{label}.csv", index=False)
        print(f"\nSaved results to nn_config_validation_mae_{label}.csv")
        
        best_row = df.loc[df['val_mae'].idxmin()]
        print(f"\nBest config: neurons={best_row['neurons']}, dropouts={best_row['dropouts']}, Validation MAE: {best_row['val_mae']:.6f}")
        
        config = best_row.to_dict()
        print("Re-training best model on the full training data...")
    
    else: # If not searching, use a single pre-defined configuration
        best_configs = {
            "Tg":      {"neurons": [256, 128, 64], "dropouts": [0.4, 0.3, 0.2]},
            "Density": {"neurons": [256, 128, 64], "dropouts": [0.4, 0.3, 0.2]},
            "FFV":     {"neurons": [512, 256, 128], "dropouts": [0.3, 0.2, 0.2]},
            "Tc":      {"neurons": [128, 64], "dropouts": [0.4, 0.3]},
            "Rg":      {"neurons": [128, 64], "dropouts": [0.4, 0.3]},
        }
        config = best_configs.get(label)
        print(f"Using pre-defined best config for {label}: Neurons={config['neurons']}, Dropouts={config['dropouts']}")

    # Final model training
    best_model = FeedforwardNet(input_dim, config["neurons"], config["dropouts"])
    show_model_summary(best_model, input_dim)
    best_model = train_model(best_model, X_train, X_val, y_train, y_val, verbose=True)
    return best_model

def train_with_xgb(label, X_train, y_train, X_val, y_val):
    print(f"Training XGB model for label: {label}")
    if label=="Tg": # Medium Data (~1.2k samples)
        Model = XGBRegressor(
            n_estimators=10000, learning_rate=0.01, max_depth=5, # <-- Reduced from 6
            colsample_bytree=1.0, reg_lambda=7.0, gamma=0.1, subsample=0.5, # <-- Increased lambda, reduced subsample
            objective="reg:absoluteerror", random_state=Config.random_state, 
            early_stopping_rounds=50, eval_metric="mae"
        )
    elif label=='Rg': # Very Low Data (~600 samples), HIGHEST PRIORITY
        Model = XGBRegressor(
            n_estimators=10000, learning_rate=0.06, max_depth=4, 
            colsample_bytree=1.0, reg_lambda=10.0, gamma=0.1, subsample=0.6, 
            objective="reg:absoluteerror", random_state=Config.random_state, 
            early_stopping_rounds=50, eval_metric="mae"
        )
    elif label=='FFV': # High Data (~8k samples), LOWEST PRIORITY
        Model = XGBRegressor(
            n_estimators=10000, learning_rate=0.06, max_depth=7, 
            colsample_bytree=0.8, reg_lambda=2.0, gamma=0.0, subsample=0.6, 
            objective="reg:absoluteerror", random_state=Config.random_state, 
            early_stopping_rounds=50, eval_metric="mae"
        )
    elif label=='Tc': # Low Data (~900 samples), HIGH PRIORITY
        Model = XGBRegressor(
            n_estimators=10000, learning_rate=0.01, max_depth=4, 
            colsample_bytree=0.8, reg_lambda=7.0, gamma=0.0, subsample=0.6, 
            objective="reg:absoluteerror", random_state=Config.random_state, 
            early_stopping_rounds=50, eval_metric="mae"
        )
    elif label=='Density': # Medium Data (~1.2k samples)
        Model = XGBRegressor(
            n_estimators=10000, learning_rate=0.06, max_depth=5, 
            colsample_bytree=1.0, reg_lambda=3.0, gamma=0.0, subsample=0.8, 
            objective="reg:absoluteerror", random_state=Config.random_state, 
            early_stopping_rounds=50, eval_metric="mae"
        )
        
    print(f"Model {label} trained with shape: {X_train.shape}, {y_train.shape}")

    Model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    return Model

def train_and_evaluate_models(label, X_main, y_main, splits, nfold, Config):
    """
    Trains models for the given label using the specified configuration.
    Returns: models, fold_maes, mean_fold_mae, std_fold_mae
    """
    # Use a prime multiplier for folds
    FOLD_PRIME = 9973   # a large prime

    models = []
    fold_maes = []
    mean_fold_mae = None
    std_fold_mae = None

    # Use stacking only if enabled in config
    if getattr(Config, 'use_stacking', False):
        Model = train_with_stacking(label, X_main, y_main)
        models.append(Model)
        save_model(Model, label, 1, Config.model_name)
    elif Config.model_name in ['autogluon']:
        Model = train_with_autogluon(label, X_main, y_main)
        models.append(Model)
        # save_model(Model, label, 1, Config.model_name)
    else:
        for fold, (train_idx, val_idx) in enumerate(splits):
            print(f"\n--- Fold {fold+1}/{nfold} ---")
            # Set a different random seed for each fold for best possible result
            # Use a deterministic but varied seed: base + fold + hash(label)
            base_seed = getattr(Config, 'random_state', 42)
            label_hash = stable_hash(label)   # replaces abs(hash(label)) % 10000
            fold_seed = base_seed + fold * FOLD_PRIME  + label_hash

            set_global_random_seed(fold_seed, config=Config)

            # Robustly handle both DataFrame and ndarray
            if isinstance(X_main, np.ndarray):
                X_train, X_val = X_main[train_idx], X_main[val_idx]
            else:
                X_train, X_val = X_main.iloc[train_idx], X_main.iloc[val_idx]
            if isinstance(y_main, np.ndarray):
                y_train, y_val = y_main[train_idx], y_main[val_idx]
            else:
                y_train, y_val = y_main.iloc[train_idx], y_main.iloc[val_idx]

            if Config.model_name == 'xgb':
                Model = train_with_xgb(label, X_train, y_train, X_val, y_val)
            elif Config.model_name in ['catboost', 'lgbm', 'extratrees', 'randomforest', 'balancedrf', 'tabnet', 'hgbm', 'autogluon', 'nn']:
                Model = train_with_other_models(Config.model_name, label, X_train, y_train, X_val, y_val)
            else:
                assert False, "No model present. Set Config.use_train_with_xgb = True to train a model."

            # Save model for later holdout prediction
            models.append(Model)
            save_model(Model, label, fold, Config.model_name)

            # Predict on validation set for this fold
            if hasattr(Model, 'forward') and not hasattr(Model, 'predict'):
                Model.eval()
                X_val_np = np.asarray(X_val) if isinstance(X_val, pd.DataFrame) else X_val
                device = next(Model.parameters()).device
                with torch.no_grad():
                    X_val_tensor = torch.tensor(X_val_np, dtype=torch.float32).to(device)
                    y_val_pred = Model(X_val_tensor).cpu().numpy().flatten()
            else:
                y_val_pred = Model.predict(X_val)

            fold_mae = mean_absolute_error(y_val, y_val_pred)
            print(f"Fold {fold+1} MAE (on validation set): {fold_mae}")
            fold_maes.append(fold_mae)
            # Save y_val, y_val_pred, and residuals in sorted order for each fold
            residuals = y_val - y_val_pred
            results_df = pd.DataFrame({
                'y_val': y_val,
                'y_val_pred': y_val_pred,
                'residual': residuals
            })
            results_df = results_df.sort_values(by='residual', ascending=False).reset_index(drop=True)
            os.makedirs(f'NeurIPS/fold_residuals/{label}', exist_ok=True)
            results_df.to_csv(f'NeurIPS/fold_residuals/{label}/fold_{fold+1}_val_pred_residuals.csv', index=False)
        mean_fold_mae = np.mean(fold_maes)
        std_fold_mae = np.std(fold_maes)
        print(f"{label} 5-Fold CV mean_absolute_error (on validation sets): {mean_fold_mae} ± {std_fold_mae}")

    return models, fold_maes, mean_fold_mae, std_fold_mae

def train_or_predict(labels, subtables, test_smiles, test_ids, output_df, train_model=True, model_dir=None):
    mae_results = []
    
    for label in labels:
        if train_model:
            print(f"\n=== Training/Predicting for label: {label} ===")
            label_data = prepare_label_data(label, subtables, Config)
            X_main = label_data["X_main"]
            X_holdout = label_data["X_holdout"]
            y_main = label_data["y_main"]
            y_holdout = label_data["y_holdout"]
            kept_columns = label_data["kept_columns"]
            scaler = label_data["scaler"]
            median_values = label_data["median_values"]
            least_important_features = label_data["least_important_features"]
            correlated_features_dropped = label_data["correlated_features_dropped"]
            selector = label_data["selector"]
            selected_cols_variance = label_data["selected_cols_variance"]
            pca = label_data["pca"]
            splits = label_data["splits"]
            nfold = label_data["nfold"]
            fold_maes = label_data["fold_maes"]
            test_preds = label_data["test_preds"]
            val_preds = label_data["val_preds"]

            # --- Save feature selection info for this label ---
            save_feature_selection_info(label, kept_columns, least_important_features, correlated_features_dropped, scaler, X_holdout, y_holdout, median_values)  
            os.makedirs('models', exist_ok=True)
            models = []

            # labels = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

            # --- Hyperparameter tuning for this label ---
            if Config.enable_param_tuning:
                if label == 'Tg':
                    db_path = f"xgb_tuning_{label}.db"
                    init_xgb_tuning_db(db_path)
                    # Example param_grid (customize as needed)
                    param_grid = {
                        'n_estimators': [3000],
                        'max_depth': [4, 5, 6, 7],
                        'learning_rate': [0.001, 0.01, 0.06, 0.1],
                        'subsample': [0.6, 0.8, 1.0],
                        'colsample_bytree': [0.8, 1.0],
                        'gamma': [0, 0.1],
                        'reg_lambda': [1.0, 5.0, 10.0],
                        'early_stopping_rounds': [50],
                        'objective': ["reg:squarederror"],
                        'eval_metric': ["rmse"]
                    }
                    # You must define X_train and y_train for tuning here
                    xgb_grid_search_with_db(X_main, y_main, param_grid, db_path=db_path)
                else:
                    continue
            # FIXME save features_keep
            # Ensure only features present in X_main are kept

            # if label in ['Tg', 'Tc', 'Density', 'Rg']:
            #     features_keep = select_features_with_lasso(X_main, y_main, label)
            #     X_main = X_main[features_keep]
            models, fold_maes, mean_fold_mae, std_fold_mae = train_and_evaluate_models(label, X_main, y_main, splits, nfold, Config)
        else:
            print(f"\n=== Loading models and data for label: {label} ===")
            mean_fold_mae, std_fold_mae = None, None
            feature_info = load_feature_selection_info(label, model_dir)
            kept_columns = feature_info["kept_columns"]
            least_important_features = feature_info["least_important_features"]
            correlated_features_dropped = feature_info["correlated_features_dropped"]
            scaler = feature_info.get("scaler", None)
            X_holdout = feature_info["X_holdout"]
            y_holdout = feature_info["y_holdout"]
            median_values = feature_info["median_values"]
            selector = None
            selected_cols_variance = None
            pca = None

            models = load_models_for_label(label, os.path.join(model_dir, 'models'))
            test_preds = []

        # Prepare test set once
        fp_df, descriptor_df, valid_smiles, invalid_indices = smiles_to_combined_fingerprints_with_descriptors(test_smiles)
        # median_values = label_data["median_values"]
        if not descriptor_df.empty:
            # Safely align test columns to training columns. 
            # This adds any missing columns and fills them with NaN.
            descriptor_df = descriptor_df.reindex(columns=kept_columns)
            if Config.model_name == 'nn':
                # Fill NaN in test with median from train
                descriptor_df = descriptor_df.fillna(median_values)
            # Scale test set using the same scaler and kept_columns, then convert to DataFrame
            if getattr(Config, 'use_standard_scaler', False) and scaler is not None:
                descriptor_df = pd.DataFrame(scaler.transform(descriptor_df), columns=kept_columns, index=descriptor_df.index)
            descriptor_df.reset_index(drop=True, inplace=True)
            if not fp_df.empty:
                fp_df = fp_df.reset_index(drop=True)
                test = pd.concat([descriptor_df, fp_df], axis=1)
            else:
                test = descriptor_df
        else:
            test = fp_df

        if len(least_important_features) > 0:
            test = test.drop(columns=least_important_features)
        if len(correlated_features_dropped) > 0:
            print(f"Dropping correlated columns from test: {correlated_features_dropped}")
            test = test.drop(correlated_features_dropped, axis=1, errors='ignore')
        if getattr(Config, 'use_variance_threshold', False):
            test_sel = selector.transform(test)
            # Convert back to DataFrame
            test = pd.DataFrame(test_sel, columns=selected_cols_variance, index=test.index)
        # Optionally apply PCA to test set if enabled
        if getattr(Config, 'use_pca', False):
            test = pca.transform(test)
        # if label in ['Tg', 'Tc', 'Density', 'Rg']:
        #     X_holdout = X_holdout[features_keep]
        #     test = test[features_keep]
        # --- Holdout set evaluation with all trained models ---
        holdout_maes = []
        for i, Model in enumerate(models):
            is_torch_model = hasattr(Model, 'forward') and not hasattr(Model, 'predict')

            if is_torch_model:
                Model.eval()
                X_holdout_np = np.asarray(X_holdout) if isinstance(X_holdout, pd.DataFrame) else X_holdout
                test_np = np.asarray(test) if isinstance(test, pd.DataFrame) else test
                device = next(Model.parameters()).device
                with torch.no_grad():
                    X_holdout_tensor = torch.tensor(X_holdout_np, dtype=torch.float32).to(device)
                    test_tensor = torch.tensor(test_np, dtype=torch.float32).to(device)
                    y_holdout_pred = Model(X_holdout_tensor).detach().cpu().numpy().flatten()
                    y_test_pred = Model(test_tensor).detach().cpu().numpy().flatten()
            else:
                y_holdout_pred = Model.predict(X_holdout)
                y_test_pred = Model.predict(test)

            holdout_mae = mean_absolute_error(y_holdout, y_holdout_pred)
            print(f"Model {i+1} holdout MAE: {holdout_mae}")
            holdout_maes.append(holdout_mae)

            if isinstance(y_test_pred, pd.Series):
                y_test_pred = y_test_pred.values.flatten()
            else:
                y_test_pred = y_test_pred.flatten()        
            test_preds.append(y_test_pred)

        mean_holdout_mae = np.mean(holdout_maes)
        std_holdout_mae = np.std(holdout_maes)
        print(f"{label} Holdout MAE (mean ± std over all models): {mean_holdout_mae:.5f} ± {std_holdout_mae:.5f}")

        mae_results.append({
            'label': label,
            'fold_mae_mean': mean_fold_mae,
            'fold_mae_std': std_fold_mae,
            'holdout_mae_mean': mean_holdout_mae,
            'holdout_mae_std': std_holdout_mae
        })

        # Average test predictions across folds
        test_preds = np.array(test_preds)
        y_pred = np.mean(test_preds, axis=0)
        print(y_pred)
        new_column_name = label
        output_df[new_column_name] = y_pred

    mae_df = pd.DataFrame(mae_results)
    mae_df.to_csv('NeurIPS/mae_results.csv', index=False)
    print("\nMean Absolute Error for each label:")
    print(mae_df)
