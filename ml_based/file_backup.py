import glob
import os
import time
import random
import json
import hashlib
import joblib
import numpy as np
import pandas as pd
import networkx as nx

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, StackingRegressor
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LassoCV, ElasticNetCV
from autogluon.tabular import TabularDataset, TabularPredictor

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import lightgbm as lgb

from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys, rdmolops, Lipinski, Crippen
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator, GetAtomPairGenerator, GetTopologicalTorsionGenerator

from mordred import Calculator, descriptors as mordred_descriptors

import shap

from transformers import AutoTokenizer, AutoModel

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
BASE_PATH = '/kaggle/input/neurips-open-polymer-prediction-2025/'


# torchinfo is optional, only used in show_model_summary
try:
    from torchinfo import summary
except ImportError:
    summary = None

# --- XGB Hyperparameter Tuning DB Utilities ---

"""
Load competition data with complete filtering of problematic polymer notation
"""

# Load external datasets with robust error handling
print("\n📂 Loading external datasets...")

external_datasets = []

# --- Outlier Detection Summary Function ---

train_df=train_extended
test_df=test
subtables = separate_subtables(train_df)

test_smiles = test_df['SMILES'].tolist()
test_ids = test_df['id'].values
labels = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
#labels = ['Tc']

# Save importance_df to Excel log file, one sheet per label

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

def select_features_with_lasso(X, y, label):
    """
    Performs feature selection using Lasso (L1) regularization.

    Args:
        X (pd.DataFrame): The input feature matrix.
        y (pd.Series or np.array): The target values.
        label (str): The name of the target property (e.g., 'Rg', 'Tc').

    Returns:
        pd.DataFrame: A DataFrame containing only the selected features.
    """
    # Lasso is sensitive to feature scaling, so we scale the data first.
    # We also need to handle any potential NaN values before scaling.
    X_filled = X.fillna(X.median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filled)

    # Use LassoCV to automatically find the best alpha (regularization strength)
    # through cross-validation. This is more robust than picking a single alpha.
    lasso_cv = LassoCV(cv=5, random_state=42, n_jobs=-1, max_iter=2000)

    # Use SelectFromModel to wrap the LassoCV regressor.
    # This will select features where the Lasso coefficient is non-zero.
    # The 'threshold="median"' can be a good strategy to select the top 50% of features
    # if LassoCV is too lenient and keeps too many. Start with the default (None).
    feature_selector = SelectFromModel(lasso_cv, prefit=False, threshold=None)

    print(f"[{label}] Fitting LassoCV to find optimal features...")
    feature_selector.fit(X_scaled, y)

    # Get the names of the features that were kept
    selected_feature_names = X.columns[feature_selector.get_support()]

    print(f"[{label}] Original number of features: {X.shape[1]}")
    print(f"[{label}] Features selected by Lasso: {len(selected_feature_names)}")

    # Return the original DataFrame with only the selected columns
    return selected_feature_names


# Utility to set random state everywhere
def set_global_random_seed(seed, config=None):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    if config is not None:
        config.random_state = seed

import hashlib
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

def save_feature_selection_info(label, kept_columns, least_important_features, correlated_features_dropped, scaler, X_holdout, y_holdout, median_values):
    holdout_dir = f"NeurIPS/feature_selection/{label}"
    os.makedirs(holdout_dir, exist_ok=True)
    feature_info = {
        "kept_columns": list(kept_columns),
        "least_important_features": list(least_important_features),
        "correlated_features_dropped": list(correlated_features_dropped),
    }
    # Save median_values if provided
    if median_values is not None:
        if hasattr(median_values, 'to_dict'):
            feature_info["median_values"] = median_values.to_dict()
        else:
            feature_info["median_values"] = median_values

    # Save X_holdout and y_holdout for this label
    X_holdout_path = os.path.join(holdout_dir, "X_holdout.csv")
    y_holdout_path = os.path.join(holdout_dir, "y_holdout.csv")
    pd.DataFrame(X_holdout).to_csv(X_holdout_path, index=False)
    pd.DataFrame({"y_holdout": y_holdout}).to_csv(y_holdout_path, index=False)

    feature_info_path = os.path.join(holdout_dir, f"{label}_feature_info.json")
    with open(feature_info_path, "w") as f:
        json.dump(feature_info, f, indent=2)

    # Save scaler object
    scaler_path = os.path.join(holdout_dir, "scaler.joblib")
    if scaler is not None:
        joblib.dump(scaler, scaler_path)

def load_feature_selection_info(label, base_dir):
    """
    Loads feature selection info saved by save_feature_selection_info for a given label.
    Returns a dict with keys: kept_columns, least_important_features, correlated_features_dropped, scaler, X_holdout, y_holdout.
    """

    holdout_dir = os.path.join(base_dir, f"NeurIPS/feature_selection/{label}")
    feature_info_path = os.path.join(holdout_dir, f"{label}_feature_info.json")
    X_holdout_path = os.path.join(holdout_dir, "X_holdout.csv")
    y_holdout_path = os.path.join(holdout_dir, "y_holdout.csv")

    if not os.path.exists(feature_info_path):
        raise FileNotFoundError(f"Feature info file not found: {feature_info_path}")

    with open(feature_info_path, "r") as f:
        feature_info = json.load(f)

    X_holdout = pd.read_csv(X_holdout_path)
    y_holdout = pd.read_csv(y_holdout_path)["y_holdout"].values

    # Note: scaler is not restored as an object (only its params or type string is saved)
    # If you need the actual scaler object, you must save it with joblib or pickle

    # Load scaler object
    scaler_path = os.path.join(holdout_dir, "scaler.joblib")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        scaler = None

    # Try to load median_values if present in feature_info, else set to empty Series
    if "median_values" in feature_info:
        median_values = pd.Series(feature_info["median_values"])
    else:
        median_values = pd.Series(dtype=float)
    return {
        "kept_columns": feature_info.get("kept_columns", []),
        "least_important_features": feature_info.get("least_important_features", []),
        "correlated_features_dropped": feature_info.get("correlated_features_dropped", []),
        "X_holdout": X_holdout,
        "y_holdout": y_holdout,
        "scaler": scaler,
        "median_values": median_values
    }

def load_models_for_label(label, models_dir="models"):
    """
    Loads all models for a given label from the specified directory.
    Model filenames must start with the label (e.g., 'Tg_fold1_xgb.joblib').
    Returns a list of loaded models.
    """

    models = []
    if not os.path.exists(models_dir):
        print(f"Models directory '{models_dir}' does not exist.")
        return models

    # Match both .joblib and .pt (for torch) files
    pattern_joblib = os.path.join(models_dir, f"{label}_*.joblib")
    pattern_pt = os.path.join(models_dir, f"{label}_*.pt")
    model_files = glob.glob(pattern_joblib) + glob.glob(pattern_pt)
    if not model_files:
        print(f"No models found for label '{label}' in '{models_dir}'.")
        return models

    for model_file in sorted(model_files):
        if model_file.endswith(".joblib"):
            try:
                model = joblib.load(model_file)
                models.append(model)
            except Exception as e:
                print(f"Failed to load model {model_file}: {e}")
        elif model_file.endswith(".pt"):
            # Torch model loading requires model class and architecture
            print(f"Skipping torch model {model_file} (requires model class definition).")
            # You can implement torch loading here if needed
    print(f"Loaded {len(models)} models for label '{label}'.")
    return models

output_df = pd.DataFrame({
    'id': test_ids
})

# --- Store and display mean_absolute_error for each label ---
mae_results = []

def train_or_predict(train_model=True, model_dir=None):
    for label in labels:
        if train_model:
            print(f"\n=== Training/Predicting for label: {label} ===")
            label_data = prepare_label_data(label, subtables, config)
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


output_df = pd.DataFrame({
    'id': test_ids
})
MODEL_DIR1 = '/kaggle/input/neurips-2025/xgb_v3'
train_or_predict(train_model=False, model_dir=MODEL_DIR1)
print(output_df)
output_df.to_csv('submission_xgb.csv', index=False)
print("Saved submission_xgb.csv")
