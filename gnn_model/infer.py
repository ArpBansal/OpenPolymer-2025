import numpy as np
import pandas as pd
import os
import joblib
import warnings
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import RobustScaler
import torch

from .data_processing import (
    clean_and_validate_smiles, add_extra_data_clean, safe_load_dataset,
    separate_subtables, smiles_to_graph_label_specific, scale_graph_features,
    TARGETS, BASE_PATH, external_datasets
)
from .train import train_gnn_model, save_gnn_model, load_gnn_model
from .prediction import predict_with_gnn

os.makedirs("NeurIPS", exist_ok=True)

class Config:
    debug = False
    use_cross_validation = True
    use_external_data = True
    random_state = 42

config = Config()

print("Loading competition data...")
train = pd.read_csv(BASE_PATH + 'train.csv')
test = pd.read_csv(BASE_PATH + 'test.csv')

if config.debug:
    print("   Debug mode: sampling 1000 training examples")
    train = train.sample(n=1000, random_state=42).reset_index(drop=True)

print(f"Training data shape: {train.shape}, Test data shape: {test.shape}")

print("Cleaning and validating SMILES...")
train['SMILES'] = train['SMILES'].apply(clean_and_validate_smiles)
test['SMILES'] = test['SMILES'].apply(clean_and_validate_smiles)

invalid_train = train['SMILES'].isnull().sum()
invalid_test = test['SMILES'].isnull().sum()

print(f"   Removed {invalid_train} invalid SMILES from training data")
print(f"   Removed {invalid_test} invalid SMILES from test data")

train = train[train['SMILES'].notnull()].reset_index(drop=True)
test = test[test['SMILES'].notnull()].reset_index(drop=True)

print(f"   Final training samples: {len(train)}")
print(f"   Final test samples: {len(test)}")

print("\n📂 Loading external datasets...")

safe_load_dataset(
    '/kaggle/input/tc-smiles/Tc_SMILES.csv',
    'Tc',
    lambda df: df.rename(columns={'TC_mean': 'Tc'}),
    'Tc data'
)

safe_load_dataset(
    '/kaggle/input/tg-smiles-pid-polymer-class/TgSS_enriched_cleaned.csv',
    'Tg', 
    lambda df: df[['SMILES', 'Tg']] if 'Tg' in df.columns else df,
    'TgSS enriched data'
)

safe_load_dataset(
    '/kaggle/input/smiles-extra-data/JCIM_sup_bigsmiles.csv',
    'Tg',
    lambda df: df[['SMILES', 'Tg (C)']].rename(columns={'Tg (C)': 'Tg'}),
    'JCIM Tg data'
)

safe_load_dataset(
    '/kaggle/input/smiles-extra-data/data_tg3.xlsx',
    'Tg',
    lambda df: df.rename(columns={'Tg [K]': 'Tg'}).assign(Tg=lambda x: x['Tg'] - 273.15),
    'Xlsx Tg data'
)

safe_load_dataset(
    '/kaggle/input/smiles-extra-data/data_dnst1.xlsx',
    'Density',
    lambda df: df.rename(columns={'density(g/cm3)': 'Density'})[['SMILES', 'Density']]
                .query('SMILES.notnull() and Density.notnull() and Density != "nylon"')
                .assign(Density=lambda x: x['Density'].astype(float) - 0.118),
    'Density data'
)

safe_load_dataset(
    BASE_PATH + 'train_supplement/dataset4.csv',
    'FFV', 
    lambda df: df[['SMILES', 'FFV']] if 'FFV' in df.columns else df,
    'dataset 4'
)

print("\n🔄 Integrating external data...")
train_extended = train[['SMILES'] + TARGETS].copy()

if getattr(config, "use_external_data", True) and  not config.debug:
    for target, dataset in external_datasets:
        print(f"   Processing {target} data...")
        train_extended = add_extra_data_clean(train_extended, dataset, target)

print(f"\n📊 Final training data:")
print(f"   Original samples: {len(train)}")
print(f"   Extended samples: {len(train_extended)}")
print(f"   Gain: +{len(train_extended) - len(train)} samples")

for target in TARGETS:
    count = train_extended[target].notna().sum()
    original_count = train[target].notna().sum() if target in train.columns else 0
    gain = count - original_count
    print(f"   {target}: {count:,} samples (+{gain})")

print(f"\n✅ Data integration complete with clean SMILES!")

train_df=train_extended
test_df=test
subtables = separate_subtables(train_df)

test_smiles = test_df['SMILES'].tolist()
test_ids = test_df['id'].values
labels = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

def train_or_predict_gnn(train_model=True, model_dir="models/gnn", n_splits=10):
    """
    (FINAL COMPLETE VERSION)
    - All data hardening (coerce, filter) and RobustScaler logic is included.
    - CV loop is modified to create a val_data_list.
    - Calls the new, optimized train_gnn_model with scheduler/early stopping.
    - Correctly passes all arguments (config['neurons'], config['dropouts']) to fix the TypeError.
    """
    
    ATOM_MAP_LEN = 20
    
    VALID_RANGES = {
        'Tg':      (-100, 500),  
        'FFV':     (0.01, 0.99), 
        'Tc':      (0, 1000),    
        'Density': (0.1, 3.0),   
        'Rg':      (0.1, 200)    
    }

    best_configs = {
        "Tg":      {"neurons": [512, 256, 128], "dropouts": [0.5, 0.4, 0.2]},
        "Density": {"neurons": [1024, 256, 64], "dropouts": [0.5, 0.4, 0.3]},
        "FFV":     {"neurons": [1024, 512, 64], "dropouts": [0.6, 0.5, 0.4]},
        "Tc":      {"neurons": [128, 64], "dropouts": [0.4, 0.3]},
        "Rg":      {"neurons": [128, 64, 64], "dropouts": [0.4, 0.3, 0.3]},
    }
    default_config = {"neurons": [128, 64], "dropouts": [0.3, 0.3]}

    output_df = pd.DataFrame({'id': test_df['id']})
    cv_mae_results = []
    os.makedirs(model_dir, exist_ok=True)
    warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)

    for label in labels: 
        print(f"\n{'='*20} Processing GNN for label: {label} {'='*20}")
        
        config = best_configs.get(label, default_config)
        print(f"Using MLP Config: Neurons={config['neurons']}, Dropouts={config['dropouts']}")
        
        ensemble_models = []
        y_scaler_path = os.path.join(model_dir, f"gnn_yscaler_{label}.joblib")
        u_scaler_path = os.path.join(model_dir, f"gnn_uscaler_{label}.joblib")
        x_scaler_path = os.path.join(model_dir, f"gnn_xscaler_{label}.joblib")
        
        if train_model:
            all_smiles_raw = subtables[label]['SMILES']
            all_y_raw = subtables[label][label] 
            
            all_y_numeric = pd.to_numeric(all_y_raw, errors='coerce')
            original_count = len(all_y_numeric)

            valid_min, valid_max = VALID_RANGES.get(label, (-np.inf, np.inf))
            valid_mask = (all_y_numeric >= valid_min) & (all_y_numeric <= valid_max) & (all_y_numeric.notna())
            
            all_y = all_y_numeric[valid_mask].reset_index(drop=True)
            all_smiles = all_smiles_raw[valid_mask].reset_index(drop=True)
            
            print(f"FILTERING: Coerced {original_count} rows. Kept {len(all_y)} valid rows within range ({valid_min}, {valid_max}).")
            
            if len(all_y) < (2 * n_splits): 
                print(f"CRITICAL: Not enough valid data ({len(all_y)}) to train for {label} with {n_splits} splits. Skipping.")
                continue

            print("Using RobustScaler for Y-Scaler.")
            y_scaler = RobustScaler()  
            all_y_scaled = y_scaler.fit_transform(all_y.values.reshape(-1, 1)).flatten()
            joblib.dump(y_scaler, y_scaler_path)
            print(f"Saved Y-Scaler for {label}")

            print("Pre-computing all graph features to fit input scalers...")
            all_train_graphs_raw = [smiles_to_graph_label_specific(s, label, None) for s in all_smiles]
            
            all_train_graphs_synced = []
            all_y_scaled_synced = [] 
            all_y_original_synced = []
            all_smiles_synced = []
            
            for i, graph in enumerate(all_train_graphs_raw):
                if graph is not None:
                    all_train_graphs_synced.append(graph)
                    all_y_scaled_synced.append(all_y_scaled[i]) 
                    all_y_original_synced.append(all_y[i])
                    all_smiles_synced.append(all_smiles[i])
            
            all_train_graphs = all_train_graphs_synced 
            all_y_scaled = np.array(all_y_scaled_synced)
            all_y_original_df = pd.Series(all_y_original_synced)
            all_smiles_df = pd.Series(all_smiles_synced)

            if not all_train_graphs:
                print(f"CRITICAL: No valid training graphs could be featurized for {label}. Skipping.")
                continue
                
            all_u_data = np.concatenate([d.u.numpy() for d in all_train_graphs], axis=0)
            print("Using RobustScaler for U-Scaler.")
            u_scaler = RobustScaler().fit(all_u_data)
            joblib.dump(u_scaler, u_scaler_path)
            print(f"Saved U-Scaler for {label}")

            all_x_data = torch.cat([d.x for d in all_train_graphs], dim=0)
            all_x_continuous = all_x_data[:, ATOM_MAP_LEN:].numpy()
            print("Using RobustScaler for X-Scaler.")
            x_scaler = RobustScaler().fit(all_x_continuous)
            joblib.dump(x_scaler, x_scaler_path)
            print(f"Saved X-Scaler for {label}")

            all_data_objects_scaled = scale_graph_features(all_train_graphs, u_scaler, x_scaler, ATOM_MAP_LEN)
            for i, data_obj in enumerate(all_data_objects_scaled):
                data_obj.y = torch.tensor([[all_y_scaled[i]]], dtype=torch.float)
            
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            fold_val_scores = []
            fold_indices_gen = kf.split(all_data_objects_scaled)

            for fold, (train_idx, val_idx) in enumerate(fold_indices_gen):
                print(f"\n--- Fold {fold+1}/{n_splits} for {label} ---")
                
                train_data_list = [all_data_objects_scaled[i] for i in train_idx]
                val_data_list = [all_data_objects_scaled[i] for i in val_idx]
                
                val_smiles_list = all_smiles_df.iloc[val_idx].tolist()
                y_val_original = all_y_original_df.iloc[val_idx].values 

                fold_model = train_gnn_model(
                    label,
                    train_data_list,
                    val_data_list,
                    config['neurons'],
                    config['dropouts'],
                    epochs=300
                )
                
                if fold_model:
                    print("Running final validation prediction on the best model...")
                    val_preds_scaled = predict_with_gnn(fold_model, val_smiles_list, label, u_scaler, x_scaler, ATOM_MAP_LEN)
                    
                    train_y_scaled_median = 0.0
                    val_preds_scaled_filled = pd.Series(val_preds_scaled).fillna(train_y_scaled_median)
                    
                    val_preds_original = y_scaler.inverse_transform(
                        val_preds_scaled_filled.values.reshape(-1, 1)
                    ).flatten()

                    mae = mean_absolute_error(y_val_original, val_preds_original)
                    print(f"✅ Fold {fold+1} Validation MAE (Original Scale): {mae:.4f}")
                    fold_val_scores.append(mae)
                    
                    model_save_name = f"{label}_fold{fold}"
                    save_gnn_model(fold_model, model_save_name, model_dir)
                    ensemble_models.append(fold_model)
                else:
                    print(f"Warning: Training failed for Fold {fold+1}. Model will be skipped.")
            
            if fold_val_scores:
                avg_cv_mae = np.mean(fold_val_scores)
                print(f"\n{'*'*10} Average CV MAE for {label} (Original Scale): {avg_cv_mae:.4f} {'*'*10}")
                cv_mae_results.append({'label': label, 'avg_cv_mae': avg_cv_mae})

        else:
            print(f"Loading {n_splits} models and ALL 3 RobustScalers for {label} ensemble...")
            model_path = '/kaggle/input/neurips-2025/GATConv_v29/models/gnn/'
            try:
                y_scaler = joblib.load(f'{model_path}gnn_yscaler_{label}.joblib')
                u_scaler = joblib.load(f'{model_path}gnn_uscaler_{label}.joblib')
                x_scaler = joblib.load(f'{model_path}gnn_xscaler_{label}.joblib')
                print("Loaded Y, U, and X RobustScalers.")
            except FileNotFoundError:
                print(f"CRITICAL: Scaler files not found for {label}. Cannot make predictions.")
                continue

            for fold in range(n_splits):
                loaded_model = load_gnn_model(f"{label}_fold{fold}", model_path.rstrip('/'))
                if loaded_model:
                    ensemble_models.append(loaded_model)
            
            if not ensemble_models: print(f"Warning: No models found for label {label}.")
            else: print(f"Successfully loaded {len(ensemble_models)} models for ensemble.")

        test_smiles = test_df['SMILES'].tolist()
        
        if ensemble_models and y_scaler and u_scaler and x_scaler:
            print(f"Making ensemble (scaled) predictions for {label} using {len(ensemble_models)} models...")
            all_fold_preds_scaled = []
            for model in ensemble_models:
                fold_test_preds_scaled = predict_with_gnn(model, test_smiles, label, u_scaler, x_scaler, ATOM_MAP_LEN)
                all_fold_preds_scaled.append(fold_test_preds_scaled)
            
            preds_stack_scaled = np.stack(all_fold_preds_scaled)
            final_ensemble_preds_scaled = np.nanmean(preds_stack_scaled, axis=0) 
            pred_series_scaled = pd.Series(final_ensemble_preds_scaled)
            
            pred_series_scaled_filled = pred_series_scaled.fillna(0.0)

            final_preds_original = y_scaler.inverse_transform(
                pred_series_scaled_filled.values.reshape(-1, 1)
            ).flatten()
            
            output_df[label] = final_preds_original
            
        else:
            print(f"No models or scalers available for {label}. Filling with (filtered) training median.")
            fallback_median = 0.0
            try:
                if 'all_y' in locals() and not all_y.empty:
                     fallback_median = all_y.median()
                else: 
                     print("Loading data to calculate fallback median...")
                     fb_y_raw = subtables[label][label]
                     fb_y_num = pd.to_numeric(fb_y_raw, errors='coerce')
                     valid_min, valid_max = VALID_RANGES.get(label, (-np.inf, np.inf))
                     fb_mask = (fb_y_num >= valid_min) & (fb_y_num <= valid_max) & (fb_y_num.notna())
                     fallback_median = fb_y_num[fb_mask].median()
                print(f"Using filtered median fallback: {fallback_median}")
            except Exception as e:
                 print(f"Error getting median, falling back to 0: {e}")
                 fallback_median = 0.0 
                 
            output_df[label] = fallback_median

    if train_model and cv_mae_results:
        print("\n" + "="*40)
        print("📊 HYBRID GNN 5-Fold CV MAE Summary (Original Scale):")
        print("="*40)
        mae_df = pd.DataFrame(cv_mae_results)
        print(mae_df.to_string(index=False))
        mae_df.to_csv("gnn_hybrid_cv_mae_results.csv", index=False)
        print("\nCV results saved to gnn_hybrid_cv_mae_results.csv")

    submission_path = 'submission_hybrid_gnn_final.csv'
    output_df.to_csv(submission_path, index=False)
    print(f"\n✅ GNN Ensemble predictions (Original Scale) saved to {submission_path}")
    
    warnings.filterwarnings("default", "Mean of empty slice", RuntimeWarning)
    
    return output_df

gnn_submission_df = train_or_predict_gnn(train_model=False)

gnn_submission_df.to_csv('submission_gnn.csv', index=False)
print("Saved submission_gnn.csv")

print("\nGNN Submission Preview:")
print(gnn_submission_df.head())
