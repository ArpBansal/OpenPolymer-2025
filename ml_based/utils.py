import sqlite3
import hashlib
import json
import numpy as np
import pandas as pd
import joblib
import torch
import random
import os
import glob
from transformers import AutoTokenizer, AutoModel
from config import config

def init_chemberta():
    model_name = "/kaggle/input/c/transformers/default/1/ChemBERTa-77M-MLM"
    chemberta_tokenizer = AutoTokenizer.from_pretrained(model_name)
    chemberta_model = AutoModel.from_pretrained(model_name)
    chemberta_model.eval()
    return chemberta_tokenizer, chemberta_model

def get_chemberta_embedding(smiles, embedding_dim=384):
    """
    Returns ChemBERTa embedding for a single SMILES string.
    Pads/truncates to embedding_dim if needed.
    """
    if smiles is None or not isinstance(smiles, str) or len(smiles) == 0:
        return np.zeros(embedding_dim)
    try:
        # Add pooling argument with default 'mean'
        pooling = getattr(config, 'chemberta_pooling', 'mean')  # can be 'mean', 'max', 'cls', 'pooler'
        chemberta_tokenizer, chemberta_model = init_chemberta()
        inputs = chemberta_tokenizer([smiles], padding=True, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = chemberta_model(**inputs)
            if pooling == 'pooler' and hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                emb = outputs.pooler_output.squeeze(0)
            elif pooling == 'cls' and hasattr(outputs, 'last_hidden_state'):
                emb = outputs.last_hidden_state[:, 0, :].squeeze(0)
            elif pooling == 'max' and hasattr(outputs, 'last_hidden_state'):
                emb = outputs.last_hidden_state.max(dim=1).values.squeeze(0)
            elif pooling == 'mean' and hasattr(outputs, 'last_hidden_state'):
                emb = outputs.last_hidden_state.mean(dim=1).squeeze(0)
            else:
                raise ValueError("Cannot extract embedding from model output")
            emb_np = emb.cpu().numpy()
            # Pad or truncate if needed
            if emb_np.shape[0] < embedding_dim:
                emb_np = np.pad(emb_np, (0, embedding_dim - emb_np.shape[0]))
            elif emb_np.shape[0] > embedding_dim:
                emb_np = emb_np[:embedding_dim]
            return emb_np
    except Exception as e:
        print(f"ChemBERTa embedding failed for SMILES '{smiles}': {e}")
        return np.zeros(embedding_dim)
    
def init_xgb_tuning_db(db_path="xgb_tuning.db"):
    """Initialize the XGB tuning database and return all existing results."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS xgb_tuning
                 (param_hash TEXT PRIMARY KEY, params TEXT, score REAL)''')
    c.execute('SELECT params, score FROM xgb_tuning')
    results = c.fetchall()
    conn.close()
    return [(json.loads(params), score) for params, score in results]

def get_param_hash(params):
    param_str = json.dumps(params, sort_keys=True)
    return hashlib.md5(param_str.encode('utf-8')).hexdigest()

def check_db_for_params(db_path, param_hash):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT score FROM xgb_tuning WHERE param_hash=?', (param_hash,))
    result = c.fetchone()
    conn.close()
    return result is not None

def save_result_to_db(db_path, param_hash, params, score):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''INSERT OR REPLACE INTO xgb_tuning (param_hash, params, score)
                 VALUES (?, ?, ?)''', (param_hash, json.dumps(params, sort_keys=True), score))
    conn.commit()
    conn.close()

# Utility: Display model summary if torchinfo is available
def show_model_summary(model, input_dim, batch_size=32):
    try:
        from torchinfo import summary
        print(summary(model, input_size=(batch_size, input_dim)))
    except ImportError:
        print("torchinfo is not installed. Install it with 'pip install torchinfo' to see model summaries.")

def display_outlier_summary(y, X=None, name="target", z_thresh=3, iqr_factor=1.5, iso_contamination=0.01, lof_contamination=0.01):
    """
    Display the percentage of data flagged as outlier by Z-score, IQR, Isolation Forest, and LOF.
    y: 1D array-like (target or feature)
    X: 2D array-like (feature matrix, required for Isolation Forest/LOF)
    name: str, name of the variable being checked
    """
    print(f"\nOutlier summary for: {name}")
    y = np.asarray(y)
    n = len(y)
    # Z-score
    z_scores = (y - np.mean(y)) / np.std(y)
    z_outliers = np.abs(z_scores) > z_thresh
    print(f"Z-score > {z_thresh}: {np.sum(z_outliers)} / {n} ({100*np.mean(z_outliers):.2f}%)")

    # IQR
    Q1 = np.percentile(y, 25)
    Q3 = np.percentile(y, 75)
    IQR = Q3 - Q1
    lower = Q1 - iqr_factor * IQR
    upper = Q3 + iqr_factor * IQR
    iqr_outliers = (y < lower) | (y > upper)
    print(f"IQR (factor {iqr_factor}): {np.sum(iqr_outliers)} / {n} ({100*np.mean(iqr_outliers):.2f}%)")

    # Isolation Forest (if X provided)
    if X is not None:
        try:
            from sklearn.ensemble import IsolationForest
            iso = IsolationForest(contamination=iso_contamination, random_state=42)
            iso_out = iso.fit_predict(X)
            iso_outliers = iso_out == -1
            print(f"Isolation Forest (contamination={iso_contamination}): {np.sum(iso_outliers)} / {len(iso_outliers)} ({100*np.mean(iso_outliers):.2f}%)")
        except Exception as e:
            print(f"Isolation Forest failed: {e}")
        # Local Outlier Factor
        try:
            from sklearn.neighbors import LocalOutlierFactor
            lof = LocalOutlierFactor(n_neighbors=20, contamination=lof_contamination)
            lof_out = lof.fit_predict(X)
            lof_outliers = lof_out == -1
            print(f"Local Outlier Factor (contamination={lof_contamination}): {np.sum(lof_outliers)} / {len(lof_outliers)} ({100*np.mean(lof_outliers):.2f}%)")
        except Exception as e:
            print(f"Local Outlier Factor failed: {e}")
    else:
        print("Isolation Forest/LOF skipped (X not provided)")


def set_global_random_seed(seed, config=None):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    if config is not None:
        config.random_state = seed

def check_inf_nan(X, y, label=None):
    """
    Checks for inf, -inf, and NaN values in X (DataFrame) and y (array/Series).
    Prints summary and returns True if any such values are found.
    """
    X_inf = np.isinf(X.values).any()
    X_nan = np.isnan(X.values).any()
    y_inf = np.isinf(y).any()
    y_nan = np.isnan(y).any()
    if label is None:
        label = ""
    else:
        label = f" [{label}]"
    if X_inf or X_nan or y_inf or y_nan:
        print(f"⚠️ Detected inf/nan in X or y{label}: X_inf={X_inf}, X_nan={X_nan}, y_inf={y_inf}, y_nan={y_nan}")
        if X_inf:
            print(f"  X columns with inf: {X.columns[np.isinf(X.values).any(axis=0)].tolist()}")
        if X_nan:
            print(f"  X columns with nan: {X.columns[np.isnan(X.values).any(axis=0)].tolist()}")
        if y_inf:
            print("  y contains inf values.")
        if y_nan:
            print("  y contains nan values.")
        return True
    else:
        print(f"No inf/nan in X or y{label}.")
        return False

def save_importance_to_excel(importance_df, label, log_path):
    import os
    from openpyxl import load_workbook
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if os.path.exists(log_path):
        with pd.ExcelWriter(log_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            importance_df.to_excel(writer, sheet_name=label, index=False)
    else:
        with pd.ExcelWriter(log_path, engine='openpyxl') as writer:
            importance_df.to_excel(writer, sheet_name=label, index=False)

def save_model(Model, label, fold, model_name):
    model_path = f"models/{label}_fold{fold+1}_{model_name}"
    try:
        if 'torch' in str(type(Model)).lower():
            # Save PyTorch model state_dict
            model_path += ".pt"
            torch.save(Model.state_dict(), model_path)
        else:
            # Save scikit-learn model
            model_path += ".joblib"
            joblib.dump(Model, model_path)
        print(f"Saved model for {label} fold {fold+1} to {model_path}")
    except Exception as e:
        print(f"Failed to save model for {label} fold {fold+1}: {e}")

def stable_hash(obj, max_value=1_000_000):
    """
    Deterministic hash for objects (e.g. labels).
    Always returns the same value across runs/machines.
    """
    # Convert to string and encode
    s = str(obj).encode("utf-8")
    # Use MD5 (fast & deterministic)
    h = hashlib.md5(s).hexdigest()
    # Convert hex digest to int and limit range
    return int(h, 16) % max_value

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
