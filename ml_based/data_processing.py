import numpy as np
import pandas as pd
from rdkit import Chem
from config import config, RDKIT_AVAILABLE, TARGETS, BASE_PATH, Config
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV
from mordred import Calculator, descriptors as mordred_descriptors
from rdkit.Chem import Descriptors, MACCSkeys, rdmolops, Lipinski, Crippen
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator, GetAtomPairGenerator, GetTopologicalTorsionGenerator
from utils import get_chemberta_embedding
import networkx as nx
import os
import joblib
from utils import check_inf_nan, save_importance_to_excel
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import shap

external_datasets = []

def clean_and_validate_smiles(smiles):
    """Completely clean and validate SMILES, removing all problematic patterns"""
    if not isinstance(smiles, str) or len(smiles) == 0:
        return None
    
    bad_patterns = [
        '[R]', '[R1]', '[R2]', '[R3]', '[R4]', '[R5]', 
        "[R']", '[R"]', 'R1', 'R2', 'R3', 'R4', 'R5',
        '*R', 'R*', '[*R', 'R*]',
        '([R])', '([R1])', '([R2])', 
    ]
    
    for pattern in bad_patterns:
        if pattern in smiles:
            return None
    
    if '][' in smiles and any(x in smiles for x in ['[R', 'R]']):
        return None
    
    if RDKIT_AVAILABLE:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                return Chem.MolToSmiles(mol, canonical=True)
            else:
                return None
        except:
            return None
    
    return smiles

def add_extra_data_clean(df_train, df_extra, target):
    """Add external data with thorough SMILES cleaning"""
    n_samples_before = len(df_train[df_train[target].notnull()])
    
    print(f"      Processing {len(df_extra)} {target} samples...")
    
    df_extra['SMILES'] = df_extra['SMILES'].apply(clean_and_validate_smiles)
    
    before_filter = len(df_extra)
    df_extra = df_extra[df_extra['SMILES'].notnull()]
    df_extra = df_extra.dropna(subset=[target])
    after_filter = len(df_extra)
    
    print(f"      Kept {after_filter}/{before_filter} valid samples")
    
    if len(df_extra) == 0:
        print(f"      No valid data remaining for {target}")
        return df_train
    
    df_extra = df_extra.groupby('SMILES', as_index=False)[target].mean()
    
    cross_smiles = set(df_extra['SMILES']) & set(df_train['SMILES'])
    unique_smiles_extra = set(df_extra['SMILES']) - set(df_train['SMILES'])

    filled_count = 0
    for smile in df_train[df_train[target].isnull()]['SMILES'].tolist():
        if smile in cross_smiles:
            df_train.loc[df_train['SMILES']==smile, target] = \
                df_extra[df_extra['SMILES']==smile][target].values[0]
            filled_count += 1
    
    extra_to_add = df_extra[df_extra['SMILES'].isin(unique_smiles_extra)].copy()
    if len(extra_to_add) > 0:
        for col in TARGETS:
            if col not in extra_to_add.columns:
                extra_to_add[col] = np.nan
        
        extra_to_add = extra_to_add[['SMILES'] + TARGETS]
        df_train = pd.concat([df_train, extra_to_add], axis=0, ignore_index=True)

    n_samples_after = len(df_train[df_train[target].notnull()])
    print(f'      {target}: +{n_samples_after-n_samples_before} samples, +{len(unique_smiles_extra)} unique SMILES')
    print(f"      Filled {filled_count} missing entries in train for {target}")
    print(f"      Added {len(extra_to_add)} new entries for {target}")
    return df_train

def safe_load_dataset(path, target, processor_func, description):
    try:
        if path.endswith('.xlsx'):
            data = pd.read_excel(path)
        else:
            data = pd.read_csv(path)
        
        data = processor_func(data)
        external_datasets.append((target, data))
        print(f"   ✅ {description}: {len(data)} samples")
        return True
    except Exception as e:
        print(f"   ⚠️ {description} failed: {str(e)[:100]}")
        return False

def separate_subtables(train_df):
    labels = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    subtables = {}
    for label in labels:
        subtables[label] = train_df[train_df[label].notna()][['SMILES', label]].reset_index(drop=True)

    for label in subtables:
        print(f"{label} NaNs per column:")
        print(subtables[label].isna().sum())
        print(subtables[label].shape)

    return subtables

def augment_smiles_dataset(smiles_list, labels, num_augments=3):
    """
    Augments a list of SMILES strings by generating randomized versions.

    Parameters:
        smiles_list (list of str): Original SMILES strings.
        labels (list or np.array): Corresponding labels.
        num_augments (int): Number of augmentations per SMILES.

    Returns:
        tuple: (augmented_smiles, augmented_labels)
    """
    augmented_smiles = []
    augmented_labels = []

    for smiles, label in zip(smiles_list, labels):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        augmented_smiles.append(smiles)
        augmented_labels.append(label)
        for _ in range(num_augments):
            rand_smiles = Chem.MolToSmiles(mol, doRandom=True)
            augmented_smiles.append(rand_smiles)
            augmented_labels.append(label)

    return augmented_smiles, np.array(augmented_labels)

def load_data():
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

    return train_extended, test

mordred_calc = Calculator(mordred_descriptors, ignore_3D=True)
def build_mordred_descriptors(smiles_list):
    # Build Mordred descriptors for test
    mols_test = [Chem.MolFromSmiles(s) for s in smiles_list]
    desc_test = mordred_calc.pandas(mols_test, nproc=1)

    # Make columns string & numeric only (no dropping beyond that)
    desc_test.columns = desc_test.columns.map(str)
    desc_test = desc_test.select_dtypes(include=[np.number]).copy()
    desc_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    return desc_test

from rdkit.Chem import Crippen, Lipinski

def smiles_to_combined_fingerprints_with_descriptors(smiles_list):
    # Set fingerprint parameters inside the function
    radius = 2
    n_bits = 128

    generator = GetMorganGenerator(radius=radius, fpSize=n_bits) if getattr(Config, "use_morgan_fp", True) else None
    atom_pair_gen = GetAtomPairGenerator(fpSize=n_bits) if getattr(Config, 'use_atom_pair_fp', False) else None
    torsion_gen = GetTopologicalTorsionGenerator(fpSize=n_bits) if getattr(Config, 'use_torsion_fp', False) else None
    
    fp_len = (n_bits if getattr(Config, 'use_morgan_fp', False) else 0) \
           + (n_bits if getattr(Config, 'use_atom_pair_fp', False) else 0) \
           + (n_bits if getattr(Config, 'use_torsion_fp', False) else 0) \
           + (167 if getattr(Config, 'use_maccs_fp', True) else 0)
    if getattr(Config, 'use_chemberta', False):
        fp_len += 384
        
    fingerprints = []
    descriptors = []
    valid_smiles = []
    invalid_indices = []
    use_any_fp = getattr(Config, "use_morgan_fp", False) or getattr(Config, "use_atom_pair_fp", False) or getattr(Config, "use_torsion_fp", False) or getattr(Config, "use_maccs_fp", False) or getattr(Config, 'use_chemberta', False)

    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            # Fingerprints (No change from your code)
            if use_any_fp:
                fps = []
                if getattr(Config, "use_morgan_fp", True) and generator is not None:
                    fps.append(np.array(generator.GetFingerprint(mol)))
                if atom_pair_gen:
                    fps.append(np.array(atom_pair_gen.GetFingerprint(mol)))
                if torsion_gen:
                    fps.append(np.array(torsion_gen.GetFingerprint(mol)))
                if getattr(Config, 'use_maccs_fp', True):
                    fps.append(np.array(MACCSkeys.GenMACCSKeys(mol)))
                if getattr(Config, "use_chemberta", False):
                    emb = get_chemberta_embedding(smiles)
                    fps.append(emb)
                
                combined_fp = np.concatenate(fps)
                fingerprints.append(combined_fp)

            if getattr(Config, 'use_descriptors', True):
                descriptor_values = {}
                for name, func in Descriptors.descList:
                    try:
                        descriptor_values[name] = func(mol)
                    except:
                        print(f"Descriptor {name} failed for SMILES at index {i}")
                        descriptor_values[name] = None

                # try:
                # --- Features for Rigidity and Complexity (for Tg, FFV) ---
                try:
                    num_heavy_atoms = mol.GetNumHeavyAtoms()
                except Exception as e:
                    num_heavy_atoms = 0

                # --- Features for Rigidity and Complexity (for Tg, FFV) ---
                try:
                    descriptor_values['NumAromaticRings'] = Lipinski.NumAromaticRings(mol)
                except Exception as e:
                    descriptor_values['NumAromaticRings'] = None
                try:
                    num_sp3_carbons = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6 and atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3)
                    descriptor_values['FractionCSP3'] = num_sp3_carbons / num_heavy_atoms if num_heavy_atoms > 0 else 0
                except Exception as e:
                    descriptor_values['FractionCSP3'] = None

                # --- Features for Bulkiness and Shape (for FFV) ---
                try:
                    descriptor_values['MolMR'] = Crippen.MolMR(mol) # Molar Refractivity (volume)
                except Exception as e:
                    descriptor_values['MolMR'] = None
                try:
                    descriptor_values['LabuteASA'] = Descriptors.LabuteASA(mol) # Accessible surface area
                except Exception as e:
                    descriptor_values['LabuteASA'] = None

                # --- Features for Heavy Atoms (for Density) ---
                try:
                    descriptor_values['NumFluorine'] = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 9)
                except Exception as e:
                    descriptor_values['NumFluorine'] = None
                try:
                    descriptor_values['NumChlorine'] = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 17)
                except Exception as e:
                    descriptor_values['NumChlorine'] = None

                # --- Features for Intermolecular Forces (for Tc) ---
                try:
                    descriptor_values['NumHDonors'] = Lipinski.NumHDonors(mol)
                except Exception as e:
                    descriptor_values['NumHDonors'] = None
                try:
                    descriptor_values['NumHAcceptors'] = Lipinski.NumHAcceptors(mol)
                except Exception as e:
                    descriptor_values['NumHAcceptors'] = None

                # --- Features for Branching and Flexibility (for Rg) ---
                try:
                    descriptor_values['BalabanJ'] = Descriptors.BalabanJ(mol) # Topological index sensitive to branching
                except Exception as e:
                    descriptor_values['BalabanJ'] = None
                try:
                    descriptor_values['Kappa2'] = Descriptors.Kappa2(mol) # Molecular shape index
                except Exception as e:
                    descriptor_values['Kappa2'] = None
                try:
                    descriptor_values['NumRotatableBonds'] = CalcNumRotatableBonds(mol) # Flexibility
                except Exception as e:
                    descriptor_values['NumRotatableBonds'] = None
                
                # Graph-based features
                try:
                    adj = rdmolops.GetAdjacencyMatrix(mol)
                    G = nx.from_numpy_array(adj)
                    if nx.is_connected(G):
                        descriptor_values['graph_diameter'] = nx.diameter(G)
                        descriptor_values['avg_shortest_path'] = nx.average_shortest_path_length(G)
                    else:
                        descriptor_values['graph_diameter'], descriptor_values['avg_shortest_path'] = 0, 0
                    descriptor_values['num_cycles'] = len(list(nx.cycle_basis(G)))
                except:
                    print(f"Graph features failed for SMILES at index {i}")
                    descriptor_values['graph_diameter'], descriptor_values['avg_shortest_path'], descriptor_values['num_cycles'] = None, None, None

                descriptors.append(descriptor_values)
            else:
                descriptors.append(None)
            valid_smiles.append(smiles)
        else:
            if use_any_fp: fingerprints.append(np.zeros(fp_len))
            if getattr(Config, "use_chemberta", False):
                descriptors.append({f'chemberta_emb_{j}': 0.0 for j in range(384)})
            else:
                descriptors.append(None)
            valid_smiles.append(None)
            invalid_indices.append(i)

    fingerprints_df = pd.DataFrame(fingerprints, columns=[f'FP_{i}' for i in range(fp_len)]) if use_any_fp else pd.DataFrame()
    descriptors_df = pd.DataFrame([d for d in descriptors if d is not None]) if any(d is not None for d in descriptors) else pd.DataFrame()

    if getattr(Config, 'use_mordred', False):
        mordred_df = build_mordred_descriptors(smiles_list)
        if descriptors_df.empty:
            descriptors_df = mordred_df
        else:
            descriptors_df = pd.concat([descriptors_df.reset_index(drop=True), mordred_df], axis=1)
    
    # Keep only unique columns in descriptors_df
    if not descriptors_df.empty:
        descriptors_df = descriptors_df.loc[:, ~descriptors_df.columns.duplicated()]
    return fingerprints_df, descriptors_df, valid_smiles, invalid_indices

required_descriptors = {'graph_diameter','num_cycles','avg_shortest_path','MolWt', 'LogP', 'TPSA', 'RotatableBonds', 'NumAtoms'}

# Utility function to combine train and val sets into X_all and y_all
def combine_train_val(X_train, X_val, y_train, y_val):
    X_train = pd.DataFrame(X_train) if isinstance(X_train, np.ndarray) else X_train
    X_val = pd.DataFrame(X_val) if isinstance(X_val, np.ndarray) else X_val
    y_train = pd.Series(y_train) if isinstance(y_train, np.ndarray) else y_train
    y_val = pd.Series(y_val) if isinstance(y_val, np.ndarray) else y_val
    X_all = pd.concat([X_train, X_val], axis=0)
    y_all = pd.concat([y_train, y_val], axis=0)
    return X_all, y_all

# --- PCA utility for train/test transformation ---
def apply_pca(X_train, X_test=None, verbose=True):
    pca = PCA(n_components=config.pca_variance, svd_solver='full', random_state=getattr(config, 'random_state', 42))
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test) if X_test is not None else None
    if verbose:
        print(f"[PCA] Reduced train shape: {X_train.shape} -> {X_train_pca.shape} (kept {pca.n_components_} components, {100*pca.explained_variance_ratio_.sum():.4f}% variance)")
    return X_train_pca, X_test_pca, pca

def augment_dataset(X, y, n_samples=1000, n_components=5, random_state=None):
    """
    Augments a dataset using Gaussian Mixture Models.

    Parameters:
    - X: pd.DataFrame or np.ndarray — feature matrix
    - y: pd.Series or np.ndarray — target values
    - n_samples: int — number of synthetic samples to generate
    - n_components: int — number of GMM components
    - random_state: int — random seed for reproducibility

    Returns:
    - X_augmented: pd.DataFrame — augmented feature matrix
    - y_augmented: pd.Series — augmented target values
    """
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    elif not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame or a NumPy array")

    X.columns = X.columns.astype(str)

    if isinstance(y, np.ndarray):
        y = pd.Series(y)
    elif not isinstance(y, pd.Series):
        raise ValueError("y must be a pandas Series or a NumPy array")

    df = X.copy()
    df['Target'] = y.values

    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    gmm.fit(df)

    synthetic_data, _ = gmm.sample(n_samples)
    synthetic_df = pd.DataFrame(synthetic_data, columns=df.columns)

    augmented_df = pd.concat([df, synthetic_df], ignore_index=True)

    X_augmented = augmented_df.drop(columns='Target')
    y_augmented = augmented_df['Target']

    return X_augmented, y_augmented

def drop_correlated_features(df, threshold=0.95):
    """
    Drops columns in a DataFrame that are highly correlated with other columns.
    Only one of each pair of correlated columns is kept.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        threshold (float): Correlation threshold for dropping columns (default 0.95).

    Returns:
        pd.DataFrame: DataFrame with correlated columns dropped.
        list: List of dropped column names.
    """
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return df.drop(columns=to_drop), to_drop

def get_canonical_smiles(smiles):
        """Convert SMILES to canonical form for consistency"""
        if not RDKIT_AVAILABLE:
            return smiles
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                return Chem.MolToSmiles(mol, canonical=True)
        except:
            pass
        return smiles

def get_least_important_features_all_methods(X, y, label, model_name=None):
    """
    Remove features in three steps:
    1. Remove features with model.feature_importances_ <= 0
    2. Remove features with permutation_importance <= 0
    3. Remove features with SHAP importance <= 0
    Returns a list of features to remove (union of all three criteria).
    """    
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=config.random_state)
    model_type = (model_name or getattr(config, 'model_name', 'xgb'))
    if model_type == 'xgb':
        model = XGBRegressor(random_state=config.random_state, n_jobs=-1, verbosity=0, early_stopping_rounds=50, eval_metric="mae", objective="reg:absoluteerror")
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    elif model_type == 'catboost':
        model = CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=6, loss_function='MAE', eval_metric='MAE', random_seed=config.random_state, verbose=False)
        model.fit(X_train, y_train, eval_set=(X_valid, y_valid), early_stopping_rounds=50, use_best_model=True)
    elif model_type == 'lgbm':
        model = LGBMRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, reg_lambda=1.0, objective='mae', random_state=config.random_state, verbose=-1, verbosity=-1)
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric='mae', callbacks=[lgb.early_stopping(stopping_rounds=50)])
    else:
        model = XGBRegressor(random_state=config.random_state, n_jobs=-1, verbosity=0, early_stopping_rounds=50, eval_metric="rmse")
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    feature_names = X_train.columns

    # 1. Remove features with model.feature_importances_ <= 0
    fi_mask = model.feature_importances_ <= 0
    fi_features = set(feature_names[fi_mask])
    # Save feature_importances_ to Excel, sorted by importance
    fi_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': model.feature_importances_,
        'importance_std': [0]*len(feature_names)
    }).sort_values('importance_mean', ascending=False)
    save_importance_to_excel(fi_importance_df, label + '_fi', getattr(Config, 'permutation_importance_log_path', 'log/permutation_importance_log.xlsx'))

    # 2. Remove features with permutation_importance <= 0
    perm_result = permutation_importance(
        model, X_valid, y_valid,
        n_repeats=1 if config.debug else 10,
        random_state=config.random_state,
        scoring='neg_mean_absolute_error'
    )
    # Save permutation importance to Excel, sorted by mean importance descending
    perm_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': perm_result.importances_mean,
        'importance_std': perm_result.importances_std
    }).sort_values('importance_mean', ascending=False)
    save_importance_to_excel(perm_importance_df, label + '_perm', getattr(Config, 'permutation_importance_log_path', 'log/permutation_importance_log.xlsx'))
    perm_mask = perm_result.importances_mean <= 0
    perm_features = set(feature_names[perm_mask])

    # 3. Remove features with SHAP importance <= 0
    explainer = shap.Explainer(model, X_valid)
    # For LGBM, disable additivity check to avoid ExplainerError
    if model_type == 'lgbm':
        shap_values = explainer(X_valid, check_additivity=False)
    else:
        shap_values = explainer(X_valid)
    shap_importance = np.abs(shap_values.values).mean(axis=0)
    shap_mask = shap_importance <= 0
    shap_features = set(feature_names[shap_mask])
    # Save SHAP importance to Excel, sorted by importance
    shap_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': shap_importance,
        'importance_std': [0]*len(feature_names)
    }).sort_values('importance_mean', ascending=False)
    save_importance_to_excel(shap_importance_df, label + '_shap', getattr(Config, 'permutation_importance_log_path', 'log/permutation_importance_log.xlsx'))

    # Union of all features to remove
    features_to_remove = fi_features | perm_features | shap_features
    print(f"Removed {len(features_to_remove)} features for {label} using all methods (fi: {len(fi_features)}, perm: {len(perm_features)}, shap: {len(shap_features)})")

    return list(features_to_remove)


def preprocess_numerical_features(X, label=None):
    # Ensure numeric types
    X_num = X.select_dtypes(include=[np.number]).copy()
    
    # Replace inf/-inf with NaN
    X_num.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    valid_cols = X_num.columns
    # # Drop columns with any NaN
    # valid_cols = [col for col in X_num.columns if not X_num[col].isnull().any()]
    
    # dropped_cols = set(X_num.columns) - set(valid_cols)
    # if dropped_cols:
    #     print(f"Dropped columns with NaN/Inf for {label}: {list(dropped_cols)}")

    # # Keep only valid columns
    # X_num = X_num[valid_cols]
    
    # Calculate median for each column (for use in test set)
    median_values = X_num.median()
    # Scale features if enabled
    if getattr(Config, 'use_standard_scaler', False):
        scaler = StandardScaler()
        X_num_scaled = scaler.fit_transform(X_num)
        X_num = pd.DataFrame(X_num_scaled, columns=valid_cols, index=X.index)
    else:
        scaler = None
        X_num = X_num.copy()
    
    # Display categorical features
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if cat_cols:
        print(f"Categorical (non-numeric) features for {label}: {cat_cols}")
    else:
        print(f"No categorical (non-numeric) features for {label}.")
    return X_num, valid_cols, scaler, median_values


def prepare_label_data(label, subtables, Config):
    print(f"Processing label: {label}")
    print(subtables[label].head())
    print(subtables[label].shape)
    original_smiles = subtables[label]['SMILES'].tolist()
    original_labels = subtables[label][label].values

    # Canonicalize SMILES and deduplicate at molecule level before augmentation
    canonical_smiles = [get_canonical_smiles(s) for s in original_smiles]
    smiles_label_df = pd.DataFrame({
        'SMILES': canonical_smiles,
        'label': original_labels
    })
    before_dedup = len(smiles_label_df)
    smiles_label_df = smiles_label_df.drop_duplicates(subset=['SMILES'], keep='first').reset_index(drop=True)
    after_dedup = len(smiles_label_df)
    num_dropped = before_dedup - after_dedup
    print(f"Dropped {num_dropped} duplicate SMILES rows for {label} before augmentation.")
    original_smiles = smiles_label_df['SMILES'].tolist()
    original_labels = smiles_label_df['label'].values

    if Config.use_augmentation and not Config.debug:
        print(f"SMILES before augmentation: {len(original_smiles)}")
        smiles_aug, labels_aug = augment_smiles_dataset(original_smiles, original_labels, num_augments=1)
        print(f"SMILES after augmentation: {len(smiles_aug)} (increase: {len(smiles_aug) - len(original_smiles)})")
        original_smiles, original_labels = smiles_aug, labels_aug

    # After augmentation, deduplicate again at molecule level (canonical SMILES)
    canonical_smiles_aug = [get_canonical_smiles(s) for s in original_smiles]
    smiles_label_aug_df = pd.DataFrame({
        'SMILES': canonical_smiles_aug,
        'label': original_labels
    })
    smiles_label_aug_df = smiles_label_aug_df.drop_duplicates(subset=['SMILES'], keep='first').reset_index(drop=True)
    original_smiles = smiles_label_aug_df['SMILES'].tolist()
    original_labels = smiles_label_aug_df['label'].values

    fp_df, descriptor_df, valid_smiles, invalid_indices = smiles_to_combined_fingerprints_with_descriptors(original_smiles)

    print(f"Invalid indices for {label}: {invalid_indices}")
    y = np.delete(original_labels, invalid_indices)
    print(fp_df.shape)
    fp_df.reset_index(drop=True, inplace=True)

    if not descriptor_df.empty:
        X = pd.DataFrame(descriptor_df)
        X, kept_columns, scaler, median_values = preprocess_numerical_features(X, label)
        X.reset_index(drop=True, inplace=True)
        if not fp_df.empty:
            X = pd.concat([X, fp_df], axis=1)
    else:
        kept_columns = []
        scaler = None
        X = fp_df

    # Remove duplicate rows in X and corresponding values in y (feature-level duplicates)
    X_dup = X.duplicated(keep='first')
    if X_dup.any():
        print(f"Found {X_dup.sum()} duplicate rows in X for {label}, removing them.")
        X = X[~X_dup]
        y = y[~X_dup]
    print(f"After concat: {X.shape}")
    # Fill NaN in train with median from train
    # Only fill NaN with median if using neural network
    if Config.model_name == 'nn':
        X = X.fillna(median_values)
    check_inf_nan(X, y, label)

    # display_outlier_summary(y, X=X, name=label, z_thresh=3, iqr_factor=1.5, iso_contamination=0.01, lof_contamination=0.01)

    # Drop least important features from X and test

    least_important_features = []
    if getattr(Config, 'use_least_important_features_all_methods', False):
        for i in range(4):
            print(f"Iteration {i+1} for least important feature removal on {label}")
            least_important_feature = get_least_important_features_all_methods(X, y, label, model_name=Config.model_name)
            least_important_features.extend(least_important_feature)
            if len(least_important_feature) > 0:
                print(f"label: {label} Dropping least important features: {least_important_feature}")
                X = X.drop(columns=least_important_feature)
                print(f"After dropping least important features: {X.shape}")

    check_inf_nan(X, y, label)
    # Drop highly correlated features using label-specific correlation threshold from Config
    correlation_threshold = Config.correlation_thresholds.get(label, 1.0)
    if correlation_threshold < 1.0:
        X, correlated_features_dropped = drop_correlated_features(pd.DataFrame(X), threshold=correlation_threshold)
    else:
        correlated_features_dropped = []

    print(f"After correlation cut (threshold={correlation_threshold}): {X.shape}, dropped columns: {correlated_features_dropped}")
    print(f"After dropping correlated features: {X.shape}")
    check_inf_nan(X, y, label)

    if getattr(Config, 'use_variance_threshold', False):
        threshold = 0.01
        selector = VarianceThreshold(threshold=threshold)
        X_sel = selector.fit_transform(X)
        # Get mask of selected features
        selected_cols_variance = X.columns[selector.get_support()]
        # Convert back to DataFrame with column names
        X = pd.DataFrame(X_sel, columns=selected_cols_variance, index=X.index)
        print(f"After variance cut: {X.shape}")
        print(f'Type of X: {type(X)}')

    if Config.add_gaussian and not Config.debug:
        n_samples = 1000
        X, y = augment_dataset(X, y, n_samples=n_samples)
        print(f"After augment cut: {X.shape}")

    # --- Hold out 10% for final MAE calculation ---
    # X_main, X_holdout, y_main, y_holdout = train_test_split(X, y, test_size=0.1, random_state=Config.random_state)
    # Bin y for stratification
    y_bins = pd.qcut(y, q=5, duplicates='drop', labels=False)
    X_main, X_holdout, y_main, y_holdout = train_test_split(X, y, test_size=0.10, random_state=Config.random_state, stratify=y_bins)
    if Config.useAllDataForTraining == True:
        X_main = X
        y_main = y
    # --- Optionally apply PCA ---
    if getattr(Config, 'use_pca', False):
        X_main, X_holdout, pca = apply_pca(X_main, X_holdout, verbose=True)
    else:
        pca = None

    # --- Cross-Validation or Single Split (for speed) ---
    fold_maes = []
    test_preds = []
    val_preds = np.zeros(len(y_main))
    if getattr(Config, 'use_cross_validation', True):
        nfold = 10
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=Config.random_state)
        # For regression, bin y_main for stratification
        y_bins = pd.qcut(y_main, q=nfold, duplicates='drop', labels=False)
        splits = skf.split(X_main, y_bins)
    else:
        # Use a single split: 80% train, 20% val
        train_idx, val_idx = train_test_split(
            np.arange(len(X_main)), test_size=0.2, random_state=Config.random_state
        )
        splits = [(train_idx, val_idx)]
        nfold = 1

    return {
        "X_main": X_main,
        "X_holdout": X_holdout,
        "y_main": y_main,
        "y_holdout": y_holdout,
        "kept_columns": kept_columns,
        "scaler": scaler,
        "median_values": median_values,
        "least_important_features": least_important_features,
        "correlated_features_dropped": correlated_features_dropped,
        "selector": selector if getattr(Config, 'use_variance_threshold', False) else None,
        "selected_cols_variance": selected_cols_variance if getattr(Config, 'use_variance_threshold', False) else None,
        "pca": pca,
        "splits": splits,
        "nfold": nfold,
        "fold_maes": fold_maes,
        "test_preds": test_preds,
        "val_preds": val_preds
    }

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

def load_label_data(label, model_dir=None):
    if model_dir is not None:
        # Load model and data for the specified label
        model_path = os.path.join(model_dir, f"{label}_model.pkl")
        data_path = os.path.join(model_dir, f"{label}_data.pkl")
        model = joblib.load(model_path)
        data = joblib.load(data_path)
        return model, data
    return None, None
