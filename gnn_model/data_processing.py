import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.mixture import GaussianMixture
from torch_geometric.data import Data
import torch

RDKIT_AVAILABLE = True
TARGETS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
BASE_PATH = '/kaggle/input/neurips-open-polymer-prediction-2025/'

ATOM_MAP = {
    'C': 0, 'N': 1, 'O': 2, 'F': 3, 'P': 4, 'S': 5, 'Cl': 6, 'Br': 7, 'I': 8, 'H': 9,
    'Si': 10, 'Na': 11, '*' : 12, 'B': 13, 'Ge': 14, 'Sn': 15, 'Se': 16, 'Te': 17, 'Ca': 18, 'Cd': 19,
}

LABEL_SPECIFIC_FEATURES = {
    'Tg': ["HallKierAlpha", "MolLogP", "NumRotatableBonds", "TPSA"],
    'FFV': ["NHOHCount", "NumRotatableBonds", "MolWt", "TPSA"],
    'Tc': ["MolLogP", "NumValenceElectrons", "SPS", "MolWt"],
    'Density': ["MolWt", "MolMR", "FractionCSP3", "NumHeteroatoms"],
    'Rg': ["HallKierAlpha", "MolWt", "NumValenceElectrons", "qed"]
}

RDKIT_DESC_CALCULATORS = {name: func for name, func in Descriptors.descList}
RDKIT_DESC_CALCULATORS['qed'] = Descriptors.qed

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
            if mol is None:
                return None
            return Chem.MolToSmiles(mol)
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
        print(f"      No valid data for {target}, skipping...")
        return df_train
    
    df_extra = df_extra.groupby('SMILES', as_index=False)[target].mean()
    
    cross_smiles = set(df_extra['SMILES']) & set(df_train['SMILES'])
    unique_smiles_extra = set(df_extra['SMILES']) - set(df_train['SMILES'])

    filled_count = 0
    for smile in df_train[df_train[target].isnull()]['SMILES'].tolist():
        if smile in cross_smiles:
            df_train.loc[df_train['SMILES'] == smile, target] = df_extra.loc[df_extra['SMILES'] == smile, target].values[0]
            filled_count += 1
    
    extra_to_add = df_extra[df_extra['SMILES'].isin(unique_smiles_extra)].copy()
    if len(extra_to_add) > 0:
        for col in df_train.columns:
            if col not in extra_to_add.columns:
                extra_to_add[col] = np.nan
        extra_to_add = extra_to_add[df_train.columns]
        df_train = pd.concat([df_train, extra_to_add], ignore_index=True)

    n_samples_after = len(df_train[df_train[target].notnull()])
    print(f'      {target}: +{n_samples_after-n_samples_before} samples, +{len(unique_smiles_extra)} unique SMILES')
    print(f"      Filled {filled_count} missing entries in train for {target}")
    print(f"      Added {len(extra_to_add)} new entries for {target}")
    return df_train

def safe_load_dataset(path, target, processor_func, description):
    try:
        print(f"   Loading {description}...")
        df = pd.read_csv(path) if path.endswith('.csv') else pd.read_excel(path)
        df = processor_func(df)
        if target not in df.columns:
            print(f"      Warning: {target} not in {description}")
            return None
        print(f"      Loaded {len(df)} rows with {target}")
        external_datasets.append((target, df))
        return (target, df)
    except Exception as e:
        print(f"   Could not load {description}: {e}")
        return None

def separate_subtables(train_df):
    labels = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    subtables = {}
    for label in labels:
        subtables[label] = train_df[train_df[label].notnull()][['SMILES', label]].reset_index(drop=True)

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
        augmented_smiles.append(smiles)
        augmented_labels.append(label)
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            for _ in range(num_augments):
                random_smiles = Chem.MolToSmiles(mol, doRandom=True)
                augmented_smiles.append(random_smiles)
                augmented_labels.append(label)

    return augmented_smiles, augmented_labels

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
        raise TypeError("X must be a pd.DataFrame or np.ndarray")

    X.columns = X.columns.astype(str)

    if isinstance(y, np.ndarray):
        y = pd.Series(y)
    elif not isinstance(y, pd.Series):
        raise TypeError("y must be a pd.Series or np.ndarray")

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

def smiles_to_graph(smiles_str: str, y_val=None):
    """
    Converts a SMILES string to a graph, adding selected global
    molecular features to each node's feature vector.
    """
    try:
        mol = Chem.MolFromSmiles(smiles_str)
        if mol is None:
            return None

        num_atoms = mol.GetNumAtoms()
        if num_atoms == 0:
            return None

        node_features = []
        for atom in mol.GetAtoms():
            atom_symbol = atom.GetSymbol()
            atom_idx = ATOM_MAP.get(atom_symbol, len(ATOM_MAP))
            one_hot = [0] * (len(ATOM_MAP) + 1)
            one_hot[atom_idx] = 1

            degree = atom.GetDegree()
            formal_charge = atom.GetFormalCharge()
            num_hs = atom.GetTotalNumHs()
            hybridization = int(atom.GetHybridization())
            is_aromatic = int(atom.IsInRing())

            feature_vector = one_hot + [degree, formal_charge, num_hs, hybridization, is_aromatic]
            node_features.append(feature_vector)

        x = torch.tensor(node_features, dtype=torch.float)

        edge_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_list.append([i, j])
            edge_list.append([j, i])

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.empty((2, 0), dtype=torch.long)

        data = Data(x=x, edge_index=edge_index)
        if y_val is not None:
            data.y = torch.tensor([y_val], dtype=torch.float)

        return data
    except Exception as e:
        return None

def smiles_to_graph_label_specific(smiles_str: str, label: str, y_val=None):
    """
    (BASELINE VERSION - SIMPLE FEATURES)
    - This is the original hybrid GNN featurizer that produced your best score.
    - Node Features (x): Atom one-hot (20) + 5 atom features = 25 features.
    - Edge Features (edge_attr): Bond type as double = 1 feature.
    - Global Features (u): Label-specific descriptors are stored separately in 'data.u'.
    """
    try:
        mol = Chem.MolFromSmiles(smiles_str)
        if mol is None:
            return None

        num_atoms = mol.GetNumAtoms()
        if num_atoms == 0:
            return None

        node_features = []
        for atom in mol.GetAtoms():
            atom_symbol = atom.GetSymbol()
            atom_idx = ATOM_MAP.get(atom_symbol, len(ATOM_MAP))
            one_hot = [0] * (len(ATOM_MAP) + 1)
            one_hot[atom_idx] = 1

            degree = atom.GetDegree()
            formal_charge = atom.GetFormalCharge()
            num_hs = atom.GetTotalNumHs()
            hybridization = int(atom.GetHybridization())
            is_aromatic = int(atom.IsInRing())

            feature_vector = one_hot + [degree, formal_charge, num_hs, hybridization, is_aromatic]
            node_features.append(feature_vector)

        x = torch.tensor(node_features, dtype=torch.float)

        edge_list = []
        edge_attrs = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_type = bond.GetBondTypeAsDouble()

            edge_list.append([i, j])
            edge_attrs.append([bond_type])
            edge_list.append([j, i])
            edge_attrs.append([bond_type])

        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float)

        global_features = []
        for feat_name in LABEL_SPECIFIC_FEATURES.get(label, []):
            calculator_func = RDKIT_DESC_CALCULATORS.get(feat_name, None)
            if calculator_func is not None:
                try:
                    val = calculator_func(mol)
                    if val is None or np.isnan(val) or np.isinf(val):
                        val = 0.0
                except:
                    val = 0.0
            else:
                val = 0.0
            global_features.append(val)

        if not global_features:
            global_features = [0.0]

        u = torch.tensor([global_features], dtype=torch.float)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, u=u)

        if y_val is not None:
            data.y = torch.tensor([y_val], dtype=torch.float)

        return data
        
    except Exception as e:
        print(f"Error processing SMILES {smiles_str}: {e}")
        return None

def scale_graph_features(data_list, u_scaler, x_scaler, atom_map_len):
    """Applies fitted scalers in-place to a list of Data objects."""
    try:
        for data in data_list:
            if data is None:
                continue
                
            u_scaled = u_scaler.transform(data.u.numpy())
            data.u = torch.tensor(u_scaled, dtype=torch.float)
            
            x_node_only = data.x[:, atom_map_len:].numpy()
            x_node_scaled = x_scaler.transform(x_node_only)
            data.x[:, atom_map_len:] = torch.tensor(x_node_scaled, dtype=torch.float)
            
    except Exception as e:
        print(f"Error scaling features: {e}")
        raise
    return data_list
