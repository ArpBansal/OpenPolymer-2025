import os

BASE_PATH = '/kaggle/input/neurips-open-polymer-prediction-2025/'
RDKIT_AVAILABLE = True
TARGETS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

os.makedirs("NeurIPS", exist_ok=True)

class Config:
    useAllDataForTraining = False
    use_standard_scaler = True
    use_least_important_features_all_methods = True
    use_variance_threshold = False
    enable_param_tuning = False
    debug = False

    use_descriptors = False
    use_mordred = True
    use_maccs_fp = False
    use_morgan_fp = False
    use_atom_pair_fp = False
    use_torsion_fp = False
    use_chemberta = False
    chemberta_pooling = 'max'

    search_nn = False
    use_stacking = False
    model_name = 'xgb'

    feature_importance_method = 'permutation_importance'
    use_cross_validation = True
    use_pca = False
    pca_variance = 0.9999
    use_external_data = True
    use_augmentation = False
    add_gaussian = False
    random_state = 42

    n_least_important_features = {
        'xgb':     {'Tg': 20, 'FFV': 20, 'Tc': 22, 'Density': 19, 'Rg': 19},
        'catboost':{'Tg': 15, 'FFV': 15, 'Tc': 18, 'Density': 15, 'Rg': 15},
        'lgbm':    {'Tg': 18, 'FFV': 18, 'Tc': 20, 'Density': 17, 'Rg': 17},
        'extratrees':{'Tg': 22, 'FFV': 15, 'Tc': 10, 'Density': 25, 'Rg': 5},
        'randomforest':{'Tg': 21, 'FFV': 19, 'Tc': 21, 'Density': 18, 'Rg': 18},
        'balancedrf':{'Tg': 20, 'FFV': 20, 'Tc': 20, 'Density': 20, 'Rg': 20},
    }

    permutation_importance_log_path = "log/permutation_importance_log.xlsx"

    correlation_threshold_value = 0.96
    correlation_thresholds = {
        "Tg": correlation_threshold_value,
        "FFV": correlation_threshold_value,
        "Tc": correlation_threshold_value,
        "Density": correlation_threshold_value,
        "Rg": correlation_threshold_value
    }

config = Config()

if config.debug or config.search_nn:
    config.use_cross_validation = False
