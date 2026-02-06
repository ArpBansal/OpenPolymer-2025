import torch
import numpy as np
from torch_geometric.loader import DataLoader as PyGDataLoader

def predict_with_gnn(trained_model, test_smiles, label, u_scaler, x_scaler, atom_map_len):
    """
    (MODIFIED for Full Scaling)
    - Requires both u_scaler (global) and x_scaler (node) to transform features.
    - Returns SCALED predictions.
    """
    from .data_processing import smiles_to_graph_label_specific, scale_graph_features
    
    if trained_model is None or u_scaler is None or x_scaler is None:
        print("Missing model or scalers!")
        return np.full(len(test_smiles), np.nan)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    test_data_list = [smiles_to_graph_label_specific(s, label, y_val=None) for s in test_smiles]
    
    valid_indices = [i for i, data in enumerate(test_data_list) if data is not None]
    valid_test_data = [data for data in test_data_list if data is not None]

    if not valid_test_data:
        print("No valid test data after featurization!")
        return np.full(len(test_smiles), np.nan)
        
    try:
        valid_test_data = scale_graph_features(valid_test_data, u_scaler, x_scaler, atom_map_len)
    except Exception as e:
        print(f"Error scaling test features: {e}")
        return np.full(len(test_smiles), np.nan)

    test_loader = PyGDataLoader(valid_test_data, batch_size=32, shuffle=False) 

    trained_model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)
            out = trained_model(batch)
            all_preds.append(out.cpu())

    test_preds_tensor = torch.cat(all_preds, dim=0).numpy().flatten()
    
    final_predictions = np.full(len(test_smiles), np.nan)
    if len(test_preds_tensor) == len(valid_indices):
        final_predictions[valid_indices] = test_preds_tensor
    else:
        print(f"Warning: Mismatch in prediction array sizes. Expected {len(valid_indices)}, got {len(test_preds_tensor)}")
        for idx, pred in zip(valid_indices[:len(test_preds_tensor)], test_preds_tensor):
            final_predictions[idx] = pred

    return final_predictions
