import torch
import os
import json
from torch_geometric.loader import DataLoader as PyGDataLoader
from sklearn.metrics import mean_absolute_error
from .model import TaskSpecificGNN

def train_gnn_model(label, train_data_list, val_data_list, mlp_neurons, mlp_dropouts, epochs=300):
    """
    (REVISED)
    - Accepts both train and val data lists.
    - Implements ReduceLROnPlateau scheduler based on val_loss.
    - Implements Early Stopping based on val_loss patience.
    """
    print(f"--- Training GNN for label: {label} ---")
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not train_data_list:
        print("Empty training data list!")
        return None
    if not val_data_list:
        print("Empty validation data list!")
        return None

    train_loader = PyGDataLoader(train_data_list, batch_size=32, shuffle=True, drop_last=True) 
    val_loader = PyGDataLoader(val_data_list, batch_size=32, shuffle=False)

    first_data = train_data_list[0]
    num_node_features = first_data.x.shape[1]
    num_global_features = first_data.u.shape[1]
    num_edge_features = first_data.edge_attr.shape[1]
    
    print(f"Model Features (Scaled): Nodes={num_node_features}, Edges={num_edge_features}, Global={num_global_features}")

    model = TaskSpecificGNN(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        num_global_features=num_global_features,
        hidden_channels_gnn=128, 
        mlp_neurons=mlp_neurons,
        mlp_dropouts=mlp_dropouts
    ).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    criterion = torch.nn.L1Loss() 

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    PATIENCE_EPOCHS = 30

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out.squeeze(), batch.y)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            train_batches += 1

        avg_train_loss = train_loss_sum / train_batches if train_batches > 0 else 0.0

        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                out = model(batch)
                loss = criterion(out.squeeze(), batch.y)
                val_loss_sum += loss.item()
                val_batches += 1

        avg_val_loss = val_loss_sum / val_batches if val_batches > 0 else 0.0

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        if epochs_no_improve >= PATIENCE_EPOCHS:
            print(f"Early stopping at epoch {epoch}. Val loss hasn't improved for {PATIENCE_EPOCHS} epochs.")
            break
            
    print(f"--- GNN training for {label} complete. Best Val Loss: {best_val_loss:.6f} ---")
    return model

def save_gnn_model(model, label, model_dir="models/gnn"):
    """
    (MODIFIED) Saves the GNN model state_dict and its full constructor config.
    """
    if model is None:
        print("No model to save!")
        return

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"gnn_model_{label}.pth")
    config_path = os.path.join(model_dir, f"gnn_config_{label}.json")

    torch.save(model.state_dict(), model_path)
    
    with open(config_path, 'w') as f:
        json.dump(model.__dict__, f)
        
    print(f"Saved final model for {label} to {model_path}")


def load_gnn_model(label, model_dir="models/gnn"):
    """
    (MODIFIED) Loads a saved GNN model using its full config file.
    """
    model_path = os.path.join(model_dir, f"gnn_model_{label}.pth")
    config_path = os.path.join(model_dir, f"gnn_config_{label}.json")
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not os.path.exists(model_path) or not os.path.exists(config_path):
        print(f"Model files not found for {label} at {model_dir}")
        return None

    with open(config_path, 'r') as f:
        config = json.load(f)
    
    try:
        model = TaskSpecificGNN(
            num_node_features=config['num_node_features'],
            num_edge_features=config['num_edge_features'],
            num_global_features=config['num_global_features'],
            hidden_channels_gnn=config['hidden_channels_gnn'],
            mlp_neurons=config.get('mlp_neurons', [128, 64]),
            mlp_dropouts=config.get('mlp_dropouts', [0.2, 0.2])
        ).to(DEVICE)

    except Exception as e:
        print(f"Error loading model for {label}: {e}")
        return None
    
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print(f"Loaded GNN model for {label} from {model_path}")
    return model
