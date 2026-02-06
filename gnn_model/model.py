import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINEConv, GATConv, global_mean_pool, global_max_pool

def create_dynamic_mlp(input_dim, layer_list, dropout_list):
    """
    Helper function to dynamically build the task-specific MLP.
    """
    layers = []
    current_dim = input_dim
    
    for neurons, dropout in zip(layer_list, dropout_list):
        layers.append(torch.nn.Linear(current_dim, neurons))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Dropout(dropout))
        current_dim = neurons
        
    layers.append(torch.nn.Linear(current_dim, 1))
    
    return torch.nn.Sequential(*layers)

class GNNModel(torch.nn.Module):
    """
    Defines the Graph Neural Network architecture.
    """
    def __init__(self, num_node_features, hidden_channels=128):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x

class TaskSpecificGNN(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, num_global_features,
                 hidden_channels_gnn, mlp_neurons, mlp_dropouts, heads=8):
        super(TaskSpecificGNN, self).__init__()
        
        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features
        self.num_global_features = num_global_features
        self.hidden_channels_gnn = hidden_channels_gnn
        
        self.conv1 = GATConv(
            in_channels=num_node_features,
            out_channels=hidden_channels_gnn,
            heads=heads,
            edge_dim=num_edge_features,
            concat=True,
            dropout=0.1
        )
        
        self.conv2 = GATConv(
            in_channels=hidden_channels_gnn * heads,
            out_channels=hidden_channels_gnn,
            heads=heads,
            edge_dim=num_edge_features,
            concat=False,
            dropout=0.1
        )
        
        combined_dim = hidden_channels_gnn + num_global_features
        self.mlp = create_dynamic_mlp(combined_dim, mlp_neurons, mlp_dropouts)

    def forward(self, data):
        x, edge_index, edge_attr, u, batch = data.x, data.edge_index, data.edge_attr, data.u, data.batch
        
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.elu(self.conv2(x, edge_index, edge_attr))
        
        graph_embed = global_mean_pool(x, batch)
        
        u_expanded = u[batch[::len(batch)//len(u)] if len(batch) > len(u) else torch.arange(len(u))]
        if u_expanded.shape[0] != graph_embed.shape[0]:
            u_expanded = u.repeat(graph_embed.shape[0], 1)
        
        combined = torch.cat([graph_embed, u_expanded], dim=1)
        
        out = self.mlp(combined)
        return out
