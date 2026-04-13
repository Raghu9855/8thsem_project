import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
from .common_blocks import PerChannelCNNExtractor

class CNNGNNModel(nn.Module):
    def __init__(self, eeg_channels, latent_dim=64, num_classes=2):
        super(CNNGNNModel, self).__init__()
        
        self.feature_dim = 16
        self.cnn_extractor = PerChannelCNNExtractor(out_features=self.feature_dim)
        
        # GCN layers
        self.gcn1 = GCNConv(self.feature_dim, 32)
        self.gcn2 = GCNConv(32, 64)
        self.relu = nn.ReLU()
        
        # We assume a fully connected graph constraint between all channels
        # edge_index is created on the fly in forward pass since channel count varies slightly 
        # based on dataset, though mostly ~20-23
        
        self.fc1 = nn.Linear(64 + latent_dim, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)
        
    def create_fully_connected_edge_index(self, num_nodes, device):
        # Create a complete graph without self loops
        row = []
        col = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    row.append(i)
                    col.append(j)
        return torch.tensor([row, col], dtype=torch.long, device=device)

    def forward(self, x_eeg, x_latent):
        # x_eeg: (Batch, Channels, FreqBins, TimeBins)
        B, C, _, _ = x_eeg.shape
        device = x_eeg.device
        
        # 1. Independent channel features
        node_features = self.cnn_extractor(x_eeg) # (B, C, feature_dim)
        
        # 2. Graph topology
        edge_index = self.create_fully_connected_edge_index(C, device)
        
        # 3. GNN Process
        # torch_geometric usually processes a batch as a single large disjoint graph
        # Here we manually batch the node features
        # Reshape to (Batch * Channels, feature_dim)
        x_gnn = node_features.view(B * C, -1)
        
        # Create batched edge_index
        # shift edges for each graph in the batch
        batch_edge_index = []
        for b in range(B):
            batch_edge_index.append(edge_index + b * C)
        batch_edge_index = torch.cat(batch_edge_index, dim=1)
        
        # Create batch vector
        batch_vec = torch.arange(B, device=device).repeat_interleave(C)
        
        # GCN Layers
        x_gnn = self.gcn1(x_gnn, batch_edge_index)
        x_gnn = self.relu(x_gnn)
        x_gnn = self.gcn2(x_gnn, batch_edge_index)
        x_gnn = self.relu(x_gnn)
        
        # Global pooling across nodes (channels) per graph
        x_graph = global_mean_pool(x_gnn, batch_vec)  # (B, 64)
        
        # 4. Integrate Latent Features
        combined = torch.cat((x_graph, x_latent), dim=1)
        
        out = self.relu(self.fc1(combined))
        out = self.dropout(out)
        out = self.fc2(out)
        return out
