import torch
from torch.nn import Linear, ModuleDict, BCEWithLogitsLoss
from torch_geometric.nn import HGTConv, Linear as PyGLinear
import torch_geometric.transforms as T
from graph_builder import load_graph_data
from sklearn.metrics import roc_auc_score
import os

class HeteroLinkPredictor(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, metadata, edge_attr_dim=4):
        super().__init__()
        
        # Project initial raw node features to the hidden dimensionality
        self.lin_dict = ModuleDict()
        for node_type in metadata[0]:
            self.lin_dict[node_type] = PyGLinear(-1, hidden_channels)
            
        # HGT layers
        self.conv1 = HGTConv(hidden_channels, hidden_channels, metadata, heads=4)
        self.conv2 = HGTConv(hidden_channels, out_channels, metadata, heads=4)
        
        # Decoder layer
        self.decoder_lin1 = Linear(out_channels * 2 + edge_attr_dim, hidden_channels)
        self.decoder_lin2 = Linear(hidden_channels, 1)

    def forward(self, x_dict, edge_index_dict, target_edge_index, target_edge_attr):
        # 1. Project nodes
        h_dict = {}
        for node_type, x in x_dict.items():
            h_dict[node_type] = self.lin_dict[node_type](x).relu()
            
        # 2. HGT Message Passing
        h_dict = self.conv1(h_dict, edge_index_dict)
        h_dict = {key: val.relu() for key, val in h_dict.items()}
        h_dict = self.conv2(h_dict, edge_index_dict)
        
        # 3. Predict edge attribute
        src, dst = target_edge_index
        z_src = h_dict['product'][src]
        z_dst = h_dict['warehouse'][dst]
        
        z = torch.cat([z_src, z_dst, target_edge_attr], dim=-1)
        z = self.decoder_lin1(z).relu()
        z = self.decoder_lin2(z)
        return z.view(-1)

def train_and_save_model():
    print("Loading Data...")
    data, suppl_map, wh_map, prod_map, shipments_df = load_graph_data()
    
    # Make graph undirected so all nodes receive messages
    data = T.ToUndirected()(data)
    
    # Initialize Model
    model = HeteroLinkPredictor(hidden_channels=32, out_channels=32, metadata=data.metadata(), edge_attr_dim=4)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = BCEWithLogitsLoss()
    
    target_edge = ('product', 'shipped_to', 'warehouse')
    edge_index = data[target_edge].edge_index
    edge_attr = data[target_edge].edge_attr
    y = data[target_edge].y
    num_edges = edge_index.size(1)

    perm = torch.randperm(num_edges)
    train_idx = perm[:int(0.8 * num_edges)]
    test_idx = perm[int(0.8 * num_edges):]

    train_edge_index = edge_index[:, train_idx]
    train_edge_attr = edge_attr[train_idx]
    train_y = y[train_idx]

    test_edge_index = edge_index[:, test_idx]
    test_edge_attr = edge_attr[test_idx]
    test_y = y[test_idx]

    print("Training Model...")
    for epoch in range(1, 101):
        model.train()
        optimizer.zero_grad()
        
        out = model(data.x_dict, data.edge_index_dict, train_edge_index, train_edge_attr)
        
        loss = criterion(out, train_y)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_out = model(data.x_dict, data.edge_index_dict, test_edge_index, test_edge_attr)
                test_loss = criterion(test_out, test_y)
                probs = torch.sigmoid(test_out).cpu().numpy()
                auc = roc_auc_score(test_y.cpu().numpy(), probs)
                print(f"Epoch {epoch:03d}, Train Loss: {loss:.4f}, Test Loss: {test_loss:.4f}, Test AUC: {auc:.4f}")

    os.makedirs("saved_models", exist_ok=True)
    torch.save(model.state_dict(), "saved_models/hetero_hgt.pth")
    print("Model saved to saved_models/hetero_hgt.pth")

if __name__ == "__main__":
    train_and_save_model()
