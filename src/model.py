import torch
import torch.nn as nn
import torch.nn.functional as F
from src.graph_learner import DynamicGraphLearner

class GATLayer(nn.Module):
    """
    Custom GAT Layer that handles Batched Adjacency Matrices.
    Standard PyTorch Geometric layers often struggle with dynamic (changing) graphs.
    Equation: H_out = Activation( A * H_in * W )
    """
    def __init__(self, in_features, out_features):
        super(GATLayer, self).__init__()
        self.weight = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x, adj):
        """
        x: (Batch, Nodes, In_Features)
        adj: (Batch, Nodes, Nodes) -> Learned by Phase 2
        """
        # 1. Linear Transform: H * W
        # Shape: (Batch, Nodes, Out_Features)
        h_trans = self.weight(x)
        
        # 2. Graph Aggregation: A * (H*W)
        # Batched Matrix Multiplication
        # (Batch, N, N) @ (Batch, N, F_out) -> (Batch, N, F_out)
        h_agg = torch.matmul(adj, h_trans)
        
        # 3. Non-linearity
        return self.act(h_agg + self.bias)

class PortfolioGNN(nn.Module):
    """
    The Full Hybrid Architecture:
    [Input] -> [LSTM (Time)] -> [Graph Learner (Structure)] -> [GAT (Space)] -> [Allocations]
    """
    def __init__(self, num_nodes, num_features, lstm_hidden=64, gnn_hidden=32):
        super(PortfolioGNN, self).__init__()
        
        # 1. Temporal Encoder (LSTM)
        # Processes the time-window for EACH node independently
        self.lstm = nn.LSTM(input_size=num_features, 
                            hidden_size=lstm_hidden, 
                            num_layers=1, 
                            batch_first=True)
        
        # 2. Graph Structure Learner (From Phase 2)
        # Uses the LSTM output to figure out market structure
        self.graph_learner = DynamicGraphLearner(input_dim=lstm_hidden, 
                                                 num_nodes=num_nodes, 
                                                 hidden_dim=gnn_hidden)
        
        # 3. Spatial GNN Layers
        # Aggregates info based on the learned structure
        self.gat1 = GATLayer(lstm_hidden, gnn_hidden)
        self.gat2 = GATLayer(gnn_hidden, gnn_hidden)
        
        # 4. Portfolio Decoder
        # Collapses the node features into a single weight per stock
        self.decoder = nn.Sequential(
            nn.Linear(gnn_hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # Output 1 score per node
        )
        
        # Softmax over the 'Nodes' dimension to ensure weights sum to 1
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Input x: (Batch, Time_Window, Nodes, Features)
        This 4D tensor needs reshaping for the LSTM.
        """
        B, T, N, F = x.shape
        
        # --- Step A: Temporal Encoding ---
        # Reshape to (Batch * Nodes, Time, Features) to feed LSTM
        x_reshaped = x.view(B * N, T, F)
        
        # Run LSTM
        lstm_out, (h_n, c_n) = self.lstm(x_reshaped)
        
        # We only care about the LAST hidden state (the "current" state)
        # h_n shape: (1, Batch*Nodes, Hidden) -> Squeeze to (Batch*Nodes, Hidden)
        embedding = h_n[-1]
        
        # Reshape back to (Batch, Nodes, Hidden) for the Graph layers
        node_embeddings = embedding.view(B, N, -1)
        
        # --- Step B: Dynamic Graph Learning ---
        # Learn Adjacency Matrix A based on current market state
        adj_matrix = self.graph_learner(node_embeddings)
        
        # --- Step C: Spatial Aggregation (GAT) ---
        # Smooth features across connected stocks
        g1 = self.gat1(node_embeddings, adj_matrix)
        g2 = self.gat2(g1, adj_matrix) # Stacked layer
        
        # --- Step D: Portfolio Weight Generation ---
        # Generate raw scores: (Batch, Nodes, 1)
        scores = self.decoder(g2).squeeze(-1) # (Batch, Nodes)
        
        # Convert to Weights (Sum = 1)
        weights = self.softmax(scores)
        
        return weights, adj_matrix

# --- UNIT TEST ---
if __name__ == "__main__":
    print("--- Testing Full PortfolioGNN Model ---")
    
    # 1. Dummy Data
    # Batch=16, Window=30 days, Nodes=5, Features=6
    B, T, N, F = 16, 30, 5, 6
    dummy_x = torch.randn(B, T, N, F)
    
    # 2. Initialize Model
    model = PortfolioGNN(num_nodes=N, num_features=F)
    
    # 3. Forward Pass
    port_weights, learned_graph = model(dummy_x)
    
    # 4. Verification
    print(f"Input Shape: {dummy_x.shape}")
    print(f"Weights Output Shape: {port_weights.shape} (Should be Batch x Nodes)")
    print(f"Graph Output Shape: {learned_graph.shape} (Should be Batch x Nodes x Nodes)")
    
    # Check if weights sum to 1
    sample_sum = port_weights[0].sum().item()
    print(f"\nSum of Portfolio Weights (Sample 0): {sample_sum:.4f}")
    
    if abs(sample_sum - 1.0) < 1e-4:
        print("✔ SUCCESS: Weights correspond to a valid portfolio allocation.")
    else:
        print("✘ ERROR: Weights do not sum to 1.")