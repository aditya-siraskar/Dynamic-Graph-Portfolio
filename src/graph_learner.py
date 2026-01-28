import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np  # <--- Added this missing import

class DynamicGraphLearner(nn.Module):
    """
    Phase 2 Module:
    Learns the Adjacency Matrix (A) dynamically from node features.
    
    Mechanism: Multi-Head Self-Attention (simplified to single head for stability).
    Input:  (Batch, Nodes, Features)
    Output: (Batch, Nodes, Nodes) -> The Adjacency Matrix
    """
    def __init__(self, input_dim, num_nodes, hidden_dim=64, sparsity_threshold=0.2):
        super(DynamicGraphLearner, self).__init__()
        
        self.num_nodes = num_nodes
        self.sparsity_threshold = sparsity_threshold
        
        # 1. Linear Projections for Attention (Query and Key)
        self.weight_query = nn.Linear(input_dim, hidden_dim)
        self.weight_key = nn.Linear(input_dim, hidden_dim)
        
        # 2. Activation for the attention scores
        self.act = nn.ReLU() 
        
    def forward(self, x):
        """
        x shape: (Batch_Size, Num_Nodes, Input_Dim)
        """
        # Step 1: Project features
        query = self.weight_query(x) 
        key = self.weight_key(x)     
        
        # Step 2: Calculate Attention (Q * K^T)
        key_transposed = key.transpose(1, 2)
        attention = torch.matmul(query, key_transposed)
        
        # Scale
        scale = torch.sqrt(torch.tensor(query.shape[-1], dtype=torch.float32, device=x.device))
        attention = attention / scale
        
        # Step 3: Activation (ReLU) & Symmetrization
        adj = self.act(attention)
        adj = (adj + adj.transpose(1, 2)) / 2
        
        # Step 4: Sparsity (Thresholding)
        mask = (adj > self.sparsity_threshold).float()
        adj = adj * mask
        
        # Step 5: Add Self-Loops (Crucial for GNNs)
        # Every stock should inherently "attend" to itself.
        batch_size, num_nodes, _ = adj.shape
        eye = torch.eye(num_nodes, device=x.device).unsqueeze(0).expand(batch_size, -1, -1)
        adj = adj + eye
        
        # Step 6: Symmetric Normalization (D^-0.5 * A * D^-0.5)
        # This keeps the matrix Symmetric!
        
        # Calculate Degree (Row Sum)
        d = torch.sum(adj, dim=2) + 1e-6 # Shape: (Batch, Nodes)
        d_inv_sqrt = torch.pow(d, -0.5)  # Shape: (Batch, Nodes)
        
        # Reshape for broadcasting: (Batch, Nodes, 1) and (Batch, 1, Nodes)
        d_mat = d_inv_sqrt.unsqueeze(2) 
        d_mat_t = d_inv_sqrt.unsqueeze(1)
        
        # Normalize: D^-0.5 * A * D^-0.5
        adj = d_mat * adj * d_mat_t
        
        return adj

# --- UNIT TEST BLOCK ---
if __name__ == "__main__":
    print("--- Testing Dynamic Graph Learner ---")
    
    # 1. Create Dummy Data
    batch_size = 32
    nodes = 10
    features = 6
    dummy_input = torch.randn(batch_size, nodes, features)
    
    # 2. Initialize Model
    learner = DynamicGraphLearner(input_dim=features, num_nodes=nodes, hidden_dim=32)
    
    # 3. Forward Pass
    adj_matrix = learner(dummy_input)
    
    # 4. Verify Output
    print(f"Output Adjacency Shape: {adj_matrix.shape} (Batch, Nodes, Nodes)")
    
    # Check properties
    first_adj = adj_matrix[0].detach().numpy()
    
    print("\nSample Adjacency Matrix (Top 3x3):")
    print(first_adj[:3, :3])
    
    # Check Symmetry
    is_symmetric = np.allclose(first_adj, first_adj.T, atol=1e-5)
    print(f"\nIs Matrix Symmetric? {is_symmetric}")
    
    if is_symmetric:
        print("✔ SUCCESS: Graph Learner is functioning correctly.")