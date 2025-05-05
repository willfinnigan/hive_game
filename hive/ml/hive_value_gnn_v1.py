import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.nn import RGCNConv, global_add_pool
from torch_geometric.data import Data, Batch # For creating dummy data/batches

# --- Configuration Placeholders ---
# You'll need to set these based on your actual feature engineering
# Example values:
NUM_NODE_FEATURES = 15  # e.g., IsPiece(1)+IsEmpty(1)+Type(8)+Color(2)+IsStacked(1)+StackLvl(1)+IsPinned(1)
NUM_RELATIONS = 3       # e.g., 0: Adjacent, 1: Stacked_On, 2: Covered_By
HIDDEN_CHANNELS = 64    # Size of hidden embeddings

class HiveValueGNN(torch.nn.Module):
    """
    A Graph Neural Network for Hive focusing *only* on predicting
    the game state value (win/loss/draw probability).
    Uses RGCNConv to handle different edge types.
    """
    def __init__(self, num_node_features, hidden_channels, num_relations):
        super().__init__()
        self.num_relations = num_relations

        # GNN Layers (using RGCN for typed edges)
        self.conv1 = RGCNConv(num_node_features, hidden_channels, num_relations)
        self.conv2 = RGCNConv(hidden_channels, hidden_channels, num_relations)
        # Add more conv layers here if needed for deeper message passing
        # self.conv3 = RGCNConv(hidden_channels, hidden_channels, num_relations)

        # Value Head MLP
        # Takes the aggregated graph embedding as input
        self.value_mlp = Sequential(
            Linear(hidden_channels, hidden_channels // 2),
            ReLU(),
            Linear(hidden_channels // 2, 1) # Output a single scalar value
        )

    def forward(self, x, edge_index, edge_type, batch):
        """
        Forward pass of the GNN.

        Args:
            x (Tensor): Node feature matrix [num_nodes, num_node_features]
            edge_index (LongTensor): Graph connectivity [2, num_edges]
            edge_type (LongTensor): Type of each edge [num_edges]
            batch (LongTensor): Batch vector [num_nodes], assigns each node to
                                its respective graph in the batch.

        Returns:
            Tensor: Predicted value for each graph in the batch [batch_size, 1]
        """
        # 1. Apply GNN layers
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = F.relu(self.conv2(x, edge_index, edge_type))
        # if self.conv3: x = F.relu(self.conv3(x, edge_index, edge_type))

        # 2. Pool node embeddings to get graph embeddings
        # global_add_pool sums node features for each graph in the batch
        graph_embedding = global_add_pool(x, batch) # [batch_size, hidden_channels]

        # 3. Pass graph embedding through Value Head MLP
        value_raw = self.value_mlp(graph_embedding) # [batch_size, 1]

        # 4. Apply Tanh activation to squash value into [-1, 1] range
        value = torch.tanh(value_raw)

        return value

# --- Example Usage ---

if __name__ == '__main__':
    # 1. Instantiate the Model
    model = HiveValueGNN(
        num_node_features=NUM_NODE_FEATURES,
        hidden_channels=HIDDEN_CHANNELS,
        num_relations=NUM_RELATIONS
    )
    print("Model Architecture:\n", model)

    # 2. Create Dummy Data (representing a batch of 2 graphs)
    # Graph 1: 5 nodes, 6 edges
    x1 = torch.randn(5, NUM_NODE_FEATURES)
    edge_index1 = torch.tensor([[0, 1, 1, 2, 3, 4],
                                [1, 0, 2, 3, 4, 0]], dtype=torch.long)
    edge_type1 = torch.randint(0, NUM_RELATIONS, (6,)) # Random edge types

    # Graph 2: 4 nodes, 4 edges
    x2 = torch.randn(4, NUM_NODE_FEATURES)
    edge_index2 = torch.tensor([[0, 1, 1, 2],
                                [1, 0, 2, 3]], dtype=torch.long)
    edge_type2 = torch.randint(0, NUM_RELATIONS, (4,))

    # Create PyG Data objects
    data1 = Data(x=x1, edge_index=edge_index1, edge_type=edge_type1)
    data2 = Data(x=x2, edge_index=edge_index2, edge_type=edge_type2)

    # Create a Batch object (PyG handles collation)
    # Note: You would typically use torch_geometric.loader.DataLoader for this
    batch = Batch.from_data_list([data1, data2])

    print("\nBatch Object:\n", batch)
    print("Batch size:", batch.num_graphs)

    # 3. Perform Forward Pass
    model.eval() # Set model to evaluation mode
    with torch.no_grad(): # Disable gradient calculation for inference
        predicted_values = model(batch.x, batch.edge_index, batch.edge_type, batch.batch)

    print("\nPredicted Values (Batch):\n", predicted_values)
    print("Shape:", predicted_values.shape) # Should be [batch_size, 1] = [2, 1]

    # --- Training Setup Snippet (Conceptual) ---
    # Assuming you have a dataset of PyG Data objects, each with a target value 'y'
    # dataset = YourHiveDataset(...) # List of Data(x=..., edge_index=..., edge_type=..., y=...)
    # loader = DataLoader(dataset, batch_size=32, shuffle=True)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # loss_fn = torch.nn.MSELoss()

    # for epoch in range(num_epochs):
    #     model.train()
    #     total_loss = 0
    #     for batch_data in loader:
    #         optimizer.zero_grad()
    #         pred_val = model(batch_data.x, batch_data.edge_index, batch_data.edge_type, batch_data.batch)
    #         target_val = batch_data.y.view(-1, 1).float() # Ensure correct shape and type
    #         loss = loss_fn(pred_val, target_val)
    #         loss.backward()
    #         optimizer.step()
    #         total_loss += loss.item() * batch_data.num_graphs
    #     avg_loss = total_loss / len(loader.dataset)
    #     print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")