import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch.nn import Linear, BatchNorm1d, Dropout


def layerwise_propagation(edge_index, seed_mask):
    """
    Performs a Breadth-First Search (BFS) starting from the seed nodes
    to determine the hop-distance (layer) of every node in the graph.
    """
    visited = seed_mask.clone()
    current_layer = seed_mask.nonzero(as_tuple=True)[0]
    layers = [current_layer]

    while current_layer.numel() > 0:
        mask = torch.isin(edge_index[0], current_layer)
        neighbors = edge_index[1][mask]

        unvisited_mask = ~visited[neighbors]
        next_layer = neighbors[unvisited_mask].unique()

        if next_layer.numel() == 0:
            break

        visited[next_layer] = True
        layers.append(next_layer)
        current_layer = next_layer

    return layers


class HierarchicalGATNet(torch.nn.Module):
    def __init__(self, in_channels=584, out_channels=2, max_hops=3):
        """
        Hierarchical Graph Attention Network (HGAT) for Pulmonary Artery-Vein classification.

        Args:
            in_channels (int): Input feature dimension per node (e.g. 584).
            out_channels (int): Number of target classes (default 2: Artery vs Vein).
            max_hops (int): Maximum hierarchy hops (default 3).
        """
        super().__init__()
        self.max_hops = max_hops
        # Concatenate one-hot hierarchy mask with node feature dimension
        gat_input_dim = in_channels + (max_hops + 1)

        self.conv1 = GATConv(gat_input_dim, 64, heads=4, concat=True)
        self.bn1 = BatchNorm1d(256)

        self.conv2 = GATConv(256, 128, heads=4, concat=False)
        self.bn2 = BatchNorm1d(128)

        self.residual_proj = Linear(in_channels, 128)
        self.gate_linear = Linear(128 + 1, 128)

        self.conv3 = GATConv(128, 64, heads=2, concat=False)
        self.bn3 = BatchNorm1d(64)
        self.classifier = Linear(64, out_channels)

    def forward(self, data):
        """
        Forward pass with hierarchy gating and residual connections.
        """
        x, edge_index = data.x, data.edge_index

        hop_counts = data.hierarchy_mask.argmax(dim=1).float()
        aug_x = torch.cat([x, data.hierarchy_mask], dim=1)

        h1 = F.dropout(F.elu(self.bn1(self.conv1(aug_x, edge_index))), p=0.1, training=self.training)
        h2 = self.bn2(self.conv2(h1, edge_index))

        gate = torch.sigmoid(self.gate_linear(torch.cat([h2, hop_counts.unsqueeze(1)], dim=1)))
        h2 = F.dropout(F.elu((h2 * gate) + self.residual_proj(x)), p=0.1, training=self.training)

        h3 = F.elu(self.bn3(self.conv3(h2, edge_index)))
        return F.log_softmax(self.classifier(h3), dim=1)

    def _compute_hierarchy_mask(self, edge_index, seed_mask):
        layers = layerwise_propagation(edge_index, seed_mask)
        hierarchy_mask = torch.zeros((seed_mask.size(0), self.max_hops + 1))
        for i, layer in enumerate(layers):
            if i > self.max_hops:
                break
            hierarchy_mask[layer, i] = 1
        return hierarchy_mask
