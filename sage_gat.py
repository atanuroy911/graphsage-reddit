from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


from torch_geometric.utils import (
    coalesce,
    to_networkx,
    train_test_split_edges,
    add_self_loops, 
    degree,
)
from torch_geometric.nn import (
    GAT, 
    GATConv,
    SAGEConv,
    global_max_pool,
    global_mean_pool,
    MessagePassing,
)

import torch_geometric.transforms as T

from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import (
    GAT, 
    GATConv,
    SAGEConv,
    global_max_pool,
    global_mean_pool,
    MessagePassing,
)

from torch_geometric.data import NeighborSampler



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(SAGEConv(in_channels, hidden_channels))
            in_channels = hidden_channels
        self.num_layers = num_layers

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_()
                x = F.dropout(x, p=0.5, training=self.training)
        return x

    @torch.no_grad()
    def inference(self, x_all, subgraph_loader):
        pbar = tqdm(total=len(subgraph_loader.dataset) * len(self.convs))
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        for i, conv in enumerate(self.convs):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                x = x_all[n_id].to(device)
                x = conv(x, adj.to(device))
                if i < len(self.convs) - 1:
                    x = x.relu_()
                xs.append(x.cpu())
                pbar.update(batch_size)
            x_all = torch.cat(xs, dim=0)
        pbar.close()
        return x_all

class NodeSampler(torch.utils.data.DataLoader):
    def __init__(self, data, size, num_layers):
        super(NodeSampler, self).__init__(data, batch_size=size, shuffle=True)

        self.size = size
        self.num_layers = num_layers

    def collate_fn(self, data_list):
        batch_size = len(data_list)
        n_id = torch.cat([data.n_id for data in data_list], dim=0)
        adj = NeighborSampler(data_list, size=self.size, num_hops=self.num_layers, batch_size=batch_size,
                              shuffle=False).adj_t
        return batch_size, n_id, adj

    

