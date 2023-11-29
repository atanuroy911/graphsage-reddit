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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SAGEN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=8, dropout=0.2))
        self.convs.append(GATConv(hidden_channels * 8, out_channels, heads=1, dropout=0.2))
       
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
            for batch in subgraph_loader:
                x = x_all[batch.n_id.to(x_all.device)].to(device)
                x = conv(x, batch.edge_index.to(device))
                if i < len(self.convs) - 1:
                    x = x.relu_()
                xs.append(x[:batch.batch_size].cpu())
                pbar.update(batch.batch_size)
            x_all = torch.cat(xs, dim=0)
        pbar.close()
        return x_all
    

