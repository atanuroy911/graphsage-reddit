
import torch

# General
from matplotlib import pyplot as plt

import copy
from tqdm import tqdm

import scipy.sparse as sp

from sklearn.metrics import accuracy_score

import torch.nn.functional as F


from torch_geometric.loader import NeighborLoader, DataLoader

from torch_geometric.datasets import Reddit

from sage_gat import NodeSampler

print(torch.__version__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


path = 'Datasets'

dataset = Reddit(path, transform=None)

# Get the number of nodes and edges in the dataset
num_nodes = dataset[0].num_nodes
num_edges = dataset[0].num_edges

# Print some information about the dataset
print("Number of nodes:", num_nodes)
print("Number of edges:", num_edges)

# # Get the node features as a numpy array
node_features = dataset[0].x.numpy()

# # Plot a histogram of the node features
# plt.hist(node_features.flatten(), bins='auto')
# plt.title("Histogram of Node Features")
# plt.xlabel("Node Feature Value")
# plt.ylabel("Frequency")

# # Save the histogram plot to a file (e.g., PNG)
# plt.savefig('histogram_plot.png')

# # Close the plot
# plt.close()

del node_features

# Compute the sparsity of the adjacency matrix
edge_index = dataset[0].edge_index
adjacency_matrix = sp.coo_matrix((torch.ones(num_edges), edge_index), shape=(num_nodes, num_nodes))
sparsity = 1 - (num_edges / (num_nodes * (num_nodes - 1)))
print("Adjacency matrix sparsity:", sparsity)

# # Plot the adjacency matrix
# plt.spy(adjacency_matrix, markersize=0.1)
# plt.title("Adjacency Matrix")
# plt.xlabel("Node ID")
# plt.ylabel("Node ID")
# # Save the adjacency matrix to a file (e.g., PNG)
# plt.savefig('adjacency_matrix.png')

# # Close the plot
# plt.close()

del edge_index, adjacency_matrix, sparsity

# Community of graph 0
print(dataset[0].y)

# Subreddits (one-hot encoded) of graph 0
num_subreddits = dataset.num_classes
print(num_subreddits)
print(dataset[0].x[:, :num_subreddits])

# Already send node features/labels to GPU for faster access during sampling:
data = dataset[0].to(device, 'x', 'y')

kwargs = {'batch_size': 512, 'num_workers': 6, 'persistent_workers': True}
train_loader = NeighborLoader(data, input_nodes=data.train_mask,
                              num_neighbors=[25, 10], shuffle=True, **kwargs)

subgraph_loader = NeighborLoader(copy.copy(data), input_nodes=None,
                                 num_neighbors=[-1], shuffle=False, **kwargs)

# No need to maintain these features during evaluation:
del subgraph_loader.data.x, subgraph_loader.data.y

# Add global node index information.
subgraph_loader.data.num_nodes = data.num_nodes
subgraph_loader.data.n_id = torch.arange(data.num_nodes)

from sage_gat import SAGE

model = SAGE(dataset.num_features, 256, dataset.num_classes, 2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

sage_train_accs, sage_val_accs, sage_test_accs = [], [], []

# SAGE TRAINING 

def train(epoch):
    model.train()

    pbar = tqdm(total=int(len(train_loader.dataset)))
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = total_examples = 0
    for batch in train_loader:
        optimizer.zero_grad()
        y = batch.y[:batch.batch_size]
        y_hat = model(batch.x, batch.edge_index.to(device))[:batch.batch_size]
        loss = F.cross_entropy(y_hat, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * batch.batch_size
        total_correct += int((y_hat.argmax(dim=-1) == y).sum())
        total_examples += batch.batch_size
        pbar.update(batch.batch_size)
    pbar.close()

    torch.save(model.state_dict(), 'sage.pt')

    return total_loss / total_examples, total_correct / total_examples

# SAGE TESTING

@torch.no_grad()
def test():
    model.eval()

    # To load the saved model back
    # Note: Make sure the model architecture is defined before loading
    # loaded_model = SAGE(dataset.num_features, 256, dataset.num_classes).to(device)
    # loaded_model.load_state_dict(torch.load('sage_model.pth'))
    # loaded_model.eval()  # Set the model to evaluation mode
    
    y_hat = model.inference(data.x, subgraph_loader).argmax(dim=-1)
    y = data.y.to(y_hat.device)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((y_hat[mask] == y[mask]).sum()) / int(mask.sum()))
    return accs

for epoch in range(1, 11):
    loss, acc = train(epoch)
    print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')
    train_acc, val_acc, test_acc = test()
    print(f'Epoch: {epoch:02d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
          f'Test: {test_acc:.4f}')

    # store accuracy values for SAGE
    sage_train_accs.append(train_acc)
    sage_val_accs.append(val_acc)
    sage_test_accs.append(test_acc)
    
    # delete intermediate tensors
    del loss, acc, train_acc, val_acc, test_acc

   # create a list of labels for each epoch
epochs = range(1, 11)
epoch_labels = [f"Epoch {i}" for i in epochs]

# Plot the training and validation accuracies
plt.plot(epochs, sage_train_accs, label='Train')
plt.plot(epochs, sage_val_accs, label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('SAGE with GAT Accuracy')
plt.legend()
plt.savefig('sage_gat_accuracy.png')
plt.close()