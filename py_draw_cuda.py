from torch_geometric.datasets import Planetoid
import numpy as np

dataset = Planetoid("./dataset", 'Cora')
data = dataset[0]

adj_matrix = np.zeros((data.num_nodes, data.num_nodes))
for edge in data.edge_index.T:
    adj_matrix[edge[0], edge[1]] = 1

from torch_geometric.loader import ClusterData, ClusterLoader
cluster_data = ClusterData(data, num_parts=16, recursive=False, save_dir=dataset.processed_dir)
train_loader = ClusterLoader(cluster_data, batch_size=1, shuffle=True, num_workers=1)

print(cluster_data.data.adj)

# for batch in train_loader:
#     print(batch.num_nodes, batch.num_edges)

m = cluster_data.data.adj.to_dense()

import matplotlib.pyplot as plt
plt.figure(3)
plt.spy(adj_matrix, markersize=1, precision=0.01)
plt.show()
pass