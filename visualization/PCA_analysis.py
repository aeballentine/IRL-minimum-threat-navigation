import io
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from evaluation import dijkstra_bwd
from nn_architecture import neighbors_of_four

warnings.filterwarnings(
    "ignore",
    message=r"You are using `torch.load` with `weights_only=False`.*"
)


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name='_load_from_bytes'):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
print(device)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# NEIGHBORS OF FOUR
neighbors = neighbors_of_four(dims=(25, 25), target=624)
neighbors_tensor = torch.from_numpy(neighbors[['left', 'right', 'up', 'down']].values).to(device)
distance = torch.from_numpy(neighbors[['dist']].values).float().to(device)
max_distance = torch.tensor([[torch.max(distance)]]).to(device)
distance = torch.concat((distance, max_distance))

# load the relevant feature functions for each
# easy threat field:
file_path_a = '2_23_heuristic_1_performance_0_finishes_624v2.pkl'
file_path_b = '2_23_heuristic_0_performance_1_finishes_624v2.pkl'
# random file:
# file_path_a = '2_23_heuristic_0_performance_1_finishes_624random.pkl'
# file_path_b = '2_23_heuristic_1_performance_2_finishes_624random.pkl'
with open(file_path_a, 'rb') as f:
    suboptimal = CPU_Unpickler(f).load()
with open(file_path_b, 'rb') as f:
    optimal = CPU_Unpickler(f).load()

# run the evaluation for heuristic = 0 (optimal and nn_policy_0)
starting_coords = np.arange(0, 624, 1)
features = []
classification = []

# find Dijkstra path
vertices = np.arange(0, 625, 1)
end_index = 624
constant = 0
heuristic = 0
my_feature_map = optimal['feature_function']
dijkstra_info = dijkstra_bwd(feature_function=my_feature_map,
                             vertices=vertices,
                             source=end_index,
                             neighbors=neighbors_tensor,
                             max_len=625,
                             constant=constant,
                             heuristic=heuristic)

for loc in starting_coords:
    node = loc
    path_features = my_feature_map[node][:3].clone()
    while node != end_index:
        previous_node = dijkstra_info[node].parent
        node = previous_node
        path_features += my_feature_map[node][:3].clone()
    features.append(path_features.numpy())
    classification.append(0)

for path in optimal['nn_paths']:
    path_features = torch.zeros((3,))
    for node in path:
        path_features += my_feature_map[node][:3].clone()
    if node != 624:
        print('failure')
    features.append(path_features.numpy())
    classification.append(1)

# run the evaluation for heuristic = 1 (suboptimal and nn_policy_1)
starting_coords = np.arange(0, 624, 1)

# find Dijkstra path
vertices = np.arange(0, 625, 1)
end_index = 624
constant = 0
heuristic = 1
my_feature_map = suboptimal['feature_function']
dijkstra_info = dijkstra_bwd(feature_function=my_feature_map,
                             vertices=vertices,
                             source=end_index,
                             neighbors=neighbors_tensor,
                             max_len=625,
                             constant=constant,
                             heuristic=heuristic)

for i, loc in enumerate(starting_coords):
    node = loc
    path_features = my_feature_map[node][:3].clone()
    counter = 0
    while node != end_index:
        previous_node = dijkstra_info[node].parent
        node = previous_node
        path_features += my_feature_map[node][:3].clone()
        counter += 1
    features.append(path_features.numpy())
    classification.append(2)

for i, path in enumerate(suboptimal['nn_paths']):
    counter = 0
    path_features = torch.zeros((3,))
    for node in path:
        path_features += my_feature_map[node][:3].clone()
        counter += 1
    if node != 624:
        print('failure')
    features.append(path_features.numpy())
    classification.append(3)

features = np.array(features)
classification = np.array(classification)

# run the PCA analysis
scaling = StandardScaler()
scaling.fit(features)
scaled_data = scaling.transform(features)
principle = PCA(n_components=3)
principle.fit(scaled_data)
output = principle.transform(scaled_data)

fig = plt.figure(figsize=(8, 8))

# choose projection 3d for creating a 3d graph
axis = fig.add_subplot(111, projection='3d')

# x[:,0]is pc1,x[:,1] is pc2 while x[:,2] is pc3
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
# axis.scatter(output[:, 0][classification == 0], output[:, 1][classification == 0], output[:, 2][classification == 0],
#              marker='s', color=colors[0],
#              label='Dijkstra Optimal', s=5)
axis.scatter(output[:, 0][classification == 1], output[:, 1][classification == 1], output[:, 2][classification == 1],
             color='mediumblue',
             label='NN Policy A', s=10, )
axis.set_xlabel("PC1", fontsize=10)
axis.set_ylabel("PC2", fontsize=10)
axis.set_zlabel("PC3", fontsize=10)
plt.legend()

# plt.show()

# fig = plt.figure(figsize=(10, 10))

# choose projection 3d for creating a 3d graph
# axis = fig.add_subplot(111, projection='3d')
# axis.scatter(output[:, 0][classification == 2], output[:, 1][classification == 2], output[:, 2][classification == 2],
#              marker='s', c=colors[2],
#              label='Dijkstra Sub-optimal', s=5)
axis.scatter(output[:, 0][classification == 3], output[:, 1][classification == 3], output[:, 2][classification == 3],
             c='darkorange',
             label='NN policy B', s=5)
axis.set_xlabel("PC1", fontsize=10)
axis.set_ylabel("PC2", fontsize=10)
axis.set_zlabel("PC3", fontsize=10)
plt.legend()

plt.show()
