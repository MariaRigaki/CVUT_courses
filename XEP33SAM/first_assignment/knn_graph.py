import numpy as np
from pyflann import *
import networkx as nx
import matplotlib.pyplot as plt
import pydot as dot

# Read the data and shape them accordingly
vec = np.fromfile('data/oxford_5k/imagedesc.dat', dtype=np.float32)
points = vec.reshape(-1, 128)

G = nx.DiGraph()
flann = FLANN()

""" Autotune results
{'algorithm': 'kmeans', 'checks': 224, 'eps': 0.0, 'sorted': 1, 'max_neighbors': -1,
'cores': 0, 'trees': 1, 'leaf_max_size': 4, 'branching': 64, 'iterations': 5,
'centers_init': 'random', 'cb_index': 0.0, 'target_precision': 0.8999999761581421,
'build_weight': 0.009999999776482582, 'memory_weight': 0.0, 'sample_fraction': 0.10000000149011612,
'table_number_': 12, 'key_size_': 20, 'multi_probe_level_': 2, 'log_level': 'info',
'random_seed': 86244857, 'speedup': 2.0744643211364746}

"""
params = {'algorithm': 'kmeans',
          'branching': 64,
          'target_precision': 0.90,
          'centers_init': 'gonzales'}

flann.build_index(points, **params)
result, dists = flann.nn_index(points, 5)
print("Using params for k-NN:", params)
# Using params for k-NN: {'algorithm': 'kmeans', 'branching': 64, 'target_precision': 0.9, 'centers_init': 'gonzales'}

for i in range(points.shape[0]):
    for j in range(len(result[i])):
        G.add_edge(i, result[i, j], weight=dists[i, j])

print(nx.info(G))

options = {
    'node_color': 'C0',
    'node_size': 100,
    'arrows': True
}

# Name: 
# Type: DiGraph
# Number of nodes: 5063
# Number of edges: 25315
# Average in degree:   5.0000
# Average out degree:   5.0000

    
nx.draw(G)
plt.show()
# nx.nx_pydot.write_dot(G, 'oxford.dot')
