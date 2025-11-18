# Network Parameters

This file contains the list of the network parameters used by the network generator (netgen) for the creation of the network json. 


## Structure

The parameters of the network must be included in a json file with this structure: 
    ```
    {
        "param_name": "param value",
        ... 
    }
    ```

## Parameters

### Network type

- `net_type`: STAR | RING | CHAIN | TREE | MESH | FULLY_CONNECTED | REGULAR_GRAPH | CUSTOM
- the type of the network

### Seed
- `seed`: integer
- the seed for the random generator

### Nodes number

- `node_range`: [min, max], positive integers value
- the range of the number of nodes; min=max for a fixed value


### Connectivity probability, degree distribution and number of interfaces for random graphs

- use one of the 3 possibilities. `if_range` che be used to bound the others

- `conn_prob`: [0.0-1.0]
- the probability for a couple of node (i, j) to be connected. If i and j are connected, the interfaces and the channels are created

- `degree_distr`: 
    {
        "type": BINOMIAL | UNIFORM | NORMAL | POWERLAW | CUSTOM
        "params": [...] 
    }
- the distribution of the grade of the nodes. If a node has grade k, k interfaces and channels are created. 

- `if_range`: [min, max], positive integers values
- the range of the number of interfaces for each node, that represent the nodes degree. The same as uniform degree distribution

### Clustering 

- `rewiring_prob`: [0.0-1.0]
- the probability of rewiring for an edge (u, v): delete the edge, choose a node w randomly and add the edge (u, w)
The highest is this probability, the lower is the local clustering coefficient

- `local_clustering_coeff`: [0.0-1.0]
- the coefficient for the local clustering

- `clusters_number_range`: [min, max]
- the number of clusters

- `nodes_range_per_cluster`: [min, max]
- the range of the number of nodes in a cluster

- `inter_clusters_coeff`: [0.0-1.0]
- the probability of the inter-clusters connection


### Nodes centrality

- `edge_per_new_node`: [min, max]
- the number of new edges added for a new node. A small value (1, 2) implies a small number of hubs with a high edges, a high value (20) implies a network densed and more balanced

- `central_nodes_min_degree`: min
- the minimum degree for a node to be considered central node. This value must be lower than number of nodes.

- `central_nodes_range`: [min, max]
- the number of central nodes. This parameter can modify the degree distribution or the connectivity probability to set the expected number of central nodes.


### Nodes Buffer size

- `buffer_size_range`: [min, max]
- the dimension of the node message queue

- `central_nodes_buffer_size`: [min, max]
- Optional, the dimension of the node message queue for the central nodes

### Channel bandwidth

- `channel_bandwidth_range`: [min, max]
- the number of links that can use the channel


