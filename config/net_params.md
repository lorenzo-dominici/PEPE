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


### Attachment metod

 - `gen_model`: RANDOM | SMART-WORLD | SCALE-FREE


### Connectivity probability, degree distribution and number of interfaces for random graphs

- `connected`: true | false
- if true, the generator try to construct a connected graph in max 1000 iterations.

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

### SMART-WORLD parameters

- `mean_degree_range`: [min, max]
- the mean degree for each node

- `rewiring_prob`: [0.0-1.0]
- the probability of rewiring for an edge (u, v): choose a node w randomly and add the edge (u, w)
The highest is this probability, the lower is the local clustering coefficient

- `delete_rewired`: true, false
- for the edge (u, v) with the rewired edge (u, w), choose if (u, v) have to be deleted or not

### SCALE-FREE PARAMETERS

- `initial_degree_range`: [min, max]
- the initial min degree for the new nodes

- `new_edges_prob`: [0.0-1.0]
- the probability to add new edges after the generation between existents nodes


### Clustering 

- `clusters_number_range`: [min, max]
- the number of clusters

- `nodes_range_per_cluster`: [min, max]
- the range of the number of nodes in a cluster

- `inter_clusters_coeff`: [0.0-1.0]
- the probability of the inter-clusters connection


### Nodes Buffer size

- `buffer_size_range`: [min, max]
- the dimension of the node message queue

- `central_nodes_buffer_size`: [min, max]
- Optional, the dimension of the node message queue for the central nodes

### Channel bandwidth

- `channel_bandwidth_range`: [min, max]
- the number of links that can use the channel


