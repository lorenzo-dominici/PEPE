# This module generates the output elements according to the input data.
import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zipf

class NetworkGenerator: 
    def __init__(self, params):
        self.seed = params.get("seed")
        self.node_number = params.get("node_number")
        self.conn_prob = params.get("conn_prob")
        self.degree_distr = params.get("degree_distr")
        self.if_range = params.get("if_range")
        self.rewiring_prob = params.get("rewiring_prob")
        self.local_clustering_coeff = params.get("local_clustering_coeff")
        self.clusters_number_range = params.get("clusters_number_range")
        self.nodes_range_per_cluster = params.get("nodes_range_per_cluster")
        self.inter_clusters_coeff = params.get("inter_clusters_coeff")
        self.central_nodes_range = params.get("central_nodes_range")
        self.central_nodes_min_degree = params.get("central_nodes_min_degree")
        self.edge_per_new_node = params.get("edge_per_new_node")
        self.buffer_size_range = params.get("buffer_size_range")
        self.central_nodes_buffer_size = params.get("central_nodes_buffer_size")
        self.channel_bandwidth_range = params.get("channel_bandwidth_range")

        self.rng = np.random.default_rng(seed = self.seed)
        random.seed(self.seed)
        self.G = nx.Graph()


    def generate_network(self):
        self._generate_topology()

        self._handle_central_nodes()

        self._assign_attributes()

        self._generate_json()

        # plt.figure(figsize=(6,6))
        # pos = nx.spring_layout(self.G, seed=self.seed)
        # nx.draw(self.G, pos, node_color='skyblue', edge_color='gray', with_labels=True)
        # plt.title(f"Grafo con {self.node_number}")
        # plt.show()

        

    
    def _generate_topology(self):
        # topology generation logic based on parameters
        # first of all, check if the clustering is required; if so, first generate subgraphs and then connect them
        clustering = self.clusters_number_range is not None and self.nodes_range_per_cluster is not None
        if clustering:
            clusters_sizes = self._partition()
            subgraphs = []
            # generate the subgraphs
            for i in range(len(clusters_sizes)):
                size = clusters_sizes[i]
                subgraphs[i] = self._generate_graph(size)
            # concatenate the subgraphs
            self._join_subgraphs(subgraphs)

        else:
            # no clusters, a single subgraph is created
            self.G = self._generate_graph()

    
    def _partition(self):
        min_c, max_c = self.clusters_number_range[0], self.clusters_number_range[1]
        clusters_number = random.randint(min_c, max_c)
        while (clusters_number * self.nodes_range_per_cluster[0] > self.node_number) or (clusters_number * self.nodes_range_per_cluster[1] < self.node_number):
            clusters_number = random.randint(min_c, max_c)

        sizes = [self.nodes_range_per_cluster[0]] * clusters_number
        while sum(sizes) < self.node_number:
            i = random.randint(0, len(sizes) - 1)
            if sizes[i] < self.nodes_range_per_cluster[1]:
                sizes[i] += 1
        return sizes
    
    
    def _generate_graph(self, size: int = None):
        if size == None:
            size = self.node_number
        
        # random graph generation with the Erdős–Rényi algorithm if the conn_prob exists
        if self.conn_prob != None:
            G = nx.gnp_random_graph(size, self.conn_prob)
            while not(nx.is_connected(G)):
                G = nx.gnp_random_graph(size, self.conn_prob)
            return G
        
        # graph generation with the Havel-Hakimi algorithm using a degree distribution
        if self.degree_distr != None:
            degrees = self._get_degrees_from_distr()
            print(degrees)
            print ("\n")
            print (len(degrees))

            min_val = min(degrees)
            max_val = max(degrees)
            bordi_bins = np.arange(min_val - 0.5, max_val + 1.5, 1)
            plt.hist(degrees, bins=bordi_bins, edgecolor='black')
            plt.xticks(np.arange(min_val, max_val + 1, 1))
            plt.show()

            return nx.Graph()
        

    def _get_degrees_from_distr(self):
        type = self.degree_distr.get("type")
        params = self.degree_distr.get("params")
        degrees = []
        match type:
            case "BINOMIAL":
                n = self.degree_distr.get("params")[0]
                if n > self.node_number:
                    n = self.node_number
                p = self.degree_distr.get("params")[1]
                degrees = self.rng.binomial(n, p, self.node_number)
                while sum(degrees) % 2 != 0:
                    degrees = self.rng.binomial(n, p, self.node_number)
            case "UNIFORM":
                min = self.degree_distr.get("params")[0]
                max = self.degree_distr.get("params")[1]
                if max > self.node_number:
                    max = self.node_number
                degrees = self.rng.integers(min, max + 1, self.node_number)
                while sum(degrees) % 2 != 0:
                    degrees = self.rng.integers(min, max + 1, self.node_number)
            case "NORMAL":
                mean = self.degree_distr.get("params")[0]
                stddev = self.degree_distr.get("params")[1]
                if mean > self.node_number:
                    mean = self.node_number
                degrees = self.rng.normal(mean, stddev, self.node_number)
                degrees = np.round(degrees).astype(int)
                while sum(degrees) % 2 != 0:
                    degrees = np.round(degrees).astype(int)
            case "POWERLAW":
                gamma = self.degree_distr.get("params")[0]
                k_min = self.degree_distr.get("params")[1]
                if k_min > self.node_number:
                    k_min = self.node_number
                degrees = zipf.rvs(a = gamma, loc = k_min, random_state = self.rng)
                while sum(degrees) % 2 != 0:
                    degrees = zipf.rvs(a = gamma, loc = k_min, random_state = self.rng)
            case "CUSTOM":
                degrees = self.degree_distr.get("params")
                if len(degrees) < self.node_number:
                    n = self.node_number - len(degrees)
                    to_append = [degrees[-1]] * n
                    degrees.extend(to_append)
                elif len(degrees) > self.node_number:
                    degrees = degrees[:self.node_number]
                while sum(degrees) % 2 != 0:
                    i = self.rng.integers(0, len(degrees) - 1)
                    if degrees[i] < self.node_number:
                        degrees[i] += 1
        return degrees

    
    def _join_subraphs(self, subgraphs):
        pass

    
    def _handle_central_nodes(self):
        # Logic to identify and handle central nodes
        pass

    
    def _assign_attributes(self):
        # Logic to assign attributes like buffer sizes and channel bandwidths
        pass

    
    def _generate_json(self):
        # Logic to convert the network structure into JSON format
        pass


        