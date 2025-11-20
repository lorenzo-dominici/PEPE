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
        self.connected = params.get("connected")

        self.gen_model = params.get("gen_model")

        self.conn_prob = params.get("conn_prob")
        self.degree_distr = params.get("degree_distr")
        self.if_range = params.get("if_range")

        self.mean_degree_range = params.get("mean_degree_range")
        self.rewiring_prob = params.get("rewiring_prob")
        self.delete_rewired = params.get("delete_rewired")

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

        if self.seed == None:
            self.seed = 42 # default seed
        self.rng = np.random.default_rng(seed = self.seed)
        random.seed(self.seed)
        self.G = nx.Graph()


    def generate_network(self):
        self._generate_topology()

        self._handle_central_nodes()

        self._assign_attributes()

        self._generate_json()

    
    def _generate_topology(self):
        # topology generation logic based on parameters
        # first of all, check if the clustering is required; if so, first generate subgraphs and then connect them
        clustering = self.clusters_number_range is not None
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

        self._draw_graph()

    
    def _partition(self):
        min_c, max_c = self.clusters_number_range[0], self.clusters_number_range[1]
        clusters_number = self.rng.integers(min_c, max_c)
        while clusters_number > self.node_number:
                clusters_number = self.rng.integers(min_c, max_c)

        # if self.nodes_range_per_cluster == None, the nodes are equally distributed in each cluster. 
        if self.nodes_range_per_cluster == None:
            nodes_per_cluster = self.node_number // clusters_number
            sizes = [nodes_per_cluster] * clusters_number
            remaining = self.node_number % clusters_number
            for i in range(remaining):
                sizes[i] += 1
                return sizes
            
        # if the min nodes per cluster is too high
        if self.nodes_range_per_cluster[0] * min_c > self.node_number:
            clusters_number = min_c
            nodes_per_cluster = self.node_number // clusters_number
            sizes = [nodes_per_cluster] * clusters_number
            remaining = self.node_number % clusters_number
            for i in range(remaining):
                sizes[i] += 1
                return sizes

        # if the max nodes per cluster is too small
        if self.nodes_range_per_cluster[1] * max_c < self.node_number:
            clusters_number = max_c
            nodes_per_cluster = self.node_number // clusters_number
            sizes = [nodes_per_cluster] * clusters_number
            remaining = self.node_number % clusters_number
            for i in range(remaining):
                sizes[i] += 1
                return sizes
            
        # if there is the nodes range per cluster
        while (clusters_number * self.nodes_range_per_cluster[0] > self.node_number or clusters_number * self.nodes_range_per_cluster[1] < self.node_number):
            clusters_number = self.rng.integers(min_c, max_c)

        sizes = [self.nodes_range_per_cluster[0]] * clusters_number
        while sum(sizes) < self.node_number:
            i = self.rng.integers(0, len(sizes) - 1)
            if sizes[i] < self.nodes_range_per_cluster[1]:
                sizes[i] += 1
        return sizes
    
    
    def _generate_graph(self, size: int = None):
        if size == None:
            size = self.node_number
        
        if self.connected == None:
            self.connected = False

        model = self.gen_model
        if model == None:
            model = "RANDOM"

        G = nx.Graph()

        match model:
            case "RANDOM":
                G = self._generate_random_graph(size)
            case "SMART-WORLD":
                G = self._generate_smart_world_graph(size)
            case "SCALE-FREE":
                G = self._generate_scale_free_graph(size)

        # if the if_range parameter exists, use it to bound the degrees of the nodes
        if self.if_range != None:
            self._bound_nodes_degrees(G)
        
        return G
    

    def _generate_random_graph(self, size):
        # random graph generation with the Erdős–Rényi algorithm if the conn_prob exists
        if self.conn_prob != None:
            G = nx.gnp_random_graph(size, self.conn_prob)
            if self.connected:
                tries = 0
                while not(nx.is_connected(G)) and tries <= 1000:
                    G = nx.gnp_random_graph(size, self.conn_prob)
                    tries += 1
                if not(nx.is_connected(G)):
                    raise nx.NetworkXUnfeasible("Can't generate a connected graph")
            return G
        
        # graph generation with the Havel-Hakimi algorithm using a degree distribution
        if self.degree_distr != None:
            degrees = self._get_degrees_from_distr()
            print(degrees)
            print ("\n")
            print (len(degrees))
            G = nx.havel_hakimi_graph(degrees)
            if self.connected:
                tries = 0
                while not(nx.is_connected(G)) and tries <=1000:
                    nx.double_edge_swap(G, nswap=5*len(G.edges()), max_tries=100*len(G.edges()))
                    tries += 1
                if not(nx.is_connected(G)):
                    raise nx.NetworkXUnfeasible("Can't generate a connected graph")
            return G
        
        if self.if_range != None:
            degrees = self._get_degrees_from_distr(type = "UNIFORM", params = self.if_range)
            G = nx.havel_hakimi_graph(degrees)
            if self.connected:
                tries = 0
                while not(nx.is_connected(G)) and tries <=1000:
                    nx.double_edge_swap(G, nswap=5*len(G.edges()), max_tries=100*len(G.edges()))
                    tries += 1
                if not(nx.is_connected(G)):
                    raise nx.NetworkXUnfeasible("Can't generate a connected graph")
            return G



            # min_val = min(degrees)
            # max_val = max(degrees)
            # bordi_bins = np.arange(min_val - 0.5, max_val + 1.5, 1)
            # plt.hist(degrees, bins=bordi_bins, edgecolor='black')
            # plt.xticks(np.arange(min_val, max_val + 1, 1))
            # plt.show()

            return nx.Graph()
        

    def _generate_smart_world_graph(self, size):
        pass


    def _generate_scale_free_graph(self, size):
        pass
    
    
    def _get_degrees_from_distr(self, type = None, params = None):
        if type == None:
            type = self.degree_distr.get("type")
        if params == None:
            params = self.degree_distr.get("params")
        degrees = []
        # generate the sequence of the degrees starting from the parameters of the degree distribution.
        # note that the sum of the degrees must be even, because each edge connect 2 nodes
        match type:
            case "BINOMIAL":
                n = params[0]
                if n > self.node_number:
                    n = self.node_number
                p = params[1]
                degrees = self.rng.binomial(n, p, self.node_number)
                while sum(degrees) % 2 != 0:
                    degrees = self.rng.binomial(n, p, self.node_number)
            case "UNIFORM":
                min = params[0]
                max = params[1]
                if max > self.node_number:
                    max = self.node_number
                degrees = self.rng.integers(min, max + 1, self.node_number)
                while sum(degrees) % 2 != 0:
                    degrees = self.rng.integers(min, max + 1, self.node_number)
            case "NORMAL":
                mean = params[0]
                stddev = params[1]
                if mean > self.node_number:
                    mean = self.node_number
                degrees = self.rng.normal(mean, stddev, self.node_number)
                degrees = np.round(degrees).astype(int)
                while sum(degrees) % 2 != 0:
                    degrees = self.rng.normal(mean, stddev, self.node_number)
                    degrees = np.round(degrees).astype(int)
            case "POWERLAW":
                gamma = params[0]
                k_min = params[1]
                if k_min > self.node_number:
                    k_min = self.node_number
                degrees = zipf.rvs(a = gamma, loc = k_min, size = self.node_number, random_state = self.rng)
                while sum(degrees) % 2 != 0:
                    degrees = zipf.rvs(a = gamma, loc = k_min, size = self.node_number, random_state = self.rng)
            case "CUSTOM":
                degrees = eval(params)
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
        
        # the degree of a node must be 0 <= deg <= node_number
        for i in range(len(degrees)):
            if degrees[i] < 0:
                degrees[i] = 0
            elif degrees[i] > self.node_number:
                degrees[i] = self.node_number

        return degrees

    
    def _bound_nodes_degrees(self, G):
        min_if, max_if = self.if_range[0], self.if_range[1]
        nodes = list(nx.nodes(G))
        modified = True
        while modified:
            modified = False
            for node_a in nodes:
                if nx.degree(G, node_a) > max_if:
                    neighbors = list(nx.neighbors(G, node_a))
                    self.rng.shuffle(neighbors)
                    for node_b in neighbors:
                        if nx.degree(G, node_b) > min_if and nx.degree(G, node_a) > max_if:
                            if self.connected == False:
                                G.remove_edge(node_a, node_b)
                                modified = True
                                break
                            else:
                                bridges = list(nx.bridges(G))
                                if (node_a, node_b) not in bridges and (node_b, node_a) not in bridges:
                                    G.remove_edge(node_a, node_b)  
                                    modified = True
                                    break

        for node_a in nodes:
            if nx.degree(G, node_a) > max_if:
                raise nx.NetworkXUnfeasible("can't bound the number of interfaces for each node")      

        modified = True
        while modified:
            modified = False
            for node_a in nodes:
                if nx.degree(G, node_a) < min_if:
                    nodes_to_check = nodes.copy()
                    nodes_to_check.remove(node_a)
                    self.rng.shuffle(nodes_to_check)
                    for node_b in nodes_to_check:
                        if nx.degree(G, node_b) < max_if and nx.degree(G, node_a) < max_if and (node_a, node_b) not in nx.edges(G, node_a) and (node_b, node_a) not in nx.edges(G, node_b):
                            G.add_edge(node_a, node_b)
                            modified = True
                            break

        for node_a in nodes:
            if nx.degree(G, node_a) < min_if:             
                raise nx.NetworkXUnfeasible("can't bound the number of interfaces for each node")
        
    
    def _join_subraphs(self, subgraphs):
        pass

    
    def _handle_central_nodes(self):
        # Logic to identify and handle central nodes
        pass


    def _draw_graph(self):
        pos = nx.spring_layout(self.G)
        plt.figure(figsize=(10, 10))
        nx.draw(self.G, pos, with_labels=True, node_size=200, node_color='skyblue', font_size=7, font_weight='bold')
        plt.title("Grafo")
        plt.axis('off')
        plt.show()

    
    def _assign_attributes(self):
        # Logic to assign attributes like buffer sizes and channel bandwidths
        pass

    
    def _generate_json(self):
        # Logic to convert the network structure into JSON format
        pass


        