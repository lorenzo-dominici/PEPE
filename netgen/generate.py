# This module generates the output elements according to the input data.
import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zipf

class NetworkGenerator: 
    def __init__(self, params):
        self.protocol = params.get("protocol")
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

        self.initial_degree_range = params.get("initial_degree_range")
        self.new_edges_prob = params.get("new_edges_prob")

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
        self.path_perc = params.get("path_perc")

        self.link_prob_failure_to_working = params.get("link_prob_failure_to_working")
        self.link_prob_retry = params.get("link_prob_retry")
        self.link_prob_sending = params.get("link_prob_sending")
        self.channel_prob_working_to_error = params.get("channel_prob_working_to_error")
        self.channel_prob_error_to_working = params.get("channel_prob_error_to_working")
        self.channel_prob_failure_to_working = params.get("channel_prob_failure_to_working")
        self.if_prob_off_to_working = params.get("if_prob_off_to_working")
        self.if_prob_off_to_error = params.get("if_prob_off_to_error")
        self.if_prob_off_to_failure = params.get("if_prob_off_to_failure")
        self.if_prob_working_to_error = params.get("if_prob_working_to_error")  
        self.ls_size_epoch_range = params.get("ls_size_epoch_range")
        self.ls_size_ratchet_range = params.get("ls_size_ratchet_range")
        self.ls_prob_session_reset = params.get("ls_prob_session_reset")
        self.ls_prob_ratchet_reset = params.get("ls_prob_ratchet_reset")
        self.ls_prob_none = params.get("ls_prob_none")
        self.ls_prob_compromised = params.get("ls_prob_compromised")
        self.sp_prob_run = params.get("sp_prob_run")


        if self.seed == None:
            self.seed = 42 # default seed
        self.rng = np.random.default_rng(seed = self.seed)
        
        if self.connected == None:
            self.connected = False

        if self.gen_model == None:
            self.gen_model = "RANDOM"


        self.G = nx.Graph()
        self.attributes = {}


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
                subgraphs.append(self._generate_graph(size))
                color = tuple(self.rng.random(3))
                nx.set_node_attributes(subgraphs[i], color, 'color')
                nx.set_edge_attributes(subgraphs[i], color, 'color')
            # concatenate the subgraphs
            self._join_subgraphs(subgraphs)

        else:
            # no clusters, a single subgraph is created
            self.G = self._generate_graph()
            nx.set_node_attributes(self.G, "orange", 'color')
            nx.set_edge_attributes(self.G, "black", 'color')


        self._draw_graph()

    
    def _partition(self):
        # return the partition of the nodes in each cluster

        min_c, max_c = self.clusters_number_range[0], self.clusters_number_range[1]
        if min_c == max_c:
            clusters_number = min_c
        else:
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
        else:  
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

        G = nx.Graph()

        match self.gen_model:
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
            G = nx.gnp_random_graph(size, self.conn_prob, seed = self.rng)
            if self.connected:
                tries = 0
                while not(nx.is_connected(G)) and tries <= 1000:
                    G = nx.gnp_random_graph(size, self.conn_prob, seed = self.rng)
                    tries += 1
                if not(nx.is_connected(G)):
                    raise nx.NetworkXUnfeasible("can't generate a connected graph with theese parameters")
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
                    raise nx.NetworkXUnfeasible("Ccan't generate a connected graph with theese parameters")
            return G
        
        # graph generation in exixts the bound for the interfces. Uses the uniform distribution 
        if self.if_range != None:
            degrees = self._get_degrees_from_distr(type = "UNIFORM", params = self.if_range)
            G = nx.havel_hakimi_graph(degrees)
            if self.connected:
                tries = 0
                while not(nx.is_connected(G)) and tries <=1000:
                    nx.double_edge_swap(G, nswap=5*len(G.edges()), max_tries=100*len(G.edges()))
                    tries += 1
                if not(nx.is_connected(G)):
                    raise nx.NetworkXUnfeasible("can't generate a connected graph with theese parameters")
            return G


    def _generate_smart_world_graph(self, size):
        # Graph generation using the Watts-Strogatz algorithm in 2 variants: deleting or not the rewired edge
        if self.mean_degree_range is None:
            mean = size // 2
        elif self.mean_degree_range[0] >= size:
            mean = size - 1
        elif self.mean_degree_range[0] == self.mean_degree_range[1]:
            mean = self.mean_degree_range[0]
        else:
            mean = self.rng.integers(self.mean_degree_range[0], self.mean_degree_range[1])
            while mean >= size:
                mean = self.rng.integers(self.mean_degree_range[0], self.mean_degree_range[1])
        
        if self.rewiring_prob is None:
            p = 0.5
        else:
            p = self.rewiring_prob

        G = nx.Graph()
        # if delete_rewired is true, use the Watts-Strogatz algorithm
        if self.delete_rewired == True:
            if self.connected:
                try:
                    G = nx.connected_watts_strogatz_graph(size, mean, p, tries = 1000, seed = self.rng)
                except Exception:
                    raise nx.NetworkXUnfeasible("Can't generate a connected graph with theese parameters")
            else:
                G = nx.watts_strogatz_graph(size, mean, p, seed = self.rng)
        # else use the Newmann-Watts_Strogatz algorithm that doesn't delete the rewired edges
        else:
            G = nx.newman_watts_strogatz_graph(size, mean, p, seed = self.rng)
        return G
                    

    def _generate_scale_free_graph(self, size):
        # generation of scale free graph with the extended Albert_Barabasi algorithm

        n = size
        if self.initial_degree_range == None:
            m = n // 2
        else:
            min_d, max_d = self.initial_degree_range[0], self.initial_degree_range[1]
            if min_d >= n:
                m = n - 1
            elif min_d == max_d:
                m = min_d
            else:
                m = self.rng.integers(min_d, max_d)
                while m >= n:
                    m = self.rng.integers(min_d, max_d)
        if self.new_edges_prob == None:
            p = 0
        else:
            p = self.new_edges_prob
        if self.rewiring_prob == None:
            q = 0
        else:
            q = self.rewiring_prob

        G = nx.Graph()
        G = nx.extended_barabasi_albert_graph(n, m, p, q, seed = self.rng)
        if self.connected:
            tries = 0
            while not(nx.is_connected(G)) and tries < 1000:
                G = nx.extended_barabasi_albert_graph(n, m, p, q, seed = self.rng)
            if not(nx.is_connected(G)):
                raise nx.NetworkXUnfeasible("can't generate a connected graph with theese parameters")
            
        return G
    
    
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
        
    
    def _join_subgraphs(self, subgraphs):
        n_clusters = len(subgraphs)
        p = self.inter_clusters_coeff
        if p == None:
            p = 0.01
        self.G = nx.disjoint_union_all(subgraphs)
        all_nodes = list(self.G.nodes())
        cluster_boundaries = []
        start = 0
        for g in subgraphs:
            cluster_boundaries.append(start)
            start += g.number_of_nodes()
        cluster_boundaries.append(start)

        # cluster_boundaries = [i * len(subgraphs[i]) for i in range(n_clusters)] + [len(all_nodes)]
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                nodes_i = np.arange(cluster_boundaries[i], cluster_boundaries[i+1])
                nodes_j = np.arange(cluster_boundaries[j], cluster_boundaries[j+1])
                u_mesh, v_mesh = np.meshgrid(nodes_i, nodes_j, indexing='ij')
                u_flat = u_mesh.ravel()
                v_flat = v_mesh.ravel()
                mask = self.rng.random(len(u_flat)) < p
                for u, v in zip(u_flat[mask], v_flat[mask]):
                    self.G.add_edge(int(u), int(v), color = "black")

    
    def _handle_central_nodes(self):
        # Logic to identify and handle central nodes
        pass


    def _draw_graph(self):
        pos = nx.spring_layout(self.G, seed = self.seed)
        centrality = nx.degree_centrality(self.G)
        # nodes_sizes = list(v * ((1 / self.node_number) * (200000000 * np.var(list(centrality.values())))) + 50 for v in centrality.values())
        scale_factor = 10000
        nodes_sizes = [ (v ** 2) * scale_factor + 50 for v in centrality.values() ]

        node_colors = [data['color'] for _, data in self.G.nodes(data=True)]
        edge_colors = [data['color'] for _, _, data in self.G.edges(data=True)]

        plt.figure(figsize=(10, 10))
        nx.draw(self.G, pos, with_labels=True, node_size=nodes_sizes, node_color=node_colors, edge_color = edge_colors, font_size=7, font_weight='bold')
        plt.title("Grafo")
        plt.axis('off')
        plt.show()

    
    def _assign_attributes(self):
        # Logic to assign attributes like buffer sizes and channel bandwidths and paths
        self._generate_paths()

        buffer_size = self.rng.integers(self.buffer_size_range[0], self.buffer_size_range[1])
        self.attributes["buffer_size"] = buffer_size

        channel_bandwidth = self.rng.integers(self.channel_bandwidth_range[0], self.channel_bandwidth_range[1])
        self.attributes["channel_bandwidth"] = channel_bandwidth

        for attr in self.attributes:
            print(f"{attr}: {self.attributes[attr]}")

        

    def _generate_paths(self):
        nodes = list(self.G.nodes())
        #if the protocol is hpke or double ratchet, we need only one-to-one paths
        if self.protocol in {"HPKE", "DOUBLE-RATCHET"}:
            paths = []
            for i in range(len(nodes)):
                for j in range(len(nodes)):
                    if i != j:
                        a = nodes[i]
                        b = nodes[j]
                        if self.rng.random() < self.path_perc:
                            path = nx.shortest_path(self.G, a, b)
                            paths.append(path)
            self.attributes["paths"] = paths
            

        # for sender key, we need one-to-many paths and all the one-to-one return paths to the sender
        else:
            # random generation of one-to-many paths
            one_to_many_paths = {}
            for i in range(len(nodes)):
                path = []
                group = []
                a = nodes[i]
                group.append(a)
                for j in range(len(nodes)):
                    if i != j and self.rng.random() < self.path_perc:
                        b = nodes[j]
                        path_to_add = nx.shortest_path(self.G, a, b)
                        path.append(path_to_add)
                        group.append(b)
                one_to_many_paths[tuple(group)] = path
            self.attributes["one_to_many_paths"] = one_to_many_paths
            # for group, paths in one_to_many_paths.items():
            #     print(f"Group: {group}")
            #     for path in paths:
            #         print(f"  Path: {path}")
            # print("\n")

            # generation of all the one-to-one return paths to the sender
            one_to_one_paths = {}
            for group, paths in one_to_many_paths.items():
                sender = group[0]
                paths_to_add = []
                for path in paths:
                    receiver = path[-1]
                    return_path = nx.shortest_path(self.G, receiver, sender)
                    paths_to_add.append(return_path)
                one_to_one_paths[tuple(group)] = paths_to_add
            self.attributes["one_to_one_return_paths"] = one_to_one_paths
            # for group, paths in one_to_one_paths.items():
            #     print(f"Group: {group}")
            #     for path in paths:
            #         print(f"  Return Path: {path}")
            # print("\n")     


    def _generate_json(self):
        # Logic to convert the network structure into JSON format
        pass