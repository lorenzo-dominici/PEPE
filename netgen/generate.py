# This module generates the output elements according to the input data.
from networkx.classes.function import neighbors
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zipf
import json


class Session:
    def __init__(self):
        self.id = None
        self.protocol = None
        self.nodes = []
        self.paths = []
        self.subsessions = []

    def set_id(self, id):
        self.id = id

    def set_protocol(self, protocol):
        self.protocol = protocol
    
    def set_nodes(self, nodes):
        self.nodes = nodes

    def add_path(self, path):
        self.paths.append(path)

    def add_subsession(self, subsession):
        self.subsessions.append(subsession)

    def get_id(self):
        return self.id

    def get_nodes(self):
        return self.nodes

    def get_paths(self):
        return self.paths

    def get_subsessions(self):
        return self.subsessions.copy()



class NetworkGenerator: 
    def __init__(self, params):
        self.params = params
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

        self._generate_paths()

        self._assign_attributes()

        return self._generate_json()

    
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
        # Logic to assign attributes like buffer sizes and channel bandwidths and all the probabilities
        
        attributes_to_add = [
            "mls_sessions_range", "filename", "support_protocol",
            "buffer_size_range", "node_prob_off_to_on", "node_prob_on_to_off",
            "channel_bandwidth_range", 
            "channel_prob_working_to_error", "channel_prob_error_to_working", "channel_prob_failure_to_working",
            "if_prob_off_to_working", "if_prob_off_to_error", "if_prob_off_to_failure",
            "if_prob_working_to_error", "if_prob_error_to_working", "if_prob_failure_to_working",
            "link_prob_working_to_error", "link_prob_error_to_working", "link_prob_failure_to_working",
            "link_prob_retry", "link_prob_sending",
            "ls_prob_session_reset", "ls_prob_ratchet_reset", "ls_prob_none", "ls_prob_compromised",
            "sp_prob_run"
        ]
        
        for attr_name in attributes_to_add:
            self.attributes[attr_name] = self.params[attr_name]

        
    def _generate_paths(self):
        nodes = list(self.G.nodes())
        #if the protocol is hpke or double ratchet, we need only one-to-one paths and return paths
        match self.protocol:
            case "HPKE" | "DOUBLE-RATCHET":
                paths = []
                return_paths = []
                sessions = []
                int id = 0
                for i in range(len(nodes)):
                    for j in range(len(nodes)):
                        if i != j:
                            a = nodes[i]
                            b = nodes[j]
                            if self.rng.random() < self.path_perc:
                                path = nx.DiGraph()
                                nx.add_path(path, nx.shortest_path(self.G, a, b))
                                # select the size of epoch and ratchet for sender and receiver
                                if self.params.get("ls_size_epoch_range")[0] == self.params.get("ls_size_epoch_range")[1]:
                                    epoch_size = self.params.get("ls_size_epoch_range")[0]
                                else:
                                    epoch_size = self.rng.integers(self.params.get("ls_size_epoch_range")[0], self.params.get("ls_size_epoch_range")[1] + 1)
                                if self.params.get("ls_size_ratchet_range")[0] == self.params.get("ls_size_ratchet_range")[1]:
                                    ratchet_size = self.params.get("ls_size_ratchet_range")[0]
                                else:
                                    ratchet_size = self.rng.integers(self.params.get("ls_size_ratchet_range")[0], self.params.get("ls_size_ratchet_range")[1] + 1)
                                
                                path.nodes[a]['epoch_size'] = epoch_size
                                path.nodes[a]['ratchet_size'] = ratchet_size
                                path.nodes[b]['epoch_size'] = epoch_size
                                path.nodes[b]['ratchet_size'] = ratchet_size

                                paths.append(path)

                                # generate the return path
                                return_path = nx.DiGraph()
                                nx.add_path(return_path, nx.shortest_path(self.G, b, a))

                                if self.params.get("ls_size_epoch_range")[0] == self.params.get("ls_size_epoch_range")[1]:
                                    epoch_size = self.params.get("ls_size_epoch_range")[0]
                                else:
                                    epoch_size = self.rng.integers(self.params.get("ls_size_epoch_range")[0], self.params.get("ls_size_epoch_range")[1] + 1)
                                if self.params.get("ls_size_ratchet_range")[0] == self.params.get("ls_size_ratchet_range")[1]:
                                    ratchet_size = self.params.get("ls_size_ratchet_range")[0]
                                else:
                                    ratchet_size = self.rng.integers(self.params.get("ls_size_ratchet_range")[0], self.params.get("ls_size_ratchet_range")[1] + 1)

                                return_path.nodes[b]['epoch_size'] = epoch_size
                                return_path.nodes[b]['ratchet_size'] = ratchet_size
                                return_path.nodes[a]['epoch_size'] = epoch_size
                                return_path.nodes[a]['ratchet_size'] = ratchet_size

                                return_paths.append(return_path)

                                session = Session()
                                session.set_id(id)
                                session.set_protocol(self.protocol)
                                session.set_nodes([a, b])
                                session.add_path(path)
                                session.add_path(return_path)
                                sessions.append(session)
                                id += 1

                self.attributes["paths"] = paths
                self.attributes["return_paths"] = return_paths
                self.attributes["sessions"] = sessions

            # for sender key, we need one-to-many paths and all the one-to-one paths sender_receiver and receiver_sender
            case "SENDER-KEY":
                # first, select senders
                senders = []
                for node in nodes:
                    if self.rng.random() < self.path_perc:
                        senders.append(node)
                # now, for each sender, select receivers and generate paths
                one_to_many_paths = []
                sessions = []
                for sender in senders:
                    if self.params.get("ls_size_epoch_range")[0] == self.params.get("ls_size_epoch_range")[1]:
                        epoch_size = self.params.get("ls_size_epoch_range")[0]
                    else:
                        epoch_size = self.rng.integers(self.params.get("ls_size_epoch_range")[0], self.params.get("ls_size_epoch_range")[1] + 1)

                    if self.params.get("ls_size_ratchet_range")[0] == self.params.get("ls_size_ratchet_range")[1]:  
                        ratchet_size = self.params.get("ls_size_ratchet_range")[0]
                    else:
                        ratchet_size = self.rng.integers(self.params.get("ls_size_ratchet_range")[0], self.params.get("ls_size_ratchet_range")[1] + 1)

                    destinations = []
                    for node in nodes:
                        if node != sender and self.rng.random() < self.path_perc:
                            destinations.append(node)
                    tree = nx.DiGraph()
                    tree.add_node(sender)
                    tree.nodes[sender]['epoch_size'] = epoch_size
                    tree.nodes[sender]['ratchet_size'] = ratchet_size
                    # for each destination, generate the shortest path from sender to destination and add it to the tree
                    for dest in destinations:
                        path = nx.shortest_path(self.G, sender, dest)
                        nx.add_path(tree, path)
                        # mark the destination node as receiver
                        tree.nodes[dest]['is_receiver'] = True
                        tree.nodes[dest]['epoch_size'] = epoch_size
                        tree.nodes[dest]['ratchet_size'] = ratchet_size
                    # for each node, set the counter of next nodes
                    for node in tree.nodes():
                        count = len(list(tree.successors(node)))
                        tree.nodes[node]['counter'] = count
                    one_to_many_paths.append(tree)

                self.attributes["one_to_many_paths"] = one_to_many_paths

                # generate all the one-to-one paths from sender to each receiver
                one_to_one_paths = []
                one_to_one_return_paths = []
                id = 0
                
                for tree in one_to_many_paths:
                    session = Session()
                    subsessions = []
                    sender = list(tree.nodes())[0]
                    receivers = [n for n, attr in tree.nodes(data=True) if attr.get('is_receiver')]
                    for receiver in receivers:
                        path = nx.DiGraph()
                        nx.add_path(path, nx.shortest_path(self.G, sender, receiver))

                        if self.params.get("ls_size_epoch_range")[0] == self.params.get("ls_size_epoch_range")[1]:
                            epoch_size = self.params.get("ls_size_epoch_range")[0]
                        else:
                            epoch_size = self.rng.integers(self.params.get("ls_size_epoch_range")[0], self.params.get("ls_size_epoch_range")[1] + 1)
                        if self.params.get("ls_size_ratchet_range")[0] == self.params.get("ls_size_ratchet_range")[1]:
                            ratchet_size = self.params.get("ls_size_ratchet_range")[0]
                        else:
                            ratchet_size = self.rng.integers(self.params.get("ls_size_ratchet_range")[0], self.params.get("ls_size_ratchet_range")[1] + 1)
                        
                        path.nodes[sender]['epoch_size'] = epoch_size
                        path.nodes[sender]['ratchet_size'] = ratchet_size
                        path.nodes[receiver]['epoch_size'] = epoch_size
                        path.nodes[receiver]['ratchet_size'] = ratchet_size

                        one_to_one_paths.append(path)

                        # one-to-one return path
                        return_path = nx.DiGraph()
                        nx.add_path(return_path, nx.shortest_path(self.G, receiver, sender))

                        if self.params.get("ls_size_epoch_range")[0] == self.params.get("ls_size_epoch_range")[1]:
                            epoch_size = self.params.get("ls_size_epoch_range")[0]
                        else:
                            epoch_size = self.rng.integers(self.params.get("ls_size_epoch_range")[0], self.params.get("ls_size_epoch_range")[1] + 1)
                        if self.params.get("ls_size_ratchet_range")[0] == self.params.get("ls_size_ratchet_range")[1]:
                            ratchet_size = self.params.get("ls_size_ratchet_range")[0]
                        else:
                            ratchet_size = self.rng.integers(self.params.get("ls_size_ratchet_range")[0], self.params.get("ls_size_ratchet_range")[1] + 1)
                        
                        return_path.nodes[sender]['epoch_size'] = epoch_size
                        return_path.nodes[sender]['ratchet_size'] = ratchet_size
                        return_path.nodes[receiver]['epoch_size'] = epoch_size
                        return_path.nodes[receiver]['ratchet_size'] = ratchet_size

                        one_to_one_return_paths.append(return_path)

                        subsession = Session()
                        subsession.set_id(id)
                        subsession.set_protocol(self.attributes["support_protocol"])
                        subsession.set_nodes([sender, receiver])
                        subsession.add_path(path)
                        subsession.add_path(return_path)
                        subsessions.append(subsession)
                        id += 1
                    # now add the subsessions to the main session. Select the session of the sender
                    session.set_id(id)
                    session.set_protocol(self.protocol)
                    session.set_nodes([sender] + receivers)
                    session.add_path(tree)
                    for subsession in subsessions:
                        session.add_subsession(subsession)
                    sessions.append(session)
                    id += 1
                
                self.attributes["one_to_one_paths"] = one_to_one_paths
                self.attributes["one_to_one_return_paths"] = one_to_one_return_paths
                self.attributes["sessions"] = sessions
            
            case "MLS":
                # for MLS, generate random sessions and one_to-many paths in each session with the one to one return paths
                # partition the nodes into sessions
                nodes = nodes.copy()
                self.rng.shuffle(nodes)
                if self.params.get("mls_sessions_range") is None:
                    sessions_number = 1
                if self.params.get("mls_sessions_range")[0] == self.params.get("mls_sessions_range")[1]:
                    sessions_number = self.params.get("mls_sessions_range")[0]
                    if sessions_number > len(nodes):
                        sessions_number = 1
                else:
                    sessions_number = self.rng.integers(self.params.get("mls_sessions_range")[0], self.params.get("mls_sessions_range")[1] + 1)
                    while sessions_number > len(nodes):
                        sessions_number = self.rng.integers(self.params.get("mls_sessions_range")[0], self.params.get("mls_sessions_range")[1] + 1)

                cut_points = sorted(self.rng.choice(range(1, len(nodes)), sessions_number - 1, replace = False))
                cut_points = [0] + cut_points + [len(nodes)]
                groups = []
                for i in range(len(cut_points) - 1):
                    start = cut_points[i]
                    end = cut_points[i + 1]
                    group_nodes = nodes[start:end]
                    groups.append(group_nodes)

                one_to_many_paths = []
                one_to_one_return_paths = []
                sessions = []
                id = 0
                # for each group, generate all the one-to-many paths
                for group in groups:
                    for sender in group:
                        if self.params.get("ls_size_epoch_range")[0] == self.params.get("ls_size_epoch_range")[1]:
                            epoch_size = self.params.get("ls_size_epoch_range")[0]
                        else:
                            epoch_size = self.rng.integers(self.params.get("ls_size_epoch_range")[0], self.params.get("ls_size_epoch_range")[1] + 1)

                        if self.params.get("ls_size_ratchet_range")[0] == self.params.get("ls_size_ratchet_range")[1]:  
                            ratchet_size = self.params.get("ls_size_ratchet_range")[0]
                        else:
                            ratchet_size = self.rng.integers(self.params.get("ls_size_ratchet_range")[0], self.params.get("ls_size_ratchet_range")[1] + 1)

                        destinations = [node for node in group if node != sender]
                        tree = nx.DiGraph()
                        tree.add_node(sender)
                        tree.nodes[sender]['epoch_size'] = epoch_size
                        tree.nodes[sender]['ratchet_size'] = ratchet_size
                        # for each destination, generate the shortest path from sender to destination and add it to the tree
                        for dest in destinations:
                            path = nx.shortest_path(self.G, sender, dest)
                            nx.add_path(tree, path)
                            # mark the destination node as receiver
                            tree.nodes[dest]['is_receiver'] = True
                            tree.nodes[dest]['epoch_size'] = epoch_size
                            tree.nodes[dest]['ratchet_size'] = ratchet_size
                        # for each node, set the counter of next nodes
                        for node in tree.nodes():
                            count = len(list(tree.successors(node)))
                            tree.nodes[node]['counter'] = count
                        one_to_many_paths.append(tree)

                        session = Session()
                        session.set_protocol(self.protocol)
                        session.set_nodes([sender] + destinations)
                        session.add_path(tree)
                        session.set_id(id)
                        id += 1

                        # now generate all the one-to-one return paths to the sender
                        one_to_one_return_paths = []
                        subsessions = []
                        
                        for dest in destinations:
                            path = nx.DiGraph()
                            nx.add_path(path, nx.shortest_path(self.G, dest, sender))

                            if self.params.get("ls_size_epoch_range")[0] == self.params.get("ls_size_epoch_range")[1]:
                                epoch_size = self.params.get("ls_size_epoch_range")[0]
                            else:
                                epoch_size = self.rng.integers(self.params.get("ls_size_epoch_range")[0], self.params.get("ls_size_epoch_range")[1] + 1)
                            if self.params.get("ls_size_ratchet_range")[0] == self.params.get("ls_size_ratchet_range")[1]:
                                ratchet_size = self.params.get("ls_size_ratchet_range")[0]
                            else:
                                ratchet_size = self.rng.integers(self.params.get("ls_size_ratchet_range")[0], self.params.get("ls_size_ratchet_range")[1] + 1)
                            
                            path.nodes[sender]['epoch_size'] = epoch_size
                            path.nodes[sender]['ratchet_size'] = ratchet_size
                            path.nodes[dest]['epoch_size'] = epoch_size
                            path.nodes[dest]['ratchet_size'] = ratchet_size

                            one_to_one_return_paths.append(path)

                            subsession = Session()
                            subsession.set_id(id)
                            subsession.set_protocol(self.attributes["support_protocol"])
                            subsession.set_nodes([dest, sender])
                            subsession.add_path(path)
                            subsessions.append(subsession)
                            id += 1
                        
                        for subsession in subsessions:
                            session.add_subsession(subsession)

                        sessions.append(session)

                self.attributes["sessions"] = sessions
                self.attributes["one_to_many_paths"] = one_to_many_paths
                self.attributes["one_to_one_return_paths"] = one_to_one_return_paths


    def _generate_json(self):
        # Logic to convert the network structure into JSON format
        # first add the constants from the consts.json file
        with open("config/netgen_files.json", "r") as f:
            files_config = json.load(f)

        with open(files_config.get("consts"), "r") as f:
            consts = json.load(f)
       
        
        network_dict = {}

        network_dict["consts"] = consts
        network_dict["items"] = []

        # add nodes
        nodes_dict = self._add_nodes_to_dict()
        network_dict["items"].append(nodes_dict)

        # add interfaces
        interfaces_dict = self._add_interfaces_to_dict()
        network_dict["items"].append(interfaces_dict)

        # add channels
        channels_dict = self._add_channels_to_dict()
        network_dict["items"].append(channels_dict)

        # add links
        links_dict = self._add_links_to_dict()
        network_dict["items"].append(links_dict)

        # add link_refs
        link_refs_dict = self._add_link_refs_to_dict()
        network_dict["items"].append(link_refs_dict)

        # add session_paths
        session_paths_dict = self._add_session_paths_to_dict()
        network_dict["items"].append(session_paths_dict)

        # add local sessions
        local_sessions_dict = self._add_local_sessions_to_dict()
        network_dict["items"].append(local_sessions_dict)

        # add session_checkers
        session_checkers_dict = self._add_session_checkers_to_dict()
        network_dict["items"].append(session_checkers_dict)


        # # Print del network_dict in modo leggibile
        # print("=== NETWORK DICTIONARY ===")
        # print(json.dumps(network_dict, indent=2, ensure_ascii=False))
        # print("=" * 27)

        return network_dict

    
    def _add_nodes_to_dict(self):
        with open("config/netgen_files.json", "r") as f:
            files_config = json.load(f)

        with open(files_config.get("ranges"), "r") as f:
            ranges = json.load(f)


        nodes_dict = {}
        nodes_dict["name"] = "node_modules"
        nodes_dict["template"] = files_config.get("node_template")
        nodes_dict["instances"] = []
        range = ranges.get("node_range_state")
        sizes = {}
        nodes = list(self.G.nodes())
        for node in nodes:
            node_to_add = {}
            node_to_add["name"] = f"node_{node}"
            node_to_add["#"] = node
            node_to_add["range_state"] = range
            node_to_add["init_state"] = 0
            if self.attributes.get("buffer_size_range")[0] == self.attributes.get("buffer_size_range")[1]:
                size_buffer = self.attributes.get("buffer_size_range")[0]
            else:
                size_buffer = int(self.rng.integers(self.attributes.get("buffer_size_range")[0], self.attributes.get("buffer_size_range")[1] + 1))
            node_to_add["size_buffer"] = size_buffer    
            sizes[node] = size_buffer
            node_to_add["init_buffer"] = 0
            if self.attributes.get("node_prob_off_to_on")[0] == self.attributes.get("node_prob_off_to_on")[1]:
                prob_off_to_on = self.attributes.get("node_prob_off_to_on")[0]
            else:
                prob_off_to_on = round(self.rng.uniform(self.attributes.get("node_prob_off_to_on")[0], self.attributes.get("node_prob_off_to_on")[1]), 2)
            node_to_add["prob_off_to_on"] = prob_off_to_on
            if self.attributes.get("node_prob_on_to_off")[0] == self.attributes.get("node_prob_on_to_off")[1]:
                prob_on_to_off = self.attributes.get("node_prob_on_to_off")[0]
            else:
                prob_on_to_off = round(self.rng.uniform(self.attributes.get("node_prob_on_to_off")[0], self.attributes.get("node_prob_on_to_off")[1]), 2)
            node_to_add["prob_on_to_off"] = prob_on_to_off

            neighbors = list(self.G.neighbors(node))
            node_to_add["interfaces"] = []
            for neighbor in neighbors:
                interface = {"#": f"{node}_{neighbor}"}
                node_to_add["interfaces"].append(interface)
            nodes_dict["instances"].append(node_to_add)
        self.attributes["nodes_buffer_sizes"] = sizes
        return nodes_dict
            

    def _add_interfaces_to_dict(self):
        with open("config/netgen_files.json", "r") as f:
            files_config = json.load(f)

        with open(files_config.get("ranges"), "r") as f:
            ranges = json.load(f)


        interfaces_dict = {}
        interfaces_dict["name"] = "interface_modules"
        interfaces_dict["template"] = files_config.get("interface_template")
        interfaces_dict["instances"] = []
        range_state = ranges.get("interface_range_state")
        for edge in self.G.edges():
            node_a = edge[0]
            node_b = edge[1]
            # interface from node_a to node_b
            interface_ab = {}
            interface_ab["name"] = f"interface_{node_a}_{node_b}"
            interface_ab["#"] = f"{node_a}_{node_b}"
            interface_ab["range_state"] = range_state
            interface_ab["init_state"] = 0
           
            # add probabilities
            interface_ab["prob_off_to_working"] = round(self.rng.uniform(self.attributes.get("if_prob_off_to_working")[0], self.attributes.get("if_prob_off_to_working")[1]), 2)
            interface_ab["prob_off_to_error"] = round(self.rng.uniform(self.attributes.get("if_prob_off_to_error")[0], self.attributes.get("if_prob_off_to_error")[1]), 2)
            interface_ab["prob_off_to_failure"] = round(self.rng.uniform(self.attributes.get("if_prob_off_to_failure")[0], self.attributes.get("if_prob_off_to_failure")[1]), 2)
            interface_ab["prob_working_to_error"] = round(self.rng.uniform(self.attributes.get("if_prob_working_to_error")[0], self.attributes.get("if_prob_working_to_error")[1]), 2)
            interface_ab["prob_error_to_working"] = round(self.rng.uniform(self.attributes.get("if_prob_error_to_working")[0], self.attributes.get("if_prob_error_to_working")[1]), 2)
            interface_ab["prob_failure_to_working"] = round(self.rng.uniform(self.attributes.get("if_prob_failure_to_working")[0], self.attributes.get("if_prob_failure_to_working")[1]), 2)
            interface_ab["ref_node_state"] = f"{node_a}"
            interfaces_dict["instances"].append(interface_ab)

        return interfaces_dict


    def _add_channels_to_dict(self):
        with open("config/netgen_files.json", "r") as f:
            files_config = json.load(f)

        with open(files_config.get("ranges"), "r") as f:
            ranges = json.load(f)


        channels_dict = {}
        channels_dict["name"] = "channel_modules"
        channels_dict["template"] = files_config.get("channel_template")
        channels_dict["instances"] = []
        range_state = ranges.get("channel_range_state")
        for edge in self.G.edges():
            node_a = edge[0]
            node_b = edge[1]
            channel = {}
            channel["name"] = f"channel_{node_a}_{node_b}"
            channel["#"] = f"{node_a}_{node_b}"
            channel["range_state"] = range_state
            init_value = int(range_state.strip('[]').split('..')[0])
            channel["init_state"] = init_value
            if self.attributes.get("channel_bandwidth_range")[0] == self.attributes.get("channel_bandwidth_range")[1]:
                size_bandwidth = self.attributes.get("channel_bandwidth_range")[0]
            else:
                size_bandwidth = int(self.rng.integers(self.attributes.get("channel_bandwidth_range")[0], self.attributes.get("channel_bandwidth_range")[1] + 1))
            channel["size_bandwidth"] = size_bandwidth
            channel["init_bandwidth"] = size_bandwidth
            channel["prob_working_to_error"] = round(self.rng.uniform(self.attributes.get("channel_prob_working_to_error")[0], self.attributes.get("channel_prob_working_to_error")[1]), 2)
            channel["prob_error_to_working"] = round(self.rng.uniform(self.attributes.get("channel_prob_error_to_working")[0], self.attributes.get("channel_prob_error_to_working")[1]), 2)
            channel["prob_failure_to_working"] = round(self.rng.uniform(self.attributes.get("channel_prob_failure_to_working")[0], self.attributes.get("channel_prob_failure_to_working")[1]), 2)
            channels_dict["instances"].append(channel)

        return channels_dict


    def _add_links_to_dict(self):
        with open("config/netgen_files.json", "r") as f:
            files_config = json.load(f)

        with open(files_config.get("ranges"), "r") as f:
            ranges = json.load(f)

        # generate links and link_ref_counters for each path

        links_dict = {}
        links_dict["name"] = "link_modules"
        links_dict["template"] = files_config.get("link_template")
        links_dict["instances"] = []
        range_state = ranges.get("link_range_state")

        sessions = self.attributes.get("sessions").copy()
        for session in self.attributes.get("sessions"):
            sessions.extend(session.get_subsessions())

        for session in sessions:
            session_id = session.get_id()
            for path in session.get_paths():
                for edge in path.edges():
                    sender = path.nodes()[0]
                    receivers = path.nodes()[1:]
                    node_a = edge[0]
                    node_b = edge[1]

                    # links instances
                    link = {}
                    link["name"] = f"link_{node_a}_{node_b}_of_path_{sender}_{session_id}"
                    link["#"] = f"{node_a}_{node_b}_{sender}_{session_id}"

                    # states
                    link["range_state"] = range_state
                    init_value = int(range_state.strip('[]').split('..')[0])
                    link["init_state"] = init_value
                    link["init_prev"] = False
                    link["init_sending"] = False
                    link["init_receiving"] = False

                    # references
                    link["ref_channel"] = f"{node_a}_{node_b}"
                    link["ref_interface_sender"] = f"{node_a}_{node_b}"
                    link["ref_node_buffer_sender"] = f"{node_a}"
                    link["ref_link_ref_counter"] = f"{node_a}_{sender}_{session_id}"
                    next_links = []
                    successors = list(path.successors(node_b))
                    for succ in successors:
                        next_links.append({"ref_link_next": f"{node_b}_{succ}_{sender}_{session_id}"})
                    link["next_links"] = next_links
                    link["ref_interface_receiver"] = f"{node_b}_{node_a}"
                    link["ref_node_buffer_receiver"] = f"{node_b}"
                    link["size_node_buffer_receiver"] = self.attributes["nodes_buffer_sizes"][node_b]

                    # probabilities
                    if self.attributes.get("prob_working_to_error")[0] == self.attributes.get("prob_working_to_error")[1]:
                        link["prob_working_to_error"] = self.attributes.get("link_prob_working_to_error")[0]
                    else:
                        link["prob_working_to_error"] = round(self.rng.uniform(self.attributes.get("link_prob_working_to_error")[0], self.attributes.get("link_prob_working_to_error")[1]), 2)
                    if self.attributes.get("prob_error_to_working")[0] == self.attributes.get("prob_error_to_working")[1]:
                        link["prob_error_to_working"] = self.attributes.get("link_prob_error_to_working")[0]
                    else:
                        link["prob_error_to_working"] = round(self.rng.uniform(self.attributes.get("link_prob_error_to_working")[0], self.attributes.get("link_prob_error_to_working")[1]), 2)
                    
                    if self.attributes.get("prob_failure_to_working")[0] == self.attributes.get("prob_failure_to_working")[1]:
                        link["prob_failure_to_working"] = self.attributes.get("link_prob_failure_to_working")[0]
                    else:
                        link["prob_failure_to_working"] = round(self.rng.uniform(self.attributes.get("link_prob_failure_to_working")[0], self.attributes.get("link_prob_failure_to_working")[1]), 2)
                    if self.attributes.get("prob_retry")[0] == self.attributes.get("prob_retry")[1]:
                        link["prob_retry"] = self.attributes.get("link_prob_retry")[0]
                    else:
                        link["prob_retry"] = round(self.rng.uniform(self.attributes.get("link_prob_retry")[0], self.attributes.get("link_prob_retry")[1]), 2)
                    if self.attributes.get("prob_sending")[0] == self.attributes.get("prob_sending")[1]:
                        link["prob_sending"] = self.attributes.get("link_prob_sending")[0]
                    else:
                        link["prob_sending"] = round(self.rng.uniform(self.attributes.get("link_prob_sending")[0], self.attributes.get("link_prob_sending")[1]), 2)

                    links_dict["instances"].append(link)

        return links_dict

    
    def _add_link_refs_to_dict(self):
        with open("config/netgen_files.json", "r") as f:
            files_config = json.load(f)

        link_refs_dict = {}
        link_refs_dict["name"] = "link_ref_modules"
        link_refs_dict["template"] = files_config.get("link_ref_template")
        link_refs_dict["instances"] = []

        sessions = self.attributes.get("sessions").copy()
        for session in self.attributes.get("sessions"):
            sessions.extend(session.get_subsessions())

        for session in sessions:
            session_id = session.get_id()
            for path in session.get_paths():
                sender = path.nodes[0]
                for node in path.nodes():
                    # link_ref_counter instances
                    link_ref = {}
                    link_ref["name"] = f"link_ref_{node}_of_path_{sender}_{session_id}"
                    link_ref["#"] = f"{node}_{sender}_{session_id}"
                    
                    if session.get_protocol() in {"HPKE", "DOUBLE-RATCHET", "SUPPORT_HPKE", "SUPPORT_DOUBLE-RATCHET"}:
                        link_ref["size_counter"] = 1
                        link_ref["init_counter"] = 1
                    else:
                        counter = path.nodes[node].get("counter", 1)
                        link_ref["size_counter"] = counter
                        link_ref["init_counter"] = counter
                    link_ref["is_receiver"] = path.nodes[node].get("is_receiver", False)
                    link_ref["ref_node"] = f"{node}"
                    link_refs_dict["instances"].append(link_ref)

        return link_refs_dict
        

    def _add_session_paths_to_dict(self):

        with open("config/netgen_files.json", "r") as f:
            files_config = json.load(f)

        with open(files_config.get("ranges"), "r") as f:
            ranges = json.load(f)

        with open(files_config.get("consts"), "r") as f:
            consts = json.load(f)

        session_paths_dict = {}
        session_paths_dict["name"] = "session_path_modules"
        session_paths_dict["template"] = files_config.get("session_path_template")
        session_paths_dict["instances"] = []
        range_state = ranges.get("session_path_state_range")
        range_system_message = ranges.get("session_path_system_message_range")
        range_data_message = ranges.get("session_path_data_message_range")
        consts_idle = consts.get("const_idle")
        const_message_data = consts.get("const_message_data")

            
        sessions = self.attributes.get("sessions").copy()
        for session in self.attributes.get("sessions"):
            sessions.extend(session.get_subsessions())

        for session in sessions:
            sessions_id = session.get_id()
            for path in session.paths:
                sender = path.nodes[0]
                receivers = path.nodes[1:]
                session_path = {}
                session_path["name"] = f"session_path_{sender}_{sessions_id}"
                session_path["#"] = f"{sender}_{sessions_id}"
                session_path["protocol"] = sessions.get_protocol()

                # states
                session_path["range_depth"] = range_state
                session_path["init_state"] = consts_idle
                session_path["range_system_message"] = range_system_message
                session_path["init_system_message"] = const_message_data 
                session_path["range_data_message"] = range_data_message
                session_path["init_data_message"] = const_message_data 
                session_path["range_prev_local_session_epoch_sender"] = path.nodes[0]['epoch_size']
                session_path["init_prev_local_session_epoch_sender"] = 0
                session_path["size_link_counter"] = len(path[0]) - 1
                session_path["init_link_counter"] = len(path[0]) - 1
                session_path["size_checker_counter"] = len(receivers)  # one session checker for each receiver
                session_path["init_checker_counter"] = len(receivers)

                # references
                session_path["ref_node_sender"] = f"{sender}"
                session_path["ref_local_session_sender"] = f"{sender}_{sessions_id}"
                session_path["size_buffer_sender"] = self.attributes["nodes_buffer_sizes"][sender]
                succ = next(path.successors(sender))
                first_links = [{"#": f"{sender}_{succ}_{sender}_{sessions_id}"}] if succ else None
                session_path["first_links"] = first_links

                # probabilities
                if session.get_protocol() in {"SUPPORT_HPKE", "SUPPORT_DOUBLE-RATCHET"}:
                    session_path["prob_run"] = 0.0
                else:
                    if self.attributes.get("sp_prob_run")[0] == self.attributes.get("sp_prob_run")[1]:
                            prob_run = self.attributes.get("sp_prob_run")[0]
                    else:
                        prob_run = round(self.rng.uniform(self.attributes.get("sp_prob_run")[0], self.attributes.get("sp_prob_run")[1]), 2)
                    session_path["prob_run"] = prob_run
                session_paths_dict["instances"].append(session_path)

        return session_paths_dict


    def _add_local_sessions_to_dict(self):
        with open("config/netgen_files.json", "r") as f:
            files_config = json.load(f)

        with open(files_config.get("ranges"), "r") as f:
            ranges = json.load(f)

        with open(files_config.get("consts"), "r") as f:
            consts = json.load(f)

        local_sessions_dict = {}
        local_sessions_dict["name"] = "local_session_modules"
        local_sessions_dict["template"] = files_config.get("local_session_template")
        local_sessions_dict["instances"] = []
        range_state = ranges.get("local_session_range_state")
        const_valid = consts.get("const_local_session_valid")

        sessions = self.attributes.get("sessions")
        sessions_copy = sessions.copy()
        for session in sessions:
            subs = session.get_subsessions()
            sessions_copy.extend(subs)
        for session in sessions_copy:
            nodes = session.get_nodes()
            id = session.get_id()
            for node in nodes:
                local_session = {}
                local_session["name"] = f"local_session_of_{node}_of_session_{id}"
                local_session["#"] = f"{node}_{id}"

                # states
                local_session["range_state"] = range_state
                local_session["init_state"] = const_valid 
                local_session["size_epoch"] = session.get_nodes()[node]['epoch_size']
                local_session["init_epoch"] = 0
                local_session["size_ratchet"] = session.get_nodes()[node]['ratchet_size']
                local_session["init_ratchet"] = 0
                local_session["init_compromise"] = False

                # probabilities
                if self.params["ls_prob_session_reset"][0] == self.params["ls_prob_session_reset"][1]:
                    local_session["prob_session_reset"] = self.params["ls_prob_session_reset"][0]
                else:
                    local_session["prob_session_reset"] = round(self.rng.uniform(self.params["ls_prob_session_reset"][0], self.params["ls_prob_session_reset"][1]), 2)
                if self.params["ls_prob_compromised"][0] == self.params["ls_prob_compromised"][1]:
                    local_session["prob_compromised"] = self.params["ls_prob_compromised"][0]
                else:
                    local_session["prob_compromised"] = round(self.rng.uniform(self.params["ls_prob_compromised"][0], self.params["ls_prob_compromised"][1]), 2)
                if self.params["ls_prob_ratchet_reset"][0] == self.params["ls_prob_ratchet_reset"][1]:
                    local_session["prob_ratchet_reset"] = self.params["ls_prob_ratchet_reset"][0]
                else:   
                    local_session["prob_ratchet_reset"] = round(self.rng.uniform(self.params["ls_prob_ratchet_reset"][0], self.params["ls_prob_ratchet_reset"][1]), 2)
                if self.params["ls_prob_none"][0] == self.params["ls_prob_none"][1]:
                    local_session["prob_none"] = self.params["ls_prob_none"][0]
                else:
                    local_session["prob_none"] = round(self.rng.uniform(self.params["ls_prob_none"][0], self.params["ls_prob_none"][1]), 2)

                local_sessions_dict["instances"].append(local_session)

        return local_sessions_dict

                    
    def _add_session_checkers_to_dict(self):
        # one session checker for each receiver in each path

        with open("config/netgen_files.json", "r") as f:
            files_config = json.load(f)

        with open(files_config.get("ranges"), "r") as f:
            ranges = json.load(f)

        with open(files_config.get("consts"), "r") as f:
            consts = json.load(f)

        state_range = ranges.get("session_checker_state_range")
        session_checker_sender_local_session_system_message_range = ranges.get("session_checker_sender_local_session_system_message_range")
        session_checker_sender_local_session_data_message_range = ranges.get("session_checker_sender_local_session_data_message_range")
        const_idle = consts.get("const_idle")
        const_message_data = consts.get("const_message_data")

        session_checkers_dict = {}
        session_checkers_dict["name"] = "session_checker_modules"
        session_checkers_dict["template"] = files_config.get("session_checker_template")
        session_checkers_dict["instances"] = []

        sessions = self.attributes.get("sessions").copy()
        for session in self.attributes.get("sessions"):
            sessions.extend(session.get_subsessions())

        for session in sessions:
            sender = session.get_nodes()[0]
            receivers = session.get_nodes()[1:]
            session_id = session.get_id()
            for receiver in receivers:
                session_checker = {}
                session_checker["name"] = f"session_checker_of_{receiver}_of_session_{session_id}"
                session_checker["#"] = f"{receiver}_{session_id}"
                session_checker["protocol"] = session.get_protocol()

                # states
                session_checker["range_state"] = state_range
                session_checker["init_state"] = const_idle
                session_checker["range_sender_local_session_epoch"] = session.get_nodes()[sender]['epoch_size']
                session_checker["init_sender_local_session_epoch"] = 0
                session_checker["range_sender_local_session_system_message"] = session_checker_sender_local_session_system_message_range
                session_checker["init_sender_local_session_system_message"] = const_message_data
                session_checker["range_sender_local_session_data_message"] = session_checker_sender_local_session_data_message_range
                session_checker["init_sender_local_session_data_message"] = const_message_data

                # references
                session_checker["ref_local_session_sender"] = f"{sender}_{session_id}"
                session_checker["ref_session_path"] = f"{sender}_{session_id}"
                session_checker["ref_local_session_receiver"] = f"{receiver}_{session_id}"
                session_checker["ref_node_receiver"] = f"{receiver}"
                if session.get_protocol() in {"HPKE", "DOUBLE-RATCHET", "SUPPORT_HPKE", "SUPPORT_DOUBLE-RATCHET"}:
                    session_checker["ref_session_path_to_sender"] = f"{receiver}_{session_id}"
                    session_checker["ref_session_checker_sender"] = f"{sender}_{session_id}"
                else:
                    subsessions = session.get_subsessions()
                    for subsession in subsessions:
                        if subsession.get_nodes()[0] == receiver or subsession.get_nodes()[1] == receiver:
                            session_checker["ref_session_path_to_sender"] = f"{receiver}_{subsession.get_id()}"
                            session_checker["ref_session_checker_sender"] = f"{sender}_{subsession.get_id()}"
                            break
            
                session_checker["size_ratchet_sender"] = session.get_nodes()[sender]['ratchet_size']

                session_checker["ref_local_session_sender_key_receiver"] = 0 # TO DO

                session_checkers_dict["instances"].append(session_checker)

        return session_checkers_dict      