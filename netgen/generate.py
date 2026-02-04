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


class Contribution:
    def __init__(self, command, condition, value):
        self.command = command
        self.condition = condition
        self.value = value


class Reward:
    def __init__(self, name):
        self.name = name
        self.contributions = []


    def add_contribution(self, contribution):
        self.contributions.append(contribution)


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

        self._assign_attributes()
        
        self._generate_paths()

        self._generate_rewards()
        
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
            "link_prob_working_to_error", "link_prob_error_to_working", "link_prob_failure_to_working", "link_prob_retry", "link_prob_sending",
            "ls_prob_session_reset", "ls_prob_ratchet_reset", "ls_prob_none", "ls_prob_compromised",
            "sp_prob_run"
        ]
        
        for attr_name in attributes_to_add:
            self.attributes[attr_name] = self.params[attr_name]

        
    def _generate_paths(self):
        nodes = list(self.G.nodes())
        #if the protocol is hpke or double ratchet, we need only one-to-one paths and return paths
        match self.protocol:
            case "hpke" | "double_ratchet":
                paths = []
                return_paths = []
                sessions = []
                id = 0
                for i in range(len(nodes)):
                    for j in range(i, len(nodes)):
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
                                
                                path.nodes[b]['is_receiver'] = True

                                paths.append(path)

                                # generate the return path
                                return_path = nx.DiGraph()
                                nx.add_path(return_path, nx.shortest_path(self.G, b, a))

                                return_path.nodes[b]['epoch_size'] = epoch_size
                                return_path.nodes[b]['ratchet_size'] = ratchet_size
                                return_path.nodes[a]['epoch_size'] = epoch_size
                                return_path.nodes[a]['ratchet_size'] = ratchet_size
                                return_path.nodes['is_receiver'] = False
                                return_path.nodes[a]['is_receiver'] = True

                                return_paths.append(return_path)

                                session = Session()
                                session.set_id(id)
                                session.set_protocol(self.protocol)
                                session.set_nodes([a, b])
                                session.add_path(path)
                                session.add_path(return_path)
                                sessions.append(session)
                                id += 1

                                # correspondence between paths and sessions
                                print(f"Session ID: {session.get_id()}, Nodes: {session.get_nodes()}")
                                print("Paths:")
                                for p in session.get_paths():
                                    print(list(p.edges()))
                                print("\n")

                self.attributes["paths"] = paths
                self.attributes["return_paths"] = return_paths
                self.attributes["sessions"] = sessions

            # for sender key, we need one-to-many paths and all the one-to-one paths sender_receiver and receiver_sender
            case "sender_key":
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
                    if len(destinations) == 0:
                        destinations.append(self.rng.choice([n for n in nodes if n != sender]))
                    tree = nx.DiGraph()
                    tree.add_node(sender)
                    tree.nodes[sender]['epoch_size'] = epoch_size
                    tree.nodes[sender]['ratchet_size'] = ratchet_size
                    # for each destination, generate the shortest path from sender to destination and add it to the tree
                    for dest in destinations:
                        path = nx.DiGraph()
                        nx.add_path(path, nx.shortest_path(self.G, sender, dest))
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
                        path.nodes[receiver]['is_receiver'] = True

                        one_to_one_paths.append(path)

                        # one-to-one return path
                        return_path = nx.DiGraph()
                        nx.add_path(return_path, nx.shortest_path(self.G, receiver, sender))
                        
                        return_path.nodes[sender]['epoch_size'] = epoch_size
                        return_path.nodes[sender]['ratchet_size'] = ratchet_size
                        return_path.nodes[receiver]['epoch_size'] = epoch_size
                        return_path.nodes[receiver]['ratchet_size'] = ratchet_size
                        return_path.nodes[sender]['is_receiver'] = True

                        one_to_one_return_paths.append(return_path)

                        subsession = Session()
                        subsession.set_id(id)
                        subsession.set_protocol(self.attributes.get("support_protocol"))
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

                # correspondence between paths and sessions
                for session in sessions:
                    print(f"Session ID: {session.get_id()}, Nodes: {session.get_nodes()}")
                    print("Paths:")
                    for p in session.get_paths():
                        print(list(p.edges()))
                    print("Subsessions:")
                    for ss in session.get_subsessions():
                        print(f"  Subsession ID: {ss.get_id()}, Nodes: {ss.get_nodes()}")
                        for p in ss.get_paths():
                            print(f"    {list(p.edges())}")
                    print("\n")
            
            case "mls":
                # for mls, generate random sessions and one-to-many paths in each session with the one to one return paths
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
                            path = nx.DiGraph()
                            nx.add_path(path, nx.shortest_path(self.G, sender, dest))
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
                        for dest in destinations:
                            path = nx.DiGraph()
                            nx.add_path(path, nx.shortest_path(self.G, dest, sender))

                            path.nodes[sender]['epoch_size'] = epoch_size
                            path.nodes[sender]['ratchet_size'] = ratchet_size
                            path.nodes[dest]['epoch_size'] = epoch_size
                            path.nodes[dest]['ratchet_size'] = ratchet_size
                            path.nodes[dest]['is_receiver'] = True
                            path['return_path'] = True

                            one_to_one_return_paths.append(path)

                            session.add_path(path)

                        sessions.append(session)

                self.attributes["sessions"] = sessions
                self.attributes["one_to_many_paths"] = one_to_many_paths
                self.attributes["one_to_one_return_paths"] = one_to_one_return_paths

                # correspondence between paths and sessions
                for session in sessions:
                    print(f"Session ID: {session.get_id()}, Nodes: {session.get_nodes()}")
                    print("Paths:")
                    for p in session.get_paths():
                        print(list(p.edges()))
                    print("Subsessions:")
                    for ss in session.get_subsessions():
                        print(f"  Subsession ID: {ss.get_id()}, Nodes: {ss.get_nodes()}")
                        for p in ss.get_paths():
                            print(f"    {list(p.edges())}")
                    print("\n")


    def _generate_rewards(self):
        self.attributes["rewards"] = []

        def generate_message_reward():
            with open("config/netgen_files.json", "r") as f:
                files_config = json.load(f)
            
            with open(files_config.get("consts"), "r") as f:
                consts = json.load(f)
            
            rewards_data = {
                "hpke": Reward("reward_data_message_hpke"),
                "double_ratchet": Reward("reward_data_message_double_ratchet"),
                "sender_key": Reward("reward_data_message_sender_key"), 
                "mls": Reward("reward_data_message_mls")
            }

            rewards_system = {
                "hpke": Reward("reward_system_message_hpke"),
                "double_ratchet": Reward("reward_system_message_double_ratchet"),
                "sender_key": Reward("reward_system_message_sender_key"), 
                "mls": Reward("reward_system_message_mls")
            }

            reward_message = Reward("reward_total_messages")

            sessions = self.attributes["sessions"].copy()
            for session in self.attributes["sessions"]:
                ss = session.get_subsessions()
                sessions.extend(ss)

            for session in sessions:
                for i, path in enumerate(session.paths):
                    id = session.id
                    session_path = f"{i}_{id}" 
                    command = f"cmd_send_session_path_{session_path}"

                    reward_message.add_contribution(Contribution(command, "true", "1"))

                    match session.protocol:
                        case "hpke":
                            condition = f"(sender_path_system_message_{session_path} = {consts['const_message_data']}) | (sender_path_system_message_{session_path} = {consts['const_message_ratchet']})"
                            value = "1" 
                            rewards_data["hpke"].add_contribution(Contribution(command, condition, value))
                            condition = f"(sender_path_system_message_{session_path} = {consts['const_message_reset']})"
                            rewards_system["hpke"].add_contribution(Contribution(command, condition, value))

                        case "double_ratchet":
                            condition = f"(sender_path_system_message_{session_path} = {consts['const_message_ratchet']})"
                            value = "1" 
                            rewards_data["double_ratchet"].add_contribution(Contribution(command, condition, value))
                            condition = f"(sender_path_system_message_{session_path} = {consts['const_message_reset']})"
                            rewards_system["double_ratchet"].add_contribution(Contribution(command, condition, value))

                        case "hpke_sender_key" | "double_ratchet_sender_key":
                            condition = "true" # all messages should be system
                            value = "1"
                            rewards_system["sender_key"].add_contribution(Contribution(command, condition, value))

                        case "sender_key":
                            condition = "true" # all messages should be data
                            value = "1"
                            rewards_data["sender_key"].add_contribution(Contribution(command, condition, value))

                        case "mls":                    
                            if path['return_path'] == True:
                                condition = "true" # all messages should be system
                                value = "1"
                                rewards_system["mls"].add_contribution(Contribution(command, condition, value))
                            else:
                                condition = f"(sender_path_system_message_{session_path} = {consts['const_message_data']}) | (sender_path_system_message_{session_path} = {consts['const_message_ratchet']})"
                                value = "1" 
                                rewards_data["mls"].add_contribution(Contribution(command, condition, value))
                                condition = f"(sender_path_system_message_{session_path} = {consts['const_message_reset']})"
                                rewards_system["mls"].add_contribution(Contribution(command, condition, value))

            self.attributes["rewards"].extend(list(rewards_data.values()) + list(rewards_system.values()) + [reward_message])
        
        generate_message_reward()


    def _generate_policies(self):
        pass


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

        # add rewards
        rewards_dict = self._add_rewards_to_dict()
        network_dict["items"].append(rewards_dict)

        return network_dict

    
    def _add_nodes_to_dict(self):
        with open("config/netgen_files.json", "r") as f:
            files_config = json.load(f)

        with open(files_config.get("ranges"), "r") as f:
            ranges = json.load(f)

        with open(files_config.get("init"), "r") as f:
            init = json.load(f)


        nodes_dict = {}
        nodes_dict["name"] = "node_modules"
        nodes_dict["template"] = files_config.get("node_template")
        nodes_dict["instances"] = []
        range = ranges.get("node_range_state")
        sizes = {}
        nodes = list(self.G.nodes())
        for node in nodes:
            node_to_add = {}
            node_to_add["name"] = f"node_{node}.prism"
            node_to_add["#"] = node
            node_to_add["range_state"] = range
            node_to_add["init_state"] = init.get("node_init_state")
            node_to_add["node_on_to_off_init"] = init.get("node_on_to_off_init")
            if self.attributes.get("buffer_size_range")[0] == self.attributes.get("buffer_size_range")[1]:
                size_buffer = self.attributes.get("buffer_size_range")[0]
            else:
                size_buffer = int(self.rng.integers(self.attributes.get("buffer_size_range")[0], self.attributes.get("buffer_size_range")[1] + 1))
            node_to_add["size_buffer"] = size_buffer    
            sizes[node] = size_buffer
            node_to_add["init_buffer"] = init.get("node_init_buffer")
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

            # commands
            node_to_add["cmd_off_to_on"] = f"cmd_off_to_on_node_{node}"
            node_to_add["cmd_on_to_off"] = f"cmd_on_to_off_node_{node}"
            node_to_add["cmd_shutting_down"] = f"cmd_shutting_down_node_{node}"

            # now i need all the links that arrives to this node
            links = []
            sessions = self.attributes.get("sessions").copy()
            for session in self.attributes.get("sessions"):
                ss = session.get_subsessions()
                sessions.extend(ss)
            for session in sessions:
                for i, path in enumerate(session.paths):
                    if node in path.nodes():
                        path_edges = list(path.edges())
                        for edge in path_edges:
                            if edge[1] == node:
                                link_name = f"link_{edge[0]}_{edge[1]}_of_path_{i}_{session.id}"
                                commands = {}
                                commands["cmd_receive_success"] = f"cmd_receive_success_{link_name}"
                                links.append(commands)

            node_to_add["links"] = links

            # here i need the list of the link_refs of this node
            link_refs = []
            for session in sessions:
                for i, path in enumerate(session.paths):
                    if node in path.nodes() and path.nodes[node].get("is_receiver") == False:
                        commands = {}
                        link_ref_name = f"link_ref_{node}_of_path_{i}_{session.id}"
                        commands["cmd_reset"] = f"cmd_reset_{link_ref_name}"
                        link_refs.append(commands)

            node_to_add["link_refs"] = link_refs
                                
            nodes_dict["instances"].append(node_to_add)
        self.attributes["nodes_buffer_sizes"] = sizes
        return nodes_dict
            

    def _add_interfaces_to_dict(self):
        with open("config/netgen_files.json", "r") as f:
            files_config = json.load(f)

        with open(files_config.get("ranges"), "r") as f:
            ranges = json.load(f)

        with open(files_config.get("init"), "r") as f:
            init = json.load(f)


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
            interface_ab["name"] = f"interface_{node_a}_{node_b}.prism"
            interface_ab["#"] = f"{node_a}_{node_b}"
            interface_ab["range_state"] = range_state
            interface_ab["init_state"] = init.get("interface_init_state")
           
            # probabilities
            if self.attributes.get("if_prob_off_to_working")[0] == self.attributes.get("if_prob_off_to_working")[1]:
                interface_ab["prob_off_to_working"] = self.attributes.get("if_prob_off_to_working")[0]
            else:
                interface_ab["prob_off_to_working"] = round(self.rng.uniform(self.attributes.get("if_prob_off_to_working")[0], self.attributes.get("if_prob_off_to_working")[1]), 2)
            if self.attributes.get("if_prob_off_to_error")[0] == self.attributes.get("if_prob_off_to_error")[1]:
                interface_ab["prob_off_to_error"] = self.attributes.get("if_prob_off_to_error")[0]
            else:
                interface_ab["prob_off_to_error"] = round(self.rng.uniform(self.attributes.get("if_prob_off_to_error")[0], self.attributes.get("if_prob_off_to_error")[1]), 2)
            
            if self.attributes.get("if_prob_off_to_failure")[0] == self.attributes.get("if_prob_off_to_failure")[1]:
                interface_ab["prob_off_to_failure"] = self.attributes.get("if_prob_off_to_failure")[0]
            else:
                interface_ab["prob_off_to_failure"] = round(self.rng.uniform(self.attributes.get("if_prob_off_to_failure")[0], self.attributes.get("if_prob_off_to_failure")[1]), 2)
            if self.attributes.get("if_prob_working_to_error")[0] == self.attributes.get("if_prob_working_to_error")[1]:
                interface_ab["prob_working_to_error"] = self.attributes.get("if_prob_working_to_error")[0]
            else:
                interface_ab["prob_working_to_error"] = round(self.rng.uniform(self.attributes.get("if_prob_working_to_error")[0], self.attributes.get("if_prob_working_to_error")[1]), 2)
            if self.attributes.get("if_prob_error_to_working")[0] == self.attributes.get("if_prob_error_to_working")[1]:
                interface_ab["prob_error_to_working"] = self.attributes.get("if_prob_error_to_working")[0]
            else:
                interface_ab["prob_error_to_working"] = round(self.rng.uniform(self.attributes.get("if_prob_error_to_working")[0], self.attributes.get("if_prob_error_to_working")[1]), 2)
            if self.attributes.get("if_prob_failure_to_working")[0] == self.attributes.get("if_prob_failure_to_working")[1]:
                interface_ab["prob_failure_to_working"] = self.attributes.get("if_prob_failure_to_working")[0]
            else:
                interface_ab["prob_failure_to_working"] = round(self.rng.uniform(self.attributes.get("if_prob_failure_to_working")[0], self.attributes.get("if_prob_failure_to_working")[1]), 2)
            
            interface_ab["ref_node_state"] = f"{node_a}"

            # commands
            interface_ab["cmd_off_to_on"] = f"cmd_off_to_on_interface_{node_a}_{node_b}"
            interface_ab["cmd_working_to_error"] = f"cmd_working_to_error_interface_{node_a}_{node_b}"
            interface_ab["cmd_error_to_working"] = f"cmd_error_to_working_interface_{node_a}_{node_b}"
            interface_ab["cmd_failure_to_working"] = f"cmd_failure_to_working_interface_{node_a}_{node_b}"
            interface_ab["cmd_shutting_down"] = f"cmd_shutting_down_node_{node_a}"

            # interface from node_b to node_a
            interface_ba = {}
            interface_ba["name"] = f"interface_{node_b}_{node_a}.prism"
            interface_ba["#"] = f"{node_b}_{node_a}"
            interface_ba["range_state"] = range_state
            interface_ba["init_state"] = 0

            # probabilities
            if self.attributes.get("if_prob_off_to_working")[0] == self.attributes.get("if_prob_off_to_working")[1]:
                interface_ba["prob_off_to_working"] = self.attributes.get("if_prob_off_to_working")[0]
            else:
                interface_ba["prob_off_to_working"] = round(self.rng.uniform(self.attributes.get("if_prob_off_to_working")[0], self.attributes.get("if_prob_off_to_working")[1]), 2)
            if self.attributes.get("if_prob_off_to_error")[0] == self.attributes.get("if_prob_off_to_error")[1]:
                interface_ba["prob_off_to_error"] = self.attributes.get("if_prob_off_to_error")[0]
            else:
                interface_ba["prob_off_to_error"] = round(self.rng.uniform(self.attributes.get("if_prob_off_to_error")[0], self.attributes.get("if_prob_off_to_error")[1]), 2)
            if self.attributes.get("if_prob_off_to_failure")[0] == self.attributes.get("if_prob_off_to_failure")[1]:
                interface_ba["prob_off_to_failure"] = self.attributes.get("if_prob_off_to_failure")[0]
            else:
                interface_ba["prob_off_to_failure"] = round(self.rng.uniform(self.attributes.get("if_prob_off_to_failure")[0], self.attributes.get("if_prob_off_to_failure")[1]), 2)
            if self.attributes.get("if_prob_working_to_error")[0] == self.attributes.get("if_prob_working_to_error")[1]:
                interface_ba["prob_working_to_error"] = self.attributes.get("if_prob_working_to_error")[0]
            else:
                interface_ba["prob_working_to_error"] = round(self.rng.uniform(self.attributes.get("if_prob_working_to_error")[0], self.attributes.get("if_prob_working_to_error")[1]), 2)
            if self.attributes.get("if_prob_error_to_working")[0] == self.attributes.get("if_prob_error_to_working")[1]:
                interface_ba["prob_error_to_working"] = self.attributes.get("if_prob_error_to_working")[0]  
            else:
                interface_ba["prob_error_to_working"] = round(self.rng.uniform(self.attributes.get("if_prob_error_to_working")[0], self.attributes.get("if_prob_error_to_working")[1]), 2)
            if self.attributes.get("if_prob_failure_to_working")[0] == self.attributes.get("if_prob_failure_to_working")[1]:
                interface_ba["prob_failure_to_working"] = self.attributes.get("if_prob_failure_to_working")[0]  
            else:
                interface_ba["prob_failure_to_working"] = round(self.rng.uniform(self.attributes.get("if_prob_failure_to_working")[0], self.attributes.get("if_prob_failure_to_working")[1]), 2)

            # commands
            interface_ba["cmd_off_to_on"] = f"cmd_off_to_on_interface_{node_b}_{node_a}"
            interface_ba["cmd_working_to_error"] = f"cmd_working_to_error_interface_{node_b}_{node_a}"
            interface_ba["cmd_error_to_working"] = f"cmd_error_to_working_interface_{node_b}_{node_a}"
            interface_ba["cmd_failure_to_working"] = f"cmd_failure_to_working_interface_{node_b}_{node_a}"
            interface_ba["cmd_shutting_down"] = f"cmd_shutting_down_node_{node_b}"

            interface_ba["ref_node_state"] = f"{node_b}"


            # now i need the list of the incoming links and outgoing links for each interface
            in_links_ab = []
            out_links_ab = []
            in_links_ba = []
            out_links_ba = []

            sessions = self.attributes.get("sessions").copy()
            for session in self.attributes.get("sessions"):
                ss = session.get_subsessions()
                sessions.extend(ss)
            
            for session in sessions:
                for i, path in enumerate(session.paths):
                    id = session.id
                    session_path = f"{i}_{id}" 
                    edges = list(path.edges())
                    for edge in edges:
                        if edge[0] == node_a and edge[1] == node_b:
                            commands = {}
                            commands["cmd_send_failure"] = f"cmd_send_failure_link {node_a}_{node_b}_of_path_{session_path}"
                            out_links_ab.append(commands)

                            commands = {}
                            commands["cmd_receive_failure"] = f"cmd_receive_failure_link {node_a}_{node_b}_of_path_{session_path}"
                            in_links_ba.append(commands)

                        if edge[0] == node_b and edge[1] == node_a:
                            commands = {}
                            commands["cmd_send_failure"] = f"cmd_send_failure_link {node_b}_{node_a}_of_path_{session_path}"
                            out_links_ba.append(commands)

                            commands = {}
                            commands["cmd_receive_failure"] = f"cmd_receive_failure_link {node_b}_{node_a}_of_path_{session_path}"
                            in_links_ab.append(commands)
            
            interface_ab["in_links"] = in_links_ab
            interface_ab["out_links"] = out_links_ab
            interface_ba["in_links"] = in_links_ba
            interface_ba["out_links"] = out_links_ba

            interfaces_dict["instances"].append(interface_ab)
            interfaces_dict["instances"].append(interface_ba)

        return interfaces_dict


    def _add_channels_to_dict(self):
        with open("config/netgen_files.json", "r") as f:
            files_config = json.load(f)

        with open(files_config.get("ranges"), "r") as f:
            ranges = json.load(f)

        with open(files_config.get("init"), "r") as f:
            init = json.load(f)


        channels_dict = {}
        channels_dict["name"] = "channel_modules"
        channels_dict["template"] = files_config.get("channel_template")
        channels_dict["instances"] = []
        range_state = ranges.get("channel_range_state")
        for edge in self.G.edges():
            node_a = edge[0]
            node_b = edge[1]

            # channel a b
            channel = {}
            channel["name"] = f"channel_{node_a}_{node_b}.prism"
            channel["#"] = f"{node_a}_{node_b}"
            channel["range_state"] = range_state
            channel["init_state"] = init.get("channel_init_state")
            if self.attributes.get("channel_bandwidth_range")[0] == self.attributes.get("channel_bandwidth_range")[1]:
                size_bandwidth = self.attributes.get("channel_bandwidth_range")[0]
            else:
                size_bandwidth = int(self.rng.integers(self.attributes.get("channel_bandwidth_range")[0], self.attributes.get("channel_bandwidth_range")[1] + 1))
            channel["size_bandwidth"] = size_bandwidth
            channel["init_bandwidth"] = size_bandwidth

            # probabilities
            if self.attributes.get("channel_prob_working_to_error")[0] == self.attributes.get("channel_prob_working_to_error")[1]:
                channel["prob_working_to_error"] = self.attributes.get("channel_prob_working_to_error")[0]
            else:
                channel["prob_working_to_error"] = round(self.rng.uniform(self.attributes.get("channel_prob_working_to_error")[0], self.attributes.get("channel_prob_working_to_error")[1]), 2)
            if self.attributes.get("channel_prob_error_to_working")[0] == self.attributes.get("channel_prob_error_to_working")[1]:
                channel["prob_error_to_working"] = self.attributes.get("channel_prob_error_to_working")[0]
            else:
                channel["prob_error_to_working"] = round(self.rng.uniform(self.attributes.get("channel_prob_error_to_working")[0], self.attributes.get("channel_prob_error_to_working")[1]), 2)
            if self.attributes.get("channel_prob_failure_to_working")[0] == self.attributes.get("channel_prob_failure_to_working")[1]:
                channel["prob_failure_to_working"] = self.attributes.get("channel_prob_failure_to_working")[0]
            else:
                channel["prob_failure_to_working"] = round(self.rng.uniform(self.attributes.get("channel_prob_failure_to_working")[0], self.attributes.get("channel_prob_failure_to_working")[1]), 2)
            
            # commands
            channel["cmd_working_to_error"] = f"cmd_working_to_error_channel_{node_a}_{node_b}"
            channel["cmd_error_to_working"] = f"cmd_error_to_working_channel_{node_a}_{node_b}"
            channel["cmd_failure_to_working"] = f"cmd_failure_to_working_channel_{node_a}_{node_b}"

            # channel b a
            channel_ba = {}
            channel_ba["name"] = f"channel_{node_b}_{node_a}.prism"
            channel_ba["#"] = f"{node_b}_{node_a}"
            channel_ba["range_state"] = range_state
            init_value = int(range_state.split('..')[0])
            channel_ba["init_state"] = init_value
            if self.attributes.get("channel_bandwidth_range")[0] == self.attributes.get("channel_bandwidth_range")[1]:
                size_bandwidth = self.attributes.get("channel_bandwidth_range")[0]
            else:
                size_bandwidth = int(self.rng.integers(self.attributes.get("channel_bandwidth_range")[0], self.attributes.get("channel_bandwidth_range")[1] + 1))
            channel_ba["size_bandwidth"] = size_bandwidth
            channel_ba["init_bandwidth"] = size_bandwidth   

            # probabilities
            if self.attributes.get("channel_prob_working_to_error")[0] == self.attributes.get("channel_prob_working_to_error")[1]:
                channel_ba["prob_working_to_error"] = self.attributes.get("channel_prob_working_to_error")[0]
            else:
                channel_ba["prob_working_to_error"] = round(self.rng.uniform(self.attributes.get("channel_prob_working_to_error")[0], self.attributes.get("channel_prob_working_to_error")[1]), 2)
            if self.attributes.get("channel_prob_error_to_working")[0] == self.attributes.get("channel_prob_error_to_working")[1]:
                channel_ba["prob_error_to_working"] = self.attributes.get("channel_prob_error_to_working")[0]
            else:
                channel_ba["prob_error_to_working"] = round(self.rng.uniform(self.attributes.get("channel_prob_error_to_working")[0], self.attributes.get("channel_prob_error_to_working")[1]), 2)
            if self.attributes.get("channel_prob_failure_to_working")[0] == self.attributes.get("channel_prob_failure_to_working")[1]:
                channel_ba["prob_failure_to_working"] = self.attributes.get("channel_prob_failure_to_working")[0]
            else:
                channel_ba["prob_failure_to_working"] = round(self.rng.uniform(self.attributes.get("channel_prob_failure_to_working")[0], self.attributes.get("channel_prob_failure_to_working")[1]), 2)    

            # commands
            channel_ba["cmd_working_to_error"] = f"cmd_working_to_error_channel_{node_b}_{node_a}"
            channel_ba["cmd_error_to_working"] = f"cmd_error_to_working_channel_{node_b}_{node_a}"
            channel_ba["cmd_failure_to_working"] = f"cmd_failure_to_working_channel_{node_b}_{node_a}"


            # now i have to find all the links that use this channel and add the synchronous commands
            links_ab = []
            links_ba = []
            sessions = self.attributes.get("sessions").copy()
            for session in self.attributes.get("sessions"):
                sessions.extend(session.get_subsessions())
            for session in sessions:
                session_id = session.get_id()
                for i, path in enumerate(session.paths):
                    for edge in path.edges():
                        if (edge[0] == node_a) and (edge[1] == node_b):
                            # this link uses the channel_ab
                            ref = f"link_{node_a}_{node_b}_of_path_{i}_{session_id}"
                            commands = {}
                            commands["cmd_send_failure"] = f"cmd_send_failure_{ref}"
                            commands["cmd_send_success"] = f"cmd_send_success_{ref}"
                            commands["cmd_receive_success"] = f"cmd_receive_success_{ref}"
                            commands["cmd_receive_failure"] = f"cmd_receive_failure_{ref}"
                            links_ab.append(commands)

                        if (edge[0] == node_b) and (edge[1] == node_a):
                            # this link uses the channel_ba
                            ref = f"link_{node_b}_{node_a}_of_path_{i}_{session_id}"
                            commands = {}
                            commands["cmd_send_failure"] = f"cmd_send_failure_{ref}"
                            commands["cmd_send_success"] = f"cmd_send_success_{ref}"
                            commands["cmd_receive_success"] = f"cmd_receive_success_{ref}"
                            commands["cmd_receive_failure"] = f"cmd_receive_failure_{ref}"
                            links_ba.append(commands)

            channel["links"] = links_ab
            channel_ba["links"] = links_ba

            channels_dict["instances"].append(channel)
            channels_dict["instances"].append(channel_ba)

        return channels_dict


    def _add_links_to_dict(self):
        with open("config/netgen_files.json", "r") as f:
            files_config = json.load(f)

        with open(files_config.get("ranges"), "r") as f:
            ranges = json.load(f)

        with open(files_config.get("init"), "r") as f:
            init = json.load(f)

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
            for i, path in enumerate(session.paths):
                for edge in path.edges():
                    sender = list(path.nodes)[0]
                    receivers = list(path.nodes)[1:]
                    node_a = edge[0]
                    node_b = edge[1]

                    # links instances
                    link = {}
                    link["name"] = f"link_{node_a}_{node_b}_of_path_{i}_{session_id}.prism"
                    link["#"] = f"{node_a}_{node_b}_{i}_{session_id}"

                    # states
                    link["range_state"] = range_state
                    link["init_state"] = init.get("link_init_state")
                    link["range_phase"] = ranges.get("link_range_phase")
                    link["init_phase"] = init.get("link_init_phase")

                    # references
                    link["ref_channel"] = f"{node_a}_{node_b}"
                    link["ref_interface_sender"] = f"{node_a}_{node_b}"
                    link["ref_node_buffer_sender"] = f"{node_a}"
                    # link["ref_session_path"] = f"{i}_{session_id}"
                    # link["ref_link_ref_counter"] = f"{node_a}_{i}_{session_id}"
                    next_links = []
                    successors = list(path.successors(node_b))
                    for succ in successors:
                        next_links.append({"ref_link_next": f"{node_b}_{succ}_{i}_{session_id}"})
                    link["next_links"] = next_links
                    link["number_next_links"] = len(successors)
                    link["ref_interface_receiver"] = f"{node_b}_{node_a}"
                    link["ref_node_buffer_receiver"] = f"{node_b}"
                    link["size_node_buffer_receiver"] = self.attributes["nodes_buffer_sizes"][node_b]
                    link["ref_channel_bandwidth"] = f"{node_a}_{node_b}"

                    # probabilities
                    if self.attributes.get("link_prob_working_to_error")[0] == self.attributes.get("link_prob_working_to_error")[1]:
                        link["prob_working_to_error"] = self.attributes.get("link_prob_working_to_error")[0]
                    else:
                        link["prob_working_to_error"] = round(self.rng.uniform(self.attributes.get("link_prob_working_to_error")[0], self.attributes.get("link_prob_working_to_error")[1]), 2)
                    if self.attributes.get("link_prob_error_to_working")[0] == self.attributes.get("link_prob_error_to_working")[1]:
                        link["prob_error_to_working"] = self.attributes.get("link_prob_error_to_working")[0]
                    else:
                        link["prob_error_to_working"] = round(self.rng.uniform(self.attributes.get("link_prob_error_to_working")[0], self.attributes.get("link_prob_error_to_working")[1]), 2)
                    
                    if self.attributes.get("link_prob_failure_to_working")[0] == self.attributes.get("link_prob_failure_to_working")[1]:
                        link["prob_failure_to_working"] = self.attributes.get("link_prob_failure_to_working")[0]
                    else:
                        link["prob_failure_to_working"] = round(self.rng.uniform(self.attributes.get("link_prob_failure_to_working")[0], self.attributes.get("link_prob_failure_to_working")[1]), 2)
                    if self.attributes.get("link_prob_retry")[0] == self.attributes.get("link_prob_retry")[1]:
                        link["prob_retry"] = self.attributes.get("link_prob_retry")[0]
                    else:
                        link["prob_retry"] = round(self.rng.uniform(self.attributes.get("link_prob_retry")[0], self.attributes.get("link_prob_retry")[1]), 2)
                    if self.attributes.get("link_prob_sending")[0] == self.attributes.get("link_prob_sending")[1]:
                        link["prob_sending"] = self.attributes.get("link_prob_sending")[0]
                    else:
                        link["prob_sending"] = round(self.rng.uniform(self.attributes.get("link_prob_sending")[0], self.attributes.get("link_prob_sending")[1]), 2)

                    # commands
                    link["cmd_working_to_error"] = f"cmd_working_to_error_link_{node_a}_{node_b}_of_path_{i}_{session_id}"
                    link["cmd_error_to_working"] = f"cmd_error_to_working_link_{node_a}_{node_b}_of_path_{i}_{session_id}"
                    link["cmd_failure_to_working"] = f"cmd_failure_to_working_link_{node_a}_{node_b}_of_path_{i}_{session_id}"
                    link["cmd_send_failure"] = f"cmd_send_failure_link_{node_a}_{node_b}_of_path_{i}_{session_id}"
                    link["cmd_send_success"] = f"cmd_send_success_link_{node_a}_{node_b}_of_path_{i}_{session_id}"
                    link["cmd_sending"] = f"cmd_sending_link_{node_a}_{node_b}_of_path_{i}_{session_id}"
                    link["cmd_receive_success"] = f"cmd_receive_success_link_{node_a}_{node_b}_of_path_{i}_{session_id}"
                    link["cmd_receive_failure"] = f"cmd_receive_failure_link_{node_a}_{node_b}_of_path_{i}_{session_id}"

                    # if the link is the first link of the path, this command is syncronous with the cmd send of the path
                    # if the link is not the first link of the path, this command is syncronous with the cmd receive of the previous link
                    if node_a == sender:
                        link["cmd_link_start"] = f"cmd_send_session_path_{i}_{session_id}"
                    else:
                        predecessor = list(path.predecessors(node_a))[0]
                        link["cmd_link_start"] = f"cmd_receive_success_link_{predecessor}_{node_a}_of_path_{i}_{session_id}"
                    
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
            for i, path in enumerate(session.paths):
                sender = list(path.nodes)[0]
                for node in path.nodes:
                    # link_ref_counter instances
                    link_ref = {}
                    link_ref["name"] = f"link_ref_{node}_of_path_{i}_{session_id}.prism"
                    link_ref["#"] = f"{node}_{i}_{session_id}"
                    
                    if session.protocol in {"hpke", "double_ratchet", "hpke_sender_key", "double_ratchet_sender_key"}:
                        link_ref["size_counter"] = 1
                        link_ref["init_counter"] = 1
                    else:
                        counter = path.nodes[node].get("counter", 1)
                        link_ref["size_counter"] = counter
                        link_ref["init_counter"] = counter
                    link_ref["is_receiver"] = path.nodes[node].get("is_receiver", False)
                    link_ref["ref_node"] = f"{node}"
                    link_ref["cmd_reset"] = f"cmd_reset_link_ref_{node}_of_path_{i}_{session_id}"

                    link_refs_dict["instances"].append(link_ref)
                    

        return link_refs_dict
        

    def _add_session_paths_to_dict(self):

        with open("config/netgen_files.json", "r") as f:
            files_config = json.load(f)

        with open(files_config.get("ranges"), "r") as f:
            ranges = json.load(f)

        with open(files_config.get("consts"), "r") as f:
            consts = json.load(f)

        with open(files_config.get("init"), "r") as f:
            init = json.load(f)

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
        for s in self.attributes.get("sessions"):
            subs = s.get_subsessions()
            sessions.extend(subs)

        for session in sessions:
            session_id = session.get_id()
            for i, path in enumerate(session.paths):
                sender = list(path.nodes)[0]
                receivers = list(path.nodes)[1:]
                session_path = {}

                if session.protocol == "hpke_sender_key" or session.protocol == "double_ratchet_sender_key":
                    session_path["name"] = f"session_path_{i}_{session_id}.prism"
                    session_path["#"] = f"{i}_{session_id}"
                else:
                    session_path["name"] = f"session_path_{i}_{session_id}.prism"
                    session_path["#"] = f"{i}_{session_id}"

                session_path["protocol"] = session.protocol

                # states
                session_path["range_depth"] = range_state
                session_path["init_state"] = init.get("session_path_init_state")
                session_path["range_system_message"] = range_system_message
                session_path["init_system_message"] = init.get("session_path_init_system_message")
                session_path["range_data_message"] = range_data_message
                session_path["init_data_message"] = init.get("session_path_init_data_message")
                session_path["size_prev_local_session_epoch_sender"] = path.nodes[sender]['epoch_size']
                session_path["init_prev_local_session_epoch_sender"] = init.get("session_path_init_prev_local_session_epoch_sender")
                session_path["size_link_counter"] = len(path) - 1
                session_path["init_link_counter"] = len(path) - 1
                session_path["size_checker_counter"] = len(receivers)  # one session checker for each receiver
                session_path["init_checker_counter"] = len(receivers)
                if session.protocol == "sender_key":
                    one_to_one = []
                    subsessions = session.get_subsessions()
                    for ss in subsessions:
                        for j, subpath in enumerate(ss.paths):
                            if list(subpath.nodes)[0] == sender:
                                one_to_one.append({"ref_session_path": f"{j}_{ss.get_id()}", 
                                                   "ref_session_checker_receiver": f"{list(subpath.nodes)[-1]}_{j}_{ss.get_id()}"})
                    session_path["one_to_one"] = one_to_one


                # references
                session_path["ref_node_sender"] = f"{sender}"
                session_path["ref_local_session_sender"] = f"{sender}_{session_id}"
                session_path["size_buffer_sender"] = self.attributes["nodes_buffer_sizes"][sender]
                # first_links = []
                # successors = list(path.successors(sender))
                # for succ in successors:
                #     if session.protocol == "hpke_sender_key" or session.protocol == "double_ratchet_sender_key":
                #         first_links.append({"#": f"{sender}_{succ}_{i}_{session_id}"})
                #     else:
                #         first_links.append({"#": f"{sender}_{succ}_{i}_{session_id}"})
                # session_path["first_links"] = first_links

                session_path["is_broadest"] = "true" #TODO fix for sender key

                # probabilities
                if session.protocol in {"hpke_sender_key", "double_ratchet_sender_key"}:
                    session_path["prob_run"] = 0.0
                else:
                    if self.attributes.get("sp_prob_run")[0] == self.attributes.get("sp_prob_run")[1]:
                        prob_run = self.attributes.get("sp_prob_run")[0]
                    else:
                        prob_run = round(self.rng.uniform(self.attributes.get("sp_prob_run")[0], self.attributes.get("sp_prob_run")[1]), 2)
                    session_path["prob_run"] = prob_run

                # commands
                if session.protocol in {"hpke_sender_key", "double_ratchet_sender_key"}:
                    session_path["cmd_run"] = f"cmd_run_session_path_{i}_{session_id}"
                    session_path["cmd_update_data"] = f"cmd_update_data_session_path_{i}_{session_id}"
                    session_path["cmd_update_failure"] = f"cmd_update_failure_session_path_{i}_{session_id}"
                    session_path["cmd_update_success"] = f"cmd_update_success_session_path_{i}_{session_id}"
                    session_path["cmd_send"] = f"cmd_send_session_path_{i}_{session_id}"
                    session_path["cmd_counter_reset"] = f"cmd_counter_reset_session_path_{i}_{session_id}"
                else:
                    session_path["cmd_run"] = f"cmd_run_session_path_{i}_{session_id}"
                    session_path["cmd_update"] = f"cmd_update_session_path_{i}_{session_id}"
                    session_path["cmd_update_failure"] = f"cmd_update_failure_session_path_{i}_{session_id}"
                    session_path["cmd_update_success"] = f"cmd_update_success_session_path_{i}_{session_id}"
                    session_path["cmd_send"] = f"cmd_send_session_path_{i}_{session_id}"
                    session_path["cmd_counter_reset"] = f"cmd_counter_reset_session_path_{i}_{session_id}"
                if session.protocol == "mls":
                    session_path["cmd_update_data"] = f"cmd_update_data_session_path_{i}_{session_id}"
                    session_path["cmd_update_not_data"] = f"cmd_update_not_data_session_path_{i}_{session_id}"
                    
                if session.protocol == "sender_key":
                    session_path["cmd_alt_run"] = f"cmd_alt_run_session_path_{i}_{session_id}"

                session_paths_dict["instances"].append(session_path)

        return session_paths_dict


    def _add_local_sessions_to_dict(self):
        with open("config/netgen_files.json", "r") as f:
            files_config = json.load(f)

        with open(files_config.get("ranges"), "r") as f:
            ranges = json.load(f)

        with open(files_config.get("consts"), "r") as f:
            consts = json.load(f)

        with open(files_config.get("init"), "r") as f:
            init = json.load(f)

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
                local_session["name"] = f"local_session_of_{node}_of_session_{id}.prism"
                local_session["#"] = f"{node}_{id}"
                local_session["protocol"] = session.protocol

                # states
                local_session["range_state"] = range_state
                local_session["init_state"] = init.get("local_session_init_state")
                local_session["size_epoch"] = list(session.paths)[0].nodes[node]['epoch_size']
                local_session["init_epoch"] = init.get("local_session_init_epoch")
                local_session["size_ratchet"] = list(session.paths)[0].nodes[node]['ratchet_size']
                local_session["init_ratchet"] = init.get("local_session_init_ratchet")
                local_session["init_compromised"] = init.get("local_session_init_compromised")

                # references
                local_session["ref_node"] = f"{node}"

                session_paths = []
                for i, path in enumerate(session.paths):
                    if path.nodes[0] == node:
                        commands = {}
                        commands["cmd_update_data"] = f"cmd_update_data_session_path_{i}_{id}"
                        session_paths.append(commands)

                local_session["session_paths"] = session_paths
                
                if session.protocol == "sender_key" and node != session.paths[0].nodes[0]:
                    for ss in session.get_subsessions():
                        if node in ss.nodes:
                            i = [e[0] for e in enumerate(ss.paths) if node == list(e[1])[0]][0]
                            local_session["ref_broadest_session_path"] = f"{i}_{ss.get_id()}"
                            break
                else:
                    i = [e[0] for e in enumerate(session.paths) if node == list(e[1])[0]][0]
                    local_session["ref_broadest_session_path"] = f"{i}_{id}"

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

                # commands
                local_session["cmd_session_update_success"] = f"cmd_session_update_success_local_session_of_{node}_of_session_{id}"
                local_session["cmd_session_update_failure"] = f"cmd_session_update_failure_local_session_of_{node}_of_session_{id}"
                local_session["cmd_session_ratchet"] = f"cmd_session_ratchet_local_session_of_{node}_of_session_{id}"
                local_session["cmd_session_check"] = f"cmd_session_check_local_session_of_{node}_of_session_{id}"
                local_session["cmd_compromise"] = f"cmd_compromise_local_session_of_{node}_of_session_{id}"
                local_session["cmd_decompromise"] = f"cmd_decompromise_local_session_of_{node}_of_session_{id}"

                local_sessions_dict["instances"].append(local_session)

        return local_sessions_dict


    def _get_receivers_from_path(self, path):
        return [node_id for node_id in path.nodes if path.nodes[node_id].get("is_receiver", False)]

                    
    def _add_session_checkers_to_dict(self):
        # one session checker for each receiver in each path

        with open("config/netgen_files.json", "r") as f:
            files_config = json.load(f)

        with open(files_config.get("ranges"), "r") as f:
            ranges = json.load(f)

        with open(files_config.get("consts"), "r") as f:
            consts = json.load(f)

        with open(files_config.get("init"), "r") as f:
            init = json.load(f)

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
            session_id = session.get_id()
            for i, path in enumerate(session.paths):
                sender = list(path)[0]
                receivers = self._get_receivers_from_path(path)
                
                for node in receivers:
                    session_checker = {}
                    session_checker["name"] = f"session_checker_of_{node}_of_path_{i}_{session_id}.prism"
                    session_checker["#"] = f"{node}_{i}_{session_id}"
                    session_checker["protocol"] = session.protocol

                    # states
                    session_checker["range_state"] = state_range
                    session_checker["init_state"] = init.get("session_checker_init_state")
                    session_checker["size_sender_local_session_epoch"] = path.nodes[sender]['epoch_size']
                    session_checker["init_sender_local_session_epoch"] = init.get("session_checker_init_sender_local_session_epoch")
                    session_checker["range_sender_local_session_system_message"] = session_checker_sender_local_session_system_message_range
                    session_checker["init_sender_local_session_system_message"] = init.get("session_checker_init_sender_local_session_system_message")
                    session_checker["range_sender_local_session_data_message"] = session_checker_sender_local_session_data_message_range
                    session_checker["init_sender_local_session_data_message"] = init.get("session_checker_init_sender_local_session_data_message")

                    # references
                    session_checker["ref_local_session_sender"] = f"{sender}_{session_id}"
                    session_checker["ref_session_path"] = f"{i}_{session_id}"
                    session_checker["ref_local_session_receiver"] = f"{node}_{session_id}"
                    session_checker["ref_node_receiver"] = f"{node}"

                    match session.protocol:
                        case protocol if protocol == "hpke" or protocol == "double_ratchet":
                            session_checker["ref_session_path_to_sender"] = f"{i}_{session_id}"
                            session_checker["size_ratchet_sender"] = sender['ratchet_size']

                            session_checker["cmd_read_reset"] = f"cmd_read_reset_session_checker_of_{node}_of_path_{i}_of_session_{session_id}"
                            session_checker["cmd_read_ratchet"] = f"cmd_read_ratchet_session_checker_of_{node}_of_path_{i}_of_session_{session_id}"


                            session_checker[f"cmd_{protocol}_reset"] = f"cmd_{protocol}_reset_session_checker_of_{node}_of_path_{i}_of_session_{session_id}"
                            session_checker[f"cmd_{protocol}_failure"] = f"cmd_{protocol}_failure_session_checker_of_{node}_of_path_{i}_of_session_{session_id}"

                        case protocol if protocol == "hpke_sender_key" or protocol == "double_ratchet_sender_key":
                            supersession = [s for s in sessions if session_id in [ss.id for ss in s.subsessions]][0]
                            session_checker["ref_session_path_to_sender"] = f"{i}_{session_id}"
                            if node == list(supersession.paths[0])[0]:
                                session_checker["ref_session_checker_sender"] = f"{sender}_0_{supersession.id}"
                            else:
                                session_checker["ref_session_checker_sender"] = f"{node}_0_{supersession.id}"
                            session_checker["ref_session_path_broadcast"] = f"0_{supersession.id}"
                            session_checker["size_ratchet_sender"] = path.nodes[sender]['ratchet_size']
                            session_checker["ref_local_session_sender_key_receiver"] = f"{node}_{supersession.id}"

                            session_checker["cmd_read_reset"] = f"cmd_read_reset_session_checker_of_{node}_of_path_{i}_of_session_{session_id}"
                            session_checker["cmd_read_ratchet"] = f"cmd_read_ratchet_session_checker_of_{node}_of_path_{i}_of_session_{session_id}"
                            session_checker["cmd_resolve_sender_new_key"] = f"cmd_resolve_sender_new_key_session_checker_of_{node}_of_path_0_of_session_{supersession.id}"
                            session_checker["cmd_resolve_sender_refresh"] = f"cmd_resolve_sender_refresh_session_checker_of_{node}_of_path_{i}_of_session_{session_id}"
                            session_checker["cmd_resolve_sender_reset"] = f"cmd_resolve_sender_reset_session_checker_of_{node}_of_path_{i}_of_session_{session_id}"

                            session_checker[f"cmd_{protocol[:-11]}_reset"] = f"cmd_{protocol[:-11]}_reset_session_checker_of_{node}_of_path_{i}_of_session_{session_id}"
                            session_checker[f"cmd_{protocol[:-11]}_failure"] = f"cmd_{protocol[:-11]}_failure_session_checker_of_{node}_of_path_{i}_of_session_{session_id}"


                        case "sender_key":
                            subsession = [ss for ss in session.subsessions if node in ss.nodes][0]
                            path_id = [e[0] for e in enumerate(subsession.paths) if node == list(e[1])[0]][0]
                            session_checker["ref_session_path_to_sender"] = f"{path_id}_{subsession.get_id()}"
                            session_checker["ref_local_session_sender_key_receiver"] = f"{node}_{session_id}"

                            session_checker["cmd_resolve_sender_new_key"] = f"cmd_resolve_sender_new_key_session_checker_of_{node}_of_path_{i}_of_session_{session_id}"
                            session_checker["cmd_sender_key_failure"] = f"cmd_sender_key_failure_session_checker_of_{node}_of_path_{i}_of_session_{session_id}"
                            session_checker["cmd_sender_key_reset"] = f"cmd_sender_key_reset_session_checker_of_{node}_of_path_{i}_of_session_{session_id}"

                        case "mls":
                            session_checker["ref_session_path_broadcast"] = f"{i}_{session_id}"
                            session_checker["size_ratchet_sender"] = sender['ratchet_size']

                            session_checker["cmd_read_reset"] = f"cmd_read_reset_session_checker_of_{node}_of_path_{i}_of_session_{session_id}"
                            session_checker["cmd_read_ratchet"] = f"cmd_read_ratchet_session_checker_of_{node}_of_path_{i}_of_session_{session_id}"
                            session_checker["cmd_read_refresh"] = f"cmd_read_refresh_session_checker_of_{node}_of_path_{i}_of_session_{session_id}"
                            session_checker["cmd_read_current"] = f"cmd_read_current_session_checker_of_{node}_of_path_{i}_of_session_{session_id}"
                            session_checker["cmd_mls_reset"] = f"cmd_mls_reset_session_checker_of_{node}_of_path_{i}_of_session_{session_id}"
                            session_checker["cmd_mls_failure"] = f"cmd_mls_failure_session_checker_of_{node}_of_path_{i}_of_session_{session_id}"


                    # commands
                    session_checker["cmd_freeze"] = f"cmd_send_session_path_{i}_{session_id}" # synchronized with the send command of the session path

                    # here i need the link to the receiver in the path. The path is a tree, so I can use predecessors to find the previous node.
                    prec = next(path.predecessors(node))
                    session_checker["cmd_trigger"] = f"cmd_receive_success_link_{prec}_{node}_of_path_{i}_{session_id}"

                    session_checker["cmd_read_data"] = f"cmd_read_data_session_checker_of_{node}_of_session_{session_id}"
                    session_checker["cmd_update_success"] = f"cmd_update_success_session_checker_of_{node}_of_session_{session_id}"
                    session_checker["cmd_update_failure"] = f"cmd_update_failure_session_checker_of_{node}_of_session_{session_id}"
                    session_checker["cmd_check_success"] = f"cmd_check_success_session_checker_of_{node}_of_session_{session_id}"
                    session_checker["cmd_resolve_data"] = f"cmd_resolve_data_session_checker_of_{node}_of_session_{session_id}"
                    session_checker["cmd_cleanup"] = f"cmd_cleanup_session_checker_of_{node}_of_session_{session_id}"
                    session_checker["cmd_check_failure"] = f"cmd_check_failure_session_checker_of_{node}_of_session_{session_id}"
                    
                    session_checkers_dict["instances"].append(session_checker)

        return session_checkers_dict      


    def _add_rewards_to_dict(self):
        with open("config/netgen_files.json", "r") as f:
            files_config = json.load(f)
            
        rewards_dict = {}
        rewards_dict["name"] = "rewards_modules"
        rewards_dict["template"] = files_config.get("rewards_template")
        rewards_dict["instances"] = []
        for reward in self.attributes.get("rewards", []):
            instance = {}
            instance["name"] = f"{reward.name}.prism"
            instance["#"] = reward.name
            instance["contributions"] = []
            for contribution in reward.contributions:
                contrib_dict = {}
                contrib_dict["command"] = contribution.command
                contrib_dict["condition"] = contribution.condition
                contrib_dict["value"] = contribution.value
                instance["contributions"].append(contrib_dict)
            rewards_dict["instances"].append(instance)
        
        return rewards_dict


