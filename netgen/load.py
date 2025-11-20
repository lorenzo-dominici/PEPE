# This module contains the logic to load and structure data from files.
import json
import random

NET_TYPES = {"STAR", "RING", "CHAIN", "TREE", "MESH", "FULLY_CONNECTED", "REGULAR_GRAPH", "CUSTOM"}
PROTOCOLS = {"HPKE", "DOUBLE_RATCHET", "SENDER_KEY", "MLS"}

def load_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
        network_params = validate_data(data)
        if network_params:
            print("Network parameters successfully validated.")
            return network_params
    
def validate_data(data):

    seed = data.get("seed")
    if seed:
        if not isinstance(seed, int):
            raise ValueError("seed must be an integer")
   
    node_range = data.get("node_range")
    if node_range:
        if (not isinstance(node_range, list) or len(node_range) != 2 or
            not all(isinstance(n, int) for n in node_range) or
            node_range[0] <= 0 or node_range[1] < node_range[0]):
            raise ValueError("node_range must be a list of two positive integers [min, max] with min <= max")
    else:
        raise ValueError("Missing parameter: node_range")
    
    node_number = random.randint(node_range[0], node_range[1])

    connected = data.get("connected")
    if connected:
        if not(isinstance(connected, bool)):
            raise ValueError("connected must be a bool")
        
    gen_model = data.get("gen_model")
    if gen_model:
        if gen_model not in {"RANDOM", "SMART-WORLD", "SCALE-FREE"}:
            raise ValueError("gen_model must be one of RANDOM, SMART-WORLD, SCALE-FREE")
        

    # RANDOM GRAPH PARAMETERS
        
    conn_prob = data.get("conn_prob")
    if conn_prob:
        if not (isinstance(conn_prob, float) and 0.0 <= conn_prob <= 1.0):
            raise ValueError("conn_prob must be a float between 0.0 and 1.0")
        
    degree_distr = data.get("degree_distr")
    if degree_distr:
        type = degree_distr.get("type")
        if type:
            if type not in {"BINOMIAL", "UNIFORM", "NORMAL", "POWERLAW", "CUSTOM"}:
                raise ValueError("degree_distr.type must be one of BINOMIAL, UNIFORM, NORMAL, POWER_LAW, CUSTOM")
            params = degree_distr.get("params")
            if params:
                if not isinstance(params, list):
                    if type == "CUSTOM":
                        if not(isinstance(params, str)):
                            raise ValueError("For CUSTOM, degree_distr.params must be a list of non-negative integers or a string")
                        if isinstance(params, str) and not isinstance(eval(params), list):
                            raise ValueError("invalid params for the CUSTOM distribuiton")
                    else:
                        raise ValueError("degree_distr.params must be a list of parameters")
                if type == "BINOMIAL":
                    if len(params) != 2 or not(isinstance(params[0], int)) or not(isinstance(params[1], float)):
                        raise ValueError("For BINOMIAL, degree_distr.params must be [n: int, p: float]")
                    if not(1 <= params[0] <= node_range[1] - 1) or not(0.0 <= params[1] <= 1.0):
                        raise ValueError("For BINOMIAL, degree_distr.params must be [1 <= n <= node_range[max] - 1, 0.0 <= p <= 1.0]")
                elif type == "UNIFORM":
                    if len(params) != 2 or not all(isinstance(p, int) for p in params):
                        raise ValueError("For UNIFORM, degree_distr.params must be [min: int, max: int]")
                    if params[0] < 0 or params[1] > node_range[1]:
                        raise ValueError("For UNIFORM, degree_distr.params must be [min >= 0, max <= node_range[max]]")
                elif type == "NORMAL":
                    if len(params) != 2 or not all(isinstance(p, int) for p in params):
                        raise ValueError("For NORMAL, degree_distr.params must be [mean: int, stddev: int]")
                    if not(0 <= params[0] <= node_range[1]) or (params[1] < 0):
                        raise ValueError("For NORMAL, degree_distr.params must be [0 <= mean <= node_range[max], stddev >= 0]")
                elif type == "POWERLAW":
                    if len(params) != 2 or not all(isinstance(p, (int, float)) for p in params):
                        raise ValueError("For POWERLAW, degree_distr.params must be [exponent: float, min: float]")
                    if params[0] <= 1 or params[1] > node_range[1]:
                        raise ValueError("For POWERLAW, degree_distr.params must be [exponent > 1: , 0 <= min <= node_range[max]]")
            else:
                raise ValueError("Missing parameter: degree_distr.params")
        else:
            raise ValueError("Missing parameter: degree_distr.type")
        
    if_range = data.get("if_range")
    if if_range:
        if (not isinstance(if_range, list) or len(if_range) != 2 or
            not all(isinstance(n, int) for n in if_range) or
            if_range[0] <= 0 or if_range[1] < if_range[0]):
            raise ValueError("if_range must be a list of two positive integers [min, max] with min <= max")
        

    # SMART-WORLD GRAPH PARAMETERS

    mean_degree_range = data.get("mean_degree_range")
    if mean_degree_range:
        if (not isinstance(mean_degree_range, list) or len(mean_degree_range) != 2 or
            not all(isinstance(n, int) for n in mean_degree_range) or
            mean_degree_range[0] <= 0 or mean_degree_range[1] < mean_degree_range[0]):
            raise ValueError("mean_degree_range must be a list of two positive integers [min, max] with min <= max")  
        
    rewiring_prob = data.get("rewiring_prob")
    if rewiring_prob:
        if not (isinstance(rewiring_prob, float) and 0.0 <= rewiring_prob <= 1.0):
            raise ValueError("rewiring_prob must be a float between 0.0 and 1.0")
        
    delete_rewired = data.get("delete_rewired")
    if delete_rewired:
        if not(isinstance(delete_rewired, bool)):
            raise ValueError("delete_rewired must be a bool")
        

    # CLUSTERING PARAMETERS
        
    local_clustering_coeff = data.get("local_clustering_coeff") 
    if local_clustering_coeff:
        if not (isinstance(local_clustering_coeff, float) and 0.0 <= local_clustering_coeff <= 1.0):
            raise ValueError("local_clustering_coeff must be a float between 0.0 and 1.0")
        
    clusters_number_range = data.get("clusters_number_range")
    if clusters_number_range:
        if (not isinstance(clusters_number_range, list) or len(clusters_number_range) != 2 or
            not all(isinstance(n, int) for n in clusters_number_range) or
            clusters_number_range[0] <= 0 or clusters_number_range[1] < clusters_number_range[0]):
            raise ValueError("clusters_number_range must be a list of two positive integers [min, max] with min <= max")
        if clusters_number_range[1] > node_number:
            raise ValueError("The maximum of clusters_number_range cannot exceed the total number of nodes")
        cluster_number = random.randint(clusters_number_range[0], clusters_number_range[1])
        
    nodes_range_per_cluster = data.get("nodes_range_per_cluster")
    if nodes_range_per_cluster:
        if (not isinstance(nodes_range_per_cluster, list) or len(nodes_range_per_cluster) != 2 or
            not all(isinstance(n, int) for n in nodes_range_per_cluster) or
            nodes_range_per_cluster[0] <= 0 or nodes_range_per_cluster[1] < nodes_range_per_cluster[0]):
            raise ValueError("nodes_range_per_cluster must be a list of two positive integers [min, max] with min <= max")
        if nodes_range_per_cluster[1] > node_number:
            raise ValueError("The maximum of nodes_range_per_cluster cannot exceed the total number of nodes")
        
    inter_clusters_coeff = data.get("inter_clusters_coeff")
    if inter_clusters_coeff:
        if not (isinstance(inter_clusters_coeff, float) and 0.0 <= inter_clusters_coeff <= 1.0):
            raise ValueError("inter_clusters_coeff must be a float between 0.0 and 1.0")
    
    central_nodes_range = data.get("central_nodes_range")
    if central_nodes_range:
        if (not isinstance(central_nodes_range, list) or len(central_nodes_range) != 2 or
            not all(isinstance(n, int) for n in central_nodes_range) or
            central_nodes_range[0] <= 0 or central_nodes_range[1] < central_nodes_range[0]):
            raise ValueError("central_nodes_range must be a list of two positive integers [min, max] with min <= max")
        if central_nodes_range[1] > node_number:
            raise ValueError("The maximum of central_nodes_range cannot exceed the total number of nodes")
        
    central_nodes_min_degree = data.get("central_nodes_min_degree")
    if central_nodes_min_degree:
        if not (isinstance(central_nodes_min_degree, int) and 0 < central_nodes_min_degree < node_number):
            raise ValueError("central_nodes_min_degree must be a positive integer less than the total number of nodes")
        
    edge_per_new_node = data.get("edge_per_new_node")
    if edge_per_new_node:
        if not (isinstance(edge_per_new_node, int) and 0 < edge_per_new_node < node_number):
            raise ValueError("edge_per_new_node must be a positive integer less than the total number of nodes")
        
    buffer_size_range = data.get("buffer_size_range")
    if buffer_size_range:
           if (not isinstance(buffer_size_range, list) or len(buffer_size_range) != 2 or
                not all(isinstance(n, int) for n in buffer_size_range) or
                buffer_size_range[0] <= 0 or buffer_size_range[1] < buffer_size_range[0]):
                raise ValueError("buffer_size_range must be a list of two positive integers [min, max] with min <= max")  
    else:
        raise ValueError("Missing parameter: buffer_size_range")
    
    central_nodes_buffer_size = data.get("central_nodes_buffer_size")
    if central_nodes_buffer_size:
        if not (isinstance(central_nodes_buffer_size, int) and central_nodes_buffer_size > 0):
            raise ValueError("central_nodes_buffer_size must be a positive integer")
    else:
        central_nodes_buffer_size = buffer_size_range  # default to buffer_size_range if not provided

    channel_bandwidth_range = data.get("channel_bandwidth_range")
    if channel_bandwidth_range:
        if (not isinstance(channel_bandwidth_range, list) or len(channel_bandwidth_range) != 2 or
            not all(isinstance(n, int) for n in channel_bandwidth_range) or
            channel_bandwidth_range[0] <= 0 or channel_bandwidth_range[1] < channel_bandwidth_range[0]):
            raise ValueError("channel_bandwidth_range must be a list of two positive integers [min, max] with min <= max")
    else:
        raise ValueError("Missing parameter: channel_bandwidth_range")

    
    network_params = {
        "seed": seed,
        "node_number": node_number,
        "connected": connected,
        "conn_prob": conn_prob,
        "degree_distr": degree_distr,
        "if_range": if_range,
        "rewiring_prob": rewiring_prob,
        "local_clustering_coeff": local_clustering_coeff,
        "clusters_number_range": clusters_number_range,
        "nodes_range_per_cluster": nodes_range_per_cluster,
        "inter_clusters_coeff": inter_clusters_coeff,
        "central_nodes_range": central_nodes_range,
        "central_nodes_min_degree": central_nodes_min_degree,
        "edge_per_new_node": edge_per_new_node,
        "buffer_size_range": buffer_size_range,
        "central_nodes_buffer_size": central_nodes_buffer_size,
        "channel_bandwidth_range": channel_bandwidth_range
    }

    return network_params

