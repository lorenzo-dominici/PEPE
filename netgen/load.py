# This module contains the logic to load and structure data from files.
from fileinput import filename
import json
import random

NET_TYPES = {"STAR", "RING", "CHAIN", "TREE", "MESH", "FULLY_CONNECTED", "REGULAR_GRAPH", "CUSTOM"}
PROTOCOLS = {"hpke", "double_ratchet", "sender_key", "mls"}

def load_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
        network_params = validate_data(data)
        if network_params:
            print("Network parameters successfully validated.")
            return network_params
    
def validate_data(data):

    protocol = data.get("protocol")
    if protocol:
        if protocol not in {"hpke", "double_ratchet", "sender_key", "mls"}:
                raise ValueError("protocol must be one of hpke, double_ratchet, sender_key, mls")
    else:
        raise ValueError("protocol parameter must be setted")

    if protocol == "mls":
        mls_sessions_range = data.get("mls_sessions_range")
        if mls_sessions_range:
            if (not isinstance(mls_sessions_range, list) or len(mls_sessions_range) != 2 or
                not all(isinstance(n, int) for n in mls_sessions_range) or
                mls_sessions_range[0] <= 0 or mls_sessions_range[1] < mls_sessions_range[0]):
                raise ValueError("mls_sessions_range must be a list of two positive integers [min, max] with min <= max")
        else:
            raise ValueError("Missing parameter: mls_sessions_range")
    else:
        mls_sessions_range = None  # default to None if not provided and not needed
        
    

    support_protocol = data.get("support_protocol")
    if support_protocol:
        if support_protocol not in {"hpke_sender_key", "double_ratchet_sender_key"}:
            raise ValueError("support_protocol must be one of hpke_sender_key, double_ratchet_sender_key")
    else:
        support_protocol = "hpke_sender_key"  # default to hpke if not provided
            
    
    seed = data.get("seed")
    if seed:
        if not isinstance(seed, int):
            raise ValueError("seed must be an integer")

    
    filename = data.get("filename")
   
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
        

    # SCALE-FREE PARAMETERS

    initial_degree_range = data.get("initial_degree_range")
    if initial_degree_range:
        if (not isinstance(initial_degree_range, list) or len(initial_degree_range) != 2 or
            not all(isinstance(n, int) for n in initial_degree_range) or
            initial_degree_range[0] <= 0 or initial_degree_range[1] < initial_degree_range[0]):
            raise ValueError("initial_degree_range must be a list of two positive integers [min, max] with min <= max")  
        if initial_degree_range[0] > node_range[1]:
            raise ValueError("min initial_degree_range must be lower than max number of nodes")
        
    new_edges_prob = data.get("new_edges_prob")
    if new_edges_prob:
        if not (isinstance(new_edges_prob, float) and 0.0 <= new_edges_prob <= 1.0):
            raise ValueError("new_edges_prob must be a float between 0.0 and 1.0")
        if rewiring_prob:
            if rewiring_prob + new_edges_prob >= 1.0:
                raise ValueError("the sum of rewiring_prob and new_edges_prob must be lower than 1")

    # CLUSTERING PARAMETERS

    clusters_number_range = data.get("clusters_number_range")
    if clusters_number_range:
        if (not isinstance(clusters_number_range, list) or len(clusters_number_range) != 2 or
            not all(isinstance(n, int) for n in clusters_number_range) or
            clusters_number_range[0] <= 0 or clusters_number_range[1] < clusters_number_range[0]):
            raise ValueError("clusters_number_range must be a list of two positive integers [min, max] with min <= max")
    
    nodes_range_per_cluster = data.get("nodes_range_per_cluster")
    if nodes_range_per_cluster:
        if (not isinstance(nodes_range_per_cluster, list) or len(nodes_range_per_cluster) != 2 or
            not all(isinstance(n, int) for n in nodes_range_per_cluster) or
            nodes_range_per_cluster[0] <= 0 or nodes_range_per_cluster[1] < nodes_range_per_cluster[0]):
            raise ValueError("nodes_range_per_cluster must be a list of two positive integers [min, max] with min <= max")
    
    inter_clusters_coeff = data.get("inter_clusters_coeff")
    if inter_clusters_coeff:
        if not (isinstance(inter_clusters_coeff, float) and 0.0 <= inter_clusters_coeff <= 1.0):
            raise ValueError("inter_clusters_coeff must be a float between 0.0 and 1.0")
        
    # central_nodes_min_degree = data.get("central_nodes_min_degree")
    # if central_nodes_min_degree:
    #     if not (isinstance(central_nodes_min_degree, int) and 0 < central_nodes_min_degree < node_number):
    #         raise ValueError("central_nodes_min_degree must be a positive integer less than the total number of nodes")
        
    
    # ATTRIBUTES

    # NODE PARAMETERS
    
    buffer_size_range = data.get("buffer_size_range")
    if buffer_size_range:
           if (not isinstance(buffer_size_range, list) or len(buffer_size_range) != 2 or
                not all(isinstance(n, int) for n in buffer_size_range) or
                buffer_size_range[0] <= 0 or buffer_size_range[1] < buffer_size_range[0]):
                raise ValueError("buffer_size_range must be a list of two positive integers [min, max] with min <= max")  
    else:
        raise ValueError("Missing parameter: buffer_size_range")

    node_prob_off_to_on = data.get("node_prob_off_to_on")
    if node_prob_off_to_on:
        if (not isinstance(node_prob_off_to_on, list) or len(node_prob_off_to_on) != 2 or
            not all(isinstance(p, float) for p in node_prob_off_to_on) or
            not all(0.0 <= p <= 1.0 for p in node_prob_off_to_on) or
            node_prob_off_to_on[0] > node_prob_off_to_on[1]):
            raise ValueError("node_prob_off_to_on must be a list of two floats [min, max] with 0.0 <= min <= max <= 1.0")
    else:
        raise ValueError("Missing parameter: node_prob_off_to_on")

    node_prob_on_to_off = data.get("node_prob_on_to_off")
    if node_prob_on_to_off:
        if (not isinstance(node_prob_on_to_off, list) or len(node_prob_on_to_off) != 2 or
            not all(isinstance(p, float) for p in node_prob_on_to_off) or
            not all(0.0 <= p <= 1.0 for p in node_prob_on_to_off) or
            node_prob_on_to_off[0] > node_prob_on_to_off[1]):
            raise ValueError("node_prob_on_to_off must be a list of two floats [min, max] with 0.0 <= min <= max <= 1.0")
    else:
        raise ValueError("Missing parameter: node_prob_on_to_off")
    
    # central_nodes_buffer_size = data.get("central_nodes_buffer_size")
    # if central_nodes_buffer_size:
    #     if not (isinstance(central_nodes_buffer_size, int) and central_nodes_buffer_size > 0):
    #         raise ValueError("central_nodes_buffer_size must be a positive integer")
    # else:
    #     central_nodes_buffer_size = buffer_size_range  # default to buffer_size_range if not provided


    # CHANNEL PARAMETERS

    channel_bandwidth_range = data.get("channel_bandwidth_range")
    if channel_bandwidth_range:
        if (not isinstance(channel_bandwidth_range, list) or len(channel_bandwidth_range) != 2 or
            not all(isinstance(n, int) for n in channel_bandwidth_range) or
            channel_bandwidth_range[0] <= 0 or channel_bandwidth_range[1] < channel_bandwidth_range[0]):
            raise ValueError("channel_bandwidth_range must be a list of two positive integers [min, max] with min <= max")
    else:
        raise ValueError("Missing parameter: channel_bandwidth_range")
    

    path_perc = data.get("path_perc")
    if path_perc:
        if not (isinstance(path_perc, float) and 0.0 <= path_perc <= 1.0):
            raise ValueError("path_perc must be a float between 0.0 and 1.0")
        

    channel_prob_working_to_error = data.get("channel_prob_working_to_error")
    if channel_prob_working_to_error:
        if (not isinstance(channel_prob_working_to_error, list) or len(channel_prob_working_to_error) != 2 or
            not all(isinstance(p, float) for p in channel_prob_working_to_error) or
            not all(0.0 <= p <= 1.0 for p in channel_prob_working_to_error) or
            channel_prob_working_to_error[0] > channel_prob_working_to_error[1]):
            raise ValueError("channel_prob_working_to_error must be a list of two floats [min, max] with 0.0 <= min <= max <= 1.0")
    else:
        raise ValueError("Missing parameter: channel_prob_working_to_error")

    channel_prob_error_to_working = data.get("channel_prob_error_to_working")
    if channel_prob_error_to_working:
        if (not isinstance(channel_prob_error_to_working, list) or len(channel_prob_error_to_working) != 2 or
            not all(isinstance(p, float) for p in channel_prob_error_to_working) or
            not all(0.0 <= p <= 1.0 for p in channel_prob_error_to_working) or
            channel_prob_error_to_working[0] > channel_prob_error_to_working[1]):
            raise ValueError("channel_prob_error_to_working must be a list of two floats [min, max] with 0.0 <= min <= max <= 1.0")
    else:
        raise ValueError("Missing parameter: channel_prob_error_to_working")

    channel_prob_failure_to_working = data.get("channel_prob_failure_to_working")
    if channel_prob_failure_to_working:
        if (not isinstance(channel_prob_failure_to_working, list) or len(channel_prob_failure_to_working) != 2 or
            not all(isinstance(p, float) for p in channel_prob_failure_to_working) or
            not all(0.0 <= p <= 1.0 for p in channel_prob_failure_to_working) or
            channel_prob_failure_to_working[0] > channel_prob_failure_to_working[1]):
            raise ValueError("channel_prob_failure_to_working must be a list of two floats [min, max] with 0.0 <= min <= max <= 1.0")
    else:
        raise ValueError("Missing parameter: channel_prob_failure_to_working")


    # INTERFACE PARAMETERS

    if_prob_off_to_working = data.get("if_prob_off_to_working")
    if if_prob_off_to_working:
        if (not isinstance(if_prob_off_to_working, list) or len(if_prob_off_to_working) != 2 or
            not all(isinstance(p, float) for p in if_prob_off_to_working) or
            not all(0.0 <= p <= 1.0 for p in if_prob_off_to_working) or
            if_prob_off_to_working[0] > if_prob_off_to_working[1]):
            raise ValueError("if_prob_off_to_working must be a list of two floats [min, max] with 0.0 <= min <= max <= 1.0")
    else:
        raise ValueError("Missing parameter: if_prob_off_to_working")

    if_prob_off_to_error = data.get("if_prob_off_to_error")
    if if_prob_off_to_error:
        if (not isinstance(if_prob_off_to_error, list) or len(if_prob_off_to_error) != 2 or
            not all(isinstance(p, float) for p in if_prob_off_to_error) or
            not all(0.0 <= p <= 1.0 for p in if_prob_off_to_error) or
            if_prob_off_to_error[0] > if_prob_off_to_error[1]):
            raise ValueError("if_prob_off_to_error must be a list of two floats [min, max] with 0.0 <= min <= max <= 1.0")
    else:
        raise ValueError("Missing parameter: if_prob_off_to_error")

    if_prob_off_to_failure = data.get("if_prob_off_to_failure")
    if if_prob_off_to_failure:
        if (not isinstance(if_prob_off_to_failure, list) or len(if_prob_off_to_failure) != 2 or
            not all(isinstance(p, float) for p in if_prob_off_to_failure) or
            not all(0.0 <= p <= 1.0 for p in if_prob_off_to_failure) or
            if_prob_off_to_failure[0] > if_prob_off_to_failure[1]):
            raise ValueError("if_prob_off_to_failure must be a list of two floats [min, max] with 0.0 <= min <= max <= 1.0")
    else:
        raise ValueError("Missing parameter: if_prob_off_to_failure")


    if_prob_working_to_error = data.get("if_prob_working_to_error")
    if if_prob_working_to_error:
        if (not isinstance(if_prob_working_to_error, list) or len(if_prob_working_to_error) != 2 or
            not all(isinstance(p, float) for p in if_prob_working_to_error) or
            not all(0.0 <= p <= 1.0 for p in if_prob_working_to_error) or
            if_prob_working_to_error[0] > if_prob_working_to_error[1]):
            raise ValueError("if_prob_working_to_error must be a list of two floats [min, max] with 0.0 <= min <= max <= 1.0")
    else:
        raise ValueError("Missing parameter: if_prob_working_to_error")

    if_prob_error_to_working = data.get("if_prob_error_to_working")
    if if_prob_error_to_working:
        if (not isinstance(if_prob_error_to_working, list) or len(if_prob_error_to_working) != 2 or
            not all(isinstance(p, float) for p in if_prob_error_to_working) or
            not all(0.0 <= p <= 1.0 for p in if_prob_error_to_working) or
            if_prob_error_to_working[0] > if_prob_error_to_working[1]):
            raise ValueError("if_prob_error_to_working must be a list of two floats [min, max] with 0.0 <= min <= max <= 1.0")
    else:
        raise ValueError("Missing parameter: if_prob_error_to_working")

    if_prob_failure_to_working = data.get("if_prob_failure_to_working")
    if if_prob_failure_to_working:
        if (not isinstance(if_prob_failure_to_working, list) or len(if_prob_failure_to_working) != 2 or
            not all(isinstance(p, float) for p in if_prob_failure_to_working) or
            not all(0.0 <= p <= 1.0 for p in if_prob_failure_to_working) or
            if_prob_failure_to_working[0] > if_prob_failure_to_working[1]):
            raise ValueError("if_prob_failure_to_working must be a list of two floats [min, max] with 0.0 <= min <= max <= 1.0")
    else:
        raise ValueError("Missing parameter: if_prob_failure_to_working")

    # Controllo che la somma delle probabilità interface sia matematicamente valida
    min_sum = if_prob_off_to_working[0] + if_prob_off_to_error[0] + if_prob_off_to_failure[0]
    max_sum = if_prob_off_to_working[1] + if_prob_off_to_error[1] + if_prob_off_to_failure[1]
    
    if min_sum >= 1.0:
        raise ValueError("the sum of minimum values of if_prob_off_to_working, if_prob_off_to_error and if_prob_off_to_failure must be lower than 1.0")
    
    if max_sum >= 1.0:
        raise ValueError("the sum of maximum values of if_prob_off_to_working, if_prob_off_to_error and if_prob_off_to_failure must be lower than 1.0")


    # LINK PARAMETERS

    link_prob_working_to_error = data.get("link_prob_working_to_error")
    if link_prob_working_to_error:
        if (not isinstance(link_prob_working_to_error, list) or len(link_prob_working_to_error) != 2 or
            not all(isinstance(p, float) for p in link_prob_working_to_error) or
            not all(0.0 <= p <= 1.0 for p in link_prob_working_to_error) or
            link_prob_working_to_error[0] > link_prob_working_to_error[1]):
            raise ValueError("link_prob_working_to_error must be a list of two floats [min, max] with 0.0 <= min <= max <= 1.0")
    else:
        raise ValueError("Missing parameter: link_prob_working_to_error")

    link_prob_error_to_working = data.get("link_prob_error_to_working")
    if link_prob_error_to_working:
        if (not isinstance(link_prob_error_to_working, list) or len(link_prob_error_to_working) != 2 or
            not all(isinstance(p, float) for p in link_prob_error_to_working) or
            not all(0.0 <= p <= 1.0 for p in link_prob_error_to_working) or
            link_prob_error_to_working[0] > link_prob_error_to_working[1]):
            raise ValueError("link_prob_error_to_working must be a list of two floats [min, max] with 0.0 <= min <= max <= 1.0")
    else:
        raise ValueError("Missing parameter: link_prob_error_to_working")

    link_prob_failure_to_working = data.get("link_prob_failure_to_working")
    if link_prob_failure_to_working:
        if (not isinstance(link_prob_failure_to_working, list) or len(link_prob_failure_to_working) != 2 or
            not all(isinstance(p, float) for p in link_prob_failure_to_working) or
            not all(0.0 <= p <= 1.0 for p in link_prob_failure_to_working) or
            link_prob_failure_to_working[0] > link_prob_failure_to_working[1]):
            raise ValueError("link_prob_failure_to_working must be a list of two floats [min, max] with 0.0 <= min <= max <= 1.0")
    else:
        raise ValueError("Missing parameter: link_prob_failure_to_working")

    link_prob_retry = data.get("link_prob_retry")
    if link_prob_retry:
        if (not isinstance(link_prob_retry, list) or len(link_prob_retry) != 2 or
            not all(isinstance(p, float) for p in link_prob_retry) or
            not all(0.0 <= p <= 1.0 for p in link_prob_retry) or
            link_prob_retry[0] > link_prob_retry[1]):
            raise ValueError("link_prob_retry must be a list of two floats [min, max] with 0.0 <= min <= max <= 1.0")
    else:
        raise ValueError("Missing parameter: link_prob_retry")

    link_prob_sending = data.get("link_prob_sending")
    if link_prob_sending:       
        if (not isinstance(link_prob_sending, list) or len(link_prob_sending) != 2 or
            not all(isinstance(p, float) for p in link_prob_sending) or
            not all(0.0 <= p <= 1.0 for p in link_prob_sending) or
            link_prob_sending[0] > link_prob_sending[1]):
            raise ValueError("link_prob_sending must be a list of two floats [min, max] with 0.0 <= min <= max <= 1.0")
    else:
        raise ValueError("Missing parameter: link_prob_sending")


    # LOCAL SESSION PARAMETERS

    ls_size_epoch_range = data.get("ls_size_epoch_range")
    if ls_size_epoch_range:
        if (not isinstance(ls_size_epoch_range, list) or len(ls_size_epoch_range) != 2 or
            not all(isinstance(n, int) for n in ls_size_epoch_range) or
            ls_size_epoch_range[0] <= 0 or ls_size_epoch_range[1] < ls_size_epoch_range[0]):
            raise ValueError("ls_size_epoch_range must be a list of two positive integers [min, max] with min <= max")
    else:
        raise ValueError("Missing parameter: ls_size_epoch_range")      

    ls_size_ratchet_range = data.get("ls_size_ratchet_range")
    if ls_size_ratchet_range:
        if (not isinstance(ls_size_ratchet_range, list) or len(ls_size_ratchet_range) != 2 or
            not all(isinstance(n, int) for n in ls_size_ratchet_range) or
            ls_size_ratchet_range[0] <= 0 or ls_size_ratchet_range[1] < ls_size_ratchet_range[0]):
            raise ValueError("ls_size_ratchet_range must be a list of two positive integers [min, max] with min <= max")
    else:
        raise ValueError("Missing parameter: ls_size_ratchet_range")

    ls_prob_session_reset = data.get("ls_prob_session_reset")
    if ls_prob_session_reset:
        if (not isinstance(ls_prob_session_reset, list) or len(ls_prob_session_reset) != 2 or
            not all(isinstance(p, float) for p in ls_prob_session_reset) or
            not all(0.0 <= p <= 1.0 for p in ls_prob_session_reset) or
            ls_prob_session_reset[0] > ls_prob_session_reset[1]):
            raise ValueError("ls_prob_session_reset must be a list of two floats [min, max] with 0.0 <= min <= max <= 1.0")
    else:
        raise ValueError("Missing parameter: ls_prob_session_reset")

    ls_prob_ratchet_reset = data.get("ls_prob_ratchet_reset")
    if ls_prob_ratchet_reset:
        if (not isinstance(ls_prob_ratchet_reset, list) or len(ls_prob_ratchet_reset) != 2 or
            not all(isinstance(p, float) for p in ls_prob_ratchet_reset) or
            not all(0.0 <= p <= 1.0 for p in ls_prob_ratchet_reset) or
            ls_prob_ratchet_reset[0] > ls_prob_ratchet_reset[1]):
            raise ValueError("ls_prob_ratchet_reset must be a list of two floats [min, max] with 0.0 <= min <= max <= 1.0")
    else:
        raise ValueError("Missing parameter: ls_prob_ratchet_reset")


    ls_prob_none = data.get("ls_prob_none")
    if ls_prob_none:
        if (not isinstance(ls_prob_none, list) or len(ls_prob_none) != 2 or
            not all(isinstance(p, float) for p in ls_prob_none) or
            not all(0.0 <= p <= 1.0 for p in ls_prob_none) or
            ls_prob_none[0] > ls_prob_none[1]):
            raise ValueError("ls_prob_none must be a list of two floats [min, max] with 0.0 <= min <= max <= 1.0")
    else:
        raise ValueError("Missing parameter: ls_prob_none")
    
    # the sum of ls_prob_none, ls_prob_session_reset and ls_prob_ratchet_reset must be lower than 1.0
    min_sum = ls_prob_none[0] + ls_prob_session_reset[0] + ls_prob_ratchet_reset[0]
    if min_sum >= 1.0:
        raise ValueError("the sum of minimum values of ls_prob_none, ls_prob_session_reset and ls_prob_ratchet_reset must be lower than 1.0")

    ls_prob_compromised = data.get("ls_prob_compromised")
    if ls_prob_compromised:
        if (not isinstance(ls_prob_compromised, list) or len(ls_prob_compromised) != 2 or
            not all(isinstance(p, float) for p in ls_prob_compromised) or
            not all(0.0 <= p <= 1.0 for p in ls_prob_compromised) or
            ls_prob_compromised[0] > ls_prob_compromised[1]):
            raise ValueError("ls_prob_compromised must be a list of two floats [min, max] with 0.0 <= min <= max <= 1.0")
    else:
        raise ValueError("Missing parameter: ls_prob_compromised")


    # SESSION PATH PARAMETERS

    sp_prob_run = data.get("sp_prob_run")
    if sp_prob_run:
        if (not isinstance(sp_prob_run, list) or len(sp_prob_run) != 2 or
            not all(isinstance(p, float) for p in sp_prob_run) or
            not all(0.0 <= p <= 1.0 for p in sp_prob_run) or
            sp_prob_run[0] > sp_prob_run[1]):
            raise ValueError("sp_prob_run must be a list of two floats [min, max] with 0.0 <= min <= max <= 1.0")
    else:
        raise ValueError("Missing parameter: sp_prob_run")


    network_params = {
        "protocol": protocol,
        "support_protocol": support_protocol,
        "mls_sessions_range": mls_sessions_range,
        "seed": seed,
        "filename": filename,
        "node_number": node_number,
        "connected": connected,
        "gen_model": gen_model,
        "conn_prob": conn_prob,
        "degree_distr": degree_distr,
        "if_range": if_range,
        "mean_degree_range": mean_degree_range,
        "rewiring_prob": rewiring_prob,
        "delete_rewired": delete_rewired,
        "initial_degree_range": initial_degree_range,
        "new_edges_prob": new_edges_prob,
        "clusters_number_range": clusters_number_range,
        "nodes_range_per_cluster": nodes_range_per_cluster,
        "inter_clusters_coeff": inter_clusters_coeff,
        "buffer_size_range": buffer_size_range,
        "node_prob_off_to_on": node_prob_off_to_on,
        "node_prob_on_to_off": node_prob_on_to_off,
        "channel_bandwidth_range": channel_bandwidth_range,
        "path_perc": path_perc,
        "channel_prob_working_to_error": channel_prob_working_to_error,
        "channel_prob_error_to_working": channel_prob_error_to_working,
        "channel_prob_failure_to_working": channel_prob_failure_to_working,
        "if_prob_off_to_working": if_prob_off_to_working,
        "if_prob_off_to_error": if_prob_off_to_error,
        "if_prob_off_to_failure": if_prob_off_to_failure,
        "if_prob_working_to_error": if_prob_working_to_error,
        "if_prob_error_to_working": if_prob_error_to_working,
        "if_prob_failure_to_working": if_prob_failure_to_working,
        "link_prob_working_to_error": link_prob_working_to_error,
        "link_prob_error_to_working": link_prob_error_to_working,
        "link_prob_failure_to_working": link_prob_failure_to_working,
        "link_prob_retry": link_prob_retry,
        "link_prob_sending": link_prob_sending,
        "ls_size_epoch_range": ls_size_epoch_range,
        "ls_size_ratchet_range": ls_size_ratchet_range,
        "ls_prob_session_reset": ls_prob_session_reset,
        "ls_prob_ratchet_reset": ls_prob_ratchet_reset,
        "ls_prob_none": ls_prob_none,
        "ls_prob_compromised": ls_prob_compromised,
        "sp_prob_run": sp_prob_run
    }

    return network_params