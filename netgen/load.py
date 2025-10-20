# This module contains the logic to load and structure data from files.
import json

NET_TYPES = {"STAR", "RING", "CHAIN", "TREE", "MESH", "FULLY_CONNECTED", "REGULAR_GRAPH", "RANDOM"}
PROTOCOLS = {"HPKE", "DOUBLE_RATCHET", "SENDER_KEY", "MLS"}

def load_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
        return validate_data(data)
    
def validate_data(data):
    expected_data = {
        "num_nodes": int,
        "net_type": str,
        "protocol": str,
        "seed": int
    }

    for key, expected_type in expected_data.items():
        if key not in data:
            raise ValueError(f"Missing parameter: {key}")
        if not isinstance(data[key], expected_type):
            raise TypeError(f"Incorrect type for {key}: expected {expected_type}, got {type(data[key])}")
        
    if data["net_type"] not in NET_TYPES:
        raise ValueError(f"Invalid network type: {data['net_type']}. Expected one of {NET_TYPES}")
    if data["protocol"] not in PROTOCOLS:
        raise ValueError(f"Invalid protocol: {data['protocol']}. Expected one of {PROTOCOLS}")
    if data["num_nodes"] <= 0:
        raise ValueError("num_nodes must be a positive integer")
    return data

