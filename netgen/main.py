# This modules is responsible for the correct execution of the whole program
from load import load_json
from generate import NetworkGenerator
from store import store_network
import json

def main(): 
    params = load_json("config/netgen.json")
    ng = NetworkGenerator(params)
    network_dict = ng.generate_network()
    if params.get("filename"):
        file_path = f"output/{params['filename']}.json"
    else:
        file_path = "output/network.json"
    store_network(network_dict, file_path)

if __name__ == "__main__":
    main()