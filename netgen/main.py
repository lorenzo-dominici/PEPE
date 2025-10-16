# This modules is responsible for the correct execution of the whole program.
from load import load_json
from generate import generate_network
from store import store_network
import json

def main(): 
    net_params = load_json("config/netgen.json")
    print("Loaded network parameters:", json.dumps(net_params, indent=4))
    network = generate_network(net_params)
    store_network(network, "output/network.json")

if __name__ == "__main__":
    main()

