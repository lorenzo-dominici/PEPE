# This modules is responsible for the correct execution of the whole program
from load import load_json
from generate import NetworkGenerator
# from store import store_network
import json

def main(): 
    params = load_json("config/netgen.json")
    ng = NetworkGenerator(params)
    ng.generate_network()
    # store_network(network, "output/network.json")

if __name__ == "__main__":
    main()

