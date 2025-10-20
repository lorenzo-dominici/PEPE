# This module generates the output elements according to the input data.
import random

class NetworkGenerator: 
    def __init__(self, params):
        self.num_nodes = params["num_nodes"]
        self.net_type = params["net_types"]
        self.protocol = params["protocol"]
        self.seed = params["seed"]

        random.seed(self.seed)

    def generate_network(self):

        network = {
            "items": self._generate_items
        }

        return network

    def _generate_items(self):
        {}
        