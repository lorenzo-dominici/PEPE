# This module contains the logic to load and structure data from files.
import json

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

