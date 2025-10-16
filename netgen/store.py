# This modulecontains the logic to handle the format and storage of files.
import json
from pathlib import Path

def store_network(network, file_path):
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(network, indent=4, ensure_ascii=False), encoding="utf-8")