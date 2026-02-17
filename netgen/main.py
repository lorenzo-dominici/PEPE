"""
main.py — Entry point for the PEPE network generator (netgen).

Orchestrates the full generation pipeline:
1. Load and validate parameters from ``config/netgen.json`` via :func:`load.load_json`.
2. Determine the output directory name (either from a CLI argument or
   auto-generated with a timestamp).
3. Instantiate :class:`generate.NetworkGenerator` and invoke the generation.
4. Persist three artefacts:
   * ``<name>.json``          — full PRISM-ready network descriptor
   * ``<name>_sessions.json`` — hierarchical sessions summary
   * ``<name>_graph.png``     — spring-layout visualisation (saved by the generator)
5. Copy the network JSON into a **fixed directory** so that downstream tools
   always find the latest output at a predictable path.

Usage
-----
::

    # Automatic timestamped directory (default)
    python netgen/main.py

    # Custom name — no timestamp appended
    python netgen/main.py my_network

The optional CLI argument overrides the ``filename`` value in the config
and suppresses the timestamp suffix, making the run fully reproducible
when re-executed with the same seed.
"""

import sys
from load import load_json
from generate import NetworkGenerator
from store import store_network
from datetime import datetime
from pathlib import Path

# Stable output path consumed by the PRISM preprocessor and other tools.
# A copy of the network JSON is always placed here regardless of the
# timestamped directory used for the primary output.
FIXED_DIR = "netgen/output/networks"


def main():
    """Run the complete network generation pipeline."""

    # --- 1. Load & validate configuration ---
    params = load_json("config/netgen.json")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # --- 2. Resolve output naming ---
    # CLI argument overrides config filename; no timestamp when name is given
    # so that repeated runs with the same name are idempotent.
    if len(sys.argv) > 1:
        base_name = sys.argv[1]
        dir_name = base_name
        fixed_json_name = base_name
    else:
        base_name = params.get("filename", "network")
        dir_name = f"{base_name}_{timestamp}"
        fixed_json_name = f"{base_name}_{timestamp}"

    # --- 3. Prepare output directory ---
    output_dir = f"netgen/output/{dir_name}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Sync the filename in params so the generator uses the correct base name
    # (e.g. for the graph image filename).
    params["filename"] = base_name

    # --- 4. Generate & persist ---
    ng = NetworkGenerator(params)
    network_dict, sessions_summary = ng.generate_network(output_dir)

    # Primary outputs inside the timestamped (or named) directory
    store_network(network_dict, f"{output_dir}/{base_name}.json")
    
    # Sessions summary as human-readable text
    sessions_path = Path(f"{output_dir}/{base_name}_sessions.txt")
    sessions_path.write_text(sessions_summary, encoding="utf-8")

    # --- 5. Fixed directory copy (network JSON only) ---
    fixed_dir = Path(FIXED_DIR)
    fixed_dir.mkdir(parents=True, exist_ok=True)
    store_network(network_dict, str(fixed_dir / f"{fixed_json_name}.json"))


if __name__ == "__main__":
    main()