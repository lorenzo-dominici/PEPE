"""
generate.py — Core generation engine for the PEPE network generator.

This module translates validated configuration parameters into a complete
network descriptor ready for the PRISM model checker preprocessor.  The
pipeline executed by :class:`NetworkGenerator` is:

1. **Topology generation** — Build an undirected NetworkX graph using one of
   three models (random / small-world / scale-free), optionally with
   clustering.  Node degrees can be bounded post-hoc via an interface range.

2. **Path generation** — Create directed communication paths (as
   ``nx.DiGraph`` trees) between nodes, according to the selected protocol:
   - *hpke* / *double_ratchet*: one-to-one bidirectional paths.
   - *sender_key*: one-to-many broadcast tree with one-to-one sub-sessions
     (using the chosen support protocol).
   - *mls*: group multicast — every member sends to all others; nodes are
     partitioned into groups of ≥ 2.

3. **Reward generation** — Assemble PRISM reward structures that evaluate
   message counts, vulnerability, bandwidth, memory, availability and
   degradation at model-checking time.

4. **JSON serialisation** — Produce a dictionary that maps directly to the
   PRISM preprocessor input (nodes, interfaces, channels, links, link_refs,
   session_paths, local_sessions, session_checkers, rewards).

Supporting data classes
-----------------------
``Session``
    Groups one or more directed path trees sharing the same protocol and
    session id; may own sub-sessions (e.g. sender_key).
``Contribution``
    A single reward-structure contribution: ``(command, condition, value)``.
``Reward``
    Named collection of :class:`Contribution` objects corresponding to one
    PRISM ``rewards "name" ... endrewards`` block.
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zipf
import json


# ═══════════════════════════════════════════════════════════════════════════════
#  Data-model classes
# ═══════════════════════════════════════════════════════════════════════════════

class Session:
    """Logical communication session between a set of nodes.

    A session owns one or more directed path trees (``nx.DiGraph``) and may
    contain sub-sessions (used by the *sender_key* protocol to wrap one-to-one
    support-protocol paths).

    Attributes
    ----------
    id : int | None
        Unique session identifier (assigned after construction).
    protocol : str | None
        Protocol name (e.g. ``"hpke"``, ``"mls"``).
    nodes : list[int]
        Participating node ids.
    paths : list[nx.DiGraph]
        Directed path trees.  Each tree's first node is the sender.
    subsessions : list[Session]
        Child sessions (empty unless the protocol requires sub-sessions).
    """

    def __init__(self):
        self.id = None
        self.protocol = None
        self.nodes = []
        self.paths = []
        self.subsessions = []

    def set_id(self, id):
        self.id = id

    def set_protocol(self, protocol):
        self.protocol = protocol
    
    def set_nodes(self, nodes):
        self.nodes = nodes

    def add_path(self, path):
        """Append a directed path tree to this session."""
        self.paths.append(path)

    def add_subsession(self, subsession):
        """Register a child session (e.g. support-protocol session)."""
        self.subsessions.append(subsession)

    def get_id(self):
        return self.id

    def get_nodes(self):
        return self.nodes

    def get_paths(self):
        return self.paths

    def get_subsessions(self):
        """Return a shallow copy to prevent external mutation."""
        return self.subsessions.copy()


class Contribution:
    """Single contribution to a PRISM reward structure.

    In the generated PRISM code this maps to a line of the form::

        [command] condition : value;

    Attributes
    ----------
    command : str
        Synchronisation label (may be empty for state-rewards).
    condition : str
        Boolean guard expression in PRISM syntax.
    value : str
        Numeric or arithmetic expression yielding the reward value.
    """

    def __init__(self, command: str, condition: str, value: str):
        self.command = command
        self.condition = condition
        self.value = value


class Reward:
    """Named PRISM reward structure, collecting multiple :class:`Contribution` s.

    Each instance maps to a ``rewards "<name>" ... endrewards`` block in the
    generated PRISM model.

    Attributes
    ----------
    name : str
        Reward structure identifier (e.g. ``"reward_total_messages_sent"``).
    contributions : list[Contribution]
        Ordered list of reward entries.
    """

    def __init__(self, name: str):
        self.name = name
        self.contributions = []

    def add_contribution(self, contribution: Contribution):
        """Append a contribution line to this reward structure."""
        self.contributions.append(contribution)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main generator class
# ═══════════════════════════════════════════════════════════════════════════════

class NetworkGenerator: 
    """End-to-end network generator targeting the PRISM model checker.

    The generator is fully deterministic for a given *seed*: all stochastic
    decisions (topology, path selection, probability sampling) are drawn from a
    single ``numpy.random.Generator`` instance.

    Attributes
    ----------
    params : dict
        Validated configuration parameters (mutated in-place for defaults).
    rng : numpy.random.Generator
        Seeded PRNG used throughout the generation pipeline.
    G : nx.Graph
        Undirected topology graph built during ``_generate_topology()``.
    nodes_buffer_sizes : dict[int, int]
        Mapping ``node_id → buffer_size``, populated when nodes are serialised.
    sessions : list[Session]
        Communication sessions, populated by ``_generate_paths()``.
    rewards : list[Reward]
        PRISM reward structures, populated by ``_generate_rewards()``.
    channel_bandwidth : dict[str, int]
        Mapping ``"nodeA_nodeB" → bandwidth``, set during channel serialisation.
    """

    def __init__(self, params: dict):
        """Initialise the generator with validated parameters.

        Parameters
        ----------
        params : dict
            Output of :func:`load.validate_data`.  The constructor applies
            default values for ``seed``, ``connected`` and ``gen_model`` when
            they are absent, so partial dicts are accepted.
        """
        self.params = params

        # Apply defaults for optional params that may be None
        if self.params.get("seed") is None:
            self.params["seed"] = 42
        if self.params.get("connected") is None:
            self.params["connected"] = False
        if self.params.get("gen_model") is None:
            self.params["gen_model"] = "random"

        # Single seeded PRNG for full reproducibility
        self.rng = np.random.default_rng(seed=self.params["seed"])
        # Undirected topology — populated by _generate_topology()
        self.G = nx.Graph()
        # Per-node buffer sizes — populated by _add_nodes_to_dict()
        self.nodes_buffer_sizes = {}


    def generate_network(self, output_dir: str) -> tuple[dict, dict]:
        """Run the full generation pipeline and return the output data.

        Parameters
        ----------
        output_dir : str
            Directory where the graph visualisation PNG is saved.

        Returns
        -------
        tuple[dict, dict]
            ``(network_dict, sessions_summary)`` — the PRISM-ready network
            descriptor and the human-readable sessions summary.
        """
        self._generate_topology(output_dir)
        self._generate_paths()
        network_dict = self._generate_json()
        sessions_summary = self._generate_sessions_summary()
        return network_dict, sessions_summary

    # ── Topology generation ──────────────────────────────────────────────────

    def _generate_topology(self, output_dir: str) -> None:
        """Build the undirected topology graph and save its visualisation.

        When clustering is enabled (``clusters_number_range`` is set), the
        method creates independent sub-graphs, assigns each a random colour,
        and wires them together via :meth:`_join_subgraphs`.  Otherwise a
        single graph is produced.

        The resulting graph is stored in ``self.G`` and a spring-layout PNG
        is saved to ``<output_dir>/<filename>_graph.png``.
        """
        # Determine whether multi-cluster topology is requested
        clustering = self.params["clusters_number_range"] is not None
        if clustering:
            clusters_sizes = self._partition()
            
            subgraphs = []
            # Generate one sub-graph per cluster and assign unique colours
            for i in range(len(clusters_sizes)):
                size = clusters_sizes[i]
                subgraphs.append(self._generate_graph(size))
                color = tuple(self.rng.random(3))
                nx.set_node_attributes(subgraphs[i], color, 'color')
                nx.set_edge_attributes(subgraphs[i], color, 'color')
            # Merge sub-graphs and add inter-cluster edges
            self._join_subgraphs(subgraphs)

        else:
            # No clusters — a single subgraph fills the whole topology
            self.G = self._generate_graph()
            nx.set_node_attributes(self.G, "orange", 'color')
            nx.set_edge_attributes(self.G, "black", 'color')

        # Persist the graph visualisation in the output directory
        base_name = self.params.get("filename", "network")
        graph_path = f"{output_dir}/{base_name}_graph.png"
        self._draw_graph(output_path=graph_path)

    
    def _partition(self) -> list[int]:
        """Partition ``node_number`` nodes into clusters.

        Returns a list of integers whose sum equals ``node_number``, each
        entry representing the size of one cluster.

        The algorithm handles five cases:

        1. No per-cluster range is given → equal split + remainder distribution.
        2. Per-cluster minimum too large for ``min_clusters`` → fall back to
           equal split with ``min_clusters``.
        3. Per-cluster maximum too small for ``max_clusters`` → fall back to
           equal split with ``max_clusters``.
        4. General case → start every cluster at ``min`` and randomly increment
           until the total reaches ``node_number``.
        5. All edge cases clip the cluster count to the valid interval.

        Returns
        -------
        list[int]
            Cluster sizes summing to ``self.params["node_number"]``.
        """
        # Resolve the number of clusters from the configured range
        min_c, max_c = self.params["clusters_number_range"][0], self.params["clusters_number_range"][1]
        if min_c == max_c:
            clusters_number = min_c
        else:
            clusters_number = self.rng.integers(min_c, max_c)
        while clusters_number > self.params["node_number"]:
                clusters_number = self.rng.integers(min_c, max_c)

        # --- Equal split when no per-cluster range is specified ---
        if self.params["nodes_range_per_cluster"] is None:
            nodes_per_cluster = self.params["node_number"] // clusters_number
            sizes = [nodes_per_cluster] * clusters_number
            remaining = self.params["node_number"] % clusters_number
            for i in range(remaining):
                sizes[i] += 1
            return sizes
        else:  
            # --- Fallback: per-cluster minimum too large for min_clusters ---
            if self.params["nodes_range_per_cluster"][0] * min_c > self.params["node_number"]:
                clusters_number = min_c
                nodes_per_cluster = self.params["node_number"] // clusters_number
                sizes = [nodes_per_cluster] * clusters_number
                remaining = self.params["node_number"] % clusters_number
                for i in range(remaining):
                    sizes[i] += 1
                    return sizes

            # --- Fallback: per-cluster maximum too small for max_clusters ---
            if self.params["nodes_range_per_cluster"][1] * max_c < self.params["node_number"]:
                clusters_number = max_c
                nodes_per_cluster = self.params["node_number"] // clusters_number
                sizes = [nodes_per_cluster] * clusters_number
                remaining = self.params["node_number"] % clusters_number
                for i in range(remaining):
                    sizes[i] += 1
                    return sizes
                
            # --- General case: resample until clusters_number fits, then
            #     start each cluster at the minimum and randomly increment ---
            while (clusters_number * self.params["nodes_range_per_cluster"][0] > self.params["node_number"] or clusters_number * self.params["nodes_range_per_cluster"][1] < self.params["node_number"]):
                clusters_number = self.rng.integers(min_c, max_c)

            sizes = [self.params["nodes_range_per_cluster"][0]] * clusters_number
            while sum(sizes) < self.params["node_number"]:
                i = self.rng.integers(0, len(sizes) - 1)
                if sizes[i] < self.params["nodes_range_per_cluster"][1]:
                    sizes[i] += 1
            return sizes
    
    
    def _generate_graph(self, size: int = None) -> nx.Graph:
        """Dispatch graph construction to the model-specific method.

        Parameters
        ----------
        size : int, optional
            Number of nodes.  Defaults to ``self.params["node_number"]``
            (used when building a single non-clustered graph).

        Returns
        -------
        nx.Graph
            The generated undirected graph with ``size`` nodes.

        Notes
        -----
        After the model-specific method returns, the ``if_range`` bound (if
        set) is enforced post-hoc by :meth:`_bound_nodes_degrees`.
        """
        if size is None:
            size = self.params["node_number"]

        G = nx.Graph()

        match self.params["gen_model"]:
            case "random":
                G = self._generate_random_graph(size)
            case "smart-world":
                G = self._generate_smart_world_graph(size)
            case "scale-free":
                G = self._generate_scale_free_graph(size)

        # Post-hoc degree bounding (min/max interfaces per node)
        if self.params["if_range"] is not None:
            self._bound_nodes_degrees(G)
        
        return G
    

    def _generate_random_graph(self, size: int) -> nx.Graph:
        """Generate a random graph using one of three strategies.

        Strategy selection follows a priority chain:

        1. **conn_prob set** → Erdős–Rényi ``G(n, p)`` via
           ``nx.gnp_random_graph``.  If ``connected`` is requested, retries
           up to 1 000 times.
        2. **degree_distr set** → Havel-Hakimi algorithm using the degree
           sequence sampled from the configured distribution.  Connectedness
           is attempted via ``nx.double_edge_swap``.
        3. **if_range set** → Havel-Hakimi with a uniform degree distribution
           bounded by the interface range.

        Parameters
        ----------
        size : int
            Number of nodes in the graph.

        Returns
        -------
        nx.Graph

        Raises
        ------
        nx.NetworkXUnfeasible
            If the ``connected`` constraint cannot be satisfied.
        """
        # --- Strategy 1: Erdős–Rényi G(n, p) ---
        if self.params["conn_prob"] is not None:
            G = nx.gnp_random_graph(size, self.params["conn_prob"], seed = self.rng)
            if self.params["connected"]:
                tries = 0
                while not(nx.is_connected(G)) and tries <= 1000:
                    G = nx.gnp_random_graph(size, self.params["conn_prob"], seed = self.rng)
                    tries += 1
                if not(nx.is_connected(G)):
                    raise nx.NetworkXUnfeasible("can't generate a connected graph with these parameters")
            return G
        
        # --- Strategy 2: Havel-Hakimi from a degree distribution ---
        if self.params["degree_distr"] is not None:
            degrees = self._get_degrees_from_distr()
            G = nx.havel_hakimi_graph(degrees)
            if self.params["connected"]:
                tries = 0
                while not(nx.is_connected(G)) and tries <=1000:
                    nx.double_edge_swap(G, nswap=5*len(G.edges()), max_tries=100*len(G.edges()))
                    tries += 1
                if not(nx.is_connected(G)):
                    raise nx.NetworkXUnfeasible("Can't generate a connected graph with these parameters")
            return G
        
        # --- Strategy 3: uniform degree from if_range bounds ---
        if self.params["if_range"] is not None:
            degrees = self._get_degrees_from_distr(distr_type = "uniform", distr_params = self.params["if_range"])
            G = nx.havel_hakimi_graph(degrees)
            if self.params["connected"]:
                tries = 0
                while not(nx.is_connected(G)) and tries <=1000:
                    nx.double_edge_swap(G, nswap=5*len(G.edges()), max_tries=100*len(G.edges()))
                    tries += 1
                if not(nx.is_connected(G)):
                    raise nx.NetworkXUnfeasible("can't generate a connected graph with these parameters")
            return G


    def _generate_smart_world_graph(self, size: int) -> nx.Graph:
        """Generate a small-world graph (Watts-Strogatz family).

        Two variants are available:

        * ``delete_rewired == True`` → classical **Watts-Strogatz** model, where
          rewired edges replace the original edges.
        * ``delete_rewired == False`` (default) → **Newman-Watts-Strogatz**
          model, where shortcut edges are *added* without removing existing ones.

        Parameters
        ----------
        size : int
            Number of nodes.

        Returns
        -------
        nx.Graph

        Raises
        ------
        nx.NetworkXUnfeasible
            If the connected variant fails after 1 000 tries.
        """
        if self.params["mean_degree_range"] is None:
            mean = size // 2
        elif self.params["mean_degree_range"][0] >= size:
            mean = size - 1
        elif self.params["mean_degree_range"][0] == self.params["mean_degree_range"][1]:
            mean = self.params["mean_degree_range"][0]
        else:
            mean = self.rng.integers(self.params["mean_degree_range"][0], self.params["mean_degree_range"][1])
            while mean >= size:
                mean = self.rng.integers(self.params["mean_degree_range"][0], self.params["mean_degree_range"][1])
        
        if self.params["rewiring_prob"] is None:
            p = 0.5
        else:
            p = self.params["rewiring_prob"]

        # Select variant based on delete_rewired flag
        # If delete_rewired is true, use the Watts-Strogatz algorithm
        if self.params["delete_rewired"] == True:
            if self.params["connected"]:
                try:
                    G = nx.connected_watts_strogatz_graph(size, mean, p, tries = 1000, seed = self.rng)
                except Exception:
                    raise nx.NetworkXUnfeasible("Can't generate a connected graph with these parameters")
            else:
                G = nx.watts_strogatz_graph(size, mean, p, seed = self.rng)
        # Newman-Watts-Strogatz: shortcuts are added (no edge deletion),
        # producing a graph that is always connected by construction.
        else:
            G = nx.newman_watts_strogatz_graph(size, mean, p, seed = self.rng)
        return G
                    

    def _generate_scale_free_graph(self, size: int) -> nx.Graph:
        """Generate a scale-free graph via the **extended Barabási-Albert** model.

        The extended BA model accepts three probabilities:

        * ``m`` — initial degree (number of edges each new node brings).
        * ``p`` — probability of adding a new edge between existing nodes.
        * ``q`` — probability of rewiring an existing edge.

        The constraint ``p + q < 1`` is checked during validation in
        :func:`load.validate_data`.

        Parameters
        ----------
        size : int
            Number of nodes.

        Returns
        -------
        nx.Graph

        Raises
        ------
        nx.NetworkXUnfeasible
            If the connected constraint cannot be met within 1 000 attempts.
        """
        # generation of scale-free graph with the extended Barabási-Albert algorithm

        n = size
        if self.params["initial_degree_range"] is None:
            m = n // 2
        else:
            min_d, max_d = self.params["initial_degree_range"][0], self.params["initial_degree_range"][1]
            if min_d >= n:
                m = n - 1
            elif min_d == max_d:
                m = min_d
            else:
                m = self.rng.integers(min_d, max_d)
                while m >= n:
                    m = self.rng.integers(min_d, max_d)
        if self.params["new_edges_prob"] is None:
            p = 0
        else:
            p = self.params["new_edges_prob"]
        if self.params["rewiring_prob"] is None:
            q = 0
        else:
            q = self.params["rewiring_prob"]

        G = nx.Graph()
        G = nx.extended_barabasi_albert_graph(n, m, p, q, seed = self.rng)
        if self.params["connected"]:
            tries = 0
            while not(nx.is_connected(G)) and tries < 1000:
                G = nx.extended_barabasi_albert_graph(n, m, p, q, seed = self.rng)
            if not(nx.is_connected(G)):
                raise nx.NetworkXUnfeasible("can't generate a connected graph with these parameters")
            
        return G
    
    
    def _get_degrees_from_distr(self, distr_type: str = None, distr_params=None) -> list:
        """Sample a graphical degree sequence from a statistical distribution.

        The method draws ``node_number`` degree values according to the
        selected distribution, enforces the *even-sum* invariant (required by
        the handshaking lemma for undirected graphs), and clips every entry to
        ``[0, node_number]``.

        Supported distributions
        -----------------------
        * **binomial** — ``numpy.random.Generator.binomial(n, p, size)``
        * **uniform** — ``numpy.random.Generator.integers(low, high+1, size)``
        * **normal** — ``numpy.random.Generator.normal(mean, stddev, size)``,
          rounded to integers.
        * **powerlaw** — ``scipy.stats.zipf.rvs(a=gamma, loc=k_min, size)``
        * **custom** — user-supplied Python list literal; padded/trimmed to
          ``node_number`` and adjusted for parity.

        Parameters
        ----------
        distr_type : str, optional
            Overrides ``self.params["degree_distr"]["type"]`` (used when the
            caller wants a uniform sequence from ``if_range``).
        distr_params : list, optional
            Overrides ``self.params["degree_distr"]["params"]``.

        Returns
        -------
        list[int]
            Degree sequence of length ``node_number`` with even sum.
        """
        if distr_type is None:
            distr_type = self.params["degree_distr"].get("type")
        if distr_params is None:
            distr_params = self.params["degree_distr"].get("params")
        degrees = []
        # Generate the degree sequence from the distribution parameters.
        # NOTE: the sum of the degrees must be even (handshaking lemma).
        match distr_type:
            case "binomial":
                n = distr_params[0]
                if n > self.params["node_number"]:
                    n = self.params["node_number"]
                p = distr_params[1]
                degrees = self.rng.binomial(n, p, self.params["node_number"])
                while sum(degrees) % 2 != 0:
                    degrees = self.rng.binomial(n, p, self.params["node_number"])
            case "uniform":
                low = distr_params[0]
                high = distr_params[1]
                if high > self.params["node_number"]:
                    high = self.params["node_number"]
                degrees = self.rng.integers(low, high + 1, self.params["node_number"])
                while sum(degrees) % 2 != 0:
                    degrees = self.rng.integers(low, high + 1, self.params["node_number"])
            case "normal":
                mean = distr_params[0]
                stddev = distr_params[1]
                if mean > self.params["node_number"]:
                    mean = self.params["node_number"]
                degrees = self.rng.normal(mean, stddev, self.params["node_number"])
                degrees = np.round(degrees).astype(int)
                while sum(degrees) % 2 != 0:
                    degrees = self.rng.normal(mean, stddev, self.params["node_number"])
                    degrees = np.round(degrees).astype(int)
            case "powerlaw":
                gamma = distr_params[0]
                k_min = distr_params[1]
                if k_min > self.params["node_number"]:
                    k_min = self.params["node_number"]
                degrees = zipf.rvs(a = gamma, loc = k_min, size = self.params["node_number"], random_state = self.rng)
                while sum(degrees) % 2 != 0:
                    degrees = zipf.rvs(a = gamma, loc = k_min, size = self.params["node_number"], random_state = self.rng)
            case "custom":
                degrees = eval(distr_params)
                if len(degrees) < self.params["node_number"]:
                    n = self.params["node_number"] - len(degrees)
                    to_append = [degrees[-1]] * n
                    degrees.extend(to_append)
                elif len(degrees) > self.params["node_number"]:
                    degrees = degrees[:self.params["node_number"]]
                while sum(degrees) % 2 != 0:
                    i = self.rng.integers(0, len(degrees) - 1)
                    if degrees[i] < self.params["node_number"]:
                        degrees[i] += 1
        
        # Clip every degree to the valid range [0, node_number]
        for i in range(len(degrees)):
            if degrees[i] < 0:
                degrees[i] = 0
            elif degrees[i] > self.params["node_number"]:
                degrees[i] = self.params["node_number"]

        return degrees

    
    def _bound_nodes_degrees(self, G: nx.Graph) -> None:
        """Enforce the ``if_range`` degree bounds on every node in *G*.

        The method runs two iterative passes:

        1. **Excess-degree removal** — Nodes whose degree exceeds ``max_if``
           have edges removed.  When ``connected`` is ``True``, bridge edges
           are skipped to preserve connectivity.
        2. **Deficit-degree addition** — Nodes whose degree is below ``min_if``
           gain edges to random non-neighbours that still have capacity.

        Both passes loop until convergence (no further modification is
        possible).  If the constraints cannot be fully satisfied,
        ``NetworkXUnfeasible`` is raised.

        Raises
        ------
        nx.NetworkXUnfeasible
            If bounds are infeasible for the current graph structure.
        """
        min_if, max_if = self.params["if_range"][0], self.params["if_range"][1]
        nodes = list(nx.nodes(G))
        modified = True
        while modified:
            modified = False
            for node_a in nodes:
                if nx.degree(G, node_a) > max_if:
                    neighbors = list(nx.neighbors(G, node_a))
                    self.rng.shuffle(neighbors)
                    for node_b in neighbors:
                        if nx.degree(G, node_b) > min_if and nx.degree(G, node_a) > max_if:
                            if self.params["connected"] == False:
                                G.remove_edge(node_a, node_b)
                                modified = True
                                break
                            else:
                                bridges = list(nx.bridges(G))
                                if (node_a, node_b) not in bridges and (node_b, node_a) not in bridges:
                                    G.remove_edge(node_a, node_b)  
                                    modified = True
                                    break

        for node_a in nodes:
            if nx.degree(G, node_a) > max_if:
                raise nx.NetworkXUnfeasible("can't bound the number of interfaces for each node")      

        modified = True
        while modified:
            modified = False
            for node_a in nodes:
                if nx.degree(G, node_a) < min_if:
                    nodes_to_check = nodes.copy()
                    nodes_to_check.remove(node_a)
                    self.rng.shuffle(nodes_to_check)
                    for node_b in nodes_to_check:
                        if nx.degree(G, node_b) < max_if and nx.degree(G, node_a) < max_if and (node_a, node_b) not in nx.edges(G, node_a) and (node_b, node_a) not in nx.edges(G, node_b):
                            G.add_edge(node_a, node_b)
                            modified = True
                            break

        for node_a in nodes:
            if nx.degree(G, node_a) < min_if:             
                raise nx.NetworkXUnfeasible("can't bound the number of interfaces for each node")
        
    
    def _join_subgraphs(self, subgraphs: list[nx.Graph]) -> None:
        """Merge *subgraphs* into ``self.G`` and add inter-cluster edges.

        The sub-graphs are concatenated via ``nx.disjoint_union_all`` (which
        relabels nodes to avoid collisions).  Then, for every pair of clusters,
        each possible inter-cluster edge is included independently with
        probability ``inter_clusters_coeff`` (defaults to 0.01).

        The implementation uses vectorised NumPy meshgrids and boolean masks
        for efficiency on large cluster pairs.
        """
        n_clusters = len(subgraphs)
        p = self.params["inter_clusters_coeff"]
        if p is None:
            p = 0.01
        self.G = nx.disjoint_union_all(subgraphs)
        all_nodes = list(self.G.nodes())
        cluster_boundaries = []
        start = 0
        for g in subgraphs:
            cluster_boundaries.append(start)
            start += g.number_of_nodes()
        cluster_boundaries.append(start)

        # cluster_boundaries = [i * len(subgraphs[i]) for i in range(n_clusters)] + [len(all_nodes)]
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                nodes_i = np.arange(cluster_boundaries[i], cluster_boundaries[i+1])
                nodes_j = np.arange(cluster_boundaries[j], cluster_boundaries[j+1])
                u_mesh, v_mesh = np.meshgrid(nodes_i, nodes_j, indexing='ij')
                u_flat = u_mesh.ravel()
                v_flat = v_mesh.ravel()
                mask = self.rng.random(len(u_flat)) < p
                for u, v in zip(u_flat[mask], v_flat[mask]):
                    self.G.add_edge(int(u), int(v), color = "black")


    def _draw_graph(self, output_path: str = None) -> None:
        """Render the topology as a spring-layout visualisation.

        Node sizes are proportional to degree centrality (squared, so that
        hub nodes are visually prominent).  Node/edge colours are taken from
        the ``'color'`` attribute set during topology generation.

        Parameters
        ----------
        output_path : str, optional
            If provided, the figure is saved as a PNG (150 dpi) and the
            matplotlib figure is closed.  Otherwise ``plt.show()`` is called.
        """
        pos = nx.spring_layout(self.G, seed = self.params["seed"])
        centrality = nx.degree_centrality(self.G)
        scale_factor = 10000
        nodes_sizes = [ (v ** 2) * scale_factor + 500 for v in centrality.values() ]

        # Determine max label length to scale font appropriately
        import math
        max_label_len = max(len(str(n)) for n in self.G.nodes())
        min_size = min(nodes_sizes)
        min_radius = math.sqrt(min_size / math.pi)
        # Font must fit the widest label inside the smallest node
        font_size = max(12, min(22, int(min_radius * 1.1 / max(max_label_len * 0.55, 1))))

        node_colors = [data['color'] for _, data in self.G.nodes(data=True)]
        edge_colors = [data['color'] for _, _, data in self.G.edges(data=True)]

        plt.figure(figsize=(10, 10))
        nx.draw(self.G, pos, with_labels=True, node_size=nodes_sizes, node_color=node_colors, edge_color = edge_colors, font_size=font_size, font_weight='bold', font_color='white')
        plt.title("Grafo")
        plt.axis('off')
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    # ── Path / session generation ────────────────────────────────────────────

    def _generate_paths(self) -> None:
        """Create directed communication paths and populate ``self.sessions``.

        The structure of the sessions depends on the protocol:

        **hpke / double_ratchet** (point-to-point):
            For every ordered pair of nodes ``(a, b)`` with ``a < b``,
            a bidirectional session is created with probability ``path_perc``.
            Each session holds two paths: ``a→b`` and ``b→a`` (return path).

        **sender_key** (one-to-many + one-to-one sub-sessions):
            Each node is selected as a sender with probability ``path_perc``.
            For each sender, other nodes are selected as receivers with
            probability ``path_perc``, with a minimum of 2 receivers guaranteed
            (true one-to-many).  A broadcast tree (``nx.DiGraph``) is built.
            Additionally, a one-to-one sub-session (using the configured
            support protocol) is created for every sender-receiver pair
            (forward + return path).

        **mls** (group multicast):
            Nodes are shuffled and partitioned into groups of ≥ 2 nodes.
            Within each group every member creates a one-to-many broadcast
            tree to all other members.  When a group has > 2 members, the
            one-to-one paths for each pair are also added.

        Every path tree is an ``nx.DiGraph`` whose first node is the sender.
        Receiver nodes carry the attribute ``is_receiver = True``.  Epoch and
        ratchet sizes are sampled per session from ``ls_size_epoch_range``
        and ``ls_size_ratchet_range``.
        """
        nodes = list(self.G.nodes())
        # Route according to the selected protocol
        match self.params["protocol"]:
            # ── HPKE / Double-Ratchet: bidirectional one-to-one paths ──
            case "hpke" | "double_ratchet":
                paths = []
                return_paths = []
                sessions = []
                id = 0
                for i in range(len(nodes)):
                    for j in range(i, len(nodes)):
                        if i != j:
                            a = nodes[i]
                            b = nodes[j]
                            if self.rng.random() < self.params["path_perc"]:
                                path = nx.DiGraph()
                                nx.add_path(path, nx.shortest_path(self.G, a, b))
                                # Sample epoch / ratchet sizes for this session
                                if self.params.get("ls_size_epoch_range")[0] == self.params.get("ls_size_epoch_range")[1]:
                                    epoch_size = int(self.params.get("ls_size_epoch_range")[0])
                                else:
                                    epoch_size = int(self.rng.integers(self.params.get("ls_size_epoch_range")[0], self.params.get("ls_size_epoch_range")[1] + 1))
                                if self.params.get("ls_size_ratchet_range")[0] == self.params.get("ls_size_ratchet_range")[1]:
                                    ratchet_size = int(self.params.get("ls_size_ratchet_range")[0])
                                else:
                                    ratchet_size = int(self.rng.integers(self.params.get("ls_size_ratchet_range")[0], self.params.get("ls_size_ratchet_range")[1] + 1))
                                
                                path.nodes[a]['epoch_size'] = epoch_size
                                path.nodes[a]['ratchet_size'] = ratchet_size
                                path.nodes[b]['epoch_size'] = epoch_size
                                path.nodes[b]['ratchet_size'] = ratchet_size
                                
                                path.nodes[b]['is_receiver'] = True

                                paths.append(path)

                                # Generate the symmetric return path (b → a)
                                return_path = nx.DiGraph()
                                nx.add_path(return_path, nx.shortest_path(self.G, b, a))

                                return_path.nodes[b]['epoch_size'] = epoch_size
                                return_path.nodes[b]['ratchet_size'] = ratchet_size
                                return_path.nodes[a]['epoch_size'] = epoch_size
                                return_path.nodes[a]['ratchet_size'] = ratchet_size
                                return_path.nodes[a]['is_receiver'] = True

                                return_paths.append(return_path)

                                session = Session()
                                session.set_id(id)
                                session.set_protocol(self.params["protocol"])
                                session.set_nodes([a, b])
                                session.add_path(path)
                                session.add_path(return_path)
                                sessions.append(session)
                                id += 1

                self.sessions = sessions

            # ── Sender-Key: one-to-many + sub-session per receiver ──
            # For sender_key, we need one-to-many paths and all the one-to-one
            # paths sender→receiver and receiver→sender (via support protocol).
            case "sender_key":
                # Step 1: select senders with probability path_perc.
                # Since sender_key is a broadcast protocol, each selected sender
                # will transmit to ALL other nodes (true one-to-many).
                senders = []
                for node in nodes:
                    if self.rng.random() < self.params["path_perc"]:
                        senders.append(node)
                # Guarantee at least one sender
                if len(senders) == 0:
                    senders.append(self.rng.choice(nodes))
                # Step 2: for each sender, select receivers using path_perc as
                # the inclusion probability per node (at least 1 guaranteed).
                # With high path_perc the paths are naturally one-to-many.
                one_to_many_paths = []
                sessions = []
                for sender in senders:
                    if self.params.get("ls_size_epoch_range")[0] == self.params.get("ls_size_epoch_range")[1]:
                        epoch_size = int(self.params.get("ls_size_epoch_range")[0])
                    else:
                        epoch_size = int(self.rng.integers(self.params.get("ls_size_epoch_range")[0], self.params.get("ls_size_epoch_range")[1] + 1))

                    if self.params.get("ls_size_ratchet_range")[0] == self.params.get("ls_size_ratchet_range")[1]:  
                        ratchet_size = int(self.params.get("ls_size_ratchet_range")[0])
                    else:
                        ratchet_size = int(self.rng.integers(self.params.get("ls_size_ratchet_range")[0], self.params.get("ls_size_ratchet_range")[1] + 1))

                    candidates = [n for n in nodes if n != sender]
                    destinations = [n for n in candidates if self.rng.random() < self.params["path_perc"]]
                    # Guarantee at least 1 receiver
                    if len(destinations) == 0:
                        destinations.append(self.rng.choice(candidates))
                    tree = nx.DiGraph()
                    tree.add_node(sender)
                    tree.nodes[sender]['epoch_size'] = epoch_size
                    tree.nodes[sender]['ratchet_size'] = ratchet_size
                    # Step 3: for each destination, compute the shortest path from
                    # sender to dest and merge it into the broadcast tree.
                    for dest in destinations:
                        path = nx.DiGraph()
                        nx.add_path(path, nx.shortest_path(self.G, sender, dest))
                        nx.add_path(tree, path)
                        # mark the destination node as receiver
                        tree.nodes[dest]['is_receiver'] = True
                        tree.nodes[dest]['epoch_size'] = epoch_size
                        tree.nodes[dest]['ratchet_size'] = ratchet_size
                    one_to_many_paths.append(tree)

                # Step 4: create bidirectional one-to-one support-protocol
                # sub-sessions between sender and each receiver.
                one_to_one_paths = []
                one_to_one_return_paths = []
                id = 0
                
                for tree in one_to_many_paths:
                    session = Session()
                    subsessions = []
                    sender = list(tree.nodes())[0]
                    receivers = [n for n, attr in tree.nodes(data=True) if attr.get('is_receiver')]
                    for receiver in receivers:
                        path = nx.DiGraph()
                        nx.add_path(path, nx.shortest_path(self.G, sender, receiver))

                        if self.params.get("ls_size_epoch_range")[0] == self.params.get("ls_size_epoch_range")[1]:
                            epoch_size = int(self.params.get("ls_size_epoch_range")[0])
                        else:
                            epoch_size = int(self.rng.integers(self.params.get("ls_size_epoch_range")[0], self.params.get("ls_size_epoch_range")[1] + 1))
                        if self.params.get("ls_size_ratchet_range")[0] == self.params.get("ls_size_ratchet_range")[1]:
                            ratchet_size = int(self.params.get("ls_size_ratchet_range")[0])
                        else:
                            ratchet_size = int(self.rng.integers(self.params.get("ls_size_ratchet_range")[0], self.params.get("ls_size_ratchet_range")[1] + 1))
                        
                        path.nodes[sender]['epoch_size'] = epoch_size
                        path.nodes[sender]['ratchet_size'] = ratchet_size
                        path.nodes[receiver]['epoch_size'] = epoch_size
                        path.nodes[receiver]['ratchet_size'] = ratchet_size
                        path.nodes[receiver]['is_receiver'] = True

                        one_to_one_paths.append(path)

                        # one-to-one return path
                        return_path = nx.DiGraph()
                        nx.add_path(return_path, nx.shortest_path(self.G, receiver, sender))
                        
                        return_path.nodes[sender]['epoch_size'] = epoch_size
                        return_path.nodes[sender]['ratchet_size'] = ratchet_size
                        return_path.nodes[receiver]['epoch_size'] = epoch_size
                        return_path.nodes[receiver]['ratchet_size'] = ratchet_size
                        return_path.nodes[sender]['is_receiver'] = True

                        one_to_one_return_paths.append(return_path)

                        subsession = Session()
                        subsession.set_id(id)
                        subsession.set_protocol(self.params["support_protocol"])
                        subsession.set_nodes([sender, receiver])
                        subsession.add_path(path)
                        subsession.add_path(return_path)
                        subsessions.append(subsession)
                        id += 1
                    # Assemble the main sender_key session: broadcast tree +
                    # all one-to-one sub-sessions.
                    session.set_id(id)
                    session.set_protocol(self.params["protocol"])
                    session.set_nodes([sender] + receivers)
                    session.add_path(tree)
                    for subsession in subsessions:
                        session.add_subsession(subsession)
                    sessions.append(session)
                    id += 1
                
                
                
                self.sessions = sessions

            # ── MLS: group multicast, one-to-many per member ──
            case "mls":
                # For MLS, partition nodes into groups and within each group
                # every member broadcasts to all other members.
                nodes = nodes.copy()
                self.rng.shuffle(nodes)

                # Determine the number of groups from mls_sessions_range
                if self.params.get("mls_sessions_range") is None:
                    sessions_number = 1
                elif self.params.get("mls_sessions_range")[0] == self.params.get("mls_sessions_range")[1]:
                    sessions_number = self.params.get("mls_sessions_range")[0]
                    if sessions_number > len(nodes) // 2:
                        sessions_number = 1
                else:
                    sessions_number = self.rng.integers(self.params.get("mls_sessions_range")[0], self.params.get("mls_sessions_range")[1] + 1)
                    while sessions_number > len(nodes) // 2:
                        sessions_number = self.rng.integers(self.params.get("mls_sessions_range")[0], self.params.get("mls_sessions_range")[1] + 1)

                # Partition nodes into groups of ≥ 2 members.
                # Phase 1: assign 2 nodes per group (minimum viable group).
                # Phase 2: distribute remaining nodes randomly.
                groups = [[] for _ in range(sessions_number)]
                idx = 0
                for g in range(sessions_number):
                    groups[g].append(nodes[idx])
                    groups[g].append(nodes[idx + 1])
                    idx += 2
                remaining = nodes[idx:]
                for node in remaining:
                    g = int(self.rng.integers(0, sessions_number))
                    groups[g].append(node)

                one_to_many_paths = []
                one_to_one_return_paths = []
                sessions = []
                id = 0
                # For each group, every member generates a broadcast tree
                # reaching all other group members.
                for group in groups:
                    session = Session()
                    session.set_protocol(self.params["protocol"])
                    session.set_nodes(group)
                    session.set_id(id)
                    for sender in group:
                        if self.params.get("ls_size_epoch_range")[0] == self.params.get("ls_size_epoch_range")[1]:
                            epoch_size = int(self.params.get("ls_size_epoch_range")[0])
                        else:
                            epoch_size = int(self.rng.integers(self.params.get("ls_size_epoch_range")[0], self.params.get("ls_size_epoch_range")[1] + 1))

                        if self.params.get("ls_size_ratchet_range")[0] == self.params.get("ls_size_ratchet_range")[1]:  
                            ratchet_size = int(self.params.get("ls_size_ratchet_range")[0])
                        else:
                            ratchet_size = int(self.rng.integers(self.params.get("ls_size_ratchet_range")[0], self.params.get("ls_size_ratchet_range")[1] + 1))

                        # Build broadcast tree: for each destination, merge the
                        # shortest path into the tree.
                        destinations = [n for n in group if n != sender]
                        tree = nx.DiGraph()
                        tree.add_node(sender)
                        tree.nodes[sender]['epoch_size'] = epoch_size
                        tree.nodes[sender]['ratchet_size'] = ratchet_size
                        for dest in destinations:
                            path = nx.DiGraph()
                            nx.add_path(path, nx.shortest_path(self.G, sender, dest))
                            nx.add_path(tree, path)
                            # Mark the destination node as a receiver
                            tree.nodes[dest]['is_receiver'] = True
                            tree.nodes[dest]['epoch_size'] = epoch_size
                            tree.nodes[dest]['ratchet_size'] = ratchet_size
                            if len(destinations) > 1:
                                path.nodes[sender]['epoch_size'] = epoch_size
                                path.nodes[sender]['ratchet_size'] = ratchet_size
                                path.nodes[dest]['epoch_size'] = epoch_size
                                path.nodes[dest]['ratchet_size'] = ratchet_size
                                path.nodes[dest]['is_receiver'] = True
                                session.add_path(path)
                        session.add_path(tree)
                    id += 1
                    sessions.append(session)

                self.sessions = sessions


    # ── Reward generation ────────────────────────────────────────────────────

    def _generate_rewards(self) -> None:
        """Build PRISM reward structures and store them in ``self.rewards``.

        The following reward families are generated:

        * **Message rewards** (per protocol):
          - ``reward_data_message_<proto>`` — counts data-message sends.
          - ``reward_system_message_<proto>`` — counts system-message sends.
        * **Aggregate message rewards**:
          - ``reward_total_messages_sent`` — total sends across all paths.
          - ``reward_total_messages_received`` — messages that reached receivers.
          - ``reward_total_messages_lost`` — messages not received.
          - ``reward_total_messages_check_success / _check_failure``.
        * **Resource rewards**:
          - ``reward_bandwidth_availability`` — average relative channel usage.
          - ``reward_memory_availability`` — average relative buffer usage.
        * **Security & availability rewards**:
          - ``reward_network_vulnerability`` — fraction of sessions with ≥ 1
            compromised node.
          - ``reward_network_availability`` — fraction of nodes ON plus
            non-failed channels/interfaces.
          - ``reward_network_degradation`` — fraction of channels/interfaces
            in the *error* state.

        All values are expressed as PRISM arithmetic expressions referencing
        the generated module variable names.
        """
        self.rewards = []

        def generate_message_reward():
            """Inner helper that builds all reward structures.

            Uses closures over ``self.sessions``, ``self.G``,
            ``self.nodes_buffer_sizes`` and ``self.channel_bandwidth``.
            Results are appended to ``self.rewards``.
            """
            with open("config/netgen_files.json", "r") as f:
                files_config = json.load(f)
            
            with open(files_config.get("consts"), "r") as f:
                consts = json.load(f)
            
            rewards_data = {
                "hpke": Reward("reward_data_message_hpke"),
                "double_ratchet": Reward("reward_data_message_double_ratchet"),
                "sender_key": Reward("reward_data_message_sender_key"), 
                "mls": Reward("reward_data_message_mls")
            }

            rewards_system = {
                "hpke": Reward("reward_system_message_hpke"),
                "double_ratchet": Reward("reward_system_message_double_ratchet"),
                "sender_key": Reward("reward_system_message_sender_key"), 
                "mls": Reward("reward_system_message_mls")
            }

            reward_message_sent = Reward("reward_total_messages_sent")
            reward_message_received = Reward("reward_total_messages_received")
            reward_message_lost = Reward("reward_total_messages_lost")
            reward_message_check_success = Reward("reward_total_messages_check_success")
            reward_message_check_failure = Reward("reward_total_messages_check_failure")

            reward_bandwidth_availability = Reward("reward_bandwidth_availability")
            reward_memory_availability = Reward("reward_memory_availability")
            reward_network_vulnerability = Reward("reward_network_vulnerability")
    
            reward_network_availability = Reward("reward_network_availability")
            reward_network_degradation = Reward("reward_network_degradation")


            # Flatten all sessions (including sub-sessions) for iteration
            sessions = self.sessions.copy()
            for session in self.sessions:
                ss = session.get_subsessions()
                sessions.extend(ss)

            # ── Per-path message rewards (data vs. system) ────────────────
            for session in sessions:
                for i, path in enumerate(session.paths):
                    id = session.id
                    session_path = f"{i}_{id}" 
                    command = f"cmd_send_session_path_{session_path}"

                    reward_message_sent.add_contribution(Contribution(command, "true", "1"))

                    match session.protocol:
                        case "hpke":
                            condition = f"(session_path_data_message_{session_path} = {consts['const_message_data']})"
                            value = "1" 
                            rewards_data["hpke"].add_contribution(Contribution(command, condition, value))
                            condition = f"(session_path_system_message_{session_path} = {consts['const_message_reset']})"
                            rewards_system["hpke"].add_contribution(Contribution(command, condition, value))

                        case "double_ratchet":
                            condition = f"(session_path_data_message_{session_path} = {consts['const_message_data']})"
                            value = "1" 
                            rewards_data["double_ratchet"].add_contribution(Contribution(command, condition, value))
                            condition = f"(session_path_system_message_{session_path} = {consts['const_message_reset']})"
                            rewards_system["double_ratchet"].add_contribution(Contribution(command, condition, value))

                        case "hpke_sender_key" | "double_ratchet_sender_key":
                            condition = "true" # all messages should be system
                            value = "1"
                            rewards_system["sender_key"].add_contribution(Contribution(command, condition, value))

                        case "sender_key":
                            condition = "true" # all messages should be data
                            value = "1"
                            rewards_data["sender_key"].add_contribution(Contribution(command, condition, value))

                        case "mls":                    
                            condition = f"(session_path_system_message_{session_path} != {consts['const_message_tree_refresh']}) & (session_path_system_message_{session_path} != {consts['const_message_current_tree']})"
                            value = "1" 
                            rewards_data["mls"].add_contribution(Contribution(command, condition, value))
                            condition = f"(session_path_system_message_{session_path} = {consts['const_message_tree_refresh']}) | (session_path_system_message_{session_path} = {consts['const_message_current_tree']})"
                            rewards_system["mls"].add_contribution(Contribution(command, condition, value))
                    
                    for node in self._get_receivers_from_path(path):
                        command = f"cmd_cleanup_session_checker_of_node_{node}_of_path_{i}_{id}"
                        condition = f"(session_checker_message_arrived_{node}_{i}_{id} = true)"
                        value = "1"
                        reward_message_received.add_contribution(Contribution(command, condition, value))
                        condition = f"(session_checker_message_arrived_{node}_{i}_{id} = false)"
                        reward_message_lost.add_contribution(Contribution(command, condition, value))

                        command = f"cmd_check_success_session_checker_of_node_{node}_of_path_{i}_{id}"
                        condition = "true"
                        value = "1"
                        reward_message_check_success.add_contribution(Contribution(command, condition, value))
                        command = f"cmd_check_failure_session_checker_of_node_{node}_of_path_{i}_{id}"
                        reward_message_check_failure.add_contribution(Contribution(command, condition, value))

            # ── Network vulnerability reward ──────────────────────────────
            # Value = fraction of sessions that have at least one compromised node.
            command = ""
            condition = "true"
            value = "("
            for session in sessions:
                value += "(("
                for node in session.nodes:
                    value += f"(local_session_compromised_{node}_{session.id} = true ? 1 : 0) + "
                value = value[:-3] + ") >= 1 ? 1 : 0) + "
            value = value[:-3] + f") / {len(sessions)}"
            reward_network_vulnerability.add_contribution(Contribution(command, condition, value))


            # ── Resource availability rewards (memory & bandwidth) ────────
            # Memory: average ratio of remaining buffer / total buffer per node.
            command = ""
            condition = "true"
            value = "("
            for node in list(self.G.nodes()):
                value += f"node_buffer_{node} / {self.nodes_buffer_sizes[node]} + "

            value = value[:-3] + f") / {len(list(self.G.nodes()))}" 
            reward_memory_availability.add_contribution(Contribution(command, condition, value))

            # Bandwidth: average ratio of remaining bandwidth / total per channel.
            value = "("
            for edge in list(self.G.edges()):
                command = ""
                condition = "true"
                value += f"channel_bandwidth_{edge[0]}_{edge[1]} / {self.channel_bandwidth[f'{edge[0]}_{edge[1]}']} + "
                value += f"channel_bandwidth_{edge[1]}_{edge[0]} / {self.channel_bandwidth[f'{edge[1]}_{edge[0]}']} + "
            value = value[:-3] + f") / {len(list(self.G.edges()))}" 
            reward_bandwidth_availability.add_contribution(Contribution(command, condition, value))


            # generation of the rewards for network availability and degradation
            command = ""
            condition = "true"
            value = "("
            for node in list(self.G.nodes()):
                value += f"(node_state_{node} = {consts['const_on']} ? 1 : 0) + "
            for edge in list(self.G.edges()):
                value += f"(channel_state_{edge[0]}_{edge[1]} != {consts['const_failure']} ? 1 : 0) + "
                value += f"(channel_state_{edge[1]}_{edge[0]} != {consts['const_failure']} ? 1 : 0) + "
                value += f"(interface_state_{edge[0]}_{edge[1]} != {consts['const_failure']} & interface_state_{edge[0]}_{edge[1]} != {consts['const_off']} ? 1 : 0) + "
                value += f"(interface_state_{edge[1]}_{edge[0]} != {consts['const_failure']} & interface_state_{edge[1]}_{edge[0]} != {consts['const_off']} ? 1 : 0) + "
            value = value[:-3] + f") / {(len(list(self.G.nodes())) + 4 * len(list(self.G.edges())))}"
            reward_network_availability.add_contribution(Contribution(command, condition, value))

            value = "("
            for edge in list(self.G.edges()):
                value += f"(channel_state_{edge[0]}_{edge[1]} = {consts['const_error']} ? 1 : 0) + "
                value += f"(channel_state_{edge[1]}_{edge[0]} = {consts['const_error']} ? 1 : 0) + "
                value += f"(interface_state_{edge[0]}_{edge[1]} = {consts['const_error']} ? 1 : 0) + "
                value += f"(interface_state_{edge[1]}_{edge[0]} = {consts['const_error']} ? 1 : 0) + "
            value = value[:-3] + f") / {(4 * len(list(self.G.edges())))}"
            reward_network_degradation.add_contribution(Contribution(command, condition, value))
            
            self.rewards.extend(list(rewards_data.values()) + list(rewards_system.values()) + [reward_message_sent, reward_message_received, reward_message_lost, reward_message_check_success, reward_message_check_failure, reward_bandwidth_availability, reward_memory_availability, reward_network_vulnerability, reward_network_availability, reward_network_degradation])
        
        generate_message_reward()


    # ── JSON serialisation ───────────────────────────────────────────────────

    def _generate_json(self) -> dict:
        """Assemble the complete PRISM-ready network descriptor.

        The output dictionary has the shape::

            {
                "consts": { ... },    # shared PRISM constants + abstraction level
                "items": [
                    { "name": "node_modules",            ... },
                    { "name": "interface_modules",       ... },
                    { "name": "channel_modules",         ... },
                    { "name": "link_modules",            ... },
                    { "name": "link_ref_modules",        ... },
                    { "name": "session_path_modules",    ... },
                    { "name": "local_session_modules",   ... },
                    { "name": "session_checker_modules",  ... },
                    { "name": "rewards_modules",         ... }
                ]
            }

        Each item in ``items`` is produced by the corresponding
        ``_add_<component>_to_dict()`` method.  The order matters for the
        preprocessor, which iterates ``items`` sequentially.

        Returns
        -------
        dict
            The full network descriptor.
        """
        # Load shared constants from the external JSON files
        with open("config/netgen_files.json", "r") as f:
            files_config = json.load(f)

        with open(files_config.get("consts"), "r") as f:
            consts = json.load(f)
       
        
        network_dict = {}

        consts["abstraction"] = self.params["abstraction"]
        network_dict["consts"] = consts
        network_dict["items"] = []

        # add nodes
        nodes_dict = self._add_nodes_to_dict()
        network_dict["items"].append(nodes_dict)

        # add interfaces
        interfaces_dict = self._add_interfaces_to_dict()
        network_dict["items"].append(interfaces_dict)

        # add channels
        channels_dict = self._add_channels_to_dict()
        network_dict["items"].append(channels_dict)

        # add links
        links_dict = self._add_links_to_dict()
        network_dict["items"].append(links_dict)

        # add link_refs
        link_refs_dict = self._add_link_refs_to_dict()
        network_dict["items"].append(link_refs_dict)

        # add session_paths
        session_paths_dict = self._add_session_paths_to_dict()
        network_dict["items"].append(session_paths_dict)

        # add local sessions
        local_sessions_dict = self._add_local_sessions_to_dict()
        network_dict["items"].append(local_sessions_dict)

        # add session_checkers
        session_checkers_dict = self._add_session_checkers_to_dict()
        network_dict["items"].append(session_checkers_dict)

        # add rewards
        self._generate_rewards()
        rewards_dict = self._add_rewards_to_dict()
        network_dict["items"].append(rewards_dict)

        return network_dict

    
    def _generate_sessions_summary(self) -> str:
        """Generate a human-readable text summary of all sessions.

        The summary provides a clearly formatted, indented view of every
        session, its paths (with sender -> receiver flow) and any
        sub-sessions.  It is saved alongside the main network JSON for
        debugging and documentation purposes.

        Returns
        -------
        str
            Multi-line text summary.
        """
        lines = []
        sep = "=" * 70
        thin_sep = "-" * 50

        lines.append(sep)
        lines.append(f"  SESSIONS SUMMARY — {len(self.sessions)} session(s)")
        lines.append(sep)
        lines.append("")

        for idx, session in enumerate(self.sessions):
            lines.append(f"  Session {session.get_id()}")
            lines.append(f"    Protocol : {session.protocol}")
            lines.append(f"    Nodes    : {session.get_nodes()}")
            lines.append(f"    Paths    : {len(session.get_paths())}")

            for i, path in enumerate(session.get_paths()):
                nodes = list(path.nodes())
                sender = nodes[0]
                receivers = [n for n, attr in path.nodes(data=True) if attr.get("is_receiver")]
                hops = " -> ".join(str(n) for n in nodes)
                lines.append(f"      Path {i}: {hops}")
                lines.append(f"               Sender    : node {sender}")
                lines.append(f"               Receivers : {receivers}")

            subsessions = session.get_subsessions()
            if subsessions:
                lines.append(f"    Sub-sessions : {len(subsessions)}")
                for sub in subsessions:
                    lines.append(f"      Sub-session {sub.get_id()}")
                    lines.append(f"        Protocol : {sub.protocol}")
                    lines.append(f"        Nodes    : {sub.get_nodes()}")
                    for j, path in enumerate(sub.get_paths()):
                        nodes = list(path.nodes())
                        sender = nodes[0]
                        receivers = [n for n, attr in path.nodes(data=True) if attr.get("is_receiver")]
                        hops = " -> ".join(str(n) for n in nodes)
                        lines.append(f"          Path {j}: {hops}")
                        lines.append(f"                   Sender    : node {sender}")
                        lines.append(f"                   Receivers : {receivers}")

            if idx < len(self.sessions) - 1:
                lines.append(f"  {thin_sep}")

        lines.append("")
        lines.append(sep)
        return "\n".join(lines)

    
    # ── Component serialisation helpers ─────────────────────────────────────
    # Each ``_add_<component>_to_dict`` method reads the shared config files
    # (ranges, init, consts) and iterates over the topology graph and/or
    # sessions to produce a dict with keys ``name``, ``template`` and
    # ``instances``.  Every instance maps 1-to-1 to a PRISM module file
    # emitted by the preprocessor.

    def _add_nodes_to_dict(self) -> dict:
        """Serialise node PRISM modules.

        For each node in the topology graph, builds an instance dict containing:

        * State range and initial values (from ``ranges.json`` / ``init.json``).
        * Buffer size (sampled from ``buffer_size_range``).
        * Transition probabilities (``off_to_on``, ``on_to_off``).
        * Synchronisation commands for power-on, power-off and shutdown.
        * Cross-references to links arriving at this node, link_refs owned by
          this node, session paths departing from it, and session checkers.

        Side-effects
        ------------
        Populates ``self.nodes_buffer_sizes`` for downstream methods.

        Returns
        -------
        dict
            ``{"name": "node_modules", "template": ..., "instances": [...]}``
        """
        with open("config/netgen_files.json", "r") as f:
            files_config = json.load(f)

        with open(files_config.get("ranges"), "r") as f:
            ranges = json.load(f)

        with open(files_config.get("init"), "r") as f:
            init = json.load(f)


        nodes_dict = {}
        nodes_dict["name"] = "node_modules"
        nodes_dict["template"] = files_config.get("node_template")
        nodes_dict["instances"] = []
        range_state = ranges.get("node_range_state")
        sizes = {}
        nodes = list(self.G.nodes())
        for node in nodes:
            node_to_add = {}
            node_to_add["name"] = f"node_{node}.prism"
            node_to_add["#"] = node
            node_to_add["range_state"] = range_state
            node_to_add["init_state"] = init.get("node_init_state")
            node_to_add["init_on_to_off"] = init.get("node_on_to_off_init")
            if self.params["buffer_size_range"][0] == self.params["buffer_size_range"][1]:
                size_buffer = self.params["buffer_size_range"][0]
            else:
                size_buffer = int(self.rng.integers(self.params["buffer_size_range"][0], self.params["buffer_size_range"][1] + 1))
            node_to_add["size_buffer"] = size_buffer    
            sizes[node] = size_buffer
            node_to_add["init_buffer"] = init.get("node_init_buffer")
            if self.params["node_prob_off_to_on"][0] == self.params["node_prob_off_to_on"][1]:
                prob_off_to_on = self.params["node_prob_off_to_on"][0]
            else:
                prob_off_to_on = round(self.rng.uniform(self.params["node_prob_off_to_on"][0], self.params["node_prob_off_to_on"][1]), 2)
            node_to_add["prob_off_to_on"] = prob_off_to_on
            if self.params["node_prob_on_to_off"][0] == self.params["node_prob_on_to_off"][1]:
                prob_on_to_off = self.params["node_prob_on_to_off"][0]
            else:
                prob_on_to_off = round(self.rng.uniform(self.params["node_prob_on_to_off"][0], self.params["node_prob_on_to_off"][1]), 2)
            node_to_add["prob_on_to_off"] = prob_on_to_off

            # commands
            node_to_add["cmd_off_to_on"] = f"cmd_off_to_on_node_{node}"
            node_to_add["cmd_on_to_off"] = f"cmd_on_to_off_node_{node}"
            node_to_add["cmd_shutting_down"] = f"cmd_shutting_down_node_{node}"

            # Collect all links that deliver messages TO this node
            # (used for buffer decrement synchronisation in the PRISM model).
            links = []
            sessions = self.sessions.copy()
            for session in self.sessions:
                ss = session.get_subsessions()
                sessions.extend(ss)
            for session in sessions:
                for i, path in enumerate(session.paths):
                    if node in path.nodes():
                        path_edges = list(path.edges())
                        for edge in path_edges:
                            if edge[1] == node:
                                link_name = f"link_{edge[0]}_{edge[1]}_of_path_{i}_{session.id}"
                                commands = {}
                                commands["cmd_receive_success"] = f"cmd_receive_success_{link_name}"
                                links.append(commands)

            node_to_add["links"] = links

            # Collect link_ref counters owned by this node (non-receiver roles)
            link_refs = []
            for session in sessions:
                for i, path in enumerate(session.paths):
                    if node in path.nodes() and path.nodes[node].get("is_receiver", False) == False:
                        commands = {}
                        link_ref_name = f"link_ref_{node}_of_path_{i}_{session.id}"
                        commands["cmd_reset"] = f"cmd_reset_{link_ref_name}"
                        link_refs.append(commands)

            node_to_add["link_refs"] = link_refs

            # Collect session paths where this node is the sender
            session_paths = []
            for session in sessions:
                for i, path in enumerate(session.paths):
                    if node in path.nodes() and path.nodes[node].get("is_receiver", False) == False:
                        commands = {}
                        session_path_name = f"session_path_{i}_{session.id}"
                        commands["cmd_send"] = f"cmd_send_{session_path_name}"
                        session_paths.append(commands)
                                
            node_to_add["session_paths"] = session_paths

            # Collect session checkers where this node is a receiver
            session_checkers = []
            for session in sessions:
                if node in session.nodes:
                    for i, path in enumerate(session.paths):
                        if node in path.nodes and path.nodes[node].get("is_receiver") == True:
                            commands = {}
                            session_checker_name = f"session_checker_of_node_{node}_of_path_{i}_{session.id}"
                            commands["cmd_cleanup"] = f"cmd_cleanup_{session_checker_name}"
                            commands["#"] = f"{node}_{i}_{session.id}"
                            session_checkers.append(commands)

            node_to_add["session_checkers"] = session_checkers

            nodes_dict["instances"].append(node_to_add)
        self.nodes_buffer_sizes = sizes
        return nodes_dict
            

    def _add_interfaces_to_dict(self) -> dict:
        """Serialise interface PRISM modules — two per undirected edge (A→B, B→A).

        Each interface instance records:

        * State range and initial value.
        * Six transition probabilities for the interface state machine
          (off→working, off→error, off→failure, working→error, error→working,
          failure→working).
        * Synchronisation commands for state transitions and node shutdown.
        * Lists of incoming / outgoing link commands that traverse this
          interface (derived by scanning all session paths).

        Returns
        -------
        dict
            ``{"name": "interface_modules", "template": ..., "instances": [...]}``
        """
        with open("config/netgen_files.json", "r") as f:
            files_config = json.load(f)

        with open(files_config.get("ranges"), "r") as f:
            ranges = json.load(f)

        with open(files_config.get("init"), "r") as f:
            init = json.load(f)


        interfaces_dict = {}
        interfaces_dict["name"] = "interface_modules"
        interfaces_dict["template"] = files_config.get("interface_template")
        interfaces_dict["instances"] = []
        range_state = ranges.get("interface_range_state")
        for edge in self.G.edges():
            node_a = edge[0]
            node_b = edge[1]

            # interface from node_a to node_b
            interface_ab = {}
            interface_ab["name"] = f"interface_{node_a}_{node_b}.prism"
            interface_ab["#"] = f"{node_a}_{node_b}"
            interface_ab["range_state"] = range_state
            interface_ab["init_state"] = init.get("interface_init_state")
           
            # probabilities
            if self.params["if_prob_off_to_working"][0] == self.params["if_prob_off_to_working"][1]:
                interface_ab["prob_off_to_working"] = self.params["if_prob_off_to_working"][0]
            else:
                interface_ab["prob_off_to_working"] = round(self.rng.uniform(self.params["if_prob_off_to_working"][0], self.params["if_prob_off_to_working"][1]), 2)
            if self.params["if_prob_off_to_error"][0] == self.params["if_prob_off_to_error"][1]:
                interface_ab["prob_off_to_error"] = self.params["if_prob_off_to_error"][0]
            else:
                interface_ab["prob_off_to_error"] = round(self.rng.uniform(self.params["if_prob_off_to_error"][0], self.params["if_prob_off_to_error"][1]), 2)
            
            if self.params["if_prob_off_to_failure"][0] == self.params["if_prob_off_to_failure"][1]:
                interface_ab["prob_off_to_failure"] = self.params["if_prob_off_to_failure"][0]
            else:
                interface_ab["prob_off_to_failure"] = round(self.rng.uniform(self.params["if_prob_off_to_failure"][0], self.params["if_prob_off_to_failure"][1]), 2)
            if self.params["if_prob_working_to_error"][0] == self.params["if_prob_working_to_error"][1]:
                interface_ab["prob_working_to_error"] = self.params["if_prob_working_to_error"][0]
            else:
                interface_ab["prob_working_to_error"] = round(self.rng.uniform(self.params["if_prob_working_to_error"][0], self.params["if_prob_working_to_error"][1]), 2)
            if self.params["if_prob_error_to_working"][0] == self.params["if_prob_error_to_working"][1]:
                interface_ab["prob_error_to_working"] = self.params["if_prob_error_to_working"][0]
            else:
                interface_ab["prob_error_to_working"] = round(self.rng.uniform(self.params["if_prob_error_to_working"][0], self.params["if_prob_error_to_working"][1]), 2)
            if self.params["if_prob_failure_to_working"][0] == self.params["if_prob_failure_to_working"][1]:
                interface_ab["prob_failure_to_working"] = self.params["if_prob_failure_to_working"][0]
            else:
                interface_ab["prob_failure_to_working"] = round(self.rng.uniform(self.params["if_prob_failure_to_working"][0], self.params["if_prob_failure_to_working"][1]), 2)
            
            interface_ab["ref_node"] = f"{node_a}"

            # commands
            interface_ab["cmd_off_to_on"] = f"cmd_off_to_on_interface_{node_a}_{node_b}"
            interface_ab["cmd_working_to_error"] = f"cmd_working_to_error_interface_{node_a}_{node_b}"
            interface_ab["cmd_error_to_working"] = f"cmd_error_to_working_interface_{node_a}_{node_b}"
            interface_ab["cmd_failure_to_working"] = f"cmd_failure_to_working_interface_{node_a}_{node_b}"
            interface_ab["cmd_shutting_down"] = f"cmd_shutting_down_node_{node_a}"

            # interface from node_b to node_a
            interface_ba = {}
            interface_ba["name"] = f"interface_{node_b}_{node_a}.prism"
            interface_ba["#"] = f"{node_b}_{node_a}"
            interface_ba["range_state"] = range_state
            interface_ba["init_state"] = init.get("interface_init_state")

            # probabilities
            if self.params["if_prob_off_to_working"][0] == self.params["if_prob_off_to_working"][1]:
                interface_ba["prob_off_to_working"] = self.params["if_prob_off_to_working"][0]
            else:
                interface_ba["prob_off_to_working"] = round(self.rng.uniform(self.params["if_prob_off_to_working"][0], self.params["if_prob_off_to_working"][1]), 2)
            if self.params["if_prob_off_to_error"][0] == self.params["if_prob_off_to_error"][1]:
                interface_ba["prob_off_to_error"] = self.params["if_prob_off_to_error"][0]
            else:
                interface_ba["prob_off_to_error"] = round(self.rng.uniform(self.params["if_prob_off_to_error"][0], self.params["if_prob_off_to_error"][1]), 2)
            if self.params["if_prob_off_to_failure"][0] == self.params["if_prob_off_to_failure"][1]:
                interface_ba["prob_off_to_failure"] = self.params["if_prob_off_to_failure"][0]
            else:
                interface_ba["prob_off_to_failure"] = round(self.rng.uniform(self.params["if_prob_off_to_failure"][0], self.params["if_prob_off_to_failure"][1]), 2)
            if self.params["if_prob_working_to_error"][0] == self.params["if_prob_working_to_error"][1]:
                interface_ba["prob_working_to_error"] = self.params["if_prob_working_to_error"][0]
            else:
                interface_ba["prob_working_to_error"] = round(self.rng.uniform(self.params["if_prob_working_to_error"][0], self.params["if_prob_working_to_error"][1]), 2)
            if self.params["if_prob_error_to_working"][0] == self.params["if_prob_error_to_working"][1]:
                interface_ba["prob_error_to_working"] = self.params["if_prob_error_to_working"][0]  
            else:
                interface_ba["prob_error_to_working"] = round(self.rng.uniform(self.params["if_prob_error_to_working"][0], self.params["if_prob_error_to_working"][1]), 2)
            if self.params["if_prob_failure_to_working"][0] == self.params["if_prob_failure_to_working"][1]:
                interface_ba["prob_failure_to_working"] = self.params["if_prob_failure_to_working"][0]  
            else:
                interface_ba["prob_failure_to_working"] = round(self.rng.uniform(self.params["if_prob_failure_to_working"][0], self.params["if_prob_failure_to_working"][1]), 2)

            # commands
            interface_ba["cmd_off_to_on"] = f"cmd_off_to_on_interface_{node_b}_{node_a}"
            interface_ba["cmd_working_to_error"] = f"cmd_working_to_error_interface_{node_b}_{node_a}"
            interface_ba["cmd_error_to_working"] = f"cmd_error_to_working_interface_{node_b}_{node_a}"
            interface_ba["cmd_failure_to_working"] = f"cmd_failure_to_working_interface_{node_b}_{node_a}"
            interface_ba["cmd_shutting_down"] = f"cmd_shutting_down_node_{node_b}"

            interface_ba["ref_node"] = f"{node_b}"


            # Build in/out link command lists by scanning all session paths
            # that traverse this edge in either direction.
            in_links_ab = []
            out_links_ab = []
            in_links_ba = []
            out_links_ba = []

            sessions = self.sessions.copy()
            for session in self.sessions:
                ss = session.get_subsessions()
                sessions.extend(ss)
            
            for session in sessions:
                for i, path in enumerate(session.paths):
                    id = session.id
                    session_path = f"{i}_{id}" 
                    edges = list(path.edges())
                    for edge in edges:
                        if edge[0] == node_a and edge[1] == node_b:
                            commands = {}
                            commands["cmd_send_failure"] = f"cmd_send_failure_link_{node_a}_{node_b}_of_path_{session_path}"
                            out_links_ab.append(commands)

                            commands = {}
                            commands["cmd_receive_failure"] = f"cmd_receive_failure_link_{node_a}_{node_b}_of_path_{session_path}"
                            in_links_ba.append(commands)

                        if edge[0] == node_b and edge[1] == node_a:
                            commands = {}
                            commands["cmd_send_failure"] = f"cmd_send_failure_link_{node_b}_{node_a}_of_path_{session_path}"
                            out_links_ba.append(commands)

                            commands = {}
                            commands["cmd_receive_failure"] = f"cmd_receive_failure_link_{node_b}_{node_a}_of_path_{session_path}"
                            in_links_ab.append(commands)
            
            interface_ab["in_links"] = in_links_ab
            interface_ab["out_links"] = out_links_ab
            interface_ba["in_links"] = in_links_ba
            interface_ba["out_links"] = out_links_ba

            interfaces_dict["instances"].append(interface_ab)
            interfaces_dict["instances"].append(interface_ba)

        return interfaces_dict


    def _add_channels_to_dict(self) -> dict:
        """Serialise channel PRISM modules — two per undirected edge (A→B, B→A).

        Each channel instance records:

        * State range, initial state, bandwidth capacity and initial bandwidth.
        * Transition probabilities (working→error, error→working,
          failure→working).
        * Synchronisation commands for state transitions.
        * Lists of link commands that use this channel (send/receive
          success/failure), derived by scanning all session paths.

        Side-effects
        ------------
        Populates ``self.channel_bandwidth`` for reward generation.

        Returns
        -------
        dict
            ``{"name": "channel_modules", "template": ..., "instances": [...]}``
        """
        with open("config/netgen_files.json", "r") as f:
            files_config = json.load(f)

        with open(files_config.get("ranges"), "r") as f:
            ranges = json.load(f)

        with open(files_config.get("init"), "r") as f:
            init = json.load(f)

        bandwidth_sizes = {}
        channels_dict = {}
        channels_dict["name"] = "channel_modules"
        channels_dict["template"] = files_config.get("channel_template")
        channels_dict["instances"] = []
        range_state = ranges.get("channel_range_state")
        for edge in self.G.edges():
            node_a = edge[0]
            node_b = edge[1]

            # channel a b
            channel = {}
            channel["name"] = f"channel_{node_a}_{node_b}.prism"
            channel["#"] = f"{node_a}_{node_b}"
            channel["range_state"] = range_state
            channel["init_state"] = init.get("channel_init_state")
            if self.params["channel_bandwidth_range"][0] == self.params["channel_bandwidth_range"][1]:
                size_bandwidth = self.params["channel_bandwidth_range"][0]
            else:
                size_bandwidth = int(self.rng.integers(self.params["channel_bandwidth_range"][0], self.params["channel_bandwidth_range"][1] + 1))
            channel["size_bandwidth"] = size_bandwidth
            channel["init_bandwidth"] = size_bandwidth
            bandwidth_sizes[f"{node_a}_{node_b}"] = size_bandwidth
            # probabilities
            if self.params["channel_prob_working_to_error"][0] == self.params["channel_prob_working_to_error"][1]:
                channel["prob_working_to_error"] = self.params["channel_prob_working_to_error"][0]
            else:
                channel["prob_working_to_error"] = round(self.rng.uniform(self.params["channel_prob_working_to_error"][0], self.params["channel_prob_working_to_error"][1]), 2)
            if self.params["channel_prob_error_to_working"][0] == self.params["channel_prob_error_to_working"][1]:
                channel["prob_error_to_working"] = self.params["channel_prob_error_to_working"][0]
            else:
                channel["prob_error_to_working"] = round(self.rng.uniform(self.params["channel_prob_error_to_working"][0], self.params["channel_prob_error_to_working"][1]), 2)
            if self.params["channel_prob_failure_to_working"][0] == self.params["channel_prob_failure_to_working"][1]:
                channel["prob_failure_to_working"] = self.params["channel_prob_failure_to_working"][0]
            else:
                channel["prob_failure_to_working"] = round(self.rng.uniform(self.params["channel_prob_failure_to_working"][0], self.params["channel_prob_failure_to_working"][1]), 2)
            
            # commands
            channel["cmd_working_to_error"] = f"cmd_working_to_error_channel_{node_a}_{node_b}"
            channel["cmd_error_to_working"] = f"cmd_error_to_working_channel_{node_a}_{node_b}"
            channel["cmd_failure_to_working"] = f"cmd_failure_to_working_channel_{node_a}_{node_b}"
            
            # channel b a
            channel_ba = {}
            channel_ba["name"] = f"channel_{node_b}_{node_a}.prism"
            channel_ba["#"] = f"{node_b}_{node_a}"
            channel_ba["range_state"] = range_state
            channel_ba["init_state"] = init.get("channel_init_state")  
            if self.params["channel_bandwidth_range"][0] == self.params["channel_bandwidth_range"][1]:
                size_bandwidth = self.params["channel_bandwidth_range"][0]
            else:
                size_bandwidth = int(self.rng.integers(self.params["channel_bandwidth_range"][0], self.params["channel_bandwidth_range"][1] + 1))
            channel_ba["size_bandwidth"] = size_bandwidth
            channel_ba["init_bandwidth"] = size_bandwidth   
            bandwidth_sizes[f"{node_b}_{node_a}"] = size_bandwidth

            # probabilities
            if self.params["channel_prob_working_to_error"][0] == self.params["channel_prob_working_to_error"][1]:
                channel_ba["prob_working_to_error"] = self.params["channel_prob_working_to_error"][0]
            else:
                channel_ba["prob_working_to_error"] = round(self.rng.uniform(self.params["channel_prob_working_to_error"][0], self.params["channel_prob_working_to_error"][1]), 2)
            if self.params["channel_prob_error_to_working"][0] == self.params["channel_prob_error_to_working"][1]:
                channel_ba["prob_error_to_working"] = self.params["channel_prob_error_to_working"][0]
            else:
                channel_ba["prob_error_to_working"] = round(self.rng.uniform(self.params["channel_prob_error_to_working"][0], self.params["channel_prob_error_to_working"][1]), 2)
            if self.params["channel_prob_failure_to_working"][0] == self.params["channel_prob_failure_to_working"][1]:
                channel_ba["prob_failure_to_working"] = self.params["channel_prob_failure_to_working"][0]
            else:
                channel_ba["prob_failure_to_working"] = round(self.rng.uniform(self.params["channel_prob_failure_to_working"][0], self.params["channel_prob_failure_to_working"][1]), 2)    

            # commands
            channel_ba["cmd_working_to_error"] = f"cmd_working_to_error_channel_{node_b}_{node_a}"
            channel_ba["cmd_error_to_working"] = f"cmd_error_to_working_channel_{node_b}_{node_a}"
            channel_ba["cmd_failure_to_working"] = f"cmd_failure_to_working_channel_{node_b}_{node_a}"


            # Scan all session paths to find links traversing this channel
            # and collect their synchronisation commands.
            links_ab = []
            links_ba = []
            sessions = self.sessions.copy()
            for session in self.sessions:
                sessions.extend(session.get_subsessions())
            for session in sessions:
                session_id = session.get_id()
                for i, path in enumerate(session.paths):
                    for edge in path.edges():
                        if (edge[0] == node_a) and (edge[1] == node_b):
                            # this link uses the channel_ab
                            ref = f"link_{node_a}_{node_b}_of_path_{i}_{session_id}"
                            commands = {}
                            commands["cmd_send_failure"] = f"cmd_send_failure_{ref}"
                            commands["cmd_send_success"] = f"cmd_send_success_{ref}"
                            commands["cmd_receive_success"] = f"cmd_receive_success_{ref}"
                            commands["cmd_receive_failure"] = f"cmd_receive_failure_{ref}"
                            links_ab.append(commands)

                        if (edge[0] == node_b) and (edge[1] == node_a):
                            # this link uses the channel_ba
                            ref = f"link_{node_b}_{node_a}_of_path_{i}_{session_id}"
                            commands = {}
                            commands["cmd_send_failure"] = f"cmd_send_failure_{ref}"
                            commands["cmd_send_success"] = f"cmd_send_success_{ref}"
                            commands["cmd_receive_success"] = f"cmd_receive_success_{ref}"
                            commands["cmd_receive_failure"] = f"cmd_receive_failure_{ref}"
                            links_ba.append(commands)

            channel["links"] = links_ab
            channel_ba["links"] = links_ba
            self.channel_bandwidth = bandwidth_sizes

            channels_dict["instances"].append(channel)
            channels_dict["instances"].append(channel_ba)

        return channels_dict


    def _add_links_to_dict(self) -> dict:
        """Serialise link PRISM modules — one per directed edge per path.

        Links are the per-hop transmission units.  Each instance records:

        * State and phase ranges, initial values, outcome flag.
        * References to the underlying channel, sender/receiver interfaces,
          sender/receiver node buffers, and channel bandwidth.
        * Transmission probabilities (working/error/failure transitions,
          retry, sending success).
        * Synchronisation commands for all link lifecycle events.
        * The ``cmd_link_start`` command, which is synchronised either with
          the session-path's ``cmd_send`` (for the first hop) or with the
          previous link's ``cmd_receive_success`` (for subsequent hops).

        Returns
        -------
        dict
            ``{"name": "link_modules", "template": ..., "instances": [...]}``
        """
        with open("config/netgen_files.json", "r") as f:
            files_config = json.load(f)

        with open(files_config.get("ranges"), "r") as f:
            ranges = json.load(f)

        with open(files_config.get("init"), "r") as f:
            init = json.load(f)

        # One link instance per directed edge per path across all sessions
        links_dict = {}
        links_dict["name"] = "link_modules"
        links_dict["template"] = files_config.get("link_template")
        links_dict["instances"] = []
        range_state = ranges.get("link_range_state")
        range_phase = ranges.get("link_range_phase")

        sessions = self.sessions.copy()
        for session in self.sessions:
            sessions.extend(session.get_subsessions())

        for session in sessions:
            session_id = session.get_id()
            for i, path in enumerate(session.paths):
                for edge in path.edges():
                    sender = list(path.nodes)[0]
                    receivers = list(path.nodes)[1:]
                    node_a = edge[0]
                    node_b = edge[1]

                    # links instances
                    link = {}
                    link["name"] = f"link_{node_a}_{node_b}_of_path_{i}_{session_id}.prism"
                    link["#"] = f"{node_a}_{node_b}_{i}_{session_id}"

                    # states
                    link["range_state"] = range_state
                    link["init_state"] = init.get("link_init_state")
                    link["range_phase"] = range_phase
                    link["init_phase"] = init.get("link_init_phase")
                    link["init_outcome"] = "false"

                    # references
                    link["ref_channel"] = f"{node_a}_{node_b}"
                    link["ref_interface_sender"] = f"{node_a}_{node_b}"
                    link["ref_node_buffer_sender"] = f"{node_a}"
                    # link["ref_session_path"] = f"{i}_{session_id}"
                    # link["ref_link_ref_counter"] = f"{node_a}_{i}_{session_id}"
                    # next_links = []
                    # successors = list(path.successors(node_b))
                    # for succ in successors:
                    #     next_links.append({"ref_link_next": f"{node_b}_{succ}_{i}_{session_id}"})
                    # link["next_links"] = next_links
                    # link["number_next_links"] = len(successors)
                    link["ref_interface_receiver"] = f"{node_b}_{node_a}"
                    link["ref_node_buffer_receiver"] = f"{node_b}"
                    link["size_node_buffer_receiver"] = self.nodes_buffer_sizes[node_b]
                    link["ref_channel_bandwidth"] = f"{node_a}_{node_b}"

                    # probabilities
                    if self.params["link_prob_working_to_error"][0] == self.params["link_prob_working_to_error"][1]:
                        link["prob_working_to_error"] = self.params["link_prob_working_to_error"][0]
                    else:
                        link["prob_working_to_error"] = round(self.rng.uniform(self.params["link_prob_working_to_error"][0], self.params["link_prob_working_to_error"][1]), 2)
                    if self.params["link_prob_error_to_working"][0] == self.params["link_prob_error_to_working"][1]:
                        link["prob_error_to_working"] = self.params["link_prob_error_to_working"][0]
                    else:
                        link["prob_error_to_working"] = round(self.rng.uniform(self.params["link_prob_error_to_working"][0], self.params["link_prob_error_to_working"][1]), 2)
                    
                    if self.params["link_prob_failure_to_working"][0] == self.params["link_prob_failure_to_working"][1]:
                        link["prob_failure_to_working"] = self.params["link_prob_failure_to_working"][0]
                    else:
                        link["prob_failure_to_working"] = round(self.rng.uniform(self.params["link_prob_failure_to_working"][0], self.params["link_prob_failure_to_working"][1]), 2)
                    if self.params["link_prob_retry"][0] == self.params["link_prob_retry"][1]:
                        link["prob_retry"] = self.params["link_prob_retry"][0]
                    else:
                        link["prob_retry"] = round(self.rng.uniform(self.params["link_prob_retry"][0], self.params["link_prob_retry"][1]), 2)
                    if self.params["link_prob_sending"][0] == self.params["link_prob_sending"][1]:
                        link["prob_sending"] = self.params["link_prob_sending"][0]
                    else:
                        link["prob_sending"] = round(self.rng.uniform(self.params["link_prob_sending"][0], self.params["link_prob_sending"][1]), 2)

                    # commands
                    link["cmd_working_to_error"] = f"cmd_working_to_error_link_{node_a}_{node_b}_of_path_{i}_{session_id}"
                    link["cmd_error_to_working"] = f"cmd_error_to_working_link_{node_a}_{node_b}_of_path_{i}_{session_id}"
                    link["cmd_failure_to_working"] = f"cmd_failure_to_working_link_{node_a}_{node_b}_of_path_{i}_{session_id}"
                    link["cmd_send_failure"] = f"cmd_send_failure_link_{node_a}_{node_b}_of_path_{i}_{session_id}"
                    link["cmd_send_success"] = f"cmd_send_success_link_{node_a}_{node_b}_of_path_{i}_{session_id}"
                    link["cmd_sending"] = f"cmd_sending_link_{node_a}_{node_b}_of_path_{i}_{session_id}"
                    link["cmd_receive_success"] = f"cmd_receive_success_link_{node_a}_{node_b}_of_path_{i}_{session_id}"
                    link["cmd_receive_failure"] = f"cmd_receive_failure_link_{node_a}_{node_b}_of_path_{i}_{session_id}"
                    link["cmd_link_cleanup"] = f"cmd_link_cleanup_link_{node_a}_{node_b}_of_path_{i}_{session_id}"

                    # Determine the link start trigger:
                    # - First hop: synchronised with the session path's send command.
                    # - Subsequent hops: synchronised with the previous link's receive_success.
                    if node_a == sender:
                        link["cmd_link_start"] = f"cmd_send_session_path_{i}_{session_id}"
                    else:
                        predecessor = list(path.predecessors(node_a))[0]
                        link["cmd_link_start"] = f"cmd_receive_success_link_{predecessor}_{node_a}_of_path_{i}_{session_id}"
                    
                    links_dict["instances"].append(link)

        return links_dict

    
    def _add_link_refs_to_dict(self) -> dict:
        """Serialise link-reference-counter PRISM modules.

        A link_ref tracks how many outgoing links from a given node within a
        given path have completed their cleanup phase.  This is used to
        coordinate fan-out in broadcast trees: the node waits for all child
        links to finish before proceeding.

        One instance is created for every non-leaf node in every path.

        Returns
        -------
        dict
            ``{"name": "link_ref_modules", "template": ..., "instances": [...]}``
        """
        with open("config/netgen_files.json", "r") as f:
            files_config = json.load(f)

        link_refs_dict = {}
        link_refs_dict["name"] = "link_ref_modules"
        link_refs_dict["template"] = files_config.get("link_ref_template")
        link_refs_dict["instances"] = []

        sessions = self.sessions.copy()
        for session in self.sessions:
            sessions.extend(session.get_subsessions())

        for session in sessions:
            session_id = session.get_id()
            for i, path in enumerate(session.paths):
                sender = list(path.nodes)[0]
                for node in path.nodes:
                    # link_ref_counter instances
                    size_counter = len(list(path.successors(node)))
                    if size_counter == 0:
                        continue
                    link_ref = {}
                    link_ref["name"] = f"link_ref_{node}_of_path_{i}_{session_id}.prism"
                    link_ref["#"] = f"{node}_{i}_{session_id}"
                    link_ref["size_counter"] = size_counter
                    link_ref["init_counter"] = size_counter
                    link_ref["is_receiver"] = path.nodes[node].get("is_receiver", False)
                    link_ref["ref_node"] = f"{node}"
                    link_ref["cmd_reset"] = f"cmd_reset_link_ref_{node}_of_path_{i}_{session_id}"
                    links = []
                    for successor in path.successors(node):
                        link_name = f"link_{node}_{successor}_of_path_{i}_{session_id}"
                        links.append({"cmd_link_cleanup": f"cmd_link_cleanup_{link_name}"})
                    link_ref["links"] = links

                    link_refs_dict["instances"].append(link_ref)
                    

        return link_refs_dict
        

    def _add_session_paths_to_dict(self) -> dict:
        """Serialise session-path PRISM modules.

        A session path represents one directed path tree within a session.
        Each instance records:

        * Protocol type, state/message ranges and initial values.
        * Epoch cache size and receiver counter.
        * Reference to the sender node and local session.
        * Scheduling probability (``prob_run``), forced to 0 for support
          sub-protocols.
        * Protocol-specific commands and flags, dispatched via a
          ``match session.protocol`` block covering all six protocol variants
          (hpke, double_ratchet, hpke_sender_key, double_ratchet_sender_key,
          sender_key, mls).
        * Lists of session-checker cleanup commands.

        Returns
        -------
        dict
            ``{"name": "session_path_modules", "template": ..., "instances": [...]}``
        """

        with open("config/netgen_files.json", "r") as f:
            files_config = json.load(f)

        with open(files_config.get("ranges"), "r") as f:
            ranges = json.load(f)

        with open(files_config.get("consts"), "r") as f:
            consts = json.load(f)

        with open(files_config.get("init"), "r") as f:
            init = json.load(f)

        session_paths_dict = {}
        session_paths_dict["name"] = "session_path_modules"
        session_paths_dict["template"] = files_config.get("session_path_template")
        session_paths_dict["instances"] = []
        range_state = ranges.get("session_path_state_range")
        range_system_message = ranges.get("session_path_system_message_range")
        range_data_message = ranges.get("session_path_data_message_range")
        range_return_pending = ranges.get("session_path_return_pending_range")
 

            
        sessions = self.sessions.copy()
        for s in self.sessions:
            subs = s.get_subsessions()
            sessions.extend(subs)

        for session in sessions:
            session_id = session.get_id()
            for i, path in enumerate(session.paths):
                sender = list(path.nodes)[0]
                receivers = self._get_receivers_from_path(path)    
                session_path = {}

                session_path["name"] = f"session_path_{i}_{session_id}.prism"
                session_path["#"] = f"{i}_{session_id}"

                session_path["protocol"] = session.protocol

                # states
                session_path["range_state"] = range_state
                session_path["init_state"] = init.get("session_path_init_state")
                session_path["range_system_message"] = range_system_message
                session_path["init_system_message"] = init.get("session_path_init_system_message")
                session_path["range_data_message"] = range_data_message
                session_path["init_data_message"] = init.get("session_path_init_data_message")
                session_path["size_cache_local_session_epoch_sender"] = path.nodes[sender]['epoch_size']
                session_path["init_cache_local_session_epoch_sender"] = init.get("session_path_init_cache_local_session_epoch_sender")
                session_path["size_receivers_counter"] = len(receivers)  # one session checker for each receiver
                session_path["init_receivers_counter"] = len(receivers)
                session_path["range_return_pending"] = range_return_pending
                session_path["init_return_pending"] = init.get("session_path_init_return_pending")

                # references
                session_path["ref_node_sender"] = f"{sender}"
                session_path["ref_local_session_sender"] = f"{sender}_{session_id}"
                session_path["size_buffer_sender"] = self.nodes_buffer_sizes[sender]

                # probabilities
                if session.protocol in {"hpke_sender_key", "double_ratchet_sender_key"}:
                    session_path["prob_run"] = 0.0
                else:
                    if self.params["sp_prob_run"][0] == self.params["sp_prob_run"][1]:
                        prob_run = self.params["sp_prob_run"][0]
                    else:
                        prob_run = round(self.rng.uniform(self.params["sp_prob_run"][0], self.params["sp_prob_run"][1]), 2)
                    session_path["prob_run"] = prob_run

                

                match session.protocol:
                    case "hpke":
                        session_path["is_broadest"] = "true"
                        session_path["cmd_hpke_reset"] = f"cmd_hpke_reset_session_path_{i}_{session_id}"
                        session_path["cmd_hpke_failure"] = f"cmd_hpke_failure_session_path_{i}_{session_id}"
                        session_path["cmd_alert"] = f"cmd_alert_session_path_{i}_{session_id}"

                    case "double_ratchet":
                        session_path["is_broadest"] = "true"
                        session_path["cmd_double_ratchet_reset"] = f"cmd_double_ratchet_reset_session_path_{i}_{session_id}"
                        session_path["cmd_double_ratchet_failure"] = f"cmd_double_ratchet_failure_session_path_{i}_{session_id}"
                        session_path["cmd_alert"] = f"cmd_alert_session_path_{i}_{session_id}"

                    case "hpke_sender_key":
                        session_path["is_broadest"] = "true"
                        session_path["cmd_hpke_reset"] = f"cmd_hpke_reset_session_path_{i}_{session_id}"
                        session_path["cmd_hpke_failure"] = f"cmd_hpke_failure_session_path_{i}_{session_id}"
                        ss = [ss for ss in sessions if session in ss.get_subsessions()][0]
                        ss_sender = ss.nodes[0]
                        if sender == ss_sender:
                            direction = "sender_to_receiver"
                            session_path["cmd_sub_run"] = f"cmd_sub_run_session_path_0_{ss.get_id()}"
                            session_path["cmd_resolve_sender_refresh"] = f"cmd_resolve_sender_refresh_session_path_{i}_{session_id}"
                        else:
                            direction = "receiver_to_sender"
                            session_path["cmd_sender_key_reset"] = f"cmd_sender_key_reset_session_path_{i}_{session_id}"
                            session_path["cmd_sender_key_failure"] = f"cmd_sender_key_failure_session_path_{i}_{session_id}"
                        session_path["direction"] = direction
                        session_path["cmd_alert"] = f"cmd_alert_session_path_{i}_{session_id}"

                    case "double_ratchet_sender_key":
                        session_path["is_broadest"] = "true"
                        session_path["cmd_double_ratchet_reset"] = f"cmd_double_ratchet_reset_session_path_{i}_{session_id}"
                        session_path["cmd_double_ratchet_failure"] = f"cmd_double_ratchet_failure_session_path_{i}_{session_id}"
                        ss = [ss for ss in sessions if session in ss.get_subsessions()][0]
                        ss_sender = ss.nodes[0]
                        if sender == ss_sender:
                            direction = "sender_to_receiver"
                            session_path["cmd_sub_run"] = f"cmd_sub_run_session_path_0_{ss.get_id()}"
                            session_path["cmd_resolve_sender_refresh"] = f"cmd_resolve_sender_refresh_session_path_{i}_{session_id}"
                        else:
                            direction = "receiver_to_sender"
                            session_path["cmd_sender_key_reset"] = f"cmd_sender_key_reset_session_path_{i}_{session_id}"
                            session_path["cmd_sender_key_failure"] = f"cmd_sender_key_failure_session_path_{i}_{session_id}"
                        session_path["direction"] = direction
                        session_path["cmd_alert"] = f"cmd_alert_session_path_{i}_{session_id}"
                        
                    case "sender_key":
                        session_path["is_broadest"] = "true"
                        self_session_checkers = []
                        ss = session.get_subsessions()
                        for subs in ss:
                            for j, subpath in enumerate(subs.paths):
                                rec = self._get_receivers_from_path(subpath)[0]
                                if rec == sender:
                                    self_session_checkers.append({"cmd_resolve_sender_reset": f"cmd_resolve_sender_reset_session_checker_of_node_{sender}_of_path_{j}_{subs.get_id()}"})
                        session_path["self_session_checkers"] = self_session_checkers
                        session_path["cmd_sub_run"] = f"cmd_sub_run_session_path_{i}_{session_id}"
                        session_path["cmd_sub_resolve"] = f"cmd_sub_resolve_session_path_{i}_{session_id}"
                        sub_session_paths = []
                        for subs in ss:
                            for s, subpath in enumerate(subs.paths):
                                if list(subpath.nodes)[0] == sender:
                                    sub_session_paths.append({"cmd_resolve": f"cmd_resolve_session_path_{s}_{subs.get_id()}"})
                        session_path["sub_session_paths"] = sub_session_paths
                        session_path["cmd_alert"] = f"cmd_alert_session_path_{i}_{session_id}"

                    case "mls":
                        is_broadest = True
                        one_to_one = False
                        for other_path in session.paths:
                            if other_path != path and list(other_path.nodes)[0] == sender:
                                if len(self._get_receivers_from_path(path)) < len(self._get_receivers_from_path(other_path)):
                                    is_broadest = False
                        if is_broadest:
                            session_path["is_broadest"] = "true"
                            if len(receivers) == 1:
                                session_path["one_to_one"] = "true"
                                one_to_one = True
                            else:
                                session_path["one_to_one"] = "false"
                                one_to_one = False
                        else:
                            session_path["is_broadest"] = "false"
                        self_session_checkers = []
                        for j, other_path in enumerate(session.paths):
                            if sender in self._get_receivers_from_path(other_path):
                                commands = {}
                                if is_broadest:
                                    commands["cmd_mls_reset"] =  f"cmd_mls_reset_session_checker_of_node_{sender}_of_path_{j}_{session_id}"
                                    if one_to_one:
                                        commands["cmd_mls_failure"] = f"cmd_mls_failure_session_checker_of_node_{sender}_of_path_{j}_{session_id}"
                                        commands["cmd_read_refresh"] = f"cmd_read_refresh_session_checker_of_node_{sender}_of_path_{j}_{session_id}"
                                else:
                                    commands["cmd_mls_failure"] = f"cmd_mls_failure_session_checker_of_node_{sender}_of_path_{j}_{session_id}"
                                    commands["cmd_read_refresh"] = f"cmd_read_refresh_session_checker_of_node_{sender}_of_path_{j}_{session_id}"
                                self_session_checkers.append(commands)
                        session_path["self_session_checkers"] = self_session_checkers
                        session_path["cmd_update_not_data"] = f"cmd_update_not_data_session_path_{i}_{session_id}"
                        if is_broadest:
                            session_path["cmd_alert"] = f"cmd_alert_session_path_{i}_{session_id}"
                                
                # commands
                session_path["cmd_run"] = f"cmd_run_session_path_{i}_{session_id}"
                session_path["cmd_process_pending"] = f"cmd_process_pending_session_path_{i}_{session_id}"
                session_path["cmd_update_data"] = f"cmd_update_data_session_path_{i}_{session_id}"
                session_path["cmd_set_message"] = f"cmd_set_message_session_path_{i}_{session_id}"
                session_path["cmd_send"] = f"cmd_send_session_path_{i}_{session_id}"
                session_path["cmd_resolve"] = f"cmd_resolve_session_path_{i}_{session_id}"
                session_checkers = []
                for r in receivers:
                    session_checkers.append({"cmd_cleanup": f"cmd_cleanup_session_checker_of_node_{r}_of_path_{i}_{session_id}"})
                session_path["session_checkers"] = session_checkers

                session_paths_dict["instances"].append(session_path)

        return session_paths_dict


    def _add_local_sessions_to_dict(self) -> dict:
        """Serialise local-session PRISM modules.

        Each node participating in a session holds a local session that tracks
        the session's cryptographic state (epoch counter, ratchet counter,
        compromised flag, mutex).  One instance is created per (node, session)
        pair.

        The method also resolves cross-references to the broadest session
        path (the one with the most receivers departing from this node) and
        the session checkers that notify this local session of incoming
        messages.

        Local-session transition probabilities (reset, ratchet, none,
        compromised) are sampled from their respective ranges with a
        normalisation step that ensures the sum stays below 1.0.

        Returns
        -------
        dict
            ``{"name": "local_session_modules", "template": ..., "instances": [...]}``
        """
        with open("config/netgen_files.json", "r") as f:
            files_config = json.load(f)

        with open(files_config.get("ranges"), "r") as f:
            ranges = json.load(f)

        with open(files_config.get("consts"), "r") as f:
            consts = json.load(f)

        with open(files_config.get("init"), "r") as f:
            init = json.load(f)

        local_sessions_dict = {}
        local_sessions_dict["name"] = "local_session_modules"
        local_sessions_dict["template"] = files_config.get("local_session_template")
        local_sessions_dict["instances"] = []
        range_state = ranges.get("local_session_range_state")

        sessions = self.sessions
        sessions_copy = sessions.copy()
        for session in sessions:
            subs = session.get_subsessions()
            sessions_copy.extend(subs)
        for session in sessions_copy:
            nodes = session.get_nodes()
            id = session.get_id()
            for node in nodes:
                local_session = {}
                local_session["name"] = f"local_session_of_{node}_of_session_{id}.prism"
                local_session["#"] = f"{node}_{id}"
                local_session["protocol"] = session.protocol

                # states
                local_session["range_state"] = range_state
                local_session["init_state"] = init.get("local_session_init_state")
                local_session["size_epoch"] = session.paths[0].nodes[list(session.paths[0].nodes)[0]]['epoch_size']
                local_session["init_epoch"] = init.get("local_session_init_epoch")
                local_session["size_ratchet"] = session.paths[0].nodes[list(session.paths[0].nodes)[0]]['ratchet_size']
                local_session["init_ratchet"] = init.get("local_session_init_ratchet")
                local_session["range_mutex"] = ranges.get("local_session_range_mutex")
                local_session["init_mutex"] = init.get("local_session_init_mutex")
                local_session["init_compromised"] = init.get("local_session_init_compromised")

                # references
                local_session["ref_node"] = f"{node}"

                session_paths = []
                for i, path in enumerate(session.paths):
                    if list(path.nodes)[0] == node:
                        commands = {}
                        commands["cmd_update_data"] = f"cmd_update_data_session_path_{i}_{id}"
                        commands["cmd_set_message"] = f"cmd_set_message_session_path_{i}_{id}"
                        session_paths.append(commands)

                local_session["session_paths"] = session_paths
                
                if session.protocol == "sender_key" and node != list(session.paths[0].nodes)[0]:
                    for ss in session.get_subsessions():
                        if node in ss.nodes:
                            i = [e[0] for e in enumerate(ss.paths) if node == list(e[1])[0]][0]
                            local_session["ref_broadest_session_path"] = f"{i}_{ss.get_id()}"
                            local_session["cmd_alert"] = f"cmd_alert_session_path_{i}_{ss.get_id()}"
                            break
                else:
                    i = [e[0] for e in enumerate(session.paths) if node == list(e[1])[0]][0]
                    local_session["ref_broadest_session_path"] = f"{i}_{id}"
                    local_session["cmd_alert"] = f"cmd_alert_session_path_{i}_{id}"

                # probabilities — generate and ensure sum < 1.0
                if self.params["ls_prob_session_reset"][0] == self.params["ls_prob_session_reset"][1]:
                    prob_reset = self.params["ls_prob_session_reset"][0]
                else:
                    prob_reset = round(self.rng.uniform(self.params["ls_prob_session_reset"][0], self.params["ls_prob_session_reset"][1]), 2)

                if self.params["ls_prob_ratchet_reset"][0] == self.params["ls_prob_ratchet_reset"][1]:
                    prob_ratchet = self.params["ls_prob_ratchet_reset"][0]
                else:
                    prob_ratchet = round(self.rng.uniform(self.params["ls_prob_ratchet_reset"][0], self.params["ls_prob_ratchet_reset"][1]), 2)

                if self.params["ls_prob_none"][0] == self.params["ls_prob_none"][1]:
                    prob_none = self.params["ls_prob_none"][0]
                else:
                    prob_none = round(self.rng.uniform(self.params["ls_prob_none"][0], self.params["ls_prob_none"][1]), 2)

                # Safety clamp: ensure prob_reset + prob_ratchet + prob_none < 1.0
                total = prob_reset + prob_ratchet + prob_none
                if total >= 1.0:
                    scale = 0.99 / total
                    prob_reset = round(prob_reset * scale, 2)
                    prob_ratchet = round(prob_ratchet * scale, 2)
                    prob_none = round(1.0 - prob_reset - prob_ratchet - 0.01, 2)

                local_session["prob_reset"] = prob_reset
                local_session["prob_ratchet"] = prob_ratchet
                local_session["prob_none"] = prob_none
                
                if self.params["ls_prob_compromised"][0] == self.params["ls_prob_compromised"][1]:
                    local_session["prob_compromised"] = self.params["ls_prob_compromised"][0]
                else:
                    local_session["prob_compromised"] = round(self.rng.uniform(self.params["ls_prob_compromised"][0], self.params["ls_prob_compromised"][1]), 2)

                # commands
                local_session["cmd_session_update"] = f"cmd_session_update_local_session_of_{node}_of_session_{id}"
                local_session["cmd_session_update_compromised"] = f"cmd_session_update_compromised_local_session_of_{node}_of_session_{id}"
                local_session["cmd_update"] = f"cmd_update_local_session_of_{node}_of_session_{id}"
                local_session["cmd_ratchet"] = f"cmd_ratchet_local_session_of_{node}_of_session_{id}"
                local_session["cmd_reset"] = f"cmd_reset_local_session_of_{node}_of_session_{id}"
                local_session["cmd_none"] = f"cmd_none_local_session_of_{node}_of_session_{id}"
                local_session["cmd_compromise"] = f"cmd_compromise_local_session_of_{node}_of_session_{id}"
                local_session["cmd_decompromise"] = f"cmd_resolve_session_path_{local_session['ref_broadest_session_path']}"

                # i need the list of all the session checker of this local session on this node
                session_checkers = []
                session_id = session.get_id()
                for i, path in enumerate(session.paths):
                    sender = list(path.nodes)[0]
                    receivers = self._get_receivers_from_path(path)
                    if node in receivers:
                        commands = {}
                        commands["cmd_read_data"] = f"cmd_read_data_session_checker_of_node_{node}_of_path_{i}_{session_id}"
                        commands["cmd_cleanup"] = f"cmd_cleanup_session_checker_of_node_{node}_of_path_{i}_{session_id}"
                        match session.protocol:
                            case "hpke" | "hpke_sender_key":
                                commands["cmd_read_ratchet"] = f"cmd_read_ratchet_session_checker_of_node_{node}_of_path_{i}_{session_id}"
                                commands["cmd_read_reset"] = f"cmd_read_reset_session_checker_of_node_{node}_of_path_{i}_{session_id}"
                                for j, local_path in enumerate(session.paths):
                                    path_list = list(local_path.nodes)
                                    if node == path_list[0] and sender == path_list[-1]:
                                        commands["cmd_hpke_failure"] = f"cmd_hpke_failure_session_path_{j}_{session_id}"

                            case "double_ratchet" | "double_ratchet_sender_key":
                                commands["cmd_read_ratchet"] = f"cmd_read_ratchet_session_checker_of_node_{node}_of_path_{i}_{session_id}"
                                commands["cmd_read_reset"] = f"cmd_read_reset_session_checker_of_node_{node}_of_path_{i}_{session_id}"
                                for j, local_path in enumerate(session.paths):
                                    path_list = list(local_path.nodes)
                                    if node == path_list[0] and sender == path_list[-1]:
                                        commands["cmd_double_ratchet_failure"] = f"cmd_double_ratchet_failure_session_path_{j}_{session_id}"
                                        
                            case "sender_key":
                                commands["cmd_resolve_sender_new_key"] = f"cmd_resolve_sender_new_key_session_checker_of_node_{node}_of_path_0_{session_id}"
                                commands["ref_session_path"] = f"0_{session_id}"

                            case "mls":
                                commands["cmd_read_ratchet"] = f"cmd_read_ratchet_session_checker_of_node_{node}_of_path_{i}_{session_id}"
                                commands["cmd_read_reset"] = f"cmd_read_reset_session_checker_of_node_{node}_of_path_{i}_{session_id}"
                                commands["cmd_read_current"] = f"cmd_read_current_session_checker_of_node_{node}_of_path_{i}_{session_id}"
                                commands["ref_session_path"] = f"{i}_{session_id}"
                        session_checkers.append(commands)

                local_session["session_checkers"] = session_checkers
                        
                local_sessions_dict["instances"].append(local_session)

        return local_sessions_dict


    def _get_receivers_from_path(self, path: nx.DiGraph) -> list[int]:
        """Return the list of receiver node ids in *path*.

        Receivers are nodes whose ``is_receiver`` attribute is ``True``.
        """
        return [node_id for node_id in path.nodes if path.nodes[node_id].get("is_receiver", False)]

                    
    def _add_session_checkers_to_dict(self) -> dict:
        """Serialise session-checker PRISM modules.

        A session checker is instantiated for **each receiver in each path**.
        It monitors whether a message has been successfully received and drives
        the protocol-specific verification / key-management logic.

        Protocol-specific behaviour is dispatched via a ``match`` block:

        * **hpke / double_ratchet**: references the return path to the sender
          for reset/failure commands; includes read_reset, read_ratchet and
          resolve_system commands.
        * **hpke_sender_key / double_ratchet_sender_key**: additionally
          references the supersession broadcast path and sender-refresh
          commands.
        * **sender_key**: references the sub-session path to the sender;
          includes sender_key reset/failure commands.
        * **mls**: includes read_refresh, read_current and mls_reset/failure
          commands; references the broadcast path.

        Common commands (freeze, trigger, read_data, check_success/failure,
        resolve_data, cleanup, update) are appended for all protocols.
        The checker also records the chain of links from sender to receiver
        for coordinated cleanup.

        Returns
        -------
        dict
            ``{"name": "session_checker_modules", "template": ..., "instances": [...]}``
        """
        # One session checker for each receiver in each path

        with open("config/netgen_files.json", "r") as f:
            files_config = json.load(f)

        with open(files_config.get("ranges"), "r") as f:
            ranges = json.load(f)

        with open(files_config.get("consts"), "r") as f:
            consts = json.load(f)

        with open(files_config.get("init"), "r") as f:
            init = json.load(f)

        state_range = ranges.get("session_checker_state_range")

        session_checkers_dict = {}
        session_checkers_dict["name"] = "session_checker_modules"
        session_checkers_dict["template"] = files_config.get("session_checker_template")
        session_checkers_dict["instances"] = []

        sessions = self.sessions.copy()
        for session in self.sessions:
            sessions.extend(session.get_subsessions())

        for session in sessions:
            session_id = session.get_id()
            for i, path in enumerate(session.paths):
                sender = list(path)[0]
                receivers = self._get_receivers_from_path(path)
                
                for node in receivers:
                    session_checker = {}
                    session_checker["name"] = f"session_checker_of_node_{node}_of_path_{i}_{session_id}.prism"
                    session_checker["#"] = f"{node}_{i}_{session_id}"
                    session_checker["protocol"] = session.protocol

                    # states
                    session_checker["range_state"] = state_range
                    session_checker["init_state"] = init.get("session_checker_init_state")
                    session_checker["init_message_arrived"] = "false"

                    # references
                    session_checker["ref_local_session_sender"] = f"{sender}_{session_id}"
                    session_checker["ref_session_path"] = f"{i}_{session_id}"
                    session_checker["ref_local_session_receiver"] = f"{node}_{session_id}"
                    session_checker["ref_node_receiver"] = f"{node}"

                    match session.protocol:
                        case protocol if protocol == "hpke" or protocol == "double_ratchet":
                            for j, local_path in enumerate(session.paths):
                                path_receivers = self._get_receivers_from_path(local_path)
                                if len(path_receivers) == 1 and path_receivers[0] == sender:
                                    session_checker["ref_session_path_to_sender"] = f"{j}_{session_id}"
                                    session_checker[f"cmd_{protocol}_reset"] = f"cmd_{protocol}_reset_session_path_{j}_{session_id}"
                                    session_checker[f"cmd_{protocol}_failure"] = f"cmd_{protocol}_failure_session_path_{j}_{session_id}"

                            session_checker["cmd_read_reset"] = f"cmd_read_reset_session_checker_of_node_{node}_of_path_{i}_{session_id}"
                            session_checker["cmd_read_ratchet"] = f"cmd_read_ratchet_session_checker_of_node_{node}_of_path_{i}_{session_id}"
                            session_checker["cmd_resolve_system"] = f"cmd_resolve_system_session_checker_of_node_{node}_of_path_{i}_{session_id}"


                        case protocol if protocol == "hpke_sender_key" or protocol == "double_ratchet_sender_key":
                            supersession = [s for s in sessions if session_id in [ss.id for ss in s.subsessions]][0]
                            for j, local_path in enumerate(session.paths):
                                path_receivers = self._get_receivers_from_path(local_path)
                                if len(path_receivers) == 1 and path_receivers[0] == sender:
                                    session_checker["ref_session_path_to_sender"] = f"{j}_{session_id}"
                                    session_checker[f"cmd_{protocol[:-11]}_reset"] = f"cmd_{protocol[:-11]}_reset_session_path_{j}_{session_id}"
                                    session_checker[f"cmd_{protocol[:-11]}_failure"] = f"cmd_{protocol[:-11]}_failure_session_path_{j}_{session_id}"
                                    session_checker["cmd_resolve_sender_refresh"] = f"cmd_resolve_sender_refresh_session_path_{j}_{session_id}"

                            session_checker["ref_session_path_broadcast"] = f"0_{supersession.id}"

                            session_checker["cmd_read_reset"] = f"cmd_read_reset_session_checker_of_node_{node}_of_path_{i}_{session_id}"
                            session_checker["cmd_read_ratchet"] = f"cmd_read_ratchet_session_checker_of_node_{node}_of_path_{i}_{session_id}"
                            session_checker["cmd_resolve_sender_new_key"] = f"cmd_resolve_sender_new_key_session_checker_of_node_{node}_of_path_0_{supersession.id}"
                            session_checker["cmd_resolve_sender_reset"] = f"cmd_resolve_sender_reset_session_checker_of_node_{node}_of_path_{i}_{session_id}"
                            session_checker["cmd_resolve_system"] = f"cmd_resolve_system_session_checker_of_node_{node}_of_path_{i}_{session_id}"
                            
                        case "sender_key":
                            subsession = [ss for ss in session.subsessions if node in ss.nodes][0]
                            path_id = [e[0] for e in enumerate(subsession.paths) if node == list(e[1])[0]][0]
                            session_checker["ref_session_path_to_sender"] = f"{path_id}_{subsession.get_id()}"

                            session_checker["cmd_resolve_sender_new_key"] = f"cmd_resolve_sender_new_key_session_checker_of_node_{node}_of_path_{i}_{session_id}"
                            session_checker["cmd_sender_key_failure"] = f"cmd_sender_key_failure_session_path_{path_id}_{subsession.get_id()}"
                            session_checker["cmd_sender_key_reset"] = f"cmd_sender_key_reset_session_path_{path_id}_{subsession.get_id()}"

                        case "mls":
                            session_checker["ref_session_path_broadcast"] = f"{i}_{session_id}"
                            for j, local_path in enumerate(session.paths):
                                path_receivers = self._get_receivers_from_path(local_path)
                                if len(path_receivers) == 1 and path_receivers[0] == sender:
                                    session_checker["ref_session_path_to_sender"] = f"{j}_{session_id}"

                            session_checker["cmd_read_reset"] = f"cmd_read_reset_session_checker_of_node_{node}_of_path_{i}_{session_id}"
                            session_checker["cmd_read_ratchet"] = f"cmd_read_ratchet_session_checker_of_node_{node}_of_path_{i}_{session_id}"
                            session_checker["cmd_read_refresh"] = f"cmd_read_refresh_session_checker_of_node_{node}_of_path_{i}_{session_id}"
                            session_checker["cmd_read_current"] = f"cmd_read_current_session_checker_of_node_{node}_of_path_{i}_{session_id}"
                            session_checker["cmd_mls_reset"] = f"cmd_mls_reset_session_checker_of_node_{node}_of_path_{i}_{session_id}"
                            session_checker["cmd_mls_failure"] = f"cmd_mls_failure_session_checker_of_node_{node}_of_path_{i}_{session_id}"
                            session_checker["cmd_resolve_system"] = f"cmd_resolve_system_session_checker_of_node_{node}_of_path_{i}_{session_id}"

                    # commands
                    # Freeze is synchronised with the session path's send command
                    session_checker["cmd_freeze"] = f"cmd_send_session_path_{i}_{session_id}"

                    # Trigger: the link delivering to this receiver completes
                    # (predecessors yields exactly one node in a tree path).
                    prec = list(path.predecessors(node))[0]
                    session_checker["cmd_trigger"] = f"cmd_receive_success_link_{prec}_{node}_of_path_{i}_{session_id}"
                    session_checker["cmd_read_data"] = f"cmd_read_data_session_checker_of_node_{node}_of_path_{i}_{session_id}"
                    session_checker["cmd_check_success"] = f"cmd_check_success_session_checker_of_node_{node}_of_path_{i}_{session_id}"
                    session_checker["cmd_resolve_data"] = f"cmd_resolve_data_session_checker_of_node_{node}_of_path_{i}_{session_id}"
                    session_checker["cmd_cleanup"] = f"cmd_cleanup_session_checker_of_node_{node}_of_path_{i}_{session_id}"
                    session_checker["cmd_check_failure"] = f"cmd_check_failure_session_checker_of_node_{node}_of_path_{i}_{session_id}"
                    session_checker["cmd_update"] = f"cmd_update_session_checker_of_node_{node}_of_path_{i}_{session_id}"

                    # Walk backwards from receiver to sender collecting the
                    # link chain for coordinated cleanup.
                    links = []
                    current_node = node
                    while current_node != sender:
                        predecessor = next(path.predecessors(current_node))
                        link_name = f"link_{predecessor}_{current_node}_of_path_{i}_{session_id}"
                        links.append({"cmd_cleanup": f"cmd_link_cleanup_{link_name}",
                                      "#": f"{predecessor}_{current_node}_{i}_{session_id}"})
                        current_node = predecessor

                    session_checker["links"] = links
                    
                    session_checkers_dict["instances"].append(session_checker)

        return session_checkers_dict      


    def _add_rewards_to_dict(self) -> dict:
        """Serialise reward PRISM modules from ``self.rewards``.

        Each :class:`Reward` becomes an instance with a list of contribution
        dicts ``{"command": ..., "condition": ..., "value": ...}``.

        Returns
        -------
        dict
            ``{"name": "rewards_modules", "template": ..., "instances": [...]}``
        """
        with open("config/netgen_files.json", "r") as f:
            files_config = json.load(f)
            
        rewards_dict = {}
        rewards_dict["name"] = "rewards_modules"
        rewards_dict["template"] = files_config.get("rewards_template")
        rewards_dict["instances"] = []
        for reward in self.rewards:
            instance = {}
            instance["name"] = f"{reward.name}.prism"
            instance["#"] = reward.name
            instance["contributions"] = []
            for contribution in reward.contributions:
                contrib_dict = {}
                contrib_dict["command"] = contribution.command
                contrib_dict["condition"] = contribution.condition
                contrib_dict["value"] = contribution.value
                instance["contributions"].append(contrib_dict)
            rewards_dict["instances"].append(instance)
        
        return rewards_dict


