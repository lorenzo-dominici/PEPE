"""
load.py — Configuration loader and validator for the PEPE network generator.

Reads the user-supplied JSON configuration file (``config/netgen.json``),
validates every parameter against domain-specific constraints, and returns a
clean ``params`` dictionary ready for consumption by
:class:`generate.NetworkGenerator`.

Validation strategy
-------------------
* Required parameters raise ``ValueError`` when absent or ``None``.
* Optional parameters fall back to sensible defaults (e.g. ``seed=42``).
* Numeric ranges are checked for order (``min <= max``) and type.
* Probability values are clamped to ``[0.0, 1.0]`` and cross-checked
  where independent probabilities must sum to less than 1.
* All string-valued parameters (protocol names, gen_model, abstraction,
  degree_distr.type) are **lowered** on input to guarantee case-insensitive
  matching against the canonical constant sets defined below.

Module-level constants
----------------------
``PROTOCOLS``
    The four end-to-end cryptographic protocols supported by the generator:
    *hpke*, *double_ratchet*, *sender_key*, *mls*.
``SUPPORT_PROTOCOLS``
    Sub-protocols used internally by *sender_key* sessions:
    *hpke_sender_key*, *double_ratchet_sender_key*.
``GEN_MODELS``
    Network topology generation models: *random* (Erdős–Rényi / Havel-Hakimi),
    *smart-world* (Watts-Strogatz), *scale-free* (Barabási-Albert extended).
``DEGREE_DISTR_TYPES``
    Distribution families for custom degree sequences: *binomial*, *uniform*,
    *normal*, *powerlaw*, *custom*.
``ABSTRACTION_LEVELS``
    PRISM model detail levels: *high*, *medium*, *low*.
"""

import json
import numpy as np

# ── Canonical constant sets (all lowercase) ───────────────────────────────────
PROTOCOLS = {"hpke", "double_ratchet", "sender_key", "mls"}
SUPPORT_PROTOCOLS = {"hpke_sender_key", "double_ratchet_sender_key"}
GEN_MODELS = {"random", "smart-world", "scale-free"}
DEGREE_DISTR_TYPES = {"binomial", "uniform", "normal", "powerlaw", "custom"}
ABSTRACTION_LEVELS = {"high", "medium", "low"}


def load_json(file_path: str) -> dict:
    """Load a JSON configuration file, validate it, and return the params dict.

    Parameters
    ----------
    file_path : str
        Path to the JSON config (typically ``config/netgen.json``).

    Returns
    -------
    dict
        Validated and normalised parameter dictionary.

    Raises
    ------
    ValueError
        On any invalid or missing parameter.
    FileNotFoundError
        If *file_path* does not exist.
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    params = validate_data(data)
    print("Network parameters successfully validated.")
    return params


# ── Validation helpers ─────────────────────────────────────────────────────────

def _require(data: dict, key: str, msg: str | None = None):
    """Return ``data[key]`` or raise ``ValueError`` if the key is missing/None.

    Parameters
    ----------
    data : dict
        The raw config dictionary.
    key : str
        The parameter name to look up.
    msg : str, optional
        Custom error message; defaults to ``"Missing required parameter: <key>"``.
    """
    val = data.get(key)
    if val is None:
        raise ValueError(msg or f"Missing required parameter: {key}")
    return val


def _validate_int_range(value, name: str, *, min_val: int = 1) -> None:
    """Validate a ``[min, max]`` pair of integers.

    Constraints:
    * Must be a list of exactly two ``int`` values.
    * ``min >= min_val`` (defaults to 1).
    * ``min <= max``.

    Raises ``ValueError`` with a descriptive message on failure.
    """
    if (not isinstance(value, list) or len(value) != 2 or
        not all(isinstance(n, int) for n in value) or
        value[0] < min_val or value[1] < value[0]):
        raise ValueError(f"{name} must be a list of two integers [min, max] with min >= {min_val} and min <= max")


def _validate_prob_range(value, name: str) -> None:
    """Validate a ``[min, max]`` probability range (both in ``[0.0, 1.0]``).

    Raises ``ValueError`` if the values are out of bounds or in wrong order.
    """
    if (not isinstance(value, list) or len(value) != 2 or
        not all(isinstance(p, (int, float)) for p in value) or
        not all(0.0 <= p <= 1.0 for p in value) or
        value[0] > value[1]):
        raise ValueError(f"{name} must be a list of two floats [min, max] with 0.0 <= min <= max <= 1.0")


def _validate_prob(value, name: str) -> None:
    """Validate a single probability (``float`` in ``[0.0, 1.0]``)."""
    if not (isinstance(value, (int, float)) and 0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be a float between 0.0 and 1.0")


def _require_prob_range(data: dict, key: str):
    """Shorthand: require the parameter *and* validate it as a probability range.

    Returns the validated ``[min, max]`` list.
    """
    val = _require(data, key)
    _validate_prob_range(val, key)
    return val


# ── Main validation ────────────────────────────────────────────────────────────

def validate_data(data: dict) -> dict:
    """Validate every field in *data* and build the canonical ``params`` dict.

    The function walks through the config in a fixed order that mirrors the
    logical grouping of parameters (protocol → topology → component attributes).
    Each group is documented with a section comment below.

    Parameters
    ----------
    data : dict
        Raw dictionary parsed from the JSON configuration file.

    Returns
    -------
    dict
        A new dictionary containing only validated, normalised values.  Keys
        match the names used throughout :mod:`generate`.

    Raises
    ------
    ValueError
        If any required parameter is missing or any value violates its domain.
    """
    params = {}

    # ── Protocol selection ─────────────────────────────────────────────────
    # The protocol dictates the session structure: one-to-one (hpke,
    # double_ratchet), one-to-many with sub-sessions (sender_key),
    # or group multicast (mls).
    protocol = _require(data, "protocol", "protocol parameter must be set")
    if isinstance(protocol, str):
        protocol = protocol.lower()
    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol must be one of {PROTOCOLS}")
    params["protocol"] = protocol

    # MLS requires an explicit range for the number of sessions (groups).
    if protocol == "mls":
        mls_range = _require(data, "mls_sessions_range")
        _validate_int_range(mls_range, "mls_sessions_range")
        params["mls_sessions_range"] = mls_range
    else:
        params["mls_sessions_range"] = data.get("mls_sessions_range")

    # Support protocol is used by sender_key to create one-to-one sub-sessions
    # between the sender and each receiver (via hpke or double_ratchet).
    support_protocol = data.get("support_protocol", "hpke_sender_key")
    if isinstance(support_protocol, str):
        support_protocol = support_protocol.lower()
    if support_protocol not in SUPPORT_PROTOCOLS:
        raise ValueError(f"support_protocol must be one of {SUPPORT_PROTOCOLS}")
    params["support_protocol"] = support_protocol

    # ── Seed & filename ────────────────────────────────────────────────────
    # The seed controls all stochastic decisions (topology, paths, probabilities)
    # for full reproducibility.
    seed = data.get("seed", 42)
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    params["seed"] = seed

    params["filename"] = data.get("filename")

    # ── Node count ─────────────────────────────────────────────────────────
    # ``node_range`` is a [min, max] pair.  A concrete ``node_number`` is
    # sampled uniformly at load time using a NumPy RNG seeded with the same
    # seed, so the count is deterministic for a given config.
    node_range = _require(data, "node_range")
    _validate_int_range(node_range, "node_range")
    rng = np.random.default_rng(seed)
    params["node_number"] = int(rng.integers(node_range[0], node_range[1] + 1))

    # ── Topology flags ─────────────────────────────────────────────────────
    # ``connected`` forces the resulting graph to be connected; the generator
    # retries up to 1 000 times or raises NetworkXUnfeasible.
    connected = data.get("connected", False)
    if not isinstance(connected, bool):
        raise ValueError("connected must be a bool")
    params["connected"] = connected

    # ``gen_model`` selects the graph family.  Must match one of GEN_MODELS.
    gen_model = data.get("gen_model", "random")
    if isinstance(gen_model, str):
        gen_model = gen_model.lower()
    if gen_model not in GEN_MODELS:
        raise ValueError(f"gen_model must be one of {GEN_MODELS}")
    params["gen_model"] = gen_model

    # ── Random graph parameters ────────────────────────────────────────────
    # Exactly ONE of conn_prob, degree_distr, or if_range should be set.
    # Priority: conn_prob > degree_distr > if_range (checked in generate.py).
    conn_prob = data.get("conn_prob")
    if conn_prob is not None:
        _validate_prob(conn_prob, "conn_prob")
    params["conn_prob"] = conn_prob

    degree_distr = data.get("degree_distr")
    if degree_distr is not None:
        degree_distr = _validate_degree_distr(degree_distr, node_range)
    params["degree_distr"] = degree_distr

    # Interface range: hard bounds on each node's degree (applied post-hoc).
    if_range = data.get("if_range")
    if if_range is not None:
        _validate_int_range(if_range, "if_range")
    params["if_range"] = if_range

    # ── Small-world (Watts-Strogatz) parameters ───────────────────────────
    mean_degree_range = data.get("mean_degree_range")
    if mean_degree_range is not None:
        _validate_int_range(mean_degree_range, "mean_degree_range")
    params["mean_degree_range"] = mean_degree_range

    # Rewiring probability: also reused by scale-free as ``q``.
    rewiring_prob = data.get("rewiring_prob")
    if rewiring_prob is not None:
        _validate_prob(rewiring_prob, "rewiring_prob")
    params["rewiring_prob"] = rewiring_prob

    # When True the original Watts-Strogatz variant (edge deletion) is used;
    # otherwise the Newman-Watts-Strogatz variant (edge addition) is chosen.
    delete_rewired = data.get("delete_rewired")
    if delete_rewired is not None and not isinstance(delete_rewired, bool):
        raise ValueError("delete_rewired must be a bool")
    params["delete_rewired"] = delete_rewired

    # ── Scale-free (Barabási-Albert extended) parameters ──────────────────
    initial_degree_range = data.get("initial_degree_range")
    if initial_degree_range is not None:
        _validate_int_range(initial_degree_range, "initial_degree_range")
        if initial_degree_range[0] > node_range[1]:
            raise ValueError("min initial_degree_range must be lower than max number of nodes")
    params["initial_degree_range"] = initial_degree_range

    # Probability of adding a new edge (``p`` in extended BA); the sum
    # p + q must be < 1.
    new_edges_prob = data.get("new_edges_prob")
    if new_edges_prob is not None:
        _validate_prob(new_edges_prob, "new_edges_prob")
        if rewiring_prob is not None and rewiring_prob + new_edges_prob >= 1.0:
            raise ValueError("the sum of rewiring_prob and new_edges_prob must be lower than 1")
    params["new_edges_prob"] = new_edges_prob

    # ── Clustering parameters ─────────────────────────────────────────────
    # When set, the topology is built as multiple sub-graphs (clusters)
    # connected via inter-cluster random edges.
    clusters_number_range = data.get("clusters_number_range")
    if clusters_number_range is not None:
        _validate_int_range(clusters_number_range, "clusters_number_range")
    params["clusters_number_range"] = clusters_number_range

    nodes_range_per_cluster = data.get("nodes_range_per_cluster")
    if nodes_range_per_cluster is not None:
        _validate_int_range(nodes_range_per_cluster, "nodes_range_per_cluster")
    params["nodes_range_per_cluster"] = nodes_range_per_cluster

    # Probability that any two nodes in different clusters share an edge.
    inter_clusters_coeff = data.get("inter_clusters_coeff")
    if inter_clusters_coeff is not None:
        _validate_prob(inter_clusters_coeff, "inter_clusters_coeff")
    params["inter_clusters_coeff"] = inter_clusters_coeff

    # ── Node attributes ───────────────────────────────────────────────────
    # Buffer size models the memory capacity of a node in the PRISM model.
    buffer_size_range = _require(data, "buffer_size_range")
    _validate_int_range(buffer_size_range, "buffer_size_range")
    params["buffer_size_range"] = buffer_size_range

    # Transition probabilities for the node state machine (off ↔ on).
    params["node_prob_off_to_on"] = _require_prob_range(data, "node_prob_off_to_on")
    params["node_prob_on_to_off"] = _require_prob_range(data, "node_prob_on_to_off")

    # ── Channel attributes ────────────────────────────────────────────────
    # Channels model the physical/logical link capacity between two nodes.
    channel_bandwidth_range = _require(data, "channel_bandwidth_range")
    _validate_int_range(channel_bandwidth_range, "channel_bandwidth_range")
    params["channel_bandwidth_range"] = channel_bandwidth_range

    # Path selection percentage: probability that a candidate pair of nodes
    # actually establishes a communication path.
    path_perc = data.get("path_perc")
    if path_perc is not None:
        _validate_prob(path_perc, "path_perc")
    params["path_perc"] = path_perc

    # Transition probabilities for the channel state machine
    # (working ↔ error, failure → working).
    params["channel_prob_working_to_error"] = _require_prob_range(data, "channel_prob_working_to_error")
    params["channel_prob_error_to_working"] = _require_prob_range(data, "channel_prob_error_to_working")
    params["channel_prob_failure_to_working"] = _require_prob_range(data, "channel_prob_failure_to_working")

    # ── Interface attributes ──────────────────────────────────────────────
    # Interfaces model the network adapters on each end of an edge.
    # Three independent transitions occur when the interface is OFF;
    # their probability-range minima and maxima must each sum to < 1.
    if_prob_off_to_working = _require_prob_range(data, "if_prob_off_to_working")
    if_prob_off_to_error = _require_prob_range(data, "if_prob_off_to_error")
    if_prob_off_to_failure = _require_prob_range(data, "if_prob_off_to_failure")
    params["if_prob_off_to_working"] = if_prob_off_to_working
    params["if_prob_off_to_error"] = if_prob_off_to_error
    params["if_prob_off_to_failure"] = if_prob_off_to_failure

    # Cross-check: competing transitions from the OFF state must leave room
    # for the implicit "stay off" probability.
    min_sum = if_prob_off_to_working[0] + if_prob_off_to_error[0] + if_prob_off_to_failure[0]
    max_sum = if_prob_off_to_working[1] + if_prob_off_to_error[1] + if_prob_off_to_failure[1]
    if min_sum >= 1.0:
        raise ValueError("the sum of minimum if_prob_off_to_* values must be lower than 1.0")
    if max_sum >= 1.0:
        raise ValueError("the sum of maximum if_prob_off_to_* values must be lower than 1.0")

    # Recovery / degradation transitions for the interface.
    params["if_prob_working_to_error"] = _require_prob_range(data, "if_prob_working_to_error")
    params["if_prob_error_to_working"] = _require_prob_range(data, "if_prob_error_to_working")
    params["if_prob_failure_to_working"] = _require_prob_range(data, "if_prob_failure_to_working")

    # ── Link attributes ───────────────────────────────────────────────────
    # Links are the per-path, per-edge transmission units.  Each link has its
    # own state machine; ``prob_retry`` and ``prob_sending`` control the
    # stochastic retry / transmission success behaviour in PRISM.
    params["link_prob_working_to_error"] = _require_prob_range(data, "link_prob_working_to_error")
    params["link_prob_error_to_working"] = _require_prob_range(data, "link_prob_error_to_working")
    params["link_prob_failure_to_working"] = _require_prob_range(data, "link_prob_failure_to_working")
    params["link_prob_retry"] = _require_prob_range(data, "link_prob_retry")
    params["link_prob_sending"] = _require_prob_range(data, "link_prob_sending")

    # ── Local session attributes ──────────────────────────────────────────
    # Each node participating in a session holds a local session with epoch
    # and ratchet counters that drive the key-rotation logic of the protocol.
    ls_size_epoch_range = _require(data, "ls_size_epoch_range")
    _validate_int_range(ls_size_epoch_range, "ls_size_epoch_range")
    params["ls_size_epoch_range"] = ls_size_epoch_range

    ls_size_ratchet_range = _require(data, "ls_size_ratchet_range")
    _validate_int_range(ls_size_ratchet_range, "ls_size_ratchet_range")
    params["ls_size_ratchet_range"] = ls_size_ratchet_range

    # The four local-session transition probabilities must leave room for a
    # non-negative "compromised" probability (i.e. sum < 1).
    ls_prob_session_reset = _require_prob_range(data, "ls_prob_session_reset")
    ls_prob_ratchet_reset = _require_prob_range(data, "ls_prob_ratchet_reset")
    ls_prob_none = _require_prob_range(data, "ls_prob_none")
    params["ls_prob_session_reset"] = ls_prob_session_reset
    params["ls_prob_ratchet_reset"] = ls_prob_ratchet_reset
    params["ls_prob_none"] = ls_prob_none

    if ls_prob_none[0] + ls_prob_session_reset[0] + ls_prob_ratchet_reset[0] >= 1.0:
        raise ValueError("the sum of minimum ls_prob_none, ls_prob_session_reset and ls_prob_ratchet_reset must be lower than 1.0")

    params["ls_prob_compromised"] = _require_prob_range(data, "ls_prob_compromised")

    # ── Session path attributes ───────────────────────────────────────────
    # ``sp_prob_run`` governs how often a session path is scheduled to run
    # in the PRISM scheduler.  For support sub-protocols it is forced to 0.
    params["sp_prob_run"] = _require_prob_range(data, "sp_prob_run")

    # ── Abstraction level ─────────────────────────────────────────────────
    # Controls PRISM model granularity; passed as a constant in the output JSON.
    abstraction = data.get("abstraction", "high")
    if isinstance(abstraction, str):
        abstraction = abstraction.lower()
    if abstraction not in ABSTRACTION_LEVELS:
        raise ValueError(f"abstraction must be one of {ABSTRACTION_LEVELS}")
    params["abstraction"] = abstraction

    return params


# ── Degree distribution validation ────────────────────────────────────────────

def _validate_degree_distr(degree_distr: dict, node_range: list) -> dict:
    """Validate and normalise the ``degree_distr`` object from the config.

    The function checks the ``type`` and ``params`` fields against the rules
    specific to each distribution family, and returns a **new** dict with
    the ``type`` lowered to guarantee case-insensitive matching downstream.

    Parameters
    ----------
    degree_distr : dict
        Raw config sub-object, e.g. ``{"type": "Binomial", "params": [5, 0.3]}``.
    node_range : list[int]
        The ``[min, max]`` node range, used to cap distribution parameters that
        must not exceed the maximum number of nodes.

    Returns
    -------
    dict
        ``{"type": <lowercase_type>, "params": <validated_params>}``

    Raises
    ------
    ValueError
        On any missing / invalid field.
    """
    distr_type = degree_distr.get("type")
    if not distr_type:
        raise ValueError("Missing parameter: degree_distr.type")
    if isinstance(distr_type, str):
        distr_type = distr_type.lower()
    if distr_type not in DEGREE_DISTR_TYPES:
        raise ValueError(f"degree_distr.type must be one of {DEGREE_DISTR_TYPES}")

    distr_params = degree_distr.get("params")
    if distr_params is None:
        raise ValueError("Missing parameter: degree_distr.params")

    # Custom distributions accept a raw Python list literal (as string) or list.
    if distr_type == "custom":
        if isinstance(distr_params, str):
            if not isinstance(eval(distr_params), list):
                raise ValueError("invalid params for the custom distribution")
        elif not isinstance(distr_params, list):
            raise ValueError("For custom, degree_distr.params must be a list or a string")
        return {"type": distr_type, "params": distr_params}

    # For built-in families, params is always a list of numbers.
    if not isinstance(distr_params, list):
        raise ValueError("degree_distr.params must be a list of parameters")

    match distr_type:
        case "binomial":
            # params = [n, p] where n = Bernoulli trials, p = success probability
            if len(distr_params) != 2 or not isinstance(distr_params[0], int) or not isinstance(distr_params[1], float):
                raise ValueError("For binomial, degree_distr.params must be [n: int, p: float]")
            if not (1 <= distr_params[0] <= node_range[1] - 1) or not (0.0 <= distr_params[1] <= 1.0):
                raise ValueError("For binomial, params must satisfy 1 <= n <= node_range[max]-1 and 0.0 <= p <= 1.0")
        case "uniform":
            # params = [min, max] — discrete uniform over degree values
            if len(distr_params) != 2 or not all(isinstance(p, int) for p in distr_params):
                raise ValueError("For uniform, degree_distr.params must be [min: int, max: int]")
            if distr_params[0] < 0 or distr_params[1] > node_range[1]:
                raise ValueError("For uniform, params must satisfy min >= 0 and max <= node_range[max]")
        case "normal":
            # params = [mean, stddev] — Gaussian (rounded to integers)
            if len(distr_params) != 2 or not all(isinstance(p, int) for p in distr_params):
                raise ValueError("For normal, degree_distr.params must be [mean: int, stddev: int]")
            if not (0 <= distr_params[0] <= node_range[1]) or distr_params[1] < 0:
                raise ValueError("For normal, params must satisfy 0 <= mean <= node_range[max] and stddev >= 0")
        case "powerlaw":
            # params = [exponent, k_min] — Zipf distribution via scipy.stats.zipf
            if len(distr_params) != 2 or not all(isinstance(p, (int, float)) for p in distr_params):
                raise ValueError("For powerlaw, degree_distr.params must be [exponent: float, min: float]")
            if distr_params[0] <= 1 or distr_params[1] > node_range[1]:
                raise ValueError("For powerlaw, params must satisfy exponent > 1 and min <= node_range[max]")

    return {"type": distr_type, "params": distr_params}
