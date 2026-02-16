# PEPE — PRISM Encryption Protocols Evaluation

PEPE is a research-oriented repository that provides parametric PRISM model templates for the quantitative evaluation of encryption communication protocols. The models are written as `.pre` template files: they contain PRISM-compatible module code enriched with a small custom templating syntax (for example, `{{...}}` placeholders and `{%...||...%}` constructs) which is preprocessed into concrete PRISM models by the included preprocessor.

This project was developed as part of a course on Quantitative Evaluation of Stochastic Models and targets protocol families such as HPKE, Double Ratchet, Sender Key and MLS. The goals are to provide modular, parameterisable building blocks so researchers can compose network, channel and cryptographic protocol models, then explore reliability, robustness and session-security properties with PRISM.

## Table of contents

- [What this repository contains](#what-this-repository-contains)
  - [Directory structure](#directory-structure)
- [How the .pre templates work](#how-the-pre-templates-work)
  - [Macro syntax](#macro-syntax)
  - [Placeholder syntax](#placeholder-syntax)
- [Modules and main variables](#modules-and-main-variables)
  - [node](#node)
  - [interface](#interface)
  - [channel](#channel)
  - [link_ref](#link_ref)
  - [link](#link)
  - [local_session](#local_session)
  - [session_path](#session_path)
  - [session_checker](#session_checker)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the preprocessor](#running-the-preprocessor)
  - [Configuration](#configuration)
  - [Data format](#data-format)
- [Project Status](#project-status)
  - [Current status](#current-status)
  - [Limitations](#limitations)
  - [Next steps](#next-steps)
- [Contributing](#contributing)
- [License & authorship](#license--authorship)
- [Contacts](#contacts)

## What this repository contains

The workspace contains `.pre` template files and a fully functional preprocessor that transforms them into PRISM modules.

### Directory structure

```text
PEPE/
├── config/                    # Configuration files
│   ├── netgen.json           # Network generator config (empty placeholder)
│   └── preprocessor.json     # Preprocessor config (empty placeholder)
├── examples/                  # Example configurations and data
│   ├── netgen.json           # Network generator example (empty)
│   ├── preprocessor_test_config.json  # Working preprocessor config
│   └── preprocessor_test_data.json    # Working test data
├── netgen/                    # Network generator (not yet implemented)
│   ├── generate.py           # Placeholder
│   ├── load.py               # Placeholder
│   ├── main.py               # Placeholder
│   └── store.py              # Placeholder
├── preprocessor/              # Template preprocessor (fully implemented)
│   ├── load.py               # JSON/template loading utilities
│   ├── logger.py             # Logging with color support & multiprocessing
│   ├── macro.py              # Macro resolution engine (match, loop)
│   ├── main.py               # CLI entry point and orchestration
│   ├── models.py             # Pydantic models for config/data validation
│   ├── process.py            # Template processing pipeline
│   ├── replace.py            # Placeholder substitution with JSON-path
│   ├── requirements.txt      # Python dependencies
│   └── store.py              # File output and joining utilities
├── templates/                 # PRISM module templates
│   ├── channel.pre           # Physical channel model
│   ├── interface.pre         # Network interface model
│   ├── link.pre              # Logical link model
│   ├── link_ref.pre          # Reference counter helper
│   ├── local_session.pre     # Per-node session state
│   ├── node.pre              # Device/node model
│   ├── policy.pre            # Policy definitions
│   ├── rewards.pre           # Reward structures
│   ├── session_checker.pre   # Receiver-side validation (WIP)
│   └── session_path.pre      # Session path controller (WIP)
└── README.md
```

## How the `.pre` templates work

Templates use a two-phase processing system implemented by the preprocessor:

### Macro syntax

Macros are control structures enclosed in `{%...%}` (configurable). Fields are separated by `||`.

**Match (conditional):**

```text
{%match||key||value1||output1||value2||output2||_||default_output%}
```

Evaluates the data key and outputs the corresponding branch. Use `_` for the default fallback.

**Loop (iteration):**

```text
{%loop||list_key||
    body template with {{list_key[].field}}
%}
```

Repeats the body for each item in the list. Use `list_key[]` to reference the current item.

**Nesting:** Macros can be nested; the preprocessor handles depth-aware parsing.

### Placeholder syntax

Placeholders use `{{...}}` (configurable) and support JSON-path-like access:

- Simple key: `{{name}}`
- Nested access: `{{user.profile.email}}`
- Array indexing: `{{items[0].price}}`, `{{items[-1]}}`
- Quoted keys: `{{data."key.with.dots"}}`

## Modules and main variables

Each template implements a PRISM module whose state is expressed with small sets of variables. Below are the most important module responsibilities and representative variables (not exhaustive):

### node

- node_state: on | off
- node_buffer: integer buffer for outgoing messages
- transitions that coordinate with attached interfaces and link_ref counters

### interface

- interface_state: working | error | failure
- probabilistic transitions modeling transient errors and repairs

### channel

- channel_state: working | error | failure
- channel_bandwidth: numeric available bandwidth

### link_ref

- link_ref_counter: reference counter to ensure a single buffer decrement across multiple outgoing links

### link

- link_state, link_prev, link_sending, link_receiving
- encoding of send/receive success, retry and bandwidth checks

### local_session

- local_session_state, local_session_epoch, local_session_ratchet, local_session_compromised
- models ratchet/epoch progression, resets, compromise and recovery transitions

### session_path

- session_path_state and counters used to orchestrate message production and multi-recipient delivery

### session_checker

- receiver-side validation logic that depends on the chosen protocol (patterns for HPKE, Double Ratchet, Sender Key)

## Installation

1. Ensure you have Python 3.10+ installed
2. Install dependencies:

```bash
pip install -r preprocessor/requirements.txt
```

Required packages:

- `pydantic` — Configuration and data validation
- `colorama` — Colored console output
- `tqdm` — Progress bar for parallel processing

## Usage

### Running the preprocessor

Run the preprocessor using the module syntax:

```bash
python -m preprocessor.main --config examples\preprocessor_test_config.json --data examples\preprocessor_test_data.json --log-file examples\preprocessor_test\_preprocessor.log
```

**Command-line arguments:**

| Argument | Required | Description |
| -------- | -------- | ----------- |
| `--config` | Yes | Path to preprocessor configuration JSON file |
| `--data` | Yes | Path to data JSON file or directory of data files |
| `--log-level` | No | Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO) |
| `--log-file` | No | Path for log file output (defaults to stderr) |

### Configuration

The preprocessor configuration file (JSON) defines:

```json
{
    "output_dir": "examples/preprocessor_test",
    "jobs": -1,
    "join_mode": "clean_join",
    "joined_file": "MODEL.prism",
    "separators": {
        "macro_open": "{%",
        "macro_separator": "||",
        "macro_close": "%}",
        "match_default": "_",
        "placeholder_open": "{{",
        "placeholder_close": "}}"
    }
}
```

| Field | Description |
| ----- | ----------- |
| `output_dir` | Directory for generated output files |
| `jobs` | Number of parallel workers (-1 or 0 = auto, based on CPU count) |
| `join_mode` | `none` (individual files), `join` (also create joined file), `clean_join` (join and delete individuals) |
| `joined_file` | Filename for the joined output |
| `separators` | Token definitions for macro and placeholder syntax |

### Data format

The data file (JSON) provides constants and template instances:

```json
{
    "consts": {
        "const_off": 0,
        "const_on": 1,
        "version": "1.0"
    },
    "items": [
        {
            "name": "node_modules",
            "template": "templates/node.pre",
            "instances": [
                {"name": "node_0.prism", "#": 0, "range_state": "0..1", ...},
                {"name": "node_1.prism", "#": 1, "range_state": "0..1", ...}
            ]
        }
    ]
}
```

- `consts`: Shared values available to all instances
- `items`: List of template groups, each with:
  - `name`: Descriptive name for the group
  - `template`: Path to the `.pre` template file
  - `instances`: List of data dictionaries (each must have a `name` key for output filename)

## Project Status

### Current status

**Preprocessor** — Fully implemented with the following features:

- Two-phase template processing: macro resolution followed by placeholder substitution
- Macro support: `match` (conditional branching) and `loop` (iteration)
- JSON-path-like placeholder syntax with dot notation and array indexing
- Pydantic-based configuration and data validation
- Parallel processing using multiprocessing with progress tracking
- Configurable output file joining (`none`, `join`, `clean_join` modes)
- Comprehensive logging with color support and multiprocessing-safe queue handlers
- CLI with flexible argument handling

**Templates** — Complete set of `.pre` template files for PRISM modules covering nodes, interfaces, channels, links, sessions, and protocol-specific logic.

**Network Generator (netgen)** — Not yet implemented. The `netgen/` directory contains placeholder files for a planned tool to generate network topologies and corresponding preprocessor data files.

### Limitations

- The `netgen` module is not implemented — data files must be created manually or via external tools.
- The repository does not include top-level property files or automated PRISM experiment scripts.
- Configuration files in `config/` are empty placeholders; use `examples/` for working configurations.

### Next steps

1. **Implement netgen**: Build the network topology generator to automatically produce data files from high-level network descriptions.

2. **Add PCTL properties**: Create a property suite for safety, availability, and session-security metrics.

3. **Experiment automation**: Add scripts to run PRISM experiments and collect results.

4. **CI integration**: Add automated syntax checks on generated models and smoke tests if PRISM is available.

## Contributing

Contributions are welcome. Suggested ways to help:

- **Implement netgen**: Build the network topology generator module.
- **Add properties**: Create PCTL property files for PRISM experiments.
- **Documentation**: Improve template documentation and add usage examples.
- **Testing**: Add unit tests for the preprocessor modules.

## License & authorship

This repository was created as course work for the Quantitative Evaluation of Stochastic Models class. If you want to reuse or extend the work, please include attribution.

## Contacts

If you want help turning templates into runnable PRISM models or extending the preprocessor, open an issue or reach out to the repository owner.
