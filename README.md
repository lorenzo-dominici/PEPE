# PEPE — PRISM Encryption Protocols Evaluation

PEPE is a research-oriented repository that provides parametric PRISM model templates for the quantitative evaluation of encryption communication protocols. The models are written as `.pre` template files: they contain PRISM-compatible module code enriched with a small custom templating syntax (for example, `{{...}}` placeholders and `^...|...$` constructs) which is intended to be preprocessed into concrete PRISM models by a separate preprocessor (not yet implemented).

This project was developed as part of a course on Quantitative Evaluation of Stochastic Models and targets protocol families such as HPKE, Double Ratchet, Sender Key and MLS. The goals are to provide modular, parameterisable building blocks so researchers can compose network, channel and cryptographic protocol models, then explore reliability, robustness and session-security properties with PRISM.

## Table of contents

- [What this repository contains](#what-this-repository-contains)
  - [Files included](#files-included)
- [How the .pre templates work](#how-the-pre-templates-work)
- [Modules and main variables](#modules-and-main-variables)
  - [node](#node)
  - [interface](#interface)
  - [channel](#channel)
  - [link_ref](#link_ref)
  - [link](#link)
  - [local_session](#local_session)
  - [session_path](#session_path)
  - [session_checker](#session_checker)
- [Suggested workflow](#suggested-workflow)
  - [Manual instantiation](manual-instantiation)
  - [Scripted preprocessing](scripted-preprocessing)
  - [Compose a top-level PRISM model](compose-a-top-level-prism-model)
  - [Run PRISM](run-prism)
- [Project Status](#project-status)
  - [Current status](#current-status)
  - [Limitations](#limitations)
  - [Next steps](#next-steps)
- [Contributing](#contributing)
- [License & authorship](#license--authorship)
- [Contacts](#contacts)

## What this repository contains

The workspace contains a set of `.pre` files. Each `.pre` file is a template for a PRISM module that, once instantiated with concrete parameters, produces a PRISM module implementing part of the system.

### Files included

- `node.pre` — device/node model: node power state, local buffer and transitions that interact with attached interfaces and sessions.
- `interface.pre` — network interface model: interface state machine (working / error / failure) and probabilistic transitions.
- `channel.pre` — physical channel model: channel state and bandwidth variable.
- `link_ref.pre` — reference-counter helper: coordinates buffer decrements across multiple outgoing links.
- `link.pre` — logical link model: sending/receiving primitives, interactions with channel/interface/node buffer and retry logic.
- `local_session.pre` — local session model: per-node session state, ratchets/epochs, and compromise/recovery transitions.
- `session_path.pre` — session path controller (WIP): drives message creation and forwarding along a path of links for a given session.
- `session_checker.pre` — session checker (WIP): receiver-side validation & message resolution logic (handling HPKE/Double Ratchet/sender-key behaviors).
- `README.md` — this file (updated)

## How the `.pre` templates work

- Placeholders: The templates use double-curly placeholders like `{{name}}` for substituting identifiers, constants and numeric parameters.
- Control directives: The repository uses special markers (for example `^...|...$`) to express higher-level constructs such as loops, matches or conditional blocks that a preprocessor should expand into valid PRISM code.
- Output: After preprocessing, each `.pre` file should become a syntactically valid `.prism` file (or be directly includable by a PRISM model) with concrete names and numeric values.

The preprocessor is intentionally left out of this repository: it is part of the future work. The README below explains the suggested workflow to proceed without a preprocessor for quick experiments and how to approach building one.

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

## Suggested workflow

### Manual instantiation

Open a `.pre` file and manually replace `{{...}}` placeholders with concrete names and values to produce a `.prism` module. This works well for single-shot tests or small variations.

### Scripted preprocessing

Implement a small preprocessor (Python/JS) that reads `.pre` files and a parameter specification (YAML/JSON) and emits `.prism` modules. The preprocessor should:

- Replace `{{...}}` placeholders with values from the parameter file.
- Expand `^...|...$` blocks according to simple rules (loop/match expansion).
- Validate basic PRISM syntax (optional) before writing output.

### Compose a top-level PRISM model

Create a top-level `.prism` file that imports the generated modules, defines global constants, and declares any rewards or properties to check.

### Run PRISM

Use PRISM to run probabilistic model checking or simulation experiments against the generated `.prism` model, measure reliability, message delivery probabilities, compromise rates, latency proxies (counters/epochs), etc.

## Project Status

### Current status

The repository contains the template `.pre` modules and documentation. The templates are the canonical specification of the intended system, but they are not yet automatically usable because the preprocessor that turns `.pre` into concrete PRISM code has not been implemented.

### Limitations

- No preprocessor included — templates must be instantiated manually or via a user-provided script.
- Some modules are marked WIP (`session_path.pre`, `session_checker.pre`) and contain partial/templated logic intended to support multiple protocol variants.
- The repository does not currently include top-level property files, experimental scripts, or automated runs against PRISM.

### Next steps

1. Implement the preprocessor (Python is a good choice). Minimal features:

    - Token substitution for `{{...}}` placeholders.
    - Simple loop and match expansions for `^...|...$` blocks.
    - Parameter file support (JSON/YAML) and a small CLI.

2. Provide a set of example parameter files and a few generated `.prism` models that exercise each protocol (HPKE, Double Ratchet, Sender Key, MLS).

3. Add at least one top-level PRISM model and a small property suite (PCTL) for safety/availability/session-security metrics.

4. Optionally add CI to run basic syntax checks on generated models and a short PRISM smoke test if PRISM is available in the environment.

## Contributing

Contributions are welcome. Suggested low-effort ways to help:

- Add the preprocessor implementation and example parameter files.
- Create example instantiations (generated `.prism` files) for each protocol.
- Add PCTL properties and scripts to run PRISM experiments.

### License & authorship

This repository was created as course work for the Quantitative Evaluation of Stochastic Models class. If you want to reuse or extend the work, please include attribution.

### Contacts

If you want help turning templates into runnable PRISM models or designing the preprocessor, open an issue or reach out to the repository owner.
