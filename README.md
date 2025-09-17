# PEPE

PRISM Encryption Protocols Evaluation

This project aims to perform a performance analysis and evaluation of different encryption communication protocols, such as ***HPKE***, ***Double Ratchet***, ***Sender Key*** and ***MLS***.

This project is being developed for the Quantitative Evaluation of Stochastic Models course of Information Engineering master degree of the University of Studies of Florence.

## Modules

These are the following modules around which the project revolve.

### Node

It represents a device on the communication network. It can have one or more *interfaces* and zero or more *sessions*.

#### Variables

- node_state := *on* | *off*

### Interface

It represents a network interface of the *node*.

#### Variables

- interface_state := *on* | *off*
- interface_buffer := [1..N]

### Session

It respresents a session between two or more *nodes*, maintained locally by *local sessions*.

#### Varaibles

- leaked := true | false

### Local Session

It respresents the local state of a *session* from the perspective of a *node*.

#### Variables

- local_session_state := [0..N]
- session_leaked := ***#session_leaked***
- local_session_compromised := true | false

### Link

It represents a communication channel between two *interfaces*.

#### Variables

- link_state := *working* | *error* | *failure*

### Single-Session Path [ WIP ]

It represents a sending path from a single sender *node* to multiple receiver *nodes*, with all these nodes sharing a common *session*. At the moment the implementation is hardly modular, but the following variables are theoretically required.

#### Variables

- state
- sender node
- receiver nodes
- session
- links tree

### Multi-Session Path [ WIP ]

It represents a sending path from a single sender *node* to multiple receiver *nodes*, where this path crosses more then one common *session*. At the moment the implementation is hardly modular, but the following variables are theoretically required.

#### Variables

- state
- single-session paths & links tree
