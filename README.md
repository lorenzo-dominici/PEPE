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
- node_buffer := [0..N]
- interfaces := [ ***#interface_state*** ]

### Interface

It represents a network interface of the *node*.

#### Variables

- interface_state := *working* | *error* | *failure*

### Session

It respresents a session between two or more *nodes*, maintained locally by *local sessions*.

#### Varaibles

- session_leaked := true | false

### Local Session

It respresents the local state of a *session* from the perspective of a *node*.

#### Variables

- local_session_state := [0..N]
- session_leaked := ***#session_leaked***
- local_session_compromised := true | false

### Channel

It represents a physical communication channel between two *interfaces*.

#### Variables

- channel_state := *working* | *error* | *failure*
- channel_bandwidth := [0..N]

### Link Ref

It represents a reference counter to assure a single sender buffer decrease in a multiple outward links scenario.

#### Variables

- link_ref_counter := [0..N]
- node_buffer := ***#node_buffer***

### Link

It represents a logical communication channel between two *interfaces*.

#### Variables

- link_state := *working* | *error* | *failure*
- link_prev := true | false
- link_sending := true | false
- link_receiving := true | false
- channel_state := ***#channel_state***
- channel_bandwidth := ***#channel_bandwidth***
- interface_state_sender := ***#interface_state***
- interface_state_receiver := ***#interface_state***
- node_buffer_sender := ***#node_buffer***
- node_buffer_receiver := ***#node_buffer***
- link_ref_counter := ***#link_ref_counter***
- session_path_link_counter := ***#session_path_link_counter***
- next_links := [ ***#link_prev*** ]

### Session Checker

It represents the session checking operation done when a message is received.

#### Variables

- session_checker_state := *idle* | *check*
- session_checker_sender_local_session_state := [0..N]
- node_buffer_receiver := ***#node_buffer***
- local_session_state_sender := ***#local_session_state***
- local_session_state_receiver := ***#local_session_state***

### Session Path [ WIP ]

It represents a sending path from a single sender *node* to multiple receiver *nodes*, where sender and receivers share a common *session*.

#### Variables

- session_path_state := *idle* | *run* | *wait*
- session_path_link_counter := [0..N]
- session_path_checker_counter := [0..N]
- node_state_sender := ***#node_state***
- node_buffer_sender := ***#node_buffer***
- first_links := [ ***#link_prev*** ]

### Multi-Session Path [ WIP ]

It represents a sending path from a single sender *node* to multiple receiver *nodes*, where this path crosses more then one common *session*. At the moment the implementation is hardly modular, but the following variables are theoretically required.

#### Variables

- state
- session paths & links tree
