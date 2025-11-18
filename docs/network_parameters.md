# Documentazione Parametri Network Generator

## Panoramica
Questo documento descrive tutti i parametri disponibili per configurare il generatore di reti del progetto PEPE.

## Struttura dei File di Configurazione

- `netgen_complete.json`: Configurazione completa con tutti i parametri possibili
- `netgen_simple.json`: Configurazione semplificata per uso rapido
- `netgen.json`: Configurazione corrente di lavoro

## Parametri Dettagliati

### 1. PARAMETRI TOPOLOGICI DI BASE

#### `net_type` - Tipo di Topologia
Determina la struttura base della rete:

- **`random`**: Connessioni casuali uniformi
  - Buono per: Modelli generali, baseline comparison
  - PRISM: Analisi probabilistiche uniformi

- **`scale_free`**: Distribuzione power-law dei gradi
  - Buono per: Internet, social networks, reti biologiche  
  - PRISM: Analisi di robustezza con hub centrali

- **`small_world`**: Alta clusterizzazione + cammini corti
  - Buono per: Reti sociali, propagazione epidemie/informazioni
  - PRISM: Modelli di diffusione rapida

- **`ring`**: Topologia ad anello
  - Buono per: Token ring, protocolli rotativi
  - PRISM: Analisi di fairness e ordine

- **`star`**: Topologia a stella con hub centrale
  - Buono per: Client-server, single point of failure
  - PRISM: Analisi di disponibilità centralizzata

- **`mesh`**: Connessione completa tra tutti i nodi
  - Buono per: Massima ridondanza, reti critiche
  - PRISM: Baseline per robustezza massima

#### `connectivity_probability` (0.0-1.0)
Per reti random/small-world, probabilità di connessione tra nodi:
- 0.1-0.3: Rete sparsa, possibili isole
- 0.4-0.6: Connettività bilanciata
- 0.7-0.9: Rete densa, alta ridondanza

### 2. METRICHE STRUTTURALI DEL GRAFO

#### `diameter`
Massima distanza tra due nodi qualsiasi:
- **Impatto su PRISM**: 
  - Proprietà temporali: `F<=diameter φ`
  - Worst-case latency bounds
  - Analisi di raggiungibilità

#### `average_path_length`
Distanza media tra tutte le coppie di nodi:
- **Impatto su PRISM**:
  - Expected delivery time
  - Load distribution analysis
  - Communication efficiency

#### `density = edges / max_possible_edges`
- 0.1: Rete molto sparsa
- 0.5: Densità bilanciata  
- 0.9: Rete quasi completa
- **Trade-off**: Ridondanza vs. complessità/costo

#### `assortativity` (-1.0 to +1.0)
Correlazione tra gradi di nodi connessi:
- +1: Hub si connettono tra loro (ricchi-con-ricchi)
- 0: Connessioni casuali
- -1: Hub si connettono a nodi periferici
- **Impatto**: Resilienza ad attacchi mirati

### 3. PARAMETRI DI CENTRALITÀ

#### `degree_centrality`
Numero di connessioni dirette:
- **Hub detection**: Nodi con grado >> media
- **Load balancing**: Distribuzione traffico
- **PRISM**: Probabilità di transito messaggi

#### `betweenness_centrality`
Frequenza su shortest paths:
- **Bridge nodes**: Criticità per comunicazione
- **Bottleneck analysis**: Punti di congestione
- **PRISM**: Colli di bottiglia probabilistici

#### `closeness_centrality`
Vicinanza media a tutti i nodi:
- **Leadership**: Nodi coordinatori naturali
- **Information spread**: Velocità di diffusione
- **PRISM**: Tempi di raggiungimento ottimali

### 4. PARAMETRI DI RESILIENZA

#### `failure_tolerance`
- **`node_failure_threshold`**: Max nodi falliti mantenendo connettività
- **`cascading_failure_probability`**: Propagazione fallimenti
- **PRISM Properties**:
  ```
  P=? [G (connected_components = 1)]
  P=? [F (failed_nodes > threshold)]
  ```

#### `attack_resistance`
- **`random_attack_resilience`**: Resistenza a fallimenti casuali
- **`targeted_attack_resilience`**: Resistenza ad attacchi mirati
- **PRISM**: Analisi security vs availability

### 5. PARAMETRI DI PERFORMANCE

#### `bandwidth_constraints`
- **`total_bandwidth`**: Capacità totale di rete
- **`bandwidth_distribution`**: Come distribuire la capacità
  - `uniform`: Stessa banda per tutti
  - `proportional_to_degree`: Più banda ai nodi centrali
- **PRISM**: Modelli di congestione e QoS

#### `latency_model`
- **`base_latency`**: Tempo minimo trasmissione
- **`distance_factor`**: Penalità per hop aggiuntivi
- **`congestion_factor`**: Rallentamento sotto carico
- **PRISM**: Proprietà real-time `P=? [F<=T delivered]`

### 6. PARAMETRI DI SICUREZZA

#### `compromise_model`
- **`initial_compromised_probability`**: Stato iniziale
- **`compromise_spread_rate`**: Velocità propagazione
- **`detection_probability`**: Efficacia monitoraggio
- **PRISM Properties**:
  ```
  P=? [G (compromised_nodes < threshold)]
  P=? [F (attack_detected = true)]
  ```

#### `trust_model`
- **`trust_levels`**: Categorie di fiducia
- **`trust_propagation_decay`**: Degrado fiducia transitiva
- **PRISM**: Modelli di reputation e trust

### 7. PARAMETRI HARDWARE

#### `node_characteristics`
- **`buffer_size_range`**: Capacità code messaggi
- **`interfaces_per_node_range`**: Punti di connessione
- **`reliability_range`**: MTBF hardware
- **PRISM**: Stati di saturazione e failure

#### `channel_characteristics`  
- **`capacity_range`**: Banda per canale
- **`error_rate_range`**: Probabilità errori trasmissione
- **PRISM**: Modelli di perdita e ritrasmissione

### 8. PARAMETRI DI PROTOCOLLO

#### `crypto_protocol`
- **`signal`**: End-to-end encryption moderno
- **`mls`**: Messaging Layer Security per gruppi
- **`tls`**: Transport Layer Security
- **PRISM**: Diversi modelli di security

#### `session_management`
- **`max_sessions_per_node`**: Limite concorrenza
- **`session_timeout`**: Scadenza sessioni
- **PRISM**: Gestione stati sessione

#### `key_management`
- **`key_rotation_frequency`**: Frequenza cambio chiavi
- **`forward_secrecy`**: Sicurezza retroattiva
- **PRISM**: Proprietà crittografiche temporali

### 9. PARAMETRI PRISM-SPECIFICI

#### `properties_to_verify`
Proprietà da verificare automaticamente:
- **`connectivity`**: Raggiungibilità e connessione
- **`security`**: Resistenza ad attacchi
- **`performance`**: Throughput e latenza
- **`reliability`**: Disponibilità e robustezza

#### `model_checking`
- **`state_space_exploration`**: Strategia di esplorazione
- **`probability_precision`**: Precisione calcoli
- **`counterexample_generation`**: Debug proprietà

## Utilizzo Raccomandato

### Per Reti Piccole (5-20 nodi)
Usa `netgen_simple.json` con:
- `net_type`: "small_world" o "random"
- Focus su parametri base: nodi, collegamenti, sicurezza

### Per Reti Medie (20-100 nodi)  
Usa subset di `netgen_complete.json`:
- Aggiungi parametri di performance
- Include metriche di resilienza
- Configura protocolli specifici

### Per Reti Grandi (100+ nodi)
Configurazione completa con:
- Ottimizzazioni PRISM (reduction techniques)
- Parametri di scalabilità
- Analisi multi-obiettivo

## Esempi di Proprietà PRISM

```prism
// Connectivity
P=? [G (connected = true)]

// Security  
P=? [F<=T (compromised_nodes > 0.2 * total_nodes)]

// Performance
R{"throughput"}=? [C<=T]

// Reliability
P=? [G (availability > 0.99)]
```

## File Correlati

- `/netgen/generate.py`: Implementazione generatore
- `/templates/*.pre`: Template moduli PRISM  
- `/netgen/consts.json`: Costanti sistema
- `/netgen/ranges.json`: Range valori ammissibili