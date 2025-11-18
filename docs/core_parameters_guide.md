# Parametri Core per Analisi Topologica

## 1. **NET_TYPE** - Tipo di Topologia di Rete

### **`random`** - Rete Casuale (Erdős–Rényi)
```
Caratteristiche:
- Ogni coppia di nodi ha probabilità p di essere connessa
- Distribuzione binomiale dei gradi
- Transizione di fase per la connettività
```
**Quando usarla:**
- Baseline per confronti
- Modelli senza struttura particolare
- Analisi probabilistiche pure

**Proprietà tipiche:**
- Diametro: O(log n)
- Clustering basso
- Distribuzione gradi: Poisson

---

### **`scale_free`** - Rete Scale-Free (Barabási–Albert)
```
Caratteristiche:
- "Ricchi diventano più ricchi" (preferential attachment)
- Distribuzione gradi: legge di potenza P(k) ∝ k^(-γ)
- Pochi hub altamente connessi + molti nodi periferici
```
**Quando usarla:**
- Internet, Web, social networks
- Reti biologiche (protein interaction)
- Sistemi con crescita preferenziale

**Proprietà tipiche:**
- Ultra-small diameter
- Alta robustezza a fallimenti casuali
- Vulnerabile ad attacchi mirati agli hub

---

### **`small_world`** - Rete Small-World (Watts-Strogatz)
```
Caratteristiche:
- Alta clusterizzazione locale + cammini globali corti
- Interpolazione tra regolare e casuale
- Parametro di rewiring β ∈ [0,1]
```
**Quando usarla:**
- Reti sociali reali
- Propagazione epidemie/informazioni
- Neural networks

**Proprietà tipiche:**
- Clustering alto (come reti regolari)
- Path length basso (come reti casuali)
- "Six degrees of separation"

---

### **`ring`** - Topologia ad Anello
```
Caratteristiche:
- Ogni nodo connesso esattamente a k vicini
- Struttura completamente regolare
- Simmetria perfetta
```
**Quando usarla:**
- Protocolli token-ring
- Algoritmi distribuiti con fairness
- Baseline per clustering massimo

---

### **`star`** - Topologia a Stella
```
Caratteristiche:
- Un nodo centrale connesso a tutti gli altri
- Tutti i path passano per il centro
- Single point of failure
```
**Quando usarla:**
- Architetture client-server
- Analisi criticità hub centrale
- Worst-case per robustezza

---

## 2. **NUM_NODES** - Numero di Nodi

### Impatto sulla Complessità
- **Piccole (5-20)**: Analisi esaustiva possibile
- **Medie (20-100)**: Buon compromesso analisi/realismo  
- **Grandi (100+)**: Require ottimizzazioni PRISM

### Scaling Laws
```
Proprietà che scalano con n:
- Archi massimi: O(n²)
- Diametro: O(log n) per random, O(n) per ring
- Clustering: Dipende dalla topologia
```

---

## 3. **CONNECTIVITY_PROBABILITY** - Probabilità di Connessione

### Per Reti Random
```
p = probabilità che esista edge (i,j)
Grado atteso di ogni nodo: (n-1) × p
Numero archi atteso: n(n-1)p/2
```

### Soglie Critiche
- **p < 1/n**: Tanti componenti isolati
- **p ≈ 1/n**: Transizione di fase (giant component)
- **p > log(n)/n**: Probabilmente connessa
- **p → 1**: Grafo completo

### Impatto su PRISM
```prism
// Probabilità di raggiungibilità
P=? [F (message_received = true)]

// Con p bassa: molti path lunghi o inesistenti
// Con p alta: molti path alternativi corti
```

---

## 4. **DEGREE_DISTRIBUTION** - Distribuzione dei Gradi

### **`uniform`** - Distribuzione Uniforme
```
Tutti i nodi hanno grado simile
P(k) = costante per k ∈ [k_min, k_max]
```
**Caratteristiche:**
- Rete molto regolare
- Nessun hub dominante
- Load balancing naturale

### **`normal`** - Distribuzione Normale
```
P(k) ∝ exp(-(k-μ)²/2σ²)
Concentrata attorno alla media
```
**Caratteristiche:**
- Maggioranza nodi con grado medio
- Pochi outliers
- Bilanciamento buono

### **`power_law`** - Legge di Potenza
```
P(k) ∝ k^(-γ)
Tipicamente γ ∈ [2, 3]
```
**Caratteristiche:**
- Pochi hub super-connessi
- "80-20 rule": 20% nodi hanno 80% connessioni
- Scale-free networks

### **`exponential`** - Distribuzione Esponenziale
```
P(k) ∝ exp(-λk)
Decadimento rapido per gradi alti
```
**Caratteristiche:**
- Predominanza nodi basso grado
- Rari nodi ad alto grado
- Random networks

---

## 5. **DIAMETER** - Diametro della Rete

### Definizione
```
diameter = max{d(u,v) : u,v ∈ V}
dove d(u,v) = lunghezza shortest path da u a v
```

### Impatto Pratico
- **Comunicazione**: Worst-case latency
- **Propagazione**: Tempo massimo diffusione
- **Robustezza**: Resilienza a disconnessioni

### Valori Tipici per Tipologia
```
Random:      O(log n)           es. 4-6 per n=100
Scale-free:  O(log log n)       es. 3-4 per n=100  
Small-world: O(log n)           es. 4-5 per n=100
Ring:        O(n)               es. 50 per n=100
Star:        2                  sempre 2
```

### Per PRISM
```prism
// Bound temporali
P=? [F<=diameter (all_nodes_informed = true)]

// Worst-case reachability
P=? [F (max_distance <= diameter)]
```

---

## 6. **AVERAGE_PATH_LENGTH** - Lunghezza Media Cammini

### Definizione
```
APL = (1/|V|(|V|-1)) × Σ{d(u,v) : u≠v}
Media di tutte le distanze tra coppie di nodi
```

### Significato Pratico
- **Performance**: Latenza media attesa
- **Efficienza**: Quanto è "compatta" la rete
- **Costi**: Overhead medio comunicazione

### Relazione con Diameter
```
APL ≤ diameter
Tipicamente: APL ≈ diameter/2
```

### Small-World Phenomenon
```
Rete è "small-world" se:
APL ≈ APL_random (bassa)
Clustering >> Clustering_random (alta)
```

---

## 7. **DENSITY** - Densità della Rete

### Formula
```
density = |E| / |E_max|
dove |E_max| = n(n-1)/2 per grafi non diretti
```

### Interpretazione
- **0.0**: Nessun arco (nodi isolati)
- **0.5**: Metà degli archi possibili
- **1.0**: Grafo completo

### Trade-offs
```
Alta Densità:
✓ Ridondanza alta
✓ Robustezza
✗ Costo alto
✗ Complessità

Bassa Densità:  
✓ Efficienza
✓ Costo basso
✗ Vulnerabilità
✗ Colli di bottiglia
```

### Soglie Pratiche
- **< 0.1**: Rete molto sparsa
- **0.1-0.3**: Rete tipica (Internet-like)
- **0.3-0.6**: Rete densa
- **> 0.6**: Quasi completa

---

## 8. **CENTRALITY** - Misure di Centralità

### **Degree Centrality** - Centralità di Grado
```
C_D(v) = deg(v) / (n-1)
Frazione di nodi direttamente connessi
```
**Significato:**
- Influenza locale diretta
- Capacità di comunicazione immediata
- Load potenziale

### **Betweenness Centrality** - Centralità di Intermediazione
```
C_B(v) = Σ{σ_st(v)/σ_st : s≠v≠t}
Frazione di shortest paths che passano per v
```
**Significato:**
- Controllo flusso informazioni
- Punti di bottleneck
- Bridge tra comunità

### **Closeness Centrality** - Centralità di Vicinanza
```
C_C(v) = (n-1) / Σ{d(v,u) : u≠v}
Inverso della somma delle distanze
```
**Significato:**
- Velocità di raggiungere tutti
- Efficienza comunicazione
- Leadership naturale

### Hub vs. Bridge vs. Leader
```
Hub:    Alto degree, alta visibilità locale
Bridge: Alta betweenness, controllo traffico  
Leader: Alta closeness, coordinamento globale
```

## Esempi di Configurazioni Tipiche

### Rete Internet-like
```json
{
    "net_type": "scale_free",
    "num_nodes": 50,
    "degree_distribution": "power_law",
    "diameter": 6,
    "average_path_length": 3.5,
    "density": 0.08
}
```

### Rete Social Network
```json
{
    "net_type": "small_world", 
    "num_nodes": 100,
    "connectivity_probability": 0.1,
    "degree_distribution": "normal",
    "diameter": 5,
    "average_path_length": 3.2,
    "density": 0.15
}
```

### Rete Corporate
```json
{
    "net_type": "tree",
    "num_nodes": 30,
    "degree_distribution": "exponential", 
    "diameter": 8,
    "average_path_length": 4.5,
    "density": 0.07
}
```