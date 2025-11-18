# Esempi Pratici di Configurazioni

## Scenario 1: Rete Aziendale Gerarchica
```json
{
    "net_type": "tree",
    "num_nodes": 25,
    "degree_distribution": "exponential",
    "target_metrics": {
        "diameter": 8,
        "average_path_length": 4.5, 
        "density": 0.08,
        "centrality": {
            "ceo_node": {"degree": 5, "betweenness": 0.6},
            "managers": {"degree": 3, "betweenness": 0.3}, 
            "employees": {"degree": 1, "betweenness": 0.0}
        }
    }
}
```
**Analisi PRISM:**
- Propagazione top-down efficiente
- Single point of failure (CEO)
- Latenza alta per comunicazione peer-to-peer

## Scenario 2: Social Network Realistico  
```json
{
    "net_type": "small_world",
    "num_nodes": 50,
    "connectivity_probability": 0.15,
    "degree_distribution": "power_law", 
    "target_metrics": {
        "diameter": 5,
        "average_path_length": 3.1,
        "density": 0.18,
        "centrality": {
            "influencers": {"count": 3, "degree": 15},
            "connectors": {"count": 5, "betweenness": 0.4},
            "regular_users": {"degree": 3}
        }
    }
}
```
**Analisi PRISM:**
- Propagazione virale rapida
- Robustezza a fallimenti casuali
- Vulnerabilità ad attacchi mirati

## Scenario 3: Rete P2P Decentralizzata
```json
{
    "net_type": "random", 
    "num_nodes": 40,
    "connectivity_probability": 0.2,
    "degree_distribution": "normal",
    "target_metrics": {
        "diameter": 4,
        "average_path_length": 2.8,
        "density": 0.2,
        "centrality": {
            "all_nodes": {"degree_variance": "low"},
            "no_super_hubs": true
        }
    }
}
```
**Analisi PRISM:**
- Load balancing naturale  
- Resilienza bilanciata
- Performance prevedibili