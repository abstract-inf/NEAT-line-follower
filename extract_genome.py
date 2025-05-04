import pickle
import json

with open("neat_results/models/best_genome_2025-04-13_22-35-44.pkl", "rb") as f:
    genome = pickle.load(f)

def genome_to_dict(genome):
    return {
        "nodes": list(genome.nodes.keys()),
        "connections": [
            {
                "input": conn.key[0],
                "output": conn.key[1],
                "weight": conn.weight,
                "enabled": conn.enabled
            }
            for conn in genome.connections.values()
        ]
    }

with open("genome_export.json", "w") as out:
    json.dump(genome_to_dict(genome), out, indent=2)
