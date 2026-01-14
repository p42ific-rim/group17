"""
dataset.py
-----------
This file defines a wrapper around the WordNet18RR dataset from PyTorch Geometric.
The goal is to expose the dataset in a *model-agnostic* way, using standard
(head, relation, tail) triples so that embedding-based and GNN-based models
can share the same training and evaluation logic. Preserves the full graph structure
for GNNs.
"""

import torch
from torch_geometric.datasets import WordNet18RR


class KnowledgeGraphDataset:
    def __init__(self, root: str = "data"):
        self.dataset = WordNet18RR(root)
        self.data = self.dataset[0]

        self.num_entities = self.data.num_nodes
        self.num_relations = int(self.data.edge_type.max()) + 1

        # Full graph structure (for GNN encoders)
        self.edge_index = self.data.edge_index
        self.edge_type = self.data.edge_type

        # Triple-based splits (for training / evaluation)
        self.train_triples = self._get_triples(self.data.train_mask)
        self.valid_triples = self._get_triples(self.data.val_mask)
        self.test_triples  = self._get_triples(self.data.test_mask)

    def _get_triples(self, mask):
        edge_index = self.data.edge_index[:, mask]
        edge_type = self.data.edge_type[mask]

        return torch.stack(
            [edge_index[0], edge_type, edge_index[1]],
            dim=1
        )
