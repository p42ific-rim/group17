import torch
import torch.nn as nn

class DistMultModel(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim=200):
        super().__init__()
        self.emb_e = nn.Embedding(num_entities, embedding_dim)
        self.emb_r = nn.Embedding(num_relations, embedding_dim)
        
        # Dettmers-style regularization
        self.bn0 = nn.BatchNorm1d(embedding_dim)
        self.bn1 = nn.BatchNorm1d(embedding_dim)
        self.inp_drop = nn.Dropout(0.2)
        self.feat_drop = nn.Dropout(0.3)

        nn.init.xavier_uniform_(self.emb_e.weight)
        nn.init.xavier_uniform_(self.emb_r.weight)

    def forward_all(self, h_idx, r_idx, cached_x=None):
        h = self.emb_e(h_idx)
        r = self.emb_r(r_idx)

        h = self.inp_drop(self.bn0(h))
        r = self.feat_drop(self.bn1(r))
        
        # Scoring: (h * r) @ E^T
        return torch.mm(h * r, self.emb_e.weight.t())

class TransEModel(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim=200, p_norm=1):
        super().__init__()
        self.emb_e = nn.Embedding(num_entities, embedding_dim)
        self.emb_r = nn.Embedding(num_relations, embedding_dim)
        self.p_norm = p_norm
        
        self.bn0 = nn.BatchNorm1d(embedding_dim)
        self.bn1 = nn.BatchNorm1d(embedding_dim)
        self.inp_drop = nn.Dropout(0.1)

        nn.init.xavier_uniform_(self.emb_e.weight)
        nn.init.xavier_uniform_(self.emb_r.weight)

    def forward_all(self, h_idx, r_idx, cached_x=None):
        h = self.inp_drop(self.bn0(self.emb_e(h_idx)))
        r = self.bn1(self.emb_r(r_idx))
        
        # (h + r)
        projected_h_r = h + r 
        
        # BCE expects high scores for true triples. 
        # Since TransE is distance-based (lower is better), we return negative distance.
        return -torch.cdist(projected_h_r, self.emb_e.weight, p=self.p_norm)