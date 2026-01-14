import torch
import torch.nn as nn

class TransEModel(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim=200):
        super().__init__()
        self.emb_e = nn.Embedding(num_entities, embedding_dim)
        self.emb_r = nn.Embedding(num_relations, embedding_dim)
        
        self.bn0 = nn.BatchNorm1d(embedding_dim)
        self.bn1 = nn.BatchNorm1d(embedding_dim)
        self.inp_drop = nn.Dropout(0.1)

        # Scale and bias map distances to logits for BCE compatibility
        self.register_parameter('scale', nn.Parameter(torch.tensor([-1.0])))
        self.register_parameter('bias', nn.Parameter(torch.tensor([0.0])))

        nn.init.xavier_uniform_(self.emb_e.weight)
        nn.init.xavier_uniform_(self.emb_r.weight)

    def forward_all(self, h_idx, r_idx, cached_x=None):
        h = self.inp_drop(self.bn0(self.emb_e(h_idx)))
        r = self.bn1(self.emb_r(r_idx))
        
        target = h + r # [batch, dim]
        all_entities = self.emb_e.weight # [num_entities, dim]
        
        # --- THE SPEED FIX: Expanded L2 Distance ---
        # Instead of cdist, we use Matrix Multiplication
        target_sq = torch.sum(target**2, dim=1, keepdim=True) # [batch, 1]
        all_e_sq = torch.sum(all_entities**2, dim=1, keepdim=False) # [num_entities]
        dot_product = torch.mm(target, all_entities.t()) # [batch, num_entities]
        
        # dist = ||a||^2 + ||b||^2 - 2<a,b>
        # Use relu to ensure no precision errors result in negative distances
        dist = torch.relu(target_sq + all_e_sq - 2 * dot_product)
        
        return self.scale * torch.sqrt(dist + 1e-9) + self.bias