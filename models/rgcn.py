import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

class RGCNModel(nn.Module):
    """
    Relational Graph Convolutional Network (R-GCN).
    
    The R-GCN acts as an 'Encoder'. It takes the raw graph structure and 
    learns high-level entity embeddings by looking at the neighbors of each node.
    It uses a Hybrid Device strategy: Message Passing (Sparse Math) on CPU, 
    and Scoring (Dense Math) on GPU/MPS.
    """
    def __init__(self, num_entities, num_relations, embedding_dim=200, data=None, device='cpu'):
        super().__init__()
        # Store the target device (e.g., 'mps' for Mac or 'cuda' for NVIDIA)
        self.device = device
        
        # Reference to the dataset object which contains edge_index and edge_types
        self.data = data
        
        # 1. BASE EMBEDDINGS
        # These are 'Lookup Tables'. Every entity and relation starts as a random vector.
        # These vectors will be updated and 'learned' during training.
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        
        # 2. TWO-LAYER GNN ARCHITECTURE
        # RGCNConv is specialized: it applies a different Weight Matrix (W) for 
        # every relation type. This allows the model to learn that 'is_capital_of' 
        # means something different than 'is_near'.
        self.conv1 = RGCNConv(embedding_dim, embedding_dim, num_relations)
        self.conv2 = RGCNConv(embedding_dim, embedding_dim, num_relations)
        
        # 3. REGULARIZATION (Batch Normalization)
        # Normalizes the 'activations' to mean 0 and variance 1.
        # This prevents gradients from exploding and speeds up training.
        self.bn = nn.BatchNorm1d(embedding_dim)
        
        # 4. WEIGHT INITIALIZATION
        # Xavier initialization sets weights to small random numbers based on layer size.
        # This ensures signals don't die out (vanishing gradients) in the first epoch.
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    def get_enriched_embeddings(self):
        """
        ENCODER PHASE: Transforms raw entity features into contextualized embeddings.
        
        This method implements a 'Hybrid Device Strategy'. It performs the memory-intensive 
        graph convolutions on the CPU to ensure system stability on Mac hardware, 
        then offloads the final vectors to the GPU (MPS) for high-speed scoring.
        """

        # --- STEP 1: DEVICE OFFLOADING (CPU) ---
        # Logic: Explicitly move the graph structure and weights to the CPU RAM.
        # Why it's important: Metal Performance Shaders (MPS) have strict buffer limits 
        # for sparse matrix operations. By using the CPU here, we bypass the 'MPS Hang' 
        # that occurs when a graph (like WN18RR + Semantic Edges) becomes too complex 
        # for the GPU's current shader kernels to allocate.
        edge_index = self.data.edge_index.cpu()
        edge_type = self.data.edge_type.cpu()
        x = self.entity_embeddings.weight.cpu()
        
        # --- STEP 2: LAYER 1 MESSAGE PASSING ---
        # Logic: Perform the first round of relational neighbor aggregation.
        # self.conv1.cpu(): Ensures the R-GCN layer itself is in CPU mode.
        # self.bn.cpu(): Applies Batch Normalization on the CPU to stabilize activations.
        # F.relu(): Adds non-linearity so the model can learn complex relationships.
        # Why it's important: This is the '1-hop' reasoning where an entity learns 
        # about its immediate neighbors and their specific relationship types.
        h = F.relu(self.bn.cpu()(self.conv1.cpu()(x, edge_index, edge_type)))
        
        # --- STEP 3: LAYER 2 & RESIDUAL CONNECTION ---
        # Logic: Perform a second round of aggregation and add a 'skip connection'.
        # h2: Represents '2-hop' reasoning (neighbors of neighbors).
        h2 = self.conv2.cpu()(h, edge_index, edge_type)
        
        # (h2 + h): This is the Residual Connection.
        # Why it's important: In Deep GNNs, entities can suffer from 'Over-smoothing' 
        # (where they all start looking identical). Adding the original 'h' back 
        # into 'h2' preserves unique features and stabilizes gradient flow.
        enriched = F.relu(h2 + h)
        
        # --- STEP 4: DEVICE RE-UPLOADING (MPS/GPU) ---
        # Logic: Move the final, high-level entity vectors back to the target device.
        # Why it's important: While the CPU is safer for the 'Sparse' graph math 
        # above, the 'Dense' matrix multiplication used in the scoring phase 
        # (forward_all) is much faster on the GPU. This return statement 
        # bridges the two devices for maximum efficiency.
        return enriched.to(self.device)

    def forward_all(self, h_idx, r_idx, cached_x=None):
        """
        DECODER STEP: Scores (Head, Relation) pairs against ALL possible entities.
        This is a '1-N' scoring strategy, which is much faster than scoring 1 triple at a time.
        """
        
        # 1. GET EMBEDDINGS
        # We use 'cached_x' if provided (standard for evaluation) to avoid 
        # re-running the expensive GNN logic multiple times per batch.
        x = cached_x if cached_x is not None else self.get_enriched_embeddings()
        
        # 2. EXTRACT BATCH FEATURES
        # h: Vector representation of the specific 'Heads' in this batch.
        # r: Vector representation of the 'Relations' in this batch.
        h = x[h_idx]
        r = self.relation_embeddings(r_idx)
        
        # 3. SCORING (DistMult-style Interaction)
        # (h * r) performs element-wise multiplication.
        # torch.mm(..., x.t()) performs a Matrix Multiplication against ALL entities.
        # Result: A score for every possible entity in the graph.
        return torch.mm(h * r, x.t())