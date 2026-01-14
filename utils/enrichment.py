import torch
from sentence_transformers import SentenceTransformer

def add_text_embeddings(dataset, model_name="all-MiniLM-L6-v2"):
    """
    Adds semantic embeddings for each entity using its WordNet definition.
    """
    # Detect the best available hardware (NVIDIA GPU, Apple Silicon, or CPU).
    # This ensures the heavy Transformer math runs as fast as possible.
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load the Sentence-Transformer model. MiniLM is chosen because it is 
    # fast and has a good balance between size and semantic accuracy.
    model = SentenceTransformer(model_name, device=device)
    
    # ATTRIBUTE LOOKUP: This logic is vital because PyTorch Geometric datasets 
    # often wrap their data in different ways. We check the top level and the 
    # internal .data object to find the list of text definitions.
    if hasattr(dataset, 'entity_glosses'):
        definitions = dataset.entity_glosses
    elif hasattr(dataset.data, 'entity_glosses'):
        definitions = dataset.data.entity_glosses
    else:
        raise AttributeError("Could not find 'entity_glosses' in dataset.")

    print(f"🧬 Encoding {len(definitions)} definitions on {device}...", flush=True)
    
    # THE CORE NLP STEP: model.encode takes the list of strings and returns 
    # a [num_entities, 384] tensor. convert_to_tensor=True keeps it in PyTorch
    # format so we don't have to convert from NumPy later.
    embeddings = model.encode(definitions, convert_to_tensor=True, show_progress_bar=True)
    
    # Store the results back into the dataset. .cpu() is used here to free 
    # up GPU/MPS memory for the upcoming training phase.
    dataset.text_embeddings = embeddings.to(torch.float).cpu()
    return dataset.text_embeddings

def add_semantic_edges(dataset, threshold=0.92, max_edges_per_node=5):
    """
    Vectorized and safe semantic edge addition.
    Caps edges per node to prevent memory explosion (no more 1.6 billion edges).
    """
    # Safety check: We can't compare definitions if we haven't encoded them yet.
    if not hasattr(dataset, "text_embeddings"):
        raise ValueError("Text embeddings not found. Run add_text_embeddings first.")

    # Move embeddings to the current working device.
    x = dataset.text_embeddings.to(dataset.data.edge_index.device)
    num_entities = x.size(0)
    
    # L2 NORMALIZATION: Crucial for Cosine Similarity.
    # By scaling every vector to length 1, the dot product (torch.mm) 
    # becomes identical to the Cosine Similarity score.
    x_norm = torch.nn.functional.normalize(x, p=2, dim=1)

    print(f"🔗 Calculating similarities (N={num_entities})...", flush=True)
    
    # VECTORIZED MATH: Instead of looping through entities (which is O(n^2)), 
    # we use one massive Matrix Multiplication. This is the difference between 
    # the script taking 5 seconds vs. 5 hours.
    sim_matrix = torch.mm(x_norm, x_norm.t())
    
    # Mask self-similarity (node i compared to itself) so we don't add 40k loops.
    sim_matrix.fill_diagonal_(0)

    print(f"🕵️ Filtering Top-{max_edges_per_node} neighbors above {threshold}...", flush=True)
    
    # TOP-K FILTERING: This is the "Safety Valve." 
    # Even if 1,000 nodes are similar, we only take the Top 5. This prevents 
    # your graph from becoming too "dense" and crashing your system memory.
    top_values, top_indices = torch.topk(sim_matrix, k=max_edges_per_node, dim=1)
    
    # THRESHOLD MASK: Only keep the Top-K if they are actually similar (>= 0.92).
    mask = top_values >= threshold
    
    # Use 'torch.where' logic to extract the specific indices (head and tail).
    rows = torch.arange(num_entities, device=x.device).view(-1, 1).expand_as(top_indices)
    h_idx = rows[mask]
    t_idx = top_indices[mask]

    num_new_edges = h_idx.size(0)
    print(f"✅ Found {num_new_edges} semantic edges.", flush=True)

    if num_new_edges > 0:
        # Prepare the new edges for the PyTorch Geometric graph.
        new_edges = torch.stack([h_idx, t_idx], dim=0)
        
        # RELATION ASSIGNMENT: We create a brand new relation ID that is 
        # higher than any existing relation ID in the original WN18RR.
        relation_id = int(dataset.data.edge_type.max().item() + 1)
        new_edge_types = torch.full((num_new_edges,), relation_id, dtype=torch.long, device=x.device)

        # CONCATENATION: The "Enrichment" moment.
        # We append the new semantic edges/types to the existing graph tensors.
        dataset.data.edge_index = torch.cat([dataset.data.edge_index, new_edges], dim=1)
        dataset.data.edge_type = torch.cat([dataset.data.edge_type, new_edge_types], dim=0)
        
    # Manual memory cleanup: The similarity matrix is huge (40k x 40k), 
    # so we delete it immediately after use to free up several GBs of RAM.
    del sim_matrix