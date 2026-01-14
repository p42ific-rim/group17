import os
import ssl
import torch
import torch.optim as optim
import time
from collections import defaultdict

# ==============================================================================
# 1. ENVIRONMENT & COMPATIBILITY SETUP
# ==============================================================================

# --- SSL BYPASS ---
# Logic: MacOS Python versions often lack pre-installed root certificates. 
# Importance: Prevents "CERTIFICATE_VERIFY_FAILED" when the script tries to 
# download Sentence-BERT models or NLTK datasets.
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

import nltk
from nltk.corpus import wordnet as wn

# Project imports for Models, Training, and Evaluation
from data.dataset import KnowledgeGraphDataset
from models.transe import TransEModel
from models.distmult import DistMultModel
from models.rgcn import RGCNModel
from training.earlystopping import EarlyStopping
from training.trainer_bce import BCETrainer
from training.evaluator import Evaluator
from utils.enrichment import add_text_embeddings, add_semantic_edges

# --- MAC/MPS MEMORY OPTIMIZATION ---
# Logic: Sets the PyTorch Metal Performance Shaders (MPS) memory threshold.
# Importance: '0.0' forces the GPU to release memory immediately after use, 
# preventing "Out of Memory" crashes on shared-memory Apple Silicon chips.
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

# ==============================================================================
# 2. DATA PREPROCESSING FUNCTIONS
# ==============================================================================

def get_nltk_glosses(dataset):
    """
    Logic: Uses NLTK WordNet to retrieve text definitions (glosses) for entities.
    Importance: WN18RR entities are numeric offsets. To do 'Semantic Enrichment', 
    we must turn those numbers back into human-readable text definitions.
    """
    print("🌐 Downloading NLTK WordNet data...", flush=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    
    num_entities = dataset.num_entities
    glosses = ["" for _ in range(num_entities)]
    
    print(f"🧠 Generating glosses for {num_entities} entities...", flush=True)
    for i in range(num_entities):
        try:
            # Convert WN18RR internal ID to WordNet synset to get the definition
            offset = int(dataset.id2entity[i])
            synset = wn.synset_from_pos_and_offset('n', offset)
            glosses[i] = synset.definition()
        except Exception:
            continue 
    return glosses

def augment_with_inverses(triples, num_relations):
    """
    Logic: For a triple (h, r, t), adds a new triple (t, r + offset, h).
    Importance: Standard KGE practice. It enables the model to learn 
    bidirectional relationships (e.g., if 'A is part of B', then 'B contains A').
    """
    h, r, t = triples[:, 0], triples[:, 1], triples[:, 2]
    inv_triples = torch.stack([t, r + num_relations, h], dim=1)
    return torch.cat([triples, inv_triples], dim=0)

# ==============================================================================
# 3. MAIN EXECUTION PIPELINE
# ==============================================================================

def main(enrich_data=True, similarity_threshold=0.92, max_edges_per_node=5):
    # --- DEVICE DETECTION ---
    # Logic: Checks hardware availability in order of performance.
    device = torch.device("cuda" if torch.cuda.is_available() else 
                          ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"🚀 Training on: {device}", flush=True)

    # --- DATASET LOADING ---
    dataset = KnowledgeGraphDataset()
    train_triples = dataset.train_triples.clone()
    num_rels_orig = dataset.num_relations

    # --- SEMANTIC ENRICHMENT PIPELINE ---
    if enrich_data:
        print("✨ Starting Textual Enrichment...", flush=True)
        entity_glosses = get_nltk_glosses(dataset)

        # Logic: Injecting text into PyTorch Geometric internal storage.
        # Importance: Custom attributes must be visible to the '.data' object 
        # for GNN message-passing functions to access them later.
        dataset.entity_glosses = entity_glosses
        dataset.data.entity_glosses = entity_glosses
        if hasattr(dataset.data, '_store'):
            dataset.data._store['entity_glosses'] = entity_glosses

        try:
            # 1. Generate NLP Embeddings (using Sentence-BERT)
            add_text_embeddings(dataset)
            
            # 2. Create Semantic Similarity Edges
            # Importance: Connects entities with similar meanings even if 
            # no relation existed, providing "semantic shortcuts" for the GNN.
            add_semantic_edges(dataset, threshold=similarity_threshold, max_edges_per_node=max_edges_per_node)

            # 3. Merge new edges into the training set
            semantic_triples = torch.cat([dataset.data.edge_index.T, dataset.data.edge_type.unsqueeze(1)], dim=1)
            semantic_rel_id = int(dataset.data.edge_type.max().item())
            semantic_only = semantic_triples[semantic_triples[:, 1] == semantic_rel_id]
            
            train_triples = torch.cat([train_triples, semantic_only], dim=0)
            num_rels_orig = semantic_rel_id + 1
            print(f"✅ Data enriched. Total triples: {train_triples.shape[0]}", flush=True)
            
        except Exception as e:
            print(f"⚠️ Enrichment logic failed: {e}. Proceeding with base data.", flush=True)
            enrich_data = False

    # --- GRAPH COMPACTION ---
    # Logic: .contiguous() ensures tensor data is stored in a single memory block.
    # Importance: Crucial for Mac/MPS stability. Prevents 'fragmented memory' errors 
    # during high-speed R-GCN message passing.
    dataset.data.edge_index = dataset.data.edge_index.contiguous()
    dataset.data.edge_type = dataset.data.edge_type.contiguous()
    
    # --- 1-N TRAINING PREPARATION ---
    # Logic: Grouping all true tails for every (Head, Relation) pair.
    # Importance: This allows the BCETrainer to score one pair against ALL 40k entities 
    # simultaneously, which is significantly faster than scoring single triples.
    print("🔄 Adding reciprocal relations...", flush=True)
    train_triples_augmented = augment_with_inverses(train_triples, num_rels_orig)
    total_relations = num_rels_orig * 2

    print("📊 Building Multi-Hot Matrix...", flush=True)
    unique_pairs, inverse_indices = torch.unique(train_triples_augmented[:, :2], dim=0, return_inverse=True)
    true_tails_matrix = torch.zeros((len(unique_pairs), dataset.num_entities), dtype=torch.uint8)
    true_tails_matrix[inverse_indices, train_triples_augmented[:, 2]] = 1

    # Logic: Collect all known facts from all splits.
    # Importance: Used for 'Filtered' evaluation—ensuring valid training facts 
    # don't penalize the model when it's testing on similar test facts.
    all_known = torch.cat([dataset.train_triples, dataset.valid_triples, dataset.test_triples], dim=0)

    # ==============================================================================
    # 4. MODEL COMPARISON LOOP
    # ==============================================================================
    model_names = ['distmult', 'transe', 'rgcn']
    for name in model_names:
        print(f"\n--- Starting {name.upper()} ---", flush=True)
        
        # MODEL INITIALIZATION
        if name == 'rgcn':
            # R-GCN requires setting the graph indices specifically for convolution
            dataset.data.edge_index = train_triples_augmented[:, [0, 2]].t().contiguous()
            dataset.data.edge_type = train_triples_augmented[:, 1].contiguous()
            model = RGCNModel(dataset.num_entities, total_relations, data=dataset, device=device).to(device)
        elif name == 'transe':
            model = TransEModel(dataset.num_entities, total_relations).to(device)
        else:
            model = DistMultModel(dataset.num_entities, total_relations).to(device)

        # TRAINER & EVALUATOR SETUP
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        trainer = BCETrainer(model, optimizer, device, true_tails_matrix, label_smoothing=0.1)
        evaluator = Evaluator(model, device, all_known_triples=all_known, filtered=True)
        stopper = EarlyStopping(model_name=name, patience=3)

        # --- TRAINING LOOP ---
        total_start = time.time()
        for epoch in range(1, 101):
            # We use a smaller batch size for RGCN to prevent Mac memory hangs
            # but keep the larger batch size for faster training on other models.
            if name == 'rgcn':
                current_batch_size = 32  # Smaller batch for R-GCN stability
            else:
                current_batch_size = 128 # Standard batch for TransE/DistMult
            
            avg_loss = trainer.train_epoch(
                unique_pairs, 
                dataset.num_entities, 
                batch_size=current_batch_size, 
                epoch_num=epoch
            )            
            # Evaluate periodically (every 10 epochs)
            if epoch % 10 == 0:
                metrics = evaluator.evaluate(dataset.test_triples, dataset.num_entities)
                print(f"🔍 [Epoch {epoch}] MRR: {metrics['MRR']:.4f} | Hits@1: {metrics['Hits@1']:.4f} | Hits@3: {metrics['Hits@3']:.4f} | Hits@10: {metrics['Hits@10']:.4f}", flush=True)
                
                # Check Early Stopping
                stopper(metrics['MRR'], model)
                if stopper.early_stop: break
        
        print(f"⏱️ Total Time for {name}: {(time.time() - total_start)/60:.2f} mins", flush=True)

if __name__ == "__main__":
    # To run without enrichment for speed tests, set enrich_data=False
    main(enrich_data=True, similarity_threshold=0.92, max_edges_per_node=3)