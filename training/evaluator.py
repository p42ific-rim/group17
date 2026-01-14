import torch
from collections import defaultdict

class Evaluator:
    """
    Evaluates Knowledge Graph models using the Link Prediction task.
    For every test triple (h, r, t), the model ranks the true tail (t) 
    against all other possible entities in the graph.
    """
    def __init__(self, model, device, all_known_triples=None, filtered=True):
        self.model = model
        self.device = device
        self.filtered = filtered
        # A dictionary mapping (head, relation) to a set of all known valid tails.
        self.hr_to_tails = defaultdict(set)
        
        # Build the filtering dictionary:
        # This includes triples from Train, Validation, and Test sets.
        # It's used to "mask" known truths during evaluation so they don't 
        # push down the rank of the specific triple we are currently testing.
        if filtered and all_known_triples is not None:
            for h, r, t in all_known_triples.tolist():
                self.hr_to_tails[(h, r)].add(t)

    @torch.no_grad() # Disables gradient tracking to save memory and compute speed
    def evaluate(self, test_triples, num_entities):
        """
        Calculates Mean Reciprocal Rank (MRR) and Hits@K metrics.
        """
        self.model.eval() # Puts model in evaluation mode (affects Dropout and BatchNorm)
        ranks = []

        # 1. OPTIMIZATION: Cache GNN embeddings
        # For R-GCN, we don't want to re-run the expensive Message Passing for every batch.
        # We run it once at the start of evaluation and reuse the 'contextual' embeddings.
        cached_x = None
        if hasattr(self.model, "get_enriched_embeddings"):
            cached_x = self.model.get_enriched_embeddings()

        # Process in batches to avoid GPU memory overflow
        for i in range(0, len(test_triples), 100):
            batch = test_triples[i:i + 100].to(self.device)
            h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]

            # 2. Get scores for ALL possible tails for these (h, r) pairs.
            # 'scores' shape: [batch_size, num_entities]
            scores = self.model.forward_all(h, r, cached_x=cached_x)

            # 3. Filtered Logic: Mask all other known true triples
            # This is the 'Filtered' setting from Dettmers et al. (ConvE).
            # We set the scores of all OTHER valid tails to a very low number (-1e9)
            # so they are not ranked higher than the current test tail.
            if self.filtered:
                for idx in range(batch.size(0)):
                    h_i, r_i, true_t = h[idx].item(), r[idx].item(), t[idx].item()
                    known_tails = self.hr_to_tails.get((h_i, r_i), set())
                    for other_t in known_tails:
                        if other_t != true_t:
                            scores[idx, other_t] = -1e9

            # 4. Rank Calculation
            # target_scores: The score the model gave to the actual true tail.
            target_scores = scores[torch.arange(batch.size(0)), t].unsqueeze(1)
            
            # Rank: Count how many entities have a score GREATER than the true tail.
            # Adding +1 converts 0-indexed count to 1-indexed rank.
            rank = torch.sum(scores > target_scores, dim=1) + 1
            ranks.extend(rank.cpu().tolist())

        # 5. Metric Aggregation
        ranks = torch.tensor(ranks, dtype=torch.float)
        return {
            # MRR: Mean of the inverse ranks (1/rank). More sensitive to top performance.
            "MRR": torch.mean(1.0 / ranks).item(),
            # Hits@K: Percentage of test triples where the true tail ranked in the top K.
            "Hits@1": torch.mean((ranks <= 1).float()).item(),
            "Hits@3": torch.mean((ranks <= 3).float()).item(),
            "Hits@10": torch.mean((ranks <= 10).float()).item(),
        }