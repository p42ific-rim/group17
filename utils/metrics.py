"""
metrics.py
----------
Utility functions for ranking-based evaluation metrics.
"""


def compute_rank(scores, true_index):
    """
    Computes the rank of the true entity given a score vector.

    Parameters
    ----------
    scores : torch.Tensor
        Scores for all candidate entities.
    true_index : int
        Index of the true entity.

    Returns
    -------
    int
        Rank (1 = best).
    """
    sorted_indices = scores.argsort(descending=True)
    return (sorted_indices == true_index).nonzero(as_tuple=True)[0].item() + 1


def print_metrics(metrics, title="Metrics"):
    print(f"--- {title} ---")
    print(f"MRR      : {metrics['MRR']:.4f}")
    print(f"Hits@1   : {metrics['Hits@1']:.4f}")
    print(f"Hits@3   : {metrics['Hits@3']:.4f}")
    print(f"Hits@10  : {metrics['Hits@10']:.4f}")
    print("--------------------")
