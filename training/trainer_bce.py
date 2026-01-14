import torch
import torch.nn as nn
import time

class BCETrainer:
    """
    Handles the training loop using Binary Cross Entropy (BCE).
    It treats link prediction as a multi-label classification task over all entities.
    """
    def __init__(self, model, optimizer, device, true_tails_matrix=None, label_smoothing=0.1):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        
        # Label Smoothing prevents the model from becoming overconfident.
        # It replaces 1.0 with ~0.9 and 0.0 with a tiny positive value.
        self.label_smoothing = label_smoothing
        
        # true_tails_matrix is a [num_unique_pairs, num_entities] multi-hot matrix.
        # It tells the trainer exactly which tails are 'true' for a given (h, r).
        self.true_tails = true_tails_matrix
        
        # BCEWithLogitsLoss combines a Sigmoid layer and the BCE loss in one class,
        # which is more numerically stable than using them separately.
        self.loss_func = nn.BCEWithLogitsLoss()

        # Efficiency logic: GPUs (CUDA/MPS) handle large dense matrices well.
        # CPUs often struggle with memory, so we branch the logic.
        self.use_dense = device.type in ["cuda", "mps"]
        if not self.use_dense:
            print("⚠️ CPU detected, switching to memory-efficient training mode.")

    def train_epoch(self, unique_pairs, num_entities, batch_size=256, epoch_num=0):
        """
        Runs one full pass over the unique (head, relation) pairs in the training set.
        """
        self.model.train()  # Enable training mode (activates Dropout/BatchNorm)
        total_loss = 0.0
        
        # Shuffle the data every epoch to ensure the model doesn't learn the order of triples.
        indices = torch.randperm(len(unique_pairs))

        # --- LABEL SMOOTHING CALCULATION ---
        # Instead of 1 (True) and 0 (False), we use these 'smoothed' values.
        # This acts as a regularizer, helping the model generalize better.
        pos_val = 1.0 - self.label_smoothing
        neg_val = self.label_smoothing / num_entities

        t0 = time.time()

        for i in range(0, len(unique_pairs), batch_size):
            # Batching: Extract a slice of indices and move corresponding pairs to GPU/MPS.
            batch_idx = indices[i:i + batch_size]
            batch_pairs = unique_pairs[batch_idx].to(self.device, non_blocking=True)

            if self.use_dense:
                # --- ACCELERATED MODE (CUDA/MPS) ---
                
                # 1. Forward Pass: Get scores for ALL entities for every (h, r) in the batch.
                # logits shape: [batch_size, num_entities]
                logits = self.model.forward_all(batch_pairs[:, 0], batch_pairs[:, 1])
                
                # 2. Target Preparation: Fetch the multi-hot row for this batch.
                batch_targets_binary = self.true_tails[batch_idx].to(self.device, non_blocking=True)
                
                # 3. Apply Smoothing: Replace 1s with pos_val and 0s with neg_val.
                targets = torch.where(batch_targets_binary > 0.5, pos_val, neg_val)

                # 4. Optimization: Zero gradients, calculate loss, backpropagate, and step.
                # set_to_none=True is a small optimization to save memory.
                self.optimizer.zero_grad(set_to_none=True)
                loss = self.loss_func(logits, targets)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.detach().item()

            else:
                # --- MEMORY-EFFICIENT MODE (CPU) ---
                # Instead of huge dense matrix math, we process pairs one by one to save RAM.
                batch_loss = 0.0
                for j in range(batch_pairs.size(0)):
                    h, r = batch_pairs[j]
                    logits = self.model.forward_all(h.unsqueeze(0), r.unsqueeze(0))

                    # Create a target vector for just this one pair.
                    target = torch.full((num_entities,), neg_val, device=self.device)
                    if self.true_tails is not None:
                        # Find indices of true tails and set them to pos_val.
                        true_indices = self.true_tails[batch_idx[j]].nonzero(as_tuple=True)[0]
                        target[true_indices] = pos_val

                    loss = self.loss_func(logits.squeeze(0), target)
                    batch_loss += loss

                # Average the individual losses for the batch.
                batch_loss = batch_loss / batch_pairs.size(0)
                self.optimizer.zero_grad(set_to_none=True)
                batch_loss.backward()
                self.optimizer.step()
                total_loss += batch_loss.detach().item()

        # Monitoring performance
        epoch_time = time.time() - t0
        avg_loss = total_loss / (len(unique_pairs) // batch_size + 1)
        print(f"📦 Epoch {epoch_num:03d} | Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")
        return avg_loss