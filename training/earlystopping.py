import os
import torch

class EarlyStopping:
    """
    Monitors a specific metric (usually MRR or Hits@10) during validation.
    If the metric stops improving for a set number of checks (patience), 
    it terminates training to prevent overfitting and saves the best model.
    """
    def __init__(self, model_name, patience=3):
        # The identifier for the model (e.g., 'transe', 'rgcn') used for the filename
        self.model_name = model_name
        
        # 'Patience' is the number of evaluation cycles to wait before giving up.
        # Too low: you might stop early due to a temporary dip. 
        # Too high: you waste time and energy training an overfit model.
        self.patience = patience  
        
        # 'counter' tracks the consecutive number of evaluations without an improvement.
        self.counter = 0          
        
        # 'best_score' keeps track of the highest validation metric achieved.
        self.best_score = None    
        
        # 'early_stop' is a boolean flag that the training loop in main.py checks
        # to decide whether to break out of the epoch loop.
        self.early_stop = False   
        
        # Initialization logic: create a folder to store the trained weights (.pt files).
        # This prevents the script from crashing if the folder is missing.
        if not os.path.exists('saved_models'): 
            os.makedirs('saved_models')

    def __call__(self, current_score, model):
        """
        The __call__ method allows the object to be used like a function: stopper(score, model).
        It is triggered every time the evaluation runs (e.g., every 10 epochs).
        """
        
        # 1. INITIALIZATION OR IMPROVEMENT
        # If this is the first run, or if the current score is strictly better 
        # than the previous best, we have found a superior set of weights.
        if self.best_score is None or current_score > self.best_score:
            self.best_score = current_score
            
            # --- CRITICAL STEP: SAVE THE BEST STATE ---
            # model.state_dict() saves only the learned weights (tensors), not the whole 
            # object structure. This is the most memory-efficient way to save PyTorch models.
            torch.save(model.state_dict(), f"saved_models/{self.model_name}_best.pt")
            
            print(f"🌟 New Best Score! Model saved.")
            
            # Reset the patience counter to 0 because we found an improvement.
            self.counter = 0 
            
        # 2. STAGNATION
        # If the score did not improve, we increment the 'strike' counter.
        else:
            self.counter += 1
            print(f"⚠️ No improvement for {self.counter}/{self.patience} evaluations.")
            
            # 3. TERMINATION TRIGGER
            # If the number of failures reaches our patience limit, we set early_stop to True.
            # The main loop will see this and exit, preserving the best weights on disk.
            if self.counter >= self.patience:
                self.early_stop = True