import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, EsmModel
from datasets import load_dataset
import random
import bisect
import numpy as np
from scipy.stats import binom

# ==========================================
# 1. Model Definition (ESM-8M + Regression Head)
# ==========================================
class ESMForAffinity(nn.Module):
    def __init__(self, model_name="facebook/esm2_t6_8M_UR50D"):
        super().__init__()
        self.esm = EsmModel.from_pretrained(model_name)
        # 320 is the hidden size for ESM-8M
        self.mlp = nn.Sequential(
            nn.Linear(320, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        # Mean Pooling over sequence length
        hidden_states = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
        sum_mask = mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        pooled = sum_embeddings / sum_mask
        
        # MLP to scalar
        affinity = self.mlp(pooled)
        return affinity.squeeze(-1) # Output shape: (Batch, )

    def reset_parameters(self):
        # Function to completely re-initialize the model
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()
        self.esm = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")

# Functional forward for Meta-Learning
def functional_forward(model, params_dict, input_ids, attention_mask):
    """
    Simulates a forward pass using explicit parameters dict (fast_weights)
    to keep the computation graph alive for outer gradients.
    (In production, libraries like functorch or torchopt are cleaner, 
    but this represents the fundamental PyTorch mechanic).
    """
    # For a complex HuggingFace model, torch.func.functional_call is highly recommended
    from torch.func import functional_call
    return functional_call(model, params_dict, (input_ids, attention_mask)).squeeze(-1)

# ==========================================
# 2. Meta-Training Logic (DataRater Bilevel Loop)
# ==========================================
def train_datarater(
    train_loader, val_loader, 
    n_meta_steps=5000, 
    n_inner_models=8,
    lifetime=2000,
    T_window=2,
    use_first_order_ablation=False, # Task 'c' (disable backprop through the unrolled state)
    sample_one_inner=False          # Task 'f' (default False: iterate all 8; True: randomly pick 1)
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Init DataRater
    data_rater = ESMForAffinity().to(device)
    rater_opt = torch.optim.Adam(data_rater.parameters(), lr=1e-4)
    
    # Init Inner Population
    population = [ESMForAffinity().to(device) for _ in range(n_inner_models)]
    inner_lr = 1e-4
    tau = 0.5 # Softmax temperature
    
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)
    
    data_rater.train()
    for step in range(n_meta_steps):
        
        # 1. Staggered resets (Task 'a'): must traverse every model in the outer loop to keep lifetime schedule aligned
        for i, inner_model in enumerate(population):
            offset = (n_inner_models - 1 - i) * (lifetime // n_inner_models)
            if step > 0 and (step + offset) % lifetime == 0:
                inner_model.reset_parameters()

        # 2. Decide how many models to compute meta-gradients for this step
        if sample_one_inner:
            models_to_process = [random.randint(0, n_inner_models - 1)]
        else:
            models_to_process = list(range(n_inner_models))
            
        rater_opt.zero_grad()
        meta_grads_accumulator = []
        
        # 3. Process the selected inner models
        for m_idx in models_to_process:
            inner_model = population[m_idx]
            fast_weights = dict(inner_model.named_parameters())
            
            # --- Inner Loop: Truncated window of 2 ---
            for t in range(T_window):
                try:
                    x_in = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    x_in = next(train_iter)
                    
                input_ids = x_in['input_ids'].to(device)
                mask = x_in['attention_mask'].to(device)
                targets = x_in['affinity'].to(device)
                
                # DataRater scoring + in-batch Softmax (Task 'd')
                raw_scores = data_rater(input_ids, mask)
                weights = F.softmax(raw_scores / tau, dim=0)
                
                # Weighted MSE loss
                preds = functional_forward(inner_model, fast_weights, input_ids, mask)
                mse_per_sample = F.mse_loss(preds, targets, reduction='none')
                inner_loss = torch.sum(weights * mse_per_sample)
                
                # Compute inner gradients
                grads = torch.autograd.grad(
                    inner_loss, 
                    fast_weights.values(), 
                    create_graph=not use_first_order_ablation
                )
                
                # Task 'c': ablation (if enabled, stop gradients through the unroll)
                if use_first_order_ablation:
                    grads = [g.detach() for g in grads]
                    
                # Functional weight update
                fast_weights = {name: w - inner_lr * g for (name, w), g in zip(fast_weights.items(), grads)}
                
            # --- Outer Loop computation ---
            try:
                x_out = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                x_out = next(val_iter)
                
            out_input_ids = x_out['input_ids'].to(device)
            out_mask = x_out['attention_mask'].to(device)
            out_targets = x_out['affinity'].to(device)
            
            # Compute outer loss using weights after T steps
            outer_preds = functional_forward(inner_model, fast_weights, out_input_ids, out_mask)
            outer_loss = F.mse_loss(outer_preds, out_targets, reduction='mean')
            
            # Compute meta-gradients and store in accumulator
            meta_grads = torch.autograd.grad(outer_loss, data_rater.parameters())
            meta_grads_accumulator.append(meta_grads)
            
            # Sync parameters and fully detach the graph (Task 'b')
            with torch.no_grad():
                for name, param in inner_model.named_parameters():
                    param.copy_(fast_weights[name].detach())
                    
        # 4. Average meta-gradients and update DataRater
        for rater_param, meta_grads_for_this_param in zip(data_rater.parameters(), zip(*meta_grads_accumulator)):
            # Stack meta-grads from all models handled this step (8 or 1) and average
            rater_param.grad = torch.stack(meta_grads_for_this_param).mean(dim=0)
            
        rater_opt.step()
        
    return data_rater

# ==========================================
# 3. CDF Building and Dataset Filtering (Task 'd')
# ==========================================
def filter_dataset(data_rater, original_dataset, N_ref=10000, B=256, keep_ratio=0.7):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_rater.eval()
    
    print(f"Building Empirical CDF using {N_ref} random samples...")
    # Get N_ref random samples
    indices = random.sample(range(len(original_dataset)), N_ref)
    ref_scores = []
    
    with torch.no_grad():
        for idx in indices:
            sample = original_dataset[idx]
            # Assumes collate handles adding batch dim
            input_ids = sample['input_ids'].unsqueeze(0).to(device)
            mask = sample['attention_mask'].unsqueeze(0).to(device)
            score = data_rater(input_ids, mask).item()
            ref_scores.append(score)
            
    ref_scores.sort() # CDF Lookup Table
    
    K = int(B * keep_ratio)
    filtered_indices = []
    
    print("Filtering dataset using P_accept(x)...")
    with torch.no_grad():
        for idx in range(len(original_dataset)):
            sample = original_dataset[idx]
            input_ids = sample['input_ids'].unsqueeze(0).to(device)
            mask = sample['attention_mask'].unsqueeze(0).to(device)
            
            # Score individually
            score = data_rater(input_ids, mask).item()
            
            # Find percentile 'p'
            pos = bisect.bisect_left(ref_scores, score)
            p = pos / N_ref
            
            # P_accept formula from DataRater
            p_accept = binom.cdf(K - 1, B - 1, 1 - p)
            
            if random.random() < p_accept:
                filtered_indices.append(idx)
                
    filtered_dataset = original_dataset.select(filtered_indices)
    print(f"Original size: {len(original_dataset)}, Filtered size: {len(filtered_dataset)}")
    return filtered_dataset
