"""minimal example"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from mingpt.utils import set_seed


class min_model(nn.Module):
    """Minimal model"""

    def __init__(self, length, num_tokens, k=True, q=False, v=False, seed=1):
        super().__init__()
        set_seed(seed)
        self.length = length # t
        self.num_tokens = num_tokens # k
        self.key, self.query, self.value = k, q, v
        self.v = nn.Embedding(length, 1)
        self.Wk = nn.Linear(num_tokens, num_tokens, bias=False) # shape k x k
        self.Wq = nn.Linear(num_tokens, num_tokens, bias=False) # shape k x k
        self.Wv = nn.Linear(num_tokens, num_tokens, bias=False) # shape k x k
        self.token_embedding = lambda x: F.one_hot(x, num_tokens).float()

        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(self.length, self.length))
                                     .view(1, self.length, self.length) == 0)
        torch.nn.init.constant_(self.v.weight, 0.02)
        torch.nn.init.constant_(self.Wk.weight, 0.02)
        torch.nn.init.normal_(self.Wq.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.Wv.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        e = self.token_embedding(idx) # shape b x t x k
        pos = torch.arange(self.length, dtype=torch.long, device=idx.device)
        pos = pos.view(-1,1) - pos.view(1, -1)
        pos[pos<0] = 0
        pos_embd = self.v(pos)[:, :, 0]
        
        # # set the upper triangular part to 0
        pos_embd = pos_embd.tril(diagonal=0)
        layer_one = pos_embd @ e

        # (M @ e) @ W_k^T
        if self.key:
            K = self.Wk(layer_one)
        else:
            K = layer_one
        if self.query:
            attention = self.Wq(e) @ K.transpose(1,2)
        else:
            attention = e @ K.transpose(1, 2)
        masked_attention = attention.masked_fill(self.bias, 0)
        if self.value:
            output = masked_attention @ self.Wv(e)
        else:
            output = masked_attention @ e
        logits = output
        
        loss = None
        if targets is not None:
            mask = targets != -1
            logits = logits.view(-1, logits.size(-1))[mask.view(-1)]
            targets = targets.view(-1)[mask.view(-1)]
            loss = F.multi_margin_loss(logits, targets, margin=10)
            # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss
        
    
    def configure_optimizers(self, train_config):
        optim_groups = []
        if self.key:
            optim_groups.append({"params": [p for _, p in self.Wk.named_parameters()], "weight_decay": 0})
        if self.query:
            optim_groups.append({"params": [p for _, p in self.Wq.named_parameters()], "weight_decay": 0})
        if self.value:
            optim_groups.append({"params": [p for _, p in self.Wv.named_parameters()], "weight_decay": 0})
        optim_groups.append({"params": [p for _, p in self.v.named_parameters()], "weight_decay": 0})

        optimizer = torch.optim.SGD(optim_groups, lr=train_config.learning_rate)
        return optimizer