"""minimal example"""
import torch
import torch.nn as nn
from torch.nn import functional as F

class min_model(nn.Module):
    """Minimal model"""

    def __init__(self, config):
        super().__init__()
        self.length = config.block_size # t
        self.num_tokens = config.vocab_size # k
        self.v = nn.Embedding(self.length, 1)
        self.W = nn.Linear(self.num_tokens, self.num_tokens, bias=False) # shape k x k
        self.wte = lambda x: F.one_hot(x, self.num_tokens).float()

        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(self.length, self.length))
                                     .view(1, self.length, self.length) == 0)
        torch.nn.init.constant_(self.v.weight, 0)
        torch.nn.init.constant_(self.W.weight, 0.01)
        # torch.nn.init.normal_(self.W.weight, mean=0.02, std=0.02)
        # torch.nn.init.normal_(self.v.weight, mean=0.02, std=0.02)

        pos = torch.arange(self.length, dtype=torch.long)
        pos = pos.view(-1,1) - pos.view(1, -1)
        pos[pos<0] = 0
        self.register_buffer("pos", pos)
    
    def forward(self, idx, targets=None):
        _, T = idx.size()
        e = self.wte(idx) # shape b x t x k
        # layer one
        pos_embd = self.v(self.pos[:T,:T]).squeeze()
        attention = pos_embd
        masked_attention = attention.tril(diagonal=0)
        attention.masked_fill(self.bias[:T,:T], float('-inf'))
        attention = F.softmax(attention, dim=-1)
        layer_one = attention @ e
        # layer_one = masked_attention @ e

        # layer two
        attention = e @ self.W(layer_one).transpose(1, 2)
        masked_attention = attention.tril(diagonal=0)
        output = masked_attention @ e
        logits = output
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss
        
    
    def configure_optimizers(self, train_config):
        optim_groups = []
        optim_groups.append({"params": [p for _, p in self.W.named_parameters()], "weight_decay": 0})
        optim_groups.append({"params": [p for _, p in self.v.named_parameters()], "weight_decay": 0, "lr": 1})
        # optim_groups.append({"params": [p for _, p in self.v.named_parameters()], "weight_decay": 0, "lr": 4e3})
        # optim_groups.append({"params": [p for _, p in self.v.named_parameters()], "weight_decay": 0, "lr": 6.5e4})

        optimizer = torch.optim.SGD(optim_groups, lr=train_config.learning_rate)
        # optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer
