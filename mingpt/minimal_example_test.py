import torch
import torch.nn as nn
from torch.nn import functional as F


class min_model(nn.Module):
    """Minimal model"""

    def __init__(self, config, key=True, query=True, value=True):
        super().__init__()
        length = config.block_size # t
        num_tokens = config.vocab_size # k
        self.length = length # t
        self.num_tokens = num_tokens # k
        self.v = nn.Embedding(length, 1)
        self.key, self.query, self.value = key, query, value
        self.Wk, self.Wq, self.Wv = nn.Linear(num_tokens, num_tokens, bias=False), nn.Linear(num_tokens, num_tokens, bias=False), nn.Linear(num_tokens, num_tokens, bias=False) # shape k x k
        self.token_embedding = lambda x: F.one_hot(x, num_tokens).float()
        self.register_buffer("bias", torch.tril(torch.ones(self.length, self.length))
                                     .view(1, self.length, self.length) == 0)
        self.apply(self._init_weights)
        # torch.nn.init.normal_(self.Wk.weight, mean=0.0, std=0.02)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        # M @ e
        e = self.token_embedding(idx) # shape b x t x k
        pos = torch.arange(self.length, dtype=torch.long, device=idx.device)
        pos = pos.view(-1,1) - pos.view(1, -1)
        pos[pos<0] = 0
        pos_embd = self.v(pos)[:, :, 0].tril(diagonal=0)
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
        logits = output + .001

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            # loss = F.mse_loss(logits[:,-1], F.one_hot(targets[:,-1], logits.shape[-1]).float())

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

        # optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        optimizer = torch.optim.SGD(optim_groups, lr=train_config.learning_rate)
        return optimizer