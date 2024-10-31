import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class min_model(nn.Module):
    """Minimal model"""

    def __init__(self, config):
        self.test = False
        length = config.block_size # t
        num_tokens = config.vocab_size # k
        super().__init__()
        self.length = length # t
        self.num_tokens = num_tokens # k
        self.Wq = nn.Linear(num_tokens, num_tokens, bias=False)
        self.Wv = nn.Linear(num_tokens, num_tokens, bias=False)
        self.v = nn.Embedding(length, 1)
        
        
        self.token_embedding = lambda x: F.one_hot(x, num_tokens).float()
        self.apply(self._init_weights)
        pos = torch.arange(self.length, dtype=torch.long)
        pos = pos.view(-1,1) - pos.view(1, -1)
        pos[pos<0] = 0
        self.register_buffer("pos", pos)
        self.register_buffer("bias", torch.tril(torch.ones(self.length, self.length))
                                     .view(1, self.length, self.length) == 0)
        
        # self.register_buffer("helper", torch.arange(1, 1 + length))
        with torch.no_grad():
            self.Wv.weight += math.e
            # self.Wq.weight = torch.nn.Parameter(torch.eye(num_tokens)*.1)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # torch.nn.init.normal_(module.weight, mean=1, std=0.02)
            torch.nn.init.constant_(module.weight, 0)
        elif isinstance(module, nn.Embedding):
            # torch.nn.init.normal_(module.weight, mean=0.01, std=0.02)
            torch.nn.init.constant_(module.weight, 0)

    def softmax_approx(self, attn):
        attn = attn.tril()
        attn += 1 - (attn.sum(axis=-1)/self.helper)[..., None]
        attn = attn.tril()
        attn /= self.helper
        return attn

    def forward(self, idx, targets=None):
        # M @ e
        e = self.token_embedding(idx) # shape b x t x k
        pos_embd = self.v(self.pos).squeeze()
        # l1 = pos_embd.tril() @ e
        attn = pos_embd
        attn = attn.masked_fill(self.bias, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        # attn = self.softmax_approx(attn)
        l1 = attn @ e
        
        # l2 = (self.Wq(e) @ l1.transpose(-2,-1)).tril() @ e #self.Wv(e)
        # l2 = (self.Wq.weight[0,0] * e @ l1.transpose(-2,-1)).tril() @ e
        attn = (self.Wq(e) @ l1.transpose(-2,-1))
        attn = attn.masked_fill(self.bias, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        # attn = self.softmax_approx(attn)
        l2 = attn @ e * self.Wv.weight[0,0]
        
        logits = l2

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None and self.test:
            # logits += .0001
            # # logits /= logits.sum(axis=-1)[..., None]
            # # logits = torch.log(logits)
            # logits += 1 - (logits.sum(axis=-1)/logits.size(-1))[..., None]
            # logits /= logits.size(-1)
            logits = F.softmax(logits, dim=-1)
            # logits = logits.log()
            # loss = F.nll_loss(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

            # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        elif targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss
        
    
    def configure_optimizers(self, train_config):
        # optimizer = torch.optim.AdamW(self.parameters(), lr=train_config.learning_rate, betas=train_config.betas)

        optim_groups = []
        optim_groups.append({"params": [p for _, p in self.Wq.named_parameters()], "weight_decay": 0})
        optim_groups.append({"params": [p for _, p in self.Wv.named_parameters()], "weight_decay": 0})
        optim_groups.append({"params": [p for _, p in self.v.named_parameters()], "weight_decay": 0})
        # optim_groups.append({"params": self.parameters(), "weight_decay": 0})
        optimizer = torch.optim.SGD(optim_groups, lr=train_config.learning_rate)
        optimizer.param_groups[-1]['lr'] = 1e0
        return optimizer