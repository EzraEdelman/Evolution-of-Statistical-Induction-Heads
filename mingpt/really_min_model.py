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
        self.v = nn.Embedding(4, 1, padding_idx=3)
        self.W = nn.Linear(self.num_tokens, self.num_tokens, bias=False) # shape k x k
        self.token_embedding = lambda x: F.one_hot(x, self.num_tokens).float()

        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(self.length, self.length))
                                     .view(1, self.length, self.length) == 0)
        torch.nn.init.constant_(self.v.weight, 0.02)
        torch.nn.init.normal_(self.v.weight, mean=0, std= 0.02)
        # torch.nn.init.normal_(self.W.weight, mean=0, std=0.02)
        # torch.nn.init.normal_(self.lm_head.weight, 0, 0.02)
        # torch.nn.init.constant_(self.lm_head.bias, 0)
        with torch.no_grad():
            torch.nn.init.constant_(self.W.weight, 0.02)
            # torch.nn.init.normal_(self.W.weight, mean=0, std=0.02)
            # self.v.weight = nn.Parameter(self.v.weight.abs())
            # self.v.weight = nn.Parameter(torch.zeros((self.length,1)).to("cuda"))
            # self.v.weight[1,0]=1
        pos = torch.arange(self.length, dtype=torch.long)
        pos = pos.view(-1,1) - pos.view(1, -1)
        pos[pos<0] = 3
        pos[pos>2] = 2
        self.register_buffer("pos", pos)
    
    def forward(self, idx, targets=None):
        e = self.token_embedding(idx) # shape b x t x k
        
        pos_embd = self.v(self.pos.to(idx.device)).squeeze()
        
        # # set the upper triangular part to 0
        pos_embd = pos_embd.tril(diagonal=0)
        layer_one = pos_embd @ e

        # (M @ e) @ W
        attention = e @ (layer_one).transpose(1, 2)
        masked_attention = attention.tril(diagonal=0)
        output = masked_attention @ e
        logits = output
        # logits = torch.log(logits +0.001)
        loss = None
        if targets is not None:
            # mask = (targets != -1).view(-1)
            # temp_logits = logits.view(-1, logits.size(-1))
            # targets = targets.view(-1)[mask]
            # loss = F.multi_margin_loss(temp_logits, targets, margin=10)
            loss = F.multi_margin_loss(logits.view(-1, logits.size(-1)), targets.view(-1), margin=10) #float("inf")
            # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        # return torch.log(logits+0.01), loss
        # return logits, loss
        # mini = logits.min(dim=-1)[0][:,:,None].expand(logits.shape)
        # print((logits - mini+0.001).shape)
        return F.softmax(logits, dim=-1), loss
        
    
    def configure_optimizers(self, train_config):
        optim_groups = []
        optim_groups.append({"params": [p for _, p in self.v.named_parameters()], "weight_decay": 0})
        optim_groups.append({"params": [p for _, p in self.W.named_parameters()], "weight_decay": 0})

        optimizer = torch.optim.SGD(optim_groups, lr=train_config.learning_rate)
        # optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer
