import torch
import torch.nn as nn
from torch.nn import functional as F

class attention(nn.Module):
    def __init__(self, length, num_tokens, v, key, query, value):
        super().__init__()
        self.length = length # t
        self.num_tokens = num_tokens # k
        self.v = v
        self.Wk = key
        self.Wq = query
        self.Wv = value

        pos = torch.arange(self.length, dtype=torch.long, device = "cuda")
        pos = pos.view(-1,1) - pos.view(1, -1)
        pos[pos<0] = 0
        self.register_buffer("pos", pos)


    def forward(self, e, previous_layer):
        # K = self.Wk(previous_layer).transpose(-2, -1)
        Q = self.Wq(e)
        V = self.Wv(e)
        pos_embd = self.v(self.pos).reshape((self.length, self.length))
        # attention = (Q @ K + torch.einsum("BTe,Tt->BTt", Q, pos_embd)).tril()
        # return attention @ V
        if self.Wk:
            attention = (Q @ previous_layer.transpose(-2,-1)).tril()
            return attention @ V
        else:
            attention = (torch.einsum("BTe,Tt->BTt", e, pos_embd)).tril()
            # print(attention @ V)
        return attention @ e

class min_model(nn.Module):
    """Minimal model"""

    def __init__(self, config):
        length = config.block_size # t
        num_tokens = config.vocab_size # k
        super().__init__()
        self.length = length # t
        self.num_tokens = num_tokens # k
        zero = lambda x: torch.zeros_like(x)
        # test = lambda x: (x >= 1)/(x+1)**2 *.6 + (x==99)*.1
        self.layer_one = attention(length, num_tokens, nn.Embedding(length, 1), False, nn.Linear(num_tokens, num_tokens, bias=False), nn.Linear(num_tokens, num_tokens, bias=False))
        self.layer_two = attention(length, num_tokens, zero, nn.Linear(num_tokens, num_tokens, bias=False), nn.Linear(num_tokens, num_tokens, bias=False), nn.Linear(num_tokens, num_tokens, bias=False))
        

        self.token_embedding = lambda x: F.one_hot(x, num_tokens).float()
        self.apply(self._init_weights)  
        # torch.nn.init.normal_(self.layer_two.Wk.weight, mean=0.0, std=0.1)
        # torch.nn.init.normal_(self.layer_two.Wk.weight, mean=0.1, std=.1)
        # torch.nn.init.normal_(self.layer_two.Wv.weight, mean=0.1, std=.1)
        # torch.nn.init.normal_(self.layer_one.Wq.weight, mean=0.1, std=.1)
        # torch.nn.init.normal_(self.layer_one.Wv.weight, mean=0.0, std=0.1)

        # self.layer_one.v.weight.data = torch.zeros_like(self.layer_one.v.weight.data)
        # self.layer_one.v.weight.data[1] = 1.0
        # self.layer_one.v.weight.data[0] = 1.0
        # self.layer_one.Wv.weight.data = torch.eye(num_tokens)
        # self.layer_one.Wq.weight.data = torch.eye(num_tokens)
        # self.layer_two.Wv.weight.data = torch.eye(num_tokens)/10
        # self.layer_two.Wq.weight.data = torch.eye(num_tokens)
        # self.layer_two.Wk.weight.data = torch.eye(num_tokens)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=.1, std=0.02)
            # torch.nn.init.constant_(module.weight, 0.1)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.1, std=0.02)
            # torch.nn.init.constant_(module.weight, 0.01)
    
    def forward(self, idx, targets=None):
        # M @ e
        e = self.token_embedding(idx) # shape b x t x k

        l1 = self.layer_one(e, e)
        output = self.layer_two(e, l1)
        logits = output

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            # logits += .0001
            # logits = torch.log(logits)
            # loss = F.nll_loss(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss
        
    
    def configure_optimizers(self, train_config):
        optim_groups = []
        # if self.key:
        #     optim_groups.append({"params": [p for _, p in self.Wk.named_parameters()], "weight_decay": 0})
        # if self.query:
        #     optim_groups.append({"params": [p for _, p in self.Wq.named_parameters()], "weight_decay": 0})
        # if self.value:
        #     optim_groups.append({"params": [p for _, p in self.Wv.named_parameters()], "weight_decay": 0})
        # optim_groups.append({"params": [p for _, p in self.v.named_parameters()], "weight_decay": 0})

        # optimizer = torch.optim.AdamW(self.parameters(), lr=train_config.learning_rate, betas=train_config.betas)
        optimizer = torch.optim.SGD(self.parameters(), lr=train_config.learning_rate)
        return optimizer