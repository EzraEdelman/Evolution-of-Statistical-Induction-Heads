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
        self.token_embedding = lambda x: F.one_hot(x, self.num_tokens).float()

        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(self.length, self.length))
                                     .view(1, self.length, self.length) == 0)
        torch.nn.init.constant_(self.v.weight, .02)
        # torch.nn.init.constant_(self.v.weight, 0)
        torch.nn.init.constant_(self.W.weight, 0)
        # torch.nn.init.normal_(self.W.weight, mean=0.5, std=0.2)
        # torch.nn.init.normal_(self.W.weight, mean=0.02, std=0.02)
        # torch.nn.init.constant_(self.W.bias, 0)
        # torch.nn.init.normal_(self.v.weight, mean=0.02, std=0.02)
        with torch.no_grad():
            # torch.nn.init.constant_(self.W.bias, 1/config.block_size)
            # self.v.weight = nn.Parameter(torch.zeros((self.length,1)).to("cuda"))
            # self.v.weight[0,0]=.02
            self.W.weight += nn.Parameter(torch.eye(self.num_tokens).to(config.device))

            # self.W.weight += torch.eye(self.num_tokens)
        pos = torch.arange(self.length, dtype=torch.long)
        pos = pos.view(-1,1) - pos.view(1, -1)
        pos[pos<0] = 0
        self.register_buffer("pos", pos)
    
    def forward(self, idx, targets=None):
        e = self.token_embedding(idx) # shape b x t x k
        
        pos_embd = self.v(self.pos.to(idx.device)).squeeze()
        # pos_embd.masked_fill(self.bias[:,:] == 0, float('-inf'))
        # pos_emdb = F.softmax(pos_embd, dim=-1)
        
        # # set the upper triangular part to 0
        pos_embd = pos_embd.tril(diagonal=0)
        # pos_embd = pos_embd ** 2
        layer_one = pos_embd @ e

        # (M @ e) @ W
        attention = e @ self.W(layer_one).transpose(1, 2)
        masked_attention = attention.tril(diagonal=0)
        output = masked_attention @ e
        logits = output
        # logits -= logits.min(dim=-1, keepdim=True)[0]
        # logits += 1e1
        # logits = torch.log(output + 1)
        # logits[output<=0] = torch.exp(output[output<=0])
        # logits = logits.maximum(torch.tensor([1e-12], device = logits.device))
        # logits = F.normalize(logits, p=1, dim=-1)
        # logits = torch.log(logits+.0001)
        # logits = torch.softmax(logits, dim=-1)
        
        loss = None
        if targets is not None:
            # mask = (targets != -1).view(-1)
            # temp_logits = logits.view(-1, logits.size(-1))
            # targets = targets.view(-1)[mask]
            # targets = targets[:,-1].squeeze()
            # print(targets[:,-1].shape, logits.shape)
            # loss = F.multi_margin_loss(logits[:,-1], targets[:,-1], margin=1000)
            # print((logits[:,-1].mean(axis=-1)).shape)
            # print((F.one_hot(targets[:,-1], logits.shape[-1]).float()).mean(axis=1).shape)
            # print(logits[:,-1].mean(axis=1))
            # loss = logits[:,-1].mean(axis=1) - (logits[:,-1] * F.one_hot(targets[:,-1], logits.shape[-1]).float()).mean(axis=1)
            # loss = loss.mean()
            # loss = F.multi_margin_loss(logits.view(-1, logits.size(-1)), targets.view(-1), margin=10)
            # print(f"shape targets: {targets.shape}, shape idx: {idx.shape}, shape logits: {logits.shape}")
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            
            # loss = F.nll_loss(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            # loss= F.mse_loss(logits[:,-1], targets[torch.arange(targets.shape[0]),idx[:,-1]])
            # loss = F.mse_loss(logits[:,-1], F.one_hot(targets[:,-1], logits.shape[-1]).float())
            # loss = (logits[:,-1]- targets[torch.arange(targets.shape[0]),idx[:,-1]]).abs().sum()
            # loss = (logits - F.one_hot(targets, logits.shape[-1])).abs().sum()
        return logits, loss
        # mini = logits.min(dim=-1)[0][:,:,None].expand(logits.shape)
        # print((logits - mini+0.001).shape)
        # return F.softmax(logits, dim=-1), loss
        
    
    def configure_optimizers(self, train_config):
        optim_groups = []
        optim_groups.append({"params": [p for _, p in self.W.named_parameters()], "weight_decay": 0})
        optim_groups.append({"params": [p for _, p in self.v.named_parameters()], "weight_decay": 0})

        optimizer = torch.optim.SGD(optim_groups, lr=train_config.learning_rate)
        # optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer
