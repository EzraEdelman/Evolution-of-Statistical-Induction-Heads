"""50
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F
import warnings
from mingpt.utils import CfgNode as CN

# -----------------------------------------------------------------------------


class Attention(nn.Module):
    """
    Multi-head Casual self-attention with relative positional encodings
    Cannot use nn.MultiheadAttention sadly since it does not support relative positional encodings
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.Q = nn.Linear(config.n_embd, config.n_embd, bias = False)
        self.K = nn.Linear(config.n_embd, config.n_embd, bias = False)
        self.V = nn.Linear(config.n_embd, config.n_embd, bias = False)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias = False)
        # relative embeddings
        self.wpe = nn.Embedding(config.block_size + 1, config.n_embd, padding_idx = config.block_size)
        self.attention_only = config.attention_only

        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.embd_dropout = nn.Dropout(config.embd_pdrop)
        self.ln = nn.LayerNorm(config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

        # creating relative position matrix
        pos = torch.arange(config.block_size, dtype=torch.long, device=config.device).unsqueeze(0)
        pos = pos.view(-1, 1) - pos.view(1, -1) #+ config.block_size - 1
        pos = torch.maximum(pos, torch.tensor(-1))
        pos[pos==-1] = config.block_size
        self.register_buffer("pos", pos)

        if not config.attention_only:
            self.ln_2 = nn.LayerNorm(config.n_embd)
            self.mlp = nn.ModuleDict(dict(
                c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
                c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
                act     = torch.nn.ReLU(),
                dropout = nn.Dropout(config.resid_pdrop),
            ))
            m = self.mlp
            self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # MLP forward

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        #layer norm
        ln_x = self.ln(x)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.K(ln_x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.Q(ln_x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.V(ln_x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        rel_pos = self.embd_dropout(self.wpe(self.pos[:T,:T]))
        rel_pos = rel_pos.view(T, T, self.n_head, C // self.n_head) # (T, T, nh, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        attn = (q @ k.transpose(-2, -1) + torch.einsum("Tthe,BhTe->BhTt", rel_pos, q)) * (1.0 / math.sqrt(k.size(-1)))
        # attn = (torch.einsum("BhTe, Bhte->BhTt", q, k) + torch.einsum("Tthe,BhTe->BhTt", rel_pos, k)) * (1.0 / math.sqrt(k.size(-1)))

        attn = attn.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        y = attn @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        x = self.resid_dropout(self.c_proj(y)) + x
        if not self.attention_only:
            x = self.mlpf(self.ln_2(x)) + x
        return x

class Relative_Transformer(nn.Module):
    """ Substantially Simplified (and Specialized) Experimental Model """

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        assert config.device is not None
        
        self.block_size = config.block_size
        self.device = config.device

        self.wte = nn.Embedding(config.vocab_size, config.n_embd) # learned
        # self.wte = lambda x: torch.stack([F.one_hot(x, config.n_embd) for x in idx]).float() # one hot

        self.drop = nn.Dropout(config.embd_pdrop)

        # self.layers = nn.ModuleList([Attention(config) for layer in range(config.n_layer)])
        layer_one = Attention(config)
        temp = config.n_head
        config.n_head = 1

        self.layers = nn.ModuleList([layer_one] + [Attention(config) for layer in range(config.n_layer - 1)])
        config.n_head = temp
        self.ln_f = nn.LayerNorm(config.n_embd)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.01) #std=0.02/math.sqrt(2 * config.n_layer)

        # report number of parameters (note we don't count the decock_size + 1, config.n_emoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.parameters())
        print(f"number of parameters: {n_params}")
        # self.layers[0].Q = torch.nn.Identity()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):

            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        # optimizer = torch.optim.SGD(optim_groups, lr=train_config.learning_rate)

        return optimizer

    def forward(self, idx, targets=None):
        b, t = idx.size()

        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"

        tok_emb = self.wte(idx) # token embeddings of shape (b, t, n_embd)

        x = self.drop(tok_emb)
        
        for layer in self.layers:
            x = layer(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss
    
    @torch.no_grad()
    def visualize_attention(self, idx):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)
        attentions = []
        attentions_softmax = []

        tok_emb = self.wte(idx) # token embeddings of shape (b, t, n_embd)

        x = self.drop(tok_emb)
        B, T, C = x.size()

        for layer in self.layers:
            ln_x = layer.ln(x)

            # calculate query, key, values for all heads in batch and move head forward to be the batch dim
            k = layer.K(ln_x).view(B, T, layer.n_head, C // layer.n_head).transpose(1, 2) # (B, nh, T, hs)
            q = layer.Q(ln_x).view(B, T, layer.n_head, C // layer.n_head).transpose(1, 2) # (B, nh, T, hs)
            v = layer.V(ln_x).view(B, T, layer.n_head, C // layer.n_head).transpose(1, 2) # (B, nh, T, hs)

            rel_pos = layer.embd_dropout(layer.wpe(layer.pos[:T,:T]))
            rel_pos = rel_pos.view(T, T, layer.n_head, C // layer.n_head) # (T, T, nh, hs)

            # causal layer-attention; layer-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
            attn = (q @ k.transpose(-2, -1) + torch.einsum("Tthe,BhTe->BhTt", rel_pos, k)) * (1.0 / math.sqrt(k.size(-1)))
            # attn = (torch.einsum("BhTe, Bhte->BhTt", q, k) + torch.einsum("Tthe,BhTe->BhTt", rel_pos, k)) * (1.0 / math.sqrt(k.size(-1)))
            attentions.append(attn)
            attn = attn.masked_fill(layer.bias[:,:,:T,:T] == 0, float('-inf'))
            attn = F.softmax(attn, dim=-1)
            attn = layer.attn_dropout(attn)
            attentions_softmax.append(attn)

            y = attn @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

            y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

            # output projection
            x = layer.resid_dropout(layer.c_proj(y)) + x
            if not layer.attention_only:
                x = layer.mlpf(layer.ln_2(x)) + x

        return attentions, attentions_softmax