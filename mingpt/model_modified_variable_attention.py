"""
Full definition of a GPT Language Model, all of it in this single file.

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
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

def association_attention(x, idx):
    return (idx.view(B, self.n_head, T).roll(1, dims=-1)==idx.view(B, T, self.n_head)).view(B, self.n_head, T, T).float()

def first_layer_attention(x, idx):
    # attention below the diagonal
    att = torch.diag(torch.ones(T-1, device = x.device), -1)
    att[0,0] = 1
    return att

class FixedAttention(nn.Module):
    """
    A chocolate multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config, attention):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, config.n_embd)
        # self.c_attn = torch.nn.Identity()
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # self.c_proj = torch.nn.Identity()
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.attention = attention

    def forward(self, x, idx, _):

        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        v  = self.c_attn(x)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        
        att = self.attention(x, idx)
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = att.masked_fill(att == 0, -3)
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        # self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias = False)

        self.Q = nn.Linear(config.n_embd, config.n_embd, bias = False)
        self.K = nn.Linear(config.n_embd, config.n_embd, bias = False)
        self.V = nn.Linear(config.n_embd, config.n_embd, bias = False)
        # # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias = False)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.causal = config.causal
        self.abs_embd = config.abs_embd

    def attention(self, x, rel_encoding):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        # k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = self.K(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.Q(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.V(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.abs_embd:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        else:
            rel_encoding = rel_encoding.reshape(T, T, self.n_head, C // self.n_head)
            att = (q @ k.transpose(-2, -1) + torch.einsum("Tthe,BhTe->BhTt", rel_encoding, k)) * (1.0 / math.sqrt(k.size(-1))) 
            # att = (q @ k.transpose(-2, -1) + torch.einsum("Tte,BhTe->BhTt", rel_encoding, x.view(B, self.n_head, T, C//self.n_head))) * (1.0 / math.sqrt(k.size(-1))) 

        if (self.causal):
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        return att, v, (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

    def forward(self, x, _, rel_encoding):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        att, v, _ = self.attention(x, rel_encoding)
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class Block(nn.Module):
    """ an assuming Transformer block """

    def __init__(self, config, head):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = head(config)
        if head is CausalSelfAttention:
            self.name = "trained"
        else:
            self.name = "fixed"
    
    def __str__(self):
        return self.name
    
    def attention(self, x, rel_encodings = None):
        return self.attn.attention(self.ln_1(x), rel_encodings)

    def forward(self, x, idx, rel_encodings = None):
        x = x + self.attn(self.ln_1(x), idx, rel_encodings)
        return x


class Attention_Only_GPT(nn.Module):
    """ GPT Language Model """
    
    @staticmethod
    def get_default_config():
        C = CN()
        C.model_type = 'gpt'
        C.n_layer = 2
        C.n_head = 2
        C.n_embd = 16
        C.vocab_size = None
        C.block_size = None
        # dropout hyperparameters
        C.embd_pdrop = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1
        C.causal = True
        C.abs_embd = True
        return C

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.block_size = config.block_size
        assert config.model_type in ["gpt", "first layer fixed", "second layer fixed", "fixed"], f"{config.model_type} not a valid model type"
        if config.model_type == "gpt":
            layers = [Block(config, CausalSelfAttention) for _ in range(config.n_layer)]
        elif config.model_type == "first layer fixed":
            layers = [Block(config, CausalSelfAttention), Block(config, AssociationAttention)]
        elif config.model_type == "second layer fixed":
            layers = [Block(config, FixedAttention), Block(config, CausalSelfAttention)]
        elif config.model_type == "fixed":
            layers = [Block(config, FixedAttention), Block(config, AssociationAttention)]
        
        if config.model_type != "gpt":
            warnings.warn("Fixed layer embeddings not implemented right now")
            assert config.n_layer == 2, "{config.model_type} model must have 2 layers"

        #added to make it easier to map attention
        self.model_type = config.model_type
        self.abs_embd = config.abs_embd
        self.n_embd = config.n_embd
        self.num_symbols = config.vocab_size

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), 
            wpe = nn.Embedding(config.block_size, config.n_embd) if config.abs_embd 
            else nn.ModuleList(nn.Embedding(2 * config.block_size - 1, config.n_embd) for layer in layers), 
            # else nn.Embedding(2, config.n_embd),  #for second layer only experiment
            drop = nn.Dropout(config.embd_pdrop),
            h = nn.ModuleList( layers),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
            # if pn.endswith('Q.weight') or pn.endswith('K.weight'):
            #     print(pn)
            #     torch.nn.init.normal_(p, mean=0.0, std=.1)
        

        # report number of parameters (note we don't count the decock_size + 1, config.n_emoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        n_params = sum(p.numel() for p in self.parameters())
        print(f"number of parameters: {n_params}")

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
        device = idx.device
        b, t = idx.size()

        # b, t, _ = idx.size() #for second layer only experiment
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)
        # forward the GPT model itself
        # tok_emb = self.transformer.wte(idx[:, :, 0]) + self.transformer.wte(self.num_symbols+ idx[:, :, 1]) # token embeddings of shape (b, t, n_embd)  #for second layer only experiment
        tok_emb = self.transformer.wte(idx[:, :]) # token embeddings of shape (b, t, n_embd)

        # tok_emb = torch.stack([F.one_hot(x, self.n_embd) for x in idx]).float() # one hot
        # tok_emb = torch.stack([F.one_hot(x[:, 0], self.n_embd)+F.one_hot(self.num_symbols + x[:, 1], self.n_embd) for x in idx]).float()  #for second layer only experiment
        if self.abs_embd:
            pos_emb = self.transformer.wpe(pos).expand((b, t, self.n_embd)) # position embeddings of shape (1, t, n_embd)
            # x = self.transformer.drop(tok_emb )  #for second layer only experiment
            x = self.transformer.drop(tok_emb + pos_emb) 
            aK, aV = None, None
        else:
            # pos = torch.arange(t, dtype=torch.long, device=device)
            pos = pos.view(-1,1) - pos.view(1, -1) + t - 1
            # pos = torch.zeros_like(pos)
            # pos[1:,:] += torch.eye(t-1, t, device = device).long()
            # pos += (2*torch.eye(t, t, device = device)).long()

            # pos = torch.clip(pos - t , min = 0, max = 1)
            x = self.transformer.drop(tok_emb)
            # rel_encoding = self.transformer.drop(self.transformer.wpe(pos)) # undo
        temp = 0 # false
        for block in self.transformer.h:
            if self.abs_embd:
                x = block(x, idx)
            else:
                # rel_encoding = self.transformer.drop(F.pad(self.transformer.wpe[temp](pos),(self.n_embd-self.n_embd//2,0)))

                rel_encoding = self.transformer.drop(self.transformer.wpe[temp](pos))
                temp += 1
                x = block(x, idx, rel_encoding)
                # if temp==1:
                #     x = block(x, idx, rel_encoding)
                #     # temp = False
                # else:
                #     x = block(x, idx, torch.zeros_like(rel_encoding))
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss
    
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
    
    @torch.no_grad()
    def visualize_attention(self, idx):
        device = idx.device
        # b, t = idx.size() #undo
        b, t = idx.size() #undo
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)
        attentions = []
        # forward the GPT model itself
        # tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)

        tok_emb = torch.stack([F.one_hot(x, self.n_embd) for x in idx]).float() #undo

        # tok_emb = torch.stack([F.one_hot(x[:, 0], self.n_embd)+F.one_hot(self.num_symbols + x[:, 1], self.n_embd) for x in idx]).float() #undo
        if self.abs_embd:
            # pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd) #undo
            # x = self.transformer.drop(tok_emb + pos_emb) #undo
            x = self.transformer.drop(tok_emb) #undo
            aK, aV = None, None
            rel_encoding = None
        else:
            # pos = torch.arange(t, dtype=torch.long, device=device)
            pos = pos.view(-1,1) - pos.view(1, -1) + t - 1
            x = self.transformer.drop(tok_emb)
            rel_encoding = self.transformer.drop(self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x, idx, rel_encoding)
            attentions.append(block.attention(x, rel_encoding)[0])
            attentions.append( block.attention(x, rel_encoding)[2])
            

        return attentions