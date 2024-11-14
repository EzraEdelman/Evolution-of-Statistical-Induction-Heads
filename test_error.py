import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import math
import datasets

@torch.no_grad
def nested_list_to_tensor(x, out_shape, out, top_level=True):
    if isinstance(x[0], list):
        for i in range(out_shape[0]):
            nested_list_to_tensor(x[i], out_shape[1:], out, top_level=False)
    else:
        out.extend(x)
    if top_level:
        return torch.stack(out).reshape(*out_shape)


@torch.no_grad
def list_list_tensor_to_tensor(x):
    out = []
    for l in x:
        for elem in l:
            out.extend(elem)


@torch.no_grad()
def stationary_distribution(prob):
    evals, evecs = np.linalg.eig(prob.T)
    evec1 = evecs[:,np.isclose(evals, 1)]

    #Since np.isclose will return an array, we've indexed with an array
    #so we still have our 2nd axis.  Get rid of it, since it's only size 1.
    evec1 = evec1[:,0]

    stationary = evec1 / evec1.sum()

    #eigs finds complex eigenvalues and eigenvectors, so you'll want the real part.
    return stationary.real


""" next token prediction using n-grams with Rule of succession
n=0: uniform probabilities
"""
@torch.no_grad()
def ngram(inp, num_symbols, n):    
    ngrams = zip(*[inp[i:] for i in range(n)])
    candidate = torch.ones(num_symbols, dtype=torch.float)
    check = tuple(inp[-n+1:])
    for i in ngrams:
        if i[:-1] == check or n == 1:
            candidate[i[-1]]+=1
    candidate = F.normalize(candidate, p=1, dim=0)
    return candidate

""" ngrams for ngrams_simple dataset
"""
@torch.no_grad()
def simple_ngram(inp, num_symbols, n, dataset_n):
    candidate = torch.ones(num_symbols, dtype=torch.float)
    if n == 0:
        return F.normalize(candidate, p=1, dim=0)
    
    if n == 1:
        for i in inp[dataset_n - 1::dataset_n]:
            candidate[i] += 1
    else:
        ngrams = zip(*[inp[i::dataset_n] for i in range(dataset_n)])
        check = tuple(inp[-n+1:])
        for i in ngrams:
            if i[-n:-1] == check or n == 1:
                candidate[i[-1]]+=1
    candidate = F.normalize(candidate, p=1, dim=0)
    return candidate

@torch.no_grad()
def test(models, dataset, device):
    batch_size = 64
    num_symbols = dataset.num_symbols
    for model in models:
        model.eval()
    loader = DataLoader(dataset, batch_size, num_workers=8, drop_last=False)
    num_samples = len(dataset)
    num_batches = math.ceil(len(dataset)/batch_size)
    n = dataset.n+1

    #set up a bunch of tensors for various outputs
    model_true_loss = torch.zeros((len(models), dataset.length), device=device)
    ngram_losses = torch.zeros((n+1, dataset.length), device = device)
    KL_divs = torch.zeros((n+1, len(models), dataset.length), device = device)
    
    for b, (x,(probs, _)) in enumerate(loader):
        # print(x.shape)
        # ground_truth = torch.stack([torch.stack([probs[i,dataset.multi_symbol_convert(x[i,j-n:j].cpu())] for i in range(x.shape[0])]) for j in range(n, x.shape[1]+1)]).to(device)
        # ground_truth = torch.stack([torch.stack([probs[i,dataset.multi_symbol_convert(x[i,j-n+1:j])] for j in range(n-1, x.shape[1]+1)]) for i in range(x.shape[0])]).transpose(1,2).to(device)
        ground_truth = torch.zeros((x.shape[0], num_symbols, x.shape[1]), device=device)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                ground_truth[i, :, j] = probs[i,dataset.multi_symbol_convert(x[i,j-n+1:j])]
        # ground_truth = torch.stack([torch.stack([probs[i,x[i,j]] for i in range(x.shape[0])]) for j in range(x.shape[1])])
        # ngram_guesses = torch.stack([torch.stack([ngram(x[i,:j+1], num_symbols, m) for i in range(x.shape[0])]) for j in range(x.shape[1])])
        
        # create guess by m-grams
        # mgram_guesses = torch.zeros(n+1, x.shape[0], num_symbols, x.shape[1],  device = device)
        # for m in range(n+1):
        #     for i in range(x.shape[0]):
        #         for j in range(x.shape[1]):
        #             mgram_guesses[m, i, :, j] = ngram(x[i,:j+1], num_symbols, m)
            # loss of m-grams
        # for m in range(n+1):
        #     temp =  F.cross_entropy(torch.log(mgram_guesses[m]), ground_truth, reduction="none").sum(dim=0) / len(dataset)
        #     ngram_losses[m] += temp
        for model_id in range(len(models)):
            x = x.to(device)
            model = models[model_id]
            logits, loss = model(x)
            logits = logits.permute(0,2,1)
            # print(F.softmax(logits, dim=1).shape, ground_truth.shape)
            #loss of model
            # model_true_loss[model_id] += F.cross_entropy(logits, ground_truth, reduction="none").sum(dim=0)/ len(dataset)
            model_true_loss[model_id] += F.kl_div(F.log_softmax(logits, dim=1), ground_truth, reduction="batchmean")#.sum(dim=(0,1))/ len(dataset)
            # for m in range(n+1):
            #     # print(torch.log(mgram_guesses[m]).shape, F.softmax(logits, dim=1).shape)
            #     KL_divs[m,model_id] += F.kl_div(torch.log(mgram_guesses[m]), F.softmax(logits, dim=1), reduction="none").sum(dim=(0,1))/len(dataset) #CHECK DIM

    return model_true_loss.cpu(), ngram_losses.cpu(), KL_divs.cpu()


@torch.no_grad()
def test_last_token(models, dataset, device, size = 1000):
    batch_size = 64*4
    num_symbols = dataset.num_symbols
    for model in models:
        model.eval()
    loader = DataLoader(dataset, batch_size, num_workers=8, drop_last=False)
    num_samples = len(dataset)
    size = min(size, num_samples)
    num_batches = size//batch_size
    n = dataset.n+1
    size = num_batches * batch_size

    #set up a bunch of tensors for various outputs
    model_true_loss = torch.zeros((len(models)), device=device)
    ngram_losses = torch.zeros((n+1), device = device)
    KL_divs = torch.zeros((n+1, len(models)), device = device)
    for b, (x,(probs, _)) in enumerate(loader):
        if b >= num_batches:
            break
        converted_symbols = dataset.multi_symbol_convert(x[:,1-n:])
        ground_truth = probs.view(batch_size*num_symbols**(n-1), num_symbols)[torch.arange(batch_size)*num_symbols**(n-1)+converted_symbols].to(device)

        # create guess by m-grams
        mgram_guesses = torch.zeros(n+1, x.shape[0], num_symbols,  device = device)
        for m in range(n+1):
            for i in range(x.shape[0]):
                mgram_guesses[m, i] = ngram(x[i], num_symbols, m)
            # loss of m-grams
        for m in range(n+1):
            # temp =  F.kl_div(torch.log(mgram_guesses[m]),ground_truth,reduction="none").sum(dim=(0,1)) / size
            ngram_losses[m] +=  F.kl_div(torch.log(mgram_guesses[m]),ground_truth,reduction="none").sum(dim=(0,1)) / size
            # temp =  F.cross_entropy(mgram_guesses[m], ground_truth, reduction="none").sum(dim=0) / size
        x = x.to(device)
        
        for model_id in range(len(models)):

            model = models[model_id]
            logits, _ = model(x)
            logits = logits[:, -1]
            # logits = logits.permute(0,2,1)
            #loss of model
            # model_true_loss[model_id] += F.cross_entropy(logits, ground_truth, reduction="none").sum(dim=0)/ size

            model_true_loss[model_id] += F.kl_div(F.log_softmax(logits, dim=1), ground_truth, reduction="none").sum(dim=(0,1))/ size
            # model_true_loss[model_id] += F.kl_div(torch.log(.0001+logits), ground_truth, reduction="none").sum(dim=(0,1))/ size
            for m in range(n+1):
                # print(torch.log(mgram_guesses[m]).shape, F.softmax(logits, dim=1).shape)
                KL_divs[m,model_id] += F.kl_div(F.log_softmax(logits, dim=1), mgram_guesses[m], reduction="none").sum(dim=(0,1))/size #CHECK DIM
                # KL_divs[m,model_id] += F.kl_div(torch.log(.0001+logits), mgram_guesses[m], reduction="none").sum(dim=(0,1))/size #CHECK DIM
    return model_true_loss.cpu(), ngram_losses.cpu(), KL_divs.cpu()

# for ngrams_simple
@torch.no_grad()
def simple_test(models, dataset, device):
    assert type(dataset) is datasets.ngrams_simple
    batch_size = 64
    num_symbols = dataset.num_symbols
    for model in models:
        model.eval()
    loader = DataLoader(dataset, batch_size, num_workers=8, drop_last=False)
    num_samples = len(dataset)
    num_batches = math.ceil(len(dataset)/batch_size)
    n = dataset.n+1
    indices = dataset.indices[len(dataset.indices)-1:]
    print(indices)
    #set up a bunch of tensors for various outputs
    model_true_loss = torch.zeros((len(models)), device=device)
    ngram_losses = torch.zeros((n+1), device = device)
    KL_divs = torch.zeros((n+1, len(models)), device = device)
    
    for b, (x,(probs, y)) in enumerate(loader):
        # print(x.shape)
        # ground_truth = torch.stack([torch.stack([probs[i,dataset.multi_symbol_convert(x[i,j-n:j].cpu())] for i in range(x.shape[0])]) for j in range(n, x.shape[1]+1)]).to(device)
        # ground_truth = torch.stack([torch.stack([probs[i,dataset.multi_symbol_convert(x[i,j-n+1:j])] for j in range(n-1, x.shape[1]+1)]) for i in range(x.shape[0])]).transpose(1,2).to(device)
        ground_truth = torch.zeros(x.shape[0], num_symbols, x.shape[1])
        for i in range(x.shape[0]):
            for j in indices:
                ground_truth[i, :, j] = probs[i, dataset.multi_symbol_convert(x[i,j-n+1+1:j+1])]
        ground_truth = ground_truth.to(device)
        # ground_truth = torch.stack([torch.stack([probs[i,x[i,j]] for i in range(x.shape[0])]) for j in range(x.shape[1])])
        # ngram_guesses = torch.stack([torch.stack([ngram(x[i,:j+1], num_symbols, m) for i in range(x.shape[0])]) for j in range(x.shape[1])])
        
        # create guess by m-grams
        mgram_guesses = torch.zeros(n+1, x.shape[0], num_symbols, x.shape[1],  device = device)
        for m in range(n+1):
            for i in range(x.shape[0]):
                for j in indices:
                    mgram_guesses[m, i, :, j] = simple_ngram(x[i,:j+1], num_symbols, m, n)
                    # mgram_guesses[m, i, :, j] = simple_ngram(x[i,], num_symbols, m, n)
            # loss of m-grams
        for m in range(n+1):
            temp =  F.kl_div(torch.log(mgram_guesses[m,:,:,indices]), ground_truth[:,:,indices], reduction="sum") / len(dataset) / len(indices)
            ngram_losses[m] += temp
        for model_id in range(len(models)):
            x = x.to(device)
            model = models[model_id]
            logits, loss = model(x)
            logits = logits.permute(0,2,1)
            # print(F.softmax(logits, dim=1).shape, ground_truth.shape)
            #loss of model
            model_true_loss[model_id] += F.kl_div(F.log_softmax(logits[:,:,indices], dim=1), ground_truth[:,:,indices], reduction="sum")/ len(dataset) / len(indices)
            for m in range(n+1):
                # print(torch.log(mgram_guesses[m]).shape, F.softmax(logits, dim=1).shape)
                KL_divs[m,model_id] += F.kl_div(F.log_softmax(logits[:,:,indices], dim=1), F.softmax(mgram_guesses[m,:,:,indices], dim=1), reduction="sum")/len(dataset)  / len(indices)#CHECK DIM

    return model_true_loss.cpu(), ngram_losses.cpu(), KL_divs.cpu()
@torch.no_grad()
def pos_enc(model_history, config):
    t = config.block_size
    num_tokens = config.vocab_size
    device = config.device

    # KP_time = [[[] for i in range(t+1)] for j in range(num_tokens)]
    m = model_history[-1]
    if config.n <= 2:
        if hasattr(m, "layers"):
            KP_time = torch.zeros(num_tokens, t, len(model_history))
        elif hasattr(m, "v"):
            KP_time = torch.zeros(num_tokens, len(m.v.weight), len(model_history))
        for i in range(len(model_history)):
            m = model_history[i]
            if hasattr(m, "layers"):
                rel = m.layers[0].wpe.weight.cpu()
                embds = m.layers[0].ln(m.wte(torch.Tensor(list(range(num_tokens))).long().to(device)).float())
                KP_time[:,:,i] = (m.layers[0].K(embds).cpu() @ rel.T)[:,:-1]

                # KP_time[:,:,i] = F.softmax((m.layers[0].K(embds).cpu() @ rel.T), dim = -1)[:,:-1]
            elif hasattr(m, "v"):
                # KP_time[:,:,i] = m.v.weight.squeeze().expand(num_tokens, -1)
                KP_time[:,:,i] = F.softmax(m.v.weight.squeeze().expand(num_tokens, -1), dim=1)
            # KP_time[0,:,i] = F.softmax((m.layers[0].K(embds).cpu() @ rel.T)[0,:-1], dim = 0)
            # KP_time[1,:,i] = F.softmax((m.layers[0].K(embds).cpu() @ rel.T)[1,:-1], dim = 0)
            # KP_time[:,:,i] = (m.layers[0].K(embds).cpu() @ rel.T)[:,:-1]
            # for j in range(num_tokens):
                # KP = (m.layers[0].K(embds).cpu() @ rel.T)[j]
                # KP = F.softmax(KP)
                # KP_time[j,:] = KP

                # for j in range(len(KP_time[i])):
                #     KP_time[i][j].append(KP[j])
    else:
        KP_time = torch.zeros(num_tokens, t, len(model_history), config.n_head)
        for i in range(len(model_history)):
            m = model_history[i]
            # T x n_embd
            rel = m.layers[0].wpe.weight.cpu().reshape(t+1,config.n_head,config.n_embd//config.n_head)
            # T x n_embd
            embds = m.layers[0].ln(m.wte(torch.Tensor(list(range(num_tokens))).long().to(device)).float())
            K = m.layers[0].K(embds).cpu().reshape(config.vocab_size, config.n_head, config.n_embd//config.n_head)
            # KP_time[:,:,i] = (m.layers[0].K(embds).cpu() @ rel.T)[:,:-1]
            for j in range(config.n_head):

                KP_time[:,:,i,j] = F.softmax(K[:,j] @ rel[:,j].T, dim = -1)[:,:-1]
    return KP_time