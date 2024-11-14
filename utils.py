import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import math

def stationary_distribution(P):
    if len(P.shape) == 2:
      pi = torch.ones((1, P.size(1)), device=P.device, dtype=P.dtype) / P.size(0)
    elif len(P.shape) == 3:
      pi = torch.ones((P.size(0), 1, P.size(1)), device=P.device, dtype=P.dtype) / P.size(0)
    else:
      raise ValueError(f"P has shape {P.size()}, but P must be 2D or 3D tensor")
    if P.size(-1) < 5:
      P_next = torch.linalg.matrix_power(P,16)
      # while not torch.allclose(P, P_next):
      #   P = P_next
      #   P_next = torch.linalg.matrix_power(P,2)
      # print(P_next[0])
      return P_next.mean(axis=-1)

    pi_next = torch.matmul(pi, P)
    i = 0
    while not torch.allclose(pi_next, pi, atol = 1e-4, rtol = 1e-4):
        i += 1
        pi = pi_next
        pi_next = torch.matmul(pi, P)
        if i >= 100:
          print("OH NO")
          for a in range(len(pi_next)):
            if not torch.allclose(pi_next[a], pi[a], atol = 1e-4, rtol = 1e-4):
              print(P[a])
              print(pi[a]-pi_next[a])
              exit()
          print("OH NO NO")
          exit()
    pi = torch.matmul(pi_next, P).squeeze()
    return pi / pi.sum(axis=-1, keepdim=True)

def mean_alg(inp, num_symbols):
  return (torch.bincount(inp, minlength = num_symbols)+1)/(len(inp)+num_symbols)

def uniform(inp, num_symbols):
  return torch.ones(num_symbols)/num_symbols
from collections import Counter

def statistical(inp, num_symbols):
  inp = inp.tolist()
  bigrams = Counter(zip(inp[:-1],inp[1:]))
  candidate = [1+bigrams[(inp[-1], i)] for i in range(num_symbols)]
  candidate = torch.tensor(candidate, dtype=torch.float)
  candidate = F.normalize(candidate, p=1, dim=0)
  return candidate

def ngram(inp, num_symbols, n):
  inp = inp.tolist()
  if len(inp) < n:
    return torch.ones(num_symbols, dtype=torch.float)
  ngrams = zip(*[inp[i:] for i in range(1+n)])
  candidate = torch.ones(num_symbols, dtype=torch.float)
  check = tuple(inp[-n:])

  for i in ngrams:
    if i[:-1] == check or n == 0:
      candidate[i[-1]]+=1
  candidate = F.normalize(candidate, p=1, dim=0)
  return candidate

def conditional_mutual_information(X, Y, Z):
  total = 0
  num_symbols = X.shape[-1]
  if num_symbols == 2:
    num_bins = 2
    bins = torch.histc((X[:,0] + ((num_bins-1) * Y[:,0]).round() + num_bins*((num_bins-1) * Z[:,0]).round()).cpu(), num_bins**3, 0,num_bins**2).reshape((num_bins,num_bins, num_bins))

    
    for k in range(num_bins):
      pz = bins[:,:,k].sum()/len(X)
      for i in range(num_bins):
        jointxz = bins[i,:,k].sum()/len(X)
        for j in range(num_bins):
          jointxyz = bins[i,j,k].sum()/len(X)
          jointyz = bins[:,i,j].sum()/len(X)
          total += jointxyz * torch.log(pz * jointxyz / (jointxz * jointyz))


  else:
    for k in range(num_symbols):
      pz = torch.mean(Z[:,k])
      for i in range(num_symbols):
        jointxz = torch.mean(X[:,i] * Z[:,k])
        for j in range(num_symbols):
          jointxyz = torch.mean(X[:,i] * Y[:,j] * Z[:,k])
          jointyz = torch.mean(Y[:,j] * Z[:,k])
          total += jointxyz * torch.log(pz * jointxyz / (jointxz * jointyz))
  # print(total)
  return total

def entropy(X):
  total = 0
  num_symbols = X.shape[-1]
  for i in range(num_symbols):
    temp = torch.mean(X[:,i])
    total -= temp*torch.log(temp)
  return total

def mutual_information(X, Y):
  return F.kl_div(X,Y, log_target=False)
  total = 0
  num_symbols = X.shape[-1]
  if num_symbols == 2:
    num_bins = 2
    bins = torch.histc((X[:,0] + ((num_bins-1) * Y[:,0]).round()).cpu(), num_bins**2, 0,num_bins).reshape((num_bins,num_bins))

    
    for i in range(num_bins):
      px = bins[i].sum()/len(X)
      for j in range(num_bins):
        py = bins[:,j].sum()/len(X)
        jointxy = bins[i,j].sum()/len(X)
        total += jointxy * torch.log(jointxy  / (px * py))


  else:
    for i in range(num_symbols):
      px = torch.mean(X[:,i])
      for j in range(num_symbols):
        py = torch.mean(Y[:,j])
        jointxy = torch.mean(X[:,i] * Y[:,j])
        total += jointxy * torch.log(jointxy  / (px * py))
  # print(total)
  return total

def performance_correlation(F, L, Y):
    return mutual_information(F, L)
    #return mutual_information(F, Y) - conditional_mutual_information(F, Y, L)


#this function is really messy/should be refactored
@torch.no_grad()
def compare_test(models, dataset, num_symbols, device):
    batch_size = 64
    for model in models:
      model.eval()
    loader = DataLoader(dataset, batch_size, num_workers=8, drop_last=False)
    num_samples = len(dataset)
    num_batches = math.ceil(len(dataset)/batch_size)

    #set up a bunch of tensors for various outputs
    model_true_loss = torch.zeros((len(models), dataset.length), device=device)
    statistic_alg_loss = torch.zeros(dataset.length, device = device)
    mean_alg_loss = torch.zeros(dataset.length, device = device)
    uniform_alg_loss = torch.zeros(dataset.length, device = device)
    stationary_distribution_loss = torch.zeros(dataset.length, device = device)
    model_statistic_similarity = torch.zeros((len(models), dataset.length), device=device)
    model_mean_similarity = torch.zeros((len(models), dataset.length), device=device)
    model_uniform_similarity = torch.zeros((len(models), dataset.length), device=device)
    for b, (x,(probs, _)) in enumerate(loader):
      x = x.to(device)
      # temp = x # for second layer only experiment
      # x = x[:,:,0] #undo  #for second layer only experiment
      ground_truth = torch.stack([torch.stack([probs[i,x[i,j]] for j in range(len(x[i]))]) for i in range(len(x))]).permute(0,2,1).to(device)
      stat_guess = torch.stack([torch.stack([statistical(x[i,:j+1], num_symbols) for j in range(len(x[i]))]) for i in range(len(x))]).permute(0,2,1).to(device)
      statistic_alg_loss += F.cross_entropy(torch.log(stat_guess), ground_truth, reduction="none").mean(dim=0)
      mean_guess = torch.stack([torch.stack([mean_alg(x[i,:j+1], num_symbols) for j in range(len(x[i]))]) for i in range(len(x))]).permute(0,2,1)
      mean_alg_loss += F.cross_entropy(torch.log(mean_guess), ground_truth, reduction="none").mean(dim=0)
      uniform_guess = (torch.ones_like(ground_truth) / num_symbols).to(device)
      uniform_alg_loss += F.cross_entropy(torch.log(uniform_guess), ground_truth, reduction="none").mean(dim=0)
      
      stationary_distribution_guess = torch.stack([torch.stack([torch.tensor(stationary_distribution(prob))]* len(x[0])) for prob in probs]).to(device).permute(0,2,1)
      # print(stationary_distribution_guess.shape, uniform_guess.shape)
      stationary_distribution_loss += F.cross_entropy(torch.log(stationary_distribution_guess), ground_truth, reduction="none").mean(dim=0)

      for model_id in range(len(models)):
        model = models[model_id]
        logits, loss = model(x)  #for second layer only experiment
        logits = logits.permute(0,2,1)

        #print(ground_truth.shape, logits.shape)
        model_true_loss[model_id] += F.cross_entropy(logits, ground_truth, reduction="none").mean(dim=0)
        model_statistic_similarity[model_id] += F.cross_entropy(logits, stat_guess, reduction="none").mean(dim=0)
        model_mean_similarity[model_id] += F.cross_entropy(logits, mean_guess, reduction="none").mean(dim=0)
        model_uniform_similarity[model_id] += F.cross_entropy(logits, uniform_guess, reduction="none").mean(dim=0)
    return model_true_loss.cpu(), statistic_alg_loss.cpu(), mean_alg_loss.cpu(), uniform_alg_loss.cpu(), model_statistic_similarity.cpu(), model_mean_similarity.cpu(), model_uniform_similarity.cpu(), stationary_distribution_loss.cpu(), []

#this function is really messy/should be refactored
@torch.no_grad()
def special_compare_test(models, dataset, num_symbols, device):
    batch_size = 64
    for model in models:
      model.eval()
    loader = DataLoader(dataset, batch_size, num_workers=8, drop_last=False)
    num_samples = len(dataset)
    num_batches = math.ceil(len(dataset)/batch_size)

    #set up a bunch of tensors for various outputs
    model_true_loss = torch.zeros((len(models), dataset.length), device=device)
    model_losses = torch.zeros((len(models), dataset.length), device=device), torch.zeros((len(models), dataset.length), device=device)
    statistic_alg_loss = torch.zeros(dataset.length, device = device)
    mean_alg_loss = torch.zeros(dataset.length, device = device)
    uniform_alg_loss = torch.zeros(dataset.length, device = device)
    stationary_distribution_loss = torch.zeros(dataset.length, device = device)
    for b, (x,(probs, _)) in enumerate(loader):
      x = x.to(device)
      # temp = x # for second layer only experiment
      # x = x[:,:,0] #undo  #for second layer only experiment
      ground_truth = torch.stack([torch.stack([probs[i,x[i,j]] for j in range(len(x[i]))]) for i in range(len(x))]).permute(0,2,1).to(device)
      stat_guess = torch.stack([torch.stack([statistical(x[i,:j+1], num_symbols) for j in range(len(x[i]))]) for i in range(len(x))]).permute(0,2,1).to(device)
      statistic_alg_loss += F.cross_entropy(torch.log(stat_guess), ground_truth, reduction="none").mean(dim=0)
      mean_guess = torch.stack([torch.stack([mean_alg(x[i,:j+1], num_symbols) for j in range(len(x[i]))]) for i in range(len(x))]).permute(0,2,1)
      mean_alg_loss += F.cross_entropy(torch.log(mean_guess), ground_truth, reduction="none").mean(dim=0)
      uniform_guess = (torch.ones_like(ground_truth) / num_symbols).to(device)
      uniform_alg_loss += F.cross_entropy(torch.log(uniform_guess), ground_truth, reduction="none").mean(dim=0)
      
      stationary_distribution_guess = torch.stack([torch.stack([torch.tensor(stationary_distribution(prob))]* len(x[0])) for prob in probs]).to(device).permute(0,2,1)
      # print(stationary_distribution_guess.shape, uniform_guess.shape)
      stationary_distribution_loss += F.cross_entropy(torch.log(stationary_distribution_guess), ground_truth, reduction="none").mean(dim=0)
      # temp0[(x==0)[:,None, :].expand(x.shape[0], 2, x.shape[1])] = -1
      for model_id in range(len(models)):
        model = models[model_id]
        logits, loss = model(x)  #for second layer only experiment
        logits = logits.permute(0,2,1)

        #print(ground_truth.shape, logits.shape)
        loss = F.cross_entropy(logits, ground_truth, reduction="none")
        model_true_loss[model_id] += loss.mean(dim=0)
        ind = (x==0)#.roll(1, dims=-1)
        print(model_true_loss[model_id])
        model_losses[0][model_id] += loss[ind].sum(dim=0)
        model_losses[1][model_id] += loss[torch.logical_not(ind)].sum(dim=0)#.mean(dim=0)
    return model_true_loss.cpu(), statistic_alg_loss.cpu(), mean_alg_loss.cpu(), uniform_alg_loss.cpu(), model_losses[0].cpu()/len(dataset), model_losses[1].cpu()/len(dataset)



@torch.no_grad()
def top_k_accuracy(models, dataset, num_symbols, device, k=1):
    batch_size = 64
    for model in models:
      model.eval()
    loader = DataLoader(dataset, batch_size, num_workers=8, drop_last=False)
    num_samples = len(dataset)
    num_batches = math.ceil(len(dataset)/batch_size)
    model_topk_accuracy = torch.zeros((len(models),dataset.length), device=device)
    #set up a bunch of tensors for various outputs
    
    for b, (x,(probs, y)) in enumerate(loader):
      x = x.to(device)

      # ground_truth = torch.stack([torch.stack([probs[i,x[i,j]] for j in range(len(x[i]))]) for i in range(len(x))]).to(device)
      
      y = y[..., None].to(device)
      for model_id in range(len(models)):
        model = models[model_id]
        logits, loss = model(x)  #for second layer only experiment
        logits = logits

        # print(torch.topk(logits, k)[1].shape, logits.shape)
        # y = torch.topk(ground_truth,1)[1]
        topk = torch.topk(logits, k)[1]
        # print(torch.eq(y[:, ...], topk).any(dim=-1).shape)
        model_topk_accuracy[model_id] += torch.eq(y, topk).any(dim=-1).float().sum(dim=0)/num_samples
    return model_topk_accuracy.cpu()


#this function is really messy/should be refactored
@torch.no_grad()
def ngram_test(models, dataset, num_symbols, device):
    batch_size = 64
    for model in models:
      model.eval()
    loader = DataLoader(dataset, batch_size, num_workers=8, drop_last=False)
    num_samples = len(dataset)
    num_batches = math.ceil(len(dataset)/batch_size)
    n = dataset.n


    #set up a bunch of tensors for various outputs
    model_true_loss = torch.zeros((len(models), dataset.length), device=device)
    statistic_alg_loss = torch.zeros(dataset.length, device = device)
    mean_alg_loss = torch.zeros(dataset.length, device = device)
    uniform_alg_loss = torch.zeros(dataset.length, device = device)
    # stationary_distribution_loss = torch.zeros(dataset.length, device = device)
    ngram_stat_losses = torch.zeros((n-1, dataset.length), device = device)
    model_statistic_similarity = torch.zeros((len(models), dataset.length), device=device)
    model_mean_similarity = torch.zeros((len(models), dataset.length), device=device)
    model_uniform_similarity = torch.zeros((len(models), dataset.length), device=device)
    for b, (x,(probs, _)) in enumerate(loader):
      x = x.to(device)
      # temp = x # for second layer only experiment
      # x = x[:,:,0] #undo  #for second layer only experiment
      ground_truth = torch.stack([torch.stack([torch.zeros(num_symbols)]*(n-1) + [probs[i,dataset.multi_symbol_convert(x[i,j-n:j].cpu())] for j in range(n, len(x[i])+1)]) for i in range(len(x))]).permute(0,2,1).to(device)
      stat_guess = torch.stack([torch.stack([statistical(x[i,:j+1], num_symbols) for j in range(len(x[i]))]) for i in range(len(x))]).permute(0,2,1).to(device)
      statistic_alg_loss += F.cross_entropy(torch.log(stat_guess), ground_truth, reduction="none").mean(dim=0)
      for m in range(2, n+1):
        ngram_guess = torch.stack([torch.stack([ngram(x[i,:j+1], num_symbols, m) for j in range(len(x[i]))]) for i in range(len(x))]).permute(0,2,1).to(device)
        ngram_stat_losses[m-2] += F.cross_entropy(torch.log(ngram_guess), ground_truth, reduction="none").mean(dim=0)
      mean_guess = torch.stack([torch.stack([mean_alg(x[i,:j+1], num_symbols) for j in range(len(x[i]))]) for i in range(len(x))]).permute(0,2,1)
      mean_alg_loss += F.cross_entropy(torch.log(mean_guess), ground_truth, reduction="none").mean(dim=0)
      uniform_guess = (torch.ones_like(ground_truth) / num_symbols).to(device)
      uniform_alg_loss += F.cross_entropy(torch.log(uniform_guess), ground_truth, reduction="none").mean(dim=0)
      
      # stationary_distribution_guess = torch.stack([torch.stack([torch.tensor(stationary_distribution(prob))]* len(x[0])) for prob in probs]).to(device).permute(0,2,1)
      # print(stationary_distribution_guess.shape, uniform_guess.shape)
      # stationary_distribution_loss += F.cross_entropy(torch.log(stationary_distribution_guess), ground_truth, reduction="none").mean(dim=0)

      for model_id in range(len(models)):
        model = models[model_id]
        logits, loss = model(x)  #for second layer only experiment
        logits = logits.permute(0,2,1)

        #print(ground_truth.shape, logits.shape)
        model_true_loss[model_id] += F.cross_entropy(logits, ground_truth, reduction="none").mean(dim=0)
        model_statistic_similarity[model_id] += F.cross_entropy(logits, stat_guess, reduction="none").mean(dim=0)
        model_mean_similarity[model_id] += F.cross_entropy(logits, mean_guess, reduction="none").mean(dim=0)
        model_uniform_similarity[model_id] += F.cross_entropy(logits, uniform_guess, reduction="none").mean(dim=0)
    return model_true_loss.cpu(), statistic_alg_loss.cpu(), mean_alg_loss.cpu(), uniform_alg_loss.cpu(), model_statistic_similarity.cpu(), model_mean_similarity.cpu(), model_uniform_similarity.cpu(), [], ngram_stat_losses.cpu()

@torch.no_grad()
def ngrams_context(models, dataset, device):
    batch_size = 64*16
    loader = DataLoader(dataset, batch_size, num_workers=8, drop_last=False)
    num_samples = len(dataset)
    num_symbols = dataset.num_symbols
    num_batches = math.ceil(len(dataset)/batch_size)
    n = dataset.n
    ngram_stat_losses = torch.zeros(n+2, device = device)
    model_true_loss = torch.zeros(len(models), device=device)
    ngram_similarity = torch.zeros(n+2, len(models), device = device)

    mutual_inf_model= torch.zeros(len(models), device = device)
    mutual_inf_ngrams = torch.zeros(n+2, device = device)
    for model in models:
        model.eval()
    for b, (x,(probs, _)) in enumerate(loader):
      x = x.to(device)
      
      ground_truth = torch.stack([probs[i,dataset.multi_symbol_convert(x[i,-n:].cpu())] for i in range(len(x))]).to(device)
      ngram_guesses = torch.zeros((n+2, len(x), num_symbols), device = device)
      for m in range(-1, n+1):
        ngram_guesses[m+1] = torch.stack([ngram(x[i], num_symbols, m) for i in range(len(x))]).to(device)
        ngram_stat_losses[m+1] += F.cross_entropy(torch.log(ngram_guesses[m+1]), ground_truth, reduction="none").mean(dim=0)
        mutual_inf_ngrams[m+1] += mutual_information(ngram_guesses[m+1], ground_truth)
      for model_id in range(len(models)):
        model = models[model_id]
        logits, loss = model(x)
        model_true_loss[model_id] += F.cross_entropy(logits[:,-1], ground_truth, reduction="none").mean(dim=0)
        probs = F.softmax(logits[:,-1])
        # probs = logits[:,-1]

        # print(probs.shape, ngram_guesses[0].shape, ground_truth.shape)
        mutual_inf_model[model_id] += mutual_information(torch.log(ground_truth), probs)
        for m in range(-1, n+1):
          # ngram_similarity[m+1, model_id] += performance_correlation(probs, ngram_guesses[m+1], ground_truth)
          
          ngram_similarity[m+1, model_id] += mutual_information(torch.log(ngram_guesses[m+1]), probs)
      # print(mutual_information(ground_truth, ground_truth))
      # print(entropy(ground_truth))
      # print(ngram_guesses[-1])
          
        
      
      # stat_guess = torch.stack([statistical(x[i], num_symbols) for i in range(len(x))]).to(device)
      # ngram_stat_losses[n+2] += F.cross_entropy(torch.log(stat_guess), ground_truth, reduction="none").mean(dim=0)
    return model_true_loss.cpu(), ngram_stat_losses.cpu(), ngram_similarity.cpu(), mutual_inf_model.cpu(), mutual_inf_ngrams.cpu(), performance_correlation(ngram_guesses[2], ngram_guesses[1], ground_truth).cpu()