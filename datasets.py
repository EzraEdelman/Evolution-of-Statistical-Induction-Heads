import torch
import numpy as np
from utils import stationary_distribution
from torch.utils.data import Dataset
from random import choices, seed, Random, sample
import pickle
from hashlib import sha3_256

def split(tensor):
  ret = int.from_bytes(sha3_256(pickle.dumps(tensor.flatten().tolist())).digest())
  return ret


class bigrams(Dataset):
    
    """
    Dataset for the in context markov learning. Each example is a series of outputs from a markov chain
    """

    def __init__(self, split, length = 101, num_symbols = 5, device="cuda", last_token_only = False, dataset_length = None):
        assert split in {'train', 'test'}
        self.length = length - 1
        self.num_symbols = num_symbols
        self.split = split
        self.last_token_only = last_token_only
        self.device = device
        if self.dataset_length is not None:
            self.dataset_length = dataset_length
        elif split == 'train':
            self.dataset_length = int(1e16)
        elif split == 'test':
            self.dataset_length = 1000

        self.generator = torch.distributions.dirichlet.Dirichlet(torch.ones((num_symbols,num_symbols), device=self.device))
        # temp = np.random.randint(1000000)
        # self.gen = np.random.default_rng(temp)
        # self.ran = Random(temp)

    def __len__(self):
        return self.dataset_length

    def get_vocab_size(self):
        return self.num_symbols

    def get_block_size(self):
        # the length of the sequence that will feed into transformer
        return self.length

    def conditional_probs(self, idx):
        while True:
            # generate random transition probabilities
            conditional_probs = np.random.dirichlet(np.ones(self.num_symbols), size=(self.num_symbols,))
            # figure out if this generated example is train or test based on transition probabilities hash
            h = split(conditional_probs)
            inp_split = 'test' if h % 4 == 0 else 'train' # designate 25% of examples as test
            if inp_split == self.split:
                break
        
        stationary = torch.tensor(stationary_distribution(conditional_probs))
        return torch.tensor(conditional_probs), stationary

    def __getitem__(self, idx):
        conditional_probs, stationary_distribution = self.conditional_probs(idx)
        thresholds = conditional_probs.cumsum(dim = 1).tolist()

        #generate sequence
        # rand = torch.rand(self.length)

        inp = choices(range(self.num_symbols), weights = stationary_distribution)
        for i in range(self.length):
            inp.extend(choices(range(self.num_symbols), cum_weights = thresholds[inp[-1]]))

        x = torch.tensor(inp[:-1], dtype=torch.long)

        if self.split == 'train':
            y = torch.tensor(inp[1:], dtype=torch.long)
            # y[:len(y)//5] = -1
            if (self.last_token_only):
                y[:-1] = -1
        else:
            y = conditional_probs, torch.tensor(inp[1:], dtype=torch.long)
            # y[1][:len(y)//2] = -1
        return x, y

class markov(Dataset):

    """
    Dataset for the in context markov learning. Each example is a series of outputs from a markov chain
    """

    def __init__(self, split, length = 101, num_symbols = 5, last_token_only = False):
        import torch
        assert split in {'train', 'test'}
        self.length = length - 1
        self.num_symbols = num_symbols
        self.split = split
        self.last_token_only = last_token_only
        # temp = np.random.randint(1000000)
        # self.gen = np.random.default_rng(temp)
        # self.ran = Random(temp)

    def __len__(self):
        if self.split == 'train':
            return int(1e16) # ...
        if self.split == 'test':
            return 1000
            # return 64
            # return int(64*4) # ...

    def get_vocab_size(self):
        return self.num_symbols

    def get_block_size(self):
        # the length of the sequence that will feed into transformer
        return self.length

    def conditional_probs(self, idx):
        while True:
            # generate random transition probabilities
            conditional_probs = np.random.dirichlet(np.ones(self.num_symbols), size=(self.num_symbols,))
            # figure out if this generated example is train or test based on transition probabilities hash
            h = split(conditional_probs)
            inp_split = 'test' if h % 4 == 0 else 'train' # designate 25% of examples as test
            if inp_split == self.split:
                break
        
        stationary = torch.tensor(stationary_distribution(conditional_probs))
        return torch.tensor(conditional_probs), stationary

    def __getitem__(self, idx):
        conditional_probs, stationary_distribution = self.conditional_probs(idx)
        thresholds = conditional_probs.cumsum(dim = 1).tolist()

        #generate sequence
        # rand = torch.rand(self.length)

        inp = choices(range(self.num_symbols), weights = stationary_distribution)
        for i in range(self.length):
            inp.extend(choices(range(self.num_symbols), cum_weights = thresholds[inp[-1]]))

        x = torch.tensor(inp[:-1], dtype=torch.long)

        if self.split == 'train':
            y = torch.tensor(inp[1:], dtype=torch.long)
            # y[:len(y)//5] = -1
            if (self.last_token_only):
                y[:-1] = -1
        else:
            y = conditional_probs, torch.tensor(inp[1:], dtype=torch.long)
            # y[1][:len(y)//2] = -1
        return x, y

class monogram(markov):

    """
    Dataset for the in context monogram learning. Each example is a series of outputs from a markov chain
    """
    def __init__(self, split, length = 101, num_symbols = 5, last_token_only = False):
        super().__init__(split, length, num_symbols, last_token_only)


    def __getitem__(self, idx):
        _, stationary_distribution = self.conditional_probs(idx)
        thresholds = stationary_distribution.cumsum(dim = 0).tolist()
        #generate sequence

        inp = choices(range(self.num_symbols), cum_weights = thresholds, k=self.length)
        

        x = torch.tensor(inp[:-1], dtype=torch.long)

        if self.split == 'train':
            y = torch.tensor(inp[1:], dtype=torch.long)
            # y[:len(y)//4] = -1
            if (self.last_token_only):
                y[:-1] = -1
        else:
            y = stationary_distribution, torch.tensor(inp[1:], dtype=torch.long)
            # y[1][:len(y)//2] = -1

        return x, y

class learn_previous(markov):

    """
    Dataset for learning the previous token
    """

    def __init__(self, split, length = 101, num_symbols = 5, last_token_only = False):
        super().__init__(split, length, num_symbols, last_token_only)


    def __getitem__(self, idx):
        conditional_probs, stationary_distribution = self.conditional_probs(idx)
        thresholds = conditional_probs.cumsum(dim = 1).tolist()

        #generate sequence

        inp = choices(range(self.num_symbols), weights = stationary_distribution)
        for i in range(self.length):
            # print(conditional_probs.shape)
            inp.extend(choices(range(self.num_symbols), cum_weights = thresholds[inp[-1]]))

        x = torch.tensor(inp[:-1], dtype=torch.long)

        if self.split == 'train':
            y = torch.zeros_like(x)
            y[1:] = x[:-1]
            y[0] = -1
            # y[:len(y)//4] = -1
            if (self.last_token_only):
                y[:-1] = -1
        else:
            y = torch.zeros_like(x)
            y[1:] = x[:-1]
            y[0] = -1
            y = conditional_probs, y
            # y[1][:len(y)//2] = -1

        return x, y

class statistics(markov):

    """
    Dataset for learning the previous token
    """

    def __init__(self, split, length = 101, num_symbols = 5, last_token_only = False):
        super().__init__(split, length, num_symbols, last_token_only)


    def __getitem__(self, idx):
        conditional_probs, stationary_distribution = self.conditional_probs(idx)
        thresholds = conditional_probs.cumsum(dim = 1).tolist()

        #generate sequence

        inp = choices(range(self.num_symbols), weights = stationary_distribution)
        for i in range(self.length + 1):
            # print(conditional_probs.shape)
            inp.extend(self.ran.choices(range(self.num_symbols), cum_weights = thresholds[inp[-1]]))
        

        x = torch.tensor(list(zip(inp[1:-1], inp[:-2])), dtype=torch.long)

        if self.split == 'train':
            y = torch.tensor(inp[2:], dtype=torch.long)
            if (self.last_token_only):
                y[:-1] = -1
        else:
            
            y = conditional_probs, torch.tensor(inp[2:], dtype=torch.long)
            # y[1][:len(y)//2] = -1

        return x, y


class ngrams(markov):
    def __init__(self, split, n, length = 101, num_symbols = 2, last_token_only = False, uniform_prior = False):
        super().__init__(split, length, num_symbols, last_token_only)
        self.n = n - 1
        self.conv = np.array([num_symbols ** k for k in range(self.n)])
        self.uniform_prior = uniform_prior

    def conditional_probs(self, idx):
        while True:
            # generate random transition probabilities
            conditional_probs = np.random.dirichlet(np.ones(self.num_symbols), size=(self.num_symbols**(self.n),))
            # figure out if this generated example is train or test based on transition probabilities hash
            h = split(conditional_probs)
            inp_split = 'test' if h % 4 == 0 else 'train' # designate 25% of examples as test
            if inp_split == self.split:
                break
        temp = torch.zeros((self.num_symbols**(self.n),self.num_symbols**(self.n)))
        for i in range(self.num_symbols ** self.n):
            for j in range(self.num_symbols):
                converted = int((i %self.num_symbols**(self.n - 1)) * self.num_symbols + j)
                temp[i, converted] = conditional_probs[i,j]
        # stationary = torch.tensor(stationary_distribution(temp))
        stationary = stationary_distribution(temp)
        return torch.tensor(conditional_probs), stationary
        
    def multi_symbol_convert(self, l):
        # assert len(l) == self.n-1
        return (l * self.conv).sum()
    
    def single_symbol_convert(self, m):
        out = [1] * (self.n )
        for i in range(self.n - 1, -1, -1):
            out[i] = m // self.num_symbols ** i
            m = m % self.num_symbols ** i
        return out

    def __getitem__(self, idx):
        conditional_probs, stationary_distribution = self.conditional_probs(idx)
        thresholds = conditional_probs.cumsum(axis = 1).tolist()

        #generate sequence
        # rand = torch.rand(self.length)
        # print(stationary_distribution)
        if self.uniform_prior:
            inp = choices(range(self.num_symbols), k = self.n)
        else:
            inp = self.single_symbol_convert(choices(range(self.num_symbols**(self.n)), weights = stationary_distribution)[0])
        for i in range(self.length - self.n + 1):
            # print(conditional_probs.shape)
            inp.extend(choices(range(self.num_symbols), cum_weights = thresholds[self.multi_symbol_convert(inp[-self.n:])]))
            #inefficient but simple way of choosing next symbol
            # ind = self.num_symbols-1
            # tot = 0
            # for j in range(self.num_symbols):
            #   if rand[i] < thresholds[inp[-1], j]:
            #     ind = j
            #     break
            # #append next symbol
            # inp.append(ind)
        # inp.extend(choices(range(self.num_symbols), k = 1))
        # inp.extend(choices(range(self.num_symbols), cum_weights = thresholds[self.multi_symbol_convert(inp[-self.n:])]))
        if self.uniform_prior:
            inp[-2] = choices(range(self.num_symbols), k=1)[0]
            inp[-1] = choices(range(self.num_symbols), cum_weights=thresholds[inp[-2]])[0]
        x = torch.tensor(inp[:-1], dtype=torch.long)

        if self.split == 'train':
            y = torch.tensor(inp[1:], dtype=torch.long)
            # y[:len(y)//5] = -1
            if (self.last_token_only):
                y[:-1] = -1
            # y = conditional_probs.float() # UNDO
        else:
            y = conditional_probs, torch.tensor(inp[1:], dtype=torch.long)
            # y[1][:len(y)//2] = -1

        return x, y

# from scipy.stats import truncnorm

def get_truncated_normal(mean=0, sd=1, low=0, upp=1):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd).rvs()

# class spectrum(markov):
#     def __init__(self, split, n, length = 101, num_symbols = 2, last_token_only = False):
#         super().__init__(split, length, num_symbols, last_token_only)
#         assert(num_symbols == 2)
#         assert(n >= 0 and n <= 1)
#         self.n = n

#     def conditional_probs(self, idx):
#         while True:
#           # generate random transition probabilities
#           a = self.gen.uniform()
#           b = a+self.n*(1-2*a)
#           # b = np.random.beta(mu, 1-mu)
#           # b = get_truncated_normal(mu, 1/3)
#           # b = 1-a + np.random.uniform(-.1,.1)
#           # b = min(1,max(0,b))
#           # b = 1-a

#           conditional_probs = np.array([[a,1-a],[b, 1-b]])
#           # figure out if this generated example is train or test based on transition probabilities hash
#           h = split(conditional_probs)
#           inp_split = 'test' if h % 4 == 0 else 'train' # designate 25% of examples as test
#           if inp_split == self.split:
#             break
        
#         stationary = torch.tensor(stationary_distribution(conditional_probs))
#         return torch.tensor(conditional_probs), stationary

class spectrum(markov):
    def __init__(self, split, p, length = 101, num_symbols = 2, last_token_only = False):
        super().__init__(split, length, num_symbols, last_token_only)
        assert(num_symbols == 2)
        assert(p >= 0 and p <= 1)
        self.p = p

    def conditional_probs(self, idx):
        while True:
            # generate random transition probabilities
            a = np.random.uniform()
            mu = a+self.p*(1-2*a)
            # b = np.random.beta(mu, 1-mu)
            # b = get_truncated_normal(mu, 1/3)
            # factor = (1/2-abs(1/3-self.p))
            factor = 0.2
            b = mu + np.random.uniform(-factor, factor)
            b = min(1,max(0,b))

            if np.random.rand() > .5:
                c = a
                a = b
                b = c
                # b = 1-a

            conditional_probs = np.array([[a,1-a],[b, 1-b]])
            # figure out if this generated example is train or test based on transition probabilities hash
            h = split(conditional_probs)
            inp_split = 'test' if h % 4 == 0 else 'train' # designate 25% of examples as test
            if inp_split == self.split:
                break
        
        stationary = torch.tensor(stationary_distribution(conditional_probs))
        return torch.tensor(conditional_probs), stationary

class interpolate(ngrams):
    def __init__(self, split, p, length = 101, num_symbols = 2, last_token_only = False):
        assert(num_symbols == 2)
        assert(p >= 0 and p <= 1)
        super().__init__(split, 2, length, num_symbols, last_token_only)
        self.p = p

    def conditional_probs(self, idx):
        while True:
            # generate random transition probabilities
            a = np.random.uniform()
            b = a + self.p * (1 - 2 * a)
            if np.random.rand() > .5: # UNDO????
                c = a
                a = b
                b = c
            
            conditional_probs = np.array([[a,1-a],[b, 1-b]])

            # figure out if this generated example is train or test based on transition probabilities hash
            h = split(conditional_probs)
            inp_split = 'test' if h % 4 == 0 else 'train' # designate 25% of examples as test
            if inp_split == self.split:
                break
        
        stationary = torch.tensor(stationary_distribution(conditional_probs))
        return torch.tensor(conditional_probs), stationary

class interpolate_unigram(ngrams):
    def __init__(self, split, p, length = 101, num_symbols = 2, last_token_only = False):
        assert(p >= 0 and p <= 1)
        super().__init__(split, 2, length, num_symbols, last_token_only)
        self.p = p
        self.benchmarks = [self.metric(np.random.dirichlet(np.ones(num_symbols), size=num_symbols)) for _ in range(1000)]
        self.benchmarks.sort()

    @staticmethod
    def metric(m):
        # return (np.abs(np.linalg.eigvals(m))).sum()
        stationary = torch.tensor(stationary_distribution(m))
        # assert not np.isnan(np.log(stationary)).any()
        return np.log(stationary) @ m @ stationary
        

    def conditional_probs(self, idx):
        while True:
            # generate random transition probabilities
            conditional_probs = np.random.dirichlet(np.ones(self.num_symbols), size=(self.num_symbols,))
            score = np.searchsorted(self.benchmarks, self.metric(conditional_probs)) / len(self.benchmarks)
            assert(score >= 0 and score <= 1)
            if np.random.rand() < abs(self.p - score) + .1:
                break
            
        stationary = torch.tensor(stationary_distribution(conditional_probs))
        return conditional_probs, stationary
import warnings
class doubly_stochastic_3(ngrams):
    def __init__(self, split, p=1, n=2, length = 101, num_symbols = 2, last_token_only = False):
        if num_symbols > 4:
            warnings.warn("uniform doubly stochastic with more than 4 symbols is very slow. will use non-uniform sampling")
        self.num_symbols = num_symbols
        assert(p in (0,1))
        super().__init__(split, n, length, num_symbols, last_token_only)
        self.p = p

    def conditional_probs(self, idx):
        k = self.num_symbols
        while True:
            #generate random transition probabilities
            if k <= 4:
                m = -np.ones((k,k))
                while (m<=0).any():
                    # generate random transition probabilities
                    m[:k-1,:k-1] = np.random.uniform(size=(k-1,k-1))
                    m[k-1,:k-1] = 1 - m[:k-1,:k-1].sum(axis=0)
                    m[:k,k-1] = 1 - m[:,:k-1].sum(axis=1)
                conditional_probs = m
            else:
                conditional_probs = doubly_stochastic_non_uniform(k)

            # figure out if this generated example is train or test based on transition probabilities hash
            h = split(conditional_probs)
            inp_split = 'test' if h % 4 == 0 else 'train' # designate 25% of examples as test
            if inp_split == self.split:
                break
        
        stationary = torch.tensor(stationary_distribution(conditional_probs))
        return torch.tensor(conditional_probs), stationary

class unigram_3(ngrams):
    def __init__(self, split, p, n = 2, length = 101, num_symbols = 2, last_token_only = False, uniform_prior = False):
        self.num_symbols = num_symbols
        assert(p in (0,1))
        super().__init__(split, n, length, num_symbols, last_token_only, uniform_prior=uniform_prior)
        self.p = p

    def conditional_probs(self, idx):
        while True:
            # generate random transition probabilities  
            r = np.random.dirichlet(np.ones(self.num_symbols))
            conditional_probs = np.stack([r]*self.num_symbols)
            # figure out if this generated example is train or test based on transition probabilities hash
            h = split(conditional_probs)
            inp_split = 'test' if h % 4 == 0 else 'train' # designate 25% of examples as test
            if inp_split == self.split:
                break
        
        stationary = torch.tensor(stationary_distribution(conditional_probs))
        return torch.tensor(conditional_probs), stationary



def doubly_stochastic_non_uniform(k=3):
    # Generate a random k x k stochastic matrix
    A = np.random.dirichlet(np.ones(k), size=k)
    
    # Iteratively adjust rows and columns
    for _ in range(100):  # Iteration limit to prevent infinite loops
        # Normalize columns
        A = A / A.sum(axis=0, keepdims=True)
        # Normalize rows
        A = A / A.sum(axis=1, keepdims=True)
    return A

class mixture(Dataset):
    def __init__(self, db1, db2, db1_length, length, p, list1 = None):
        self.db1 = db1
        self.db2 = db2
        self.length = int(length)
        if list1 is None:
            self.list1 = [db1.__getitem__(i) for i in range(db1_length)]
        else:
            self.list1 = list1
            assert len(list1) >= db1_length
        self.p = p

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if idx % 100 <= 100 * self.p:
            return self.list1[int(idx * self.p)]
        else:
            return self.db2.__getitem__(idx)
            


print("ngrams_simple using uniform prior for first symbol")
class ngrams_simple(ngrams):
    def __init__(self, split, n, length = 101, num_symbols = 2):
        super().__init__(split, n, length, num_symbols, last_token_only=False)
        self.indices = [i for i in range(self.length) if (i + 1) % (self.n + 1) == self.n]

    def __getitem__(self, idx):
        conditional_probs, stationary_distribution = self.conditional_probs(idx)
        thresholds = conditional_probs.cumsum(dim = 1).tolist()

        #generate sequence
        # base = choices(range(self.num_symbols**(self.n)), weights = stationary_distribution, k = self.length//self.n + 1)
        base = choices(range(self.num_symbols**(self.n)), k = self.length//(self.n + 1)+1)
        inp = []
        for i in range(self.length // (self.n + 1)+1):
            next_start = base[i]
            inp.extend(self.single_symbol_convert(next_start) + choices(range(self.num_symbols), cum_weights = thresholds[next_start]))
        inp = torch.tensor(inp, dtype=torch.long)
        x = inp[:self.length]
        y = torch.ones_like(x, dtype=torch.long) * -1
        y[self.indices] = inp[1:][self.indices]
        if self.split == 'test':
                y = conditional_probs, y
        return x, y
    
class random(Dataset):
    def __init__(self, split='train', length=101, num_symbols=2, last_token_only=False):
        self.length = length
        self.num_symbols = num_symbols
        self.last_token_only = last_token_only
        self.split = split
        self.data = torch.randint(0, self.num_symbols, (200,self.length), dtype=torch.long)
    
    def __len__(self):
        if self.split == 'train':
            return int(1e16) # ...
        if self.split == 'test':
            return 1000
    
    def __getitem__(self, idx):
        x = self.data[idx % 200]
        y = x[1:].clone()
        if self.last_token_only:
            y[:-1] = -1
        return x[:-1], y
