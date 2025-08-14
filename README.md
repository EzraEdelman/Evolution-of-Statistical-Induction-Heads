# The Evolution of Statistical Induction Heads: In-Context Learning Markov Chains

[![Paper](https://img.shields.io/badge/NeurIPS-2024-blue)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/75b0edb869e2cd509d64d0e8ff446bc1-Abstract-Conference.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of **"The Evolution of Statistical Induction Heads: In-Context Learning Markov Chains"** (NeurIPS 2024).

[Ezra Edelman*](https://ezraedelman.com/), [Nikolaos Tsilivis*](https://cims.nyu.edu/~nt2231/page.html), [Benjamin L. Edelman](https://benjaminedelman.com/), [Eran Malach](https://www.eranmalach.com/), [Surbhi Goel](https://www.surbhigoel.com/)

*Equal contribution

[Blog post](https://unprovenalgos.github.io/statistical-induction-heads)

## Abstract

Large language models have the ability to generate text that mimics patterns in their inputs. We introduce a simple Markov Chain sequence modeling task in order to study how this in-context learning capability emerges. In our setting, each example is sampled from a Markov chain drawn from a prior distribution over Markov chains. Transformers trained on this task form *statistical induction heads* which compute accurate next-token probabilities given the bigram statistics of the context. During the course of training, models pass through multiple phases: after an initial stage in which predictions are uniform, they learn to sub-optimally predict using in-context single-token statistics (unigrams); then, there is a rapid phase transition to the correct in-context bigram solution. We conduct an empirical and theoretical investigation of this multi-phase process, showing how successful learning results from the interaction between the transformer's layers, and uncovering evidence that the presence of the simpler unigram solution may delay formation of the final bigram solution. We examine how learning is affected by varying the prior distribution over Markov chains, and consider the generalization of our in-context learning of Markov chains (ICL-MC) task to n-grams for n > 2.

## Key Findings

1. **Transformers learn statistical induction heads to optimally solve ICL-MC**: Transformers develop mechanisms that compute the correct conditional (posterior) probability of the next token given all previous occurrences of the prior token in context, achieving performance approaching that of the Bayes-optimal predictor.

2. **Transformers learn predictors of increasing complexity and undergo phase transitions**: Learning appears separated into distinct phases with rapid drops in loss between them. Different phases correspond to learning models of increased complexity—first unigrams, then bigrams—and this pattern extends to n-grams for n > 2.

3. **Simplicity bias may slow down learning**: Evidence suggests that the model's inherent bias towards simpler solutions (in-context unigrams) causes learning of the optimal solution to be delayed. Changing the distribution of in-context examples to remove the usefulness of unigrams leads to faster convergence.

4. **Alignment of layers is crucial**: The transition from learning the simple-but-inadequate solution to the complex-and-correct solution happens due to an alignment between the layers of the model: the learning signal for the first layer is tied to the extent to which the second layer approaches its correct weights.

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{edelman2024evolution,
  title={The Evolution of Statistical Induction Heads: In-Context Learning Markov Chains},
  author={Edelman, Ezra and Tsilivis, Nikolaos and Edelman, Benjamin L. and Malach, Eran and Goel, Surbhi},
  booktitle={Advances in Neural Information Processing Systems},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

Code adapted from [minGPT](https://github.com/karpathy/minGPT).
