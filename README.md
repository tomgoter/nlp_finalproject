# Extension of Unsupervised Data Augmentation to XLNet and Application to Game of Thrones Text Classification
## Final Project for W266: Natural Language Processing with Deep Learning
### Thomas Goter
### Fall 2019

The Unsupervised Data Augmentation (UDA) methodology presented in https://arxiv.org/pdf/1904.12848.pdf is further extended beyond its initial use with BERT for use with the [XLNet](https://arxiv.org/pdf/1906.08237.pdf) transformer-based neural language model for evaluation on a novel text classification dataset. The results discussed herein show that the benefits of UDA are reproducible and extensible to other modeling architectures, namely XLNet. For the novel *Song of Ice and Fire* text classification problem presented herein, absolute error rate reductions of up to 5\% were shown to be possible with an optimized UDA model.  Additionally, it is shown UDA can achieve the same accuracy as a finetuned model with as little as 67\% of the labeled data. However, UDA is not a magic bullet for enabling the use of complex neural architectures for cases with very limited sets of labeled data, and the complexity of its use and time associated with its optimization should be weighed against the cost of simply labeling additional data.


**Acknowledgement**  
This work builds on the work presented in https://arxiv.org/pdf/1904.12848.pdf and extends it to optionally make use of the [XLNet](https://arxiv.org/pdf/1906.08237.pdf) architecture as well. As such, much of the open source code from the respective GitHub repositories for [UDA](https://github.com/google-research/uda) and [XLNet](https://github.com/zihangdai/xlnet) was leveraged for this project. Thank you!


