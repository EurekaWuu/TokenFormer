---
license: apache-2.0
---

The *TokenFormer* is a **fully attention-based architecture** 
that unifies the computations of token-token and token-parameter interactions 
by entirely employing the attention mechanism, **maximizes the flexibility of neural network**.[(see paper)](https://arxiv.org/pdf/2410.23168). 
It contains four models of sizes 
150M, 450M, 900M, 1.5B. For each size, it's trained based on [gpt-neox](https://github.com/EleutherAI/gpt-neox) code base and uses [Pile](https://huggingface.co/datasets/EleutherAI/pile) with 300B tokens. 
All 4 model sizes are trained on the exact 
same data, in the exact same order.

# TokenFormer-150M

## Model Details

- Developed by: [Haiyang Wang](https://haiyang-w.github.io/)
- Model type: TokenFormer-based Language Model
- Language: English
- Learn more: [TokenFormer's GitHub repository](https://github.com/Haiyang-W/TokenFormer)
 for training procedure, config files, and details on how to use.
 [See paper](https://arxiv.org/pdf/2410.23168) for more evals and implementation
 details.
- Library: [GPT-NeoX](https://github.com/EleutherAI/gpt-neox)
- License: Apache 2.0
- Contact: to ask questions about this model, please email Haiyang Wang.

<figure>

| TokenFormer model | Layers | #QKV Param Tokens | #Output Param Tokens | #FFN Param Tokens | Model Dim | Heads | Batch Size | Learning Rate         | Training Iterations         |
| ----------------: | -----: | :---------------: | :------------------: | :---------------: | :-------: | :---: | :--------: | :-------------------: | :-------------------------: |
| 150M              | 12     | 768               | 768                  | 3072              | 768       | 12    | 2M         | 6.0 x 10<sup>-4</sup> | 143000                      |
| 450M              | 24     | 1024              | 1024                 | 4096              | 1024      | 16    | 2M         | 6.0 x 10<sup>-4</sup> | 143000                      | 
| 900M              | 32     | 1280              | 1280                 | 5120              | 1280      | 16    | 2M         | 6.0 x 10<sup>-4</sup> | 143000                      |
| 1.5B              | 40     | 1536              | 1536                 | 6144              | 1536      | 16    | 2M         | 6.0 x 10<sup>-4</sup> | 143000                      | 
<figcaption>Engineering details for the <i>TokenFormer</i>. </figcaption>
</figure>

## Training

### Training data

[The Pile](https://pile.eleuther.ai/) is a 825GiB general-purpose dataset in 
English. It was created by EleutherAI specifically for training large language 
models. It contains texts from 22 diverse sources, roughly broken down into 
five categories: academic writing (e.g. arXiv), internet (e.g. CommonCrawl), 
prose (e.g. Project Gutenberg), dialogue (e.g. YouTube subtitles), and 
miscellaneous (e.g. GitHub, Enron Emails). See [the Pile 
paper](https://arxiv.org/abs/2101.00027) for a breakdown of all data sources, 
methodology, and a discussion of ethical implications. Consult [the 
datasheet](https://arxiv.org/abs/2201.07311) for more detailed documentation 
about the Pile and its component datasets. The Pile can be downloaded from 
the [official website](https://pile.eleuther.ai/), or from a [community 
mirror](https://the-eye.eu/public/AI/pile/).<br>

### Training procedure
We follow the default training strategy of [Pythia](https://arxiv.org/abs/2304.01373) in [gpt-neox](https://github.com/EleutherAI/gpt-neox), 
including the dataset processing, hyper-parameter and code base.
All models were trained on the exact same data, in the exact same order. Each 
model saw 299,892,736,000 tokens during training. 

All *TokenFormer* models trained for 143000 steps at a batch size 
of 2M (2,097,152 tokens).<br>
See [GitHub](https://github.com/Haiyang-W/TokenFormer) for more details on training
 procedure.<br>
TokenFormer uses the same tokenizer as [GPT-NeoX-
20B](https://huggingface.co/EleutherAI/gpt-neox-20b).

## Evaluations

All *TokenFormer* models were evaluated using the [LM Evaluation 
Harness](https://github.com/EleutherAI/lm-evaluation-harness). 
You can run the evaluation with our [instruction](https://github.com/Haiyang-W/TokenFormer?tab=readme-ov-file#evaluations).<br>
Expand the sections below to see plots of evaluation results for all 
TokenFormer compared with Opensource Transformer-based LLMs.

<figure>

| Model        | #Param   |  LAMBADA | HellaSwag | PIQA | Arc-E  | Arc-C | WinoGrande | Average  |
| :----:       | :------: | :------: | :-------: | :--: | :---:  | :---: | :--------: | :------: |
| Pythia       |    150M  | 35.4     |     30.3  | 62.3 |  43.6  | 23.6  |  51.3      |   40.1   | 
| **TokenFormer**  |    150M  | **45.0**     |     **35.5**  | **64.9** |  **47.3**  | **24.9**  |  **50.4**      |   **44.7**   |
| Pythia       |    410M  | 51.4     |     40.6  | 66.9 |  52.1  | 24.6  |  53.8      |   48.2   |
| **TokenFormer**  |    450M  | **57.3**     |     **47.5**  | **69.5** |  **56.2**  | **26.7**  |  **54.6**      |   **52.0**   |
| Pythia       |    1B    | 56.1     |     47.2  | 70.7 |  57.0  | 27.1  |  53.5      |   51.9   | 
| **TokenFormer**  |    900M  | **64.0**     |     **55.3**  | **72.4** |  **59.9**  | **30.6**  |  **56.4**      |   **56.4**   | 
| GPT-Neo      |    1.3B  | 57.2     |     48.9  | 71.1 |  56.2  | 25.9  |  54.9      |   52.4   | 
| OPT          |    1.3B  | 58.0     |     53.7  | 72.4 |  56.7  | 29.6  |  59.5      |   55.0   |
| Pythia       |    1.3B  | 61.7     |     52.1  | 71.0 |  60.5  | 28.5  |  57.2      |   55.2   |
| GPT-Neo      |    2.7B  | 62.2     |     55.8  | 71.1 |  61.1  | 30.2  |  57.6      |   56.5   |
| OPT          |    2.7B  | 63.6     |     60.6  | 74.8 |  60.8  | 31.3  |  61.0      |   58.7   | 
| Pythia       |    2.8B  | 64.7     |     59.3  | 74.0 |  64.1  | 32.9  |  59.7      |   59.1   | 
| **TokenFormer**  |    1.5B  | **64.7**     |     60.0  | **74.8** |  **64.8**  | 32.0  |  59.7      |   **59.3**   |
<figcaption>Zero-shot evaluation of Language Modeling. </figcaption>
</figure>

