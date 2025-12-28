# Engine Code Structure

An Engine (reference: ModelRunner in SGLang) is a Python Class that initialize:

1. `Model` + `Sampler`
2. `Distributed Communication`
3. `KVCache` and `PageTable`
4. `Attention Backend`

## Model + Sampler

Model is the base LLM model defined in `minisgl/models`.
Normally, it is composed of an embedding layer, multiple decoder layers, and one language modelling head.

### Embedding Layer

The input of a model is a batch of tokens from different requests.
However, the input length of different requests.
Naively padding the input length of requests to the maximum of them will lead to substantial resource waste.
With continuous batching introduced in [Orca](https://www.usenix.org/conference/osdi22/presentation/yu),
modern LLM serving engines flatten the input tokens into an 1D tensor.
Naive padding will have a 2D tensor with shape $[batch-size, max(seq-len in batch)]$,
while flattened tokens is an 1D tensor with shape $[sum(seq-len)]$.
This could reduce the redundant computation and memory caused by padding.

After passing the 1D flattened tokens to the embedding layer, we will get a 2D-tensor
shaped like $[sum(seq-len), hidden-size]$

### Decoder Layer

The input of a decoder layer is a tensor with shape $[sum(seq-len), hidden-size]$.

For a dense model, an decoder layer are mainly composed of an attention layer and an FFN layer.

#### Attention Layer

The input of a attention layer is a tensor with shape $[sum(seq-len), hidden-size]$.

As is mentioned in `FlashAttention` [paper](https://arxiv.org/abs/2205.14135), a naive implementation
of attention can be inefficient due to redundant HBM access.
Currently, there're many open-source high performance attention implementation, such as FlashAttention,
FlashInfer, FlashMLA. Each of them requires do some extra preparation work for attention computation,
such as pre-computing a cumulative sequence length tensor and page-table, which we call metadata.
Different attention implementations requires different attention metadata.
Supporting all these variants with all models will bring a rise in code complexity:
suppose there's $n$ models and $m$ backend, then we will have to write $n \times m$ code.
To eliminate the redundant code, we introduce `AttentionBackend`. It will handle the preparation work
of its metadata as well as call the inner implementation in the forward pass.
With this abstraction of `AttentionBackend`, we only need to write $n$ models + $m$ attention backends,
instead of $n \times m$ code.

The output of a attention layer is still a 2D tensor with shape $[sum(seq-len), hidden-size]$.

#### Feed Forward Network (FFN) Layer

The input of a feed-forward network layer is a 2D tensor with shape $[sum(seq-len), hidden-size]$.

In the FFN layer, the input tensor will first scales up to intermediate size (usually larger than hidden-size),
performs an activation function, and then scales down to the original shape.

For MoE models, the mechanism will be much trickier.

The output of a attention layer is still a 2D tensor with shape $[sum(seq-len), hidden-size]$.

### Language Modelling Head (LM HEAD)

LM Head will map the input tensor with shape $[sum(seq-len), hidden-size]$ into 1D tokens
with shape $[sum(seq-len), hidden-size]$.

### Sampler

After we get all the tokens, we will sample the them and get the next token id due to the auto-regressive nature of LLMs.
In samlp

## Distributed Communication

As the models are scaling up, the inferece cost is rising: the memory and compute demand of a single LLM is increasing sharply,
beyond the capacity of a single GPU.
To effectively serve the model on multiple GPUs, we need distributed communication.

For most layers, there computation can be split to different smaller computations, and finally added by each partition.
The action of adding the input from different GPUs is called `all_reduce`, which is an important distributed primitives.

## KVCache

Without KVCache, we will have to recompute the full sequence for each output token.
Due to the mathmatical fact that the keys and values of past tokens in the attention layer remains unchanged,
along with the fact that computing the new keys and values only requires the new query tensors,
we can `cache` the past key values, and only recompute for the new query, key, value tensors.

For example, suppose there's a input request with $seq-len$ where the first n tokens is cached,
then we only need to compute the uncached $seq-len - n$ part, which is denoted as $seq-len-q$
(since only these parts of query tensors is needed in computation).
With this optimization, we can reduce the input tensor shape of a decoder layer
from $[sum(seq-len), hidden-size]$ to $[sum(seq-len-q), hidden-size]$.

Specifically, when $seq-len-q = 1$ for all requests, the input tensor shape of a decoder layer
will be $[batch-size, hidden-size]$. This special computation phase is called decode.
Correspondingly, other phases are called prefill (or `extend`).

## PageTable

Similar to memory management in OS, there could be memory fragmentation in KVCache management.
To mitigate this, vLLM introduced Page Attention. The serving engine manages a page-table that
maps each request to the real location of their KVCache, which means the storage of consecutive tokens
in one request may not be necessarily contiguous in memory storage. Similar to the page-table in OS,
they may actual live in segmented memory and the page table stores mapping from "virtual address" to "physical address"

In implementation, it is usually an 2D tensor. The value at index $[req-id, n]$ is the "physical address"
(i.e. offset in the paged KVCache) of the $n$-th token of the $req-id$.
