import torch, torch.nn as nn, torch.nn.functional as F, numpy as np
from torch.nn.parameter import Parameter
from allennlp.modules.attention.attention import Attention
from allennlp.nn import Activation

import umap
import pandas as pd

# Matplotlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import plotly.express as px

####################################
# Evaluation utils

def mean_reciprocal_rank(correct_gene_ranks):
    return torch.mean(1/correct_gene_ranks)

def average_rank(correct_gene_ranks):
    return torch.mean(correct_gene_ranks)

def top_k_acc(correct_gene_ranks, k):
    return torch.sum(correct_gene_ranks <= k) / len(correct_gene_ranks)


###########################################

# below functions from AllenNLP

def masked_mean(
    vector: torch.Tensor, mask: torch.BoolTensor, dim: int, keepdim: bool = False
    ) -> torch.Tensor:
    """
    To calculate mean along certain dimensions on masked values
    # Parameters
    vector : `torch.Tensor`
        The vector to calculate mean.
    mask : `torch.BoolTensor`
        The mask of the vector. It must be broadcastable with vector.
    dim : `int`
        The dimension to calculate mean
    keepdim : `bool`
        Whether to keep dimension
    # Returns
    `torch.Tensor`
        A `torch.Tensor` of including the mean values.
    """
    replaced_vector = vector.masked_fill(~mask, 0.0)

    value_sum = torch.sum(replaced_vector, dim=dim, keepdim=keepdim)
    value_count = torch.sum(mask, dim=dim, keepdim=keepdim)
    return value_sum / value_count.float().clamp(min=tiny_value_of_dtype(torch.float))


def tiny_value_of_dtype(dtype: torch.dtype):
    """
    Returns a moderately tiny value for a given PyTorch data type that is used to avoid numerical
    issues such as division by zero.
    This is different from `info_value_of_dtype(dtype).tiny` because it causes some NaN bugs.
    Only supports floating point dtypes.
    """
    if not dtype.is_floating_point:
        raise TypeError("Only supports floating point dtypes.")
    if dtype == torch.float or dtype == torch.double:
        return 1e-13
    elif dtype == torch.half:
        return 1e-4
    else:
        raise TypeError("Does not support dtype " + str(dtype))


def masked_softmax(
    vector: torch.Tensor,
    mask: torch.BoolTensor,
    dim: int = -1,
    memory_efficient: bool = False,
    ) -> torch.Tensor:
    """
    `torch.nn.functional.softmax(vector)` does not work if some elements of `vector` should be
    masked.  This performs a softmax on just the non-masked portions of `vector`.  Passing
    `None` in for the mask is also acceptable; you'll just get a regular softmax.
    `vector` can have an arbitrary number of dimensions; the only requirement is that `mask` is
    broadcastable to `vector's` shape.  If `mask` has fewer dimensions than `vector`, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    If `memory_efficient` is set to true, we will simply use a very large negative number for those
    masked positions so that the probabilities of those positions would be approximately 0.
    This is not accurate in math, but works for most cases and consumes less memory.
    In the case that the input vector is completely masked and `memory_efficient` is false, this function
    returns an array of `0.0`. This behavior may cause `NaN` if this is used as the last layer of
    a model that uses categorical cross-entropy loss. Instead, if `memory_efficient` is true, this function
    will treat every element as equal, and do softmax over equal numbers.
    """
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the mask, we zero these out.
            result = torch.nn.functional.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (
                result.sum(dim=dim, keepdim=True) + tiny_value_of_dtype(result.dtype)
            )
        else:
            masked_vector = vector.masked_fill(~mask, min_value_of_dtype(vector.dtype))
            result = torch.nn.functional.softmax(masked_vector, dim=dim)
    return result


def weighted_sum(matrix: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
    """
    Takes a matrix of vectors and a set of weights over the rows in the matrix (which we call an
    "attention" vector), and returns a weighted sum of the rows in the matrix.  This is the typical
    computation performed after an attention mechanism.
    Note that while we call this a "matrix" of vectors and an attention "vector", we also handle
    higher-order tensors.  We always sum over the second-to-last dimension of the "matrix", and we
    assume that all dimensions in the "matrix" prior to the last dimension are matched in the
    "vector".  Non-matched dimensions in the "vector" must be `directly after the batch dimension`.
    For example, say I have a "matrix" with dimensions `(batch_size, num_queries, num_words,
    embedding_dim)`.  The attention "vector" then must have at least those dimensions, and could
    have more. Both:
        - `(batch_size, num_queries, num_words)` (distribution over words for each query)
        - `(batch_size, num_documents, num_queries, num_words)` (distribution over words in a
          query for each document)
    are valid input "vectors", producing tensors of shape:
    `(batch_size, num_queries, embedding_dim)` and
    `(batch_size, num_documents, num_queries, embedding_dim)` respectively.
    """
    # We'll special-case a few settings here, where there are efficient (but poorly-named)
    # operations in pytorch that already do the computation we need.
    if attention.dim() == 2 and matrix.dim() == 3:
        return attention.unsqueeze(1).bmm(matrix).squeeze(1)
    if attention.dim() == 3 and matrix.dim() == 3:
        return attention.bmm(matrix)
    if matrix.dim() - 1 < attention.dim():
        expanded_size = list(matrix.size())
        for i in range(attention.dim() - matrix.dim() + 1):
            matrix = matrix.unsqueeze(1)
            expanded_size.insert(i + 1, attention.size(i + 1))
        matrix = matrix.expand(*expanded_size)
    intermediate = attention.unsqueeze(-1).expand_as(matrix) * matrix
    return intermediate.sum(dim=-2)



###########################################
# wandb plots

def mrr_vs_percent_overlap(correct_gene_rank, percent_overlap_train):
    df = pd.DataFrame({"Percent of Phenotypes Found in Single Train Patient": percent_overlap_train.squeeze(), "Rank of Correct Gene": correct_gene_rank})
    fig = px.scatter(df, x = "Percent of Phenotypes Found in Single Train Patient", y = "Rank of Correct Gene")
    return fig

def plot_softmax(softmax):
    softmax = [s.detach().item() for s in softmax]
    df = pd.DataFrame({'softmax':softmax})
    fig = px.histogram(df, x="softmax")
    return fig

def fit_umap(embed, labels={}, n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', random_state=3):
    embed = embed.detach().cpu()
    mapping = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, metric=metric, random_state=random_state).fit(embed)
    embedding = mapping.transform(embed)

    data = {"x": embedding[:, 0], "y": embedding[:, 1]}
    if len(labels) > 0: data.update(labels)
    df = pd.DataFrame(data)
    if len(labels) > 0: fig = px.scatter(df, x = "x", y = "y", color = "Node Type", hover_data=list(labels.keys()))
    else: fig = px.scatter(df, x = "x", y = "y")
    return fig

def plot_degree_vs_attention(attn_weights, phenotype_names, single_patient=False):
    if single_patient:
        phenotype_names = phenotype_names[0]
        attn_weights = attn_weights[0]
        data = [(w.item(), p_name[1]) for w, p_name in zip(attn_weights, phenotype_names)]
    else:
        data = [(attn_weights[i], phenotype_names[i]) for i in range(len(phenotype_names))]
        data = [(w.item(), p_name[1]) for attn_w, phen_name in data for w, p_name in zip(attn_w, phen_name)] 
    attn_weights, degrees = zip(*data)
    df = pd.DataFrame({"Node Degree": degrees, "Attention Weight": attn_weights})
    fig = px.scatter(df, x = "Node Degree", y = "Attention Weight")
    return fig

def plot_nhops_to_gene_vs_attention(attn_weights, phenotype_names, nhops_g_p, single_patient=False):
    if single_patient:
        attn_weights = attn_weights[0]
        nhops_g_p = nhops_g_p[0]
        phenotype_names = phenotype_names[0]
        data = [(w.item(), hops) for w, hops in zip(attn_weights, nhops_g_p)]
    else:
        data = [(attn_weights[i], nhops_g_p[i]) for i in range(len(phenotype_names))]
        data = [(w.item(), hop) for attn_w, nhops in data for w, hop in zip(attn_w, nhops)] 
    attn_weights, n_hops_g_p = zip(*data)
    df = pd.DataFrame({"Number of Hops to Gene in KG": n_hops_g_p, "Attention Weight": attn_weights})
    fig = px.scatter(df, x = "Number of Hops to Gene in KG", y = "Attention Weight")
    return fig

def plot_gene_rank_vs_x_intrain(corr_gene_ranks, in_train):
    if sum(in_train == 1) == 0: 
        values_in_train = -1
        err_in_train = 0
    else: 
        values_in_train = torch.mean(corr_gene_ranks[in_train == 1].float())
        err_in_train = torch.std(corr_gene_ranks[in_train == 1].float())
    if sum(in_train == 0) == 0: 
        values_not_in_train = -1
        err_not_in_train = 0
    else: 
        values_not_in_train = torch.mean(corr_gene_ranks[in_train == 0].float())
        err_not_in_train = torch.std(corr_gene_ranks[in_train == 0].float())
    df = pd.DataFrame({"Average Rank of Correct Gene": [values_in_train, values_not_in_train], "In Train or Not": ["True", "False"], "Error Bars": [err_in_train, err_not_in_train]})
    fig = px.bar(df, x = "In Train or Not", y = "Average Rank of Correct Gene", error_y = "Error Bars")
    return fig

def plot_gene_rank_vs_numtrain(corr_gene_ranks, correct_gene_nid, train_corr_gene_nid):
    gene_counts = [train_corr_gene_nid[g] if g in train_corr_gene_nid else 0 for g in list(correct_gene_nid.numpy())]
    data = {"Rank of Correct Gene": corr_gene_ranks.cpu(), "Number of Times Seen": gene_counts, "Gene ID": correct_gene_nid}
    df = pd.DataFrame(data)
    fig = px.scatter(df, x = "Number of Times Seen", y = "Rank of Correct Gene", hover_data=list(data.keys()))
    return fig, gene_counts


def plot_gene_rank_vs_trainset(corr_gene_ranks, correct_gene_nid, gene_counts): # train_corr_gene_nid has dimension num_gene x num_sets (corr, cand, sparse, target)
    trainset_labels = ["-".join([str(int(l)) for l in list((gene_counts[i, :] > 0).numpy())]) for i in range(gene_counts.shape[0])]
    gene_ranks_dict = {}
    for l, r in zip(trainset_labels, corr_gene_ranks): # (corr, cand, sparse, target)
        l_full = []
        if l.split("-")[0] == "1": l_full.append("Corr")
        if l.split("-")[1] == "1": l_full.append("Cand")
        if l.split("-")[2] == "1": l_full.append("Sparse")
        if l.split("-")[3] == "1": l_full.append("Target")
        if len(l_full) == 0: l_full = "None"
        else: l_full = "-".join(l_full)
        if l_full not in gene_ranks_dict: gene_ranks_dict[l_full] = []
        gene_ranks_dict[l_full].append(int(r))
    avg_gene_ranks = {l: np.mean(r) for l, r in gene_ranks_dict.items()}
    df = pd.DataFrame({"Train Set": list(avg_gene_ranks.keys()), "Average Rank of Correct Gene": list(avg_gene_ranks.values())})
    fig = px.bar(df, x = "Train Set", y = "Average Rank of Correct Gene")
    return fig


def plot_gene_rank_vs_fraction_phenotype(corr_gene_ranks, frac_p):
    df = pd.DataFrame({"Rank of Correct Gene": corr_gene_ranks, "Fraction of Phenotypes": frac_p})
    df = df[df["Fraction of Phenotypes"] > -1]
    fig = px.scatter(df, x = "Fraction of Phenotypes", y = "Rank of Correct Gene")
    return fig


def plot_gene_rank_vs_hops(corr_gene_ranks, n_hops):
    mean_hops = []
    min_hops = []
    for hops in n_hops:
        if type(hops) == list: # gene to phenotype n_hops
            mean_hops.append(np.mean(hops))
            min_hops.append(np.min(hops))
        else: # phenotype to phenotype n_hops
            filtered_hops = torch.cat([hops[i][hops[i] > 0] for i in range(hops.shape[0])]).float()
            mean_hops.append(torch.mean(filtered_hops).item())
            min_hops.append(torch.min(filtered_hops).item())
        
    df = pd.DataFrame({"Rank of Correct Gene": corr_gene_ranks, "Mean Number of Hops": mean_hops})
    fig_mean = px.scatter(df, x = "Mean Number of Hops", y = "Rank of Correct Gene")
    df = pd.DataFrame({"Rank of Correct Gene": corr_gene_ranks, "Min Number of Hops": min_hops})
    fig_min = px.scatter(df, x = "Min Number of Hops", y = "Rank of Correct Gene")
    return fig_mean, fig_min
