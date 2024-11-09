import torch.nn as nn

def partition_model(model, rank, n_ranks):
    """
    model must be a nn.Sequential
    切分一个模型为 ``nranks`` 个子模型，并返回其中的第 ``rank`` 个子模型
    """
    n_layers = len(model)
    splits = [n_layers // n_ranks] * n_ranks
    remainder = n_layers % n_ranks
    for i in range(remainder):
        splits[i] += 1
    
    start = sum(splits[:rank])
    end = start + splits[rank]
    return nn.Sequential(*model[start:end])
    
