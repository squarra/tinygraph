import torch_geometric.transforms as T
from torch_geometric import datasets

from tinygraph.data import Data, HeteroData


def cora(normalize=False):
    transform = T.NormalizeFeatures() if normalize else None
    return Data.from_pyg(datasets.Planetoid(root="/tmp/Cora", name="Cora", transform=transform)[0])


def reddit():
    return Data.from_pyg(datasets.Reddit(root="/tmp/Reddit")[0])


def dblp():
    data = datasets.DBLP(root="/tmp/DBLP", transform=T.Constant(node_types="conference"))[0]
    del data["conference"].num_nodes
    return HeteroData.from_pyg(data)


def imdb():
    return HeteroData.from_pyg(datasets.IMDB(root="/tmp/IMDB")[0])
