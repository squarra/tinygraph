from torch_geometric import datasets
from torch_geometric.transforms import NormalizeFeatures
from tinygraph.data import Data, HeteroData


def cora(normalize=False):
    transform = NormalizeFeatures() if normalize else None
    return Data.from_pyg(datasets.Planetoid(root="/tmp/Cora", name="Cora", transform=transform)[0])


def reddit():
    return Data.from_pyg(datasets.Reddit(root="/tmp/Reddit")[0])


def dblp():
    return HeteroData.from_pyg(datasets.DBLP(root="/tmp/DBLP")[0])


def imdb():
    return HeteroData.from_pyg(datasets.IMDB(root="/tmp/IMDB")[0])
