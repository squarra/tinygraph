from torch_geometric.datasets import Planetoid, Reddit
from torch_geometric.transforms import NormalizeFeatures
from tinygraph.data import Data


def cora():
    return Data.from_pyg(Planetoid(root="/tmp/Cora", name="Cora", transform=NormalizeFeatures())[0])


def reddit():
    return Data.from_pyg(Reddit(root="/tmp/Reddit")[0])
