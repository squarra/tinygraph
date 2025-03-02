from tinygrad import Tensor
from tinygraph.utils import add_self_loops, degree


class GCNConv:
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.weight = Tensor.glorot_uniform(out_features, in_features)
        self.bias = Tensor.zeros(out_features) if bias else None

    def __call__(self, x: Tensor, edge_index: Tensor) -> Tensor:
        num_nodes = x.shape[0]
        edge_index = add_self_loops(edge_index, num_nodes)
        x = x.linear(self.weight.transpose())
        src, dst = edge_index
        deg_inv_sqrt = degree(dst, num_nodes).pow(-0.5)
        norm = deg_inv_sqrt[src] * deg_inv_sqrt[dst]
        messages = norm.unsqueeze(1) * x[src]
        index = dst.unsqueeze(1).expand(messages.shape)
        out = Tensor.zeros_like(x).scatter_reduce(0, index, messages, reduce="sum")
        return out if self.bias is None else out.add(self.bias)


class SAGEConv:
    def __init__(self, in_features: int, out_features: int, aggr: str = "mean", normalize: bool = False, bias: bool = True):
        self.weight_self = Tensor.glorot_uniform(out_features, in_features)
        self.weight_neigh = Tensor.glorot_uniform(out_features, in_features)
        self.bias = Tensor.zeros(out_features) if bias else None
        self.normalize = normalize
        self.aggr = aggr

    def __call__(self, x: Tensor, edge_index: Tensor) -> Tensor:
        src, dst = edge_index
        out_self = x.linear(self.weight_self.transpose())
        messages = x[src]
        index = dst.unsqueeze(1).expand(messages.shape)
        out_neigh = Tensor.zeros_like(x).scatter_reduce(0, index, messages, reduce=self.aggr, include_self=False)
        out_neigh = out_neigh.linear(self.weight_neigh.transpose())
        out = out_self + out_neigh

        if self.bias is not None:
            out = out.add(self.bias)

        if self.normalize:
            norm = out.pow(2).sum(axis=1).sqrt().unsqueeze(1)
            out = out / (norm + 1e-7)  # Add epsilon to prevent division by zero

        return out
