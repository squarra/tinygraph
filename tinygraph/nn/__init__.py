from tinygrad import Tensor


class AdjGCNConv:
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.weight = Tensor.glorot_uniform(out_features, in_features)
        self.bias = Tensor.zeros(out_features) if bias else None

    def __call__(self, x: Tensor, adj: Tensor) -> Tensor:
        adj_hat = adj + Tensor.eye(adj.shape[0])
        deg_inv_sqrt = adj_hat.sum(axis=1) ** -0.5
        adj_norm = deg_inv_sqrt.reshape(-1, 1) * adj_hat * deg_inv_sqrt.reshape(1, -1)
        xw = x.linear(self.weight.transpose())
        return adj_norm.linear(xw, self.bias)


class GCNConv:
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.weight = Tensor.glorot_uniform(out_features, in_features)
        self.bias = Tensor.zeros(out_features) if bias else None

    def __call__(self, x: Tensor, edge_index: Tensor) -> Tensor:
        num_nodes = x.shape[0]
        self_loops = Tensor.stack(Tensor.arange(num_nodes), Tensor.arange(num_nodes))
        edge_index = edge_index.cat(self_loops, dim=1)
        xw = x.linear(self.weight.transpose())
        src, dst = edge_index
        deg_inv_sqrt = Tensor.zeros(num_nodes).scatter(0, dst, Tensor.ones(dst.shape[0]), reduce="add") ** (-0.5)
        norm = deg_inv_sqrt[src] * deg_inv_sqrt[dst]
        messages = norm.unsqueeze(1) * xw[src]
        scatter_index = dst.unsqueeze(1).expand(messages.shape[0], messages.shape[1])
        out = Tensor.zeros_like(xw).scatter(0, scatter_index, messages, reduce="add")
        return out if self.bias is None else out.add(self.bias)
