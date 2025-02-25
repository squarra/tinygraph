from tinygrad import Tensor


def add_self_loops(edge_index: Tensor, num_nodes: int) -> Tensor:
    return edge_index.cat(Tensor.stack(Tensor.arange(num_nodes), Tensor.arange(num_nodes)), dim=1)


def degree(index: Tensor, num_nodes: int) -> Tensor:
    assert index.ndim == 1
    return Tensor.zeros(num_nodes).scatter(0, index, Tensor.ones(index.shape[0]), reduce="add")
