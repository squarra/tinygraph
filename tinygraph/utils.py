from tinygrad import Tensor


def add_self_loops(edge_index, num_nodes):
    return edge_index.cat(Tensor.stack(Tensor.arange(num_nodes), Tensor.arange(num_nodes)), dim=1)
