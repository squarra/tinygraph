from tinygrad import Tensor


def generate_random_edge_index(num_edges, num_nodes):
    src = Tensor.randint(num_edges, low=0, high=num_nodes)
    offsets = Tensor.randint(num_edges, low=1, high=num_nodes)
    dst = src.add(offsets).mod(num_nodes).contiguous()
    return src.stack(dst)
