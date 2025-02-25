import unittest

import torch
import torch_geometric.utils as pyg

from test.helpers import generate_random_edge_index
from tinygraph.utils import add_self_loops, degree


class TestNN(unittest.TestCase):
    def test_add_self_loops(self):
        num_edges, num_nodes = 10, 10
        edge_index = generate_random_edge_index(num_edges, num_nodes)
        new_edge_index = add_self_loops(edge_index, num_nodes)
        self.assertEqual(new_edge_index.shape[1], num_edges + num_nodes)

    def test_degree(self):
        num_edges, num_nodes = 10, 10
        index = generate_random_edge_index(num_edges, num_nodes)[1]
        deg = degree(index, num_nodes)
        pyg_degree = pyg.degree(torch.tensor(index.numpy(), dtype=torch.long), num_nodes=num_nodes)
        self.assertEqual(deg.tolist(), pyg_degree.tolist())
