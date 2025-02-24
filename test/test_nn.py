import unittest

import numpy as np
import torch
import torch_geometric.nn
from tinygrad import Tensor

from tinygraph.nn import GCNConv


class TestNN(unittest.TestCase):
    def test_gcnconv(self):
        x = Tensor.randn(6, 5)
        edge_index = Tensor([[0, 2, 3, 4], [1, 1, 4, 5]])
        in_featues, out_features = x.shape[1], 3
        conv = GCNConv(in_featues, out_features)
        out = conv(x, edge_index)

        with torch.no_grad():
            pyg_conv = torch_geometric.nn.GCNConv(in_featues, out_features)
            pyg_conv.lin.weight[:] = torch.tensor(conv.weight.numpy())
            pyg_conv.bias[:] = torch.tensor(conv.bias.numpy())
            pyg_x = torch.tensor(x.numpy())
            pyg_edge_index = torch.tensor(edge_index.numpy())
            pyg_out = pyg_conv(pyg_x, pyg_edge_index)

        np.testing.assert_allclose(out.numpy(), pyg_out.detach().numpy(), rtol=1e-5)
