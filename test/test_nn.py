import unittest

import numpy as np
import torch
import torch_geometric.nn

from tinygraph.nn import GCNConv, SAGEConv
from tinygraph.nn.datasets import cora

data = cora()
x, edge_index = data.x, data.edge_index
in_featues, out_features = x.shape[1], len(set(data.y.tolist()))


class TestNN(unittest.TestCase):
    def test_gcnconv(self):
        def _test_gcnconv(bias):
            conv = GCNConv(in_featues, out_features, bias=bias)
            out = conv(x, edge_index)

            with torch.no_grad():
                pyg_conv = torch_geometric.nn.GCNConv(in_featues, out_features, bias=bias)
                pyg_conv.lin.weight[:] = torch.tensor(conv.weight.numpy())
                if bias:
                    pyg_conv.bias[:] = torch.tensor(conv.bias.numpy())
                pyg_x = torch.tensor(x.numpy())
                pyg_edge_index = torch.tensor(edge_index.numpy())
                pyg_out = pyg_conv(pyg_x, pyg_edge_index)

            np.testing.assert_allclose(out.numpy(), pyg_out.detach().numpy(), atol=1e-5, rtol=1e-5)
        
        for bias in [True, False]:
            _test_gcnconv(bias)

    def test_sageconv(self):
        def _test_sageconv(aggr, normalize, bias):
            conv = SAGEConv(in_featues, out_features, aggr=aggr, normalize=normalize, bias=bias)
            out = conv(x, edge_index)

            with torch.no_grad():
                aggr = "max" if aggr == "amax" else aggr
                pyg_conv = torch_geometric.nn.SAGEConv(in_featues, out_features, aggr=aggr, normalize=normalize, bias=bias)
                pyg_conv.lin_l.weight[:] = torch.tensor(conv.weight_neigh.numpy())
                pyg_conv.lin_r.weight[:] = torch.tensor(conv.weight_self.numpy())
                if bias:
                    pyg_conv.lin_l.bias[:] = torch.tensor(conv.bias.numpy())
                pyg_x = torch.tensor(x.numpy())
                pyg_edge_index = torch.tensor(edge_index.numpy()).long()
                pyg_out = pyg_conv(pyg_x, pyg_edge_index)

            np.testing.assert_allclose(out.numpy(), pyg_out.detach().numpy(), atol=1e-5, rtol=1e-5)

        for aggr in ["mean", "amax"]:
            for normalize in [True, False]:
                for bias in [True, False]:
                    _test_sageconv(aggr, normalize, bias)


if __name__ == "__main__":
    unittest.main()
