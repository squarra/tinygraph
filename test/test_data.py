import unittest

from tinygrad import Tensor

from test.helpers import generate_random_edge_index
from tinygraph.data import Data, HeteroData


class TestData(unittest.TestCase):
    def test_data(self):
        num_nodes, num_features = 10, 16
        x = Tensor.rand(num_nodes, num_features)
        num_edges = 10
        edge_index = generate_random_edge_index(num_edges, num_nodes)
        y = Tensor.randint(num_nodes)
        data = Data(x=x, edge_index=edge_index, y=y)

        # contains
        self.assertTrue("x" in data)
        self.assertFalse("a" in data)
        # delitem
        del data["y"]
        with self.assertRaises(AttributeError):
            data["y"]
        data.y = y
        # getattr
        self.assertListEqual(data.x.tolist(), x.tolist())
        with self.assertRaises(AttributeError):
            data.a
        # getitem
        self.assertListEqual(data["x"].tolist(), x.tolist())
        with self.assertRaises(AttributeError):
            data["a"]
        # setattr
        a = Tensor.zeros(1)
        data.a = a
        self.assertListEqual(data.a.tolist(), a.tolist())
        with self.assertRaises(TypeError):
            data.a = 4
        del data.a
        # setitem
        data["a"] = a
        self.assertListEqual(data.a.tolist(), a.tolist())
        with self.assertRaises(TypeError):
            data["a"] = 4
        del data.a

    def test_hetero_data(self):
        one_node, two_node = "one", "two"
        one_num_nodes, one_num_features = 10, 16
        one_x = Tensor.rand(one_num_nodes, one_num_features)
        one_y = Tensor.randint(one_num_nodes)
        two_num_nodes, two_num_features = 5, 8
        two_x = Tensor.rand(two_num_nodes, two_num_features)
        one_edge, two_edge = ("one", "to", "two"), ("two", "one")
        one_edge_index = Tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 10, 11, 11, 12, 12, 13, 13, 14, 14]])
        two_edge_index = Tensor([[10, 11, 12, 13, 14], [0, 1, 0, 1, 0]])

        data = HeteroData()
        data[one_node].x = one_x
        data[one_node].y = one_y
        data[two_node].x = two_x
        data[one_edge].edge_index = one_edge_index
        data[two_edge].edge_index = two_edge_index

        # contains
        self.assertTrue(one_node in data)
        self.assertFalse("a" in data)
        # getitem
        self.assertListEqual(data[one_node].x.tolist(), one_x.tolist())
        self.assertListEqual(data[one_edge].edge_index.tolist(), one_edge_index.tolist())
        with self.assertRaises(TypeError):
            data[4]
        # setitem
        data["a"] = data[one_node]
        self.assertListEqual(data["a"].x.tolist(), one_x.tolist())
        with self.assertRaises(TypeError):
            data["a"] = 4
        with self.assertRaises(TypeError):
            data[4] = data[one_node]
        del data["a"]
        # node_types
        node_types = [one_node, two_node]
        self.assertListEqual(data.node_types, node_types)
        # edge_types
        edge_types = [one_edge, two_edge]
        self.assertListEqual(data.edge_types, edge_types)
        # x_dict
        self.assertListEqual(list(data.x_dict.keys()), node_types)
        self.assertIsInstance(data.x_dict[one_node], Tensor)
        # edge_index_dict
        self.assertListEqual(list(data.edge_index_dict.keys()), edge_types)
        self.assertIsInstance(data.edge_index_dict[one_edge], Tensor)
        # metadata
        self.assertTupleEqual(data.metadata(), (node_types, edge_types))


if __name__ == "__main__":
    unittest.main()
