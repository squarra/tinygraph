import unittest

from tinygraph.nn import datasets


class TestNN(unittest.TestCase):
    def test_cora(self):
        keys = {"x", "edge_index", "y", "train_mask", "val_mask", "test_mask"}
        self.assertEqual(keys, datasets.cora().keys())

    def test_reddit(self):
        keys = {"x", "edge_index", "y", "train_mask", "val_mask", "test_mask"}
        self.assertEqual(keys, datasets.reddit().keys())

    def test_dblp(self):
        node_types = ["author", "paper", "term", "conference"]
        edge_types = [
            ("author", "to", "paper"),
            ("paper", "to", "author"),
            ("paper", "to", "term"),
            ("paper", "to", "conference"),
            ("term", "to", "paper"),
            ("conference", "to", "paper"),
        ]
        self.assertEqual((node_types, edge_types), datasets.dblp().metadata())

    def test_imdb(self):
        node_types = ["movie", "director", "actor"]
        edge_types = [("movie", "to", "director"), ("movie", "to", "actor"), ("director", "to", "movie"), ("actor", "to", "movie")]
        self.assertEqual((node_types, edge_types), datasets.imdb().metadata())


if __name__ == "__main__":
    unittest.main()
