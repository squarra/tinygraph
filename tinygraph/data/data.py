from tinygrad import Tensor
from torch_geometric import data


class Storage:
    def __init__(self, **kwargs):
        for name, value in kwargs.items():
            self[name] = value

    def __contains__(self, key):
        return hasattr(self, key)

    def __getitem__(self, key):
        if not hasattr(self, key):
            raise KeyError(key)
        return getattr(self, key)

    def __repr__(self):
        return ", ".join(f"{k}={list(v.shape)}" for k, v in self.items())

    def __setattr__(self, name, value):
        if not isinstance(value, Tensor):
            raise TypeError(f"Value for '{name}' must be a Tensor, got {type(value).__name__}")
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def items(self):
        return self.__dict__.items()

    def keys(self):
        return list(self.__dict__.keys())


class Data(Storage):
    def __repr__(self):
        return f"Data({super().__repr__()})"

    @staticmethod
    def from_pyg(data: data.Data):
        return Data(**{key: Tensor(value.numpy(), requires_grad=False) for key, value in data.items()})


class HeteroData:
    def __init__(self):
        self._nodes = {}
        self._edges = {}

    def __contains__(self, key):
        return key in self.keys()

    def __getitem__(self, key):
        if isinstance(key, str):
            if key not in self._nodes:
                self._nodes[key] = Storage()
            return self._nodes[key]
        elif isinstance(key, tuple):
            if key not in self._edges:
                self._edges[key] = Storage()
            return self._edges[key]
        raise NotImplementedError

    def __repr__(self):
        node_str = ",\n".join([f"  {k}={{{v}}}" for k, v in self._nodes.items()])
        edge_str = ",\n".join([f"  ({', '.join(elem for elem in k)})={{{v}}}" for (k), v in self._edges.items()])
        return f"HeteroData(\n{node_str},\n{edge_str}\n)"

    def __setitem__(self, key, value):
        if isinstance(key, str):
            self._nodes[key] = value
        elif isinstance(key, tuple):
            self._edges[key] = value
        else:
            raise NotImplementedError

    @property
    def node_types(self):
        return list(self._nodes)

    @property
    def edge_types(self):
        return list(self._edges)

    def items(self):
        return self.__dict__.items()

    def keys(self):
        return list(self._nodes) + list(self._edges)

    def metadata(self):
        return self.node_types, self.edge_types

    @staticmethod
    def from_pyg(data: data.HeteroData):
        result = HeteroData()

        for node_type in data.node_types:
            kwargs = {}
            for key in data[node_type]:
                value = data[node_type][key]
                if key == "num_nodes" and isinstance(value, int):
                    kwargs["x"] = Tensor.arange(value)
                else:
                    kwargs[key] = Tensor(value.numpy(), requires_grad=False)
            result[node_type] = Storage(**kwargs)

        for edge_type in data.edge_types:
            kwargs = {}
            edge_dict = data[edge_type]
            for key in edge_dict:
                kwargs[key] = Tensor(edge_dict[key].numpy(), requires_grad=False)
            result[edge_type] = Storage(**kwargs)

        return result
