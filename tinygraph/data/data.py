from tinygrad import Tensor
from torch_geometric import data


class Storage:
    def __init__(self, **kwargs):
        for name, value in kwargs.items():
            self[name] = value

    def __contains__(self, key):
        return hasattr(self, key)

    def __delitem__(self, key):
        delattr(self, key)

    def __getattr__(self, name) -> Tensor:
        if hasattr(super(), "__getattr__"):
            return super().__getattr__(name)
        raise AttributeError(name)

    def __getitem__(self, key) -> Tensor:
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

    def __delitem__(self, key):
        if isinstance(key, str):
            if key in self._nodes:
                del self._nodes[key]
            else:
                raise KeyError
        elif isinstance(key, tuple):
            if key in self._edges:
                del self._edges[key]
            else:
                raise KeyError
        else:
            raise TypeError

    def __getitem__(self, key) -> Storage:
        if isinstance(key, str):
            if key not in self._nodes:
                self._nodes[key] = Storage()
            return self._nodes[key]
        elif isinstance(key, tuple):
            if key not in self._edges:
                self._edges[key] = Storage()
            return self._edges[key]
        raise TypeError

    def __repr__(self):
        node_str = ",\n".join([f"  {k}={{{v}}}" for k, v in self._nodes.items()])
        edge_str = ",\n".join([f"  ({', '.join(elem for elem in k)})={{{v}}}" for (k), v in self._edges.items()])
        return f"HeteroData(\n{node_str},\n{edge_str}\n)"

    def __setitem__(self, key, value):
        if not isinstance(value, Storage):
            raise TypeError
        if isinstance(key, str):
            self._nodes[key] = value
        elif isinstance(key, tuple):
            self._edges[key] = value
        else:
            raise TypeError

    @property
    def node_types(self):
        return list(self._nodes)

    @property
    def edge_types(self):
        return list(self._edges)

    @property
    def x_dict(self):
        return {node_type: self._nodes[node_type].x for node_type in self.node_types}

    @property
    def edge_index_dict(self):
        return {edge_type: self._edges[edge_type].edge_index for edge_type in self.edge_types}

    def items(self):
        return self.__dict__.items()

    def keys(self):
        return list(self._nodes) + list(self._edges)

    def metadata(self):
        return self.node_types, self.edge_types

    @staticmethod
    def from_pyg(data: data.HeteroData):
        result = HeteroData()
        for attr in data.node_types + data.edge_types:
            result[attr] = Storage(**{k: Tensor(v.numpy(), requires_grad=False) for k, v in data[attr].items()})
        return result
