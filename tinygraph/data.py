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
        return super().__getattr__(self, name)

    def __getitem__(self, key) -> Tensor:
        return getattr(self, key)

    def __repr__(self):
        return "{" + ", ".join(f"{k}={list(v.shape)}" for k, v in self.__dict__.items()) + "}"

    def __setattr__(self, name, value):
        if not isinstance(value, Tensor):
            raise TypeError(f"Value for '{name}' must be a Tensor, got {type(value).__name__}")
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        setattr(self, key, value)


class Data(Storage):
    def __repr__(self):
        return f"Data({super().__repr__()})"

    def items(self):
        return self.__dict__.items()

    def keys(self):
        return self.__dict__.keys()

    @staticmethod
    def from_pyg(data: data.Data):
        return Data(**{k: Tensor(v.numpy(), requires_grad=False) for k, v in data.items()})


class HeteroData:
    def __init__(self):
        self.nodes = {}
        self.edges = {}

    def __contains__(self, key):
        return key in self.keys()

    def __delitem__(self, key):
        if isinstance(key, str):
            del self.nodes[key]
        elif isinstance(key, tuple):
            del self.edges[key]
        else:
            raise TypeError

    def __getitem__(self, key) -> Storage:
        if isinstance(key, str):
            if key not in self.nodes:
                self.nodes[key] = Storage()
            return self.nodes[key]
        elif isinstance(key, tuple):
            if key not in self.edges:
                self.edges[key] = Storage()
            return self.edges[key]
        raise TypeError

    def __repr__(self):
        node_str = ",\n".join([f"  {k}={v}" for k, v in self.nodes.items()])
        edge_str = ",\n".join([f"  ({', '.join(elem for elem in k)})={v}" for k, v in self.edges.items()])
        return f"HeteroData(\n{node_str},\n{edge_str}\n)"

    def __setitem__(self, key, value):
        if not isinstance(value, Storage):
            raise TypeError
        if isinstance(key, str):
            self.nodes[key] = value
        elif isinstance(key, tuple):
            self.edges[key] = value
        else:
            raise TypeError

    @property
    def node_types(self):
        return list(self.nodes)

    @property
    def edge_types(self):
        return list(self.edges)

    @property
    def x_dict(self):
        return {node_type: self.nodes[node_type].x for node_type in self.node_types}

    @property
    def edge_index_dict(self):
        return {edge_type: self.edges[edge_type].edge_index for edge_type in self.edge_types}

    def items(self):
        return {**self.nodes, **self.edges}.items()

    def keys(self):
        return {**self.nodes, **self.edges}.keys()

    def metadata(self):
        return self.node_types, self.edge_types

    @staticmethod
    def from_pyg(data: data.HeteroData):
        result = HeteroData()
        for attr in data.node_types + data.edge_types:
            result[attr] = Storage(**{k: Tensor(v.numpy(), requires_grad=False) for k, v in data[attr].items()})
        return result
