from tinygrad import Tensor
from torch_geometric import data


class Data:
    def __init__(self, x: Tensor, edge_index: Tensor, y: Tensor, **kwargs):
        self.x = x
        self.edge_index = edge_index
        self.y = y

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        info = ", ".join(f"{k}={list(v.shape)}" for k, v in self.__dict__.items())
        return f"Data({info})"

    def to(self, device):
        for v in self.__dict__.values():
            print(v)
            if isinstance(v, Tensor):
                v = v.to(device)
            else:
                raise ValueError(f"{v} is not a tensor")
        return self

    @staticmethod
    def from_pyg(data: data.Data):
        return Data(**{key: Tensor(data[key].numpy(), requires_grad=False) for key in data.keys()})
