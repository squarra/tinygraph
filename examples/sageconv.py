from tinygrad import Tensor, TinyJit, nn

from tinygraph.nn import SAGEConv
from tinygraph.nn.datasets import cora
from tinygraph.utils import convert_mask


class Model:
    def __init__(self, in_features, hidden_features, out_features, dropout=0):
        self.conv1 = SAGEConv(in_features, hidden_features)
        self.conv2 = SAGEConv(hidden_features, out_features)
        self.dropout = dropout

    def __call__(self, x, edge_index):
        x = self.conv1(x, edge_index).relu().dropout(p=self.dropout)
        return self.conv2(x, edge_index).softmax(axis=1)


if __name__ == "__main__":
    data = cora()
    in_features = data.x.shape[1]
    out_features = len(set(data.y.tolist()))
    model = Model(in_features, 16, out_features, dropout=0.5)
    params = nn.state.get_parameters(model)
    print(f"number of parameters: {sum(p.numel() for p in params)}")
    optimizer = nn.optim.Adam(params, lr=0.01)

    train_mask, val_mask, test_mask = convert_mask(data.train_mask), convert_mask(data.val_mask), convert_mask(data.test_mask)

    @TinyJit
    @Tensor.train()
    def train():
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = out[train_mask].cross_entropy(data.y[train_mask]).backward()
        optimizer.step()
        return loss.item()

    @Tensor.test()
    def test():
        out = model(data.x, data.edge_index)
        accuracies = []
        for mask in [train_mask, val_mask, test_mask]:
            pred = out[mask].argmax(axis=1)
            accuracy = pred.eq(data.y[mask]).sum() / mask.shape[0]
            accuracies.append(accuracy.item())
        return accuracies

    for epoch in range(101):
        loss = train()
        if epoch % 10 == 0:
            train_acc, val_acc, test_acc = test()
            print(f"epoch {epoch:03d} loss {loss:.4f} train acc {train_acc:.4f} val acc {val_acc:.4f} test acc {test_acc:.4f}")
