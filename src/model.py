import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
class GCN2(torch.nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dimension, num_classes, dropout, norm=None) -> None:
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.num_layers = num_layers
        self.dropout = dropout
        if num_layers == 1:
            self.convs.append(GCNConv(input_dim, num_classes, cached=False,
                             normalize=True))
        else:
            self.convs.append(GCNConv(input_dim, hidden_dimension, cached=False,
                             normalize=True))
            if norm:
                self.norms.append(torch.nn.BatchNorm1d(hidden_dimension))
            else:
                self.norms.append(torch.nn.Identity())

            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_dimension, hidden_dimension, cached=False,
                             normalize=True))
                if norm:
                    self.norms.append(torch.nn.BatchNorm1d(hidden_dimension))
                else:
                    self.norms.append(torch.nn.Identity())

            self.convs.append(GCNConv(hidden_dimension, num_classes, cached=False, normalize=True))

    def forward(self, data):
        x, edge_index, edge_weight= data.x, data.edge_index, data.edge_weight
        for i in range(self.num_layers):
            x = F.dropout(x, p=self.dropout, training=self.training)
            if edge_weight != None:
                x = self.convs[i](x, edge_index, edge_weight)
            else:
                x = self.convs[i](x, edge_index)
            if i != self.num_layers - 1:
                x = self.norms[i](x)
                x = F.relu(x)
        return x


def train(data,model,optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def test(data,model):
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)
    accs = []
    for mask in [data.train_mask, data.val_mask]:
        correct = pred[mask].eq(data.y[mask]).sum().item()
        accs.append(correct / mask.sum().item())
    return accs  # train_acc, val_acc, test_acc
def test_final(data,model):
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)
    accs = []
    for mask in [data.test_mask]:
        correct = pred[mask].eq(data.y[mask]).sum().item()
        accs.append(correct / mask.sum().item())
    return accs[0]  # train_acc, val_acc, test_acc
