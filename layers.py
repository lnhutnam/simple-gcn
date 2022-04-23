import math
import torch


class GCNLayer(torch.nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 out_feats,
                 activation,
                 dropout,
                 bias=True):
        super(GCNLayer, self).__init__()
        self.g = g
        self.weight = torch.nn.Parameter(torch.Tensor(in_feats, out_feats))
        if dropout:
            self.dropout = torch.nn.Dropout(p=dropout)
        else:
            self.dropout = 0.

        class NodeApplyModule(torch.nn.Module):
            def __init__(self, out_feats, activation=None, bias=True):
                super(NodeApplyModule, self).__init__()
                if bias:
                    self.bias = torch.nn.Parameter(torch.Tensor(out_feats))
                else:
                    self.bias = None
                self.activation = activation
                self.reset_parameters()

            def reset_parameters(self):
                if self.bias is not None:
                    stdv = 1. / math.sqrt(self.bias.size(0))
                self.bias.data.uniform_(-stdv, stdv)

            def forward(self, nodes):
                h = nodes.data['h']
                if self.bias is not None:
                    h = h + self.bias
                if self.activation:
                    h = self.activation(h)
                return {'h': h}

        self.node_update = NodeApplyModule(out_feats, activation, bias)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, h):
        def gcn_msg(edge):
            msg = edge.src['h'] * edge.src['norm']
            return {'m': msg}

        def gcn_reduce(node):
            accum = torch.sum(node.mailbox['m'], 1) * node.data['norm']
            return {'h': accum}
        if self.dropout:
            h = self.dropout(h)
        self.g.ndata['h'] = torch.mm(h, self.weight)
        self.g.update_all(gcn_msg, gcn_reduce, self.node_update)
        h = self.g.ndata.pop('h')
        return h
