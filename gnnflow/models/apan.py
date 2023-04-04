from pkgutil import ImpImporter
import torch

from gnnflow.models.modules.layers import TransfomerAttentionLayer, IdentityNormLayer, EdgePredictor
from gnnflow.models.modules.memory import Memory
from gnnflow.models.modules.memory_updater import TransformerMemoryUpdater
from .base import Model


class APAN(Model):

    def __init__(self, dim_node, dim_edge, num_nodes, sample_history=1,
                 memory_dim_out=100, memory_dim_time=100, layer=1,
                 gnn_attn_head=2, dropout=0.1, attn_dropout=0.1,
                 mailbox_size=10, mail_combine='last',
                 deliver_to_neighbors=True):
        super(APAN, self).__init__()
        self.dim_node = dim_node
        self.dim_node_input = dim_node
        self.dim_edge = dim_edge

        self.sample_history = sample_history
        self.memory_dim_out = memory_dim_out
        self.memory_dim_time = memory_dim_time
        self.gnn_dim_out = memory_dim_out
        self.gnn_dim_time = memory_dim_time
        self.gnn_attn_head = gnn_attn_head
        self.gnn_layer = layer
        self.dropout = dropout
        self.attn_dropout = attn_dropout

        # TODO:
        self.memory = Memory(num_nodes, dim_edge, dim_memory,
                             memory_device, memory_shared,
                             kvstore_client)

        # Memory updater
        self.memory_updater = TransformerMemoryUpdater(
            mailbox_size, gnn_attn_head,
            2 * memory_dim_out + dim_edge,
            memory_dim_out,
            memory_dim_time,
            dropout, attn_dropout)

        self.dim_node_input = memory_dim_out

        self.layers = torch.nn.ModuleDict()

        self.gnn_layer = 1
        for h in range(sample_history):
            self.layers['l0h' +
                        str(h)] = IdentityNormLayer(self.dim_node_input)

        self.last_updated = None
        self.edge_predictor = EdgePredictor(memory_dim_out)

    def forward(self, mfgs, neg_samples=1):
        super().forward(mfgs)
        out = list()
        for l in range(self.gnn_layer):
            for h in range(self.sample_history):
                rst = self.layers['l' + str(l) + 'h' + str(h)](mfgs[l][h])
                if l != self.gnn_layer - 1:
                    mfgs[l + 1][h].srcdata['h'] = rst
                else:
                    out.append(rst)

        if self.sample_history == 1:
            out = out[0]
        else:
            out = torch.stack(out, dim=0)
            out = self.combiner(out)[0][-1, :, :]
        return self.edge_predictor(out, neg_samples=neg_samples)

    def get_emb(self, mfgs):
        self.memory_updater(mfgs[0])
        out = list()
        for l in range(self.gnn_layer):
            for h in range(self.sample_history):
                rst = self.layers['l' + str(l) + 'h' + str(h)](mfgs[l][h])
                if l != self.gnn_layer - 1:
                    mfgs[l + 1][h].srcdata['h'] = rst
                else:
                    out.append(rst)
        if self.sample_history == 1:
            out = out[0]
        else:
            out = torch.stack(out, dim=0)
            out = self.combiner(out)[0][-1, :, :]
        return out
