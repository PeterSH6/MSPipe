from typing import Dict, Optional, Union
import torch
from gnnflow.distributed.kvstore import KVStoreClient

from gnnflow.models.modules.layers import TransfomerAttentionLayer, IdentityNormLayer, EdgePredictor
from gnnflow.models.modules.apan_memory import Memory
from gnnflow.models.modules.memory_updater import TransformerMemoryUpdater


class APAN(torch.nn.Module):

    def __init__(self, dim_node, dim_edge, dim_time=100,
                 dim_embed=100, num_layers=1, num_snapshots=1,
                 att_head=2, dropout=0.1, att_dropout=0.1,
                 use_memory=True, dim_memory=100,
                 num_nodes: Optional[int] = None,
                 memory_device: Union[torch.device, str] = 'cpu',
                 memory_shared: bool = False,
                 kvstore_client: Optional[KVStoreClient] = None,
                 mailbox_size=10,
                 *args, **kwargs
                 ):
        super(APAN, self).__init__()
        self.dim_node = dim_node
        self.dim_node_input = dim_node
        self.dim_edge = dim_edge

        self.dim_time = dim_time
        self.dim_embed = dim_embed
        self.num_layers = num_layers
        self.num_snapshots = num_snapshots
        self.att_head = att_head
        self.dropout = dropout
        self.att_dropout = att_dropout
        self.use_memory = use_memory

        self.gnn_layer = num_layers
        self.dropout = dropout
        self.attn_dropout = att_dropout

        self.mailbox_size = mailbox_size

        # TODO:....
        self.memory = Memory(num_nodes, dim_edge, dim_memory,
                             memory_device, memory_shared, mailbox_size, kvstore_client)

        # Memory updater
        self.memory_updater = TransformerMemoryUpdater(
            mailbox_size, att_head,
            2 * dim_memory + dim_edge,
            dim_memory,
            dim_time,
            dropout, att_dropout)

        self.dim_node_input = dim_memory

        self.layers = torch.nn.ModuleDict()

        self.gnn_layer = 1
        for h in range(num_snapshots):
            self.layers['l0h' +
                        str(h)] = IdentityNormLayer(self.dim_node_input)

        self.last_updated = None
        self.edge_predictor = EdgePredictor(dim_memory)

    def forward(self, mfgs, neg_samples=1):
        out = list()
        for l in range(self.gnn_layer):
            for h in range(self.num_snapshots):
                rst = self.layers['l' + str(l) + 'h' + str(h)](mfgs[l][h])
                if l != self.gnn_layer - 1:
                    mfgs[l + 1][h].srcdata['h'] = rst
                else:
                    out.append(rst)

        if self.num_snapshots == 1:
            out = out[0]
        else:
            out = torch.stack(out, dim=0)
            out = self.combiner(out)[0][-1, :, :]
        return self.edge_predictor(out)

    def get_emb(self, mfgs):
        self.memory_updater(mfgs[0])
        out = list()
        for l in range(self.gnn_layer):
            for h in range(self.num_snapshots):
                rst = self.layers['l' + str(l) + 'h' + str(h)](mfgs[l][h])
                if l != self.gnn_layer - 1:
                    mfgs[l + 1][h].srcdata['h'] = rst
                else:
                    out.append(rst)
        if self.num_snapshots == 1:
            out = out[0]
        else:
            out = torch.stack(out, dim=0)
            out = self.combiner(out)[0][-1, :, :]
        return out

    def reset(self):
        if self.use_memory:
            self.memory.reset()

    def resize(self, num_nodes: int):
        if self.use_memory:
            self.memory.resize(num_nodes)

    def has_memory(self):
        return self.use_memory

    def backup_memory(self) -> Dict:
        if self.use_memory:
            return self.memory.backup()
        return {}

    def restore_memory(self, backup: Dict):
        if self.use_memory:
            self.memory.restore(backup)
