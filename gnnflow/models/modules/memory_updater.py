"""
This code is based on the implementation of TGL's memory module.

Implementation at:
    https://github.com/amazon-research/tgl/blob/main/memorys.py
"""
import logging
import os
import torch
from dgl.heterograph import DGLBlock

from gnnflow.models.modules.layers import TimeEncode


class GRUMemeoryUpdater(torch.nn.Module):
    """
    GRU memory updater proposed by TGN
    """

    def __init__(self, dim_node: int, dim_edge: int, dim_time: int,
                 dim_embed: int, dim_memory: int):
        """
        Args:
            dim_node: dimension of node features/embeddings
            dim_edge: dimension of edge features
            dim_time: dimension of time features
            dim_embed: dimension of output embeddings
            dim_memory: dimension of memory
        """
        super(GRUMemeoryUpdater, self).__init__()
        self.dim_message = 2 * dim_memory + dim_edge
        self.dim_node = dim_node
        self.dim_time = dim_time
        self.dim_embed = dim_embed
        self.updater = torch.nn.GRUCell(
            self.dim_message + self.dim_time, dim_memory)

        self.use_time_enc = dim_time > 0
        if self.use_time_enc:
            self.time_enc = TimeEncode(dim_time)

        if dim_node > 0 and dim_node != dim_memory:
            self.node_feat_proj = torch.nn.Linear(dim_node, dim_memory)

    def forward(self, b: DGLBlock):
        """
        Update the memory of nodes

        Args:
           b: sampled message flow graph (mfg), where
                `b.num_dst_nodes()` is the number of target nodes to sample,
                `b.srcdata['ID']` is the node IDs of all nodes, and
                `b.srcdata['ts']` is the timestamp of all nodes.

        Return:
            last_updated: {
                "last_updated_nid": node IDs of the target nodes
                "last_updated_memory": updated memory of the target nodes
                "last_updated_ts": timestamp of the target nodes
            }
        """
        device = b.device

        if self.use_time_enc:
            time_feat = self.time_enc(b.srcdata['ts'] - b.srcdata['mem_ts'])
            b.srcdata['mem_input'] = torch.cat(
                [b.srcdata['mem_input'], time_feat], dim=1)

        updated_memory = self.updater(
            b.srcdata['mem_input'], b.srcdata['mem'])

        # if int(os.environ['LOCAL_RANK']) == 0:
        #     logging.info('mem input: {}'.format(b.srcdata['mem_input']))
        #     logging.info('mem : {}'.format(b.srcdata['mem']))
        #     logging.info('updated_memory: {}'.format(updated_memory))
        #     for name, param in self.updater.named_parameters():
        #         logging.info("name: {} param: {}".format(name, param[0]))

        num_dst_nodes = b.num_dst_nodes()
        last_updated_nid = b.srcdata['ID'][:num_dst_nodes].clone(
        ).detach().to(device)
        last_updated_memory = updated_memory[:num_dst_nodes].clone(
        ).detach().to(device)
        last_updated_ts = b.srcdata['ts'][:num_dst_nodes].clone(
        ).detach().to(device)

        if self.dim_node > 0:
            if self.dim_node == self.dim_embed:
                b.srcdata['h'] += updated_memory
            else:
                b.srcdata['h'] = updated_memory + \
                    self.node_feat_proj(b.srcdata['h'])
        else:
            b.srcdata['h'] = updated_memory

        return {
            "last_updated_nid": last_updated_nid,
            "last_updated_memory": last_updated_memory,
            "last_updated_ts": last_updated_ts
        }


class RNNMemeoryUpdater(torch.nn.Module):
    """
    RNN memory updater proposed by JODIE
    """

    def __init__(self, dim_node: int, dim_edge: int, dim_time: int,
                 dim_embed: int, dim_memory: int):
        """
        Args:
            dim_node: dimension of node features/embeddings
            dim_edge: dimension of edge features
            dim_time: dimension of time features
            dim_embed: dimension of output embeddings
            dim_memory: dimension of memory
        """
        super(RNNMemeoryUpdater, self).__init__()
        self.dim_message = 2 * dim_memory + dim_edge
        self.dim_node = dim_node
        self.dim_time = dim_time
        self.dim_embed = dim_embed
        self.updater = torch.nn.RNNCell(
            self.dim_message + self.dim_time, dim_memory)

        self.use_time_enc = dim_time > 0
        if self.use_time_enc:
            self.time_enc = TimeEncode(dim_time)

        if dim_node > 0 and dim_node != dim_memory:
            self.node_feat_proj = torch.nn.Linear(dim_node, dim_memory)

    def forward(self, b: DGLBlock):
        """
        Update the memory of nodes

        Args:
           b: sampled message flow graph (mfg), where
                `b.num_dst_nodes()` is the number of target nodes to sample,
                `b.srcdata['ID']` is the node IDs of all nodes, and
                `b.srcdata['ts']` is the timestamp of all nodes.

        Return:
            last_updated: {
                "last_updated_nid": node IDs of the target nodes
                "last_updated_memory": updated memory of the target nodes
                "last_updated_ts": timestamp of the target nodes
            }
        """
        device = b.device

        if self.use_time_enc:
            time_feat = self.time_enc(b.srcdata['ts'] - b.srcdata['mem_ts'])
            b.srcdata['mem_input'] = torch.cat(
                [b.srcdata['mem_input'], time_feat], dim=1)

        updated_memory = self.updater(
            b.srcdata['mem_input'], b.srcdata['mem'])

        # if int(os.environ['LOCAL_RANK']) == 0:
        #     logging.info('mem input: {}'.format(b.srcdata['mem_input']))
        #     logging.info('mem : {}'.format(b.srcdata['mem']))
        #     logging.info('updated_memory: {}'.format(updated_memory))
        #     for name, param in self.updater.named_parameters():
        #         logging.info("name: {} param: {}".format(name, param[0]))

        num_dst_nodes = b.num_dst_nodes()
        last_updated_nid = b.srcdata['ID'][:num_dst_nodes].clone(
        ).detach().to(device)
        last_updated_memory = updated_memory[:num_dst_nodes].clone(
        ).detach().to(device)
        last_updated_ts = b.srcdata['ts'][:num_dst_nodes].clone(
        ).detach().to(device)

        if self.dim_node > 0:
            if self.dim_node == self.dim_embed:
                b.srcdata['h'] += updated_memory
            else:
                b.srcdata['h'] = updated_memory + \
                    self.node_feat_proj(b.srcdata['h'])
        else:
            b.srcdata['h'] = updated_memory

        return {
            "last_updated_nid": last_updated_nid,
            "last_updated_memory": last_updated_memory,
            "last_updated_ts": last_updated_ts
        }


class TransformerMemoryUpdater(torch.nn.Module):

    def __init__(self, mailbox_size, att_head, dim_in, dim_out, dim_time, dropout, att_dropout):
        super(TransformerMemoryUpdater, self).__init__()
        self.mailbox_size = mailbox_size
        self.dim_time = dim_time
        self.att_h = att_head
        if dim_time > 0:
            self.time_enc = TimeEncode(dim_time)
        self.w_q = torch.nn.Linear(dim_out, dim_out)
        self.w_k = torch.nn.Linear(dim_in + dim_time, dim_out)
        self.w_v = torch.nn.Linear(dim_in + dim_time, dim_out)
        self.att_act = torch.nn.LeakyReLU(0.2)
        self.layer_norm = torch.nn.LayerNorm(dim_out)
        self.mlp = torch.nn.Linear(dim_out, dim_out)
        self.dropout = torch.nn.Dropout(dropout)
        self.att_dropout = torch.nn.Dropout(att_dropout)
        self.last_updated_memory = None
        self.last_updated_ts = None
        self.last_updated_nid = None

    def forward(self, b):
        # for b in mfg:
        Q = self.w_q(b.srcdata['mem']).reshape(
            (b.num_src_nodes(), self.att_h, -1))
        # logging.info("b.srcdata['mem_input'] {}".format(b.srcdata['mem_input'].shape))
        mails = b.srcdata['mem_input'].reshape(
            (b.num_src_nodes(), self.mailbox_size, -1))
        if self.dim_time > 0:
            time_feat = self.time_enc(b.srcdata['ts'][:, None] - b.srcdata['mail_ts']).reshape(
                (b.num_src_nodes(), self.mailbox_size, -1))
            mails = torch.cat([mails, time_feat], dim=2)
        K = self.w_k(mails).reshape(
            (b.num_src_nodes(), self.mailbox_size, self.att_h, -1))
        V = self.w_v(mails).reshape(
            (b.num_src_nodes(), self.mailbox_size, self.att_h, -1))
        att = self.att_act((Q[:, None, :, :]*K).sum(dim=3))
        att = torch.nn.functional.softmax(att, dim=1)
        att = self.att_dropout(att)
        rst = (att[:, :, :, None]*V).sum(dim=1)
        rst = rst.reshape((rst.shape[0], -1))
        rst += b.srcdata['mem']
        rst = self.layer_norm(rst)
        rst = self.mlp(rst)
        rst = self.dropout(rst)
        rst = torch.nn.functional.relu(rst)
        b.srcdata['h'] = rst
        self.last_updated_memory = rst.detach().clone()
        self.last_updated_nid = b.srcdata['ID'].detach().clone()
        self.last_updated_ts = b.srcdata['ts'].detach().clone()

        return {
            "last_updated_nid": self.last_updated_nid,
            "last_updated_memory": self.last_updated_memory,
            "last_updated_ts": self.last_updated_ts
        }
