import pandas as pd
import numpy as np
from tgl.sampler_core import ParallelSampler
import torch
import dgl

def load_graph(d):
    # df = pd.read_csv('DATA/{}/edges.csv'.format(d))
    g = np.load('../data/{}/ext_full.npz'.format(d))
    return g

def to_dgl_blocks(ret, hist, reverse=False):
    mfgs = list()
    for r in ret:
        if not reverse:
            b = dgl.create_block(
                (r.col(), r.row()), num_src_nodes=r.dim_in(), num_dst_nodes=r.dim_out())
            b.srcdata['ID'] = torch.from_numpy(r.nodes())
            b.edata['dt'] = torch.from_numpy(r.dts())[b.num_dst_nodes():]
            b.srcdata['ts'] = torch.from_numpy(r.ts())
        else:
            b = dgl.create_block(
                (r.row(), r.col()), num_src_nodes=r.dim_out(), num_dst_nodes=r.dim_in())
            b.dstdata['ID'] = torch.from_numpy(r.nodes())
            b.edata['dt'] = torch.from_numpy(r.dts())[b.num_src_nodes():]
            b.dstdata['ts'] = torch.from_numpy(r.ts())
        b.edata['ID'] = torch.from_numpy(r.eid())

        mfgs.append(b)
    mfgs = list(map(list, zip(*[iter(mfgs)] * hist)))
    mfgs.reverse()
    return mfgs

