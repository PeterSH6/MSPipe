
import torch
import logging
from typing import Iterable
from dgl.heterograph import DGLBlock
from gnnflow.utils import mfgs_to_cuda, node_to_dgl_blocks
from tgl.utils import load_graph, to_dgl_blocks

def sample(model, distributed, train_loader, sampler, queue_out):
    # logging.info('Sample Started')
    sample_time_sum = 0
    for i, (target_nodes, ts, eid) in enumerate(train_loader):
        if sampler is not None:
            model_name = type(model.module).__name__ if distributed else type(model).__name__
            if model_name == 'APAN':
                mfgs = node_to_dgl_blocks(target_nodes, ts)
                target_pos = len(target_nodes) * 2 // 3
                sampler.sample(
                    target_nodes[:target_pos], ts[:target_pos])
                ret = sampler.get_ret()
                block = to_dgl_blocks(ret, 1, reverse=True)[0][0]
            else:
                mfgs = sampler.sample(target_nodes, ts)
                sampler.sample(target_nodes, ts)
                ret = sampler.get_ret()
                mfgs = to_dgl_blocks(ret, 1)
                block = None
        else:
            mfgs = node_to_dgl_blocks(target_nodes, ts)
            block = None
        queue_out.put((mfgs, block, eid))
        # mfgs.share_memory()
        # logging.info('Sample done and send to feature fetching')
    # add signal that we are done
    queue_out.put(None)
    # logging.info('Sample in one epoch done')


def left_training(mfgs, eid, model, device, cache, distributed, optimizer, criterion):
    mfgs_to_cuda(mfgs, device)
    # logging.info('move to cuda done')
    mfgs = cache.fetch_feature(
        mfgs, eid, target_edge_features=True)  # because all use memory
    b = mfgs[0][0]  # type: DGLBlock
    if distributed:
        model.module.memory.prepare_input(b)
        # model.module.last_updated = model.module.memory_updater(b)
    else:
        model.memory.prepare_input(b)
        # model.last_updated = model.memory_updater(b)
    last_updated = model.module.memory_updater(mfgs[0][0])
    # logging.info('gnn mfgs {}'.format(mfgs))
    optimizer.zero_grad()
    pred_pos, pred_neg = model(mfgs)
    loss = criterion(pred_pos, torch.ones_like(pred_pos))
    loss += criterion(pred_neg, torch.zeros_like(pred_neg))
    # total_loss += float(loss) * num_target_nodes
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        # use one function
        if distributed:
            model.module.memory.update_mem_mail(
                **last_updated, edge_feats=cache.target_edge_features.get(),
                neg_sample_ratio=1)
        else:
            model.memory.update_mem_mail(
                **last_updated, edge_feats=cache.target_edge_features.get(),
                neg_sample_ratio=1)


def feature_fetching(cache, device, queue_in, queue_out, stream):
    # logging.info('feature fetching start')
    while True:
        # retrive from queue
        item = queue_in.get()
        # check for stop
        if item is None:
            queue_out.put(item)
            break
        global mfgs
        mfgs, block, eid = item
        # logging.info('feature mfgs {}'.format(mfgs))
        # with torch.cuda.stream(stream):
        mfgs_to_cuda(mfgs, device)
        # logging.info('move to cuda done')
        mfgs = cache.fetch_feature(
            mfgs, eid, target_edge_features=True)  # because all use memory
        queue_out.put((mfgs, block))
    #     logging.info('fetch feature done')
    # logging.info('feature fetching done')


def memory_fetching(model, distributed, queue_in, queue_out, stream):
    while True:
        # retrive from queue
        item = queue_in.get()
        # check for stop
        if item is None:
            queue_out.put(item)
            break
        mfgs, block = item
        # logging.info('memory mfgs {}'.format(mfgs))
        # with torch.cuda.stream(stream):
        b = mfgs[0][0]  # type: DGLBlock
        if distributed:
            model.module.memory.prepare_input(b)
            # model.module.last_updated = model.module.memory_updater(b)
        else:
            model.memory.prepare_input(b)
            # model.last_updated = model.memory_updater(b)

        queue_out.put((mfgs, block))

# TODO: first test GNN first and Memory Last


def gnn_training(model, optimizer, criterion, queue_in, queue_out, stream):
    # logging.info('gnn training start')
    neg_sample_ratio = 1
    while True:
        # retrive from queue
        item = queue_in.get()
        # check for stop
        if item is None:
            queue_out.put(item)
            break
        mfgs, block = item
        # with torch.cuda.stream(stream):
        last_updated = model.module.memory_updater(mfgs[0][0])
        # logging.info('gnn mfgs {}'.format(mfgs))
        optimizer.zero_grad()
        pred_pos, pred_neg = model(mfgs)
        loss = criterion(pred_pos, torch.ones_like(pred_pos))
        loss += criterion(pred_neg, torch.zeros_like(pred_neg))
        # total_loss += float(loss) * num_target_nodes
        loss.backward()
        optimizer.step()
        queue_out.put((last_updated, block))  # TODO: may not need?


def memory_update(model, distributed, cache, queue_in, stream):
    # logging.info('memory update start')
    while True:
        # retrive from queue
        item = queue_in.get()
        # check for stop
        if item is None:
            break
        last_updated, block = item
        # NB: no need to do backward here
        # with torch.cuda.stream(stream):
        with torch.no_grad():
            # use one function
            if distributed:
                model.module.memory.update_mem_mail(
                    **last_updated, edge_feats=cache.target_edge_features.get(),
                    neg_sample_ratio=1, block=block)
            else:
                model.memory.update_mem_mail(
                    **last_updated, edge_feats=cache.target_edge_features.get(),
                    neg_sample_ratio=1, block=block)
