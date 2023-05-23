import argparse
from ast import arg
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait
import logging
import math
from multiprocessing.pool import ThreadPool
from queue import Queue
from threading import Lock, Thread
import torch.multiprocessing as mp
import os
import random
import threading
import time

import GPUtil
import numpy as np
import torch
import torch.distributed
import torch.nn
import torch.nn.parallel
import torch.utils.data
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import BatchSampler, SequentialSampler
from tqdm import tqdm

import gnnflow.cache as caches
from gnnflow.config import get_default_config
from gnnflow.data import (DistributedBatchSampler, EdgePredictionDataset,
                          RandomStartBatchSampler, default_collate_ndarray)
from gnnflow.models.apan import APAN
from gnnflow.models.dgnn import DGNN
from gnnflow.models.gat import GAT
from gnnflow.models.graphsage import SAGE
from gnnflow.models.jodie import JODIE
from gnnflow.temporal_sampler import TemporalSampler
from gnnflow.utils import (DstRandEdgeSampler, EarlyStopMonitor, allocate_pinned_apan_memory_buffers, allocate_pinned_memory_buffers,
                           build_dynamic_graph, get_pinned_buffers,
                           get_project_root_dir, load_dataset, load_feat, load_most_similar,
                           mfgs_to_cuda, node_to_dgl_blocks)
from scripts.pipeline import feature_fetching, gnn_training, memory_fetching, memory_update, sample
from scripts.train import training_batch

datasets = ['REDDIT', 'GDELT', 'LASTFM', 'MAG', 'MOOC', 'WIKI']
model_names = ['TGN', 'TGAT', 'DySAT', 'GRAPHSAGE', 'GAT', 'APAN', 'JODIE']
cache_names = sorted(name for name in caches.__dict__
                     if not name.startswith("__")
                     and callable(caches.__dict__[name]))

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=model_names, required=True,
                    help="model architecture" + '|'.join(model_names))
parser.add_argument("--data", choices=datasets, required=True,
                    help="dataset:" + '|'.join(datasets))
parser.add_argument("--epoch", help="maximum training epoch",
                    type=int, default=100)
parser.add_argument("--lr", help='learning rate', type=float, default=0.0001)
parser.add_argument("--num-workers", help="num workers for dataloaders",
                    type=int, default=8)
parser.add_argument("--num-chunks", help="number of chunks for batch sampler",
                    type=int, default=8)
parser.add_argument("--print-freq", help="print frequency",
                    type=int, default=100)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--ingestion-batch-size", type=int, default=1000,
                    help="ingestion batch size")

# optimization
parser.add_argument("--cache", choices=cache_names, help="feature cache:" +
                    '|'.join(cache_names))
parser.add_argument("--edge-cache-ratio", type=float, default=0,
                    help="cache ratio for edge feature cache")
parser.add_argument("--node-cache-ratio", type=float, default=0,
                    help="cache ratio for node feature cache")
parser.add_argument("--snapshot-time-window", type=float, default=0,
                    help="time window for sampling")

args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG)
logging.info(args)

checkpoint_path = os.path.join(get_project_root_dir(),
                               '{}.pt'.format(args.model))



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(args.seed)

training = True


def gpu_load():
    global training
    time.sleep(5)
    while True:
        # stop when training is done
        # use a global variable to stop the thread
        if not training:
            break
        gpus = GPUtil.getGPUs()
        avg_load = sum([gpu.load for gpu in gpus]) / len(gpus)
        logging.info("GPU load: {:.2f}%".format(avg_load * 100))
        time.sleep(1)


def neg_sample(target_nodes, ts, nodes):
    batch_size = len(target_nodes) // 3
    target_pos = len(target_nodes) * 2 // 3
    num_nodes = len(nodes)
    target_nodes_pos = target_nodes[:target_pos]
    ts_pos = ts[:batch_size]
    dst = target_nodes[target_pos:2 * target_pos]
    neg_sample = None
    for i in range(batch_size):
        temp = np.delete(nodes, np.where(nodes == dst[i]))
        temp = np.reshape(temp, (len(temp), 1))
        if neg_sample is None:
            neg_sample = temp
        else:
            neg_sample = np.hstack([neg_sample, temp])
    num_neg = neg_sample.shape[0]
    neg_sample = neg_sample.flatten('C')
    target_nodes = np.concatenate([target_nodes_pos, neg_sample])
    ts = np.tile(ts_pos, num_neg + 2)
    return target_nodes, ts, num_neg
    


def evaluate(dataloader, sampler, model, criterion, cache, device, nodes=None, num_neg=1):
    model.eval()
    val_losses = list()
    aps = list()
    aucs_mrrs = list()

    with torch.no_grad():
        total_loss = 0
        for target_nodes, ts, eid in dataloader:
            if sampler is not None:
                model_name = type(model.module).__name__ if args.distributed else type(model).__name__
                if model_name == 'APAN':
                    mfgs = node_to_dgl_blocks(target_nodes, ts)
                    target_pos = len(target_nodes) * 2 // 3
                    block = sampler.sample(
                        target_nodes[:target_pos], ts[:target_pos], reverse=True)[0][0]
                else:
                    mfgs = sampler.sample(target_nodes, ts)
                    block = None
            else:
                mfgs = node_to_dgl_blocks(target_nodes, ts)
                block = None

            mfgs_to_cuda(mfgs, device)
            mfgs = cache.fetch_feature(
                mfgs, eid)

            if args.use_memory:
                b = mfgs[0][0]
                if args.distributed:
                    model.module.memory.prepare_input(b)
                    model.module.last_updated = model.module.memory_updater(b)
                else:
                    model.memory.prepare_input(b)
                    model.last_updated = model.memory_updater(b)

            pred_pos, pred_neg = model(mfgs, neg_samples=num_neg)

            if args.use_memory:
                # NB: no need to do backward here
                # use one function
                if args.distributed:
                    model.module.memory.update_mem_mail(
                        **model.module.last_updated, edge_feats=cache.target_edge_features.get(),
                        neg_sample_ratio=num_neg, block=block)
                else:
                    model.memory.update_mem_mail(
                        **model.last_updated, edge_feats=cache.target_edge_features.get(),
                        neg_sample_ratio=num_neg, block=block)

            total_loss += criterion(pred_pos, torch.ones_like(pred_pos))
            total_loss += criterion(pred_neg, torch.zeros_like(pred_neg))
            y_pred = torch.cat([pred_pos, pred_neg], dim=0).sigmoid().cpu()
            y_true = torch.cat(
                [torch.ones(pred_pos.size(0)),
                 torch.zeros(pred_neg.size(0))], dim=0)
            if num_neg > 1:
                aucs_mrrs.append(torch.reciprocal(torch.sum(pred_pos.squeeze() < pred_neg.squeeze().reshape(num_neg, -1), dim=0) + 1).type(torch.float))
            else:
                aucs_mrrs.append(roc_auc_score(y_true, y_pred))
            aps.append(average_precision_score(y_true, y_pred))

        val_losses.append(float(total_loss))

    ap = float(torch.tensor(aps).mean())
    if num_neg > 1:
        auc_mrr = float(torch.cat(aucs_mrrs).mean())
    else:
        auc_mrr = float(torch.tensor(aucs_mrrs).mean())
    return ap, auc_mrr



def main():
    args.distributed = int(os.environ.get('WORLD_SIZE', 0)) > 1
    if args.distributed:
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.local_world_size = int(os.environ['LOCAL_WORLD_SIZE'])
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group('nccl')
        args.rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()
    else:
        args.local_rank = args.rank = 0
        args.local_world_size = args.world_size = 1

    logging.info("rank: {}, world_size: {}".format(args.rank, args.world_size))

    model_config, data_config = get_default_config(args.model, args.data)
    model_config["snapshot_time_window"] = args.snapshot_time_window
    args.use_memory = model_config['use_memory']

    if args.distributed:
        # graph is stored in shared memory
        data_config["mem_resource_type"] = "shared"

    train_data, val_data, test_data, full_data = load_dataset(args.data)
    train_num_nodes = len(np.unique(np.concatenate(
        (train_data['src'], train_data['dst']))))
    full_num_nodes = len(np.unique(np.concatenate(
        (full_data['src'], full_data['dst']))))
    logging.info("train_num_nodes: {} full_num_nodes: {}".format(
        train_num_nodes, full_num_nodes))

    train_rand_sampler = DstRandEdgeSampler(
        np.concatenate((train_data['src'].to_numpy(dtype=np.int32),
                        train_data['dst'].to_numpy(dtype=np.int32))))
    val_rand_sampler = DstRandEdgeSampler(
        np.concatenate((full_data['src'].to_numpy(dtype=np.int32),
                        full_data['dst'].to_numpy(dtype=np.int32))))
    test_rand_sampler = DstRandEdgeSampler(
        np.concatenate((full_data['src'].to_numpy(dtype=np.int32),
                        full_data['dst'].to_numpy(dtype=np.int32))))

    train_ds = EdgePredictionDataset(train_data, train_rand_sampler)
    val_ds = EdgePredictionDataset(val_data, val_rand_sampler)
    test_ds = EdgePredictionDataset(test_data, test_rand_sampler)

    batch_size = model_config['batch_size']
    # NB: learning rate is scaled by the number of workers
    args.lr = args.lr * math.sqrt(args.world_size)
    logging.info("batch size: {}, lr: {}".format(batch_size, args.lr))
    # test_batch_size = batch_size // (args.world_size * 10)
    test_batch_size = batch_size // args.world_size
    logging.info("test batch size: {}".format(test_batch_size))

    if args.distributed:
        train_sampler = DistributedBatchSampler(
            SequentialSampler(train_ds), batch_size=batch_size,
            drop_last=False, rank=args.rank, world_size=args.world_size,
            num_chunks=args.num_chunks)
        val_sampler = DistributedBatchSampler(
            SequentialSampler(val_ds),
            batch_size=batch_size, drop_last=False, rank=args.rank,
            world_size=args.world_size)
        test_sampler = DistributedBatchSampler(
            SequentialSampler(test_ds),
            batch_size=test_batch_size, drop_last=False, rank=args.rank,
            world_size=args.world_size)
    else:
        train_sampler = RandomStartBatchSampler(
            SequentialSampler(train_ds), batch_size=batch_size, drop_last=False)
        val_sampler = BatchSampler(
            SequentialSampler(val_ds), batch_size=batch_size, drop_last=False)
        test_sampler = BatchSampler(
            SequentialSampler(test_ds),
            batch_size=batch_size, drop_last=False)

    train_loader = torch.utils.data.DataLoader(
        train_ds, sampler=train_sampler,
        collate_fn=default_collate_ndarray, num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(
        val_ds, sampler=val_sampler,
        collate_fn=default_collate_ndarray, num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(
        test_ds, sampler=test_sampler,
        collate_fn=default_collate_ndarray, num_workers=args.num_workers)

    dgraph = build_dynamic_graph(
        **data_config, device=args.local_rank)

    if args.distributed:
        torch.distributed.barrier()
    # insert in batch
    for i in tqdm(range(0, len(full_data), args.ingestion_batch_size)):
        batch = full_data[i:i + args.ingestion_batch_size]
        src_nodes = batch["src"].values.astype(np.int64)
        dst_nodes = batch["dst"].values.astype(np.int64)
        timestamps = batch["time"].values.astype(np.float32)
        eids = batch["eid"].values.astype(np.int64)
        dgraph.add_edges(src_nodes, dst_nodes, timestamps,
                         eids, add_reverse=False)
        if args.distributed:
            torch.distributed.barrier()

    num_nodes = dgraph.max_vertex_id() + 1
    num_edges = dgraph.num_edges()
    # put the features in shared memory when using distributed training
    node_feats, edge_feats = load_feat(
        args.data, shared_memory=args.distributed,
        local_rank=args.local_rank, local_world_size=args.local_world_size)

    dim_node = 0 if node_feats is None else node_feats.shape[1]
    dim_edge = 0 if edge_feats is None else edge_feats.shape[1]

    device = torch.device('cuda:{}'.format(args.local_rank))
    logging.debug("device: {}".format(device))

    if args.model == "GRAPHSAGE":
        model = SAGE(dim_node, model_config['dim_embed'])
    elif args.model == 'GAT':
        model = DGNN(dim_node, dim_edge, **model_config, num_nodes=num_nodes,
                     memory_device=device, memory_shared=args.distributed)
    elif args.model == 'APAN':
        model = APAN(dim_node, dim_edge, **model_config, num_nodes=num_nodes,
                     memory_device=device, memory_shared=args.distributed)
    elif args.model == 'JODIE':
        model = JODIE(dim_node, dim_edge, **model_config, num_nodes=num_nodes,
                      memory_device=device, memory_shared=args.distributed)
    else:
        model = DGNN(dim_node, dim_edge, **model_config, num_nodes=num_nodes,
                     memory_device=device, memory_shared=args.distributed)
    model.to(device)

    if type(model).__name__ == 'JODIE':
        sampler = None
    else:
        sampler = TemporalSampler(dgraph, **model_config)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], find_unused_parameters=True)

    # pinned_nfeat_buffs, pinned_efeat_buffs = None, None
    pinned_nfeat_buffs, pinned_efeat_buffs = get_pinned_buffers(
        model_config['fanouts'], model_config['num_snapshots'], batch_size,
        dim_node, dim_edge)

    # Cache
    cache = caches.__dict__[args.cache](args.edge_cache_ratio, args.node_cache_ratio,
                                        num_nodes, num_edges, device,
                                        node_feats, edge_feats,
                                        dim_node, dim_edge,
                                        pinned_nfeat_buffs,
                                        pinned_efeat_buffs,
                                        None,
                                        False)

    # # set pinned for memory
    if args.model == 'APAN':
        pinned_node_memory_buffs, pinned_node_memory_ts_buffs, \
        pinned_mailbox_buffs, pinned_mailbox_ts_buffs = allocate_pinned_apan_memory_buffers(
            model_config['fanouts'],
            model_config['num_snapshots'],
            batch_size, model_config['dim_memory'],
            2 * model_config['dim_memory'] + dim_edge,
            mailbox_shape=10)
    else:
        pinned_node_memory_buffs, pinned_node_memory_ts_buffs, \
        pinned_mailbox_buffs, pinned_mailbox_ts_buffs = allocate_pinned_memory_buffers(
            model_config['fanouts'],
            model_config['num_snapshots'],
            batch_size, model_config['dim_memory'],
            2 * model_config['dim_memory'] + dim_edge)
    if args.distributed:
        model.module.memory.set_pinned(pinned_node_memory_buffs, pinned_node_memory_ts_buffs, pinned_mailbox_buffs, pinned_mailbox_ts_buffs)
    else:
        model.memory.set_pinned(pinned_node_memory_buffs, pinned_node_memory_ts_buffs, pinned_mailbox_buffs, pinned_mailbox_ts_buffs)

    # only gnnlab static need to pass param
    if args.cache == 'GNNLabStaticCache':
        cache.init_cache(sampler=sampler, train_df=train_data,
                         pre_sampling_rounds=2)
    else:
        cache.init_cache()

    logging.info("cache mem size: {:.2f} MB".format(
        cache.get_mem_size() / 1000 / 1000))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    # most_similar = None
    # most_similar = load_most_similar('/home/gmsheng/repos/TGNN-Staleness/{}_most_similar_2.npy'.format(args.data))
    # most_similar = most_similar[:, 0:10]
    best_e = train(train_loader, val_loader, sampler,
                   model, optimizer, criterion, cache, device,test_loader)
    
    if args.rank == 0:
        logging.info('The best ap is reached at epoch {}...'.format(best_e))

    if args.distributed:
        torch.distributed.barrier()


sampling_time_sum = 0
iter_mem_update = 0

def train(train_loader, val_loader, sampler, model, optimizer, criterion,
          cache, device, test_loader, most_similar=None):
    global training
    best_ap = 0
    best_test_ap = 0
    best_test_auc = 0
    best_e = 0
    epoch_time_sum = 0

    # create threadpool
    num_process = 3
    stream_pool = [torch.cuda.Stream(device=device, priority=0)
                   for _ in range(num_process)]

    lock_pool = [Lock() for _ in range(5)]  # 5 stages
    signal_queue = Queue(maxsize=4)

    logging.info('Start training...')
    for e in range(args.epoch):
        model.train()
        cache.reset()
        if e > 0:
            if args.distributed:
                model.module.reset()
            else:
                model.reset()
        total_loss = 0
        cache_edge_ratio_sum = 0
        cache_node_ratio_sum = 0
        total_sampling_time = 0
        total_feature_fetch_time = 0
        total_memory_fetch_time = 0
        total_memory_update_time = 0
        total_memory_write_back_time = 0
        total_model_train_time = 0
        total_samples = 0

        pool = ThreadPool(processes=num_process)
        epoch_time_start = time.time()

    
        for i, (target_nodes, ts, eid) in enumerate(train_loader):
            pool.apply_async(training_batch, args=(model, sampler, cache, target_nodes,
                                                   ts, eid, device, args.distributed, optimizer, criterion, stream_pool[i % num_process], signal_queue, lock_pool, i, args.rank, most_similar))

        pool.close()
        pool.join()

        epoch_time = time.time() - epoch_time_start
        epoch_time_sum += epoch_time

        # logging.info("epoch time: {}".format(epoch_time))
        # if args.distributed:
        #     memory = model.module.memory.node_memory
        # else:
        #     memory = model.memory.node_memory
        
        # np.save('{}_memory_pipethread.npy'.format(args.data), memory.numpy())

        # Validation
        val_start = time.time()
        val_ap, val_auc = evaluate(
            val_loader, sampler, model, criterion, cache, device)

        ap, auc = evaluate(test_loader, sampler, model,
                           criterion, cache, device)

        if args.distributed:
            val_res = torch.tensor([val_ap, val_auc, ap, auc]).to(device)
            torch.distributed.all_reduce(val_res)
            val_res /= args.world_size
            val_ap, val_auc, ap, auc = val_res.tolist()

        val_end = time.time()
        val_time = val_end - val_start

        if args.distributed:
            metrics = torch.tensor([val_ap, val_auc, ap, auc, cache_edge_ratio_sum,
                                    cache_node_ratio_sum, total_samples,
                                    total_sampling_time, total_feature_fetch_time,
                                    total_memory_update_time,
                                    total_memory_write_back_time,
                                    total_model_train_time]).to(device)
            torch.distributed.all_reduce(metrics)
            metrics /= args.world_size
            val_ap, val_auc, ap, auc, cache_edge_ratio_sum, cache_node_ratio_sum, \
                total_samples, total_sampling_time, total_feature_fetch_time, \
                total_memory_update_time, total_memory_write_back_time, \
                total_model_train_time = metrics.tolist()

        if args.rank == 0:
            logging.info("Epoch {:d}/{:d} | Validation ap {:.4f} | Validation auc {:.4f} | Train time {:.2f} s | Validation time {:.2f} s | Train Throughput {:.2f} samples/s |".format(
                e + 1, args.epoch, val_ap, val_auc, epoch_time, val_time, total_samples * args.world_size / epoch_time))
            logging.info('Test ap:{:4f}  test auc:{:4f}'.format(ap, auc))

        if args.rank == 0 and val_ap > best_ap:
            best_e = e + 1
            best_ap = val_ap
            best_test_auc = auc
            best_test_ap = ap
            if args.distributed:
                model_to_save = model.module
            else:
                model_to_save = model
            torch.save({
                'model': model_to_save.state_dict(),
                'memory': model_to_save.memory.backup() if args.use_memory else None
            }, checkpoint_path)
            logging.info(
                "Best val AP: {:.4f} & val AUC: {:.4f}".format(val_ap, val_auc))


    if args.rank == 0:
        logging.info('Avg epoch time: {}'.format(epoch_time_sum / args.epoch))

    if args.distributed:
        torch.distributed.barrier()

    if args.local_rank == 0:
        training = False

    return best_e


if __name__ == '__main__':
    main()
