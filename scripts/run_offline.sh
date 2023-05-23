#!/bin/bash

MODEL=$1
DATA=$2
NPROC_PER_NODE=${3:-1}
CACHE="${4:-LRUCache}"
EDGE_CACHE_RATIO="${5:-0}" # default 0% of cache
NODE_CACHE_RATIO="${6:-0}" # default 0% of cache
TIME_WINDOW="${7:-0}" # default 0

if [[ $NPROC_PER_NODE -gt 1 ]]; then
    cmd="torchrun \
        --nnodes=1 --nproc_per_node=$NPROC_PER_NODE \
        --standalone \
        offline_edge_prediction_pipethread.py --model $MODEL --data $DATA \
        --cache $CACHE --edge-cache-ratio $EDGE_CACHE_RATIO \
        --node-cache-ratio $NODE_CACHE_RATIO --snapshot-time-window $TIME_WINDOW \
        --ingestion-batch-size 10000000"
else
    cmd="python offline_edge_prediction_pipethread.py --model $MODEL --data $DATA \
        --cache $CACHE --edge-cache-ratio $EDGE_CACHE_RATIO \
        --node-cache-ratio $NODE_CACHE_RATIO --snapshot-time-window $TIME_WINDOW \
        --ingestion-batch-size 10000000"
fi

echo $cmd
OMP_NUM_THREADS=8 exec $cmd > ${MODEL}_${DATA}_${NPROC_PER_NODE}_MSPipe.log 2>&1
