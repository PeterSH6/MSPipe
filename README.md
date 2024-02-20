# MSPipe

This repository is the official implementation of MSPipe: Efficient Temporal GNN Training via Staleness-aware Pipeline

## Install

Our development environment:

- Ubuntu 20.04LTS
- g++ 9.4
- CUDA 11.3 / 11.6
- cmake 3.23

Dependencies:

- torch >= 1.10
- dgl (CUDA version) 

Compile and install the MSPipe: 

```sh
git submodule update --init --recursive
pip install -r requirements.txt
python setup.py install
```

For debug mode,

```sh
DEBUG=1 pip install -v -e .
```

Compile and install the TGL (presample version):

```sh
cd tgl
python setup_tgl.py build_ext --inplace
```

## Prepare data

```sh
cd scripts/ && ./download_data.sh
```

## Train

**MSPipe**

Training [TGN](https://arxiv.org/pdf/2006.10637v2.pdf) model on the REDDIT dataset with MSPipe on 4 GPUs.

```sh
cd scripts
./run_offline.sh TGN REDDIT 4
```

**Presample (TGL)** 

Training [TGN](https://arxiv.org/pdf/2006.10637v2.pdf) model on the REDDIT dataset with Presample on 4 GPUs.

```sh
cd tgl
./run_tgl.sh TGN REDDIT 4
```



**Distributed training**

Training TGN model on the GDELT dataset on more than 1 servers, each server is required to do the following step:

1. change the `INTERFACE` to your netcard name (can be found using`ifconfig`)
2. change the
   - `HOST_NODE_ADDR`: IP address of the host machine
   - `HOST_NODE_PORT`: The port of the host machine
   - `NNODES`: Total number of servers
   - `NPROC_PER_NODE`: The number of GPU for each servers

```sh
cd script
./run_offline_dist.sh TGN GDELT
```
