import time
import faiss
import numpy as np
from utils import *
from argparse import ArgumentParser

parser = ArgumentParser(description="Example script: "
                                    "\n\tpython script/generate_knng_ivfpq.py --datadir /data1/data --dataset sift1M")

## Required parameters
parser.add_argument("--datadir", type=str, required=True, help="The root location of your dataset folder")
parser.add_argument("--dataset", type=str, required=True, help="Dataset name (Folder name must match file prefix)")

## Optional parameters
parser.add_argument("--nlist", type=int, default=1024, help="Number of clusters to search in the index")
parser.add_argument("--pq_m", type=int, default=16, help="Number of subquantizers for product quantization")
parser.add_argument("--nbit", type=int, default=8, help="Size of each subquantizer in bits")
parser.add_argument("--batch_size", type=int, default=32, help="Number of nearest neighbors to search for")
parser.add_argument("--save_k", type=int, default=32, help="Number of nearest neighbors to save in the index")

## generate para
args = parser.parse_args()

## load dataset
xb, xq, xt, gt = load_data(args.datadir, args.dataset)
nq, dim = xq.shape
nb = xb.shape[0]
print("nq: [%d], dim: [%d]" % (nq, dim))

## set default para
if args.save_k is None:
    args.save_k = args.nlist
print(f"para: datadir[{args.datadir}],dataset[{args.dataset}]"
      f",nlist[{args.nlist}],nbit[{args.nbit}],pq_m[{args.pq_m}],batch_size[{args.batch_size}]")

## train data
quantizer = faiss.IndexFlatL2(dim)
index = faiss.IndexIVFPQ(quantizer, dim, args.nlist, args.pq_m, args.nbit)
index.verbose = True
t0 = time.time()
index.train(xb)
t1 = time.time()
index.add(xb)
t2 = time.time()
print(f"train time : {t1 - t0:.2f} seconds\nadd time : {t2 - t1:.2f} seconds")

## search
print(f"nprobe;QPS;Recall@{args.save_k}")
for nprobe in range(args.batch_size, args.nlist // 2 + 1, args.batch_size):
    index.nprobe = nprobe
    t0 = time.time()
    D, I = index.search(xq, args.save_k * 3)
    t1 = time.time()
    qps = round(nq / (t1 - t0), 2)
    recall = matrix_recall(I, gt, args.save_k)
    print("%d;%.2f;%.4f" % (nprobe, qps, recall))
    if recall > 0.95:
        t0 = time.time()
        D, I = index.search(xb, args.save_k * 3)
        t1 = time.time()
        print(f"search knn time : {t1 - t0:.2f} seconds")
        save_vecs(f'data/{args.dataset}/{args.dataset}_knn_ivfpq.ivecs', I[:, :args.save_k].astype(np.int32))
        break