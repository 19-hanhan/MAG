import time
import cupy as cp
from utils import *
from cuvs.neighbors import brute_force
from argparse import ArgumentParser

parser = ArgumentParser(description="Example script: "
                                    "\n\tpython script/generate_knng_bruteforce.py --datadir /data1/data --dataset sift1M --save_k 32")

## Required parameters
parser.add_argument("--datadir", type=str, required=True, help="The root location of your dataset folder")
parser.add_argument("--dataset", type=str, required=True, help="Dataset name (Folder name must match file prefix)")
parser.add_argument("--save_k", type=int, required=True, help="Number of nearest neighbors to save in the index")

## generate para
args = parser.parse_args()

## load dataset
xb, xq, xt, gt = map(cp.asarray, load_data(args.datadir, args.dataset))
nq, dim = xq.shape
nb = xb.shape[0]
print("nq: [%d], dim: [%d]" % (nq, dim))

t0 = time.time()
index = brute_force.build(xb, metric="sqeuclidean")
D, I = map(cp.asarray, brute_force.search(index, xb, args.save_k))
save_vecs(f'data/{args.dataset}/{args.dataset}_knn_bruteforce.ivecs', I.astype(cp.int32).get())
t1 = time.time()
print(f"search knn time : {t1 - t0:.2f} seconds, QPS: {nb / (t1 - t0):.2f}")
