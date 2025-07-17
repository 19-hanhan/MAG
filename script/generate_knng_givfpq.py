import time
import time
import cupy as cp
from utils import *
from cuvs.neighbors import ivf_pq
from argparse import ArgumentParser

parser = ArgumentParser(description="Example script: "
                                    "\n\tpython script/generate_knng_givfpq.py --datadir /data1/data --dataset sift1M --save_k=16")

## Required parameters
parser.add_argument("--datadir", type=str, required=True, help="The root location of your dataset folder")
parser.add_argument("--dataset", type=str, required=True, help="Dataset name (Folder name must match file prefix)")
parser.add_argument("--save_k", type=int, required=True, help="Number of nearest neighbors to save in the index")

## Optional parameters
parser.add_argument("--nlist", type=int, default=1024, help="Number of clusters to search in the index")
parser.add_argument("--pq_m", type=int, default=16, help="Number of subquantizers for product quantization")
parser.add_argument("--nbit", type=int, default=8, help="Size of each subquantizer in bits")
parser.add_argument("--batch_size", type=int, default=32, help="Number of nearest neighbors to search for")

## generate para
args = parser.parse_args()

def evaluate(npts, ids, gt):
    return cp.sum(ids == gt[:, :1], axis=1).sum() / float(npts)

## load dataset
xb, xq, xt, gt = map(cp.asarray, load_data(args.datadir, args.dataset))
nq, dim = xq.shape
nb = xb.shape[0]
print("nq: [%d], dim: [%d]" % (nq, dim))

## build givfpq
t0 = time.time()
index = ivf_pq.build(ivf_pq.IndexParams(
    n_lists=args.nlist,
    metric='sqeuclidean',
    metric_arg=2.0,
    kmeans_n_iters=20,
    kmeans_trainset_fraction=0.5,
    pq_bits=args.nbit, pq_dim=args.pq_m,
    codebook_kind='subspace',
    force_random_rotation=False,
    add_data_on_build=True,
    conservative_memory_allocation=False,
    max_train_points_per_pq_code=256
), xb)
t1 = time.time()
print(f"build time : {t1 - t0:.2f} seconds")

## search in givfpq
print(f"nprobe;QPS;Recall@{args.save_k}")
for nprobe in range(args.batch_size, args.nlist // 2 + 1, args.batch_size):
    search_params = ivf_pq.SearchParams(
        n_probes=nprobe,
        lut_dtype=cp.float32,
        internal_distance_dtype=cp.float32
    )
    t0 = time.time()
    D, I = map(cp.asarray, ivf_pq.search(search_params, index, xq, args.save_k * 3))
    t1 = time.time()
    qps = round(nq / (t1 - t0), 2)
    recall = matrix_recall(I.get(), gt.get(), args.save_k)
    print("%d;%.2f;%.4f" % (nprobe, qps, recall))
    if recall > 0.95:
        t0 = time.time()
        D, I = map(cp.asarray, ivf_pq.search(search_params, index, xb, args.save_k * 3))
        save_vecs(f'data/{args.dataset}/{args.dataset}_knn{args.save_k}_givfpq.ivecs', I[:, :args.save_k].astype(cp.int32).get())
        t1 = time.time()
        print(f"search knn time : {t1 - t0:.2f} seconds")
        break
