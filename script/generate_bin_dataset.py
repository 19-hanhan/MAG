import time
import faiss
import numpy as np
from utils import *
from argparse import ArgumentParser

parser = ArgumentParser(description="Example script: "
                        "\n\tpython script/generate_bin_dataset.py --datadir /data1/data --dataset sift1M")

## Required parameters
parser.add_argument("--datadir", type=str, required=True, help="The root location of your dataset folder")
parser.add_argument("--dataset", type=str, required=True, help="Dataset name (Folder name must match file prefix)")

## generate para
args = parser.parse_args()

## load dataset
xb, xq, xt, gt = load_data(args.datadir, args.dataset)
nq, dim = xq.shape
nb = xb.shape[0]
print("nq: [%d], dim: [%d]" % (nq, dim))

def save_bin_vecs(fname, data):
    with open(fname, 'wb') as f:
        f.write(data.tobytes())

print(type(xb[0][0]))
print(type(xq[0][0]))
save_bin_vecs(f'dataset/{args.dataset}/{args.dataset}_base.fbin', xb)
save_bin_vecs(f'dataset/{args.dataset}/{args.dataset}_query.fbin', xq)

index = faiss.IndexFlatIP(dim)
index.add(xb)
t0 = time.time()
D, I = index.search(xq, 100)
t1 = time.time()
print(f"search IP gt time : {t1 - t0:.2f} seconds")
save_vecs(f'dataset/{args.dataset}/{args.dataset}_groundtruth.ivecs', I.astype(np.int32))