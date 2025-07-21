import csv
import time
import faiss
import numpy as np
from utils import *
from argparse import ArgumentParser

parser = ArgumentParser(description="Example script: "
                        "\n\tpython script/test_faiss.py --datadir /data1/data --dataset sift1M --m 8 --query_k 100 --at 100 --to_csv")

## Required parameters
parser.add_argument("--datadir", type=str, required=True, help="The root location of your dataset folder")
parser.add_argument("--dataset", type=str, required=True, help="Dataset name (Folder name must match file prefix)")

## Optional parameters
parser.add_argument("--m", type=int, default=5, help="Neighborhood size for building the graph (default: 5)")
parser.add_argument("--efc", type=int, default=300, help="EfConstruction for building the graph (default: 300)")
parser.add_argument("--query_k", type=int, default=10, help="Number of results to query, caculate recall_k@1 (default: 10)")
parser.add_argument("--at", type=int, default=1, help="the recall at")
parser.add_argument("--slow_test", action='store_true', help="Run slow test with big efSearch (default: False)")
parser.add_argument("--use_ls", action='store_true', help="Use local scaling distance for build (default: False)")
parser.add_argument("--to_csv", action='store_true', help="Save results to csv")

## global
efs = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000]
efs_big = [6000, 7000, 8000, 9000, 10000, 20000]
csvinfo = []

## generate para
args = parser.parse_args()
hnsw_stats = faiss.cvar.hnsw_stats
if args.slow_test:
    efs.extend(efs_big)
csv_file = f'data/{args.dataset}/{args.dataset}-hnswip_M{2 * args.m}.csv'
print(f"para: datadir[{args.datadir}],dataset[{args.dataset}],m[{args.m}],efc[{args.efc}],query_k[{args.query_k}],at[{args.at}]]")

## load dataset
xb, xq, xt, gt = load_data(args.datadir, args.dataset)
gt = ivecs_read(f'data/{args.dataset}/{args.dataset}_inner_product_groundtruth.ivecs')
nq, d = xq.shape
nb = xb.shape[0]
print("nq: [%d], d: [%d]" % (nq, d))


## set build para
index = faiss.IndexHNSWFlat(d, args.m, faiss.METRIC_INNER_PRODUCT)
index.verbose = True
index.hnsw.efConstruction = args.efc

## build index
t0 = time.time()
index.add(xb)
t1 = time.time()
print("Graph construction time(s)", t1 - t0)

## search hnsw index
print(f"efsearch;QPS;Recall@{args.at};dis_cnt")
csvinfo.append(['efsearch', 'QPS', f'Recall@{args.at}', 'dis_cnt'])
for efSearch in efs:
    index.hnsw.efSearch = efSearch
    hnsw_stats.reset()
    t0 = time.time()
    D, I = index.search(xq, args.query_k)
    t1 = time.time()
    qps = round(nq/(t1-t0), 2)
    recall = matrix_recall(I, gt, args.at)
    avg_dist = hnsw_stats.ndis // nq
    print("%d;%.2f;%.4f;%d" % (efSearch, qps, recall, avg_dist))
    csvinfo.append([efSearch, round(qps, 2), round(recall, 4), avg_dist])
    if recall == 1.0:
        break

if args.to_csv:
    with open(csv_file, 'w', newline='') as file:
        csv.writer(file).writerows(csvinfo)