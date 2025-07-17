import re
import csv
import subprocess
from utils import *
from argparse import ArgumentParser

parser = ArgumentParser(description="Example script: "
                                    "\n\tpython script/test_mag.py --data_dir dataset --dataset sift1M")

## Required parameters
parser.add_argument("--data_dir", type=str, required=True, help="Dataset path")
parser.add_argument("--dataset", type=str, required=True, help="Dataset name (folder name must match file prefix)")
parser.add_argument("--dim", type=int, required=True, help="dimension of base & query")

## Optional parameters build
parser.add_argument("--knn_method", type=str, default="ivfpq", help="Method to use for knn (ivfpq, givfpq, bruteforce)")
parser.add_argument("--L", type=int, default=300, help="inital pool size")
parser.add_argument("--R", type=int, default=8, help="select degree for knng graph")
parser.add_argument("--C", type=int, default=300, help="candidate pool size")
parser.add_argument("--R_IP", type=int, default=16, help="max num of ip_neighbors")
parser.add_argument("--M", type=int, default=16, help="neighborhood size of output mag index")
parser.add_argument("--Threshold", type=int, default=8, help="ip threshold (min ipneighbors)")

## Optional parameters search
parser.add_argument("--K", type=int, default=10, help="the result size")
parser.add_argument("--at", type=int, default=1, help="the recall at")
parser.add_argument("--slow_test", action='store_true', help="Run slow test with big efSearch (default: False)")
parser.add_argument("--to_csv", action='store_true', help="Save results to csv")

## global para
search_Ls = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000]
search_Ls_big = [6000, 7000, 8000, 9000, 10000, 20000]
csvinfo = []

## generate para
args = parser.parse_args()
if args.slow_test:
    search_Ls.extend(search_Ls_big)
base_file = f'{args.data_dir}/{args.dataset}/{args.dataset}_base.fbin'
query_file = f'{args.data_dir}/{args.dataset}/{args.dataset}_query.fbin'
knng_file = f'{args.data_dir}/{args.dataset}/{args.dataset}_knn{args.M}_{args.knn_method}.ivecs'
gt_file = f'{args.data_dir}/{args.dataset}/{args.dataset}_groundtruth.ivecs'
mag_file = f'data/{args.dataset}/{args.dataset}_{args.knn_method}.mag'
result_file = f'data/{args.dataset}/{args.dataset}_result.txt'
csv_file = f'data/{args.dataset}/{args.dataset}-{args.knn_method}_M{args.M}_R{args.R}.csv'
build_command = 'cmake --build build'
run_command_index = ['build/test/test_mag', base_file, knng_file, str(args.L), str(args.R), str(args.C)]
run_command_index.extend([mag_file, "index", str(args.dim), str(args.R_IP), str(args.M), str(args.Threshold)])
run_command_index = ' '.join(run_command_index)

## run mag
print(build_command)
subprocess.run(build_command, shell=True)
print(run_command_index)
subprocess.run(run_command_index, shell=True)

## count recall
print(f"search_L;QPS;Recall@{args.at};dis_cnt")
csvinfo.append(['search_L', 'QPS', f'Recall@{args.at}', 'dis_cnt'])
for search_L in search_Ls:
    run_command_search = ['build/test/test_mag', base_file, query_file, mag_file, str(search_L), str(args.K)]
    run_command_search.extend([result_file, "search", str(args.dim)])
    run_command_search = ' '.join(run_command_search)
    context = subprocess.run(run_command_search, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout

    qps_pattern = r"QPS:\s*([-+]?\d*\.?\d+)"
    qps_match = re.search(qps_pattern, context)
    qps = float(qps_match.group(1))

    dis_cnt_pattern = r"Average metric computations:\s*([-+]?\d*\.?\d+)"
    dis_cnt_match = re.search(dis_cnt_pattern, context)
    dis_cnt = float(dis_cnt_match.group(1))

    gt = ivecs_read(gt_file)
    I = np.loadtxt(result_file, dtype=int)
    recall = matrix_recall(I, gt, args.at)

    print("%d;%.2f;%.4f;%.2f" % (search_L, qps, recall, dis_cnt))
    csvinfo.append([search_L, round(qps, 2), round(recall, 4), round(dis_cnt, 2)])
    if recall == 1.0:
        break

if args.to_csv:
    with open(csv_file, 'w', newline='') as file:
        csv.writer(file).writerows(csvinfo)