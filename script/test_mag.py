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
parser.add_argument("--L", type=int, default=300, help="inital pool size")
parser.add_argument("--R", type=int, default=8, help="select degree for knng graph")
parser.add_argument("--C", type=int, default=300, help="candidate pool size")
parser.add_argument("--R_IP", type=int, default=16, help="max num of ip_neighbors")
parser.add_argument("--M", type=int, default=16, help="neighborhood size of output mag index")
parser.add_argument("--Threshold", type=int, default=8, help="ip threshold (min ipneighbors)")

## Optional parameters search
parser.add_argument("--search_L", type=int, default=300,
                    help="search pool size, the larger the better but slower (must larger than K)")
parser.add_argument("--K", type=int, default=10, help="the result size")
parser.add_argument("--at", type=int, default=1, help="the recall at")

## generate para
args = parser.parse_args()
base_file = f'{args.data_dir}/{args.dataset}/{args.dataset}_base.fbin'
query_file = f'{args.data_dir}/{args.dataset}/{args.dataset}_query.fbin'
knng_file = f'{args.data_dir}/{args.dataset}/{args.dataset}_knn.ivecs'
gt_file = f'{args.data_dir}/{args.dataset}/{args.dataset}_groundtruth.ivecs'
mag_file = f'index/{args.dataset}.mag'
result_file = f'result/{args.dataset}.knn'
build_command = 'cmake --build build'
run_command_index = ['build/test/test_mag', base_file, knng_file, str(args.L), str(args.R), str(args.C)]
run_command_index.extend([mag_file, "index", str(args.dim), str(args.R_IP), str(args.M), str(args.Threshold)])
run_command_index = ' '.join(run_command_index)
run_command_search = ['build/test/test_mag', base_file, query_file, mag_file, str(args.search_L), str(args.K)]
run_command_search.extend([result_file, "search", str(args.dim)])
run_command_search = ' '.join(run_command_search)

## run mag
print(build_command)
subprocess.run(build_command, shell=True)
print(run_command_index)
subprocess.run(run_command_index, shell=True)
print(run_command_search)
subprocess.run(run_command_search, shell=True)

## count recall
gt = ivecs_read(gt_file)
I = np.loadtxt(result_file, dtype=int)
recall = matrix_recall(I, gt, args.at)
print(f"recall@{args.at} = {recall}")