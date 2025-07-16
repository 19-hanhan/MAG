import numpy as np
from utils import *
from argparse import ArgumentParser

parser = ArgumentParser(description="Example script: "
                                    "\n\tpython script/get_recall.py --gt_path dataset/sift1M/sift1M_groundtruth.ivecs --knn_result result/sift1M.knn --at 1")

## Required parameters
parser.add_argument("--gt_path", type=str, required=True, help="Ground Truth file path")
parser.add_argument("--knn_result", type=str, required=True, help="The root location of your dataset folder")

## Optional parameters
parser.add_argument("--at", type=int, default=1, help="Dataset name (Folder name must match file prefix)")

## generate para
args = parser.parse_args()

## load dataset
gt = ivecs_read(args.gt_path)

I = np.loadtxt(args.knn_result, dtype=int)
recall = matrix_recall(I, gt, args.at)
print(f"recall@{args.at} = {recall}")
