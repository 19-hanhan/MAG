
python script/test_mag.py --data_dir data --dataset gist --dim 100 --M 16 --R 16 --R_IP 0 --Threshold 0 --knn_method givfpq --K 100 --at 100 --to_csv
python script/test_mag.py --data_dir data --dataset gist --dim 100 --M 16 --R 16 --R_IP 16 --Threshold 0 --knn_method givfpq --K 100 --at 100 --to_csv

python script/test_mag.py --data_dir data --dataset glove1M --dim 100 --M 16 --R 16 --R_IP 0 --Threshold 0 --knn_method givfpq --K 100 --at 100 --to_csv
python script/test_mag.py --data_dir data --dataset glove1M --dim 100 --M 16 --R 16 --R_IP 16 --Threshold 0 --knn_method givfpq --K 100 --at 100 --to_csv

python script/test_mag.py --data_dir data --dataset sift1M --dim 128 --M 16 --R 16 --R_IP 0 --Threshold 0 --knn_method givfpq --K 100 --at 100 --to_csv
python script/test_mag.py --data_dir data --dataset sift1M --dim 128 --M 16 --R 16 --R_IP 16 --Threshold 0 --knn_method givfpq --K 100 --at 100 --to_csv