## 1. Building Instruction

### Prerequisites

- GCC 4.9+ with OpenMP
- CMake 2.8+
- Boost 1.55+
- Faiss (optional)
- Cuvs (optional)

```shell
sudo apt install libomp-dev libboost-all-dev  -y

conda install faiss-cpu -y

conda install cupy cuvs -c conda-forge -c nvidia -c rapidsai -y
```

### Compile On Linux

```shell
cmake -B build && cmake --build build -j
```

## 2. Test MAG indexing

### a. mode is index

- `DATA_PATH`: the path of the base data in `bin` format.
- `KNNG_PATH`: the path of the pre-built kNN graph in *Step 1.*.
- `L`: inital pool size.
- `R`: select degree for knng graph.
- `C`: candidate pool size.
- `INDEX_PATH`: output index path.
- `MODE`: index.
- `DIM`: dimension of dataset.
- `R_IP`: max ip_neighbors.
- `M`: neighborhood size of output index (mix index of L2 & IP).
- `T`: ip threshold.

```shell
./build/test/test_mag DATA_PATH KNNG_PATH L R C INDEX_PATH index DIM R_IP M Threshold

## example
./build/test/test_mag data/sift1M/sift1M_base.fbin data/sift1M/sift1M_knn_ivfpq.ivecs 300 8 300 data/sift1M/sift1M.mag index 128 16 16 8
```

### b. mode is search

- `DATA_PATH`: the path of the base data in `bin` format.
- `QUERY_PATH`: the path of the query data in `bin` format.
- `INDEX_PATH`: index path for serach.
- `search_L`: search pool size, the larger the better but slower (must larger than K).
- `K`: the result size.
- `RESULT_PATH`: knn result of search.
- `MODE`: search.
- `DIM`: dimension of query.

```shell
./build/test/test_mag DATA_PATH QUERY_PATH INDEX_PATH search_L K RESULT_PATH search DIM

## example
./build/test/test_mag data/sift1M/sift1M_base.fbin data/sift1M/sift1M_query.fbin data/sift1M/sift1M.mag 300 10 data/sift1M/sift1M_result.txt search 128
```

## 3. Run Test

```shell
mkdir -p data/sift1M

## generate data file
python script/generate_bin_dataset.py --datadir /data1/data --dataset sift1M
python script/generate_knng_ivfpq.py --datadir /data1/data --dataset sift1M --pq_m 32 --save_k 10
python script/generate_knng_givfpq.py --datadir /data1/data --dataset sift1M --pq_m 32 --save_k 10
python script/generate_knng_bruteforce.py --datadir /data1/data --dataset sift1M --save_k 10

## run MAG test
python script/test_mag.py --data_dir data --dataset sift1M --dim 128
```