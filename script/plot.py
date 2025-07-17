import os
import re
import pandas as pd
import matplotlib.pyplot as plt

## global para
X_LABEL, Y_LABEL = ('Recall@100', 'dis_cnt')
FLODER = 'data/sift1M'
PATTERN = r'sift1M-givfpq_M32_R(\d+)\.csv'
CSVFILE = os.path.join(FLODER, 'sift1M-givfpq_M32.png')

def get_matching_files(folder, pattern):
    matching_files = []
    try:
        if not os.path.isdir(folder):
            raise ValueError(f"Floder [{folder}] does not exist or is not a directory.")

        regex = re.compile(pattern)
        for filename in os.listdir(folder):
            if os.path.isfile(os.path.join(folder, filename)) and regex.search(filename):
                matching_files.append(os.path.join(folder, filename))

    except Exception as e:
        print(f"Error: {e}")
        exit()
    
    return matching_files

def get_plot_data(files):
    data = []
    for fname in files:
        dic = {}
        dic['method'] = fname.split('_')[0].rsplit('/')[-1]
        match_m = re.search(r'_M(\d+)', fname)
        if match_m:
            dic['M'] = match_m.group(1)
        match_r = re.search(r'_R(\d+)', fname)
        if match_r:
            dic['R'] = match_r.group(1)
        df = pd.read_csv(fname)
        dic['x'] = df[X_LABEL].tolist()
        dic['y'] = df[Y_LABEL].tolist()
        data.append(dic)
    return data

def plot_scatter(output, data_list):
    plt.figure(figsize=(10, 6))
    
    for item in data_list:
        label = f"M{item['M']}_R{item['R']}"
        x = item['x']
        y = item['y']
        plt.scatter(x, y, label=label, s=30)
        plt.plot(x, y, alpha=0.5)
    
    plt.title('Scatter Plot with Connecting Lines')
    plt.xlabel(X_LABEL)
    plt.ylabel(Y_LABEL)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output)

files = get_matching_files(FLODER, PATTERN)
plot_list = get_plot_data(files)
plot_scatter(CSVFILE, plot_list)