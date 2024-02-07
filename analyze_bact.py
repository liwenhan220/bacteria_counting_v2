import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
import pdb

OUTPUT_DIR = "extra_data/output_imgs_4"
MEAN_VIEW = False
def get_info(name): # Is bacteria, scale
    if name[0] == "-":
        return False, 0
    elif name[:3] == "log":
        return True, int(name[3])
    elif name[:5] == "S LOG":
        return True, int(name[5])
    elif name[:11] == "E.coli. log":
        return True, int(name[11])
    return None

image_names = np.load(OUTPUT_DIR + '/image_names.npy')
bact_counts = np.load(OUTPUT_DIR + "/bact_counts.npy")
noise_counts = np.load(OUTPUT_DIR + '/noise_counts.npy')

def get_scale(i, bact_counts, noise_counts, mode = 0, epsilon = 0):
    if mode == 0:
        return np.log10(bact_counts[i] + epsilon)
    elif mode == 1:
        return np.log10(bact_counts[i] + noise_counts[i] + epsilon)
    else:
        return np.log10(noise_counts[i] + epsilon)
    

def get_graph_info(mode = 0, pad=False):
    graph_infos = {}
    max_len = 0
    for i, image_name in enumerate(image_names):
        info = get_info(image_name) 
        if info is None:
            continue
        _, scale = info
        if scale not in graph_infos:
            graph_infos[scale] = [get_scale(i, bact_counts, noise_counts, mode)]
        else:
            graph_infos[scale].append(get_scale(i, bact_counts, noise_counts, mode))
        max_len = max(max_len, len(graph_infos[scale]))

    if pad:
        for key in graph_infos:
            while len(graph_infos[key]) < max_len:
                graph_infos[key].append(None)
    return graph_infos


def gen_scattered_points(graph_infos):
    x = []
    y = []
    for key in range(0, 9):
        if key in graph_infos:
            if MEAN_VIEW:
                x.append(key)
                y.append(np.mean(graph_infos[key]))
            else:
                x += [key] * len(graph_infos[key])
                y += graph_infos[key]
    return x, y

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13, 5))

x1, y1 = gen_scattered_points(get_graph_info(mode=2))
x2, y2 = gen_scattered_points(get_graph_info(mode=0))

# axes[0].bar(x1, np.array(y1), label="noise concentration")
# axes[0].bar(x2, np.array(y2), bottom=y1, label = "bacteria concentration")
# axes[0].legend()
x, y = gen_scattered_points(get_graph_info(mode = 1))
axes[0].set_title('Raw')
axes[0].set_xlabel('expected level')
axes[0].set_ylabel('predicted level')
axes[0].scatter(x, y)

x, y = gen_scattered_points(get_graph_info(mode = 0))
axes[1].set_title('LeNet')
axes[1].set_xlabel('expected level')
axes[1].set_ylabel('predicted level')
axes[1].scatter(x, y)


import scipy.stats
import itertools
import pandas as pd

data = get_graph_info()
results = {}

results_df = pd.DataFrame(index=data.keys(), columns=data.keys())

results_df = results_df.sort_index()

# Sort the DataFrame by column index (class names)
results_df = results_df.sort_index(axis=1)

# Iterate through all pairs of classes
for (class_1, data_1), (class_2, data_2) in itertools.combinations(data.items(), 2):
    _, p_value = scipy.stats.ttest_ind(data_1, data_2)
    p_value /= 2
    if p_value < 0.05:

        results_df.loc[class_1, class_2] = "SIG" + f", {p_value:.3f}"
        results_df.loc[class_2, class_1] = "SIG" + f", {p_value:.3f}"
    else:
        results_df.loc[class_1, class_2] = "INS" + f", {p_value:.3f}"
        results_df.loc[class_2, class_1] = "INS" + f", {p_value:.3f}"

# Diagonal elements represent a class compared with itself, which is not applicable
for class_name in data.keys():
    results_df.loc[class_name, class_name] = "N/A"

# Export to Excel
results_df.to_excel('t_test_results.xlsx', index=True)

data = get_graph_info(pad=True)
data_table = pd.DataFrame(data)
data_table.sort_index(axis=1, inplace=True)
results_df.to_excel('results/t_test_results.xlsx')
data_table.to_excel('results/data.xlsx')
    
plt.show()
plt.savefig('results/quantification.jpg')


