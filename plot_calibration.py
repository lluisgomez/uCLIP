import os,sys,glob
import json
import numpy as np
import matplotlib.pyplot as plt

from metrics import compute_calibration_metrics

data = {}
results_table = {}
for fname in sys.argv[1:]:
    with open(fname) as f:
        stem = os.path.splitext(os.path.basename(fname))[0]
        parts = stem.split('_')
        if len(parts) >= 3 and parts[0] == 'metrics' and parts[1] == 'calibration':
            k = '_'.join(parts[2:])
        else:
            k = stem

        print(k)
        data[k] = json.load(f)
        results_table[k] = {}

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 3))

for method in data.keys():
  for model in data[method].keys():
    for ax,k in zip(axes.ravel(), data[method][model].keys()):
        y = np.array([c[2] for c in data[method][model][k]])[::-1]
        x = np.arange(y.shape[0])+1
        #ax.plot(d[:,0], d[:,1], label='Upper-bound', color='black', linestyle='dashed')
        ax.plot(x, y, label=method, marker='o')
        ax.set_title(k)
        ax.set_xticks(x)
        ax.set_ylabel('R@1')
        ax.set_xlabel('uncertainty level')

        spearman_corr, r_squared = compute_calibration_metrics(y,x)
        print(method,model,k,spearman_corr,r_squared)
        results_table[method][k] = (float(spearman_corr),float(r_squared),float(-spearman_corr*r_squared))


for method in results_table.keys():
    i2t = results_table[method]['text_retrieval_recall@1']
    t2i = results_table[method]['image_retrieval_recall@1']

    print(f'{method} & {i2t[0]:.2f} & {i2t[1]:.2f} & {i2t[2]:.2f} & {t2i[0]:.2f} & {t2i[1]:.2f} & {t2i[2]:.2f} \\\\')

plt.legend()
plt.show()

