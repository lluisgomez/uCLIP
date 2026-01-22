import os,sys,glob
import json
import numpy as np
import matplotlib.pyplot as plt

title_dic = {'single_adversarial':'Adversarial',
             'single_adversarial_lin': 'Adversarial lin.',
             'single_top1similarity': 'Top1similarity',
             'montecarlo_adversarial': 'Adversarial (MCD)',
             'montecarlo_adversarial_lin': 'Adversarial lin. (MCD)',
             'montecarlo_top1consistency': 'Top1consistency (MCD)',
             'montecarlo_top1similarity': 'Top1similarity (MCD)',
             'ensemble_adversarial': 'Adversarial (Ens.)',
             'ensemble_adversarial_lin': 'Adversarial lin. (Ens.)',
             'ensemble_top1consistency': 'Top1consistency (Ens.)',
             'ensemble_top1similarity': 'Top1similarity (Ens.)'}

data = {}
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

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

aux = {}
for ax in axes.ravel():
    aux[ax] = []

for method in data.keys():
  for model in data[method].keys():
    for ax,k in zip(axes.ravel(), data[method][model].keys()):
        d = np.array(data[method][model][k])
        print(d[:,0])
        # Compute the area under the curve using the trapezoidal rule
        area = np.trapz(d[:,1], x=d[:,0]/np.max(d[:,0]))
        if 'oracle' in method:
            label = f'Upper-bound ({area:0.2f})'
            line_handle, = ax.plot(d[:,0], d[:,1], label=label, color='black', linestyle='dashed')
        else:
            label = f'{title_dic[method]} ({area:0.2f})'
            line_handle, = ax.plot(d[:,0], d[:,1], label=label)

        aux[ax].append((line_handle, label, area))
        ax.set_title(k)

for ax in axes.ravel():
    sorted_data = sorted(aux[ax], key=lambda tup: tup[2], reverse=True)
    sorted_handles = [tup[0] for tup in sorted_data]
    sorted_labels = [tup[1] for tup in sorted_data]
    ax.legend(sorted_handles, sorted_labels)

plt.tight_layout()
plt.show()
