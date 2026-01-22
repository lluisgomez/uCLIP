# Uncertainty-Aware Cross-Modal Retrieval with CLIP

This repository provides code to reproduce the main experimental results of the paper:

**Uncertainty-Aware Cross-Modal Retrieval with CLIP**  
*UAI 2023*

---

## Running the experiments

The evaluation scripts use *open_clip_pytorch* models on MSCOCO and Flickr30K standard splits from HF *datasets* Python module.

```
python eval_mscoco.py
```

```
python eval_mscoco.py
```

Each script:

- saves CLIP embeddings in `results/` folder. 
- prints retrieval metrics and calibration summaries (Table 1 in the paper) to stdout.
- saves calibration and rejection curves as JSON files in `results/`

To generate figures from the paper:

```
python plot_calibration.py results/metrics_calibration_*.json
python plot_rejection.py results/metrics_rejection_*.json
```

---

## Citation

If you use this code, please cite the paper.

```
@inproceedings{gomezover,
  title={Over the Top-1: Uncertainty-Aware Cross-Modal Retrieval with CLIP},
  author={Gomez, Lluis},
  booktitle={The 41st Conference on Uncertainty in Artificial Intelligence}
}
```


