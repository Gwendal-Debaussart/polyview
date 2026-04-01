# polyview

**Multi-view clustering and embedding toolkit for Python.**

`polyview` provides sklearn-compatible algorithms for datasets where each
sample is described by multiple independent feature sets (views) — such as
audio + video, text + images, or multi-sensor readings.

> Actively maintained successor to the dormant `mvlearn` library.

---

## Installation

```bash
pip install polyview
```

For development:

```bash
git clone https://github.com/gwendal-debaussart/polyview
cd polyview
pip install -e ".[dev]"
```

---

## Quick start

```python
import numpy as np
import polyview as pv

# Two views of the same 100 samples
X_audio  = np.random.rand(100, 13)
X_visual = np.random.rand(100, 512)

# Wrap in a dataset container
mvd = pv.MultiViewDataset(
    [X_audio, X_visual],
    view_names=["audio", "visual"], # optional name for user readability
)

# Co-regularized multi-view K-Means
model = pv.cluster.MultiviewKMeans(n_clusters=4, random_state=0)
labels = model.fit_predict(mvd)

fused = pv.fusion.NormalizedFusion().fit_transform(mvd)
```

---

## What's included

| Module | Contents |
|---|---|
| `polyview.datasets` | `MultiViewDataset` — typed container with save/load, split, subsetting |
| `polyview.fusion` | `ConcatFusion`, `WeightedFusion`, `NormalizedFusion` |
| `polyview.cluster` | `MultiviewKMeans`, `MultiviewCoRegSpectralClustering`, `MultiviewCoTrainSpectralClustering` |
| `polyview.embed` | *(coming)* MVMDS, MultiviewTSNE, GCCA, ... |
| `polyview.metrics` | *(coming)* consensus score, view agreement |
| `polyview.semisupervised` | *(coming)* CoTraining, label propagation |
| `polyview.viz` | *(coming)* view scatter grids, embedding plots |

All estimators are **sklearn-compatible**: they work with `GridSearchCV`,
`cross_val_score`, `Pipeline`, and `clone`.


---

## ⚖️ License

MIT License. See [LICENSE](LICENSE) for details.