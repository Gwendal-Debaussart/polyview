# polyview

**Multi-view clustering and embedding toolkit for Python.**

[PyPI project page](https://pypi.org/project/polyview/) · [Documentation](https://gwendal-debaussart.github.io/polyview/index.html/)

`polyview` provides sklearn-compatible algorithms for datasets where each
sample is described by multiple independent feature sets (views) — such as
audio + video, text + images, or multi-sensor readings.

> Actively maintained successor to the dormant `mvlearn` library.

---

## Installation

Install from PyPI:

```bash
pip install polyview
```

Project page: https://pypi.org/project/polyview/

For development:

```bash
git clone https://github.com/gwendal-debaussart/polyview
cd polyview
pip install -e ".[dev]"
```

---

## Quick start

The snippet below demonstrates three common workflows:
- native multi-view clustering,
- early fusion into single-view clustering,
- per-view sklearn models followed by late fusion.

```python
import numpy as np
import polyview as pv
from polyview.dataset.make_multiview_gaussian import make_multiview_gaussian
from sklearn.cluster import KMeans as SKKMeans
from sklearn.preprocessing import StandardScaler

# Synthetic multiview dataset
mvd = make_multiview_gaussian(
    n_samples=200,
    n_features=20,
    n_views=3,
    centers=4,
    random_state=0,
)
mvd.view_names = ["audio", "vision", "imu"]

# 1) Native multiview clustering
mv_model = pv.cluster.MultiViewKMeans(n_clusters=4, random_state=0)
mv_labels = mv_model.fit_predict(mvd)
print("MV labels:", mv_labels.shape)

# 2) Early fusion + single-view estimator
X_fused = pv.fusion.NormalizedFusion().fit_transform(mvd)
sv_model = SKKMeans(n_clusters=4, random_state=0)
sv_labels = sv_model.fit_predict(X_fused)
print("Fused shape:", X_fused.shape)

# 3) Per-view sklearn model + late fusion
# PolyPipeline runs one cloned SK estimator per view when final step is single-view.
pipe = pv.PolyPipeline(
    steps=[
        ("scale", StandardScaler()),
        ("kmeans", SKKMeans(n_clusters=4, random_state=0)),
    ]
)
labels_by_view = pipe.fit_predict(mvd.views)  # list of label vectors

late = pv.fusion.MajorityVote(weights=[0.2, 0.5, 0.3], tie_break="first")
late_labels = late.fit_predict(labels_by_view)
print("Late-fused labels:", late_labels.shape)
```

---

## More possibilities

You can also build hybrid pipelines that transition from multi-view to
single-view representations before classification.

```python
import polyview as pv
from polyview.dataset.make_multiview_gaussian import make_multiview_gaussian
from polyview.embed.gcca import GCCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

mvd = make_multiview_gaussian(n_samples=300, n_features=15, n_views=2, random_state=42)

# MV -> SV transition inside one pipeline:
# scale each view independently -> GCCA shared embedding -> SVM classifier
clf = pv.PolyPipeline(
    steps=[
        ("scale", StandardScaler()),
        ("gcca", GCCA(n_components=6, output="concat", regularisation=1e-3)),
        ("svm", SVC(kernel="rbf", C=1.0, random_state=42)),
    ]
)
clf.fit(mvd.views, mvd.labels)
y_hat = clf.predict(mvd.views)
print("Training accuracy:", (y_hat == mvd.labels).mean())
```

---

## Why this is useful

- Stay in true multi-view mode when your method supports it (`MultiViewKMeans`, co-regularized spectral clustering).
- Transition to single-view only when needed (`ConcatFusion`, `NormalizedFusion`, `GCCA(output="concat")`).
- Reuse sklearn estimators per view with `PolyPipeline`, then combine predictions with late fusion.
- Mix native polyview estimators and sklearn tools in one workflow.

---

## What's included

| Module | Contents |
|---|---|
| `polyview.dataset` | `MultiViewDataset`, `make_multiview_gaussian` |
| `polyview.fusion` | `ConcatFusion`, `WeightedFusion`, `NormalizedFusion`, `MajorityVote`, `KernelFusion` |
| `polyview.cluster` | `MultiViewKMeans`, `MultiViewCoRegSpectralClustering`, `MultiViewCoTrainSpectralClustering` |
| `polyview.embed` | `GCCA`, ... |
| `polyview.pipeline` | `PolyPipeline` for multi-view pipelines with per-view steps |
| `polyview.metrics` | *(coming)* consensus score, view agreement |
| `polyview.semisupervised` | *(coming)* CoTraining, label propagation |
| `polyview.viz` | *(coming)* view scatter grids, embedding plots |

All estimators are **sklearn-compatible** and work with tools like
`GridSearchCV`, `cross_val_score`, `Pipeline`, and `clone`.

Multi-view datasets are handled with `MultiViewDataset`, which provides
utilities for splitting and subsetting, plus readable per-view naming.


---

## ⚖️ License

MIT License. See [LICENSE](LICENSE) for details.

<!-- ## 📄 Citation

If you use `polyview` in your research, please consider citing:

```bibtex
``` -->