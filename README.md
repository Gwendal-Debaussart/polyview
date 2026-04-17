# polyview

[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/deepcharles/ruptures/graphs/commit-activity)
[![CI](https://github.com/Gwendal-Debaussart/polyview/actions/workflows/doc.yml/badge.svg)](https://github.com/Gwendal-Debaussart/polyview/actions/workflows/doc.yml/badge.svg)
[![PyPI version](https://badge.fury.io/py/polyview.svg)](https://badge.fury.io/py/polyview)
<a href="https://github.com/psf/black"><img alt="Code style: ruff" src="https://img.shields.io/badge/code%20style-ruff-000000.svg"></a>

<p align="center">
  <img src="docs/source/_static/polyview.svg" alt="polyview Logo" width="250"/>
</p>

**polyview** is a Python library for multi-view learning, providing a consistent API for multi-view clustering, embedding, and fusion methods. It allows you to easily build pipelines that combine native multi-view estimators with single-view sklearn tools, while keeping track of per-view data and predictions.

Documentation is available at: [https://gwendal-debaussart.github.io/polyview/](https://gwendal-debaussart.github.io/polyview/)

## Installation

Install from PyPI:

```bash
pip install polyview
```


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

Additional examples are available in the documentation: [https://gwendal-debaussart.github.io/polyview/examples/](https://gwendal-debaussart.github.io/polyview/examples/)


## Contributing

Contributions are very welcome! If you plan to contribute, please see the [CONTRIBUTING](CONTRIBUTING.md) guidelines.

## ⚖️ License

GNU General Public License. See [LICENSE](LICENSE) for details.

<!-- ## 📄 Citation

If you use `polyview` in your research, please consider citing:

```bibtex
``` -->