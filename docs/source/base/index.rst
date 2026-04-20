Base API
========

Core base classes and mixins for building polyview algorithms.

Overview
--------

The ``polyview.base`` module provides a unified architecture for multi-view learning algorithms.
All polyview estimators inherit from one of the base classes, which handle:

- **Input validation** and normalization of multi-view data
- **Sklearn compatibility** (introspection, parameter validation, cloning)
- **Semantic contracts** to signal algorithm intent to consumers
- **Common workflows** (fit, transform, predict, fit_transform, fit_predict)

Why Multiple Base Classes?
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The architecture uses multiple specialized base classes rather than a single monolithic base for several important reasons:

1. **Semantic clarity**: Each base class signals what the estimator does:

   - ``BaseFusion`` → "This combines views into a single array"
   - ``BaseMultiViewClusterer`` → "This produces cluster labels"
   - ``BaseMultiViewEmbedder`` → "This produces a low-dimensional embedding"

   This allows downstream code (like ``PolyPipeline``) to make intelligent decisions about visualization, mode transitions, and validation.

2. **Enforced contracts**: Each base provides specific interface contracts:

   - ``BaseMultiViewTransformer`` requires ``transform()``
   - ``BaseMultiViewClusterer`` requires ``labels_`` attribute after fitting
   - ``BaseLateFusion`` requires ``predict()`` on per-view predictions

   These contracts prevent accidental misuse and enable automated checks.

3. **Extensibility**: New specialized bases can be added without breaking existing code:

   - ``BaseFusion`` supports both early and intermediate fusion paradigms
   - New bases (e.g., ``BaseMultiViewRegressor``) can be added as the library grows
   - Consumers can check ``isinstance(step, BaseFusion)`` for specific behavior

4. **Composition via mixins**: Mixins (``MultiViewTransformerMixin``, ``MultiViewClusterMixin``, ``MultiViewEmbedderMixin``) separate concerns and can be mixed with different base classes as needed.

This design pattern, borrowed from sklearn and scikit-learn-compatible libraries, provides clarity and flexibility without overwhelming complexity.

Design Contract
~~~~~~~~~~~~~~~

Every polyview estimator follows these conventions:

- ``__init__`` stores hyperparameters only — no data, no fitting.
- Every hyperparameter must be a keyword argument with a default.
- Learned attributes are written as ``attr_`` (trailing underscore), following sklearn convention, so ``check_is_fitted`` works.
- Views are always passed as ``list[np.ndarray]`` (or a ``MultiViewDataset``), *never* as a single array.

Inheritance Hierarchy
~~~~~~~~~~~~~~~~~~~~~

The base classes form a hierarchy:

.. code-block:: text

    BaseMultiView (ABC)
    ├── BaseMultiViewTransformer (+ MultiViewTransformerMixin)
    │   ├── BaseFusion
    │   │   └── Used by: Fusion methods (ConcatFusion, KernelFusion, etc.)
    │   └── Used by: Canonical correlation and other general transformers
    ├── BaseMultiViewClusterer (+ MultiViewClusterMixin)
    │   └── Used by: Co-training, spectral clustering, NMF methods
    └── BaseMultiViewEmbedder (+ MultiViewEmbedderMixin)
        └── Used by: MultiViewMDS, and other embedding-specific algorithms

    BaseEstimator (ABC)
    └── BaseLateFusion
        └── Used by: Late fusion voting and prediction aggregation


MultiViewDataset
----------------

**Purpose**: Unified container for multi-view datasets.

While all estimators accept lists of views (``list[np.ndarray]``), the ``MultiViewDataset`` class provides a richer container that:

- **Stores views** as an ordered, immutable collection
- **Attaches metadata** (view names, sample labels) to the data
- **Enables convenient access** via indexing (e.g., ``mvd["audio"]``) or iteration
- **Provides utilities** like ``train_test_split()`` for common workflows
- **Simplifies pipelines** by bundling data and metadata together

**When to use**:

- Building complete multi-view datasets with labels and names
- Organizing complex experiments with many views
- Creating reproducible data loaders for benchmarking
- Passing data through pipelines that track view identity

**Example**:

.. code-block:: python

    import numpy as np
    from polyview import MultiViewDataset

    # Create multi-view dataset with names and labels
    X1 = np.random.rand(100, 10)   # acoustic features
    X2 = np.random.rand(100, 20)   # visual features
    y = np.random.randint(0, 3, 100)  # class labels

    mvd = MultiViewDataset(
        [X1, X2],
        labels=y,
        view_names=["audio", "video"]
    )

    # Access by view name or index
    print(mvd["audio"].shape)        # (100, 10)
    print(mvd.views[1].shape)        # (100, 20)
    print(mvd.n_views)               # 2
    print(mvd.n_samples)             # 100

    # Convenient data splitting
    train, test = mvd.train_test_split(test_size=0.2, random_state=42)

**Key attributes**:

- ``views`` — List of numpy arrays (one per view)
- ``labels`` — Sample labels (if provided)
- ``view_names`` — Human-readable names for each view
- ``n_views`` — Number of views
- ``n_samples`` — Number of samples (shared across views)

**Note**: While ``MultiViewDataset`` provides convenience, all polyview estimators accept plain lists of arrays. Use ``MultiViewDataset`` when you need the additional metadata and utilities.


Base Classes & Mixins
---------------------

BaseMultiView
^^^^^^^^^^^^^

**Role**: Foundation for all polyview algorithms.

**Responsibilities**:

- Validates and normalizes list-of-views input
- Stores view dimensions (``n_views_in_``, ``n_features_in_``, ``n_samples_``)
- Supports sklearn utilities (``get_params()``, ``set_params()``, ``clone()``)
- Enforces view consistency across ``fit`` and ``transform``

**Key methods**:

- ``_validate_views(views, *, accept_sparse=False, reset=True)`` — Validates and coerces multi-view input
- ``fit(views, y=None)`` — Abstract; subclasses must implement

**Example**:

.. code-block:: python

    class MyAlgorithm(BaseMultiView):
        def fit(self, views, y=None):
            self._validate_views(views)
            self.is_fitted_ = True
            return self


MultiViewTransformerMixin
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Role**: Signals that an algorithm produces a *combined* output from multiple views.

**Responsibilities**:

- Enforces ``transform(views)`` interface
- Provides default ``fit_transform()`` that chains ``fit`` and ``transform``
- Output is typically a single array (fused representation)

**Key methods**:

- ``transform(views)`` — Abstract; must return ``ndarray of shape (n_samples, n_components)``
- ``fit_transform(views, y=None)`` — Default implementation calls ``fit`` then ``transform``

**Used by**:

- :class:`polyview.fusion.early.ConcatFusion` — concatenates views feature-wise
- :class:`polyview.fusion.early.WeightedFusion` — weighted sum of views
- :class:`polyview.fusion.early.NormalizedFusion` — normalized view fusion
- :class:`polyview.fusion.KernelFusion` — kernel-based fusion
- :class:`polyview.embed.GCCA` — generalized canonical correlation analysis
- :class:`polyview.embed.MCCA` — multiview canonical correlation analysis


MultiViewClusterMixin
^^^^^^^^^^^^^^^^^^^^^^

**Role**: Signals that an algorithm produces cluster labels.

**Responsibilities**:

- Enforces ``fit()`` to store ``labels_`` attribute
- Provides ``fit_predict()`` convenience method
- Output is a 1-D integer label array

**Key methods**:

- ``fit_predict(views, y=None)`` — Calls ``fit`` then returns ``labels_``

**Used by**:

- :class:`polyview.cluster.mv_kmeans.MultiViewKMeans` — multi-view k-means clustering
- :class:`polyview.cluster.mv_coreg_sc.MultiViewCoRegSpectralClustering` — co-regularized spectral clustering
- :class:`polyview.cluster.mv_cotrain_sc.MultiViewCoTrainSpectralClustering` — co-training spectral clustering
- :class:`polyview.cluster.mv_nmf.MultiViewNMF` — multi-view non-negative matrix factorization


MultiViewEmbedderMixin
^^^^^^^^^^^^^^^^^^^^^^

**Role**: Signals that an algorithm produces a *low-dimensional embedding*.

**Responsibilities**:

- Extends ``MultiViewTransformerMixin`` (inherits ``transform`` and ``fit_transform``)
- Provides ``embedding_`` property (with getter/setter) that stores the learned embedding
- Semantically signals that output is an embedding, not just a feature transformation

**Key properties**:

- ``embedding_`` — The learned low-dimensional representation (``ndarray of shape (n_samples, n_components)``)

**Used by**:

- :class:`polyview.embed.multiviewmds.MultiViewMDS` — multi-view multidimensional scaling with adaptive weighting


Concrete Base Classes
---------------------

BaseMultiViewTransformer
^^^^^^^^^^^^^^^^^^^^^^^^^

**Role**: Ready-to-subclass base for building multi-view *transformation* algorithms (both fusion and embedding preparation).

**Combines**:

- ``BaseMultiView`` — Input validation, sklearn compatibility
- ``MultiViewTransformerMixin`` — ``transform()`` and ``fit_transform()`` interface

**Subclasses must implement**:

- ``fit(views, y=None)`` — Learn parameters from multi-view data
- ``transform(views)`` — Apply transformation and return a single fused array

**Subclassed by**:

- ``BaseFusion`` — For methods that combine views (early, intermediate, or other fusion paradigms)
- ``BaseMultiViewEmbedder`` — For methods that produce embeddings

**Example**:

.. code-block:: python

    class MyTransformer(BaseMultiViewTransformer):
        def fit(self, views, y=None):
            views = self._validate_views(views, reset=True)
            # Learn from views
            return self

        def transform(self, views):
            views = self._validate_views(views, reset=False)
            # Combine views into single output
            return np.concatenate(views, axis=1)


BaseFusion
^^^^^^^^^^

**Role**: Ready-to-subclass base for *fusion* methods that combine views into a single fused representation.

Supports multiple fusion paradigms:

- **Early fusion**: Combines views before any feature extraction
- **Intermediate fusion**: Combines views after per-view processing (future capability)

**Combines**:

- ``BaseMultiViewTransformer`` — Full transformer interface and multi-view handling
- Semantic signal to consumers (e.g., ``PolyPipeline``) that output is single-view

**Subclasses must implement**:

- ``fit(views, y=None)`` — Learn fusion parameters from multi-view data
- ``transform(views)`` — Fuse views and return a single array

**Current implementations (early fusion)**:

- :class:`polyview.fusion.early.ConcatFusion` — horizontal concatenation
- :class:`polyview.fusion.early.WeightedFusion` — weighted concatenation
- :class:`polyview.fusion.early.NormalizedFusion` — normalized concatenation
- :class:`polyview.fusion.KernelFusion` — kernel-based fusion

**Example**:

.. code-block:: python

    class MyConcatFusion(BaseFusion):
        def fit(self, views, y=None):
            views = self._validate_views(views, reset=True)
            self.n_features_out_ = sum(v.shape[1] for v in views)
            return self

        def transform(self, views):
            views = self._validate_views(views, reset=False)
            return np.concatenate(views, axis=1)

**Note**: All fusion methods are automatically recognized in ``PolyPipeline`` as producing single-view output, enabling correct diagram coloring and mode transitions.


BaseMultiViewClusterer
^^^^^^^^^^^^^^^^^^^^^^

**Role**: Ready-to-subclass base for building *multi-view clustering* algorithms.

**Combines**:

- ``BaseMultiView`` — Input validation, sklearn compatibility
- ``MultiViewClusterMixin`` — ``fit_predict()`` convenience method

**Subclasses must implement**:

- ``fit(views, y=None)`` — Learn clustering model, store labels in ``labels_`` attribute

**Example**:

.. code-block:: python

    class MyClusterer(BaseMultiViewClusterer):
        def __init__(self, n_clusters=2):
            super().__init__()
            self.n_clusters = n_clusters

        def fit(self, views, y=None):
            views = self._validate_views(views, reset=True)
            # Clustering algorithm logic
            self.labels_ = np.zeros(self.n_samples_, dtype=int)
            return self


BaseMultiViewEmbedder
^^^^^^^^^^^^^^^^^^^^^

**Role**: Ready-to-subclass base for building *multi-view embedding* algorithms.

**Combines**:

- ``BaseMultiView`` — Input validation, sklearn compatibility
- ``MultiViewEmbedderMixin`` — ``transform()`` and ``embedding_`` property

**Subclasses must implement**:

- ``fit(views, y=None)`` — Learn embedding parameters, store in ``embedding_`` attribute
- ``transform(views)`` — Return the embedding (typically non-parametric, returns stored ``embedding_``)

**Example**:

.. code-block:: python

    class MyEmbedder(BaseMultiViewEmbedder):
        def fit(self, views, y=None):
            views = self._validate_views(views, reset=True)
            # Compute embedding
            self.embedding_ = np.random.rand(self.n_samples_, 2)
            return self

        def transform(self, views):
            self._validate_views(views, reset=False)
            return self.embedding_  # Return learned embedding


BaseLateFusion
^^^^^^^^^^^^^^

**Role**: Abstract base for *late-fusion* aggregation methods that combine per-view predictions.

**Inheritance**:

- Inherits from ``BaseEstimator`` and ``ABC``
- Uses ``@abstractmethod`` decorator to enforce implementation (early detection of incomplete subclasses)

**Contract**:

- Takes **list of 1-D prediction vectors** (one per view), not multi-view feature matrices
- Returns a single **aggregated 1-D prediction** vector
- Used in :class:`polyview.pipeline.PolyPipeline` to aggregate per-view classifier outputs

**Abstract methods** (subclasses must implement):

- ``fit(preds_by_view, y=None)`` — Learn aggregation parameters from view-specific predictions
- ``predict(preds_by_view)`` — Aggregate predictions and return combined output

**Concrete methods**:

- ``fit_predict(preds_by_view, y=None)`` — Convenience method that calls ``fit`` then ``predict``

**Used by**:

- :class:`polyview.fusion.late.MajorityVote` — aggregates predictions via majority voting
- Custom voting or averaging schemes in pipelines

**Example**:

.. code-block:: python

    class MyLateFusion(BaseLateFusion):
        def fit(self, preds_by_view, y=None):
            # Learn aggregation parameters
            return self

        def predict(self, preds_by_view):
            # Average predictions across views
            P = np.vstack([np.asarray(p, dtype=float) for p in preds_by_view])
            return P.mean(axis=0)


Usage Patterns
--------------

**Choosing the Right Base**:

+---------------------------------------+---------------------------------------------------+
| Goal                                  | Base Class                                        |
+=======================================+===================================================+
| Fuse views into single representation  | ``BaseFusion``                                    |
+---------------------------------------+---------------------------------------------------+
| Generic multi-view transformation     | ``BaseMultiViewTransformer``                      |
+---------------------------------------+---------------------------------------------------+
| Learn low-dimensional embedding       | ``BaseMultiViewEmbedder``                         |
+---------------------------------------+---------------------------------------------------+
| Perform multi-view clustering         | ``BaseMultiViewClusterer``                        |
+---------------------------------------+---------------------------------------------------+
| Aggregate per-view predictions        | ``BaseLateFusion``                                |
+---------------------------------------+---------------------------------------------------+


API Reference
-------------

.. automodule:: polyview.base
   :members:
   :show-inheritance:
   :undoc-members:
