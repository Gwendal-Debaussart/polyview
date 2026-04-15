from sklearn.datasets import make_blobs
from polyview.dataset.multiviewdataset import MultiViewDataset
import numpy as np


def make_multiview_gaussian(
    n_samples=300,
    n_features=10,
    latent_dim=3,
    centers=3,
    n_views=2,
    noise_std=0.1,
    random_state=None,
):
    """
    Generate a synthetic multi-view dataset using a shared latent Gaussian model.

    Parameters
    ----------
    n_samples : int, default=300
        Number of samples.
    n_features : int, default=10
        Number of features per view.
    latent_dim : int, default=3
        Dimension of shared latent space.
    centers : int, default=3
        Number of clusters in latent space.
    n_views : int, default=2
        Number of views.
    noise_std : float, default=0.1
        Standard deviation of Gaussian noise.
    random_state : int or None, default=None
        Seed.

    Returns
    -------
    MultiViewDataset
        A MultiViewDataset object containing the generated views and labels.
    """

    rng = np.random.RandomState(random_state)

    latent_centers = rng.randn(centers, latent_dim) * 5.0
    y = rng.randint(0, centers, size=n_samples)
    z = latent_centers[y] + rng.randn(n_samples, latent_dim)
    views = []
    for v in range(n_views):
        view_rng = np.random.RandomState(None if random_state is None else random_state + v + 1)
        A = view_rng.randn(n_features, latent_dim)
        A /= np.linalg.norm(A, axis=0, keepdims=True) + 1e-8
        X = z @ A.T + noise_std * view_rng.randn(n_samples, n_features)
        views.append(X)

    return MultiViewDataset(views=views, labels=y)