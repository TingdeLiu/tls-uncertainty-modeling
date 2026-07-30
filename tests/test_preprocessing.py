import numpy as np
import pytest

from umosls_dl.preprocessing import normalize_point_cloud


def test_normalize_point_cloud_centres_and_scales_points():
    points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

    normalized = normalize_point_cloud(points)

    np.testing.assert_allclose(normalized.mean(axis=0), np.zeros(3))
    assert np.linalg.norm(normalized, axis=1).max() == pytest.approx(1.0)


def test_normalize_point_cloud_rejects_zero_extent():
    with pytest.raises(ValueError, match="zero spatial extent"):
        normalize_point_cloud(np.ones((3, 3)))
