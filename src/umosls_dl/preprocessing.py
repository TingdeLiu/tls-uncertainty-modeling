"""Preprocessing utilities for TLS measurements."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class TLSMeasurements:
    """Filtered values required by the RePN experiment."""

    xyz: NDArray[np.floating]
    physical_features: NDArray[np.floating]
    range_residual: NDArray[np.floating]
    scan_id: NDArray[np.generic]


def normalize_point_cloud(points: NDArray[np.floating]) -> NDArray[np.floating]:
    """Centre a point cloud and scale it by its largest radial distance."""
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if values.shape[0] == 0:
        raise ValueError("points must not be empty")

    centered = values - values.mean(axis=0)
    maximum_norm = np.linalg.norm(centered, axis=1).max()
    if maximum_norm == 0:
        raise ValueError("cannot normalize a point cloud with zero spatial extent")
    return centered / maximum_norm


def load_tls_ply(
    path: str | Path,
    *,
    maximum_absolute_residual_m: float = 0.01,
    minimum_incidence_angle_deg: float = 10.0,
) -> TLSMeasurements:
    """Load and filter the PLY schema used by the original experiment."""
    try:
        from plyfile import PlyData
    except ImportError as error:
        raise ImportError("Install the 'plyfile' package to read PLY data.") from error

    ply_path = Path(path)
    if not ply_path.is_file():
        raise FileNotFoundError(ply_path)

    records = PlyData.read(ply_path).elements[0].data
    required = {
        "x",
        "y",
        "z",
        "scalar_Scan_ID",
        "scalar_intensity",
        "scalar_incidence",
        "scalar_sigma_dist",
        "scalar_range_measured",
        "scalar_range_residual",
    }
    available = set(records.dtype.names or ())
    missing = sorted(required - available)
    if missing:
        raise ValueError(f"PLY data is missing required properties: {missing}")

    xyz = np.column_stack((records["x"], records["y"], records["z"]))
    features = np.column_stack(
        (
            records["scalar_intensity"],
            records["scalar_incidence"],
            records["scalar_sigma_dist"],
            records["scalar_range_measured"],
        )
    )
    residual = np.asarray(records["scalar_range_residual"])
    incidence = np.asarray(records["scalar_incidence"])
    mask = (np.abs(residual) < maximum_absolute_residual_m) & (
        incidence > minimum_incidence_angle_deg
    )

    return TLSMeasurements(
        xyz=xyz[mask],
        physical_features=features[mask],
        range_residual=residual[mask],
        scan_id=np.asarray(records["scalar_Scan_ID"])[mask],
    )
