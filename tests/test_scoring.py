import numpy as np
import xarray as xr
import pytest
from app.domain.scoring import _score_chunk, score_zarr_multi


# --- _score_chunk ---

def test_score_chunk_ascending():
    x = np.array([0.0, 5.0, 12.0, 25.0])
    result = _score_chunk(x, [0, 10, 20], [1, 2, 3])
    assert list(result) == [1.0, 1.0, 2.0, 3.0]


def test_score_chunk_descending():
    # descending: higher value → lower score (e.g. DR)
    x = np.array([100.0, 50.0, 10.0])
    result = _score_chunk(x, [80, 40, 0], [1, 2, 3])
    assert result[0] < result[2]  # high value scores lower


def test_score_chunk_nan_preserved():
    x = np.array([5.0, np.nan])
    result = _score_chunk(x, [0, 10], [1, 2])
    assert not np.isnan(result[0])
    assert np.isnan(result[1])


def test_score_chunk_positive_inf():
    x = np.array([np.inf])
    result = _score_chunk(x, [0, 10, 20], [1, 2, 3])
    assert result[0] == 3.0  # last score


def test_score_chunk_negative_inf():
    x = np.array([-np.inf])
    result = _score_chunk(x, [0, 10, 20], [1, 2, 3])
    assert result[0] == 1.0  # first score


def test_score_chunk_float16_output():
    x = np.array([5.0])
    result = _score_chunk(x, [0, 10], [1, 2])
    assert result.dtype == np.float16


# --- score_zarr_multi ---

def test_score_zarr_multi_scoring_dim():
    data = xr.DataArray(np.array([5.0, 15.0, 25.0]), dims=["x"])
    result = score_zarr_multi(
        data,
        scales=["5", "10"],
        thresholds_per_scale={
            "5":  ([0, 10, 20], [1, 2, 3]),
            "10": ([0, 10, 20], [2, 4, 6]),
        },
    )
    assert "scoring" in result.dims
    assert list(result.scoring.values) == ["5", "10"]


def test_score_zarr_multi_values():
    data = xr.DataArray(np.array([5.0]), dims=["x"])
    result = score_zarr_multi(
        data,
        scales=["5"],
        thresholds_per_scale={"5": ([0, 10, 20], [1, 2, 3])},
    )
    val = float(result["score"].sel(scoring="5").values[0])
    assert val == 1.0


def test_score_zarr_multi_returns_dataset():
    data = xr.DataArray(np.array([5.0, 15.0]), dims=["x"])
    result = score_zarr_multi(
        data,
        scales=["5"],
        thresholds_per_scale={"5": ([0, 10, 20], [1, 2, 3])},
    )
    assert isinstance(result, xr.Dataset)
    assert "score" in result.data_vars
