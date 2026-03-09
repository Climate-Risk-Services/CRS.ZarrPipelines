import pandas as pd
from app.domain.combine import _build_combined, ALL_HAZARD_CODES, TIME_PERIOD_ORDER


def _make_gadm_meta():
    # GID_1 is the join key for gadm_level=1 (after _build_combined renames gadm_id → GID_1)
    return pd.DataFrame({
        "GID_1": ["X.1_1"],
        "GID_0": ["X"],
        "COUNTRY": ["Country"],
        "NAME_1": ["Prov"],
        "ENGTYPE_1": ["Province"],
        "VARNAME_1": [""],
    })


def _make_hs_df(gadm_id="X.1_1", time="Cc", score=3.0):
    """Minimal per-hazard long-format DataFrame matching what aggregate_gadm writes.
    Uses 'gadm_id' column — the join key expected by _extract_wide."""
    return pd.DataFrame({
        "gadm_id": [gadm_id],
        "scenario": ["RCP85"],
        "scoring": ["5"],
        "time": [time],
        "score_mean": [score],
    })


def _call_build(hazard_dfs):
    return _build_combined(
        gadm_level=1,
        scenario="RCP85",
        stat_col="score_mean",
        scoring_scale="5",
        gadm_meta=_make_gadm_meta(),
        hazard_dfs=hazard_dfs,
    )


def test_fixed_schema_56_cols():
    # Only HS has data — all 56 hazard cols must still appear
    hazard_dfs = {code: (_make_hs_df() if code == "HS" else None) for code in ALL_HAZARD_CODES}
    result = _call_build(hazard_dfs)
    expected = [f"{c}_{t}" for c in ALL_HAZARD_CODES for t in TIME_PERIOD_ORDER]
    assert len(expected) == 56
    assert all(c in result.columns for c in expected)


def test_missing_hazards_filled_9999():
    # HS has Cc only — all other hazards and time periods → -9999
    hazard_dfs = {code: (_make_hs_df() if code == "HS" else None) for code in ALL_HAZARD_CODES}
    result = _call_build(hazard_dfs)
    for code in ALL_HAZARD_CODES:
        for tp in TIME_PERIOD_ORDER:
            col = f"{code}_{tp}"
            if code == "HS" and tp == "Cc":
                continue  # this one has real data
            assert (result[col] == -9999).all(), f"{col} should be -9999"


def test_column_order():
    hazard_dfs = {code: (_make_hs_df() if code == "HS" else None) for code in ALL_HAZARD_CODES}
    result = _call_build(hazard_dfs)
    expected_order = [f"{c}_{t}" for c in ALL_HAZARD_CODES for t in TIME_PERIOD_ORDER]
    meta_cols = {"GID_1", "GID_0", "COUNTRY", "NAME_1", "ENGTYPE_1", "VARNAME_1"}
    actual_hazard_cols = [c for c in result.columns if c not in meta_cols]
    assert actual_hazard_cols == expected_order


def test_all_none_returns_empty():
    # When all hazard_dfs are None, _build_combined returns empty DataFrame
    hazard_dfs = {code: None for code in ALL_HAZARD_CODES}
    result = _call_build(hazard_dfs)
    assert result.empty


def test_hs_cc_has_real_value():
    hazard_dfs = {code: (_make_hs_df() if code == "HS" else None) for code in ALL_HAZARD_CODES}
    result = _call_build(hazard_dfs)
    assert result["HS_Cc"].iloc[0] == 3
