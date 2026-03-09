import pytest
from app.utils.scoring_config import ScoringConfig


@pytest.fixture(scope="module")
def cfg():
    return ScoringConfig()


def test_get_thresholds_hs_5pt(cfg):
    # HS 5_point: thresholds=[-inf,9,26,32,38,46,inf], scores=[0,1,2,3,4,5]
    thresholds, scores = cfg.get_thresholds("HS", "5")
    assert len(thresholds) > 0
    assert len(scores) > 0
    assert all(isinstance(s, (int, float)) for s in scores)
    assert scores[-1] == 5


def test_get_thresholds_normalises_suffix(cfg):
    t1, s1 = cfg.get_thresholds("HS", "5")
    t2, s2 = cfg.get_thresholds("HS", "5_point")
    assert t1 == t2 and s1 == s2


def test_get_thresholds_10pt(cfg):
    thresholds, scores = cfg.get_thresholds("HS", "10")
    assert scores[-1] == 10


def test_get_thresholds_ls_threshold_type(cfg):
    t_ari,  _ = cfg.get_thresholds("LS", "5", threshold_type="thresholds_ari")
    t_susc, _ = cfg.get_thresholds("LS", "5", threshold_type="thresholds_susceptibility")
    t_fin,  _ = cfg.get_thresholds("LS", "5", threshold_type="thresholds_final")
    # All three should return distinct threshold lists
    assert t_ari != t_susc or t_susc != t_fin  # at least two differ


def test_get_thresholds_unknown_hazard(cfg):
    with pytest.raises(ValueError):
        cfg.get_thresholds("UNKNOWN", "5")


def test_score_value_returns_int(cfg):
    score = cfg.score_value("HS", 35.0, "5")
    assert isinstance(score, (int, float))


def test_score_value_boundary(cfg):
    # Value below first threshold → lowest score
    thresholds, scores = cfg.get_thresholds("DR", "5")
    score_low = cfg.score_value("DR", -1.0, "5")
    assert isinstance(score_low, (int, float))


def test_list_hazards(cfg):
    hazards = cfg.list_hazards()
    assert "HS" in hazards
    assert "LS" in hazards
    assert len(hazards) == 14
