import pandas as pd

from screener.pipeline.scoring import percentile_rank, winsorize_series


def test_winsorize_clips_outliers():
    s = pd.Series([1, 2, 3, 4, 100])
    out = winsorize_series(s, 0.1, 0.9)
    assert out.max() < 100
    assert out.min() >= 1


def test_percentile_rank_higher_is_better():
    s = pd.Series({"A": 1, "B": 2, "C": 3})
    out = percentile_rank(s, higher_is_better=True)
    assert out["C"] > out["B"] > out["A"]


def test_percentile_rank_lower_is_better_inverts():
    s = pd.Series({"A": 1, "B": 2, "C": 3})
    out = percentile_rank(s, higher_is_better=False)
    assert out["A"] > out["B"] > out["C"]


def test_percentile_rank_missing_stays_nan():
    s = pd.Series({"A": 1.0, "B": None, "C": 3.0})
    out = percentile_rank(s, higher_is_better=True)
    assert pd.isna(out["B"])
