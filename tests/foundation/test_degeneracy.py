from qldpc.probe.degeneracy import verdict_from_gaps


def test_level_killswitch_fires():
    v = verdict_from_gaps([{"scale": "s", "ratio": 2.5}])
    assert v["verdict"] == "C" and "level" in v["reason"]


def test_trend_killswitch_fires():
    v = verdict_from_gaps([{"scale": "a", "ratio": 1.1, "bb": True},
                           {"scale": "b", "ratio": 1.2, "bb": True},
                           {"scale": "c", "ratio": 1.35, "bb": True}])
    assert v["verdict"] == "C" and "trend" in v["reason"]


def test_b_mle_holds():
    v = verdict_from_gaps([{"scale": "a", "ratio": 1.1, "bb": True},
                           {"scale": "b", "ratio": 1.05, "bb": True}])
    assert v["verdict"] == "B-MLE"
