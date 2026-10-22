"""FWI System: the operational baseline must reproduce the published
reference values (Van Wagner & Pickett 1985 test case) and behave like a
fire-danger index (hot/dry/windy raises it, rain lowers it)."""
import numpy as np
import pytest

from wfpin.fwi import DC0, DMC0, FFMC0, OUTPUT_COLUMNS, run_series, step

START = (FFMC0, DMC0, DC0)


def test_matches_published_reference_case():
    # T=17 C, RH=42 %, wind=25 km/h, no rain, April, standard startup codes.
    out = step(START, 17.0, 42.0, 25.0, 0.0, 4)
    expected = dict(ffmc=87.692, dmc=8.545, dc=19.014, isi=10.853, bui=8.490, fwi=10.096)
    for key, value in expected.items():
        assert float(out[key]) == pytest.approx(value, abs=0.005), key


def test_hotter_drier_windier_is_more_dangerous():
    base = float(step(START, 20.0, 60.0, 10.0, 0.0, 4)["fwi"])
    assert float(step(START, 30.0, 60.0, 10.0, 0.0, 4)["fwi"]) > base
    assert float(step(START, 20.0, 20.0, 10.0, 0.0, 4)["fwi"]) > base
    assert float(step(START, 20.0, 60.0, 30.0, 0.0, 4)["fwi"]) > base


def test_rain_lowers_every_moisture_code():
    dry = (90.0, 60.0, 300.0)
    before = step(dry, 20.0, 50.0, 10.0, 0.0, 4)
    after = step(dry, 20.0, 50.0, 10.0, 20.0, 4)
    for code in ("ffmc", "dmc", "dc", "fwi"):
        assert float(after[code]) < float(before[code]), code


def test_vectorised_over_points_and_restart():
    n_pts, n_days = 3, 400
    dates = np.arange("2020-01-01", "2021-02-04", dtype="datetime64[D]")[:n_days]
    t = np.full((n_pts, n_days), 25.0)
    rh = np.full((n_pts, n_days), 30.0)
    w = np.full((n_pts, n_days), 15.0)
    rain = np.zeros((n_pts, n_days))
    t[1] += 8  # point 1 is hotter -> must end up more dangerous
    out = run_series(dates, t, rh, w, rain)
    assert set(out) == set(OUTPUT_COLUMNS) and out["fwi"].shape == (n_pts, n_days)
    assert np.isfinite(out["fwi"]).all()
    assert out["fwi"][1, 200] > out["fwi"][0, 200]
    jan1_2021 = int(np.flatnonzero(dates == np.datetime64("2021-01-01"))[0])
    assert out["dc"][0, jan1_2021 - 1] > out["dc"][0, jan1_2021]  # codes reset on Jan 1


def test_fwi_is_causal():
    """Truncating the weather after day k must not change any output up to k
    (so the baseline cannot leak the future into the panel)."""
    rng = np.random.default_rng(1)
    dates = np.arange("2020-02-01", "2020-05-01", dtype="datetime64[D]")
    n = len(dates)
    t, rh = rng.uniform(5, 35, (2, n)), rng.uniform(15, 90, (2, n))
    w, rain = rng.uniform(0, 30, (2, n)), rng.exponential(2, (2, n))
    full = run_series(dates, t, rh, w, rain)
    k = 40
    cut = run_series(dates[:k], t[:, :k], rh[:, :k], w[:, :k], rain[:, :k])
    for code in OUTPUT_COLUMNS:
        np.testing.assert_allclose(full[code][:, :k], cut[code], atol=1e-9)


def test_missing_day_does_not_poison_later_days():
    dates = np.arange("2020-03-01", "2020-03-11", dtype="datetime64[D]")
    t = np.full((1, 10), 25.0)
    rh, w, rain = np.full((1, 10), 30.0), np.full((1, 10), 15.0), np.zeros((1, 10))
    clean = run_series(dates, t, rh, w, rain)
    t_gap = t.copy()
    t_gap[0, 4] = np.nan  # one missing observation day
    gap = run_series(dates, t_gap, rh, w, rain)
    assert np.isnan(gap["fwi"][0, 4])        # the missing day itself is unknown
    assert np.isfinite(gap["fwi"][0, 5:]).all()  # but the season carries on
    # With constant weather, skipping one day just delays the series by one day.
    for code in ("ffmc", "dmc", "dc"):
        np.testing.assert_allclose(gap[code][0, 5:], clean[code][0, 4:9], atol=1e-6)
