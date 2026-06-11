"""Unit tests for M5 — `pev_synth.plug_in_model` (v2.0).

Covers the Wave 2 Dev E acceptance criteria 7-11 and the orchestrator
brief's unit-test list:

    1. test_fit_plug_ins_residential_runs
    2. test_fit_plug_ins_workplace_picks_K_via_bic
    3. test_fit_plug_ins_w9_invariant_no_cluster_exceeds_6h_window
    4. test_fit_plug_ins_workplace_beta2_higher_than_residential
    5. test_fit_plug_ins_workplace_aggregate_rate_in_band
    6. test_compute_p_plug_workplace_anchor_in_band
    7. test_compute_p_plug_residential_anchor_in_band
    8. test_no_workplace_plug_in_when_dest_type_not_work
    9. test_weekend_dampening_workplace_stronger_than_residential
   10. test_assign_clusters_uses_K_from_fit
   11. test_seed_reproducible_residential
   12. test_provenance_table_documents_workplace_beta_priors

The fixtures use the existing residential fleet (1000-EV M3 output)
sliced to 5-25 EVs so the suite finishes in under a minute.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make `code/` importable when run from anywhere.
_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from pev_synth import plug_in_model as pim  # noqa: E402
from pev_synth import travel_week_builder as twb  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
# v2.0 per-region layout (the flat v1.1 `processed/residential_ev_synth`
# path no longer exists). The M3 fleet now lives under the region subdir.
_FLEET_PATH = (
    _REPO_ROOT / "data/pev/processed/bay_area/residential_ev_synth/fleet.parquet"
)
_NHTS_RAW = _REPO_ROOT / "data/pev/raw/nhts2017"
_WORKPLACE_COHORT = _REPO_ROOT / "data/pev/raw/evwatts_cohorts/workplace_subset.parquet"
_SPEECH_RESIDENTIAL = _REPO_ROOT / "data/pev/raw/speech/speech_k16_residential.json"
_SPEECH_WORKPLACE = _REPO_ROOT / "data/pev/raw/speech/speech_k16_workplace.json"


def _have_minimum_inputs() -> bool:
    if not _FLEET_PATH.is_file():
        return False
    if (_NHTS_RAW / "nhts2017_ca_hh.parquet").is_file():
        return True
    if (_NHTS_RAW / "hhpub.csv").is_file():
        return True
    return False


pytestmark = pytest.mark.skipif(
    not _have_minimum_inputs(),
    reason="needs M1 NHTS inputs + M3 fleet.parquet present in repo data tree",
)

# The workplace M5 fit requires the EVWatts public workplace cohort
# (`workplace_subset.parquet`), which is built from raw EVWatts CSVs that
# are NOT on disk here / in CI. Tests that drive `fit_and_assign_plug_ins(
# profile_type="workplace")` are gated on it; residential tests are not.
_needs_workplace_cohort = pytest.mark.skipif(
    not _WORKPLACE_COHORT.is_file(),
    reason=(
        f"workplace EVWatts cohort not present at {_WORKPLACE_COHORT}; "
        "build it from EVWatts CSVs first"
    ),
)


# --------------------------------------------------------------------------- #
# Synthetic fixtures.
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def fleet_full() -> pd.DataFrame:
    return pd.read_parquet(_FLEET_PATH)


@pytest.fixture(scope="module")
def small_fleet(fleet_full: pd.DataFrame) -> pd.DataFrame:
    return fleet_full.sort_values("ev_id").head(20).reset_index(drop=True)


@pytest.fixture(scope="module")
def mini_fleet(fleet_full: pd.DataFrame) -> pd.DataFrame:
    return fleet_full.sort_values("ev_id").head(5).reset_index(drop=True)


@pytest.fixture(scope="module")
def mobility_residential(small_fleet: pd.DataFrame) -> pd.DataFrame:
    return twb.build_mobility(
        fleet=small_fleet,
        region="bay_area",
        profile_type="residential",
        seed=pim.MASTER_SEED_DEFAULT,
        nhts_data_dir=_NHTS_RAW,
    )


@pytest.fixture(scope="module")
def mobility_workplace(small_fleet: pd.DataFrame) -> pd.DataFrame:
    return twb.build_mobility(
        fleet=small_fleet,
        region="bay_area",
        profile_type="workplace",
        seed=pim.MASTER_SEED_DEFAULT,
        nhts_data_dir=_NHTS_RAW,
    )


@pytest.fixture(scope="module")
def m5_residential(
    small_fleet: pd.DataFrame, mobility_residential: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pim.ClusterFit]:
    return pim.fit_and_assign_plug_ins(
        fleet=small_fleet,
        mobility=mobility_residential,
        region="bay_area",
        profile_type="residential",
        seed=pim.MASTER_SEED_DEFAULT,
    )


@pytest.fixture(scope="module")
def m5_workplace(
    small_fleet: pd.DataFrame, mobility_workplace: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pim.ClusterFit]:
    # The workplace acceptance suite below pins K_chosen ∈ {2, 3, 4, 5}
    # and the W9 invariant — both behaviours unique to the v2 BIC-sweep
    # cohort fit. v3.0's SPEECh K=16 path is exercised via the
    # ``test_fit_plug_ins_workplace_speech_k16`` test instead, so this
    # fixture keeps the legacy ``cluster_source="bic_sweep"`` route.
    return pim.fit_and_assign_plug_ins(
        fleet=small_fleet,
        mobility=mobility_workplace,
        region="bay_area",
        profile_type="workplace",
        cluster_source="bic_sweep",
        seed=pim.MASTER_SEED_DEFAULT,
        cohort_parquet=_WORKPLACE_COHORT,
    )


# --------------------------------------------------------------------------- #
# Acceptance tests (criteria 7-11).
# --------------------------------------------------------------------------- #


def test_fit_plug_ins_residential_runs(
    m5_residential: tuple[pd.DataFrame, pd.DataFrame, pim.ClusterFit],
) -> None:
    """v3.0: residential clusters come from Powell-2022 SPEECh K=16.

    K_chosen reflects the **union** of weekday and weekend present
    groups (9 + 8 = 9 unique groups for the residential `home` segment
    per ``speech_k16_residential.json``). Per-EV cluster_z is in
    {1..K_chosen}.
    """
    augmented_fleet, events, fit = m5_residential
    # v3.0 default path uses SPEECh K=16; K_chosen is the union size
    # (9 for residential `home`), not the legacy BIC K ∈ {3, 4, 6}.
    assert fit.speech_params is not None, (
        "v3.0 residential fit must carry SPEECh K=16 params"
    )
    assert fit.K_chosen == len(
        set(fit.speech_params.groups_present_weekday)
        | set(fit.speech_params.groups_present_weekend)
    )
    assert fit.K_chosen >= 8  # at least 8 groups present in residential `home`.
    assert fit.K_chosen <= pim.SPEECH_K_TOTAL
    assert "cluster_z" in augmented_fleet.columns
    assert augmented_fleet["cluster_z"].notna().all()
    assert (augmented_fleet["cluster_z"] >= 1).all()
    assert (augmented_fleet["cluster_z"] <= fit.K_chosen).all()
    # Events table has UTC timestamps and the v2.0 schema.
    assert len(events) > 0
    assert str(events["arrival_ts"].dtype) == "datetime64[ns, UTC]"
    assert str(events["departure_ts"].dtype) == "datetime64[ns, UTC]"
    assert "event_id" in events.columns
    assert "profile_type_event" in events.columns


def test_fit_plug_ins_residential_bic_sweep_back_compat(
    small_fleet: pd.DataFrame,
    mobility_residential: pd.DataFrame,
) -> None:
    """v2.1 back-compat path: ``cluster_source='bic_sweep'`` keeps K ∈ {3, 4, 6}.

    This exercises the legacy code path that ``legacy_v2_1_compat=True``
    on ``cache_regen.one(...)`` will route through.
    """
    _, _, fit = pim.fit_and_assign_plug_ins(
        fleet=small_fleet,
        mobility=mobility_residential,
        region="bay_area",
        profile_type="residential",
        cluster_source="bic_sweep",
        seed=pim.MASTER_SEED_DEFAULT,
    )
    assert fit.speech_params is None
    assert fit.K_chosen in pim.K_SWEEP_RESIDENTIAL
    assert len(fit.cluster_centres_h) == fit.K_chosen


@_needs_workplace_cohort
def test_fit_plug_ins_workplace_picks_K_via_bic(
    m5_workplace: tuple[pd.DataFrame, pd.DataFrame, pim.ClusterFit],
) -> None:
    """Acceptance #8: workplace BIC sweep covers K ∈ {2, 3, 4, 5} and
    picks K_chosen ∈ {2, 3, 4, 5}.

    The default K=3 fires only when BIC is ambiguous (rel-gap < 1%).
    Note: when cohort < 150 vehicles, plan §4.4.3 caps K ≤ 4, so the
    effective sweep is {2, 3, 4} and K=5 is reported in `K_sweep` but
    absent from `bic_per_K`.
    """
    _, _, fit = m5_workplace
    assert tuple(fit.K_sweep) == pim.K_SWEEP_WORKPLACE
    assert fit.K_chosen in fit.K_sweep
    # BIC values populated for every K that was actually attempted (K ≤ 4
    # when cohort is small).
    attempted = [k for k in fit.K_sweep if k in fit.bic_per_K]
    assert len(attempted) >= 3, (
        f"workplace BIC sweep attempted only {attempted} of {fit.K_sweep}"
    )
    assert not fit.degenerate_flag, (
        f"workplace BIC sweep degenerate: {fit.note}"
    )


@_needs_workplace_cohort
def test_fit_plug_ins_w9_invariant_no_cluster_exceeds_6h_window(
    m5_workplace: tuple[pd.DataFrame, pd.DataFrame, pim.ClusterFit],
) -> None:
    """Acceptance #9 / W9: no selected cluster has start-time window > 6 h."""
    _, _, fit = m5_workplace
    for w in fit.cluster_windows_h:
        assert w <= pim.WORKPLACE_CLUSTER_WINDOW_MAX_H + 1e-6, (
            f"cluster window {w:.2f}h exceeds W9 max "
            f"{pim.WORKPLACE_CLUSTER_WINDOW_MAX_H}h"
        )


def test_fit_plug_ins_workplace_beta2_higher_than_residential() -> None:
    """Acceptance #10: workplace β₂ = 2.8 > residential β₂ = 2.5."""
    assert pim.BETA_WORKPLACE["beta_2"] == 2.8
    assert pim.BETA_RESIDENTIAL["beta_2"] == 2.5
    assert pim.BETA_WORKPLACE["beta_2"] > pim.BETA_RESIDENTIAL["beta_2"]
    # PROVENANCE table contains an entry naming the S-2 sign flip.
    flips = [
        r for r in pim.PROVENANCE
        if r.parameter == "beta_2_workplace"
    ]
    assert len(flips) == 1
    assert "FLIPPED" in flips[0].source.upper() or "flipped" in flips[0].source


@_needs_workplace_cohort
def test_fit_plug_ins_workplace_aggregate_rate_in_band(
    m5_workplace: tuple[pd.DataFrame, pd.DataFrame, pim.ClusterFit],
    mobility_workplace: pd.DataFrame,
) -> None:
    """Acceptance #11: workplace aggregate plug-in rate ∈ [0.70, 0.95]."""
    _, events, _ = m5_workplace
    rate = pim.aggregate_plug_in_rate(mobility_workplace, events, "workplace")
    assert 0.70 <= rate <= 0.95, (
        f"workplace aggregate plug-in rate {rate:.3f} outside [0.70, 0.95]"
    )


# --------------------------------------------------------------------------- #
# β-anchor tests (revised plan §4.4.4 / §5.1).
# --------------------------------------------------------------------------- #


def test_compute_p_plug_workplace_anchor_in_band() -> None:
    """Plan §4.4.4 anchor: σ() at (SoC=0.55, on-cluster, dwell=8h) ≈ 0.995.

    The PM brief test specifies the band [0.85, 0.99]; the plan caveat
    explicitly tolerates the σ(5.26)≈0.995 anchor.  We band-check at
    [0.85, 0.9955] to honour both.
    """
    p = pim.compute_p_plug(
        soc_at_arrival_frac=0.55,
        is_correct_location=True,
        in_preferred_window=True,
        dwell_h=8.0,
        is_weekend=False,
        K=3, z=2,
        profile_type="workplace",
    )
    assert 0.85 <= p <= 0.9955, (
        f"workplace anchor σ()={p:.4f} outside the plan-tolerated [0.85, 0.9955] band"
    )
    # Off-cluster (β₃ suppressed): σ(3.76) ≈ 0.977.
    p_off = pim.compute_p_plug(
        soc_at_arrival_frac=0.55,
        is_correct_location=True,
        in_preferred_window=False,
        dwell_h=8.0,
        is_weekend=False,
        K=3, z=2,
        profile_type="workplace",
    )
    assert p_off < p
    # Weekend (β₅=-3.0): off-cluster + weekend should drop sharply.
    p_we_off = pim.compute_p_plug(
        soc_at_arrival_frac=0.55,
        is_correct_location=True,
        in_preferred_window=False,
        dwell_h=8.0,
        is_weekend=True,
        K=3, z=2,
        profile_type="workplace",
    )
    # σ(-0.2 + 0.54 + 2.8 + 0 + 0.624 - 3.0) = σ(0.764) ≈ 0.68.
    # Strict band: off-cluster weekend should at minimum be below the
    # in-cluster weekday value by ≥ 0.20 (the β₃ + β₅ combined hit).
    assert p_we_off < p - 0.20, (
        f"workplace off-cluster weekend σ()={p_we_off:.4f} not "
        f"strongly dampened relative to on-cluster weekday {p:.4f}"
    )


def test_compute_p_plug_residential_anchor_in_band() -> None:
    """Plan §5.1 anchor: evening cluster (K=3, z=2), dwell=14h, SoC=0.6 → σ ≈ 0.997."""
    p = pim.compute_p_plug(
        soc_at_arrival_frac=0.6,
        is_correct_location=True,
        in_preferred_window=True,
        dwell_h=14.0,
        is_weekend=False,
        K=3, z=2,
        profile_type="residential",
    )
    assert 0.99 <= p <= 1.0
    # Numeric: x = -0.5 + 1.8 * 0.4 + 2.5 + 1.5 + 0.6 * ln(14)
    x_exp = -0.5 + 1.8 * 0.4 + 2.5 + 1.5 + 0.6 * float(np.log(14.0))
    expected = 1.0 / (1.0 + float(np.exp(-x_exp)))
    assert abs(p - expected) < 1e-4


@_needs_workplace_cohort
def test_no_workplace_plug_in_when_dest_type_not_work(
    small_fleet: pd.DataFrame, mobility_workplace: pd.DataFrame,
) -> None:
    """Workplace plug-in events only fire on `dest_type == 'work'` trips."""
    _, events, _ = pim.fit_and_assign_plug_ins(
        fleet=small_fleet,
        mobility=mobility_workplace,
        region="bay_area",
        profile_type="workplace",
        cluster_source="bic_sweep",
        seed=pim.MASTER_SEED_DEFAULT,
        cohort_parquet=_WORKPLACE_COHORT,
    )
    # All event location_type values must be "work" (or NACS_workplace,
    # but we do not tag that in v2.0-rev without an EVSE_kw routing).
    assert (events["location_type"].astype(str) == "work").all()
    # And every event must point to a UTC arrival.
    assert str(events["arrival_ts"].dtype) == "datetime64[ns, UTC]"


def test_weekend_dampening_workplace_stronger_than_residential() -> None:
    """β₅(workplace) = −3.0 ≪ β₅(residential) = −0.2.

    Implication: same arrival on a weekend should be far less likely
    to plug in for workplace than for residential.
    """
    assert pim.BETA_WORKPLACE["beta_5"] == -3.0
    assert pim.BETA_RESIDENTIAL["beta_5"] == -0.2
    res = pim.compute_p_plug(
        soc_at_arrival_frac=0.5,
        is_correct_location=True,
        in_preferred_window=True,
        dwell_h=10.0,
        is_weekend=True,
        K=3, z=2,
        profile_type="residential",
    )
    wp = pim.compute_p_plug(
        soc_at_arrival_frac=0.5,
        is_correct_location=True,
        in_preferred_window=True,
        dwell_h=10.0,
        is_weekend=True,
        K=3, z=2,
        profile_type="workplace",
    )
    assert wp < res, (
        f"workplace weekend σ()={wp:.4f} not lower than residential {res:.4f}"
    )


# --------------------------------------------------------------------------- #
# Per-EV cluster assignment unit tests.
# --------------------------------------------------------------------------- #


def test_assign_clusters_uses_K_from_fit(small_fleet: pd.DataFrame) -> None:
    """`assign_clusters` produces z ∈ {1..K_chosen} for every EV and
    matches the K it was given."""
    for K in (3, 4, 6):
        z = pim.assign_clusters(
            small_fleet, K_chosen=K, profile_type="residential",
            master_seed=pim.MASTER_SEED_DEFAULT,
        )
        assert len(z) == len(small_fleet)
        assert z.dtype == np.int8
        assert int(z.min()) >= 1
        assert int(z.max()) <= K
    for K in (2, 3, 4, 5):
        z = pim.assign_clusters(
            small_fleet, K_chosen=K, profile_type="workplace",
            master_seed=pim.MASTER_SEED_DEFAULT,
        )
        assert int(z.min()) >= 1 and int(z.max()) <= K


def test_assign_clusters_is_per_ev_seeded(small_fleet: pd.DataFrame) -> None:
    """Permuting fleet rows must not change z (z is keyed on ev_id)."""
    z_a = pim.assign_clusters(small_fleet, K_chosen=3, profile_type="residential")
    perm = small_fleet.sample(frac=1.0, random_state=99).reset_index(drop=True)
    z_b = pim.assign_clusters(perm, K_chosen=3, profile_type="residential")
    aligned = np.empty_like(z_a)
    for i, ev in enumerate(small_fleet["ev_id"].astype(int).tolist()):
        pos = perm.index[perm["ev_id"] == ev][0]
        aligned[i] = z_b[pos]
    assert np.array_equal(z_a, aligned)


# --------------------------------------------------------------------------- #
# Reproducibility.
# --------------------------------------------------------------------------- #


def test_seed_reproducible_residential(
    small_fleet: pd.DataFrame, mobility_residential: pd.DataFrame, tmp_path: Path,
) -> None:
    """Two runs with the same seed produce byte-identical events parquet."""
    a_fleet, a_events, _ = pim.fit_and_assign_plug_ins(
        fleet=small_fleet,
        mobility=mobility_residential,
        region="bay_area",
        profile_type="residential",
        seed=pim.MASTER_SEED_DEFAULT,
    )
    b_fleet, b_events, _ = pim.fit_and_assign_plug_ins(
        fleet=small_fleet,
        mobility=mobility_residential,
        region="bay_area",
        profile_type="residential",
        seed=pim.MASTER_SEED_DEFAULT,
    )
    p_a = tmp_path / "ev_a.parquet"
    p_b = tmp_path / "ev_b.parquet"
    pim.write_plug_in_events(a_events, p_a)
    pim.write_plug_in_events(b_events, p_b)
    sha_a = hashlib.sha256(p_a.read_bytes()).hexdigest()
    sha_b = hashlib.sha256(p_b.read_bytes()).hexdigest()
    assert sha_a == sha_b
    # Fleet cluster_z column is also reproducible.
    pd.testing.assert_series_equal(a_fleet["cluster_z"], b_fleet["cluster_z"])


# --------------------------------------------------------------------------- #
# Provenance.
# --------------------------------------------------------------------------- #


def test_provenance_table_documents_workplace_beta_priors() -> None:
    """PROVENANCE block must name every workplace β prior AND the W9 invariant."""
    needed = {
        "beta_2_workplace",
        "beta_3_workplace",
        "beta_5_workplace",
        "beta_0_workplace_K3",
        "workplace_cluster_window_max_h",
    }
    found = {r.parameter for r in pim.PROVENANCE}
    missing = needed - found
    assert not missing, f"PROVENANCE missing entries: {missing}"
    # Kreft 2024 must be cited as the workplace β₁ source (NOT Köhler 2024).
    sources = " | ".join(r.source for r in pim.PROVENANCE)
    assert "Kreft" in sources, "Kreft 2024 missing from PROVENANCE sources"
    # The "Köhler 2024" mis-citation must NOT appear as a positive source —
    # it MAY appear in a "NOT 'Köhler 2024'" note clarifying the correction.
    for r in pim.PROVENANCE:
        bad = "köhler" in r.source.lower() or "kohler" in r.source.lower()
        if bad:
            assert "NOT" in r.source or "Kreft" in r.source, (
                f"PROVENANCE row {r.parameter} cites Köhler 2024 without correction"
            )


def test_irregular_cluster_window_covers_all_24h_residential() -> None:
    """K=4 z=4 (residential) is the irregular cluster — every hour in-window."""
    for hour in range(24):
        assert pim._in_cluster_window(
            float(hour),
            z=4,
            centres=pim.WINDOW_CENTRES_RESIDENTIAL[4],
            K=4,
            profile_type="residential",
        )


def test_workplace_has_no_irregular_cluster() -> None:
    """Workplace clusters are bounded — no 24h-irregular catch-all (RC-3)."""
    # IRREGULAR_CLUSTER_IDS_BY_K is residential-only.  At evaluation time,
    # workplace profile_type must NOT trigger the irregular short-circuit.
    for K in (2, 3, 4, 5):
        # An hour far from any workplace centre must be out of window.
        far_hour = 22.0  # 10pm — far from any workplace centre at 08-12
        for z in range(1, K + 1):
            centres = (
                pim.WINDOW_CENTRES_WORKPLACE_PRIOR.get(K, (8.0, 10.0))
            )
            in_win = pim._in_cluster_window(
                far_hour, z=z, centres=centres, K=K, profile_type="workplace",
            )
            # Workplace centres are 7-13; 22:00 is > 8h away → never in window.
            assert in_win is False, (
                f"workplace K={K} z={z} at 22:00 should not be in window"
            )


# --------------------------------------------------------------------------- #
# Convenience: BIC sweep on workplace cohort (no fleet needed).
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    not _WORKPLACE_COHORT.is_file(),
    reason="workplace_subset.parquet not present",
)
def test_run_bic_sweep_workplace_cohort_directly() -> None:
    """`run_bic_sweep` on the workplace cohort returns a populated ClusterFit."""
    cohort = pd.read_parquet(_WORKPLACE_COHORT)
    fit = pim.run_bic_sweep(
        cohort,
        profile_type="workplace",
        feature_cols=("plug_in_hr_med", "dur_h_med", "avg_kw_med"),
        K_sweep=pim.K_SWEEP_WORKPLACE,
        seed=pim.MASTER_SEED_DEFAULT,
    )
    assert fit.K_chosen in pim.K_SWEEP_WORKPLACE
    assert not fit.degenerate_flag
    assert len(fit.cluster_centres_h) == fit.K_chosen
    assert len(fit.cluster_weights) == fit.K_chosen
    # W9: every selected cluster window ≤ 6h.
    for w in fit.cluster_windows_h:
        assert w <= pim.WORKPLACE_CLUSTER_WINDOW_MAX_H + 1e-6


def test_cluster_window_h_helper() -> None:
    """_cluster_window_h returns 95% IPR for ≥4 members, else (max-min)."""
    small = np.array([8.0, 8.5, 9.0])
    assert pim._cluster_window_h(small) == 1.0
    big = np.array([7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5])
    # 95% IPR = p97.5 - p2.5
    ipr = float(np.percentile(big, 97.5) - np.percentile(big, 2.5))
    assert abs(pim._cluster_window_h(big) - ipr) < 1e-9


# --------------------------------------------------------------------------- #
# v3.0 SPEECh K=16 loader tests.
# --------------------------------------------------------------------------- #


_needs_speech_residential = pytest.mark.skipif(
    not _SPEECH_RESIDENTIAL.is_file(),
    reason=(
        f"SPEECh residential JSON not present at {_SPEECH_RESIDENTIAL}; "
        "run data/pev/raw/speech/_convert_pickles.py first"
    ),
)
_needs_speech_workplace = pytest.mark.skipif(
    not _SPEECH_WORKPLACE.is_file(),
    reason=(
        f"SPEECh workplace JSON not present at {_SPEECH_WORKPLACE}; "
        "run data/pev/raw/speech/_convert_pickles.py first"
    ),
)


@_needs_speech_residential
def test_load_speech_k16_params_residential() -> None:
    """Loader returns a populated SpeechParams with expected shapes."""
    sp = pim.load_speech_k16_params("residential")
    assert sp.profile_type == "residential"
    assert len(sp.P_G) == pim.SPEECH_K_TOTAL == 16
    # All renormalised P(G) values are finite and non-negative; sum ≈ 1.0.
    assert all(np.isfinite(p) and p >= 0.0 for p in sp.P_G)
    assert abs(sum(sp.P_G) - 1.0) < 1e-9
    # Weekday and weekend group lists are non-empty subsets of 0..15.
    assert 1 <= len(sp.groups_present_weekday) <= 16
    assert 1 <= len(sp.groups_present_weekend) <= 16
    for g in sp.groups_present_weekday + sp.groups_present_weekend:
        assert 0 <= g <= 15
    # Each present group has at least one GMM component on its day_type,
    # all means finite in hours-LT, all covs non-negative (variances).
    assert len(sp.weekday_means_h) == len(sp.groups_present_weekday)
    assert len(sp.weekday_covs_h) == len(sp.groups_present_weekday)
    assert len(sp.weekday_weights) == len(sp.groups_present_weekday)
    for means, covs, weights in zip(
        sp.weekday_means_h, sp.weekday_covs_h, sp.weekday_weights, strict=True,
    ):
        assert len(means) >= 1
        assert len(means) == len(covs) == len(weights)
        assert all(np.isfinite(m) for m in means)
        # Variances are non-negative (start-time axis is the (0,0) entry
        # of the 3x3 SPEECh covariance and is always var ≥ 0 modulo
        # float-drift; bound at -1e-9 for safety).
        assert all(c >= -1e-9 for c in covs)
        assert all(w >= 0.0 for w in weights)


@_needs_speech_workplace
def test_load_speech_k16_params_workplace() -> None:
    """Workplace SPEECh loader produces the same structural invariants."""
    sp = pim.load_speech_k16_params("workplace")
    assert sp.profile_type == "workplace"
    assert len(sp.P_G) == 16
    assert abs(sum(sp.P_G) - 1.0) < 1e-9
    # Workplace `work` segment has 13 weekday + 12 weekend present groups
    # per the S1 conversion (see the SPEECh K16 modeling notes).
    assert len(sp.groups_present_weekday) >= 10
    assert len(sp.groups_present_weekend) >= 10


@_needs_speech_residential
def test_speech_cluster_fit_axes_and_union(
    small_fleet: pd.DataFrame, mobility_residential: pd.DataFrame,
) -> None:
    """v3.0 default SPEECh path: ClusterFit.axes=='start_h_LT' and
    ``speech_params`` is attached; ``K_chosen`` matches the weekday∪weekend
    union size."""
    _, _, fit = pim.fit_and_assign_plug_ins(
        fleet=small_fleet,
        mobility=mobility_residential,
        region="bay_area",
        profile_type="residential",
        seed=pim.MASTER_SEED_DEFAULT,
    )
    assert fit.axes == ("start_h_LT",)
    assert fit.speech_params is not None
    expected_K = len(
        set(fit.speech_params.groups_present_weekday)
        | set(fit.speech_params.groups_present_weekend)
    )
    assert fit.K_chosen == expected_K
    assert len(fit.cluster_centres_h) == expected_K
    assert len(fit.cluster_weights) == expected_K


def test_load_speech_k16_params_falls_back_to_bundled(tmp_path: Path) -> None:
    """With no locally-derived copy under ``data_root``, the loader transparently
    falls back to the SPEECh K=16 JSON bundled as package data (so a fresh
    ``pip install`` can build). ``tmp_path`` has no ``data/pev/raw/speech`` tree.
    """
    # Resolution points at the bundled package-data copy, which exists.
    resolved = pim._resolve_speech_json_path("residential", tmp_path)
    assert resolved == pim._bundled_speech_json_path("residential")
    assert resolved.is_file()
    # And the load succeeds rather than raising.
    sp = pim.load_speech_k16_params("residential", data_root=tmp_path)
    assert sp.profile_type == "residential"
    assert len(sp.P_G) == pim.SPEECH_K_TOTAL


def test_load_speech_k16_params_missing_everywhere_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If neither a local nor the bundled copy exists (a broken install), the
    loader still raises a clear FileNotFoundError."""
    monkeypatch.setattr(pim, "_BUNDLED_SPEECH_DIR", tmp_path / "nonexistent")
    with pytest.raises(FileNotFoundError, match="broken install"):
        pim.load_speech_k16_params("residential", data_root=tmp_path)
