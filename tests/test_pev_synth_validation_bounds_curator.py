"""Tests for ``pev_synth.validation_bounds_curator`` (M8).

Covers the acceptance criteria from the M8 dev brief plus the four phase6
expert-review blocking fixes (Fix 1-4):

* Every check parquet carries every required column.
* No row sources from the disqualified Pecan Street May 2025 article.
* The ``_manifest.json`` enumerates every check parquet that was written.
* Rows flagged ``unbounded_v1 = true`` include a non-empty
  ``provenance_note`` explaining why no published bound was available.
* Every ``source_url`` matches a plausible URL regex.
* Two invocations of ``curate_bounds`` produce byte-identical parquets and a
  byte-identical manifest (reproducibility check).
* The shared-contract ``check_id`` column is the first column and matches
  the filename stem on every row (Fix 4 / B7).
* Every EPA Find.do URL resolves to the cited vehicle archetype, asserted
  against an in-module snapshot to avoid hitting the network in CI (Fix 1).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from pev_synth.validation_bounds_curator import (
    ALL_CHECK_IDS,
    BANNED_SOURCE_FRAGMENTS,
    CHECK_IDS,
    DST_CHECK_IDS,
    EPA_ARCHETYPE_EXPECTATIONS,
    EPA_URL_SNAPSHOT,
    REQUIRED_COLUMNS,
    WORKPLACE_CHECK_IDS,
    curate_bounds,
)

# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def bounds_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run the curator once per module and reuse the output for all tests."""

    out = tmp_path_factory.mktemp("bounds")
    curate_bounds(out)
    return out


@pytest.fixture(scope="module")
def bounds_dir(bounds_root: Path) -> Path:
    """Per-region / per-profile-type subdir used by per-check parquet tests.

    Post RFC-009.r the curator writes ONLY to the nested layout
    ``<bounds_root>/<region>/<profile_type>/<check_id>.parquet``. Tests
    that previously read flat-layout parquets from ``<bounds_root>/`` now
    read from the bay_area / workplace subdir (a stable v2.0 cache
    location that carries every applicable check_id).
    """
    return bounds_root / "bay_area" / "workplace"


def _resolve_check_path(bounds_root: Path, check_id: str) -> Path:
    """Return the path to ``<check_id>.parquet`` in the first applicable
    nested subdir under bay_area/{workplace,residential}.

    Workplace is preferred (it historically contained every check id);
    v2.1 added a handful of residential-only checks that only land under
    ``bay_area/residential/``.
    """

    for profile_type in ("workplace", "residential"):
        candidate = bounds_root / "bay_area" / profile_type / f"{check_id}.parquet"
        if candidate.exists():
            return candidate
    return bounds_root / "bay_area" / "workplace" / f"{check_id}.parquet"


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #


def test_every_check_has_required_columns(bounds_root: Path) -> None:
    """Every check parquet must expose every column in :data:`REQUIRED_COLUMNS`
    and must contain at least one row.

    v2.1 — uses :func:`_resolve_check_path` so profile-type-specific
    checks (``evse_brand_distribution_{residential,workplace}``,
    ``bundle_rule_take_rates``) are located in the correct subdir.
    """

    for check_id in ALL_CHECK_IDS:
        path = _resolve_check_path(bounds_root, check_id)
        assert path.exists(), f"missing parquet for {check_id} at {path}"
        table = pq.read_table(path)
        missing = set(REQUIRED_COLUMNS) - set(table.column_names)
        assert not missing, f"{check_id} missing columns: {missing}"
        assert table.num_rows >= 1, f"{check_id} has zero rows"

        # Required-not-null columns must be fully populated.
        for col in (
            "check_id",
            "metric_name",
            "unit",
            "source_url",
            "source_title",
            "source_table_or_figure",
            "digitisation_date",
            "extraction_method",
            "unbounded_v1",
        ):
            arr = table.column(col)
            assert arr.null_count == 0, (
                f"{check_id}.{col} has {arr.null_count} nulls"
            )


def test_no_pecan_street_2025_as_source(bounds_root: Path) -> None:
    """RC-6 hard rule: the Pecan Street May 2025 article is disqualified as
    a numeric-bound source. Assert that no row cites it.
    """

    for check_id in ALL_CHECK_IDS:
        table = pq.read_table(_resolve_check_path(bounds_root, check_id))
        urls = table.column("source_url").to_pylist()
        titles = table.column("source_title").to_pylist()
        for url, title in zip(urls, titles, strict=True):
            lower_url = url.lower()
            for banned in BANNED_SOURCE_FRAGMENTS:
                assert banned not in lower_url, (
                    f"{check_id} cites banned Pecan Street URL {url!r}"
                )
            # Defensive check on the title field too.
            assert "pecan street" not in title.lower() or "may 2025" not in title.lower(), (
                f"{check_id} title appears to cite the banned Pecan Street "
                f"May 2025 article: {title!r}"
            )


def test_manifest_lists_all_check_parquets(
    bounds_root: Path, bounds_dir: Path
) -> None:
    """The manifest (at ``<bounds_root>/_manifest.json``) must list every
    check id that is present on disk, and every parquet must be reachable
    from the manifest sha256. Post RFC-009.r each parquet lives at
    ``<bounds_root>/<region>/<profile_type>/<filename>``.

    v3.0 — region-specific checks (currently just
    ``battery_capacity_distribution``) carry a ``sha256_by_region`` map on
    their manifest entry; per-region parquet bytes differ. Shared checks
    still carry a single ``sha256`` that matches every on-disk copy. The
    test below verifies the SHA against ``sha256_by_region["bay_area"]``
    when present, falling back to the shared ``sha256`` for non-region-
    specific checks.
    """

    manifest = json.loads((bounds_root / "_manifest.json").read_text())
    manifest_ids = {entry["check_id"] for entry in manifest["checks"]}
    assert manifest_ids == set(ALL_CHECK_IDS), (
        f"manifest check ids {manifest_ids} != {set(ALL_CHECK_IDS)}"
    )

    for entry in manifest["checks"]:
        path = _resolve_check_path(bounds_root, entry["check_id"])
        assert path.exists(), f"manifest references missing parquet {path}"
        # v3.0: prefer per-region sha when published; the parquet under
        # ``bay_area/<profile>/`` carries the bay_area regional bytes.
        sha_by_region = entry.get("sha256_by_region")
        if sha_by_region:
            expected_sha = sha_by_region["bay_area"]
        else:
            expected_sha = entry["sha256"]
        actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual_sha == expected_sha, (
            f"sha256 mismatch for {entry['parquet']}: "
            f"manifest={expected_sha}, disk={actual_sha}"
        )


def test_unbounded_v1_rows_have_provenance_note(bounds_root: Path) -> None:
    """Any row flagged ``unbounded_v1 = True`` must carry a non-empty
    ``provenance_note`` explaining why no bound was available.
    """

    for check_id in ALL_CHECK_IDS:
        table = pq.read_table(_resolve_check_path(bounds_root, check_id))
        unbounded = table.column("unbounded_v1").to_pylist()
        notes = table.column("provenance_note").to_pylist()
        metrics = table.column("metric_name").to_pylist()
        for metric, u, note in zip(metrics, unbounded, notes, strict=True):
            if u:
                assert note is not None and note.strip(), (
                    f"{check_id}.{metric}: unbounded_v1 row needs a "
                    f"provenance_note"
                )


_URL_RE = re.compile(r"^https?://[^\s]+$")


def test_source_urls_are_valid_format(bounds_root: Path) -> None:
    """Every ``source_url`` must match a plausible URL regex (cheap CI check;
    live HEAD requests are out of scope per the brief).
    """

    for check_id in ALL_CHECK_IDS:
        table = pq.read_table(_resolve_check_path(bounds_root, check_id))
        urls = table.column("source_url").to_pylist()
        for url in urls:
            assert _URL_RE.match(url), (
                f"{check_id}: source_url {url!r} fails regex validation"
            )


def test_reruns_are_byte_identical(
    tmp_path: Path,
) -> None:
    """The brief requires deterministic output: two invocations of
    ``curate_bounds`` on fresh directories must produce byte-identical
    parquets and manifest files. Post RFC-009.r we compare the parquets
    inside the nested ``bay_area / workplace`` subdir (every applicable
    check parquet is byte-identical across regions / profile_types).
    """

    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    curate_bounds(out_a)
    curate_bounds(out_b)

    nested_a = out_a / "bay_area" / "workplace"
    nested_b = out_b / "bay_area" / "workplace"
    for check_id in ALL_CHECK_IDS:
        path_a = nested_a / f"{check_id}.parquet"
        path_b = nested_b / f"{check_id}.parquet"
        if not path_a.exists():
            # residential-only check; verify under residential subdir instead.
            path_a = out_a / "bay_area" / "residential" / f"{check_id}.parquet"
            path_b = out_b / "bay_area" / "residential" / f"{check_id}.parquet"
        bytes_a = path_a.read_bytes()
        bytes_b = path_b.read_bytes()
        assert hashlib.sha256(bytes_a).digest() == hashlib.sha256(bytes_b).digest(), (
            f"{check_id}.parquet is not byte-identical across runs"
        )

    manifest_a = (out_a / "_manifest.json").read_bytes()
    manifest_b = (out_b / "_manifest.json").read_bytes()
    assert manifest_a == manifest_b, "_manifest.json differs between runs"


def test_bounds_ordered_when_present(bounds_root: Path) -> None:
    """Where both ``lower_bound`` and ``upper_bound`` are populated,
    ``lower_bound <= upper_bound``. Where ``value`` is also populated, it
    should fall inside [lower_bound, upper_bound].
    """

    for check_id in ALL_CHECK_IDS:
        table = pq.read_table(_resolve_check_path(bounds_root, check_id))
        values = table.column("value").to_pylist()
        lowers = table.column("lower_bound").to_pylist()
        uppers = table.column("upper_bound").to_pylist()
        metrics = table.column("metric_name").to_pylist()
        for metric, v, lo, hi in zip(metrics, values, lowers, uppers, strict=True):
            if lo is not None and hi is not None:
                assert lo <= hi, f"{check_id}.{metric}: lower {lo} > upper {hi}"
            if v is not None and lo is not None:
                assert lo <= v + 1e-9, (
                    f"{check_id}.{metric}: value {v} < lower {lo}"
                )
            if v is not None and hi is not None:
                assert v <= hi + 1e-9, (
                    f"{check_id}.{metric}: value {v} > upper {hi}"
                )


def test_every_expected_check_parquet_present(
    bounds_root: Path, bounds_dir: Path
) -> None:
    """At least 22 check parquets exist (v1.1 11 residential + v2.0 10
    workplace + 1 DST). Combined with the manifest test, this nails down
    the exact set. Post RFC-009.r the parquets live in the nested
    ``<bounds_root>/<region>/<profile_type>/`` layout — every applicable
    check id appears under ``bay_area/workplace`` (the workplace subtree
    receives every check) and every residential-only check appears under
    ``bay_area/residential``.
    """

    workplace_parquets = sorted(p.name for p in bounds_dir.glob("*.parquet"))
    residential_parquets = sorted(
        p.name
        for p in (bounds_root / "bay_area" / "residential").glob("*.parquet")
    )
    combined = set(workplace_parquets) | set(residential_parquets)
    assert len(combined) >= 22, (
        f"expected >= 22 check parquets, got {sorted(combined)}"
    )
    for check_id in ALL_CHECK_IDS:
        assert f"{check_id}.parquet" in combined, (
            f"missing {check_id}.parquet across both nested subtrees"
        )


# --------------------------------------------------------------------------- #
# Fix 4 — check_id column                                                     #
# --------------------------------------------------------------------------- #


def test_check_id_column_present_and_matches_filename(
    bounds_root: Path,
) -> None:
    """Per PM review B7 (Fix 4): every parquet must carry a ``check_id``
    column as its FIRST column, and every row's value must equal the
    filename stem.
    """

    for check_id in ALL_CHECK_IDS:
        path = _resolve_check_path(bounds_root, check_id)
        table = pq.read_table(path)

        # check_id is the first column.
        assert table.column_names[0] == "check_id", (
            f"{check_id}.parquet: first column is "
            f"{table.column_names[0]!r}, expected 'check_id'"
        )

        # Every row's check_id matches the filename stem.
        ids = table.column("check_id").to_pylist()
        assert ids, f"{check_id}.parquet has no rows"
        for value in ids:
            assert value == check_id, (
                f"{check_id}.parquet row has check_id={value!r}, "
                f"expected {check_id!r}"
            )


# --------------------------------------------------------------------------- #
# Fix 1 — EPA URL snapshot resolution                                         #
# --------------------------------------------------------------------------- #


def test_every_epa_url_resolves_to_correct_vehicle(bounds_dir: Path) -> None:
    """For every row in ``wh_per_mi_at_70F.parquet`` whose ``source_url`` is
    an EPA fueleconomy.gov ``Find.do`` link, assert that:

    * The URL is present in :data:`EPA_URL_SNAPSHOT` (a snapshot built by
      the dev at fix time via WebFetch — see the module docstring).
    * The snapshot's resolved vehicle identity contains every expected
      token for the cited archetype (per
      :data:`EPA_ARCHETYPE_EXPECTATIONS`).

    This catches the phase6 review's "fabricated EPA ID" failure mode
    without hitting the network in CI.
    """

    table = pq.read_table(bounds_dir / "wh_per_mi_at_70F.parquet")
    metric_names = table.column("metric_name").to_pylist()
    urls = table.column("source_url").to_pylist()

    epa_prefix = "https://www.fueleconomy.gov/feg/Find.do"
    seen_archetypes: set[str] = set()
    for metric, url in zip(metric_names, urls, strict=True):
        if not url.startswith(epa_prefix):
            continue
        # Drop the "wh_per_mi_" prefix to recover the archetype id.
        assert metric.startswith("wh_per_mi_"), (
            f"unexpected EPA-cited metric_name {metric!r}"
        )
        archetype = metric[len("wh_per_mi_"):]
        seen_archetypes.add(archetype)

        # URL is in the snapshot.
        assert url in EPA_URL_SNAPSHOT, (
            f"{archetype}: source_url {url!r} is not in EPA_URL_SNAPSHOT; "
            f"refresh the snapshot via WebFetch before committing a new ID."
        )

        # Snapshot resolves to a vehicle matching the cited archetype.
        resolved = EPA_URL_SNAPSHOT[url]
        expected_tokens = EPA_ARCHETYPE_EXPECTATIONS.get(archetype)
        assert expected_tokens is not None, (
            f"{archetype}: no expectation registered in "
            f"EPA_ARCHETYPE_EXPECTATIONS"
        )
        for token in expected_tokens:
            assert token.lower() in resolved.lower(), (
                f"{archetype}: snapshot {resolved!r} for {url} is missing "
                f"expected token {token!r}. The Find.do id resolves to the "
                f"WRONG vehicle — fix the ID, not the test."
            )

    # Every archetype with a registered expectation must have actually been
    # checked (catches typos in metric_name that would silently bypass).
    assert seen_archetypes == set(EPA_ARCHETYPE_EXPECTATIONS), (
        f"archetypes covered by EPA URL check {seen_archetypes} != "
        f"registered {set(EPA_ARCHETYPE_EXPECTATIONS)}"
    )


def test_no_borlaug_2023_slide33_obc_citation(bounds_dir: Path) -> None:
    """Per Fix 2 / B5: the Borlaug 2023 (NREL/PR-5400-87021) slide-33 OBC
    citation was confirmed non-existent by the phase6 expert review. The
    OBC parquet must not cite that PDF as a numeric-share source.
    """

    table = pq.read_table(bounds_dir / "obc_distribution.parquet")
    urls = table.column("source_url").to_pylist()
    unbounded = table.column("unbounded_v1").to_pylist()
    for url, u in zip(urls, unbounded, strict=True):
        # The Borlaug PR-5400-87021 deck must NOT be used to bound a numeric
        # OBC share. If a row cites that URL, it must be unbounded_v1=True
        # AND we still prefer it not be used at all per the fix.
        assert "fy24osti/87021" not in url.lower(), (
            f"obc_distribution still cites the disproven Borlaug 2023 "
            f"slide 33 PDF as a numeric source: {url!r} (unbounded_v1={u})"
        )


def test_obc_distribution_all_unbounded_v1(bounds_dir: Path) -> None:
    """Per Fix 2: with the Borlaug citation removed and no replacement
    numeric source available, every OBC share row must be unbounded_v1=True
    and carry a provenance_note explaining the gap.
    """

    table = pq.read_table(bounds_dir / "obc_distribution.parquet")
    unbounded = table.column("unbounded_v1").to_pylist()
    notes = table.column("provenance_note").to_pylist()
    metrics = table.column("metric_name").to_pylist()
    for metric, u, note in zip(metrics, unbounded, notes, strict=True):
        assert u is True, (
            f"obc_distribution.{metric}: expected unbounded_v1=True "
            f"after Fix 2 (no published numeric OBC share table exists)"
        )
        assert note is not None and note.strip(), (
            f"obc_distribution.{metric}: unbounded_v1 row needs a "
            f"provenance_note"
        )


# --------------------------------------------------------------------------- #
# Fix 3 — Argonne per-bin PMF                                                 #
# --------------------------------------------------------------------------- #


def test_battery_capacity_distribution_has_per_bin_pmf(bounds_dir: Path) -> None:
    """Per Fix 3 / M-cross.1: ``battery_capacity_distribution.parquet`` must
    publish the per-bin Argonne sales-weighted PMF (not just a median +
    TVD threshold) so that M9 can evaluate the TVD test from the parquet
    alone.

    Asserts at least six bins are present, that each bin's value is a
    fraction in [0, 1], and that the masses sum to 1.0 within tolerance.
    """

    table = pq.read_table(bounds_dir / "battery_capacity_distribution.parquet")
    metric_names = table.column("metric_name").to_pylist()
    values = table.column("value").to_pylist()
    units = table.column("unit").to_pylist()

    pmf_rows: list[tuple[str, float]] = []
    for metric, value, unit in zip(metric_names, values, units, strict=True):
        if metric.startswith("pmf_bin_") and metric.endswith("_kwh"):
            assert unit == "fraction", (
                f"{metric} unit is {unit!r}, expected 'fraction'"
            )
            assert value is not None, f"{metric} value is null"
            assert 0.0 <= value <= 1.0, f"{metric} value {value} out of [0,1]"
            pmf_rows.append((metric, value))

    assert len(pmf_rows) >= 6, (
        f"expected >= 6 per-bin PMF rows, got {len(pmf_rows)}: {pmf_rows}"
    )
    total_mass = sum(v for _, v in pmf_rows)
    assert abs(total_mass - 1.0) < 1e-9, (
        f"per-bin PMF does not sum to 1.0: sum={total_mass}, rows={pmf_rows}"
    )


# --------------------------------------------------------------------------- #
# v3.0 — region-aware battery_capacity_distribution                           #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("region", "expected_source", "expected_url_fragment"),
    [
        ("bay_area", "cvrp_ca", "cleanvehiclerebate.org"),
        ("la_basin", "cvrp_ca", "cleanvehiclerebate.org"),
        ("new_york_metro", "nyserda_drive_clean", "nyserda.ny.gov"),
        ("boston", "argonne_national", "anl.gov"),
        ("chicago", "argonne_national", "anl.gov"),
        ("dallas_fort_worth", "argonne_national", "anl.gov"),
        ("seattle", "argonne_national", "anl.gov"),
    ],
)
def test_battery_capacity_distribution_is_region_aware(
    bounds_root: Path,
    region: str,
    expected_source: str,
    expected_url_fragment: str,
) -> None:
    """v3.0 — each region's ``battery_capacity_distribution.parquet`` is
    sourced from the region-specific sales-mix CSV per
    :data:`REGION_TO_BATTERY_SALES_MIX_SOURCE` (CVRP-CA for bay_area /
    la_basin, NYSERDA for new_york_metro, Argonne for everyone else).

    Verifies (a) the parquet exists at
    ``<bounds_root>/<region>/<profile_type>/battery_capacity_distribution.parquet``
    for both residential and workplace profile types, (b) the canonical
    ``tvd_vs_regional_sales_pmf`` v3 metric_name is published with bound
    [0, 0.05], (c) the back-compat ``tvd_vs_argonne_pmf`` alias is
    present and flagged ``unbounded_v1=True``, (d) every row's
    ``source_url`` carries the expected per-region URL fragment, and
    (e) the BEV PMF still has >= 6 bins and sums to 1.0.
    """
    from pev_synth.validation_bounds_curator import (
        REGION_TO_BATTERY_SALES_MIX_SOURCE,
    )

    assert REGION_TO_BATTERY_SALES_MIX_SOURCE[region] == expected_source

    for profile_type in ("residential", "workplace"):
        path = (
            bounds_root / region / profile_type / "battery_capacity_distribution.parquet"
        )
        assert path.exists(), (
            f"v3 region-aware battery PMF parquet missing for "
            f"{region}/{profile_type}: {path}"
        )
        table = pq.read_table(path)
        metric_names = table.column("metric_name").to_pylist()
        values = table.column("value").to_pylist()
        lowers = table.column("lower_bound").to_pylist()
        uppers = table.column("upper_bound").to_pylist()
        unbounded = table.column("unbounded_v1").to_pylist()
        urls = table.column("source_url").to_pylist()

        # (b) live v3 metric is published with the 5 pp band.
        live_idx = [
            i for i, n in enumerate(metric_names)
            if n == "tvd_vs_regional_sales_pmf"
        ]
        assert len(live_idx) == 1, (
            f"{region}/{profile_type}: expected exactly one live "
            f"'tvd_vs_regional_sales_pmf' row, got {len(live_idx)}"
        )
        i = live_idx[0]
        assert not unbounded[i], (
            f"{region}/{profile_type}: live tvd row must not be "
            f"unbounded_v1"
        )
        assert lowers[i] == 0.0 and abs(uppers[i] - 0.05) < 1e-12, (
            f"{region}/{profile_type}: tvd_vs_regional_sales_pmf bound "
            f"is [{lowers[i]}, {uppers[i]}], expected [0.0, 0.05]"
        )

        # (c) back-compat alias is present and flagged unbounded_v1.
        alias_idx = [
            i for i, n in enumerate(metric_names)
            if n == "tvd_vs_argonne_pmf"
        ]
        assert len(alias_idx) == 1, (
            f"{region}/{profile_type}: expected exactly one "
            f"'tvd_vs_argonne_pmf' back-compat alias, got {len(alias_idx)}"
        )
        assert unbounded[alias_idx[0]] is True, (
            f"{region}/{profile_type}: 'tvd_vs_argonne_pmf' alias must "
            f"be flagged unbounded_v1=True"
        )

        # (d) source URLs carry the per-region URL fragment.
        for url in urls:
            assert expected_url_fragment in url, (
                f"{region}/{profile_type}: source_url {url!r} missing "
                f"per-region fragment {expected_url_fragment!r}"
            )

        # (e) PMF still well-formed.
        pmf_rows = [
            (n, v) for n, v in zip(metric_names, values, strict=True)
            if n.startswith("pmf_bin_") and n.endswith("_kwh")
        ]
        assert len(pmf_rows) >= 6, (
            f"{region}/{profile_type}: expected >= 6 per-bin PMF rows, "
            f"got {len(pmf_rows)}"
        )
        total_mass = sum(v for _, v in pmf_rows if v is not None)
        assert abs(total_mass - 1.0) < 1e-9, (
            f"{region}/{profile_type}: per-bin PMF sum {total_mass} != 1.0"
        )


def test_battery_capacity_distribution_per_region_differs(
    bounds_root: Path,
) -> None:
    """v3.0 — the per-region battery PMF parquets are NOT byte-identical
    across regions when the underlying sales-mix CSV differs.

    Asserts SHA-256 distinct between (bay_area=CVRP), (new_york_metro=
    NYSERDA), and (chicago=Argonne). Regions sharing the same source
    (bay_area + la_basin both = CVRP-CA) must remain byte-identical.
    """

    def _sha(region: str) -> str:
        path = (
            bounds_root / region / "residential"
            / "battery_capacity_distribution.parquet"
        )
        return hashlib.sha256(path.read_bytes()).hexdigest()

    sha_bay = _sha("bay_area")
    sha_la = _sha("la_basin")
    sha_ny = _sha("new_york_metro")
    sha_chi = _sha("chicago")
    sha_bos = _sha("boston")

    # Same source ⇒ byte-identical.
    assert sha_bay == sha_la, (
        "bay_area + la_basin both use cvrp_ca; parquets must be "
        f"byte-identical (bay={sha_bay} la={sha_la})"
    )
    # Different sources ⇒ distinct.
    assert sha_bay != sha_ny, (
        "bay_area (cvrp_ca) and new_york_metro (nyserda) must have "
        "distinct parquet bytes"
    )
    assert sha_bay != sha_chi, (
        "bay_area (cvrp_ca) and chicago (argonne_national) must have "
        "distinct parquet bytes"
    )
    assert sha_ny != sha_chi, (
        "new_york_metro (nyserda) and chicago (argonne_national) must "
        "have distinct parquet bytes"
    )
    # Same source again ⇒ byte-identical.
    assert sha_chi == sha_bos, (
        "chicago + boston both use argonne_national; parquets must be "
        f"byte-identical (chi={sha_chi} bos={sha_bos})"
    )


def test_battery_capacity_manifest_carries_sha256_by_region(
    bounds_root: Path,
) -> None:
    """v3.0 — the manifest entry for the region-aware
    ``battery_capacity_distribution`` check carries a non-empty
    ``sha256_by_region`` map whose keys cover the 7 v2.0 regions and
    whose values match each region's on-disk parquet bytes.

    Non-region-specific checks (e.g. ``obc_distribution``) MUST NOT carry
    ``sha256_by_region``; their shared ``sha256`` matches every on-disk
    copy.
    """

    manifest = json.loads((bounds_root / "_manifest.json").read_text())

    entries = {entry["check_id"]: entry for entry in manifest["checks"]}
    battery = entries["battery_capacity_distribution"]
    assert "sha256_by_region" in battery, (
        "battery_capacity_distribution manifest entry must carry "
        "sha256_by_region in v3.0"
    )
    sha_by_region = battery["sha256_by_region"]
    expected_regions = {
        "bay_area", "la_basin", "new_york_metro",
        "boston", "chicago", "dallas_fort_worth", "seattle",
    }
    assert set(sha_by_region) == expected_regions, (
        f"sha256_by_region keys {set(sha_by_region)} != {expected_regions}"
    )
    for region, expected_sha in sha_by_region.items():
        for profile_type in ("residential", "workplace"):
            path = (
                bounds_root / region / profile_type
                / "battery_capacity_distribution.parquet"
            )
            assert path.exists(), f"missing per-region battery parquet: {path}"
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            assert actual == expected_sha, (
                f"sha256_by_region[{region}] = {expected_sha} but disk "
                f"sha for {profile_type} = {actual}"
            )

    # Non-region-specific checks: no sha256_by_region.
    obc = entries["obc_distribution"]
    assert "sha256_by_region" not in obc, (
        "obc_distribution is NOT region-specific; manifest must not "
        "carry sha256_by_region"
    )


def test_battery_capacity_bay_area_uses_cvrp_pmf_shape(bounds_root: Path) -> None:
    """v3.0 — bay_area's parquet must reflect the CVRP-CA PMF shape (the
    Tesla-heavy California rebate distribution skews mass into the
    [75, 85) kWh bin). Sanity-check that the [75,85) bin is the modal bin
    for bay_area, asserting the regional source actually drives the
    output rather than re-using the Argonne (national) shape.
    """

    table = pq.read_table(
        bounds_root / "bay_area" / "residential" / "battery_capacity_distribution.parquet"
    )
    metric_names = table.column("metric_name").to_pylist()
    values = table.column("value").to_pylist()
    pmf = {n: v for n, v in zip(metric_names, values, strict=True) if n.startswith("pmf_bin_")}

    modal = max(pmf, key=pmf.get)
    assert modal == "pmf_bin_75_85_kwh", (
        f"bay_area / CVRP-CA PMF modal bin should be [75, 85) kWh but "
        f"observed modal bin is {modal!r}; PMF={pmf}"
    )
    # CVRP-CA shape is markedly different from the Argonne national
    # rollup: in argonne_national the [100, 150] bin carries ~22% mass
    # (pickup-heavy national mix), whereas CVRP-CA's [100, 150] bin
    # carries ~10% (California buys fewer F-150 Lightnings / Silverado
    # EVs). Assert the bay_area parquet reflects the CA-specific shape.
    assert pmf["pmf_bin_100_150_kwh"] < 0.15, (
        f"bay_area / CVRP-CA [100, 150] kWh mass should be < 0.15 "
        f"(CA pickup share is low); observed {pmf['pmf_bin_100_150_kwh']}"
    )


# --------------------------------------------------------------------------- #
# v2.0 — workplace W1-W10 + DST bound parquets                                #
# --------------------------------------------------------------------------- #


def test_workplace_bound_parquets_have_required_columns(bounds_dir: Path) -> None:
    """User-brief unit test 1: every W1-W10 parquet exists and carries every
    v1.1 schema column with non-null source URLs."""

    assert len(WORKPLACE_CHECK_IDS) == 10, (
        f"expected 10 workplace W1-W10 check ids, got {len(WORKPLACE_CHECK_IDS)}"
    )
    for check_id in WORKPLACE_CHECK_IDS:
        path = bounds_dir / f"{check_id}.parquet"
        assert path.exists(), f"missing workplace parquet for {check_id}"
        table = pq.read_table(path)
        missing = set(REQUIRED_COLUMNS) - set(table.column_names)
        assert not missing, f"{check_id} missing columns: {missing}"
        assert table.num_rows >= 1, f"{check_id} has zero rows"
        # source_url is non-null and non-empty everywhere.
        urls = table.column("source_url").to_pylist()
        for url in urls:
            assert url and url.strip(), (
                f"{check_id}: empty source_url"
            )


def test_workplace_bound_parquets_no_pecan_street_2025(bounds_dir: Path) -> None:
    """User-brief unit test 2: no workplace bound row sources from the
    Pecan Street May 2025 article (banned per v1.1 contract / RC-6).
    """

    for check_id in WORKPLACE_CHECK_IDS:
        table = pq.read_table(bounds_dir / f"{check_id}.parquet")
        urls = table.column("source_url").to_pylist()
        titles = table.column("source_title").to_pylist()
        notes = table.column("provenance_note").to_pylist()
        for url, title, note in zip(urls, titles, notes, strict=True):
            lower_url = url.lower()
            lower_title = title.lower()
            lower_note = (note or "").lower()
            for banned in BANNED_SOURCE_FRAGMENTS:
                assert banned not in lower_url, (
                    f"{check_id} workplace bound cites banned Pecan Street "
                    f"URL {url!r}"
                )
            assert not (
                "pecan street" in lower_title and "may 2025" in lower_title
            ), (
                f"{check_id} workplace bound cites Pecan Street May 2025 "
                f"article in title: {title!r}"
            )
            assert not (
                "pecan street" in lower_note and "may 2025" in lower_note
            ), (
                f"{check_id} workplace bound cites Pecan Street May 2025 "
                f"article in provenance_note"
            )


def test_workplace_w1_explained_fail_cites_cohort_caveat(bounds_dir: Path) -> None:
    """User-brief unit test 3: the W1 workplace plug-in CDF bound parquet
    documents (in its ``provenance_note``) that an EXPLAINED_FAIL on this
    metric should cite M5's ``workplace_cohort_caveat``.

    The validator's actual EXPLAINED_FAIL wiring is tested in
    ``test_pev_synth_validator``; this test asserts the M8 bound parquet
    surfaces the caveat to the curator's downstream consumer.
    """

    table = pq.read_table(bounds_dir / "W1_workplace_plug_in_cdf.parquet")
    notes = table.column("provenance_note").to_pylist()
    combined_notes = " ".join(n or "" for n in notes).lower()
    assert "workplace_cohort_caveat" in combined_notes or (
        "cluster centres skew" in combined_notes
        and "explained_fail" in combined_notes
    ), (
        "W1 provenance_note must surface the workplace_cohort_caveat: "
        f"got notes = {notes!r}"
    )


def test_dst_verification_bound_parquet_present(bounds_dir: Path) -> None:
    """The DST verification parquet (M9 check #12) must exist and encode the
    expected zones (LA, NY, Chicago, UTC).
    """

    for check_id in DST_CHECK_IDS:
        path = bounds_dir / f"{check_id}.parquet"
        assert path.exists(), f"missing DST parquet for {check_id}"
        table = pq.read_table(path)
        metric_names = table.column("metric_name").to_pylist()
        # Expect one boolean row per zone.
        expected_slugs = (
            "los_angeles", "new_york", "chicago", "utc",
        )
        for slug in expected_slugs:
            assert any(slug in m for m in metric_names), (
                f"DST bound missing zone slug {slug!r}; metrics = {metric_names!r}"
            )


def test_manifest_tags_workplace_and_dst_check_kinds(
    bounds_root: Path,
) -> None:
    """The v2.0 manifest must tag every entry with ``check_kind`` and
    ``applicable_profile_types`` so the validator can route checks per
    cache. Manifest lives at ``<bounds_root>/_manifest.json`` (NOT inside
    the nested per-region subdirs)."""

    manifest = json.loads((bounds_root / "_manifest.json").read_text())
    by_id = {e["check_id"]: e for e in manifest["checks"]}
    for cid in CHECK_IDS:
        assert by_id[cid]["check_kind"] == "residential"
        assert "residential" in by_id[cid]["applicable_profile_types"]
    for cid in WORKPLACE_CHECK_IDS:
        assert by_id[cid]["check_kind"] == "workplace"
        assert by_id[cid]["applicable_profile_types"] == ["workplace"]
    for cid in DST_CHECK_IDS:
        assert by_id[cid]["check_kind"] == "dst"
        assert by_id[cid]["applicable_profile_types"] == [
            "residential",
            "workplace",
        ]
