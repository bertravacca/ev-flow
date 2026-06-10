"""Single typed writer for the per-cache ``meta.json`` artifact.

RFC-021: seven independent modules (M2 – M7 + cache_regen.finalise +
validator) currently read-modify-write ``meta.json`` with ad-hoc keys.
Symptoms include the known ``donor_borrow_rate`` nesting bug and lots of
silently-overlapping top-level keys.

This module:

* documents the canonical sub-namespaces (one per producer) in
  :class:`MetaSchema`,
* exposes :class:`MetaWriter` with ``setdefault``-semantics so the
  multi-producer pattern stays append-only.

Existing per-module helpers (``update_meta_with_m5``,
``update_meta_with_m7``, …) continue to work in v2.0-rev — they just
hand off the atomic write to :func:`MetaWriter.write`. The CLI surface
is untouched.

Math notation: ASCII.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "MetaSchema",
    "MetaWriter",
    "load_meta",
    "write_meta_atomic",
]


#: Canonical RMW (read-modify-write) stage keys for ``MetaWriter.update``.
#: Each maps a stage label to (sub_namespace_key, module_provenance_key).
RMW_STAGES: dict[str, tuple[str, str]] = {
    "donor_matcher": ("donor_matcher", "m3_donor_matcher"),
    "soc_trajectory": ("soc_trajectory", "m6_soc_trajectory"),
    "hourly_resampler": ("hourly_resampler", "m7_hourly_resampler"),
}


@dataclass
class MetaSchema:
    """Documented canonical schema for ``meta.json``.

    Each top-level key is owned by one producer; secondary writers MUST
    use ``setdefault`` semantics to avoid clobbering. This dataclass is
    informational — it does not enforce key presence at runtime (yet);
    runtime validation lives in :func:`pev_synth.validator.run`.
    """

    #: Set by M2 / cache_regen at bootstrap.
    methodology_version: str = ""
    master_seed: int = 0
    region: dict[str, Any] = field(default_factory=dict)
    profile_type: str = ""
    cache_provenance: str = ""
    storage_timezone: str = ""
    run_timestamp_utc: str = ""
    git_commit: str = ""
    #: One sub-namespace per producer module.
    module_provenance: dict[str, str] = field(default_factory=dict)
    vehicle_archetypes: dict[str, Any] = field(default_factory=dict)
    donor_matcher: dict[str, Any] = field(default_factory=dict)
    travel_week_builder: dict[str, Any] = field(default_factory=dict)
    plug_in_model: dict[str, Any] = field(default_factory=dict)
    soc_trajectory: dict[str, Any] = field(default_factory=dict)
    hourly_resampler: dict[str, Any] = field(default_factory=dict)
    validator: dict[str, Any] = field(default_factory=dict)
    #: Aggregate metrics derived from sub-namespaces.
    donor_borrow_rate: float = 0.0
    consumption_model: dict[str, Any] = field(default_factory=dict)

    # ---- v3.0 OPTIONAL provenance extensions ---------------------------
    # All default to ``None`` (or absent on disk via setdefault) so v2.1
    # meta.json files still validate against this schema unchanged.

    #: Realized NHTS vintage mix actually drawn for this cache (e.g.
    #: ``{"2017": 1.0}`` for v2.1-style default, or ``{"2017": 0.8,
    #: "2022_nextgen": 0.2}`` when NextGen enrichment is opted in).
    nhts_vintage_mix: dict[str, float] | None = None

    #: HOUSEID stitching mode used by M4 — ``"v2_per_travday"`` (legacy
    #: per-TRAVDAY donor-day reuse) or ``"v3_within_hh"`` (intra-HOUSEID
    #: person-day reuse with weekday/weekend partition; v3.0 default).
    houseid_stitch_mode: str | None = None

    #: True when the v3 PHEV gasoline extension (gas_kwh_equivalent
    #: ledger) is active for this cache; False/None for BEV-only or
    #: legacy v2.1 runs.
    phev_modeled: bool | None = None

    #: True when ACS PUMS raking was applied to the household-mix
    #: expected PMF for this region/profile combination.
    acs_calibration_applied: bool | None = None

    #: Free-form descriptor of the SPEECh K=16 axis projection in use
    #: (e.g. ``"start_time_marginal_with_P(G)_prior"``). ``None`` when
    #: the K=16 SPEECh path is not active (legacy K=6 fit).
    speech_k16_axis_projection: str | None = None


def load_meta(meta_path: str | os.PathLike[str]) -> dict[str, Any]:
    """Idempotent loader — returns ``{}`` if the file is absent."""
    p = Path(meta_path)
    if not p.is_file():
        return {}
    with p.open("r", encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)
    return data


def write_meta_atomic(
    meta_path: str | os.PathLike[str],
    meta: dict[str, Any],
    *,
    default: Any = None,
) -> Path:
    """Atomic JSON write (tmp + os.replace) with sorted keys + indent=2.

    ``default`` is forwarded to :func:`json.dump` to coerce non-trivial
    payloads (numpy scalars, ``Path``); ``None`` is the standard JSON
    behaviour.
    """
    p = Path(meta_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True, default=default)
    os.replace(tmp, p)
    return p


class MetaWriter:
    """Append-only ``meta.json`` writer with ``setdefault`` semantics.

    Usage:

        with MetaWriter(meta_path) as m:
            m.set_default("methodology_version", "v3.0")
            m.merge("plug_in_model", {"K_chosen_by_BIC": 4, ...})

    The writer reloads the file on enter and persists atomically on
    exit. Concurrent writers on the same cache are NOT serialised — the
    contract is per-stage, sequential.
    """

    def __init__(self, meta_path: str | os.PathLike[str]) -> None:
        self.meta_path = Path(meta_path)
        self.meta: dict[str, Any] = {}

    def __enter__(self) -> MetaWriter:
        self.meta = load_meta(self.meta_path)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        if exc_type is None:
            write_meta_atomic(self.meta_path, self.meta)

    # ---- ergonomic helpers -------------------------------------------

    def set_default(self, key: str, value: Any) -> None:
        """``setdefault`` semantics — never overwrite an existing key."""
        self.meta.setdefault(key, value)

    def set(self, key: str, value: Any) -> None:
        """Force-set; use only when the producer is the sole owner."""
        self.meta[key] = value

    def merge(self, key: str, payload: dict[str, Any]) -> None:
        """Update sub-namespace ``key`` with ``payload`` (shallow merge)."""
        block = self.meta.setdefault(key, {})
        if isinstance(block, dict):
            block.update(payload)
        else:
            self.meta[key] = payload

    def record_module(self, module_name: str, methodology_version: str) -> None:
        """Stamp ``module_provenance[module] = methodology_version``."""
        prov = self.meta.setdefault("module_provenance", {})
        if isinstance(prov, dict):
            prov[module_name] = methodology_version

    # ---- RFC-021.r unified update entry point ------------------------

    @classmethod
    def update(
        cls,
        meta_path: str | os.PathLike[str],
        *,
        stage: str,
        payload: dict[str, Any],
        methodology_version: str | None = None,
        top_level: dict[str, Any] | None = None,
        json_default: Any = None,
    ) -> Path:
        """Single-call read-modify-write for one RMW stage.

        Parameters
        ----------
        meta_path:
            Path to the cache's ``meta.json``.
        stage:
            One of the keys in :data:`RMW_STAGES` (``donor_matcher``,
            ``soc_trajectory``, ``hourly_resampler``). Determines which
            sub-namespace block and which ``module_provenance`` key the
            payload writes through.
        payload:
            Block-level dict appended under ``meta[<sub_namespace>]``.
        methodology_version:
            If given, also stamped into
            ``meta.module_provenance[<prov_key>]``. When omitted, the
            value is read from ``payload['methodology_version']`` if
            present.
        top_level:
            Optional sibling keys to merge at the root of ``meta`` (e.g.
            ``{"donor_borrow_rate": 0.07, "package_versions": {...}}``).
            Uses shallow ``dict.update`` for nested dict values.
        json_default:
            Forwarded to :func:`json.dump` (e.g. a numpy coercer).

        Returns
        -------
        Path
            The path the meta was written to (same as ``meta_path``).
        """
        if stage not in RMW_STAGES:
            raise KeyError(
                f"unknown RMW stage {stage!r}; "
                f"valid keys: {sorted(RMW_STAGES)}"
            )
        sub_key, prov_key = RMW_STAGES[stage]

        p = Path(meta_path)
        meta = load_meta(p)

        # Sub-namespace block: force-set (single owner per RMW stage).
        meta[sub_key] = dict(payload)

        # Module provenance stamp.
        version = methodology_version
        if version is None:
            version = payload.get("methodology_version")
        if version is not None:
            prov = meta.setdefault("module_provenance", {})
            if isinstance(prov, dict):
                prov[prov_key] = version

        # Top-level sibling merges (e.g. donor_borrow_rate, run_timestamp_utc,
        # package_versions). Nested dicts are shallow-merged; scalars overwrite.
        if top_level:
            for k, v in top_level.items():
                if (
                    isinstance(v, dict)
                    and isinstance(meta.get(k), dict)
                ):
                    meta[k].update(v)
                else:
                    meta[k] = v

        return write_meta_atomic(p, meta, default=json_default)
