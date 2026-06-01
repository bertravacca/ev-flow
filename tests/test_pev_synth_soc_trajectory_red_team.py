"""Red-team M7 regressions for ``pev_synth.soc_trajectory``.

Covers the promotion of ``DCFC_NAMEPLATE_KW`` to a configurable field on
:class:`pev_synth.soc_trajectory.SocTrajectoryConfig` (red-team item M7).
The module-level constant is kept as a back-compat alias and must equal
the config-field default.
"""

from __future__ import annotations

from pev_synth.soc_trajectory import DCFC_NAMEPLATE_KW, SocTrajectoryConfig


def test_dcfc_nameplate_kw_is_configurable() -> None:
    """``SocTrajectoryConfig`` must expose ``dcfc_nameplate_kw`` as a field."""
    cfg = SocTrajectoryConfig(dcfc_nameplate_kw=200.0)
    assert cfg.dcfc_nameplate_kw == 200.0


def test_dcfc_nameplate_kw_back_compat_parity() -> None:
    """Module-level constant must equal the config-field default.

    The constant is retained as a back-compat alias of
    ``SocTrajectoryConfig.dcfc_nameplate_kw``; the two must not drift.
    """
    assert DCFC_NAMEPLATE_KW == SocTrajectoryConfig().dcfc_nameplate_kw
    # Pin the v2.0 default explicitly: 50 kW (pre-2020 fleets) is held
    # to preserve regression fixtures that pin session ``max_kw``.
    assert DCFC_NAMEPLATE_KW == 50.0
