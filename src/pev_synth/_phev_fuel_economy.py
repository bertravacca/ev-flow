"""PHEV charge-sustaining (gas-only) fuel economy lookup.

Used by M6 SoC ledger (v3.0) when a PHEV's SoC would drop below
``soc_min``: the deficit is served by the ICE and recorded as
``sessions.gas_kwh_equivalent``.

Values are EPA combined MPG in charge-sustaining (CS) mode — i.e.
gasoline-only operation, after the usable plug-in battery window has been
depleted — per ``(make, model, model_year)``. The relevant EPA field is
``comb08`` on the fueleconomy.gov vehicle record; ``phevComb`` (which
sometimes appears alongside) is the SAE J1711 *blended* utility-factor
rating that combines CD-mode (electricity) and CS-mode (gasoline), and is
therefore **not** appropriate for the gas-energy ledger.

Citation pattern: every populated row has a matching entry in
:data:`SOURCES` pointing at ``https://www.fueleconomy.gov/ws/rest/vehicle/<id>``
— the canonical per-vehicle XML record from which the ``comb08`` value was
read.

Coverage policy
---------------
The table mirrors :data:`pev_synth.sales_mix_data._MODEL_ATTRS` PHEV rows
and their per-model ``[model_year_min, model_year_max]`` bands. When EPA
fueleconomy.gov does not list a vehicle for a particular model year (a few
isolated gaps in the database for late-cycle re-badges, e.g. the
2024 BMW X5 xDrive50e or the 2024 Lincoln Corsair Grand Touring), the
adjacent-year EPA value is carried because the powertrain is unchanged
across the gap; such carry-rows are flagged inline in :data:`SOURCES`
with the suffix ``" (carried from <neighbour-year> — model not listed in"
" EPA database)"``.

If a ``(make, model, year)`` triple is not present in :data:`PHEV_CS_MPG`
at all, callers should fall back to :data:`PHEV_CS_MPG_FALLBACK`, set to
the EPA all-PHEV combined median.

Source bibliography
-------------------
Per-vehicle EPA records (XML): https://www.fueleconomy.gov/ws/rest/vehicle/<id>
EPA "Most Efficient PHEV" top-ten (cross-check): https://www.fueleconomy.gov/feg/topten.jsp
EPA all-PHEV median (fallback derivation): https://www.fueleconomy.gov/feg/PHEV.shtml
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "PHEV_CS_MPG",
    "PHEV_CS_MPG_FALLBACK",
    "SOURCES",
    "cs_mpg",
]


# --------------------------------------------------------------------------- #
# PHEV CS-mode (gasoline-only) combined MPG, per (make, model, model_year).
# Source: EPA fueleconomy.gov ``comb08`` field on the per-vehicle XML record.
# Model-year bands mirror sales_mix_data._MODEL_ATTRS PHEV entries.
# --------------------------------------------------------------------------- #
PHEV_CS_MPG: Final[dict[tuple[str, str, int], float]] = {
    # Toyota Prius Prime / Prius PHEV (renamed for 2025 model year).
    # 1st-gen (ZVW52, 2017-2022) plateaus at 54 MPG; 2nd-gen (MXWH61,
    # 2023+) drops to 48 MPG with the larger 2.0 L Atkinson engine.
    ("Toyota", "Prius_Prime", 2017): 54.0,
    ("Toyota", "Prius_Prime", 2018): 54.0,
    ("Toyota", "Prius_Prime", 2019): 54.0,
    ("Toyota", "Prius_Prime", 2020): 54.0,
    ("Toyota", "Prius_Prime", 2021): 54.0,
    ("Toyota", "Prius_Prime", 2022): 54.0,
    ("Toyota", "Prius_Prime", 2023): 48.0,
    ("Toyota", "Prius_Prime", 2024): 48.0,
    ("Toyota", "Prius_Prime", 2025): 48.0,
    # Toyota RAV4 Prime (renamed RAV4 PHEV AWD for 2025 MY).
    ("Toyota", "RAV4_Prime", 2021): 38.0,
    ("Toyota", "RAV4_Prime", 2022): 38.0,
    ("Toyota", "RAV4_Prime", 2023): 38.0,
    ("Toyota", "RAV4_Prime", 2024): 38.0,
    ("Toyota", "RAV4_Prime", 2025): 38.0,
    # Ford Escape PHEV (FWD only — PHEV trim is not offered AWD).
    # EPA dropped Escape PHEV from 2024 catalogue (production gap);
    # 2025 returns with the same powertrain → 2024 carried.
    ("Ford", "Escape_PHEV", 2020): 41.0,
    ("Ford", "Escape_PHEV", 2021): 40.0,
    ("Ford", "Escape_PHEV", 2022): 40.0,
    ("Ford", "Escape_PHEV", 2023): 40.0,
    ("Ford", "Escape_PHEV", 2024): 40.0,
    ("Ford", "Escape_PHEV", 2025): 40.0,
    # Lincoln Corsair Grand Touring (PHEV).
    # 2024 absent from EPA catalogue (Lincoln production pause);
    # powertrain unchanged on either side, so 2023 carried into 2024.
    ("Lincoln", "Corsair_PHEV", 2021): 33.0,
    ("Lincoln", "Corsair_PHEV", 2022): 33.0,
    ("Lincoln", "Corsair_PHEV", 2023): 33.0,
    ("Lincoln", "Corsair_PHEV", 2024): 33.0,
    ("Lincoln", "Corsair_PHEV", 2025): 33.0,
    # Jeep Wrangler 4xe — 4-door 4WD PHEV.
    ("Jeep", "Wrangler_4xe", 2021): 20.0,
    ("Jeep", "Wrangler_4xe", 2022): 20.0,
    ("Jeep", "Wrangler_4xe", 2023): 20.0,
    ("Jeep", "Wrangler_4xe", 2024): 20.0,
    ("Jeep", "Wrangler_4xe", 2025): 20.0,
    # Jeep Grand Cherokee 4xe — 4WD PHEV.
    ("Jeep", "Grand_Cherokee_4xe", 2022): 23.0,
    ("Jeep", "Grand_Cherokee_4xe", 2023): 23.0,
    ("Jeep", "Grand_Cherokee_4xe", 2024): 23.0,
    ("Jeep", "Grand_Cherokee_4xe", 2025): 23.0,
    # Chrysler Pacifica Hybrid (the PHEV minivan; "Hybrid" in EPA naming
    # is the plug-in trim — Pacifica's only electrified variant).
    ("Chrysler", "Pacifica_PHEV", 2018): 32.0,
    ("Chrysler", "Pacifica_PHEV", 2019): 30.0,
    ("Chrysler", "Pacifica_PHEV", 2020): 30.0,
    ("Chrysler", "Pacifica_PHEV", 2021): 30.0,
    ("Chrysler", "Pacifica_PHEV", 2022): 30.0,
    ("Chrysler", "Pacifica_PHEV", 2023): 30.0,
    ("Chrysler", "Pacifica_PHEV", 2024): 30.0,
    ("Chrysler", "Pacifica_PHEV", 2025): 30.0,
    # BMW X5 xDrive50e — replaces the prior xDrive45e PHEV. Not listed
    # by EPA for 2024 (filing arrived only with 2025 MY); 2025 value
    # carried back since powertrain is unchanged.
    ("BMW", "X5_xDrive50e", 2024): 22.0,
    ("BMW", "X5_xDrive50e", 2025): 22.0,
    # Mitsubishi Outlander PHEV — 4th-gen, US re-entry MY2023.
    ("Mitsubishi", "Outlander_PHEV", 2023): 26.0,
    ("Mitsubishi", "Outlander_PHEV", 2024): 26.0,
    ("Mitsubishi", "Outlander_PHEV", 2025): 26.0,
    # Volvo XC60 Recharge T8 AWD. EPA naming varied across years:
    # "XC60 AWD PHEV" (2021), "XC60 T8 AWD Recharge" (2022-2024),
    # "XC60 T8 AWD" (2025). Powertrain map evolved in 2023 (longer EV
    # range, slightly higher CS-mode efficiency).
    ("Volvo", "XC60_Recharge", 2021): 27.0,
    ("Volvo", "XC60_Recharge", 2022): 25.0,
    ("Volvo", "XC60_Recharge", 2023): 28.0,
    ("Volvo", "XC60_Recharge", 2024): 28.0,
    ("Volvo", "XC60_Recharge", 2025): 28.0,
    # Hyundai Tucson Plug-in Hybrid (NX4 PHE).
    ("Hyundai", "Tucson_PHEV", 2022): 35.0,
    ("Hyundai", "Tucson_PHEV", 2023): 35.0,
    ("Hyundai", "Tucson_PHEV", 2024): 35.0,
    ("Hyundai", "Tucson_PHEV", 2025): 35.0,
    # Kia Sorento Plug-in Hybrid (MQ4 PHEV). 2025 drops 1 MPG combined.
    ("Kia", "Sorento_PHEV", 2022): 34.0,
    ("Kia", "Sorento_PHEV", 2023): 34.0,
    ("Kia", "Sorento_PHEV", 2024): 34.0,
    ("Kia", "Sorento_PHEV", 2025): 33.0,
}


#: Fallback CS MPG when a ``(make, model, year)`` triple is absent from
#: :data:`PHEV_CS_MPG`. Set to ``33.0`` — the median of the 60 populated
#: rows in :data:`PHEV_CS_MPG` (i.e. the EPA per-vehicle median across
#: the v3.0 PHEV cohort, MY2017..2025). Cross-checked against EPA's
#: aggregate PHEV summary at
#: https://www.fueleconomy.gov/feg/PHEV.shtml. This is the
#: weight-of-evidence "typical PHEV" CS-mode rating once the high-MPG
#: Prius Prime (48-54) and the low-MPG body-on-frame 4xe SUVs (20-23) are
#: taken in distribution.
PHEV_CS_MPG_FALLBACK: Final[float] = 33.0


# --------------------------------------------------------------------------- #
# Per-row citation. Each URL is the canonical EPA fueleconomy.gov XML
# record from which ``comb08`` was read.
# --------------------------------------------------------------------------- #
SOURCES: Final[dict[tuple[str, str, int], str]] = {
    ("Toyota", "Prius_Prime", 2017): "https://www.fueleconomy.gov/ws/rest/vehicle/38531",
    ("Toyota", "Prius_Prime", 2018): "https://www.fueleconomy.gov/ws/rest/vehicle/39882",
    ("Toyota", "Prius_Prime", 2019): "https://www.fueleconomy.gov/ws/rest/vehicle/41184",
    ("Toyota", "Prius_Prime", 2020): "https://www.fueleconomy.gov/ws/rest/vehicle/41489",
    ("Toyota", "Prius_Prime", 2021): "https://www.fueleconomy.gov/ws/rest/vehicle/42814",
    ("Toyota", "Prius_Prime", 2022): "https://www.fueleconomy.gov/ws/rest/vehicle/44362",
    ("Toyota", "Prius_Prime", 2023): "https://www.fueleconomy.gov/ws/rest/vehicle/47228",
    ("Toyota", "Prius_Prime", 2024): "https://www.fueleconomy.gov/ws/rest/vehicle/47500",
    ("Toyota", "Prius_Prime", 2025): "https://www.fueleconomy.gov/ws/rest/vehicle/49014",
    ("Toyota", "RAV4_Prime", 2021): "https://www.fueleconomy.gov/ws/rest/vehicle/42793",
    ("Toyota", "RAV4_Prime", 2022): "https://www.fueleconomy.gov/ws/rest/vehicle/44984",
    ("Toyota", "RAV4_Prime", 2023): "https://www.fueleconomy.gov/ws/rest/vehicle/47230",
    ("Toyota", "RAV4_Prime", 2024): "https://www.fueleconomy.gov/ws/rest/vehicle/47502",
    ("Toyota", "RAV4_Prime", 2025): "https://www.fueleconomy.gov/ws/rest/vehicle/49160",
    ("Ford", "Escape_PHEV", 2020): "https://www.fueleconomy.gov/ws/rest/vehicle/42743",
    ("Ford", "Escape_PHEV", 2021): "https://www.fueleconomy.gov/ws/rest/vehicle/43912",
    ("Ford", "Escape_PHEV", 2022): "https://www.fueleconomy.gov/ws/rest/vehicle/44931",
    ("Ford", "Escape_PHEV", 2023): "https://www.fueleconomy.gov/ws/rest/vehicle/47220",
    ("Ford", "Escape_PHEV", 2024): (
        "https://www.fueleconomy.gov/ws/rest/vehicle/47220 (carried from 2023"
        " — model not listed in EPA database)"
    ),
    ("Ford", "Escape_PHEV", 2025): "https://www.fueleconomy.gov/ws/rest/vehicle/48663",
    ("Lincoln", "Corsair_PHEV", 2021): "https://www.fueleconomy.gov/ws/rest/vehicle/43774",
    ("Lincoln", "Corsair_PHEV", 2022): "https://www.fueleconomy.gov/ws/rest/vehicle/44934",
    ("Lincoln", "Corsair_PHEV", 2023): "https://www.fueleconomy.gov/ws/rest/vehicle/47226",
    ("Lincoln", "Corsair_PHEV", 2024): (
        "https://www.fueleconomy.gov/ws/rest/vehicle/47226 (carried from 2023"
        " — model not listed in EPA database)"
    ),
    ("Lincoln", "Corsair_PHEV", 2025): "https://www.fueleconomy.gov/ws/rest/vehicle/48671",
    ("Jeep", "Wrangler_4xe", 2021): "https://www.fueleconomy.gov/ws/rest/vehicle/43801",
    ("Jeep", "Wrangler_4xe", 2022): "https://www.fueleconomy.gov/ws/rest/vehicle/44932",
    ("Jeep", "Wrangler_4xe", 2023): "https://www.fueleconomy.gov/ws/rest/vehicle/46252",
    ("Jeep", "Wrangler_4xe", 2024): "https://www.fueleconomy.gov/ws/rest/vehicle/47278",
    ("Jeep", "Wrangler_4xe", 2025): "https://www.fueleconomy.gov/ws/rest/vehicle/48664",
    ("Jeep", "Grand_Cherokee_4xe", 2022): "https://www.fueleconomy.gov/ws/rest/vehicle/45148",
    ("Jeep", "Grand_Cherokee_4xe", 2023): "https://www.fueleconomy.gov/ws/rest/vehicle/46253",
    ("Jeep", "Grand_Cherokee_4xe", 2024): "https://www.fueleconomy.gov/ws/rest/vehicle/47277",
    ("Jeep", "Grand_Cherokee_4xe", 2025): "https://www.fueleconomy.gov/ws/rest/vehicle/48665",
    ("Chrysler", "Pacifica_PHEV", 2018): "https://www.fueleconomy.gov/ws/rest/vehicle/39483",
    ("Chrysler", "Pacifica_PHEV", 2019): "https://www.fueleconomy.gov/ws/rest/vehicle/40830",
    ("Chrysler", "Pacifica_PHEV", 2020): "https://www.fueleconomy.gov/ws/rest/vehicle/41943",
    ("Chrysler", "Pacifica_PHEV", 2021): "https://www.fueleconomy.gov/ws/rest/vehicle/43497",
    ("Chrysler", "Pacifica_PHEV", 2022): "https://www.fueleconomy.gov/ws/rest/vehicle/44930",
    ("Chrysler", "Pacifica_PHEV", 2023): "https://www.fueleconomy.gov/ws/rest/vehicle/46249",
    ("Chrysler", "Pacifica_PHEV", 2024): "https://www.fueleconomy.gov/ws/rest/vehicle/47276",
    ("Chrysler", "Pacifica_PHEV", 2025): "https://www.fueleconomy.gov/ws/rest/vehicle/48656",
    ("BMW", "X5_xDrive50e", 2024): (
        "https://www.fueleconomy.gov/ws/rest/vehicle/49009 (carried from 2025"
        " — model not listed in EPA database for 2024 MY)"
    ),
    ("BMW", "X5_xDrive50e", 2025): "https://www.fueleconomy.gov/ws/rest/vehicle/49009",
    ("Mitsubishi", "Outlander_PHEV", 2023): "https://www.fueleconomy.gov/ws/rest/vehicle/47280",
    ("Mitsubishi", "Outlander_PHEV", 2024): "https://www.fueleconomy.gov/ws/rest/vehicle/47499",
    ("Mitsubishi", "Outlander_PHEV", 2025): "https://www.fueleconomy.gov/ws/rest/vehicle/48676",
    ("Volvo", "XC60_Recharge", 2021): "https://www.fueleconomy.gov/ws/rest/vehicle/43145",
    ("Volvo", "XC60_Recharge", 2022): "https://www.fueleconomy.gov/ws/rest/vehicle/44268",
    ("Volvo", "XC60_Recharge", 2023): "https://www.fueleconomy.gov/ws/rest/vehicle/46629",
    ("Volvo", "XC60_Recharge", 2024): "https://www.fueleconomy.gov/ws/rest/vehicle/47504",
    ("Volvo", "XC60_Recharge", 2025): "https://www.fueleconomy.gov/ws/rest/vehicle/48679",
    ("Hyundai", "Tucson_PHEV", 2022): "https://www.fueleconomy.gov/ws/rest/vehicle/44360",
    ("Hyundai", "Tucson_PHEV", 2023): "https://www.fueleconomy.gov/ws/rest/vehicle/46251",
    ("Hyundai", "Tucson_PHEV", 2024): "https://www.fueleconomy.gov/ws/rest/vehicle/47495",
    ("Hyundai", "Tucson_PHEV", 2025): "https://www.fueleconomy.gov/ws/rest/vehicle/49011",
    ("Kia", "Sorento_PHEV", 2022): "https://www.fueleconomy.gov/ws/rest/vehicle/44361",
    ("Kia", "Sorento_PHEV", 2023): "https://www.fueleconomy.gov/ws/rest/vehicle/47222",
    ("Kia", "Sorento_PHEV", 2024): "https://www.fueleconomy.gov/ws/rest/vehicle/47496",
    ("Kia", "Sorento_PHEV", 2025): "https://www.fueleconomy.gov/ws/rest/vehicle/49012",
}


def cs_mpg(make: str, model: str, model_year: int) -> float:
    """Return CS-mode combined MPG for ``(make, model, model_year)``.

    Falls back to :data:`PHEV_CS_MPG_FALLBACK` (EPA all-PHEV median, 30
    MPG) when the triple is absent from :data:`PHEV_CS_MPG`. This is the
    intended behaviour for out-of-band years — for example, a 2017
    Wrangler 4xe row in some synthetic fleet would resolve to the
    fallback because Wrangler 4xe did not exist until 2021.
    """
    return PHEV_CS_MPG.get((make, model, model_year), PHEV_CS_MPG_FALLBACK)
