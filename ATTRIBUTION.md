# Data sources, licensing & attribution

`ev-flow` is MIT-licensed (see [LICENSE](LICENSE)). It is grounded in several
upstream public data sources. This file discharges the attribution and notice
obligations of those sources. Two notices below are **legally required** and
are reproduced verbatim; the remainder are courtesy citations (the data is
either U.S. Government public-domain or carries no binding attribution clause,
but we credit it as a matter of scientific good practice).

`ev-flow`'s code is built to consume the SPEECh Original Model — its loader,
schema, and CC-BY citations live in `pev_synth.plug_in_model`. A SPEECh-derived
K=16 parameter set (start-time marginals + group priors, `speech_k16_*.json`) is
**redistributed in the wheel and sdist** under `pev_synth/data/speech/`, which
CC-BY-4.0 permits with attribution — the required notice is reproduced below and
in the source. Shipping it lets a fresh `pip install ev-flow` build without
obtaining the upstream Mendeley source. (A maintainer's locally-derived copy
under `data/pev/raw/speech/`, if present, still takes precedence.) This
ATTRIBUTION file is shipped in the sdist.

---

## Required notices

### SPEECh Original Model — CC BY 4.0

> Contains data from "SPEECh Original Model" by Siobhan Powell, Gustavo Vianna
> Cezar, and Ram Rajagopal, Mendeley Data, V1 (2021),
> <https://doi.org/10.17632/gvk34mybtb.1>, licensed under CC BY 4.0
> (<https://creativecommons.org/licenses/by/4.0/>). **Modified:** a subset of
> the home and work driver-group Gaussian-mixture components was converted from
> pickle to JSON and reweighted.

Associated publication: Powell, S.; Cezar, G. V.; Rajagopal, R. (2022).
*Scalable probabilistic estimates of electric vehicle charging given observed
driver behavior.* Applied Energy 309, 118382.
<https://doi.org/10.1016/j.apenergy.2021.118382>

### U.S. Census Bureau Data API — required disclaimer

`pev_synth.acs_loader` accesses the Census Bureau Data API at runtime. Per the
[Census Data API Terms of Service](https://www.census.gov/data/developers/about/terms-of-service.html),
any product using the API must display the following notice:

> This product uses the Census Bureau Data API but is not endorsed or certified
> by the Census Bureau.

(The underlying ACS PUMS statistics are public domain; this is an
endorsement-disclaimer, not a copyright attribution. The Census terms also
prohibit using the data to re-identify any individual, household, or entity.)

---

## Courtesy citations

These sources impose no binding attribution requirement on `ev-flow`, but are
credited here.

| Source | Used for | License / status | Citation |
|---|---|---|---|
| **NHTS 2017** (& NextGen 2022) — USDOT/FHWA, hosted by ORNL | Travel micro-data backbone (`nhts_loader`, `nhts_nextgen_loader`) | U.S. Government work — public domain (17 U.S.C. §105); public-use confidentiality condition (no respondent re-identification) | U.S. Department of Transportation, Federal Highway Administration. 2017 National Household Travel Survey (and 2022 NextGen NHTS). <https://nhts.ornl.gov> |
| **U.S. Census ACS 5-Year PUMS** | Household-mix raking/calibration (`acs_loader`) | U.S. Government work — public domain; API use governed by Census Data API ToS (see required notice above) | U.S. Census Bureau, ACS 5-Year PUMS, via the Census Data API. <https://api.census.gov/data> |
| **EPA / DOE fueleconomy.gov** | PHEV combined fuel economy (`_phev_fuel_economy`) | U.S. Government data — public domain (17 U.S.C. §105); site administered by ORNL for DOE/EPA | U.S. EPA & U.S. DOE, fueleconomy.gov vehicle data. <https://www.fueleconomy.gov/ws/> |
| **EV WATTS Public Database** — DOE/Energetics (INL, NREL, PNNL) | Workplace cohort calibration & validation bounds (`sales_mix_data`, `validation_bounds_curator`) | DOE-sponsored public dataset; attribution requested, not required | Pritchard, E. *EV Watts Public Database.* U.S. DOE, 2023. <https://doi.org/10.15483/1970735> |
| **California CVRP rebate statistics** | CA regional BEV/PHEV sales-mix weights (`sales_mix_data`) | © Center for Sustainable Energy (for CARB); no open-data license; aggregate facts, partly re-derived | California Clean Vehicle Rebate Project (CVRP), Center for Sustainable Energy / CARB. <https://cleanvehiclerebate.org/en/rebate-statistics> |
| **NYSERDA Drive Clean Rebate** | NY regional sales-mix weights (approximation) (`sales_mix_data`) | NY State Open Data (Open NY) Terms of Use — permissive, no attribution required | NYSERDA Electric Vehicle Drive Clean Rebate Data (Open NY). <https://data.ny.gov/Energy-Environment/NYSERDA-Electric-Vehicle-Drive-Clean-Rebate-Data-B/thd2-fu8y> |
| **Argonne National Laboratory — EV-FACTS** | National BEV/PHEV sales-mix shape (`sales_mix_data`) | Published DOE/Argonne analysis (© UChicago Argonne, LLC); aggregate share data is non-copyrightable fact | Argonne National Laboratory, Light-Duty Electric Drive Vehicles Monthly Sales Updates / EV-FACTS. <https://www.anl.gov/ev-facts/model-sales> |
| **NOAA NCEI** — Integrated Surface Database / U.S. Climate Normals 1991–2020 | Winter ambient-temperature uplift (`travel_week_builder`, `regions`) | U.S. Government work — public domain (NCEI applies CC0-1.0); attribution requested | NOAA National Centers for Environmental Information, Integrated Surface Database / U.S. Climate Normals 1991–2020. |
| **NREL technical reports** (TP-5400-81065, TP-5400-69031, CP-5400-85902) & EVI-Pro Lite | Modeling priors & validation bounds (`vehicle_archetypes`, `validation_bounds_curator`) | U.S. Government-sponsored publications; scholarly citation | National Renewable Energy Laboratory, U.S. DOE. (Report numbers cited inline in source.) |

**CAISO** market prices are *not* shipped with `ev-flow`. The optional
`caiso_data` dependency (imported behind a guard in `validator`) is a separate
package; if a downstream user supplies it, CAISO's own terms apply to that user.

---

*This audit was performed against each source's official terms. Where a source
could not be verified to high confidence, it is credited conservatively. If you
are a rights holder and believe attribution here is incorrect, please open an
issue.*
