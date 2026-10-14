# STIX solar flare catalogue

Source: challenge handout. One row per solar flare observed by the STIX X-ray telescope
aboard Solar Orbiter, February 2021 to February 2026.

## Background

- **GOES class** states flare strength as a letter A, B, C, M or X, each step ten times
  stronger than the last, plus a number for fine grading (`M1.5` is medium-strong, `X9` huge).
  GOES satellites orbit Earth and see only the Earth-facing side of the Sun.
- **STIX** orbits the Sun, so it also records flares on the far side of the Sun; roughly
  half of the catalogue consists of such events. Its distance to the Sun varies, and the
  same flare produces more counts when the spacecraft is closer.
- STIX counts X-ray photons in energy bands from 4 to 84 keV. The 4-10 keV band responds to
  essentially every flare and is the general size measure. Counts per flare span many orders
  of magnitude; small events are far more common than large ones.
- During very bright flares a protective **attenuator** moves in front of the detector
  (`att_in = True`), which sharply reduces the measured counts.
- This is real operational data. No column is guaranteed clean or consistently typed.

## Data dictionary

| Column | Meaning |
|---|---|
| `flare_id` | Catalog identifier of the flare |
| `start_UTC`, `end_UTC`, `peak_UTC` | Start, end, and peak time of the flare (string formatted, utc-like) |
| `4-10 keV`, `10-15 keV`, `15-25 keV`, `25-50 keV`, `50-84 keV` | X-ray counts per energy band; 4-10 keV is the general size measure, higher bands respond to strong flares |
| `att_in` | Whether the protective attenuator was inserted during the flare |
| `hpc_x_solo`, `hpc_y_solo` | Flare position in arcseconds from disk center, as seen from Solar Orbiter |
| `hpc_x_earth`, `hpc_y_earth` | The same position converted to the Earth viewpoint; empty for far-side flares |
| `visible_from_earth` | Whether the flare was on the Earth-facing side of the Sun |
| `hgs_lon`, `hgs_lat` | Heliographic longitude and latitude of the flare in degrees; longitude 0 is disk center as seen from Earth; empty when no position could be determined |
| `solo_position_AU_distance` | Solar Orbiter's distance from the Sun in AU (Earth-Sun distance = 1); about 0.29 to 1.02 |
| `GOES_class_time_of_flare`, `GOES_flux_time_of_flare` | GOES class and X-ray flux (W/m2) measured by GOES at the time of the flare |
| `goes_estimated_mean_class`, `goes_estimated_mean_flux` | GOES-equivalent class and flux estimated from STIX's own measurement |
| `error_with_imaging` | True if position imaging failed for this event |

The file contains further columns that are not documented here.
