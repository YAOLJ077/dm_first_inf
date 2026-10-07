# Guangxi S-route data layout

The local analysis is restricted to the six Guangxi site-years listed in
`disease_data/grape_downy_mildew_data.xlsx` (the source workbook is also kept
as `disease_onset.xlsx` at the data root).

- `disease_data/`: observed first visible downy mildew dates.
- `weather_data/site_year/`: one validated Beijing-time hourly JSON per
  site-year. The files use the `CropSeason` weather contract consumed by
  `model.FSIM`.
- `spore_data/`: the CK airborne-sporangia workbook and its observation
  notes. It contains five monitored site-years; the sixth Guangxi site-year
  has no CK rows and can run the base FSIM route only.

Blank CK cells are missing collection results. Numeric zeroes are observed
zero-count collections. The first measured collection interval starts three
days before its first nonblank date; later intervals are bounded by adjacent
nonblank collection dates.
