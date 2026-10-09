# Guangxi S-route data layout

The local analysis is restricted to the six Guangxi site-years listed in
`disease_data/grape_downy_mildew_data.xlsx`.

- `disease_data/`: observed first visible downy mildew dates.
- `weather_data/site_year/`: one validated Beijing-time hourly JSON per
  site-year. The files use the `CropSeason` weather contract consumed by
  `model.FSIM`.
- `spore_data/`: the CK airborne-sporangia workbook and its observation
  notes for all six Guangxi site-years.
- `phenology_data/`: the field workbook and compact local Table/Wine
  calibration inputs; literature observations and obsolete merged weather
  tables have been removed.

Blank CK cells are missing collection results. Numeric zeroes are observed
zero-count collections. The first measured collection interval starts three
days before its first nonblank date; later intervals are bounded by adjacent
nonblank collection dates.
