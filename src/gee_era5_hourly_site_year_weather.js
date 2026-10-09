/**
 * Export ERA5 hourly weather for the Guangxi site-years used in this study.
 *
 * Upload tmp/era5_site_year_index.csv to Earth Engine as a table asset and
 * replace INDEX_ASSET below with the resulting asset ID. Each index row
 * creates two Drive export tasks:
 *
 *   <site_year>_prev_sep_dec_BJT.csv
 *       September 1 of the previous year through December 31 of the
 *       previous year, in Asia/Shanghai local time.
 *   <site_year>_target_year_BJT.csv
 *       January 1 through December 31 of the target year, in local time.
 *
 * The two exports together cover the requested window for each site-year.
 * Timestamps are exported in both local time and UTC. Precipitation is the
 * ERA5 hourly total precipitation for the interval ending at the timestamp,
 * converted from m to mm. Relative humidity is calculated from 2-m air
 * temperature and 2-m dew-point temperature using the Magnus relationship.
 *
 * Source dataset:
 * https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_HOURLY
 */

var INDEX_ASSET = 'projects/glossy-essence-477012-s7/assets/era5_site_year_index';
var DRIVE_FOLDER = 'ERA5_dm_first_inf_site_year_BJT';
var TIME_ZONE = 'Asia/Shanghai';
var DATASET = 'ECMWF/ERA5/HOURLY';

// Use batches when the Earth Engine task list becomes large.
var START_INDEX = 0;
var MAX_SITE_YEARS = null;
var CHECK_TIME_COVERAGE = true;

var BANDS = [
  'temperature_2m',
  'dewpoint_temperature_2m',
  'total_precipitation',
  'u_component_of_wind_10m',
  'v_component_of_wind_10m'
];
var era5 = ee.ImageCollection(DATASET).select(BANDS);
var nativeProjection = ee.Image(era5.first())
  .select('temperature_2m').projection();

var SELECTORS = [
  'site', 'site_year', 'province', 'latitude', 'longitude', 'year', 'period',
  'weather_year_bjt', 'time_bjt', 'date_bjt', 'hour_bjt', 'timezone',
  'time_utc', 'time_start_millis', 'precipitation_interval_start_bjt',
  'temperature_c', 'dew_point_c', 'relative_humidity_pct',
  'precipitation_mm', 'wind_speed_m_s', 'valid_input_band_count', 'source'
];

function numericProperty(properties, name) {
  var value = properties[name];
  return value === null || value === undefined || String(value).trim() === ''
    ? NaN : Number(value);
}

function textProperty(properties, name) {
  var value = properties[name];
  return value === null || value === undefined
    ? '' : String(value).replace(/^\uFEFF/, '').trim();
}

function geometryCoordinate(feature, position) {
  var geometry = feature.geometry;
  if (!geometry || geometry.type !== 'Point' ||
      !geometry.coordinates || geometry.coordinates.length < 2) {
    return NaN;
  }
  return Number(geometry.coordinates[position]);
}

function makePlan(row, period) {
  var startYear = period === 'target_year' ? row.year : row.year - 1;
  var startMonth = period === 'target_year' ? 1 : 9;
  var endYear = period === 'target_year' ? row.year + 1 : row.year;
  var start = ee.Date.fromYMD(startYear, startMonth, 1, TIME_ZONE);
  var end = ee.Date.fromYMD(endYear, 1, 1, TIME_ZONE);
  var expectedHours = (Date.UTC(endYear, 0, 1) -
    Date.UTC(startYear, startMonth - 1, 1)) / 3600000;
  return {
    row: row,
    period: period,
    start: start,
    end: end,
    expectedHours: expectedHours,
    images: era5.filterDate(start, end).sort('system:time_start'),
    filename: row.site_year + '_' + period + '_BJT'
  };
}

function exportPlan(plan) {
  var row = plan.row;
  var point = ee.Geometry.Point([row.longitude, row.latitude]);
  var hourly = ee.FeatureCollection(plan.images.map(function(image) {
    image = ee.Image(image);
    var time = ee.Date(image.get('system:time_start'));
    var temperature = image.select('temperature_2m').subtract(273.15)
      .rename('temperature_c');
    var dewPoint = image.select('dewpoint_temperature_2m').subtract(273.15)
      .rename('dew_point_c');
    var precipitationRaw = image.select('total_precipitation');
    var u = image.select('u_component_of_wind_10m');
    var v = image.select('v_component_of_wind_10m');
    var relativeHumidity = temperature.expression(
      '100 * exp(17.62 * Td / (243.12 + Td) - ' +
      '17.62 * T / (243.12 + T))',
      {T: temperature, Td: dewPoint}
    ).clamp(0, 100).rename('relative_humidity_pct');
    var validCount = image.mask().unmask(0).reduce(ee.Reducer.sum())
      .rename('valid_input_band_count');
    var weather = temperature
      .addBands(dewPoint)
      .addBands(relativeHumidity)
      .addBands(precipitationRaw.multiply(1000).rename('precipitation_mm'))
      .addBands(u.pow(2).add(v.pow(2)).sqrt().rename('wind_speed_m_s'))
      .addBands(validCount);
    var values = weather.reduceRegion({
      reducer: ee.Reducer.first(),
      geometry: point,
      crs: nativeProjection,
      maxPixels: 10000
    });
    return ee.Feature(null, values).set({
      site: row.site,
      site_year: row.site_year,
      province: row.province,
      latitude: row.latitude,
      longitude: row.longitude,
      year: row.year,
      period: plan.period,
      source: DATASET,
      timezone: TIME_ZONE,
      weather_year_bjt: time.get('year', TIME_ZONE),
      time_bjt: time.format('yyyy-MM-dd HH:mm:ss', TIME_ZONE),
      date_bjt: time.format('yyyy-MM-dd', TIME_ZONE),
      hour_bjt: time.get('hour', TIME_ZONE),
      time_utc: time.format('yyyy-MM-dd HH:mm:ss', 'UTC'),
      time_start_millis: time.millis(),
      precipitation_interval_start_bjt: time.advance(-1, 'hour')
        .format('yyyy-MM-dd HH:mm:ss', TIME_ZONE)
    });
  })).sort('time_start_millis');

  Export.table.toDrive({
    collection: hourly,
    description: 'ERA5_' + plan.filename,
    folder: DRIVE_FOLDER,
    fileNamePrefix: plan.filename,
    fileFormat: 'CSV',
    selectors: SELECTORS
  });
}

// The index is small, so read only its metadata client-side to create tasks.
var indexInfo = ee.FeatureCollection(INDEX_ASSET).getInfo();
if (!indexInfo || !indexInfo.features || !indexInfo.features.length) {
  throw new Error('The site-year index asset is empty or could not be read.');
}
print('Index asset first feature:', indexInfo.features[0]);
var seen = {};
var rows = indexInfo.features.map(function(feature, index) {
  var properties = {};
  Object.keys(feature.properties || {}).forEach(function(key) {
    properties[key.replace(/^\uFEFF/, '').trim().toLowerCase()] =
      feature.properties[key];
  });
  // Earth Engine may promote latitude/longitude columns to point geometry
  // during CSV table import and omit them from feature.properties.
  var latitude = numericProperty(properties, 'latitude');
  var longitude = numericProperty(properties, 'longitude');
  if (!isFinite(latitude)) latitude = geometryCoordinate(feature, 1);
  if (!isFinite(longitude)) longitude = geometryCoordinate(feature, 0);
  var year = numericProperty(properties, 'year');
  var site = textProperty(properties, 'site');
  var siteYear = textProperty(properties, 'site_year');

  // A table uploaded with the CSV header retained as a feature produces a
  // first row whose values are the field names. Ignore that row explicitly.
  if (site.toLowerCase() === 'site' &&
      siteYear.toLowerCase() === 'site_year') {
    return null;
  }
  if (!siteYear && site && isFinite(year)) {
    siteYear = site + '_' + year;
  }
  if (!isFinite(latitude) || !isFinite(longitude) || !isFinite(year) ||
      !site || siteYear !== site + '_' + year || Math.floor(year) !== year ||
      year < 1940 || latitude < -90 || latitude > 90 ||
      longitude < -180 || longitude > 180) {
    throw new Error('Invalid site-year index row ' + (index + 1) + ': ' +
      JSON.stringify(properties));
  }
  var key = latitude + '/' + longitude + '/' + year;
  if (seen[key] || seen[siteYear]) {
    throw new Error('Duplicate site-year: ' + siteYear);
  }
  seen[key] = true;
  seen[siteYear] = true;
  return {
    site: site,
    site_year: siteYear,
    province: String(properties.province || ''),
    latitude: latitude,
    longitude: longitude,
    year: year
  };
}).filter(function(row) {
  return row !== null;
}).sort(function(a, b) {
  return a.site_year.localeCompare(b.site_year);
});

if (MAX_SITE_YEARS !== null &&
    (!isFinite(MAX_SITE_YEARS) || MAX_SITE_YEARS < 1 ||
     Math.floor(MAX_SITE_YEARS) !== MAX_SITE_YEARS)) {
  throw new Error('MAX_SITE_YEARS must be null or a positive integer.');
}
var selectedRows = rows.slice(
  START_INDEX,
  MAX_SITE_YEARS === null ? rows.length : START_INDEX + MAX_SITE_YEARS
);
if (!selectedRows.length) throw new Error('No site-years in the selected batch.');

var plans = [];
selectedRows.forEach(function(row) {
  plans.push(makePlan(row, 'target_year'));
  plans.push(makePlan(row, 'prev_sep_dec'));
});

if (CHECK_TIME_COVERAGE) {
  var audit = ee.FeatureCollection(plans.map(function(plan) {
    return ee.Feature(null, {
      filename: plan.filename,
      expected_hours: plan.expectedHours,
      actual_hours: plan.images.size(),
      unique_hours: plan.images.aggregate_count_distinct('system:time_start'),
      start_bjt: plan.start.format('yyyy-MM-dd HH:mm:ss', TIME_ZONE),
      last_bjt: plan.end.advance(-1, 'hour')
        .format('yyyy-MM-dd HH:mm:ss', TIME_ZONE)
    });
  }));
  var auditRows = audit.getInfo().features.map(function(feature) {
    return feature.properties;
  });
  auditRows.forEach(function(row) {
    if (row.actual_hours !== row.expected_hours ||
        row.unique_hours !== row.expected_hours) {
      throw new Error('Incomplete or duplicate ERA5 hours: ' +
        JSON.stringify(row));
    }
  });
  print('ERA5 hourly coverage audit:', auditRows);
}

plans.forEach(exportPlan);
print('Created export tasks:', plans.length);
print('Selected site-years:', selectedRows);
print('Start the generated tasks in the Earth Engine Tasks panel.');
Map.setCenter(108.5, 23.5, 6);
