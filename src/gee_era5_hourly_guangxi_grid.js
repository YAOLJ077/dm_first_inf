/**
 * Export one unified ERA5 hourly weather archive for Guangxi grid cells.
 *
 * Upload tmp/gx_grid_year_index.csv as an Earth Engine table asset and set
 * INDEX_ASSET below. The index contains one row per Guangxi grid cell and
 * target year (342 cells x 25 years = 8,550 rows). This script creates one
 * task per year and grid batch, rather than one task per grid-year.
 *
 * Each exported CSV contains one row per local hour and grid cell. The
 * columns are the direct weather contract used by the local FSIM reader:
 * datetime, dew_point, precip_acc_period_adjusted, relative_humidity,
 * air_temp, and wind_speed, plus grid metadata and audit columns.
 *
 * ERA5 precipitation is the hourly total for the interval ending at the
 * timestamp. precipitation_interval_start_bjt records the interval start.
 * Timestamps are retained as Asia/Shanghai local clock values and are not
 * shifted to UTC in the exported datetime field.
 */

var INDEX_ASSET = 'projects/glossy-essence-477012-s7/assets/gx_grid_year_index';
var DRIVE_FOLDER = 'ERA5_dm_first_inf_guangxi_grid';
var TIME_ZONE = 'Asia/Shanghai';
var DATASET = 'ECMWF/ERA5/HOURLY';

// Use these controls to run the task list in manageable batches.
// The existing local archive is present for the other years; 2004 needs a
// fresh export because its JSON files lack dew_point and relative_humidity.
var START_YEAR = 2004;
var END_YEAR = 2004;
var GRID_BATCH_SIZE = 50;
var START_GRID_BATCH = 0;
var MAX_GRID_BATCHES = null;
var CHECK_TIME_COVERAGE = false;

var BANDS = [
  'temperature_2m',
  'dewpoint_temperature_2m',
  'total_precipitation',
  'u_component_of_wind_10m',
  'v_component_of_wind_10m'
];
var era5 = ee.ImageCollection(DATASET).select(BANDS);
var nativeProjection = ee.Image(era5.first()).select('temperature_2m').projection();

var SELECTORS = [
  'site', 'site_year', 'province', 'grid_id', 'weather_grid_id',
  'latitude', 'longitude', 'year', 'datetime', 'time_bjt', 'date_bjt', 'hour_bjt',
  'timezone', 'time_utc', 'time_start_millis',
  'precipitation_interval_start_bjt', 'air_temp', 'dew_point',
  'relative_humidity', 'precip_acc_period_adjusted', 'wind_speed',
  'valid_input_band_count', 'source'
];

function numberProperty(properties, name) {
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

function makeYearPlan(row, points) {
  var start = ee.Date.fromYMD(row.year, 1, 1, TIME_ZONE);
  var end = ee.Date.fromYMD(row.year + 1, 1, 1, TIME_ZONE);
  return {
    year: row.year,
    batch: row.batch,
    points: points,
    start: start,
    end: end,
    expectedHours: (Date.UTC(row.year + 1, 0, 1) -
      Date.UTC(row.year, 0, 1)) / 3600000,
    images: era5.filterDate(start, end).sort('system:time_start')
  };
}

function exportPlan(plan) {
  var imageList = plan.images.toList(plan.images.size());
  var hourly = ee.FeatureCollection(imageList.map(function(item) {
    var image = ee.Image(item);
    var time = ee.Date(image.get('system:time_start'));
    var temperature = image.select('temperature_2m').subtract(273.15)
      .rename('air_temp');
    var dewPoint = image.select('dewpoint_temperature_2m').subtract(273.15)
      .rename('dew_point');
    var precipitation = image.select('total_precipitation').multiply(1000)
      .rename('precip_acc_period_adjusted');
    var u = image.select('u_component_of_wind_10m');
    var v = image.select('v_component_of_wind_10m');
    var relativeHumidity = temperature.expression(
      '100 * exp(17.62 * Td / (243.12 + Td) - ' +
      '17.62 * T / (243.12 + T))',
      {T: temperature, Td: dewPoint}
    ).clamp(0, 100).rename('relative_humidity');
    var validCount = image.mask().unmask(0).reduce(ee.Reducer.sum())
      .rename('valid_input_band_count');
    var weather = temperature
      .addBands(dewPoint)
      .addBands(relativeHumidity)
      .addBands(precipitation)
      .addBands(u.pow(2).add(v.pow(2)).sqrt().rename('wind_speed'))
      .addBands(validCount);

    return weather.reduceRegions({
      collection: plan.points,
      reducer: ee.Reducer.first(),
      crs: nativeProjection,
      scale: nativeProjection.nominalScale(),
      tileScale: 4
    }).map(function(feature) {
      return feature.set({
        year: plan.year,
        timezone: TIME_ZONE,
        datetime: time.format('yyyy-MM-dd HH:mm:ss', TIME_ZONE),
        time_bjt: time.format('yyyy-MM-dd HH:mm:ss', TIME_ZONE),
        date_bjt: time.format('yyyy-MM-dd', TIME_ZONE),
        hour_bjt: time.get('hour', TIME_ZONE),
        time_utc: time.format('yyyy-MM-dd HH:mm:ss', 'UTC'),
        time_start_millis: time.millis(),
        precipitation_interval_start_bjt: time.advance(-1, 'hour')
          .format('yyyy-MM-dd HH:mm:ss', TIME_ZONE),
        source: DATASET
      });
    });
  })).flatten();

  var prefix = 'ERA5_GX_' + plan.year + '_batch_' +
    String(plan.batch).padStart(3, '0');
  Export.table.toDrive({
    collection: hourly,
    description: prefix,
    folder: DRIVE_FOLDER,
    fileNamePrefix: prefix,
    fileFormat: 'CSV',
    selectors: SELECTORS
  });
}

// The index is queried one year at a time. Querying all 8,550 rows with
// getInfo() exceeds Earth Engine's 5,000-element client-side limit.
var indexTable = ee.FeatureCollection(INDEX_ASSET);

function readYearRows(year) {
  var indexInfo = indexTable.filter(ee.Filter.eq('year', year)).getInfo();
  if (!indexInfo || !indexInfo.features || !indexInfo.features.length) {
    throw new Error('No Guangxi grid rows found for ' + year + '.');
  }
  return indexInfo.features.map(function(feature, index) {
  var properties = feature.properties || {};
  var site = textProperty(properties, 'site');
  var siteYear = textProperty(properties, 'site_year');
  var province = textProperty(properties, 'province') || 'Guangxi';
  var gridId = numberProperty(properties, 'grid_id');
  var weatherGridId = numberProperty(properties, 'weather_grid_id');
  var latitude = numberProperty(properties, 'latitude');
  var longitude = numberProperty(properties, 'longitude');
  // Earth Engine may promote latitude/longitude to point geometry during
  // table upload and omit them from feature.properties.
  if (!isFinite(latitude)) latitude = geometryCoordinate(feature, 1);
  if (!isFinite(longitude)) longitude = geometryCoordinate(feature, 0);
  var year = numberProperty(properties, 'year');

  if (!isFinite(gridId) || !isFinite(weatherGridId) ||
      !isFinite(latitude) || !isFinite(longitude) || !isFinite(year) ||
      !site || !siteYear || Math.floor(year) !== year ||
      year < START_YEAR || year > END_YEAR ||
      latitude < -90 || latitude > 90 || longitude < -180 || longitude > 180) {
    throw new Error('Invalid Guangxi grid index row ' + (index + 1) + ': ' +
      JSON.stringify(properties));
  }
  return {
    site: site,
    site_year: siteYear,
    province: province,
    grid_id: gridId,
    weather_grid_id: weatherGridId,
    latitude: latitude,
    longitude: longitude,
    year: year
  };
  }).sort(function(a, b) { return a.grid_id - b.grid_id; });
}

var plans = [];
for (var year = START_YEAR; year <= END_YEAR; year++) {
  var yearRows = readYearRows(year);
  for (var offset = 0, batch = 0; offset < yearRows.length;
       offset += GRID_BATCH_SIZE, batch += 1) {
    if (batch < START_GRID_BATCH) continue;
    if (MAX_GRID_BATCHES !== null &&
        batch >= START_GRID_BATCH + MAX_GRID_BATCHES) break;
    var batchRows = yearRows.slice(offset, offset + GRID_BATCH_SIZE);
    var points = ee.FeatureCollection(batchRows.map(function(row) {
      return ee.Feature(ee.Geometry.Point([row.longitude, row.latitude]), row);
    }));
    plans.push(makeYearPlan({year: year, batch: batch}, points));
  }
}

if (!plans.length) throw new Error('No export plans selected.');

if (CHECK_TIME_COVERAGE) {
  var audits = plans.map(function(plan) {
    return {
      year: plan.year,
      batch: plan.batch,
      expected_hours: plan.expectedHours,
      actual_hours: plan.images.size(),
      unique_hours: plan.images.aggregate_count_distinct('system:time_start'),
      start_bjt: plan.start.format('yyyy-MM-dd HH:mm:ss', TIME_ZONE),
      last_bjt: plan.end.advance(-1, 'hour').format(
        'yyyy-MM-dd HH:mm:ss', TIME_ZONE)
    };
  });
  var auditRows = ee.FeatureCollection(audits.map(function(row) {
    return ee.Feature(null, row);
  })).getInfo().features.map(function(feature) {
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
print('Each task exports one year and up to ' + GRID_BATCH_SIZE + ' grids.');
print('Start the generated tasks in the Earth Engine Tasks panel.');
Map.setCenter(108.5, 23.5, 6);
