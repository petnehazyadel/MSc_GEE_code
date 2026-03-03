// 1. Terület definíciója
var kisBalatonArea = ee.Geometry.Rectangle([17.00, 46.50, 17.40, 46.80]);

// 2. Sentinel-2 képgyűjtemény lekérdezése (2024-ben pl.)
var s2 = ee.ImageCollection('COPERNICUS/S2_SR')
  .filterBounds(kisBalatonArea)
  .filterDate('2024-01-01', '2024-12-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
  .select(['B4', 'B8']);

// 3. Képek listává alakítása
var imageList = s2.toList(s2.size());

// 4. ET0 számításához szükséges meteorológiai adatok
function getEra5Image(date) {
  var era5 = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
    .filterDate(date, date.advance(1, 'day'))
    .first();
  return era5;
}

// 5. FAO-56 Penman–Monteith ET0 számítás (MJ/m2/day → mm/day)
function calcET0(eraImage) {
  var Rn = eraImage.select('surface_net_solar_radiation_sum').divide(1e6); // MJ/m2/day
  var T = eraImage.select('temperature_2m').subtract(273.15); // Celsius
  var Td = eraImage.select('dewpoint_temperature_2m').subtract(273.15); // Celsius
  var u10 = eraImage.select('u_component_of_wind_10m');
  var v10 = eraImage.select('v_component_of_wind_10m');
  var u2 = u10.pow(2).add(v10.pow(2)).sqrt(); // Szélsebesség
  
  var es = T.expression(
    '0.6108 * exp((17.27 * T) / (T + 237.3))',
    {T: T}
  );
  
  var ea = Td.expression(
    '0.6108 * exp((17.27 * Td) / (Td + 237.3))',
    {Td: Td}
  );
  
  var delta = T.expression(
    '4098 * (0.6108 * exp((17.27*T)/(T+237.3))) / pow((T+237.3),2)',
    {T: T}
  );
  
  var gamma = ee.Image.constant(0.066); // pszichrometrikus állandó (kPa/°C)
  
  var ET0 = delta.multiply(Rn)
    .add(gamma.multiply(900).divide(T.add(273)).multiply(u2).multiply(es.subtract(ea)))
    .divide(delta.add(gamma.multiply(ee.Image(1).add(u2.multiply(0.34)))))
    .rename('ET0_mm_day');
  
  return ET0;
}

// 6. Kc hozzárendelés NDVI osztályok alapján :
function assignKc(ndvi) {
  var kc = ndvi
    .where(ndvi.lt(0.1), 0.25)                // <0.1 csupasz talaj
    .where(ndvi.gte(0.1).and(ndvi.lt(0.2)), 1.05)  // 0.1-0.2 víz
    .where(ndvi.gte(0.2).and(ndvi.lt(0.3)), 0.4)   // 0.2-0.3
    .where(ndvi.gte(0.3).and(ndvi.lt(0.4)), 0.7)   // 0.3-0.4
    .where(ndvi.gte(0.4).and(ndvi.lt(0.5)), 0.9)   // 0.4-0.5
    .where(ndvi.gte(0.5).and(ndvi.lt(0.6)), 1.05)  // 0.5-0.6
    .where(ndvi.gte(0.6).and(ndvi.lt(0.7)), 1.10)  // 0.6-0.7
    .where(ndvi.gte(0.7), 1.15);                    // >0.7 állandó növényzet
  return kc.rename('Kc');
}

// 7. Export ciklus az első 10 Sentinel-2 képre 2024-ből
var exportCount = 10;

for (var i = 0; i < exportCount; i++) {
  var image = ee.Image(imageList.get(i));
  var date = ee.Date(image.get('system:time_start'));
  var dateString = date.format('YYYY-MM-dd').getInfo();

  // NDVI számítása
  var ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI').clip(kisBalatonArea);

  // Kc hozzárendelés
  var kc = assignKc(ndvi);

  // ERA5 adat lekérése
  var eraImage = getEra5Image(date);

  // ET0 számítás
  var ET0 = calcET0(eraImage).clip(kisBalatonArea);

  // Tényleges ET
  var ETa = ET0.multiply(kc).rename('ETa_mm_day');

  // Exportálás Google Drive-ra
  Export.image.toDrive({
    image: ETa,
    description: 'ETa_KisBalaton_' + dateString,
    folder: 'EarthEngine',
    fileNamePrefix: 'ETa_KisBalaton_' + dateString,
    region: kisBalatonArea,
    scale: 10,
    crs: 'EPSG:4326',
    maxPixels: 1e13
  });
}