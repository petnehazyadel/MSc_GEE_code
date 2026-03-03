
// 1. Terület definíciója
var kisBalatonArea = ee.Geometry.Rectangle([17.00, 46.50, 17.40, 46.80]);

// 2. Sentinel-2 képgyűjtemény lekérdezése
var s2 = ee.ImageCollection('COPERNICUS/S2_SR')
  .filterBounds(kisBalatonArea)
  .filterDate('2024-01-01', '2024-12-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
  .select(['B4', 'B3', 'B2']);

// 3. Képek számának kiíratása
print('Összes Sentinel-2 kép 2024-ben:', s2.size());



NDVI számítás és letöltés:

// 1. Terület definíciója
var kisBalatonArea = ee.Geometry.Rectangle([17.00, 46.50, 17.40, 46.80]);

// 2. Sentinel-2 képgyűjtemény lekérdezése
var s2 = ee.ImageCollection('COPERNICUS/S2_SR')
  .filterBounds(kisBalatonArea)
  .filterDate('2024-01-01', '2024-12-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
  .select(['B4', 'B8']);  // B4 = Red, B8 = NIR

// 3. Képek listává alakítása
var imageList = s2.toList(s2.size());

// 4. Export mennyisége – most csak 10 kép
var exportCount = 10;

// 5. Ciklus az NDVI számításra és exportálásra
for (var i = 0; i < exportCount; i++) {
  var image = ee.Image(imageList.get(i));

  // NDVI számítása: (NIR - RED) / (NIR + RED)
  var ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI');

  // Kép dátuma (export fájlnévhez)
  var date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd').getInfo();

  // Export parancs
  Export.image.toDrive({
    image: ndvi.clip(kisBalatonArea),
    description: 'NDVI_KisBalaton_' + date,
    folder: 'EarthEngine',
    fileNamePrefix: 'NDVI_KisBalaton_' + date,
    region: kisBalatonArea,
    scale: 10,
    crs: 'EPSG:4326',
    maxPixels: 1e13
  });
}

