var area = ee.Geometry.Rectangle([17.00, 46.50, 17.40, 46.80]);

var img = ee.ImageCollection('COPERNICUS/S2_SR')
  .filterBounds(area)
  .filterDate('2018-01-01', '2018-12-31')
  .sort('CLOUDY_PIXEL_PERCENTAGE')
  .first();

print('Talált kép:', img);

var ndvi = img.normalizedDifference(['B8', 'B4'])
  .rename('NDVI')
  .clip(area);

Map.centerObject(area, 9);
Map.addLayer(ndvi, {min: -0.2, max: 0.8, palette: ['brown','yellow','green']}, 'NDVI 2018');

Export.image.toDrive({
  image: ndvi,
  description: 'NDVI_KisBalaton_2018',
  folder: 'EarthEngine',
  fileNamePrefix: 'NDVI_KisBalaton_2018',
  region: area,
  scale: 10,
  maxPixels: 1e13
});
print(
  'Kiválasztott kép dátuma:',
  ee.Date(img.get('system:time_start')).format('YYYY-MM-dd')
);
