import geopandas as gpd

# Path to your .shp file
path = "shrug-pc11-village-poly-shp/village_modified.shp"

# Load the shapefile
gdf = gpd.read_file(path)

# Show first few rows and metadata
print(gdf.head())
print(gdf.crs)
print(gdf.shape)
