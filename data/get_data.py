# %%
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
from sentinelhub import CRS, BBox, DataCollection, SHConfig


config= SHConfig(
    instance_id="b0be4a8f-70c8-48ad-afa7-a4c87bf9d08d",
    sh_client_id='f721baa0-b97c-4d13-8987-94b7a5049330',
    sh_client_secret='|_k2TatTELEJ9PWmT{&3<@+_]3.Jkj73G:dv4IBg',
    sh_base_url='https://services.sentinel-hub.com',
   sh_auth_base_url='https://services.sentinel-hub.com',
)
if config.sh_client_id == "" or config.sh_client_secret == "":
    print("Warning! To use Sentinel Hub Catalog API, please provide the credentials (client ID and client secret).")

#%%
from sentinelhub import SentinelHubCatalog

catalog = SentinelHubCatalog(config=config)

catalog.get_info()
collections = catalog.get_collections()

collections = [collection for collection in collections if not collection["id"].startswith(("byoc", "batch"))]

collections
print("")
#%%
from sentinelhub import WebFeatureService, BBox, CRS, SHConfig
import geopandas as gpd
def get_Sentinel_Data(bbox, config):
    search_bbox = BBox(bbox=bbox, crs=CRS.WGS84)
    search_time_interval = ('2022-01-01T00:00:00', '2022-12-31T23:59:59')

    wfs_iterator = WebFeatureService(
        search_bbox,
        search_time_interval,
        data_collection=DataCollection.SENTINEL2_L1C,
        maxcc=0.0,
        config=config
    )

    ids = []
    coords = []
    # for tile_info in wfs_iterator:
    #     print("\n New tile: \n")
    #     print("coordinates info: {} \n".format(tile_info['geometry']['coordinates']))
    #     print("id info: {} \n".format(tile_info['properties']['id']))
    #     print("cloud info: {} \n".format(tile_info['properties']['cloudCoverPercentage']))
    #     ids.append(tile_info['properties']['id'])
    #     coords.append("coordinates info: {} \n".format(tile_info['geometry']['coordinates']))
        
    return wfs_iterator


sentinel_ = get_Sentinel_Data((47.39,8.04,47.4,8.1),config)

# list_geom = []
# for i in sentinel_:
#     list_geom.append(i["geometry"]["coordinates"])


#%%
# list_geom
