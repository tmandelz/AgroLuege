# %%
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
from datetime import datetime
import datetime
from dateutil.relativedelta import relativedelta
# Your client credentials
client_id = 'sh-3e5d21dd-e557-43be-a843-eb31c444dbc0'
client_secret = 'hDgFQ77SU5QTpMcqAYiDfF9yhLqjciAB'

# Create a session
client = BackendApplicationClient(client_id=client_id)
oauth = OAuth2Session(client=client)

# Get token for the session
token = oauth.fetch_token(token_url='https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token',
                          client_secret=client_secret)

#%%
evalscript = """
//VERSION=3
function setup() {
  return {
    input: ["B02", "B03", "B04"],
    mosaicking: Mosaicking.ORBIT,
    output: { id:"default", bands: 3}
  }
}

function updateOutputMetadata(scenes, inputMetadata, outputMetadata) {
    outputMetadata.userData = { "scenes":  scenes.orbits }
}

function evaluatePixel(samples) {
  return [ 2.5 * samples[0].B04, 2.5 * samples[0].B03, 2.5 * samples[0].B02 ]
}
"""
request = {
    "input": {
        "bounds": {
             "bbox": [
                13.822174072265625,
                45.85080395917834,
                14.55963134765625,
                46.29191774991382
            ]
        },
        "data": [
            {
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    "timeRange": {
                        "from": "2022-01-01T00:00:00Z",
                        "to": "2022-01-31T23:59:59Z"
                    }
                }
            }
        ]
    },
    "output": {
        "width": 512,
        "height": 512,
        "responses": [
            {
                "identifier": "default",
                "format": {
                    "type": "image/tiff"
                }
            },
            {
                "identifier": "userdata",
                "format": {
                    "type": "application/json"
                }
            }
        ]
    }
,    "evalscript": evalscript,
}

url = "https://sh.dataspace.copernicus.eu/api/v1/process"
response = oauth.post(url, json=request)
f = open(f'../data/here_.png', 'wb')
f.write(response.content)
f.close()

#%%
response.content
# %%
evalscript = """
//VERSION=3
function setup() {
  return {
    input: ["B02", "B03", "B04"],
    output: { bands: 3 },
    sampleType: "AUTO", // default value - scales the output values from [0,1] to [0,255].
  }
}

function evaluatePixel(sample) {
  return [2.5 * sample.B04, 2.5 * sample.B03, 2.5 * sample.B02]
}
"""
year = 2021  # Change this to the desired year
# Iterate through the months (from January to December)
for month in range(1, 13):
    # Get the first day of the month
    first_day = datetime.datetime(year, month, 1, 0, 0, 0, 0)

    # Calculate the last day of the month
    if month == 12:
        last_day = datetime.datetime(year + 1, 1, 1, 23, 59, 59, 999999)
    else:
        last_day = datetime.datetime(
            year, month + 1, 1, 0, 0, 0, 0) - datetime.timedelta(microseconds=1)

    # Print the first and last day of the month
    print(f"Month {month}: First Day = {first_day}, Last Day = {last_day}")
    request = {
        "input": {
            "bounds": {
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
                "bbox": [
                    # lon,lat,lon,lat
                    7.676219, 46.812455, 7.681600, 46.814992
                ],
            },
            "data": [
                {
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {
                            "from": first_day.strftime("%Y-%m-%dT%H:%M:%SZ"),
                            "to": last_day.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        }
                    },
                }
            ],
        },
        "output": {
            "width": 512,
            "height": 512,
        },
        "evalscript": evalscript,
    }

    url = "https://sh.dataspace.copernicus.eu/api/v1/process"
    response = oauth.post(url, json=request)
    f = open(f'../data/here_{first_day.month}.png', 'wb')
    f.write(response.content)
    f.close()
# %%
response
# %%
