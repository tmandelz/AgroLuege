# %%
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
from datetime import datetime
# Your client credentials
client_id = 'sh-3e5d21dd-e557-43be-a843-eb31c444dbc0'
client_secret = 'hDgFQ77SU5QTpMcqAYiDfF9yhLqjciAB'

# Create a session
client = BackendApplicationClient(client_id=client_id)
oauth = OAuth2Session(client=client)

# Get token for the session
token = oauth.fetch_token(token_url='https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token',
                          client_secret=client_secret)


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

# Define the start and end months (as datetime objects)
start_month = datetime(2023, 1, 1)
end_month = datetime(2023, 2, 1)

# Create a loop to iterate through the months
current_month = start_month
while current_month <= end_month:
    print(current_month)  # Print the current month and year
    # print(end_month)
    # Increment to the next month
    if current_month.month == 12:
        current_month = current_month.replace(year=current_month.year + 1, month=1)
    else:
        current_month = current_month.replace(month=current_month.month + 1)
    
    request = {
        "input": {
            "bounds": {
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
                "bbox": [
                    # lon,lat,lon,lat
                    # 46.812455,7.676219,46.814992,7.681600
                    7.676219,46.812455,7.681600,46.814992
                    # 7.676219,46.812455,7.681600,46.814992
                ],
            },
            "data": [
                {
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {
                            "from": f"2022-{month}-01T00:00:00Z",
                            "to": f"2022-{month}-28T00:00:00Z",
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
    f = open(f'../data/here_{month}.png', 'wb')
    f.write(response.content)
    f.close()

# %%
type(response.content)
# %%
f = open('../data/here_3.png', 'wb')
f.write(response.content)
f.close()
# %%
f = open('../data/here_2.png', 'wb')
f.write(response.content)
f.close()
# %%
f = open('../data/here.png', 'wb')
f.write(response.content)
f.close()

#%%


# %%
