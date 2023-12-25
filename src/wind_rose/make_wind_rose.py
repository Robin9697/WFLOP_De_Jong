import xarray as xr
import pandas as pd
import numpy as np
import math as m
import requests 
from pathlib import Path
import os

download = False
folder = "src\wind_rose"

# =============== Input =====================
api_key = 'eyJvcmciOiI1ZTU1NGUxOTI3NGE5NjAwMDEyYTNlYjEiLCJpZCI6IjNjOTVkZTIxOGUyNDQxN2M5M2M3ODgzOTNlMTliNjBhIiwiaCI6Im11cm11cjEyOCJ9' #get API key at https://developer.dataplatform.knmi.nl/get-started#obtain-an-api-key
dataset_name = 'wins50_ctl_nl_ts_singlepoint' #dataset details can be found at https://dataplatform.knmi.nl/dataset/?q=wins50
dataset_version = '1' #senario with present (operations) wind farms as of 2020
lookup = pd.read_csv(os.path.join(os.path.abspath(os.getcwd()), folder, "wins50_singlepoint_lookuptable.csv")) #from https://wins50.nl/data/

parcel = pd.read_csv(os.path.join(os.path.abspath(os.getcwd()), folder, "parcel.csv")) #ijmuiden ver
hub_height = 150. #IEA_15MW

# ============== Get grid point ========================

xc = np.min(parcel["Longitude"]) + 0.5*(np.max(parcel["Longitude"])-np.min(parcel["Longitude"]))
yc = np.min(parcel["Latitude"])  + 0.5*(np.max(parcel["Latitude"])-np.min(parcel["Latitude"]))

distances = np.sqrt((lookup["lon"] - xc)**2 + (lookup["lat"] - yc)**2)
min_index = np.argmin(distances)
grid_point = lookup.iloc[np.argmin(distances)]
ix = int(grid_point["ix"])
iy = int(grid_point["iy"])

# ================ Download data ============================

if download:
    datadir = "./"
    api_url = "https://api.dataplatform.knmi.nl/open-data"
    api_version = "v1"

    f = (f"WINS50_43h21_fERA5_CTL_ptA_NETHERLANDS.NL_ix{ix:03d}_iy{iy:03d}_2019010100-2022010100_v1.0.nc".format(ix=ix, iy=iy))
        
    get_file_response = requests.get(f"{api_url}/{api_version}/datasets/{dataset_name}/versions/{dataset_version}/files/{f}/url", 
                                        headers={"Authorization": api_key})
    if (get_file_response.status_code==200):
        print(f"Found file {f}! Downloading...")
        download_url = get_file_response.json().get("temporaryDownloadUrl")
        dataset_file_response = requests.get(download_url)
        
        # Write dataset file to disk
        filename = f"{datadir}/{f}"
        p = Path(filename)
        p.write_bytes(dataset_file_response.content)

    else:
        print(f"Something went wrong... check input...{f} not found" )
else:
    filename = './/WINS50_43h21_fERA5_CTL_ptA_NETHERLANDS.NL_ix058_iy110_2019010100-2022010100_v1.0.nc'
    print(filename)

# ============== Make wind rose ====================

ds = xr.open_dataset(os.path.join(os.path.abspath(os.getcwd()), folder, filename))
df = ds.to_dataframe()
df = df.dropna() #drop 2020 data because it is empty, just 2019 data is left
df = df.reset_index() #flatten the dataset
df = df[df['height'] == hub_height] #get data from hub height
df = df.reset_index(drop=True)
ws_list, wd_list, freq_list = [], [], []

step_ws = 1
step_wd = 1

for ws in range(0, m.ceil(np.max(df['wspeed']))+1, step_ws):
    for wd in range(0, 360, step_wd):
        df2 = df.copy()
        freq = len(df2[(df2["wspeed"]< ws+0.5*step_ws) & (df2["wspeed"]>= ws-0.5*step_ws) & (df2["wdir"]< wd+0.5*step_wd) & (df2["wdir"]>= wd-0.5*step_wd)])
        ws_list.append(float(ws))
        wd_list.append(float(wd))
        if wd == 0:
            freq2 = len(df2[(df2["wspeed"]< ws+0.5*step_ws) & (df2["wspeed"]>= ws-0.5*step_ws) & (df2["wdir"]>= 360-0.5*step_wd)])
            freq = freq + freq2
        freq_list.append(freq)

wind_rose = pd.DataFrame(data={'ws': ws_list,'wd': wd_list, 'freq_val': freq_list/np.sum(freq_list)})  
name = 'wind_rose.csv'
wind_rose.to_csv(os.path.join(os.path.abspath(os.getcwd()), folder, name), index=False)