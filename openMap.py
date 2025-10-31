import math
import requests
import numpy as np
import pandas as pd
import time

center_lat = 33.3716
center_lon = 77.5946
radius_km = 5
step_km = 0.08
batch_size = 1000
output_csv = "terrain_elevation.csv"
deg_per_km = 1 / 111.32
step_deg = step_km * deg_per_km
radius_deg = radius_km * deg_per_km
lats, lons = [], []
for lat in np.arange(center_lat - radius_deg, center_lat + radius_deg, step_deg):
    for lon in np.arange(center_lon - radius_deg, center_lon + radius_deg, step_deg):
        if math.dist((lat, lon), (center_lat, center_lon)) <= radius_deg:
            lats.append(lat)
            lons.append(lon)

coords = [{"latitude": la, "longitude": lo} for la, lo in zip(lats, lons)]
print(f"Generated {len(coords)} points to query...")
all_results = []
for i in range(0, len(coords), batch_size):
    batch = coords[i:i+batch_size]
    try:
        response = requests.post("https://api.open-elevation.com/api/v1/lookup", json={"locations": batch}, timeout=20)
        if response.status_code == 200 and response.text.strip():
            results = response.json().get("results", [])
            all_results.extend(results)
            print(f"Batch {i//batch_size+1}: got {len(results)} points")
        else:
            print(f"Batch {i//batch_size+1}: empty or invalid response (status {response.status_code})")
    except Exception as e:
        print(f"Error in batch {i//batch_size+1}: {e}")
    time.sleep(1)  # avoid rate limit
df = pd.DataFrame(all_results)
df.to_csv(output_csv, index=False)
print(f"\nSaved {len(df)} elevation points to {output_csv}")

