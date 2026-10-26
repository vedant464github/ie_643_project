import requests
import pandas as pd

def fetch_weather(name, lat, lon, start_date, end_date):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean",
        "timezone": "Asia/Kolkata",
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()["daily"]
    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
    df.to_csv(f"data/raw/climate/{name}.csv")
    print(name, df.shape, df.index.min(), "to", df.index.max())
    return df

if __name__ == "__main__":
    # Hill station
    fetch_weather("shimla", lat=31.1048, lon=77.1734, start_date="2015-01-01", end_date="2026-09-30")
    # Plains
    fetch_weather("delhi", lat=28.7041, lon=77.1025, start_date="2015-01-01", end_date="2026-09-30")