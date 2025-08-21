from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import date, timedelta
import httpx, numpy as np

app = FastAPI(title="Crop Calendar (Live Weather)", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# baseline windows by crop (month,day); used as anchor then adjusted by weather
BASE = {
    "wheat": {"sowing_start": (10,15), "sowing_end": (11,30), "maturity_days": 120},
    "rice":  {"sowing_start": (6,1),   "sowing_end": (7,15),  "maturity_days": 130},
    "maize": {"sowing_start": (6,1),   "sowing_end": (7,31),  "maturity_days": 110},
}

class WeatherDay(BaseModel):
    date: str
    tmin: float
    tmax: float
    rain_mm: float

class CalendarResp(BaseModel):
    crop: str
    sowing_window: dict
    harvest_window: dict
    weather_summary: dict
    narrative: str

async def fetch_weather(lat: float, lon: float):
    # Open-Meteo free API (no key); next 16 days forecast + recent 10 days to get signal
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": "temperature_2m,precipitation",
        "daily": "temperature_2m_min,temperature_2m_max,precipitation_sum",
        "forecast_days": 16, "past_days": 10, "timezone": "auto"
    }
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        d = r.json()
    days = []
    for i, day in enumerate(d["daily"]["time"]):
        days.append(WeatherDay(
            date=day,
            tmin=float(d["daily"]["temperature_2m_min"][i]),
            tmax=float(d["daily"]["temperature_2m_max"][i]),
            rain_mm=float(d["daily"]["precipitation_sum"][i]),
        ))
    return days

def plan(crop: str, days: list[WeatherDay]):
    cfg = BASE[crop]
    # compute 10-day sliding means for rain and temperature
    rain = np.array([d.rain_mm for d in days], dtype="float32")
    tmin = np.array([d.tmin for d in days], dtype="float32")
    tmax = np.array([d.tmax for d in days], dtype="float32")
    tmean = (tmin + tmax) / 2

    # heuristic conditions
    if crop == "wheat":
        good_temp = (tmean >= 18) & (tmean <= 26)
        good_rain = (rain >= 2) & (rain <= 20)  # light/moderate
    elif crop == "rice":
        good_temp = (tmean >= 22) & (tmean <= 32)
        good_rain = (rain >= 5)                  # plenty of water
    else:  # maize
        good_temp = (tmean >= 20) & (tmean <= 30)
        good_rain = (rain >= 2)

    ok = good_temp & good_rain
    idxs = np.where(ok)[0]
    if len(idxs) == 0:
        # fallback to baseline window in current year
        y = date.today().year
        sow_start = date(y, *cfg["sowing_start"]).isoformat()
        sow_end   = date(y, *cfg["sowing_end"]).isoformat()
        maturity  = timedelta(days=cfg["maturity_days"])
        return {
            "sowing_window": {"start": sow_start, "end": sow_end},
            "harvest_window": {"start": (date.fromisoformat(sow_start)+maturity).isoformat(),
                               "end":   (date.fromisoformat(sow_end)+maturity).isoformat()},
            "summary": {"note": "Using baseline dates; current weather not favorable in forecast window."}
        }

    # choose earliest 5-day stretch that is favorable
    start_i = int(idxs[0])
    # extend while ok continues
    end_i = start_i
    while end_i+1 < len(ok) and ok[end_i+1]:
        end_i += 1
    sow_start = days[start_i].date
    sow_end   = days[min(end_i, start_i+4)].date  # 5-day sowing window
    maturity  = timedelta(days=cfg["maturity_days"])
    h_start   = (date.fromisoformat(sow_start) + maturity).isoformat()
    h_end     = (date.fromisoformat(sow_end)   + maturity).isoformat()

    return {
        "sowing_window": {"start": sow_start, "end": sow_end},
        "harvest_window": {"start": h_start, "end": h_end},
        "summary": {
            "avg_temp": float(tmean[start_i:end_i+1].mean()),
            "avg_rain": float(rain[start_i:end_i+1].mean()),
            "days_considered": int(end_i - start_i + 1)
        }
    }

@app.get("/calendar", response_model=CalendarResp)
async def calendar(lat: float, lon: float, crop: str):
    crop = crop.lower()
    if crop not in BASE:
        raise HTTPException(404, "Crop not supported yet (wheat|rice|maize)")
    days = await fetch_weather(lat, lon)
    plan_out = plan(crop, days)
    narrative = (
        f"For {crop}, sowing window is {plan_out['sowing_window']['start']} to {plan_out['sowing_window']['end']} "
        f"based on local forecast (avg temp/rain {plan_out['summary'].get('avg_temp','?'):.1f}°C/"
        f"{plan_out['summary'].get('avg_rain','?'):.1f} mm). "
        f"Expected harvest window: {plan_out['harvest_window']['start']} to {plan_out['harvest_window']['end']}."
    )
    return CalendarResp(
        crop=crop,
        sowing_window=plan_out["sowing_window"],
        harvest_window=plan_out["harvest_window"],
        weather_summary=plan_out["summary"],
        narrative=narrative
    )
