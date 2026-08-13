import os

import requests
from langchain_core.tools import tool


@tool
def simpleinterest(p: float, r: float, t: int) -> float:
    """Calculate simple interest using the formula: SI = (P * R * T) / 100"""
    return (p * r * t) / 100
@tool
def electricity_bill(units: float) -> float:
    """Calculate electricity bill based on units consumed."""
    billamount=0
    rate = 0
    if units < 100:
        rate=3
    elif units>=100 and units < 200:
        rate=4
    elif units>=200 and units < 300:
            rate=5      
    else:
        rate=7
    billamount=units*rate
    return billamount    

billamont= electricity_bill.invoke({"units": 250})
print("Electricity bill is:", billamont)

@tool
def weather(location: str) -> str:
    """Get current temperature and humidity for a city or location."""
    api_key = os.getenv("OPENWEATHER_API_KEY", "your_api_key_here")
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": location,
        "units": "metric",
        "appid": api_key,
    }
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()
    temp = data["main"]["temp"]
    humidity = data["main"]["humidity"]
    description = data["weather"][0]["description"]
    return (
        f"Location: {data['name']}\n"
        f"Temperature: {temp}°C\n"
        f"Humidity: {humidity}%\n"
        f"Condition: {description}"
    )


si = simpleinterest.invoke({
    "p": 1000,
    "r": 5,
    "t": 2,
})
print("Simple Interest is:", si)

weather_info = weather.invoke({"location": "Bangalore"})
print("Weather is:")
print(weather_info)

