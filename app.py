import streamlit as st
import pandas as pd
import joblib
import requests

SENSOR_KEYS = ['aqi', 'co', 'dew', 'h', 'no2', 'o3', 'pm10', 'pm25', 'so2', 't', 'w']

locations = {
    "Aceh": 'indonesia/aceh',
    "Bengkulu": 'indonesia/bengkulu',
    "Daerah Khusus Ibukota Jakarta": 'indonesia/daerah-khusus-ibukota-jakarta',
    "Jakarta": 'indonesia/jakarta',
    "Jambi": 'indonesia/jambi',
    "Jawa Barat": 'indonesia/jawa-barat',
    "Jawa Tengah": 'indonesia/jawa-tengah',
    "Jawa Timur": 'indonesia/jawa-timur',
    "Kalimantan Barat": 'indonesia/kalimantan-barat',
    "Kalimantan Selatan": 'indonesia/kalimantan-selatan',
    "Kalimantan Tengah": 'indonesia/kalimantan-tengah',
    "Kalimantan Timur": 'indonesia/kalimantan-timur',
    "Kepulauan Riau": 'indonesia/kepulauan-riau',
    "Lampung": 'indonesia/lampung',
    "North Sulawesi": 'indonesia/north-sulawesi',
    "Papua Barat": 'indonesia/papua-barat',
    "Riau": 'indonesia/riau',
    "Sulawesi Selatan": 'indonesia/sulawesi-selatan',
    "Sulawesi Tengah": 'indonesia/sulawesi-tengah',
    "Sumatera Barat": 'indonesia/sumatera-barat',
    "Sumatera Selatan": 'indonesia/sumatera-selatan',
    "Sumatera Utara": 'indonesia/sumatera-utara',
    "Yogyakarta": 'indonesia/yogyakarta'
}

@st.cache_data(show_spinner=False)
def getDetails(slug):
    headers = {
      'accept': '*/*',
      'accept-language': 'en-US,en;q=0.9,id;q=0.8',
      'authorization': 'bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJodHRwczovL2FpcnF1YWxpdHkuYXFpLmluL2FwaS92MS9sb2dpbiIsImlhdCI6MTc1MTk1NzI2NiwiZXhwIjoxNzUzMTY2ODY2LCJuYmYiOjE3NTE5NTcyNjYsImp0aSI6Im5OWU5TRmZRWE5IOFZHMUUiLCJzdWIiOiIyOTE2OCIsInBydiI6IjIzYmQ1Yzg5NDlmNjAwYWRiMzllNzAxYzQwMDg3MmRiN2E1OTc2ZjcifQ.737ejQ9aFU5OeFwI47yO23IzW0lisWKXkUtvVBYgnNM',
      'origin': 'https://www.aqi.in',
      'priority': 'u=1, i',
      'referer': 'https://www.aqi.in/',
      'sec-ch-ua': '"Google Chrome";v="137", "Chromium";v="137", "Not/A)Brand";v="24"',
      'sec-ch-ua-mobile': '?1',
      'sec-ch-ua-platform': '"Android"',
      'sec-fetch-dest': 'empty',
      'sec-fetch-mode': 'cors',
      'sec-fetch-site': 'same-site',
      'slug': slug,
      'type': '2',
      'user-agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Mobile Safari/537.36',
  }
    resp = requests.get(
        'https://airquality.aqi.in/api/v1/getLocationDetailsBySlugNew',
        headers=headers
    )
    data = resp.json().get("data", [])
    if not data:
        return {}
    airquality = data[0].get("airquality", [])
    return {item["sensorName"]: item.get("sensorData", 0.0)
            for item in airquality
            if item.get("sensorName") in SENSOR_KEYS}

@st.cache_resource
def load_model(path="RFAQI_ScratchNew.joblib"):
    return joblib.load(path)

model = load_model()

for key in SENSOR_KEYS:
    if key not in st.session_state:
        st.session_state[key] = 0.0

st.title("AQI Prediction Interface")
st.write("Choose input mode and set sensor values for prediction.")
mode = st.radio("Input Mode", ["Manual", "From Location"])

if mode == "From Location":
    loc = st.selectbox("Location", list(locations.keys()))
    if st.button("Fetch Sensors"):
        slug = locations[loc]
        with st.spinner(f"Fetching data for {loc}..."):
            vals = getDetails(slug)
        if vals:
            for k, v in vals.items():
                st.session_state[k] = v
            st.success(f"Loaded data for {loc}")
        else:
            st.error("Failed to fetch data.")

else:
    st.subheader("Manual Sensor Inputs")
    for key in SENSOR_KEYS:
        label = key.upper() if len(key) <= 3 else key.capitalize()
        st.session_state[key] = st.number_input(
            label,
            min_value=0.0,
            max_value=1000.0,
            value=float(st.session_state.get(key, 0.0))
        )

st.subheader("Current Sensor Values")
st.table(pd.DataFrame({
    'Sensor': [k.upper() for k in SENSOR_KEYS],
    'Value': [st.session_state[k] for k in SENSOR_KEYS]
}))

st.subheader("Predicted AQI")
if st.button("Predict AQI"):
    input_df = pd.DataFrame({key: [st.session_state[key]] for key in SENSOR_KEYS})
    aqi_pred = model.predict(input_df)[0]
    st.success(f"Predicted AQI: {aqi_pred:.2f}")