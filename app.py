import streamlit as st
import pandas as pd
import joblib
import requests

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
      'authorization': 'bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJodHRwczovL2FpcnF1YWxpdHkuYXFpLmluL2FwaS92MS9sb2dpbiIsImlhdCI6MTc1MDc2ODM1NiwiZXhwIjoxNzUxOTc3OTU2LCJuYmYiOjE3NTA3NjgzNTYsImp0aSI6Ijg4akE5RnNyUDFBUmpsMjIiLCJzdWIiOiIyOTE2OCIsInBydiI6IjIzYmQ1Yzg5NDlmNjAwYWRiMzllNzAxYzQwMDg3MmRiN2E1OTc2ZjcifQ.KzN_x9HfQ4FLKLxYx9I-y9GXnQitDAbDm_0A7ue5y2M',
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
    keys = ["pm25","pm10","co","so2","no2","o3"]
    return {item["sensorName"]: item["sensorData"] for item in airquality if item["sensorName"] in keys}

@st.cache_resource
def load_model(path="RFAQI.joblib"):
    return joblib.load(path)

model = load_model()

for p in ['pm25','pm10','co','so2','no2','o3']:
    if p not in st.session_state:
        st.session_state[p] = 0.0

st.title("AQI Prediction Interface")
st.write("Choose an input mode and set pollutant levels.")
mode = st.radio("Input Mode", ["Manual","From Location"])

if mode == "From Location":
    loc = st.selectbox("Location", list(locations.keys()))
    if st.button("Fetch Pollutants"):
        slug = locations[loc]
        with st.spinner(f"Fetching for {loc}..."):
            vals = getDetails(slug)
        if vals:
            for k,v in vals.items():
                st.session_state[k] = v
            st.success(f"Loaded data for {loc}")
        else:
            st.error("Failed to fetch data.")
else:
    st.subheader("Manual Inputs")
    st.session_state['pm25'] = st.number_input('PM2.5', min_value=0.0, max_value=500.0, value=float(st.session_state.get('pm25', 0.0)))
    st.session_state['pm10'] = st.number_input('PM10', min_value=0.0, max_value=500.0, value=float(st.session_state.get('pm10', 0.0)))
    st.session_state['co'] = st.number_input('CO', min_value=0.0, max_value=500.0, value=float(st.session_state.get('co', 0.0)))
    st.session_state['so2'] = st.number_input('SO2', min_value=0.0, max_value=500.0, value=float(st.session_state.get('so2', 0.0)))
    st.session_state['no2'] = st.number_input('NO2', min_value=0.0, max_value=500.0, value=float(st.session_state.get('no2', 0.0)))
    st.session_state['o3'] = st.number_input('O3', min_value=0.0, max_value=500.0, value=float(st.session_state.get('o3', 0.0)))

st.subheader("Pollutant Levels")
st.table(pd.DataFrame({
    'Pollutant':['PM2.5','PM10','CO','SO2','NO2','O3'],
    'Value':[st.session_state['pm25'], st.session_state['pm10'],
             st.session_state['co'], st.session_state['so2'],
             st.session_state['no2'], st.session_state['o3']]
}))

st.subheader("Predicted AQI")
if st.button("Predict AQI"):
    input_df = pd.DataFrame({
        'pm25':[st.session_state['pm25']],
        'pm10':[st.session_state['pm10']],
        'co':  [st.session_state['co']],
        'so2': [st.session_state['so2']],
        'no2': [st.session_state['no2']],
        'o3':  [st.session_state['o3']]
    })
    aqi = model.predict(input_df)[0]
    st.success(f"{aqi:.2f}")
