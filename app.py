import streamlit as st
import pandas as pd
import joblib
import requests
import numpy as np

class DecisionTreeRegressorScratch:
    def __init__(self, max_depth=None, min_samples_split=2, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.tree_ = None

    class Node:
        def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
            self.feature_idx = feature_idx
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value

    def fit(self, X, y):
        n_features = X.shape[1]
        if self.max_features is None:
            self.max_features = n_features
        else:
            self.max_features = min(self.max_features, n_features)
        self.tree_ = self._build_tree(X, y, depth=0)
        return self

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        if (self.max_depth is not None and depth >= self.max_depth) or n_samples < self.min_samples_split:
            return self.Node(value=y.mean())

        feat_idxs = np.random.choice(n_features, self.max_features, replace=False)

        best_feat, best_thresh, best_mse = None, None, np.inf
        for feat in feat_idxs:
            thresholds = np.unique(X[:, feat])
            for thr in thresholds:
                left_mask = X[:, feat] <= thr
                if left_mask.sum() == 0 or left_mask.sum() == n_samples:
                    continue
                y_left, y_right = y[left_mask], y[~left_mask]
                mse_left = np.var(y_left) * y_left.shape[0]
                mse_right = np.var(y_right) * y_right.shape[0]
                mse_total = (mse_left + mse_right) / n_samples
                if mse_total < best_mse:
                    best_feat, best_thresh, best_mse = feat, thr, mse_total

        if best_feat is None:
            return self.Node(value=y.mean())

        left_mask = X[:, best_feat] <= best_thresh
        left = self._build_tree(X[left_mask], y[left_mask], depth+1)
        right = self._build_tree(X[~left_mask], y[~left_mask], depth+1)
        return self.Node(feature_idx=best_feat, threshold=best_thresh, left=left, right=right)

    def _predict_one(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree_) for x in X])

class RandomForestRegressorScratch:
    def __init__(self, n_estimators=10, max_depth=None, min_samples_split=2,
                 max_features='sqrt', bootstrap=True, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.bootstrap = bootstrap
        self.trees_ = []
        self.random_state = random_state
        self.max_features = max_features

    def fit(self, X, y):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape

        if self.max_features == 'sqrt':
            max_feats = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            max_feats = int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            max_feats = self.max_features
        else:
            max_feats = n_features

        self.trees_ = []
        for i in range(self.n_estimators):
            if self.bootstrap:
                idxs = np.random.choice(n_samples, size=n_samples, replace=True)
                X_sample, y_sample = X[idxs], y[idxs]
            else:
                X_sample, y_sample = X, y

            tree = DecisionTreeRegressorScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=max_feats
            )
            tree.fit(X_sample, y_sample)
            self.trees_.append(tree)
        return self

    def predict(self, X):
        preds = np.array([tree.predict(X) for tree in self.trees_])
        return np.mean(preds, axis=0)


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
    input_vals = np.array([[st.session_state[k] for k in SENSOR_KEYS]])
    aqi_pred = model.predict(input_vals)[0]
    st.success(f"Predicted AQI: {aqi_pred:.2f}")