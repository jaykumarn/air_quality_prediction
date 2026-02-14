from flask import Flask, request, render_template, url_for, jsonify
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
import requests as http_requests

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Train model on startup (avoids pickle version mismatch entirely)
# ---------------------------------------------------------------------------
df = pd.read_csv('Dataset/Airquality_index.csv')
df = df.dropna()
df = df[df['PM 2.5'] > 0]
X = df[['T', 'TM', 'Tm', 'SLP', 'H', 'VV', 'V', 'VM']]
y = df['PM 2.5']
model = ExtraTreesRegressor(
    n_estimators=600,
    max_depth=51,
    min_samples_split=3,
    max_features='log2',
    random_state=42,
    n_jobs=-1
)
model.fit(X, y)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/weather', methods=['GET'])
def weather():
    """
    Accepts ?city=<name>
    1. Resolves city → lat/lon via Open-Meteo Geocoding API (free, no key needed)
    2. Fetches current weather via Open-Meteo Forecast API
    3. Returns JSON with pre-mapped form field values
    """
    city = request.args.get('city', '').strip()
    if not city:
        return jsonify({'error': 'City name is required.'}), 400

    try:
        # --- Step 1: Geocoding ------------------------------------------------
        geo_url = (
            'https://geocoding-api.open-meteo.com/v1/search'
            f'?name={city}&count=1&language=en&format=json'
        )
        geo_resp = http_requests.get(geo_url, timeout=10)
        geo_data = geo_resp.json()

        if not geo_data.get('results'):
            return jsonify({
                'error': f"City '{city}' not found. Please check the spelling and try again."
            }), 404

        result   = geo_data['results'][0]
        lat      = result['latitude']
        lon      = result['longitude']
        # Build a readable label: "London, United Kingdom"
        city_label = result['name']
        if result.get('country'):
            city_label += f", {result['country']}"

        # --- Step 2: Current weather + today's min/max -----------------------
        weather_url = (
            'https://api.open-meteo.com/v1/forecast'
            f'?latitude={lat}&longitude={lon}'
            '&current=temperature_2m,relative_humidity_2m,'
            'pressure_msl,wind_speed_10m,wind_gusts_10m,visibility'
            '&daily=temperature_2m_max,temperature_2m_min'
            '&wind_speed_unit=kmh'
            '&timezone=auto'
        )
        wx_resp = http_requests.get(weather_url, timeout=10)
        wx_data = wx_resp.json()

        c = wx_data['current']
        d = wx_data['daily']

        # --- Step 3: Map API fields → AQI form fields ------------------------
        # T   – Mean temperature (°C)         → current temperature_2m
        # TM  – Max temperature  (°C)         → today's temperature_2m_max
        # Tm  – Min temperature  (°C)         → today's temperature_2m_min
        # SLP – Sea-level pressure (hPa)      → current pressure_msl
        # H   – Relative humidity (%)         → current relative_humidity_2m
        # VV  – Visibility (km)               → current visibility (m) ÷ 1000
        # V   – Mean wind speed (km/h)        → current wind_speed_10m
        # VM  – Max wind speed  (km/h)        → current wind_gusts_10m
        return jsonify({
            'city': city_label,
            'T':    round(c['temperature_2m'], 2),
            'TM':   round(d['temperature_2m_max'][0], 2),
            'Tm':   round(d['temperature_2m_min'][0], 2),
            'SLP':  round(c['pressure_msl'], 2),
            'H':    round(float(c['relative_humidity_2m']), 2),
            'VV':   round(c['visibility'] / 1000, 2),
            'V':    round(c['wind_speed_10m'], 2),
            'VM':   round(c['wind_gusts_10m'], 2),
        })

    except http_requests.exceptions.ConnectionError:
        return jsonify({'error': 'Network error. Please check your internet connection.'}), 503
    except http_requests.exceptions.Timeout:
        return jsonify({'error': 'Weather service timed out. Please try again.'}), 504
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500


@app.route('/predict', methods=['POST'])
def predict():
    AQI_predict = None
    if request.method == 'POST':
        AQI_predict = model.predict([[
            request.form['T'],
            request.form['TM'],
            request.form['Tm'],
            request.form['SLP'],
            request.form['H'],
            request.form['VV'],
            request.form['V'],
            request.form['VM'],
        ]])
    return render_template('result.html', prediction=AQI_predict)


if __name__ == '__main__':
    app.run(debug=True)
