from flask import Flask, request, render_template
import pickle
import numpy as np
import time

app = Flask(__name__)


weather_classes = [
    'clear','foggy','drizzly',
    'cloudy','hazey', 'misty','rainy','smokey','thunderstorm'
]


def load_model(model_path='model/model.pkl'):
    """Load the trained model from disk."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def classify_weather(features):
    """Return predicted class label and inference latency (ms)."""
    model = load_model()
    start = time.time()
    prediction_index = model.predict(features)[0]
    latency = round((time.time() - start) * 1000, 2)
    prediction = weather_classes[int(prediction_index)]
    return prediction, latency



REQUIRED_FIELDS = [
    'temperature','pressure','humidity',
    'wind_speed','wind_deg'
]


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':


        for field in REQUIRED_FIELDS:
            if field not in request.form or request.form[field].strip() == '':
                error_msg = f"Missing field: {field}"
                return render_template('form.html', error=error_msg), 400

        try:

            temperature = float(request.form['temperature'])
            pressure = float(request.form['pressure'])
            humidity = float(request.form['humidity'])
            wind_speed = float(request.form['wind_speed'])
            wind_deg = float(request.form['wind_deg'])
            rain_1h = float(request.form.get('rain_1h', 0) or 0)
            rain_3h = float(request.form.get('rain_3h', 0) or 0)
            snow = float(request.form.get('snow', 0) or 0)
            clouds = float(request.form.get('clouds', 0) or 0)

            features = np.array([
                temperature, pressure, humidity,
                wind_speed, wind_deg, rain_1h,
                rain_3h, snow, clouds
            ]).reshape(1, -1)

            prediction, latency = classify_weather(features)

            return render_template(
                'result.html',
                prediction=prediction,
                latency=latency
            )

        except Exception as e:
            error_msg = f"Error processing input: {e}"
            return render_template('form.html', error=error_msg), 400


    return render_template('form.html')

#CD TEST 13


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)

