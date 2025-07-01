from flask import Flask, jsonify, render_template, request
from datetime import datetime
import numpy as np
import joblib
import threading

app = Flask(__name__)

# Load model and scaler
clf = joblib.load('model.pkl')      # trained model
scaler = joblib.load('scaler.pkl')  # scaler used during training

# Locations: initialize buffers and logs
location_buffers = {
    'Hyderabad': [],
    'Chennai': [],
    'Durgapur': []
}
classification_logs = {
    'Hyderabad': [],
    'Chennai': [],
    'Durgapur': []
}
buffer_lock = threading.Lock()

WINDOW_SIZE = 1000  # number of samples for one classification

@app.route('/')
def index():
    return render_template('live_classification.html')

@app.route('/upload', methods=['POST'])
def upload():
    data = request.get_json()
    location = data.get('location')
    voltage = data.get('voltage')

    if location not in location_buffers:
        return jsonify({'error': 'Invalid location'}), 400

    with buffer_lock:
        buf = location_buffers[location]
        buf.append(voltage)

        if len(buf) >= WINDOW_SIZE:
            window = np.array(buf[:WINDOW_SIZE]).reshape(1, -1)
            scaled = scaler.transform(window)
            prediction = clf.predict(scaled)[0]

            timestamp = datetime.now().strftime("%H:%M:%S")
            if prediction == 0:
                # Don't log if no disturbance, just clear the buffer
                location_buffers[location] = buf[WINDOW_SIZE:]
                return jsonify({'status': 'ok'})

            result = f"{timestamp} [{location}] ⚠️ Disturbance Detected"
            classification_logs[location].append(result)

            # Clear used samples
            location_buffers[location] = buf[WINDOW_SIZE:]

    return jsonify({'status': 'ok'})

@app.route('/status')
def status():
    with buffer_lock:
        # Return last 10 disturbances for each location
        return jsonify({
            loc: logs[-10:] for loc, logs in classification_logs.items()
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
