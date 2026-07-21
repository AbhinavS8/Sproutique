from flask import Flask, request, jsonify
from pest_risk_model import predict_pest_level_xgb
from numpy import float32
import send_to_cloud
import serial

send_to_cloud.setup_gsm()

app = Flask(__name__)

    
@app.route('/data', methods=['POST'])

def receive_data():
    data = request.json
    temperature = float(data['temperature'])
    humidity = float(data['humidity'])
    ph = float(data['ph'])
    
    # Predict pest risk level
    prediction, risk = predict_pest_level_xgb(temperature, humidity, ph)
    print(f"Prediction: {prediction}, Risk: {risk}")
    
    # Send data to cloud
    send_to_cloud.send_sensor_data_to_cloud(humidity, temperature, ph, float(prediction))
    
    return jsonify({"status": "success", "prediction": float(prediction), "risk": risk}), 200

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("Exiting...")
          
