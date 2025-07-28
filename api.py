from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Inicializar la app
app = Flask(__name__)

# Cargar el modelo y el scaler
model = joblib.load("notebooks/kmeans2.pkl")
scaler = joblib.load("notebooks/scaler2.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        expected_cols = ['recency_z', 'frequency_z', 'monetary_z', 'payment_entropy']
        df = df[expected_cols]

        # Escalar los datos
        df_scaled = scaler.transform(df)

        # Predicción
        cluster = int(model.predict(df_scaled)[0])
        return jsonify({'cluster': cluster})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001)

add Flask API for KMeans prediction
