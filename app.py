from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Memuat model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    features = np.array([[
        float(data["bill_length_mm"]),
        float(data["bill_depth_mm"]),
        float(data["flipper_length_mm"]),
        float(data["body_mass_g"])
    ]])

    prediction = model.predict(features)[0]

    return jsonify({
        "prediction": prediction
    })


if __name__ == "__main__":
    import os
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )