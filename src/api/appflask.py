from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os

# -------- Init App --------
app = Flask(__name__)
CORS(app) # enable CORS for all routes

# -------- Load Model --------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "tfidf_svm.pkl")
model = joblib.load(MODEL_PATH)

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status" : "ok","message": "API is running"}), 200

@app.route("/predict", methods = ["POST"])
def predict():
    try :
        data = request.get_json()
        text = data.get("text", None)

        if not text :
            return jsonify({"error" : "Missing 'text'in request body"}) , 400
        
        prediction = model.predict([text])[0]
        return jsonify({"prediction" : prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__" :
    app.run(host="0.0.0.0", port=8000, debug=True)