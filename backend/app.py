# backend/app.py
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from knn_model import load_or_train_model, predict_sample

app = Flask(__name__)
CORS(app)

# 앱 시작 시 모델을 로드하거나 없으면 학습 후 저장
MODEL_PATH = os.path.join(os.path.dirname(__file__), "knn_model.pkl")
model, metadata = load_or_train_model(MODEL_PATH)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": metadata}), 200

@app.route("/predict", methods=["POST"])
def predict():
    """
    요청 바디(JSON):
    {
        "features": [sepal_length, sepal_width, petal_length, petal_width]
    }
    """
    data = request.get_json(silent=True)
    if not data or "features" not in data:
        return jsonify({"error": "Missing 'features' in JSON body"}), 400

    features = data["features"]
    try:
        pred_label, pred_prob = predict_sample(model, features)
        return jsonify({
            "prediction": pred_label,
            "probabilities": pred_prob  # 각 클래스별 확률 딕셔너리
        }), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "Internal server error", "detail": str(e)}), 500

if __name__ == "__main__":
    # 개발용 서버 실행
    app.run(host="0.0.0.0", port=5000, debug=True)
