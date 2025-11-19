# backend/knn_model.py
import os
import pickle
from typing import Tuple, Dict, Any, List

import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def train_model() -> Tuple[Any, Dict[str, Any]]:
    """
    Iris 데이터셋으로 KNN 파이프라인을 학습합니다.
    파이프라인: StandardScaler -> KNeighborsClassifier(n_neighbors=5)
    반환: (model, metadata)
    """
    iris = load_iris()
    X, y = iris.data, iris.target
    class_names = iris.target_names.tolist()
    feature_names = iris.feature_names

    # 학습/검증 분할 (간단하게 80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=5))
    ])
    pipeline.fit(X_train, y_train)

    accuracy = pipeline.score(X_test, y_test)

    metadata = {
        "dataset": "iris",
        "feature_names": feature_names,
        "class_names": class_names,
        "test_accuracy": round(float(accuracy), 4),
        "model_type": "StandardScaler + KNeighborsClassifier(k=5)"
    }
    return pipeline, metadata

def save_model(model: Any, metadata: Dict[str, Any], path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump({"model": model, "metadata": metadata}, f)

def load_model(path: str) -> Tuple[Any, Dict[str, Any]]:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return payload["model"], payload["metadata"]

def load_or_train_model(path: str) -> Tuple[Any, Dict[str, Any]]:
    """
    경로에 모델이 있으면 로드, 없으면 학습 후 저장.
    """
    if os.path.exists(path):
        return load_model(path)
    model, metadata = train_model()
    # 경로가 존재하지 않으면 생성
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_model(model, metadata, path)
    return model, metadata

def predict_sample(model: Any, features: List[float]) -> Tuple[str, Dict[str, float]]:
    """
    단일 샘플 예측.
    features 길이는 4여야 함: [sepal_length, sepal_width, petal_length, petal_width]
    반환: (predicted_label_name, probabilities_by_class)
    """
    # 입력 검증
    if not isinstance(features, (list, tuple)) or len(features) != 4:
        raise ValueError("Features must be a list of four numeric values.")

    X = np.array(features, dtype=float).reshape(1, -1)
    pred_idx = model.predict(X)[0]

    # 클래스 이름 가져오기
    # metadata는 모델 외부에 있지만, 저장 시 포함되므로 모델에 직접 없을 수 있음
    # 간단히 iris target_names를 다시 사용
    class_names = load_iris().target_names.tolist()
    pred_label = class_names[int(pred_idx)]

    # 확률 계산 (KNN의 predict_proba 사용)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        prob_dict = {class_names[i]: float(round(probs[i], 4)) for i in range(len(class_names))}
    else:
        # predict_proba가 없으면 근사값 없이 단일 클래스 1.0
        prob_dict = {pred_label: 1.0}

    return pred_label, prob_dict
