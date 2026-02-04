"""
Inference script for SageMaker.
Loads the model (w, b, scaler, selected_features) and exposes model_fn, input_fn, predict_fn, output_fn.
"""
import json
import numpy as np
import os


def model_fn(model_dir):
    """Carga los artefactos del modelo desde model_dir."""
    w = np.load(os.path.join(model_dir, "w.npy"))
    b = float(np.load(os.path.join(model_dir, "b.npy")))
    from sklearn.preprocessing import StandardScaler
    import joblib
    scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
    with open(os.path.join(model_dir, "selected_features.txt"), "r") as f:
        selected_features = [line.strip() for line in f.readlines()]
    return {"w": w, "b": b, "scaler": scaler, "selected_features": selected_features}


def input_fn(request_body, content_type):
    """Parse the request body (JSON)."""
    if content_type == "application/json":
        return json.loads(request_body)
    raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, model):
    """Predicción: mismo orden de features que selected_features, escala y σ(w·x + b)."""
    w, b, scaler, _ = model["w"], model["b"], model["scaler"], model["selected_features"]
    X = np.array(input_data)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    X_scaled = scaler.transform(X)
    logits = np.dot(X_scaled, w) + b
    # Sigmoid
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))
    return probs.flatten().tolist()


def output_fn(prediction, accept):
    """Return the response as JSON."""
    if accept == "application/json":
        return json.dumps({"predictions": prediction}), accept
    raise ValueError(f"Unsupported accept: {accept}")
