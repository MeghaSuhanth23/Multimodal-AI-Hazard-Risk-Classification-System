from pathlib import Path
from collections import Counter
import cv2
import numpy as np
import pandas as pd
import joblib
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "notebooks" / "models"

DETECTOR_PATH = MODEL_DIR / "best.pt"
CLASSIFIER_PATH = MODEL_DIR / "risk_classifier.pkl"

RISK_ID_TO_NAME = {0: "Low", 1: "Medium", 2: "High"}
RISK_NAME_TO_ID = {"Low": 0, "Medium": 1, "High": 2}

CLASS_ID_TO_NAME = {
    0: "person",
    1: "helmet",
    2: "vest",
    3: "gloves",
    4: "goggles"
}

FEATURE_COLS = [
    "person_count",
    "helmet_count",
    "vest_count",
    "gloves_count",
    "goggles_count",
    "helmet_ratio",
    "vest_ratio",
    "gloves_ratio",
    "goggles_ratio",
    "missing_helmet_count",
    "missing_vest_count",
    "missing_gloves_count",
    "missing_goggles_count",
    "weighted_score",
    "person_conf_mean",
    "helmet_conf_mean",
    "vest_conf_mean",
    "gloves_conf_mean",
    "goggles_conf_mean"
]

def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# Model loading
def load_models():
    if not DETECTOR_PATH.exists():
        raise FileNotFoundError(f"Detector not found at: {DETECTOR_PATH}")
    if not CLASSIFIER_PATH.exists():
        raise FileNotFoundError(f"Classifier not found at: {CLASSIFIER_PATH}")

    detector = YOLO(str(DETECTOR_PATH))
    classifier = joblib.load(CLASSIFIER_PATH)
    return detector, classifier


# Utility helpers
def safe_ratio(count: int, person_count: int) -> float:
    if person_count == 0:
        return 0.0
    return min(count / person_count, 1.0)


def make_feature_frame(feats: dict) -> pd.DataFrame:
    return pd.DataFrame([[feats[col] for col in FEATURE_COLS]], columns=FEATURE_COLS)



# Feature extraction from YOLO predictions
def features_from_prediction_result(result, conf_threshold: float = 0.25) -> dict:
    empty_features = {
        "person_count": 0,
        "helmet_count": 0,
        "vest_count": 0,
        "gloves_count": 0,
        "goggles_count": 0,
        "helmet_ratio": 0.0,
        "vest_ratio": 0.0,
        "gloves_ratio": 0.0,
        "goggles_ratio": 0.0,
        "missing_helmet_count": 0,
        "missing_vest_count": 0,
        "missing_gloves_count": 0,
        "missing_goggles_count": 0,
        "weighted_score": 0.0,
        "person_conf_mean": 0.0,
        "helmet_conf_mean": 0.0,
        "vest_conf_mean": 0.0,
        "gloves_conf_mean": 0.0,
        "goggles_conf_mean": 0.0
    }

    if result.boxes is None or len(result.boxes) == 0:
        return empty_features

    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()

    keep_mask = confs >= conf_threshold
    cls_ids = cls_ids[keep_mask]
    confs = confs[keep_mask]

    if len(cls_ids) == 0:
        return empty_features

    person_mask = cls_ids == 0
    helmet_mask = cls_ids == 1
    vest_mask = cls_ids == 2
    gloves_mask = cls_ids == 3
    goggles_mask = cls_ids == 4

    person_count = int(person_mask.sum())
    helmet_count = int(helmet_mask.sum())
    vest_count = int(vest_mask.sum())
    gloves_count = int(gloves_mask.sum())
    goggles_count = int(goggles_mask.sum())

    helmet_ratio = safe_ratio(helmet_count, person_count)
    vest_ratio = safe_ratio(vest_count, person_count)
    gloves_ratio = safe_ratio(gloves_count, person_count)
    goggles_ratio = safe_ratio(goggles_count, person_count)

    weighted_score = (
        0.35 * helmet_ratio +
        0.30 * vest_ratio +
        0.20 * gloves_ratio +
        0.15 * goggles_ratio
    )

    return {
        "person_count": person_count,
        "helmet_count": helmet_count,
        "vest_count": vest_count,
        "gloves_count": gloves_count,
        "goggles_count": goggles_count,
        "helmet_ratio": helmet_ratio,
        "vest_ratio": vest_ratio,
        "gloves_ratio": gloves_ratio,
        "goggles_ratio": goggles_ratio,
        "missing_helmet_count": max(person_count - helmet_count, 0),
        "missing_vest_count": max(person_count - vest_count, 0),
        "missing_gloves_count": max(person_count - gloves_count, 0),
        "missing_goggles_count": max(person_count - goggles_count, 0),
        "weighted_score": weighted_score,
        "person_conf_mean": float(confs[person_mask].mean()) if person_mask.sum() else 0.0,
        "helmet_conf_mean": float(confs[helmet_mask].mean()) if helmet_mask.sum() else 0.0,
        "vest_conf_mean": float(confs[vest_mask].mean()) if vest_mask.sum() else 0.0,
        "gloves_conf_mean": float(confs[gloves_mask].mean()) if gloves_mask.sum() else 0.0,
        "goggles_conf_mean": float(confs[goggles_mask].mean()) if goggles_mask.sum() else 0.0
    }


# Explanation logic

def risk_explanation_from_features(feats: dict, predicted_risk_id: int) -> str:
    reasons = []

    if feats["missing_helmet_count"] > 0:
        reasons.append("helmet shortage detected")
    if feats["missing_vest_count"] > 0:
        reasons.append("vest shortage detected")
    if feats["missing_gloves_count"] > 0:
        reasons.append("glove shortage detected")
    if feats["missing_goggles_count"] > 0:
        reasons.append("goggle shortage detected")

    if len(reasons) == 0:
        return f"{RISK_ID_TO_NAME[predicted_risk_id]} risk: detected workers appear compliant with selected PPE."

    return f"{RISK_ID_TO_NAME[predicted_risk_id]} risk: " + ", ".join(reasons) + "."


# Visual grounding

def draw_risk_grounding(img_bgr, result, predicted_risk_id: int, conf_threshold: float = 0.25):
    img = img_bgr.copy()

    if result.boxes is None or len(result.boxes) == 0:
        return img

    xyxy = result.boxes.xyxy.cpu().numpy().astype(int)
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()

    for box, cls_id, conf in zip(xyxy, cls_ids, confs):
        if conf < conf_threshold:
            continue

        x1, y1, x2, y2 = box.tolist()

        if cls_id == 0:
            color = (0, 255, 0) if predicted_risk_id == 0 else (0, 165, 255) if predicted_risk_id == 1 else (0, 0, 255)
        else:
            color = (255, 255, 0)

        label = f"{CLASS_ID_TO_NAME.get(cls_id, 'obj')} {conf:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img,
            label,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    footer = f"Predicted Risk: {RISK_ID_TO_NAME[predicted_risk_id]}"
    cv2.putText(img, footer, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3)
    cv2.putText(img, footer, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

    return img


# Main prediction wrapper

def predict_image_array(image_bgr, detector_model, classifier_model, device=None, conf: float = 0.25):
    if device is None:
        device = get_device()

    result = detector_model.predict(
        source=image_bgr,
        conf=conf,
        imgsz=640,
        device=device,
        verbose=False
    )[0]

    feats = features_from_prediction_result(result, conf_threshold=conf)
    single_input = make_feature_frame(feats)
    pred_risk = int(classifier_model.predict(single_input)[0])
    explanation = risk_explanation_from_features(feats, pred_risk)

    grounded_img = draw_risk_grounding(image_bgr, result, pred_risk, conf_threshold=conf)

    return {
        "risk_id": pred_risk,
        "risk_name": RISK_ID_TO_NAME[pred_risk],
        "features": feats,
        "explanation": explanation,
        "grounded_bgr": grounded_img
    }