import joblib
import cv2
import numpy as np
from pre_processing import WebcamStream

MODEL_FILE = "gesture_model.joblib"
LE_FILE = "label_encoder.joblib"


def extract_features(contour, fingertips):
    if contour is None or len(contour) < 3:
        return None

    area = cv2.contourArea(contour)
    if area <= 0:
        return None

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull) if len(hull) >= 3 else 0
    perimeter = cv2.arcLength(contour, True)

    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h if h != 0 else 0
    rect_area = w * h
    extent = float(area) / rect_area if rect_area != 0 else 0
    solidity = float(area) / hull_area if hull_area != 0 else 0
    hull_ratio = hull_area / area if area != 0 else 0

    finger_count = len(fingertips) if fingertips is not None else 0
    avg_tip_dist = 0.0
    if finger_count > 1:
        dists = []
        for i in range(finger_count):
            for j in range(i + 1, finger_count):
                dists.append(np.linalg.norm(np.array(fingertips[i]) - np.array(fingertips[j])))
        avg_tip_dist = float(np.mean(dists)) if dists else 0.0

    return [
        finger_count,
        avg_tip_dist,
        aspect_ratio,
        extent,
        solidity,
        hull_ratio,
        area,
        perimeter
    ]

def main():
    model = joblib.load(MODEL_FILE)
    le = joblib.load(LE_FILE)
    stream = WebcamStream()

    while True:
        frame = stream.read_frame()
        mask = stream.get_skinmask(frame)
        mask = stream.morpho_mask(mask)

        frame, results = stream.get_finger_tips(mask, frame)
        if len(results) > 0:
            contour = results[0]["contour"]
            fingertips = results[0]["fingertips"]

            features = extract_features(contour, fingertips)
            if features:
                X = np.array(features).reshape(1, -1)
                pred = model.predict(X)[0]
                label = le.inverse_transform([pred])[0]
                prob = None
                if hasattr(model, "predict_proba"):
                    prob = model.predict_proba(X).max()

                text = f"{label}"
                if prob:
                    text += f" ({prob:.2f})"
                cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow("Live Prediction", frame)
        cv2.imshow("Mask", mask)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    stream.release()

if __name__ == "__main__":
    main()