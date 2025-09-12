import csv
import time
import os
import numpy as np
import cv2
from pre_processing import WebcamStream


GESTURES = ["point", "pinch"]
SAMPLES_PER_GESTURE = 150
OUTPUT_CSV = "gesture_dataset.csv"


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

def collect(output_csv=OUTPUT_CSV):
    stream = WebcamStream(multi_hand=False)

    file_exists = os.path.isfile(output_csv)
    with open(output_csv, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            header = ["finger_count", "avg_tip_dist", "aspect_ratio", "extent", "solidity", "hull_ratio", "area", "perimeter", "label"]
            writer.writerow(header)

        for gesture in GESTURES:
            print(f"\nReady to collect: {gesture.upper()}. Press ENTER when ready.")
            input()
            print("Starting in 3 seconds...")
            time.sleep(3)

            collected = 0
            while collected < SAMPLES_PER_GESTURE:
                frame = stream.read_frame()
                mask = stream.get_skinmask(frame)
                mask = stream.morpho_mask(mask)

                frame, results = stream.get_finger_tips(mask, frame)
                if len(results) == 0:
                    cv2.imshow("Collecting", frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                    continue

                contour = results[0]["contour"]
                fingertips = results[0]["fingertips"]

                features = extract_features(contour, fingertips)
                if features:
                    writer.writerow(features + [gesture])
                    collected += 1

                cv2.putText(frame, f"{gesture.upper()} {collected}/{SAMPLES_PER_GESTURE}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.imshow("Collecting", frame)

                if cv2.waitKey(1) & 0xFF == 27:
                    break

            print(f"Collected {collected} samples for {gesture.upper()}.")

    stream.release()
    print("Data collection complete. Saved to", output_csv)

if __name__ == "__main__":
    collect()