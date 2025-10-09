import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score


CSV_FILE = "gesture_dataset.csv"
MODEL_FILE = "gesture_model.joblib"
LE_FILE = "label_encoder.joblib"


def train(csv_file=CSV_FILE):
    df = pd.read_csv(csv_file)
    if "label" not in df.columns:
        raise ValueError("CSV must include 'label' column")

    X = df.drop(columns=["label"]).values
    y = df["label"].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", probability=True, random_state=42)),
    ])

    param_grid = {
        "svc__C": [1, 10, 50],
        "svc__gamma": ["scale", 0.01, 0.001],
    }

    clf = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1, verbose=1)
    clf.fit(X_train, y_train)

    best = clf.best_estimator_
    y_pred = best.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    joblib.dump(best, MODEL_FILE)
    joblib.dump(le, LE_FILE)
    print(f"Model saved as {MODEL_FILE}")

if __name__ == "__main__":
    train()
