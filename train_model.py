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