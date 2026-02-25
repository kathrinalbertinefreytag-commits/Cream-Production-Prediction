import joblib
from sklearn.metrics import classification_report, confusion_matrix

# Loading Model and Encoder
pipeline = joblib.load("logistic_model.pkl")
le = joblib.load("label_encoder.pkl")

# preparing Test Data
from data_preparation import prepare_data
X_test, y_test, _ = prepare_data("data/cream_quality_data.csv")
# Achtung: prepare_data sollte hier die Testdaten **nicht mischen**!

# Prediction & Evaluation
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))
print(confusion_matrix(y_test, y_pred))
