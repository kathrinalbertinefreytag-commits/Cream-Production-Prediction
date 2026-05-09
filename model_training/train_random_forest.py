import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, classification_report



# load CSV
data = pd.read_csv("data/cream_quality_data_english.csv")

#  encode target variable
label_encoder = LabelEncoder()

y = label_encoder.fit_transform(data["quality_label"])
# Features
X = data.drop("quality_label", axis=1)

# split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# training
model.fit(X_train, y_train)

# prediction
predictions = model.predict(X_test)

print(type(y_test))
print(type(predictions))
print(predictions[:10])
print(y_test[:10])
# classification report
print(classification_report(y_test, predictions))

# accuracy
print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))



# save
joblib.dump(model, "random_forest_model.pkl")
joblib.dump(label_encoder, "label_encoder_rf.pkl")

print("Random Forest saved")

