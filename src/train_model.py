from data_preparation import prepare_data
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# pull Data
X, y, le = prepare_data("data/cream_quality_data.csv")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Creating Model 
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs"))
])

#transformation on training data and training model
pipeline.fit(X_train, y_train)

# Saving Model
joblib.dump(pipeline, "logistic_model.pkl")  # Pipeline incl. Scaler + Model
joblib.dump(le, "label_encoder.pkl") 