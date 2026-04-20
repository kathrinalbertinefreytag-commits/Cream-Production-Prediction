import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

def prepare_data(path):
    df = pd.read_csv(path)

#Production Parameters
    X = df[
        [ "mixing_time", "temperature", "stirring_speed",
         "fat_content", "water_content", "ph_value"]
    ]

#Qualification, 3 Labels
    y = df["quality_label"]

# 0= bad, 1=mediocre, 2=good
    le = LabelEncoder()
    y = le.fit_transform(y)

# scaling the features, at moment pipeline in train_model
   # scaler = StandardScaler()
    #X_scaled = scaler.fit_transform(X)


    return X, y, le
