import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


sns.set(style="whitegrid")

# load data
df = pd.read_csv("data/cream_quality_data.csv")

# classification
sns.countplot(x="quality_label", data=df)
plt.title("Verteilung der Qualitätsklassen")
plt.show()

# feature distribution
features = [
    "mixing_time",
    "temperature",
    "stirring_speed",
    "fat_content",
    "water_content",
    "ph_value"
]

df[features].hist(bins=30, figsize=(12, 8))
plt.tight_layout()
plt.show()
