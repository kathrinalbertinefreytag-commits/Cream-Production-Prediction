import numpy as np
import pandas as pd

np.random.seed(42)

def generate_cream_data(n_samples=1000):
    data = []

    for _ in range(n_samples):
        mixing_time = np.random.uniform(5, 30)          # Minuten
        temperature = np.random.uniform(20, 80)         # °C
        stirring_speed = np.random.uniform(200, 1500)   # rpm
        fat_content = np.random.uniform(5, 40)          # %
        water_content = np.random.uniform(50, 90)       # %
        ph_value = np.random.uniform(4.5, 7.5)

        # consistency (1–10)
        consistency = (
            0.25 * mixing_time
            + 0.2 * fat_content
            - 0.03 * abs(temperature - 40)
            - 0.05 * water_content
            - 0.3 * abs(ph_value - 6)
            + np.random.normal(0, 1.0)
        )

        consistency = np.clip(consistency, 1, 10)

        # humidity (%)
        moisture = (
            0.6 * water_content
            - 0.2 * fat_content
            + np.random.normal(0, 2)
        )
        moisture = np.clip(moisture, 40, 95)

        # colour (0–100, höher = heller)
        color = (
            80
            - 0.4 * fat_content
            - 0.1 * temperature
            + np.random.normal(0, 3)
        )
        color = np.clip(color, 20, 95)

        # quality label
        quality_score = (consistency + (color / 10)) / 2

        if quality_score >= 7:
            quality_label = "good"
        elif quality_score >= 5:
            quality_label = "mediocre"
        else:
            quality_label = "bad"

        data.append([
            mixing_time, temperature, stirring_speed,
            fat_content, water_content, ph_value,
            consistency, moisture, color, quality_label
        ])

    columns = [
        "mixing_time", "temperature", "stirring_speed",
        "fat_content", "water_content", "ph_value",
        "consistency", "moisture", "color", "quality_label"
    ]

    return pd.DataFrame(data, columns=columns)


# example: 3000 trainingsdata created
df = generate_cream_data(3000)
df.to_csv("cream_quality_data_english.csv", index=False)

print(df.head(100))
