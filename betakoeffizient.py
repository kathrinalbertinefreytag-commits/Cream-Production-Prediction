import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression

# ===============================
# 1️⃣ Beispiel-Daten
# ===============================
X = np.array([
    [12, 65, 350],
    [10, 60, 300],
    [15, 70, 400],
    [11, 62, 320],
    [14, 68, 370]
])
feature_names = ["mixing_time", "temperature", "stirring_speed"]

y_linear = np.array([7.5, 6.0, 8.2, 6.8, 7.9])
y_logistic = np.array([1, 0, 1, 0, 1])

# ===============================
# 2️⃣ Modelle trainieren
# ===============================
lin_model = LinearRegression()
lin_model.fit(X, y_linear)

log_model = LogisticRegression(max_iter=1000, solver='lbfgs')
log_model.fit(X, y_logistic)

# ===============================
# 3️⃣ Beta-Koeffizienten
# ===============================
beta_linear = lin_model.coef_
beta_logistic = log_model.coef_[0]

# ===============================
# 4️⃣ Plot erstellen
# ===============================
x = np.arange(len(feature_names))
width = 0.35

fig, ax = plt.subplots(figsize=(8,5))
rects1 = ax.bar(x - width/2, beta_linear, width, label='Linear Regression', color="#4DB6AC")
rects2 = ax.bar(x + width/2, beta_logistic, width, label='Logistische Regression', color="#FF8A65")

# Achsen und Titel
ax.set_ylabel("Beta-Koeffizient (β)")
ax.set_title("Einfluss der Features auf die Vorhersage")
ax.set_xticks(x)
ax.set_xticklabels(feature_names)
ax.legend()
ax.axhline(0, color='black', linewidth=0.8)  # Null-Linie

# Werte über den Balken
for rects in [rects1, rects2]:
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0,3),
                    textcoords="offset points",
                    ha='center', va='bottom')

plt.tight_layout()
plt.show()
