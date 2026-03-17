import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

# Example dataset: [area, bedrooms]
X = np.array([
    [1000, 2],
    [1500, 3],
    [2000, 4],
    [2500, 4],
    [3000, 5]
])

# Prices
y = np.array([200000, 300000, 400000, 500000, 600000])

model = LinearRegression()
model.fit(X, y)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("Model trained and saved!")