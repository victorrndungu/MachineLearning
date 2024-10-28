import numpy
import matplotlib
import numpy as np
import pandas as pd

data = pd.read_csv("C:/Users/kahin/Downloads/Nairobi Office Price Ex.csv")

x = data['SIZE'] # Feature
y = data['PRICE'] # Price

def mean_squared_error(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

def gradient_descent( x, y, m, c, learning_rate):
    n = len(x)
    y_pred = m*x + c
    #Calculate gradients
    m_grad = (2 / n) * np.sum(x * ( y - y_pred))
    c_grad = (2 / n) * np.sum( y - y_pred)
    # Update weights
    m -= learning_rate * m_grad
    c -= learning_rate * c_grad
    return m, c

# Initialize m, c with random values
m, c = np.random.rand(), np.random.rand()
learning_rate = 0.01
epochs = 10

for epoch in range(epochs):
    m, c = gradient_descent(x, y, m, c, learning_rate)
    y_pred = m * x + c
    error = mean_squared_error(y, y_pred)
    print(f"Epoch {epoch+1}: Mean Squared Error = {error}")

import matplotlib.pyplot as plt

# Plot data points
plt.scatter(x, y, color="blue", label="Data Points")

# Plot regression line
y_pred = m * x + c
plt.plot(x, y_pred, color="red", label="Regression Line")
plt.xlabel("Office Size")
plt.ylabel("Office Price")
plt.legend()
plt.show()


