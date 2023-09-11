#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from scipy.stats import norm

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Function to calculate Black-Scholes prices for call options
def vectorized_EU_call_bs(S, K, r, sigma, t=0, T=1):
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * (T-t)) / (sigma * np.sqrt(T-t))
    d2 = d1 - sigma * np.sqrt(T-t)
    return S * norm.cdf(d1) - K * np.exp(-r * (T-t)) * norm.cdf(d2)

# Generate synthetic dataset
mu = 0.01
sample_size = 10000
sample = np.random.lognormal(mean=mu, sigma=0.8, size=sample_size) * 100
stock = sample
strike = sample * np.random.uniform(0.4, 1, sample_size)
time = np.random.uniform(size=sample_size)
sigma_values = np.random.uniform(0.1, 0.8, sample_size)
r_values = np.random.uniform(0.01, 0.05, sample_size)
bs_prices = vectorized_EU_call_bs(stock, strike, r_values, sigma_values, t=0, T=time)

# Convert to dataframe
data = {
    'Stock': stock,
    'Strike': strike,
    'Time': time,
    'sigma': sigma_values,
    'r': r_values,
    'BS': bs_prices
}

# Split dataset into training and testing sets
train_data, test_data = train_test_split(pd.DataFrame(data), test_size=0.3, random_state=42)

# Standardize features and transform response
features = ['Stock', 'Strike', 'Time', 'sigma', 'r']
scaler = StandardScaler()
train_x = scaler.fit_transform(train_data[features])
test_x = scaler.transform(test_data[features])
train_y = np.log(train_data['BS'].values)
test_y = np.log(test_data['BS'].values)


# Define base models for stacking
base_models = [
    ('mlp1', MLPRegressor(hidden_layer_sizes=(200, 150, 50), max_iter=100)),
    ('mlp2', MLPRegressor(hidden_layer_sizes=(250, 150, 75, 25), max_iter=100)),
    ('mlp3', MLPRegressor(hidden_layer_sizes=(300, 200, 100, 50, 10), max_iter=100)),
    ('mlp4', MLPRegressor(hidden_layer_sizes=(350, 250, 150, 75, 25, 10), max_iter=100))
]

# Define the final estimator (linear regression)
final_estimator = LinearRegression()

# Define the stacking model
stacking_model = StackingRegressor(estimators=base_models, final_estimator=final_estimator)

# Train the stacking model on the training data
stacking_model.fit(train_x, train_y)

# Get predictions from the stacking model
stacking_pred = stacking_model.predict(test_x)

# Evaluate the model on the test data
mse_stacking = mean_squared_error(test_y, stacking_model.predict(test_x))
print("MSE for Stacking Model:", mse_stacking)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(test_y, stacking_pred))
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Calculate R-squared
r2 = r2_score(test_y, stacking_pred)
print(f"R-squared: {r2}")

# Generate Residual Plot
residuals = test_y - stacking_pred
plt.scatter(stacking_pred, residuals, alpha=0.5)
plt.title('Residual Plot')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')

plt.show()

# Generate Prediction vs Actual Plot
plt.scatter(test_y, stacking_pred, alpha=0.5)
plt.plot([min(test_y), max(test_y)], [min(test_y), max(test_y)], color='red', linestyle='--')
plt.title('Prediction vs Actual Plot')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

plt.show()


# In[ ]:




