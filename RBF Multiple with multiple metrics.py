#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Importing necessary libraries
import numpy as np
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

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

# Define the SVR models with different configurations
models = {
    'svr_model_1': SVR(kernel='rbf', C=1.0, epsilon=0.1, gamma='scale'),
    'svr_model_2': SVR(kernel='rbf', C=2.0, epsilon=0.1, gamma='scale'),
    'svr_model_3': SVR(kernel='rbf', C=1.0, epsilon=0.05, gamma='scale'),
    'svr_model_4': SVR(kernel='rbf', C=2.0, epsilon=0.05, gamma='scale')
}

# Train the models and compute the metrics
for model_name, model in models.items():
    # Train the model on the training data
    model.fit(train_x, train_y)
    
    # Predict using the model
    y_pred = model.predict(test_x)
    
    # Evaluate the model on the test data
    mse_ensemble = mean_squared_error(test_y, y_pred)
    print(f"MSE for {model_name}: {mse_ensemble}")
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(test_y, y_pred))
    print(f"Root Mean Squared Error (RMSE) for {model_name}: {rmse}")

    # Calculate R-squared
    r2 = r2_score(test_y, y_pred)
    print(f"R-squared for {model_name}: {r2}")

    # Generate Residual Plot
    residuals = test_y - y_pred
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.title(f'Residual Plot for {model_name}')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.savefig(f'{model_name}_residuals.png')
    plt.show()

    # Generate Prediction vs Actual Plot
    plt.scatter(test_y, y_pred, alpha=0.5)
    plt.plot([min(test_y), max(test_y)], [min(test_y), max(test_y)], color='red', linestyle='--')
    plt.title(f'Prediction vs Actual Plot for {model_name}')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.savefig(f'{model_name}_pred_vs_actual.png')
    plt.show()

