import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Example stock data (replace this with actual stock data)
data = {
    "Open": [100, 102, 105, 110, 108, 112, 115, 117],
    "High": [105, 106, 108, 112, 111, 115, 118, 120],
    "Low": [98, 101, 103, 108, 106, 110, 113, 115],
    "Close": [104, 105, 107, 111, 109, 114, 117, 119],
    "Volume": [1000, 1500, 1200, 1600, 1550, 1700, 1650, 1750]
}

# Create DataFrame
df = pd.DataFrame(data)

# Features (X) and label (y)
X = df[['Open', 'High', 'Low', 'Volume']]
y = df['Close']

# Feature Engineering: Add new features
X['Price_Range'] = X['High'] - X['Low']  # Price range as a new feature
X['Average_Price'] = (X['High'] + X['Low']) / 2  # Average price as a new feature

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline with StandardScaler, Polynomial Features, and Linear Regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Feature scaling
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),  # Add polynomial features
    ('model', LinearRegression())  # Linear regression model
])

# Train the model
pipeline.fit(X_train, y_train)

# Predict on test set
y_pred = pipeline.predict(X_test)

# Model evaluation using Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"R-squared: {r2}")

# Hyperparameter tuning using GridSearchCV (for polynomial degree)
param_grid = {
    'poly__degree': [1, 2, 3],  # Tune the degree of the polynomial features
}
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='r2')
grid_search.fit(X_train, y_train)

# Best hyperparameters from GridSearchCV
print("Best hyperparameters:", grid_search.best_params_)

# Save the best model using joblib
joblib.dump(grid_search.best_estimator_, 'stock_price_model.pkl')

# To load the model and predict
loaded_model = joblib.load('stock_price_model.pkl')

# Example prediction: predict stock price given new data (Open, High, Low, Volume, Price_Range, Average_Price)
new_data = np.array([[116, 119, 114, 1800, 119-114, (119+114)/2]])  # New data with engineered features
predicted_price = loaded_model.predict(new_data)

print(f"Predicted stock price: {predicted_price[0]}")