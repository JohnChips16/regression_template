import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Load the CSV file
df = pd.read_csv('aapl.csv')

# Check the first few rows of the dataframe to ensure it's loaded correctly
print(df.head())

# Features (X) and label (y)
X = df[['Open', 'High', 'Low', 'Close', 'Volume']]  # Features: Open, High, Low, Close, Volume
y = df['Adj Close']  # Label: Adj Close (what you want to predict)

# Split data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with StandardScaler, Polynomial Features, and Linear Regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Feature scaling
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),  # Polynomial features
    ('model', LinearRegression())  # Linear regression model
])

# Train the model
pipeline.fit(X_train, y_train)

# Predict on the test set
y_pred = pipeline.predict(X_test)

# Model evaluation using Mean Absolute Error (MAE) and R-squared (R2)
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

# Load the model and make a prediction with new data
loaded_model = joblib.load('stock_price_model.pkl')

# Example prediction: predict stock price given new data (Open, High, Low, Close, Volume)
new_data = np.array([[116, 119, 114, 117, 1800]])  # New data without the engineered features
predicted_price = loaded_model.predict(new_data)

print(f"Predicted stock price: {predicted_price[0]}")