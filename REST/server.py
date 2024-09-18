from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

app = Flask(__name__)

# Paths to save model and data
TRAINED_DATA_FILE = 'aapl.csv'  # Using aapl.csv for the dataset
MODEL_FILE = 'aapl.pkl'

# Global variable to store grid search model
grid_search = None

# Train route
@app.route('/train', methods=['POST'])
def train():
    global grid_search  # Ensure we can use grid_search globally

    # Load the dataset
    if os.path.exists(TRAINED_DATA_FILE):
        df = pd.read_csv(TRAINED_DATA_FILE)
    else:
        return jsonify({'error': 'Dataset not found.'}), 400

    # Features (X) and label (y)
    X = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    y = df['Adj Close']

    # Feature Engineering
    X['Price_Range'] = X['High'] - X['Low']  # Range of the price
    X['Average_Price'] = (X['High'] + X['Low']) / 2  # Average of high and low prices
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Pipeline with StandardScaler, PolynomialFeatures, and LinearRegression
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('model', LinearRegression())
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Predict on the test set
    y_pred = pipeline.predict(X_test)

    # Model evaluation
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Hyperparameter tuning with GridSearchCV
    param_grid = {'poly__degree': [1, 2, 3]}  # Hyperparameter tuning for polynomial degree
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2')  # Cross-validation
    grid_search.fit(X_train, y_train)

    # Return the metrics and best hyperparameters
    return jsonify({
        'Mean Absolute Error': mae,
        'R-squared': r2,
        'Best hyperparameters': grid_search.best_params_
    })

# Save route
@app.route('/save', methods=['POST'])
def save_model():
    global grid_search

    if grid_search is None:
        return jsonify({'error': 'Model has not been trained yet.'}), 400

    try:
        # Save the best model using joblib
        joblib.dump(grid_search.best_estimator_, MODEL_FILE)
        return jsonify({'message': 'Model saved successfully.'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Load/Use route
@app.route('/load', methods=['POST'])
def load_model():
    try:
        # Load the saved model
        loaded_model = joblib.load(MODEL_FILE)
        
        # Load new data from the request
        new_data = request.get_json()
        new_data = np.array([[
            new_data['Open'], new_data['High'], new_data['Low'],
            new_data['Close'],
            new_data['Volume'],
            new_data['High'] - new_data['Low'],
            (new_data['High'] + new_data['Low']) / 2,
        ]])

        # Predict using the loaded model
        predicted_price = loaded_model.predict(new_data)

        # Return the prediction
        return jsonify({'Predicted Adj Close': predicted_price[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Diagnostic route to return model performance
@app.route('/diagnostic', methods=['GET'])
def diagnostic():
    try:
        # Load the trained data from CSV
        df = pd.read_csv(TRAINED_DATA_FILE)

        # Features (X) and label (y)
        X = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        y = df['Adj Close']

        # Feature Engineering
        X['Price_Range'] = X['High'] - X['Low']
        X['Average_Price'] = (X['High'] + X['Low']) / 2
        

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Load the model
        loaded_model = joblib.load(MODEL_FILE)

        # Predict using the loaded model
        y_pred = loaded_model.predict(X_test)

        # Calculate diagnostics
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return jsonify({
            'Mean Absolute Error': mae,
            'R-squared': r2
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
