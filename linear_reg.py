import pandas as pd
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Example JSON data
data = '''
[
  {"size": 1000, "rooms": 2, "price": 150000},
  {"size": 1500, "rooms": 3, "price": 200000},
  {"size": 2000, "rooms": 4, "price": 250000},
  {"size": 2500, "rooms": 4, "price": 300000},
  {"size": 3000, "rooms": 5, "price": 350000}
]
'''

# Step 1: Load JSON data into a Pandas DataFrame
df = pd.read_json(StringIO(data))

# Step 2: Define features (X) and target/label (y)
X = df[['size', 'rooms']]  # Features
y = df['price']            # Target label

# Step 3: Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Standardize the features (optional but recommended)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Train a Linear Regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Save the model
joblib.dump(model, 'linear_regression_model.pkl')

# Load the model
loaded_model = joblib.load('linear_regression_model.pkl')

# Step 6: Make predictions on the test set
y_pred = loaded_model.predict(X_test_scaled)

# Step 7: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Step 8: Predict for a new house (e.g., size=1600, rooms=3)
new_data = pd.DataFrame([[1600, 3]], columns=['size', 'rooms'])
new_data_scaled = scaler.transform(new_data)
new_prediction = loaded_model.predict(new_data_scaled)
print(f"Predicted price for house (1600 sq ft, 3 rooms): ${new_prediction[0]:,.2f}")