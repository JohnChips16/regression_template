from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder

app = Flask(__name__)

# Path to save the trained model
MODEL_PATH = "people_recommender_model.pkl"

# Define the global variables for the current profile and the other profiles
current_profile = None
other_profiles = None
mlb = MultiLabelBinarizer()
one_hot_encoder = OneHotEncoder(sparse=False)

# Utility function to preprocess profiles into a feature vector
def preprocess_profiles(profiles):
    # Convert topics into a binary format using MultiLabelBinarizer
    topics_encoded = mlb.transform([p['topics'] for p in profiles])
    # Convert the country into one-hot encoding
    countries_encoded = one_hot_encoder.transform([[p['country']] for p in profiles])
    return np.hstack([countries_encoded, topics_encoded])

# Route to train the model
@app.route('/train', methods=['POST'])
def train_model():
    global current_profile, other_profiles, mlb, one_hot_encoder
    
    data = request.json
    current_profile = data['current_profile']
    other_profiles = data['other_profiles']
    
    # Combine current profile and other profiles for training
    all_profiles = [current_profile] + other_profiles
    
    # Prepare encoders
    countries = [p['country'] for p in all_profiles]
    topics = [p['topics'] for p in all_profiles]
    
    # Fit encoders
    mlb = MultiLabelBinarizer()
    mlb.fit(topics)
    one_hot_encoder = OneHotEncoder(sparse=False)
    one_hot_encoder.fit(np.array(countries).reshape(-1, 1))
    
    # Preprocess profiles
    X = preprocess_profiles(all_profiles)
    
    # Train nearest neighbors model
    knn = NearestNeighbors(n_neighbors=len(other_profiles), metric='cosine')
    knn.fit(X[1:])  # Fit on other profiles (excluding current profile)
    
    # Save the model and other data
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump((knn, mlb, one_hot_encoder), f)
    
    return jsonify({"message": "Model trained and saved successfully!"})

# Route to load and use the trained model
@app.route('/recommend', methods=['POST'])
def recommend():
    # Load the model
    with open(MODEL_PATH, 'rb') as f:
        try:
            knn, mlb, one_hot_encoder = pickle.load(f)
        except ValueError:
            return jsonify({"error": "Failed to load the model correctly. Ensure the model was saved properly."}), 500
    
    data = request.json
    current_profile = data['current_profile']
    
    # Preprocess current profile
    current_profile_vector = preprocess_profiles([current_profile])
    
    # Find nearest neighbors
    distances, indices = knn.kneighbors(current_profile_vector)
    
    recommendations = []
    for idx, dist in zip(indices[0], distances[0]):
        recommendations.append({
            "profile": other_profiles[idx],
            "similarity_score": 1 - dist  # Convert cosine distance to similarity
        })
    
    # Order recommendations by highest similarity
    recommendations = sorted(recommendations, key=lambda x: x['similarity_score'], reverse=True)
    
    return jsonify({"recommendations": recommendations})

if __name__ == '__main__':
    app.run(debug=True)