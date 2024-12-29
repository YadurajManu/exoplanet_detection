# Advanced Exoplanet Classification Machine Learning Model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

# Simplified Machine Learning Imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Set random seed for reproducibility
np.random.seed(42)

# Generate Advanced Simulated Dataset
def generate_advanced_exoplanet_data(n_samples=2000):
    data = {
        'planet_radius': np.random.lognormal(2.3, 0.5, n_samples),
        'orbital_period': np.random.lognormal(2, 1, n_samples),
        'equilibrium_temp': np.random.normal(300, 100, n_samples),
        'stellar_magnitude': np.random.normal(10, 2, n_samples),
        'is_confirmed': np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
    }
    return pd.DataFrame(data)

# Generate dataset
df = generate_advanced_exoplanet_data()

# Prepare Features and Target
X = df.drop('is_confirmed', axis=1)
y = df['is_confirmed']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    stratify=y,
    random_state=42
)

# Scale Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest Classifier
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train_scaled, y_train)

# Evaluate Model
y_pred = model.predict(X_test_scaled)

print("Advanced Model Performance:")
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

# Feature Importance Visualization
feature_importance = model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(10, 6))
indices = np.argsort(feature_importance)
plt.title('Feature Importances in Exoplanet Classification')
plt.barh(range(len(indices)), feature_importance[indices])
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Relative Importance')
plt.tight_layout()
plt.savefig('data/advanced_feature_importance.png')
plt.close()

# Advanced Prediction Function
def predict_exoplanet_advanced(radius, period, temp, magnitude):
    input_data = np.array([[radius, period, temp, magnitude]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]
    
    print("\nAdvanced Exoplanet Prediction:")
    print(f"Radius: {radius:.2f} Earth radii")
    print(f"Orbital Period: {period:.2f} days")
    print(f"Equilibrium Temperature: {temp:.2f} Kelvin")
    print(f"Stellar Magnitude: {magnitude:.2f}")
    print(f"\nConfirmation Prediction: {'Confirmed' if prediction[0] == 1 else 'Not Confirmed'}")
    print(f"Confirmation Probability: {probability*100:.2f}%")

# Example prediction
predict_exoplanet_advanced(
    radius=10.5, 
    period=15.3, 
    temp=350, 
    magnitude=9.5
)

print("\nAdvanced Exoplanet Classification Model Complete!")
